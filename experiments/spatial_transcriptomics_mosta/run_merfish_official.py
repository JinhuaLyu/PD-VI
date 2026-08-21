"""
Run official MERFISH results: ours algorithm, seeds 1-5, all using circ r=90 neighbors.
Rebuilds graph files before training to ensure consistent neighbor config.

Usage:
    cd bio_pipeline
    python run_merfish_official.py [--n_gpus 5] [--config config_merfish.yaml]
"""
import argparse
import os
import queue
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import yaml
from sklearn.neighbors import NearestNeighbors

# ── Fixed params for this official run ────────────────────────────────────────
SEEDS      = [1, 2, 3, 4, 5]
CIRC_R     = 90.0
SHAPE      = 'circ'

ARI_RE  = re.compile(r'\bari=(-?[0-9][0-9.]*(?:[eE][+-]?[0-9]+)?)')
ELBO_RE = re.compile(r'\bobjective=(-?[0-9][0-9.]*(?:[eE][+-]?[0-9]+)?)')


def parse_last(stdout, pattern):
    m = pattern.findall(stdout)
    return float(m[-1]) if m else float('nan')


def compute_tangents(coords, init_labels, tcfg, eps=1e-12):
    N  = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=tcfg['n_neighbors'],
                          algorithm='kd_tree').fit(coords.numpy())
    _, ind  = nn.kneighbors(coords.numpy())
    nbr_idx = torch.from_numpy(ind[:, 1:]).long()
    I2      = torch.eye(2, dtype=coords.dtype)
    tangents = torch.zeros((N, 2), dtype=coords.dtype)
    for i in range(N):
        nbr = nbr_idx[i]; Xi = coords[nbr]
        yi = init_labels[i].item(); y_nbr = init_labels[nbr]
        ms = (y_nbr == yi); md = ~ms
        if ms.sum().item() < tcfg['min_same'] or md.sum().item() < tcfg['min_diff']:
            continue
        Xp, Xq = Xi[ms], Xi[md]
        mu_p, mu_q = Xp.mean(0), Xq.mean(0)
        Sw  = (Xp-mu_p).T@(Xp-mu_p) + (Xq-mu_q).T@(Xq-mu_q)
        dmu = mu_p - mu_q
        try:
            w = torch.linalg.solve(Sw + tcfg['lam']*I2, dmu)
        except RuntimeError:
            continue
        if torch.linalg.norm(w) < 1e-6:
            continue
        t = torch.stack([-w[1], w[0]])
        t = t / (torch.linalg.norm(t) + eps)
        if t[0] < 0: t = -t
        tangents[i] = t
    return tangents


def build_circ_graph(coords, tangents, feat_t, radius, lambda_sim, eps=1e-12):
    """Build circular neighbor graph (ignores tangent direction entirely)."""
    N     = coords.shape[0]
    cell  = float(2 * radius)
    xy_np = coords.numpy()
    xmin, ymin = xy_np.min(0)
    cx = np.floor((xy_np[:, 0]-xmin) / cell).astype(np.int32)
    cy = np.floor((xy_np[:, 1]-ymin) / cell).astype(np.int32)
    key   = (cx.astype(np.int64) << 32) ^ (cy.astype(np.int64) & 0xffffffff)
    order = np.argsort(key)
    ukeys, sidx, cnt = np.unique(key[order], return_index=True, return_counts=True)
    c2r   = {int(k): (int(s), int(s+c)) for k,s,c in zip(ukeys, sidx, cnt)}

    srcs, dsts = [], []
    for i in range(N):
        cxi, cyi = int(cx[i]), int(cy[i])
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (np.int64(cxi+dx) << 32) ^ (np.int64(cyi+dy) & 0xffffffff)
                seg = c2r.get(int(k))
                if seg: cand.append(order[seg[0]:seg[1]])
        if not cand: continue
        cand = np.concatenate(cand).astype(np.int64)
        cand = cand[cand != i]
        if cand.size == 0: continue
        ct   = torch.from_numpy(cand)
        diff = coords[ct] - coords[i]
        keep = torch.linalg.norm(diff, dim=1) <= radius
        if keep.any():
            nbr = ct[keep].numpy()
            srcs.append(np.full(nbr.shape[0], i, dtype=np.int64))
            dsts.append(nbr.astype(np.int64))

    if not srcs:
        raise RuntimeError(f"No edges found for circ r={radius}")

    src = np.concatenate(srcs); dst = np.concatenate(dsts)
    s2  = np.concatenate([src, dst]); d2 = np.concatenate([dst, src])
    msk = s2 != d2; s2, d2 = s2[msk], d2[msk]
    ke  = np.unique(s2 * np.int64(N) + d2)
    su  = (ke // np.int64(N)).astype(np.int64); du = (ke % np.int64(N)).astype(np.int64)
    oe  = np.argsort(su); su, du = su[oe], du[oe]
    deg = np.bincount(su, minlength=N).astype(np.int64)
    ip  = np.zeros(N+1, dtype=np.int64); ip[1:] = np.cumsum(deg)
    ipt = torch.from_numpy(ip); idxt = torch.from_numpy(du)
    nnz = idxt.numel()

    z   = feat_t / (torch.linalg.norm(feat_t, dim=1, keepdim=True) + eps)
    val = torch.empty(nnz, dtype=coords.dtype)
    for i in range(N):
        st, en = int(ip[i]), int(ip[i+1])
        if st == en: continue
        ni  = idxt[st:en]
        e   = coords[ni] - coords[i]
        u   = e / torch.linalg.norm(e, dim=1, keepdim=True).clamp_min(eps)
        ti  = tangents[i]; tj = tangents[ni]
        dtt = (ti[None,:]*tj).sum(1)
        tan = torch.where(dtt[:,None] < 0, (ti-tj)*0.5, (ti+tj)*0.5)
        wg  = torch.abs((u*tan).sum(1))
        cs  = (z[ni]*z[i][None,:]).sum(1)
        val[st:en] = wg + lambda_sim*(cs + 1.0)

    return ipt, idxt, val, float(deg.mean())


def run_seed(seed, cfg, dataset_id, base_dir, inter_dir, lambda_sim,
             nbr_num, tangent_cfg, script_dir, gpu_idx, lock):
    """Rebuild graph (circ r=90) then train. Returns (seed, ari, elbo, elapsed)."""
    t0 = time.time()

    # Load tensor file
    tp = os.path.join(base_dir,
                      f"tensor_pca_xy_true_initial_{dataset_id}_seed_{seed}.pt")
    d        = torch.load(tp, map_location="cpu")
    coords   = d["spatial"]; feat_t = d["pca_data"]; init_lbl = d["initial_labels"]

    with lock:
        print(f"  [seed={seed}] loaded tensor, N={coords.shape[0]}, computing tangents...")

    tangents = compute_tangents(coords, init_lbl, tangent_cfg)

    indptr, indices, values, mean_deg = build_circ_graph(
        coords, tangents, feat_t, CIRC_R, lambda_sim)

    gpath = os.path.join(inter_dir,
                         f"weights_{nbr_num}_lambda_{lambda_sim}_{dataset_id}_seed_{seed}.pt")
    torch.save({
        "indptr":  indptr,
        "indices": indices,
        "weights": values,
        "meta": {"N": int(coords.shape[0]), "lambda_similarity": float(lambda_sim),
                 "width": 0, "height": 0, "radius": CIRC_R, "shape": SHAPE},
    }, gpath)

    with lock:
        print(f"  [seed={seed}] graph saved (circ r={CIRC_R}, "
              f"edges={indices.numel()}, deg_mean={mean_deg:.1f}), starting training...")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    train_script = os.path.join(script_dir, "train.py")
    # Use the config as-is; graph path in intermediate_dir matches
    cmd = [sys.executable, train_script,
           "--config",    os.path.join(script_dir, "config_merfish.yaml"),
           "--algorithm", "ours",
           "--seed",      str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=script_dir, env=env)

    elapsed = time.time() - t0
    if proc.returncode != 0:
        with lock:
            print(f"  [seed={seed}] FAILED after {elapsed:.0f}s")
            print(proc.stderr[-500:])
        return seed, float('nan'), float('nan'), elapsed

    ari  = parse_last(proc.stdout, ARI_RE)
    elbo = parse_last(proc.stdout, ELBO_RE)
    with lock:
        print(f"  [seed={seed}] done  ARI={ari:.4f}  ELBO={elbo:.3e}  ({elapsed:.0f}s)")
    return seed, ari, elbo, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config_merfish.yaml")
    parser.add_argument("--n_gpus", type=int, default=5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path   = os.path.join(script_dir, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset_id  = cfg['data']['dataset_id']
    base_dir    = cfg['data']['base_dir']
    inter_dir   = cfg['data']['intermediate_dir']
    lambda_sim  = cfg['graph']['weights']['lambda_similarity']
    nbr_num     = cfg['graph']['neighbors']['num']
    tangent_cfg = cfg['graph']['tangent']
    max_iter    = cfg['training']['max_iter']

    # Resolve relative paths (config paths are relative to bio_pipeline/)
    if not os.path.isabs(base_dir):
        base_dir = os.path.normpath(os.path.join(script_dir, base_dir))
    if not os.path.isabs(inter_dir):
        inter_dir = os.path.normpath(os.path.join(script_dir, inter_dir))

    print(f"dataset_id={dataset_id}  seeds={SEEDS}  circ_r={CIRC_R}")
    print(f"max_iter={max_iter}  n_gpus={args.n_gpus}")
    print(f"base_dir={base_dir}")
    print(f"inter_dir={inter_dir}\n")

    gpu_q  = queue.Queue()
    for i in range(args.n_gpus):
        gpu_q.put(i)
    lock   = __import__('threading').Lock()
    results = {}

    def worker(seed):
        gpu = gpu_q.get()
        try:
            return run_seed(seed, cfg, dataset_id, base_dir, inter_dir,
                            lambda_sim, nbr_num, tangent_cfg, script_dir, gpu, lock)
        finally:
            gpu_q.put(gpu)

    with ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
        futs = [pool.submit(worker, s) for s in SEEDS]
        for fut in as_completed(futs):
            seed, ari, elbo, elapsed = fut.result()
            results[seed] = (ari, elbo, elapsed)

    aris  = [results[s][0] for s in SEEDS]
    elbos = [results[s][1] for s in SEEDS]

    print("\n" + "=" * 60)
    print(f"  {'seed':>5}  {'final ARI':>10}  {'ELBO':>13}  {'time(s)':>8}")
    print("-" * 60)
    for s in SEEDS:
        ari, elbo, elapsed = results[s]
        print(f"  {s:>5}  {ari:>10.4f}  {elbo:>13.3e}  {elapsed:>8.0f}")
    print("-" * 60)
    print(f"  {'mean':>5}  {np.nanmean(aris):>10.4f}  {np.nanmean(elbos):>13.3e}")
    print(f"  {'std':>5}  {np.nanstd(aris):>10.4f}  {np.nanstd(elbos):>13.3e}")
    print("=" * 60)
    print("\nDone. Run make_table.py to update table.md.")


if __name__ == "__main__":
    main()
