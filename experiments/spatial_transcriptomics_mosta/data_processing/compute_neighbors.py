"""
Step 4: Build spatial k-NN graph (anisotropic, based on tangent directions).
Output is in CSR format (indptr, indices).

Usage:
    python compute_neighbors.py --config ../config.yaml --seed 1
"""
import argparse
import yaml
import os
import time
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.yaml")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

seed = args.seed if args.seed is not None else cfg['training']['seed']
base_dir = cfg['data']['base_dir']
inter_dir = cfg['data']['intermediate_dir']
nbr_cfg = cfg['graph']['neighbors']

width = nbr_cfg['width']
height = nbr_cfg['height']
radius = nbr_cfg['radius']
eps = 1e-12

half_w = width / 2
half_h = height / 2

start_time = time.time()

tangents_path = f"{inter_dir}/tangents_E165_E1S3_seed_{seed}.pt"
in_path = f"{base_dir}/tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt"
out_path = f"{inter_dir}/neighbors_25_E165_E1S3_seed_{seed}.pt"

tangents = torch.load(tangents_path, map_location="cpu")
data = torch.load(in_path, map_location="cpu")
coords = data["spatial"]

N = coords.shape[0]
assert coords.shape == (N, 2) and tangents.shape == (N, 2)

normal = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)

# Build uniform grid for candidate search
cell = float(max(width, height, 2 * radius))
xy = coords.numpy()
xmin, ymin = xy.min(axis=0)

cx = np.floor((xy[:, 0] - xmin) / cell).astype(np.int32)
cy = np.floor((xy[:, 1] - ymin) / cell).astype(np.int32)

key = (cx.astype(np.int64) << 32) ^ (cy.astype(np.int64) & 0xffffffff)
order = np.argsort(key)
key_sorted = key[order]

unique_keys, start_idx, counts = np.unique(key_sorted, return_index=True, return_counts=True)
cell2range = {int(k): (int(s), int(s + c)) for k, s, c in zip(unique_keys, start_idx, counts)}


def gather_candidates(i):
    cxi, cyi = int(cx[i]), int(cy[i])
    cand = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            k = (np.int64(cxi + dx) << 32) ^ (np.int64(cyi + dy) & 0xffffffff)
            r = cell2range.get(int(k))
            if r is None:
                continue
            s, e = r
            cand.append(order[s:e])
    if len(cand) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(cand, axis=0).astype(np.int64)


src_list, dst_list = [], []
coords_t = coords

for i in range(N):
    cand = gather_candidates(i)
    if cand.size == 0:
        continue
    cand = cand[cand != i]
    if cand.size == 0:
        continue

    cand_t = torch.from_numpy(cand)
    diff = coords_t[cand_t] - coords_t[i]

    if torch.linalg.norm(tangents[i]) < eps:
        d = torch.linalg.norm(diff, dim=1)
        keep = d <= radius
    else:
        u = diff @ tangents[i]
        v = diff @ normal[i]
        keep = (torch.abs(u) <= half_w) & (torch.abs(v) <= half_h)

    if keep.any():
        nbr = cand_t[keep].numpy()
        src_list.append(np.full(nbr.shape[0], i, dtype=np.int64))
        dst_list.append(nbr.astype(np.int64))

if len(src_list) == 0:
    raise RuntimeError("No edges found. Check width/height/radius or coordinate scale.")

src = np.concatenate(src_list, axis=0)
dst = np.concatenate(dst_list, axis=0)

# Symmetrize by union
src2 = np.concatenate([src, dst], axis=0)
dst2 = np.concatenate([dst, src], axis=0)

mask = src2 != dst2
src2, dst2 = src2[mask], dst2[mask]

key_e = np.unique(src2 * np.int64(N) + dst2)
src_u = (key_e // np.int64(N)).astype(np.int64)
dst_u = (key_e % np.int64(N)).astype(np.int64)

order_e = np.argsort(src_u)
src_u = src_u[order_e]
dst_u = dst_u[order_e]

deg = np.bincount(src_u, minlength=N).astype(np.int64)
indptr = np.zeros(N + 1, dtype=np.int64)
indptr[1:] = np.cumsum(deg)
indices = dst_u

deg_stats = np.array(deg)
print(f"deg mean/p50/p95/max: {deg_stats.mean():.1f} / "
      f"{np.percentile(deg_stats, 50):.1f} / "
      f"{np.percentile(deg_stats, 95):.1f} / {deg_stats.max()}")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(
    {
        "indptr": torch.from_numpy(indptr),
        "indices": torch.from_numpy(indices),
        "meta": {
            "N": N, "width": width, "height": height,
            "radius": radius, "cell": cell, "symmetrize": "union",
        },
    },
    out_path,
)
print(f"Saved CSR neighbors to: {out_path}")
print(f"Time: {time.time() - start_time:.1f}s")
