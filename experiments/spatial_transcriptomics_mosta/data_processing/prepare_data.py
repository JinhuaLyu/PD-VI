"""
Steps 2-5 合并：从预处理后的 h5ad 一步生成训练所需的全部数据。

中间结果（tangents、neighbors）只在内存中计算，不落盘。
最终只保存两个文件：
  - tensor_pca_xy_true_initial_{dataset_id}_seed_{seed}.pt  (特征数据 + 标签)
  - weights_{num}_lambda_{lam}_{dataset_id}_seed_{seed}.pt  (带权重的空间图)

PCA 自动跳过：若预处理后 n_genes < pca_components，直接使用特征矩阵。

Usage:
    python prepare_data.py --config ../config.yaml
    python prepare_data.py --config ../config_merfish.yaml --dataset_id MERFISH_0.09 --seed 2
"""
import argparse
import yaml
import os
import time
import numpy as np
import torch
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.yaml")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--dataset_id", default=None, help="Override config dataset_id")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

seed       = args.seed if args.seed is not None else cfg['training']['seed']
dataset_id = args.dataset_id if args.dataset_id is not None else cfg['data']['dataset_id']
label_col  = cfg['preprocessing']['label_col']

np.random.seed(seed)
torch.manual_seed(seed)

base_dir  = cfg['data']['base_dir']
inter_dir = cfg['data']['intermediate_dir']
n_comps   = cfg['preprocessing']['pca_components']

tangent_cfg       = cfg['graph']['tangent']
n_neighbors_tangent = tangent_cfg['n_neighbors']
lam               = tangent_cfg['lam']
min_same          = tangent_cfg['min_same']
min_diff          = tangent_cfg['min_diff']

nbr_cfg = cfg['graph']['neighbors']
width, height, radius = nbr_cfg['width'], nbr_cfg['height'], nbr_cfg['radius']
neighbors_num     = nbr_cfg['num']
lambda_similarity = cfg['graph']['weights']['lambda_similarity']

eps = 1e-12
t0  = time.time()

# ─────────────────────────────────────────────────────────────
# Step 2: Feature extraction + K-means
# ─────────────────────────────────────────────────────────────
print(f"[1/4] Feature extraction + K-means (dataset={dataset_id}, seed={seed})...")
in_path = f"{base_dir}/{dataset_id}.prep.h5ad"
adata   = sc.read_h5ad(in_path)
xy          = adata.obsm["spatial"]
true_labels = adata.obs[label_col].cat.codes.to_numpy()
n_clusters  = int(adata.obs[label_col].nunique())

n_genes_prep = adata.shape[1]
if n_genes_prep >= n_comps:
    print(f"    Running PCA ({n_genes_prep} genes → {n_comps} components)...")
    sc.tl.pca(adata, n_comps=n_comps)
    features = adata.obsm["X_pca"]
    feat_dim = n_comps
else:
    print(f"    Skipping PCA (n_genes={n_genes_prep} < pca_components={n_comps}); using features directly.")
    import scipy.sparse as sp_mod
    X = adata.X
    features = X.toarray() if sp_mod.issparse(X) else np.asarray(X)
    feat_dim = n_genes_prep

# ── K-means initialization ─────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler as _SS
kmeans_cfg     = cfg['preprocessing'].get('kmeans', {})
use_spatial    = kmeans_cfg.get('use_spatial', False)
spatial_weight = float(kmeans_cfg.get('spatial_weight', 3.0))
n_init         = kmeans_cfg.get('n_init', 'auto')
use_log1p_feat = kmeans_cfg.get('use_log1p_features', False)

if use_log1p_feat and "log1p" in adata.layers:
    import scipy.sparse as sp_mod2
    X_log = adata.layers["log1p"]
    X_log = X_log.toarray() if sp_mod2.issparse(X_log) else np.asarray(X_log)
    kmeans_feat = _SS().fit_transform(X_log)
    print(f"    K-means: using log1p features (pre-scale)")
else:
    kmeans_feat = _SS().fit_transform(features)

if use_spatial:
    xy_s = _SS().fit_transform(xy)
    kmeans_feat = np.hstack([kmeans_feat, xy_s * spatial_weight])
    print(f"    K-means: adding spatial coords (weight={spatial_weight})")

init_method = kmeans_cfg.get('init_method', 'kmeans')
if init_method == 'gmm_tied':
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                         n_init=n_init, random_state=seed, max_iter=300)
    gm.fit(kmeans_feat)
    initial_labels = gm.predict(kmeans_feat)
    print(f"    init_method=gmm_tied")
else:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit(kmeans_feat)
    initial_labels = km.labels_

coords  = torch.from_numpy(xy.copy()).float()          # (N, 2)
feat_t  = torch.from_numpy(features.copy()).float()    # (N, d)
true_t  = torch.from_numpy(true_labels.copy()).long()
init_t  = torch.from_numpy(initial_labels.copy()).long()

N, d = feat_t.shape
ari = adjusted_rand_score(true_labels, initial_labels)
print(f"    N={N}, d={d}, K={n_clusters}, K-means ARI={ari:.4f}, time={time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 3: 切向量（内存中计算，不保存）
# ─────────────────────────────────────────────────────────────
print("[2/4] Computing tangent vectors (in memory)...")

coords_np = coords.numpy()
nn_model  = NearestNeighbors(n_neighbors=n_neighbors_tangent, algorithm='kd_tree').fit(coords_np)
_, ind    = nn_model.kneighbors(coords_np)
nbr_idx   = torch.from_numpy(ind[:, 1:]).long()  # (N, n_neighbors-1)

I2      = torch.eye(2, dtype=coords.dtype)
tangents = torch.zeros((N, 2), dtype=coords.dtype)

for i in range(N):
    nbr   = nbr_idx[i]
    Xi    = coords[nbr]
    yi    = init_t[i].item()
    y_nbr = init_t[nbr]

    mask_same = (y_nbr == yi)
    mask_diff = ~mask_same

    if mask_same.sum().item() < min_same or mask_diff.sum().item() < min_diff:
        continue

    Xp, Xq   = Xi[mask_same], Xi[mask_diff]
    mu_p, mu_q = Xp.mean(dim=0), Xq.mean(dim=0)
    Sw  = (Xp - mu_p).T @ (Xp - mu_p) + (Xq - mu_q).T @ (Xq - mu_q)
    dmu = mu_p - mu_q

    try:
        w = torch.linalg.solve(Sw + lam * I2, dmu)
    except RuntimeError:
        continue

    norm_w = torch.linalg.norm(w)
    if norm_w < 1e-6:
        continue

    t_vec = torch.stack([-w[1], w[0]])
    t_vec = t_vec / (torch.linalg.norm(t_vec) + eps)
    if t_vec[0] < 0:
        t_vec = -t_vec
    tangents[i] = t_vec

print(f"    done, time={time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 4: 空间 k-NN 图（内存中计算，不保存）
# ─────────────────────────────────────────────────────────────
print("[3/4] Building spatial neighbor graph (in memory)...")

normal    = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
half_w, half_h = width / 2, height / 2

cell   = float(max(width, height, 2 * radius))
xy_np  = coords.numpy()
xmin, ymin = xy_np.min(axis=0)

cx = np.floor((xy_np[:, 0] - xmin) / cell).astype(np.int32)
cy = np.floor((xy_np[:, 1] - ymin) / cell).astype(np.int32)
key        = (cx.astype(np.int64) << 32) ^ (cy.astype(np.int64) & 0xffffffff)
order      = np.argsort(key)
key_sorted = key[order]

unique_keys, start_idx, counts_arr = np.unique(key_sorted, return_index=True, return_counts=True)
cell2range = {int(k): (int(s), int(s + c)) for k, s, c in zip(unique_keys, start_idx, counts_arr)}

src_list, dst_list = [], []
for i in range(N):
    cxi, cyi = int(cx[i]), int(cy[i])
    cand = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            k = (np.int64(cxi + dx) << 32) ^ (np.int64(cyi + dy) & 0xffffffff)
            r = cell2range.get(int(k))
            if r:
                cand.append(order[r[0]:r[1]])
    if not cand:
        continue
    cand = np.concatenate(cand).astype(np.int64)
    cand = cand[cand != i]
    if cand.size == 0:
        continue

    cand_t = torch.from_numpy(cand)
    diff   = coords[cand_t] - coords[i]

    if torch.linalg.norm(tangents[i]) < eps:
        keep = torch.linalg.norm(diff, dim=1) <= radius
    else:
        u    = diff @ tangents[i]
        v    = diff @ normal[i]
        keep = (torch.abs(u) <= half_w) & (torch.abs(v) <= half_h)

    if keep.any():
        nbr = cand_t[keep].numpy()
        src_list.append(np.full(nbr.shape[0], i, dtype=np.int64))
        dst_list.append(nbr.astype(np.int64))

if not src_list:
    raise RuntimeError("No edges found. Check width/height/radius.")

src = np.concatenate(src_list)
dst = np.concatenate(dst_list)

# 对称化
src2 = np.concatenate([src, dst])
dst2 = np.concatenate([dst, src])
mask = src2 != dst2
src2, dst2 = src2[mask], dst2[mask]

key_e = np.unique(src2 * np.int64(N) + dst2)
src_u = (key_e // np.int64(N)).astype(np.int64)
dst_u = (key_e %  np.int64(N)).astype(np.int64)

order_e     = np.argsort(src_u)
src_u, dst_u = src_u[order_e], dst_u[order_e]

deg        = np.bincount(src_u, minlength=N).astype(np.int64)
indptr_np  = np.zeros(N + 1, dtype=np.int64)
indptr_np[1:] = np.cumsum(deg)
indices_np = dst_u

indptr  = torch.from_numpy(indptr_np)
indices = torch.from_numpy(indices_np)
nnz     = indices.numel()
print(f"    edges={nnz}, deg_mean={deg.mean():.1f}, time={time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 5: 边权重（在 CPU 上计算）
# ─────────────────────────────────────────────────────────────
print("[4/4] Computing edge weights...")

z      = feat_t / (torch.linalg.norm(feat_t, dim=1, keepdim=True) + eps)
values = torch.empty(nnz, dtype=coords.dtype)

for i in range(N):
    start = int(indptr_np[i])
    end   = int(indptr_np[i + 1])
    if start == end:
        continue
    nbr_i = indices[start:end]

    e    = coords[nbr_i] - coords[i]
    dist = torch.linalg.norm(e, dim=1).clamp_min(eps)
    u    = e / dist[:, None]

    ti     = tangents[i]
    tj     = tangents[nbr_i]
    dot_tt = (ti[None, :] * tj).sum(dim=1)
    tan    = torch.where(dot_tt[:, None] < 0, (ti - tj) * 0.5, (ti + tj) * 0.5)
    w_geom = torch.abs((u * tan).sum(dim=1))

    cos_ij         = (z[nbr_i] * z[i][None, :]).sum(dim=1)
    values[start:end] = w_geom + lambda_similarity * (cos_ij + 1.0)

print(f"    done, time={time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# 保存：只存两个训练需要的文件
# ─────────────────────────────────────────────────────────────
os.makedirs(base_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)

tensor_path = f"{base_dir}/tensor_pca_xy_true_initial_{dataset_id}_seed_{seed}.pt"
torch.save({
    "pca_data":      feat_t,
    "spatial":       coords,
    "true_labels":   true_t,
    "initial_labels": init_t,
    "seed":          seed,
}, tensor_path)
print(f"Saved: {tensor_path}")

graph_path = f"{inter_dir}/weights_{neighbors_num}_lambda_{lambda_similarity}_{dataset_id}_seed_{seed}.pt"
torch.save({
    "indptr":  indptr,
    "indices": indices,
    "weights": values,
    "meta": {
        "N": int(N), "lambda_similarity": float(lambda_similarity),
        "width": width, "height": height, "radius": radius,
    },
}, graph_path)
print(f"Saved: {graph_path}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Ready for training.")
