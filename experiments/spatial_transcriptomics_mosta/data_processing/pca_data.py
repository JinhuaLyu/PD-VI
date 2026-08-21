"""
Step 2: PCA dimensionality reduction + K-means initialization.

Usage:
    python pca_data.py --config ../config.yaml --seed 1
"""
import argparse
import yaml
import os
import time
import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.yaml")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

seed = args.seed if args.seed is not None else cfg['training']['seed']

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

base_dir = cfg['data']['base_dir']
prep_file = cfg['data']['prep_file']
n_comps = cfg['preprocessing']['pca_components']
output_dir = cfg['data']['output_dir']

in_path = f"{base_dir}/{prep_file}"
out_path = f"{base_dir}/tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt"

time_start = time.time()

adata = sc.read_h5ad(in_path)
xy = adata.obsm["spatial"]
x, y = xy[:, 0], xy[:, 1]
true_labels = adata.obs["annotation"].cat.codes.to_numpy()
n_clusters = int(adata.obs["annotation"].nunique())

sc.tl.pca(adata, n_comps=n_comps)
data_pca = adata.obsm["X_pca"]

km = KMeans(n_clusters=n_clusters, random_state=seed).fit(data_pca)
initial_labels = km.predict(data_pca)

ari = adjusted_rand_score(true_labels, initial_labels)
print(f"K-means ARI: {ari:.4f}")

c1 = plt.get_cmap("tab20").colors
c2 = plt.get_cmap("tab20b").colors
cmap23 = ListedColormap(list(c1) + list(c2))

os.makedirs(output_dir, exist_ok=True)
plt.figure()
plt.scatter(x, y, s=5, c=initial_labels, cmap=cmap23)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title("KMeans labels")
plt.savefig(f"{output_dir}/initial_kmeans_labels_seed_{seed}.png")
plt.close()

save_obj = {
    "pca_data": torch.from_numpy(data_pca.copy()).float(),
    "spatial": torch.cat([
        torch.from_numpy(x.copy()).float().view(-1, 1),
        torch.from_numpy(y.copy()).float().view(-1, 1),
    ], dim=1),
    "true_labels": torch.from_numpy(true_labels.copy()).long(),
    "initial_labels": torch.from_numpy(initial_labels.copy()).long(),
    "seed": seed,
}

os.makedirs(base_dir, exist_ok=True)
torch.save(save_obj, out_path)
print(f"Saved to: {out_path}")
print(f"Time: {time.time() - time_start:.1f}s")
