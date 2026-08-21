"""
One-time exploratory script: print key statistics for all datasets
and save spatial scatter plots colored by cell type labels.

Usage:
    python explore_datasets.py
"""
import os
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE    = os.path.join(os.path.dirname(__file__), "../../data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../output/explore")
os.makedirs(OUT_DIR, exist_ok=True)

# (name, path, domain_label_col, celltype_label_col or None)
DATASETS = [
    ("MERFISH_0.04", f"{BASE}/MERFISH_hypothalamic/MERFISH_0.04.h5ad", "ground_truth", "cell_class"),
    ("MERFISH_0.09", f"{BASE}/MERFISH_hypothalamic/MERFISH_0.09.h5ad", "ground_truth", "cell_class"),
    ("MERFISH_0.14", f"{BASE}/MERFISH_hypothalamic/MERFISH_0.14.h5ad", "ground_truth", "cell_class"),
    ("MERFISH_0.19", f"{BASE}/MERFISH_hypothalamic/MERFISH_0.19.h5ad", "ground_truth", "cell_class"),
    ("MERFISH_0.24", f"{BASE}/MERFISH_hypothalamic/MERFISH_0.24.h5ad", "ground_truth", "cell_class"),
    ("osmFISH",      f"{BASE}/osmfish_cortex/osmfish.h5ad",             "ground_truth", "ClusterName"),
    ("ST_AD",        f"{BASE}/ST_AD/NG_LateAD_Dec_20_2021_Human7.h5ad","annotation",   None),
]

# Cell type label used for training / plots
CELLTYPE_COL = {
    "MERFISH_0.04": "cell_class",
    "MERFISH_0.09": "cell_class",
    "MERFISH_0.14": "cell_class",
    "MERFISH_0.19": "cell_class",
    "MERFISH_0.24": "cell_class",
    "osmFISH":      "ClusterName",
    "ST_AD":        "annotation",
}

# Domain label col (None = same as cell type col)
DOMAIN_COL = {
    "MERFISH_0.04": "ground_truth",
    "MERFISH_0.09": "ground_truth",
    "MERFISH_0.14": "ground_truth",
    "MERFISH_0.19": "ground_truth",
    "MERFISH_0.24": "ground_truth",
    "osmFISH":      "ground_truth",
    "ST_AD":        None,           # annotation serves as both
}


def make_cmap(n):
    c1 = list(plt.get_cmap("tab20").colors)
    c2 = list(plt.get_cmap("tab20b").colors)
    colors = (c1 + c2)[:n]
    return ListedColormap(colors)


def x_to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


for name, path, domain_col, celltype_col in DATASETS:
    print(f"\n{'═' * 55}")
    print(f"  {name}")
    print(f"{'═' * 55}")

    if not os.path.exists(path):
        print(f"  [FILE NOT FOUND]: {path}")
        continue

    adata = sc.read_h5ad(path)
    n_cells, n_genes = adata.shape
    print(f"  shape:             {n_cells} cells × {n_genes} genes")

    # Label counts
    if domain_col and domain_col in adata.obs.columns:
        n_domain = adata.obs[domain_col].nunique()
        print(f"  domain labels:     {domain_col!r:20s} → {n_domain} unique")
    else:
        print(f"  domain labels:     [{domain_col!r} NOT FOUND]")

    if celltype_col and celltype_col in adata.obs.columns:
        n_ct = adata.obs[celltype_col].nunique()
        print(f"  cell type labels:  {celltype_col!r:20s} → {n_ct} unique")
    elif celltype_col:
        print(f"  cell type labels:  [{celltype_col!r} NOT FOUND]")
    else:
        print(f"  cell type labels:  (none)")

    # Layers
    has_count = "count" in adata.layers
    print(f"  has layers[count]: {has_count}")
    print(f"  all layers:        {list(adata.layers.keys()) or '(none)'}")

    # X statistics
    X = x_to_dense(adata.X)
    print(f"  X dtype:           {X.dtype}")
    print(f"  X range:           min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}")
    n_nonzero = np.count_nonzero(X)
    sparsity  = 1.0 - n_nonzero / X.size
    print(f"  X sparsity:        {sparsity:.1%}  (non-zero: {n_nonzero})")

    # Spatial coords
    if "spatial" in adata.obsm:
        xy = adata.obsm["spatial"]
        print(f"  spatial shape:     {xy.shape}")
        print(f"  x range:           [{xy[:, 0].min():.2f}, {xy[:, 0].max():.2f}]")
        print(f"  y range:           [{xy[:, 1].min():.2f}, {xy[:, 1].max():.2f}]")
    else:
        print(f"  spatial:           [obsm['spatial'] NOT FOUND]")
        xy = None

    # All obs columns
    print(f"  all obs cols:      {list(adata.obs.columns)}")

    # ── Spatial scatter plots ──────────────────────────────────────────────
    def save_spatial_plot(col, suffix, title_suffix):
        if xy is None or col not in adata.obs.columns:
            return
        labels     = adata.obs[col]
        categories = labels.cat.categories if hasattr(labels, "cat") else sorted(labels.unique())
        n_cats     = len(categories)
        cmap       = make_cmap(n_cats)
        label_codes = labels.cat.codes.to_numpy() if hasattr(labels, "cat") \
                      else np.array([list(categories).index(v) for v in labels])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(xy[:, 0], xy[:, 1], c=label_codes, cmap=cmap,
                   s=4, linewidths=0, alpha=0.85, vmin=0, vmax=n_cats - 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name}  —  {col}  ({n_cats} {title_suffix})", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(i / max(n_cats - 1, 1)),
                       markersize=6, label=str(cat))
            for i, cat in enumerate(categories)
        ]
        ax.legend(handles=handles, title=col,
                  bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, title_fontsize=8, frameon=False)
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, f"{name}_{suffix}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {fig_path}")

    save_spatial_plot(CELLTYPE_COL[name], "celltype", "cell types")
    domain_col = DOMAIN_COL[name]
    if domain_col is not None:
        save_spatial_plot(domain_col, "domain", "domains")

print(f"\n{'═' * 55}\n")
