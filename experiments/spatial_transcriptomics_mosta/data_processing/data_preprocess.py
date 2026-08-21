"""
Step 1: Preprocess raw h5ad data.
  - Select highly variable genes (auto-skipped if n_genes <= n_top_genes)
  - Normalize total counts (skipped if do_normalize: false in config)
  - log1p, scale

Usage:
    python data_preprocess.py --config ../config.yaml
    python data_preprocess.py --config ../config_merfish.yaml --raw_file MERFISH_0.09.h5ad --dataset_id MERFISH_0.09
"""
import argparse
import yaml
import numpy as np
import scanpy as sc
import anndata as ad

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.yaml")
parser.add_argument("--raw_file", default=None, help="Override config raw_file (filename only)")
parser.add_argument("--dataset_id", default=None, help="Override config dataset_id; output: {dataset_id}.prep.h5ad")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

base_dir      = cfg['data']['base_dir']
raw_file      = args.raw_file if args.raw_file is not None else cfg['data']['raw_file']
dataset_id    = args.dataset_id if args.dataset_id is not None else cfg['data']['dataset_id']
label_col     = cfg['preprocessing']['label_col']
do_normalize  = cfg['preprocessing']['do_normalize']
n_top_genes   = cfg['preprocessing']['n_top_genes']
target_sum    = float(cfg['preprocessing']['target_sum'])
max_value     = cfg['preprocessing']['max_value']

in_path  = f"{base_dir}/{raw_file}"
out_path = f"{base_dir}/{dataset_id}.prep.h5ad"

print(f"Input:  {in_path}")
print(f"Output: {out_path}")

adata = sc.read_h5ad(in_path)
n_cells, n_genes = adata.shape
print(f"Loaded: {n_cells} cells × {n_genes} genes")

# ── Build clean AnnData ────────────────────────────────────────────────────
# Use layers["count"] if available (embryo), otherwise fall back to X
if "count" in adata.layers:
    X_src = adata.layers["count"].copy()
    print("Count source: layers['count']")
else:
    X_src = adata.X.copy()
    print("Count source: X")

import scipy.sparse as sp_mod
if sp_mod.issparse(X_src):
    X_src = X_src.astype(np.float32)
else:
    X_src = np.asarray(X_src, dtype=np.float32)

adata_clean = ad.AnnData(X=X_src, obs=adata.obs[[label_col]].copy())
adata_clean.obsm["spatial"] = adata.obsm["spatial"].copy()

# ── HVG selection (auto-skip when n_genes <= n_top_genes) ─────────────────
if n_genes > n_top_genes:
    print(f"Selecting top {n_top_genes} HVGs from {n_genes} genes...")
    sc.pp.highly_variable_genes(adata_clean, flavor="seurat_v3", n_top_genes=n_top_genes)
    adata_clean = adata_clean[:, adata_clean.var["highly_variable"]].copy()
    print(f"After HVG: {adata_clean.shape[1]} genes")
else:
    print(f"Skipping HVG (n_genes={n_genes} <= n_top_genes={n_top_genes})")

# ── Normalization ──────────────────────────────────────────────────────────
if do_normalize:
    print(f"Normalizing (target_sum={float(target_sum):.0e}) + log1p...")
    sc.pp.normalize_total(adata_clean, target_sum=target_sum)
    sc.pp.log1p(adata_clean)
else:
    print("Skipping normalize_total (do_normalize=false); applying log1p only...")
    sc.pp.log1p(adata_clean)

adata_clean.layers["log1p"] = adata_clean.X.copy()  # save before scale clipping
sc.pp.scale(adata_clean, zero_center=False, max_value=max_value)

print(adata_clean)
adata_clean.write_h5ad(out_path)
print(f"Saved to: {out_path}, shape: {adata_clean.shape}")
