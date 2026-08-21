"""
Step 1b (optional): Save raw integer count data (200 HVGs, no normalization).

Usage:
    python integer_data_process.py --config ../config.yaml
"""
import argparse
import yaml
import scanpy as sc
import anndata as ad

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

base_dir = cfg['data']['base_dir']
raw_file = cfg['data']['raw_file']
out_file = cfg['data']['prep_count_file']
n_top_genes = cfg['preprocessing']['n_top_genes']

in_path = f"{base_dir}/{raw_file}"
out_path = f"{base_dir}/{out_file}"

adata = sc.read_h5ad(in_path)

adata_clean = ad.AnnData(
    X=adata.layers["count"].copy(),
    obs=adata.obs[["annotation"]].copy(),
)
adata_clean.obsm["spatial"] = adata.obsm["spatial"].copy()

sc.pp.highly_variable_genes(adata_clean, flavor="seurat_v3", n_top_genes=n_top_genes)
adata_clean = adata_clean[:, adata_clean.var["highly_variable"]].copy()

adata_clean.write_h5ad(out_path)
print(f"Saved to: {out_path}, shape: {adata_clean.shape}")
