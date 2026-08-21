"""
Step 5: Compute edge weights combining geometric alignment + data-space cosine similarity.

Usage:
    python compute_weight.py --config ../config.yaml --seed 1
"""
import argparse
import yaml
import os
import time
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

neighbors_num = cfg['graph']['neighbors']['num']
lambda_similarity = cfg['graph']['weights']['lambda_similarity']
eps = 1e-10

start_time = time.time()
device = "cuda"

tangents = torch.load(f"{inter_dir}/tangents_E165_E1S3_seed_{seed}.pt", map_location="cpu")
data = torch.load(f"{base_dir}/tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt", map_location="cpu")
nbr = torch.load(f"{inter_dir}/neighbors_25_E165_E1S3_seed_{seed}.pt", map_location="cpu")

coords = data["spatial"].to(device)
tangents = tangents.to(device)
z = data["pca_data"].to(device)
z = z / (torch.linalg.norm(z, dim=1, keepdim=True) + eps)

indptr_cpu = nbr["indptr"]
indices = nbr["indices"].to(device)

N = coords.shape[0]
nnz = indices.numel()
print(f"N={N}, nnz={nnz}")

values = torch.empty((nnz,), device=device, dtype=coords.dtype)

for i in range(N):
    start = int(indptr_cpu[i])
    end = int(indptr_cpu[i + 1])
    k = end - start
    nbr_i = indices[start:end]

    xi = coords[i]
    ti = tangents[i]
    zi = z[i]
    xj = coords[nbr_i]
    tj = tangents[nbr_i]
    zj = z[nbr_i]

    e = xj - xi
    dist = torch.linalg.norm(e, dim=1).clamp_min(eps)
    u = e / dist[:, None]

    dot_tt = (ti[None, :] * tj).sum(dim=1)
    tan = torch.where(dot_tt[:, None] < 0, (ti - tj) * 0.5, (ti + tj) * 0.5)
    w_geom = torch.abs((u * tan).sum(dim=1))

    cos_ij = (zj * zi[None, :]).sum(dim=1)
    values[start:end] = w_geom + lambda_similarity * (cos_ij + 1.0)

out_path = f"{inter_dir}/weights_{neighbors_num}_lambda_{lambda_similarity}_E165_E1S3_seed_{seed}.pt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

torch.save(
    {
        "indptr": indptr_cpu.detach().cpu(),
        "indices": indices.detach().cpu(),
        "weights": values.detach().cpu(),
        "meta": {
            "N": int(N),
            "lambda_similarity": float(lambda_similarity),
            "eps": float(eps),
            "format": "CSR (indptr, indices, values aligned)",
        },
    },
    out_path,
)

print(f"Saved to: {out_path}")
print(f"Time: {time.time() - start_time:.1f}s")
