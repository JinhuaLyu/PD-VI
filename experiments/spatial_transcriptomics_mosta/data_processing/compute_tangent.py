"""
Step 3: Compute local manifold tangent vectors (Fisher LDA per cell).

Usage:
    python compute_tangent.py --config ../config.yaml --seed 1
"""
import argparse
import yaml
import os
import torch
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.yaml")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

seed = args.seed if args.seed is not None else cfg['training']['seed']
base_dir = cfg['data']['base_dir']
inter_dir = cfg['data']['intermediate_dir']
tangent_cfg = cfg['graph']['tangent']

n_neighbors = tangent_cfg['n_neighbors']
lam = tangent_cfg['lam']
min_same = tangent_cfg['min_same']
min_diff = tangent_cfg['min_diff']


def local_fisher_tangents(coords, labels, neighbors, lam=1e-2, min_same=5, min_diff=5,
                           eps=1e-12, enforce_positive_x=True):
    N = coords.shape[0]
    tangents = torch.zeros((N, 2), device=coords.device, dtype=coords.dtype)
    I = torch.eye(2, device=coords.device, dtype=coords.dtype)

    for i in range(N):
        nbr = neighbors[i]
        Xi = coords[nbr]
        yi = labels[i]
        y_nbr = labels[nbr]

        mask_same = (y_nbr == yi)
        mask_diff = ~mask_same

        if mask_same.sum().item() < min_same or mask_diff.sum().item() < min_diff:
            continue

        Xp = Xi[mask_same]
        Xq = Xi[mask_diff]
        mu_p = Xp.mean(dim=0)
        mu_q = Xq.mean(dim=0)

        Cp = Xp - mu_p
        Cq = Xq - mu_q
        Sw = Cp.T @ Cp + Cq.T @ Cq
        Sw_reg = Sw + lam * I
        dmu = mu_p - mu_q

        try:
            w = torch.linalg.solve(Sw_reg, dmu)
        except RuntimeError:
            continue

        norm_w = torch.linalg.norm(w)
        if norm_w < 1e-6:
            continue

        t = torch.stack([-w[1], w[0]])
        t = t / (torch.linalg.norm(t) + eps)

        if enforce_positive_x and t[0] < 0:
            t = -t

        tangents[i] = t

    return tangents


in_path = f"{base_dir}/tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt"
data = torch.load(in_path)
coords = data["spatial"]
initial_labels = data["initial_labels"]

coords_np = coords.detach().cpu().numpy()
nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(coords_np)
_, ind = nn.kneighbors(coords_np)
nbr_idx = torch.from_numpy(ind[:, 1:]).long().to(coords.device)

tangents = local_fisher_tangents(coords, initial_labels, nbr_idx, lam=lam,
                                  min_same=min_same, min_diff=min_diff)

os.makedirs(inter_dir, exist_ok=True)
out_path = f"{inter_dir}/tangents_E165_E1S3_seed_{seed}.pt"
torch.save(tangents, out_path)
print(f"Saved to: {out_path}")
