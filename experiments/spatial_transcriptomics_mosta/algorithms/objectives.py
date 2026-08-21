"""
Shared objective functions and utilities used by all clustering algorithms.
"""
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import adjusted_rand_score as sk_ari


# ─────────────────────────────────────────────────────────────────────────────
# ELBO objectives
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def global_neg_elbo_without_graph(
    x: torch.Tensor,          # (n, d)
    alpha: torch.Tensor,      # (n, K) logits
    m: torch.Tensor,          # (K, d)
    rho: torch.Tensor,        # (K, d) rho = log s^2
    sigma0_2: torch.Tensor,   # (d,)
    sigma1_2: torch.Tensor,   # (d,)
    xi: torch.Tensor | None = None,
    T: float = 1.0,
    eps: float = 1e-12,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute global negative ELBO (without graph)."""
    device = x.device
    n, d = x.shape
    K = alpha.shape[1]

    if xi is None:
        xi = torch.zeros(d, device=device, dtype=x.dtype)
    else:
        xi = xi.to(device=device, dtype=x.dtype)

    T = float(T)
    inv_sigma0_2 = (1.0 / sigma0_2.to(device=device, dtype=x.dtype)).view(1, d)
    inv_sigma1_2 = (1.0 / sigma1_2.to(device=device, dtype=x.dtype)).view(1, d)

    exp_rho = torch.exp(rho)
    m2_plus_s2 = m * m + exp_rho

    term_entropy = x.new_zeros(())
    term_data = x.new_zeros(())

    for st in range(0, n, chunk_size):
        ed = min(n, st + chunk_size)
        x_blk = x[st:ed]
        alpha_blk = alpha[st:ed]
        phi_blk = torch.softmax(alpha_blk / T, dim=-1)
        phi_safe = phi_blk.clamp_min(eps)

        term_entropy = term_entropy + (phi_safe * phi_safe.log()).sum()

        const_k = 0.5 * (m2_plus_s2 * inv_sigma0_2).sum(dim=1)
        xm = torch.einsum("bd,kd->bk", x_blk * inv_sigma0_2, m)
        C = const_k.view(1, K) - xm
        term_data = term_data + (phi_blk * C).sum()

    prior_per_k = (((m2_plus_s2 - 2.0 * (xi.view(1, d) * m)) * inv_sigma1_2) - rho).sum(dim=1)
    term_prior = 0.5 * prior_per_k.sum()
    return term_entropy + term_data + term_prior


@torch.no_grad()
def global_neg_elbo_with_patch_graph(
    x: torch.Tensor,
    alpha: torch.Tensor,
    m: torch.Tensor,
    rho: torch.Tensor,
    sigma0_2: torch.Tensor,
    sigma1_2: torch.Tensor,
    patch_indices: list,
    patch_edge_index_local: list,
    patch_edge_weight: list,
    xi: torch.Tensor | None = None,
    T_for_phi: float | None = None,
    eps: float = 1e-12,
    chunk_size: int = 4096,
) -> torch.Tensor:
    T_eff = T_for_phi if T_for_phi is not None else 1.0
    base = global_neg_elbo_without_graph(
        x=x, alpha=alpha, m=m, rho=rho,
        sigma0_2=sigma0_2, sigma1_2=sigma1_2,
        xi=xi, T=T_eff, eps=eps, chunk_size=chunk_size
    )

    device = x.device
    dtype = x.dtype

    if T_for_phi is None:
        phi_all = torch.softmax(alpha, dim=-1)
    else:
        phi_all = torch.softmax(alpha / float(T_for_phi), dim=-1)

    term_patch_graph = x.new_zeros(())
    B = len(patch_indices)

    for b in range(B):
        St = patch_indices[b]
        if St.numel() == 0:
            continue

        edge_l = patch_edge_index_local[b]
        w = patch_edge_weight[b].to(device=device, dtype=dtype)

        if edge_l.numel() == 0:
            continue

        phi_b = phi_all[St]
        src = edge_l[0].to(device)
        dst = edge_l[1].to(device)

        dots = (phi_b[src] * phi_b[dst]).sum(dim=1)
        term_patch_graph = term_patch_graph - 0.5 * (w * dots).sum()

    return base + term_patch_graph


def patch_neg_elbo_with_graph(
    x_b: torch.Tensor,
    alpha_b: torch.Tensor,
    m0: torch.Tensor,
    rho0: torch.Tensor,
    sigma0_2: torch.Tensor,
    sigma1_2: torch.Tensor,
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    T: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    device = x_b.device
    n_b, d = x_b.shape
    K = alpha_b.shape[1]

    inv_sigma0_2 = (1.0 / sigma0_2.to(device=device, dtype=x_b.dtype)).view(1, 1, d)
    inv_sigma1_2 = (1.0 / sigma1_2.to(device=device, dtype=x_b.dtype)).view(1, d)

    phi = torch.softmax(alpha_b / float(T), dim=-1)
    phi_safe = phi.clamp_min(eps)

    term_entropy = (phi_safe * phi_safe.log()).sum()

    x_ = x_b.view(n_b, 1, d)
    m_ = m0.view(1, K, d)
    exp_rho = torch.exp(rho0).view(1, K, d)
    C_ik = 0.5 * ((m_ * m_ + exp_rho - 2.0 * x_ * m_) * inv_sigma0_2).sum(dim=-1)
    term_data = (phi * C_ik).sum()

    scale = float(n_b) / float(n)
    prior_per_k = ((m0 * m0 + torch.exp(rho0)) * inv_sigma1_2 - rho0).sum(dim=-1)
    term_prior = 0.5 * scale * prior_per_k.sum()

    if edge_index.numel() == 0:
        term_graph = x_b.new_zeros(())
    else:
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        w = edge_weight.to(device=device, dtype=x_b.dtype)
        dots = (phi[src] * phi[dst]).sum(dim=1)
        term_graph = -0.5 * (w * dots).sum()

    return term_entropy + term_data + term_prior + term_graph


# ─────────────────────────────────────────────────────────────────────────────
# Graph utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_patch_edges_from_csr(St, indptr, indices, weights, N):
    device = St.device

    in_patch = torch.zeros(N, dtype=torch.bool, device=device)
    in_patch[St] = True

    starts = indptr[St]
    ends = indptr[St + 1]
    deg = ends - starts
    total_deg = int(deg.sum().item())

    base = torch.repeat_interleave(starts, deg)
    offset = torch.arange(total_deg, device=device) - torch.repeat_interleave(
        torch.cumsum(deg, dim=0) - deg, deg
    )
    nbr_pos = base + offset
    nbr = indices[nbr_pos]
    w = weights[nbr_pos]

    src = torch.repeat_interleave(St, deg)

    keep = in_patch[nbr]
    src = src[keep]
    dst = nbr[keep]
    w = w[keep]

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, w


def remap_edge_index_global_to_local(St, edge_index_global, N):
    device = St.device
    g2l = torch.full((N,), -1, dtype=torch.long, device=device)
    g2l[St] = torch.arange(St.numel(), device=device)
    return g2l[edge_index_global]


def remap_labels_to_0K(labels: torch.Tensor):
    uniq = torch.unique(labels)
    inv = torch.bucketize(labels, uniq)
    return inv, uniq.numel()


# ─────────────────────────────────────────────────────────────────────────────
# Sigma estimation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma0_sigma1_from_initlabels(y: torch.Tensor, initial_labels: torch.Tensor, eps=1e-12, sigma0_agg='median'):
    device = y.device
    n, d = y.shape

    labels, K = remap_labels_to_0K(initial_labels.to(device))

    ones = torch.ones(n, device=device, dtype=y.dtype)
    counts = torch.zeros(K, device=device, dtype=y.dtype).index_add_(0, labels, ones)
    counts_clamped = counts.clamp_min(1.0)

    sum_y = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y.index_add_(0, labels, y)
    mean_y = sum_y / counts_clamped.unsqueeze(1)

    sum_y2 = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y2.index_add_(0, labels, y * y)
    Ey2 = sum_y2 / counts_clamped.unsqueeze(1)
    var_within = (Ey2 - mean_y * mean_y).clamp_min(0.0)
    std_within = torch.sqrt(var_within + eps)

    if sigma0_agg == 'mean':
        sigma0_diag = std_within.mean(dim=0)
    else:
        sigma0_diag = std_within.median(dim=0).values
    sigma1_diag = mean_y.std(dim=0, unbiased=True)

    return sigma0_diag * 10, sigma1_diag, mean_y, var_within, counts


# ─────────────────────────────────────────────────────────────────────────────
# SVI utilities (shared between svi_constant and svi_faster)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def phi_kkt_grad_norm(phi, C, edge_index, edge_weight, T, eps=1e-12):
    device = phi.device
    dtype = phi.dtype
    T = float(T)

    phi_safe = phi.clamp_min(eps)
    log_phi = phi_safe.log()

    s = torch.zeros_like(phi)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        w = edge_weight.to(device=device, dtype=dtype).view(-1, 1)
        s.index_add_(0, dst, phi[src] * w)

    g = log_phi + 1.0 + (C / T) - (0.5 * s / T)
    g_proj = g - g.mean(dim=1, keepdim=True)

    grad_norm = g_proj.norm().item()
    max_row_norm = g_proj.norm(dim=1).max().item()
    return grad_norm, max_row_norm


def stochastic_cavi_patch_update_onepass(
    x_b: torch.Tensor,
    alpha_b_old: torch.Tensor,
    m0: torch.Tensor,
    rho0: torch.Tensor,
    sigma0_2: torch.Tensor,
    sigma1_2: torch.Tensor,
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    T: float,
    fp_steps: int = 10,
    eps: float = 1e-12,
) -> tuple:
    device = x_b.device
    dtype = x_b.dtype
    n_b, d = x_b.shape
    K = alpha_b_old.shape[1]
    T = float(T)

    inv_sigma0_2 = (1.0 / sigma0_2.to(device=device, dtype=dtype)).view(1, 1, d)

    x_ = x_b.view(n_b, 1, d)
    m_ = m0.view(1, K, d)
    s2_ = torch.exp(rho0).view(1, K, d)
    C = 0.5 * ((m_ * m_ + s2_ - 2.0 * x_ * m_) * inv_sigma0_2).sum(dim=-1)
    logits = -C

    phi = torch.softmax(alpha_b_old / T, dim=-1)
    s = torch.zeros_like(phi)

    if edge_index.numel() > 0:
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        w = edge_weight.to(device=device, dtype=dtype).view(-1, 1)
        for _ in range(fp_steps):
            s.zero_()
            s.index_add_(0, dst, phi[src] * w)
            phi = torch.softmax((logits + 0.5 * s) / T, dim=-1)
    else:
        phi = torch.softmax(logits / T, dim=-1)

    eps_phi = 1e-2
    phi = (1.0 - eps_phi) * phi + eps_phi / K
    alpha_new = (T * phi.clamp_min(eps).log())

    scale = float(n) / float(max(n_b, 1))
    S_k = (phi.sum(dim=0) * scale).clamp_min(eps)
    T_kd = (phi.t() @ x_b) * scale

    inv_sigma0_2_kd = (1.0 / sigma0_2.to(device=device, dtype=dtype)).view(1, d)
    inv_sigma1_2_kd = (1.0 / sigma1_2.to(device=device, dtype=dtype)).view(1, d)

    denom_m = (S_k.view(K, 1) * inv_sigma0_2_kd) + inv_sigma1_2_kd
    numer_m = (T_kd * inv_sigma0_2_kd)
    m_intermediate = numer_m / denom_m
    s2_intermediate = 1.0 / denom_m

    return alpha_new.detach(), m_intermediate.detach(), s2_intermediate.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Shared setup: data loading + patch construction
# ─────────────────────────────────────────────────────────────────────────────

def load_and_setup(cfg: dict, device: str) -> dict:
    """Load data and graph, build spatial patches. Returns all tensors needed by algorithms."""
    t = cfg['training']
    seed = t['seed']
    neighbors_num = t['neighbors_num']
    lambda_similarity = t['lambda_similarity']
    weights_scale = t['weights_scale']
    nx, ny = t['nx'], t['ny']

    base_dir = cfg['data']['base_dir']
    inter_dir = cfg['data']['intermediate_dir']

    dataset_id = cfg['data']['dataset_id']
    data_path  = f"{base_dir}/tensor_pca_xy_true_initial_{dataset_id}_seed_{seed}.pt"
    graph_path = f"{inter_dir}/weights_{neighbors_num}_lambda_{lambda_similarity}_{dataset_id}_seed_{seed}.pt"

    data = torch.load(data_path, map_location=device)
    coords = data["spatial"]
    data_pca = data["pca_data"]
    true_labels = data["true_labels"]
    initial_labels = data["initial_labels"]

    n = data_pca.shape[0]
    d = data_pca.shape[1]
    K = torch.unique(true_labels).numel()

    sigma0_agg = cfg.get('preprocessing', {}).get('sigma0_agg', 'median')
    sigma0, sigma1, mean_k, var_k, counts = estimate_sigma0_sigma1_from_initlabels(data_pca, initial_labels, sigma0_agg=sigma0_agg)
    sigma0_2 = sigma0 ** 2
    sigma1_2 = sigma1 ** 2

    graph = torch.load(graph_path, map_location=device)
    indptr = graph["indptr"].to(device)
    indices = graph["indices"].to(device)
    weights = graph["weights"].to(device) * weights_scale

    B = nx * ny
    x_edges = torch.linspace(coords[:, 0].min(), coords[:, 0].max(), steps=nx + 1, device=coords.device)
    y_edges = torch.linspace(coords[:, 1].min(), coords[:, 1].max(), steps=ny + 1, device=coords.device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)

    patch_indices = [None] * B
    for ix in range(nx):
        x_lo, x_hi = x_edges[ix], x_edges[ix + 1]
        in_x = (coords[:, 0] >= x_lo) & (coords[:, 0] < x_hi if ix < nx - 1 else coords[:, 0] <= x_hi)
        for iy in range(ny):
            y_lo, y_hi = y_edges[iy], y_edges[iy + 1]
            in_y = (coords[:, 1] >= y_lo) & (coords[:, 1] < y_hi if iy < ny - 1 else coords[:, 1] <= y_hi)
            mask = in_x & in_y
            b = ix * ny + iy
            patch_indices[b] = all_idx[mask]

    patch_edge_index_local = [None] * B
    patch_edge_weight = [None] * B
    for b in range(B):
        St = patch_indices[b]
        edge_g, w = build_patch_edges_from_csr(St, indptr, indices, weights, n)
        edge_l = remap_edge_index_global_to_local(St, edge_g, N=n)
        patch_edge_index_local[b] = edge_l
        patch_edge_weight[b] = w

    # Filter out empty patches to avoid zero-size alpha_b with None gradient
    non_empty = [i for i in range(B) if len(patch_indices[i]) > 0]
    patch_indices = [patch_indices[i] for i in non_empty]
    patch_edge_index_local = [patch_edge_index_local[i] for i in non_empty]
    patch_edge_weight = [patch_edge_weight[i] for i in non_empty]
    B = len(patch_indices)

    return {
        "coords": coords,
        "data_pca": data_pca,
        "true_labels": true_labels,
        "initial_labels": initial_labels,
        "n": n, "d": d, "K": K,
        "sigma0_2": sigma0_2, "sigma1_2": sigma1_2,
        "mean_k": mean_k, "var_k": var_k,
        "B": B,
        "patch_indices": patch_indices,
        "patch_edge_index_local": patch_edge_index_local,
        "patch_edge_weight": patch_edge_weight,
        "data_path": data_path,
        "graph_path": graph_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared output: save results + plot
# ─────────────────────────────────────────────────────────────────────────────

def save_and_plot(results: dict, coords, final_z_pred, K: int, output_folder: str,
                  fig_name: str, title: str, initial_labels_cpu=None):
    os.makedirs(output_folder, exist_ok=True)

    # save results
    save_name = results.pop("_save_name")
    save_path = os.path.join(output_folder, save_name)
    torch.save(results, save_path)
    print(f"Saved results to: {save_path}")

    # plot predictions
    coords_cpu = coords.detach().cpu().numpy()
    z_pred_cpu = final_z_pred.detach().cpu().numpy().astype(int)

    c1 = plt.get_cmap("tab20").colors
    c2 = plt.get_cmap("tab20b").colors
    cmap23 = ListedColormap(list(c1) + list(c2))

    plt.figure()
    plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=5, c=z_pred_cpu, cmap=cmap23)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.savefig(os.path.join(output_folder, fig_name))
    plt.close()

    # optionally plot initial labels
    if initial_labels_cpu is not None:
        init_cpu = initial_labels_cpu.numpy().astype(int) if hasattr(initial_labels_cpu, 'numpy') else initial_labels_cpu
        plt.figure()
        plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=5, c=init_cpu, cmap=cmap23)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial labels (K-means)")
        plt.savefig(os.path.join(output_folder, "initial_labels.png"))
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Shared grad-norm diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def compute_grad_norm_wo_graph(data_pca, alpha_all, m0, rho0, sigma0_2, sigma1_2, T_cur, eps):
    with torch.enable_grad():
        m0_g = m0.detach().clone().requires_grad_(True)
        rho0_g = rho0.detach().clone().requires_grad_(True)
        loss_wo_graph = global_neg_elbo_without_graph.__wrapped__(
            x=data_pca,
            alpha=alpha_all.detach() if hasattr(alpha_all, 'detach') else alpha_all,
            m=m0_g,
            rho=rho0_g,
            sigma0_2=sigma0_2,
            sigma1_2=sigma1_2,
            xi=None,
            T=T_cur,
            eps=eps,
        )
        loss_wo_graph.backward()
        g_m = m0_g.grad
        g_r = rho0_g.grad
        return torch.sqrt(g_m.pow(2).sum() + g_r.pow(2).sum()).item()
