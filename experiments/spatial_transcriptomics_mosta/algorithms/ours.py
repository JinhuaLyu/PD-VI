"""
Our method: Federated ADMM for spatial GMM clustering.
"""
import torch
import time
import os
from sklearn.metrics import adjusted_rand_score as sk_ari

from .objectives import (
    global_neg_elbo_with_patch_graph,
    global_neg_elbo_without_graph,
    load_and_setup,
    save_and_plot,
    compute_grad_norm_wo_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm-specific functions
# ─────────────────────────────────────────────────────────────────────────────

def alpha_update(data_batch, m, rho, sigma0_2, alpha, edge_index, edge_weight, T, inner_fp_steps=10, eps=1e-12):
    device = data_batch.device
    x = data_batch
    n_b, d = x.shape
    K = alpha.shape[1]

    inv_sigma0_2 = (1.0 / sigma0_2.to(device)).view(1, 1, d)
    x_ = x.view(n_b, 1, d)
    m_ = m.view(1, K, d)
    rho_ = rho.view(1, K, d)
    C = 0.5 * torch.sum(
        (m_ * m_ + torch.exp(rho_) - 2.0 * x_ * m_) * inv_sigma0_2,
        dim=-1
    )
    logits = -C

    phi = torch.softmax(alpha / T, dim=-1)
    s = torch.zeros_like(phi)
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    w = edge_weight.to(device)

    w_col = w.view(-1, 1)
    for _ in range(inner_fp_steps):
        s.zero_()
        s.index_add_(0, dst, phi[src] * w_col)
        phi = torch.softmax((logits + 0.5 * s) / T, dim=-1)

    alpha_new = (logits + 0.5 * s).detach()
    phi = torch.softmax(alpha_new / T, dim=-1)
    eps_phi = 1e-2
    phi = (1 - eps_phi) * phi + eps_phi / K
    alpha_new = (T * phi.log()).detach()

    return alpha_new, phi


def local_update(mu_batch, gamma_batch, data_batch, alpha_batch, m0, rho0,
                 edge_index, edge_weight, n, eta_m, eta_s, device,
                 sigma0_2, sigma1_2, T, num_steps=10, newton_steps=5):
    K = m0.shape[0]
    m = m0.unsqueeze(0).clone()
    d = data_batch.shape[1]
    n_b = data_batch.shape[0]
    rho = rho0.unsqueeze(0).clone()
    alpha = alpha_batch.clone()
    x = data_batch

    inv_sigma0_2 = (1.0 / sigma0_2).view(1, d)
    inv_eta_m = 1.0 / eta_m
    inv_eta_s = 1.0 / eta_s
    inv_sigma1_2 = (1.0 / sigma1_2).view(1, d)
    scale = float(n_b) / float(n)
    inv_nb_over_n_sigma1_2 = (scale * inv_sigma1_2)

    for it in range(num_steps):
        alpha, phi = alpha_update(data_batch, m, rho, sigma0_2, alpha, edge_index, edge_weight, T)

        S_k = phi.sum(dim=0)
        T_kd = phi.t() @ x

        denom = (S_k.view(K, 1) * inv_sigma0_2) + inv_nb_over_n_sigma1_2 + inv_eta_m
        numer = (T_kd * inv_sigma0_2) + (m0 * inv_eta_m) - mu_batch
        m = (numer / denom).unsqueeze(0)

        Sk = S_k.view(K, 1)
        A = 0.5 * (Sk * inv_sigma0_2 + inv_nb_over_n_sigma1_2)
        const = -0.5 * scale
        rho_kd = rho.squeeze(0)
        for _ in range(newton_steps):
            exp_rho = torch.exp(rho_kd)
            g = exp_rho * A + const + gamma_batch + (rho_kd - rho0) * inv_eta_s
            gp = exp_rho * A + inv_eta_s
            rho_kd = rho_kd - g / gp
        rho = rho_kd.unsqueeze(0)

    return alpha, m.squeeze(0), rho.squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# run(cfg) entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    t = cfg['training']
    alg_cfg = cfg['algorithms']['ours']

    seed = t['seed']
    device = t['device']
    max_iter = t['max_iter']
    print_every = t['print_every']
    grad_every = t['grad_every']
    nx, ny = t['nx'], t['ny']
    T = t['T']
    T_init = t['T_init']
    anneal_iters = t['anneal_iters']
    neighbors_num = t['neighbors_num']
    lambda_similarity = t['lambda_similarity']
    weights_scale = t['weights_scale']
    eps = 1e-12

    eta_m = alg_cfg['eta_m']
    eta_s = alg_cfg['eta_s']
    num_steps = alg_cfg['num_steps']
    newton_steps = alg_cfg['newton_steps']

    inner_seed = seed + 20
    torch.manual_seed(inner_seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    s = load_and_setup(cfg, device)
    coords = s['coords']
    data_pca = s['data_pca']
    true_labels = s['true_labels']
    initial_labels = s['initial_labels']
    n, d, K = s['n'], s['d'], s['K']
    sigma0_2, sigma1_2 = s['sigma0_2'], s['sigma1_2']
    mean_k, var_k = s['mean_k'], s['var_k']
    B = s['B']
    patch_indices = s['patch_indices']
    patch_edge_index_local = s['patch_edge_index_local']
    patch_edge_weight = s['patch_edge_weight']
    data_path, graph_path = s['data_path'], s['graph_path']

    initial_labels_cpu = initial_labels.detach().cpu()

    output_folder = os.path.join(cfg['data']['output_dir'], f"preconditioned_{max_iter}_seed_{seed}")

    alpha_all = 1e-2 * torch.randn(n, K, device=device)
    m0 = mean_k
    rho0 = torch.log(var_k + 1e-12)

    mu = torch.zeros(B, K, d, device=device)
    gamma = torch.zeros(B, K, d, device=device)
    h_t_m = torch.zeros(K, d, device=device)
    h_t_rho = torch.zeros(K, d, device=device)

    m = m0.clone()
    rho = rho0.clone()
    m0_prev = m0.clone() + 1000
    rho0_prev = rho0.clone() + 1000

    log_iter, log_objective, log_ari = [], [], []
    log_residual, log_delta_consensus = [], []
    log_time_passed, log_grad_norm = [], []
    final_z_pred = None
    time_start = time.time()

    for it in range(max_iter + 1):
        t_anneal = min(1.0, it / anneal_iters)
        T_cur = T_init * (1.0 - t_anneal) + T * t_anneal

        if it % print_every == 0:
            neg_elbo = global_neg_elbo_with_patch_graph(
                x=data_pca, alpha=alpha_all, m=m0, rho=rho0,
                sigma0_2=sigma0_2, sigma1_2=sigma1_2,
                patch_indices=patch_indices,
                patch_edge_index_local=patch_edge_index_local,
                patch_edge_weight=patch_edge_weight,
                xi=None, T_for_phi=T_cur,
            )

            phi = torch.softmax(alpha_all / T_cur, dim=-1)
            z_pred = torch.argmax(phi, dim=-1)
            final_z_pred = z_pred.detach().clone()
            ari = sk_ari(true_labels.detach().cpu().numpy(), z_pred.detach().cpu().numpy())

            rel_m = torch.linalg.norm(m - m0) / (torch.linalg.norm(m0) + eps)
            rel_r = torch.linalg.norm(rho - rho0) / (torch.linalg.norm(rho0) + eps)
            primal_residual = (rel_m + rel_r).item()

            delta_m0 = torch.linalg.norm(m0 - m0_prev) / (torch.linalg.norm(m0_prev) + eps)
            delta_r0 = torch.linalg.norm(rho0 - rho0_prev) / (torch.linalg.norm(rho0_prev) + eps)
            delta = (delta_m0 + delta_r0).item()

            time_passed = time.time() - time_start

            if it % grad_every == 0:
                grad_norm = compute_grad_norm_wo_graph(data_pca, alpha_all, m0, rho0, sigma0_2, sigma1_2, T_cur, eps)
            else:
                grad_norm = float("nan")

            print(
                f"iter {it}/{max_iter}, objective={neg_elbo.item():.6e}, ari={ari:.6e}, "
                f"residual={primal_residual:.6e}, delta_consensus={delta:.6e}, "
                f"time={time_passed:.2f}s, T={T_cur:.6f}"
            )
            if it % grad_every == 0:
                print(f"  grad_norm(m0,rho0) w/o graph: {grad_norm:.6e}")

            log_iter.append(int(it))
            log_objective.append(float(neg_elbo.item()))
            log_ari.append(float(ari))
            log_residual.append(float(primal_residual))
            log_delta_consensus.append(float(delta))
            log_time_passed.append(float(time_passed))
            log_grad_norm.append(float(grad_norm))

            m0_prev = m0.clone()
            rho0_prev = rho0.clone()

        b = torch.randint(0, B, (1,), device=device).item()
        mu_batch = mu[b]
        gamma_batch = gamma[b]
        St = patch_indices[b]
        data_batch = data_pca[St]
        alpha_batch = alpha_all[St]
        edge_index = patch_edge_index_local[b]
        edge_weight = patch_edge_weight[b]

        alpha, m, rho = local_update(
            mu_batch, gamma_batch, data_batch, alpha_batch,
            m0, rho0, edge_index, edge_weight, n,
            eta_m, eta_s, device, sigma0_2, sigma1_2, T_cur,
            num_steps=num_steps, newton_steps=newton_steps,
        )

        alpha_all[St] = alpha
        m_diff = m - m0
        rho_diff = rho - rho0

        mu[b] += m_diff / eta_m
        gamma[b] += rho_diff / eta_s

        h_t_m += m_diff / B
        h_t_rho += rho_diff / B
        m0 = m + h_t_m
        rho0 = rho + h_t_rho

    if final_z_pred is None:
        phi_final = torch.softmax(alpha_all / T_cur, dim=-1)
        final_z_pred = torch.argmax(phi_final, dim=-1)

    results = {
        "_save_name": f"results_ours_{max_iter}_seed_{seed}_eta_m_{eta_m:.0e}_eta_s_{eta_s:.0e}.pt",
        "algorithm": "ours",
        "eta_m": float(eta_m), "eta_s": float(eta_s),
        "nx": int(nx), "ny": int(ny), "B": int(B),
        "max_iter": int(max_iter),
        "neighbors_num": int(neighbors_num),
        "lambda_similarity": float(lambda_similarity),
        "weights_scale": float(weights_scale),
        "T": float(T), "T_init": float(T_init), "anneal_iters": int(anneal_iters),
        "print_every": int(print_every), "grad_every": int(grad_every),
        "data_path": str(data_path), "graph_path": str(graph_path),
        "log_iter": log_iter, "objective": log_objective, "ari": log_ari,
        "residual": log_residual, "delta_consensus": log_delta_consensus,
        "time_passed": log_time_passed,
        "grad_norm_wo_graph_wrt_m0_rho0": log_grad_norm,
        "final_z_pred": final_z_pred.detach().cpu(),
        "final_m0": m0.detach().cpu(),
        "final_rho0": rho0.detach().cpu(),
    }

    save_and_plot(results, coords, final_z_pred, K, output_folder,
                  "final_labels.png", "Final labels (Ours - Federated ADMM)",
                  initial_labels_cpu=initial_labels_cpu)
