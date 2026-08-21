"""
SGD + CV (SAGA-style control variates) baseline for spatial GMM clustering.
"""
import torch
import time
import os
from sklearn.metrics import adjusted_rand_score as sk_ari

from .objectives import (
    global_neg_elbo_with_patch_graph,
    global_neg_elbo_without_graph,
    patch_neg_elbo_with_graph,
    load_and_setup,
    save_and_plot,
    compute_grad_norm_wo_graph,
)


def run(cfg: dict):
    t = cfg['training']
    alg_cfg = cfg['algorithms']['cv']

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

    lr_alpha = alg_cfg['lr_alpha']
    lr_m0 = alg_cfg['lr_m0']
    lr_rho0 = alg_cfg['lr_rho0']

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    s = load_and_setup(cfg, device)
    coords = s['coords']
    data_pca = s['data_pca']
    true_labels = s['true_labels']
    n, d, K = s['n'], s['d'], s['K']
    sigma0_2, sigma1_2 = s['sigma0_2'], s['sigma1_2']
    mean_k, var_k = s['mean_k'], s['var_k']
    B = s['B']
    patch_indices = s['patch_indices']
    patch_edge_index_local = s['patch_edge_index_local']
    patch_edge_weight = s['patch_edge_weight']
    data_path, graph_path = s['data_path'], s['graph_path']
    initial_labels = s['initial_labels']

    output_folder = os.path.join(cfg['data']['output_dir'], f"sgd_cv_{max_iter}_seed_{seed}")

    alpha_init = torch.full((n, K), 0.5 / (K - 1), device=device, dtype=data_pca.dtype)
    alpha_init.scatter_(1, initial_labels.to(device).unsqueeze(1), 0.5)
    alpha_init += 5e-2 * torch.randn(n, K, device=device, dtype=data_pca.dtype)
    alpha_all = alpha_init
    m0 = mean_k.clone()
    rho0 = torch.log(var_k + 1e-12).clone()

    # SAGA state for globals (m0, rho0)
    gm_table = torch.zeros(B, K, d, device=device, dtype=data_pca.dtype)
    gr_table = torch.zeros(B, K, d, device=device, dtype=data_pca.dtype)
    Gm = torch.zeros(K, d, device=device, dtype=data_pca.dtype)
    Gr = torch.zeros(K, d, device=device, dtype=data_pca.dtype)

    def patch_grad_wrt_globals(b_idx, m0_cur, rho0_cur, T_use):
        St = patch_indices[b_idx]
        x_b = data_pca[St]
        edge_index = patch_edge_index_local[b_idx]
        edge_weight = patch_edge_weight[b_idx]
        alpha_b_fix = alpha_all[St].detach()

        m0_g = m0_cur.detach().clone().requires_grad_(True)
        rho0_g = rho0_cur.detach().clone().requires_grad_(True)

        loss_g = patch_neg_elbo_with_graph(
            x_b=x_b, alpha_b=alpha_b_fix, m0=m0_g, rho0=rho0_g,
            sigma0_2=sigma0_2, sigma1_2=sigma1_2, n=n,
            edge_index=edge_index, edge_weight=edge_weight, T=T_use, eps=eps,
        )
        g_m, g_r = torch.autograd.grad(loss_g, [m0_g, rho0_g], create_graph=False)
        return g_m.detach(), g_r.detach()

    print("[CV init] building gradient table for all patches...")
    for b0 in range(B):
        gmb, grb = patch_grad_wrt_globals(b0, m0, rho0, T_init)
        gm_table[b0] = gmb
        gr_table[b0] = grb
    Gm = gm_table.mean(dim=0)
    Gr = gr_table.mean(dim=0)
    print("[CV init] done.")

    log_iter, log_objective, log_ari, log_time_passed, log_grad_norm = [], [], [], [], []
    final_z_pred = None
    time_start = time.time()

    for it in range(max_iter + 1):
        t_anneal = min(1.0, it / anneal_iters)
        T_cur = T_init * (1.0 - t_anneal) + T * t_anneal

        b = torch.randint(0, B, (1,), device=device).item()
        St = patch_indices[b]
        x_b = data_pca[St]
        edge_index = patch_edge_index_local[b]
        edge_weight = patch_edge_weight[b]

        alpha_b = alpha_all[St].detach().clone().requires_grad_(True)
        m0_tmp = m0.detach().clone().requires_grad_(True)
        rho0_tmp = rho0.detach().clone().requires_grad_(True)

        loss = patch_neg_elbo_with_graph(
            x_b=x_b, alpha_b=alpha_b, m0=m0_tmp, rho0=rho0_tmp,
            sigma0_2=sigma0_2, sigma1_2=sigma1_2, n=n,
            edge_index=edge_index, edge_weight=edge_weight, T=T_cur, eps=eps,
        )
        g_alpha, g_m, g_r = torch.autograd.grad(loss, [alpha_b, m0_tmp, rho0_tmp], create_graph=False)

        with torch.no_grad():
            alpha_all[St] = alpha_b - lr_alpha * g_alpha

            g_old_m = gm_table[b]
            g_old_r = gr_table[b]
            g_hat_m = g_m - g_old_m + Gm
            g_hat_r = g_r - g_old_r + Gr

            m0 = m0 - lr_m0 * g_hat_m
            rho0 = rho0 - lr_rho0 * g_hat_r

            Gm = Gm + (g_m - g_old_m) / float(B)
            Gr = Gr + (g_r - g_old_r) / float(B)
            gm_table[b] = g_m
            gr_table[b] = g_r

        if it % print_every == 0:
            with torch.no_grad():
                global_obj = global_neg_elbo_with_patch_graph(
                    x=data_pca, alpha=alpha_all, m=m0, rho=rho0,
                    sigma0_2=sigma0_2, sigma1_2=sigma1_2,
                    patch_indices=patch_indices,
                    patch_edge_index_local=patch_edge_index_local,
                    patch_edge_weight=patch_edge_weight,
                    xi=None, T_for_phi=1e-4,
                )
                phi_all = torch.softmax(alpha_all / float(T_cur), dim=-1)
                z_pred = torch.argmax(phi_all, dim=-1)
                final_z_pred = z_pred.detach().clone()
                ari = sk_ari(true_labels.detach().cpu().numpy(), z_pred.detach().cpu().numpy())
                time_passed = time.time() - time_start

            if it % grad_every == 0:
                grad_norm = compute_grad_norm_wo_graph(data_pca, alpha_all, m0, rho0, sigma0_2, sigma1_2, 1e-4, eps)
            else:
                grad_norm = float("nan")

            print(
                f"iter {it}/{max_iter}, objective={global_obj.item():.6e}, ari={ari:.6e}, "
                f"time={time_passed:.2f}s, T={T_cur:.6f}"
            )
            if it % grad_every == 0:
                print(f"  grad_norm(m0,rho0) w/o graph: {grad_norm:.6e}")

            log_iter.append(int(it))
            log_objective.append(float(global_obj.item()))
            log_ari.append(float(ari))
            log_time_passed.append(float(time_passed))
            log_grad_norm.append(float(grad_norm))

    if final_z_pred is None:
        phi_all = torch.softmax(alpha_all / float(T_cur), dim=-1)
        final_z_pred = torch.argmax(phi_all, dim=-1)

    results = {
        "_save_name": f"results_cv_{max_iter}_seed_{seed}_lrA_{lr_alpha:.0e}_lrM_{lr_m0:.0e}_lrR_{lr_rho0:.0e}.pt",
        "algorithm": "cv",
        "nx": int(nx), "ny": int(ny), "B": int(B),
        "max_iter": int(max_iter),
        "neighbors_num": int(neighbors_num), "lambda_similarity": float(lambda_similarity),
        "weights_scale": float(weights_scale),
        "T": float(T), "T_init": float(T_init), "anneal_iters": int(anneal_iters),
        "print_every": int(print_every), "grad_every": int(grad_every),
        "lr_alpha": float(lr_alpha), "lr_m0": float(lr_m0), "lr_rho0": float(lr_rho0),
        "data_path": str(data_path), "graph_path": str(graph_path),
        "log_iter": log_iter, "objective": log_objective, "ari": log_ari,
        "time_passed": log_time_passed,
        "grad_norm_wo_graph_wrt_m0_rho0": log_grad_norm,
        "final_z_pred": final_z_pred.detach().cpu(),
        "final_m0": m0.detach().cpu(),
        "final_rho0": rho0.detach().cpu(),
    }

    save_and_plot(results, coords, final_z_pred, K, output_folder,
                  "final_labels.png", "Predicted labels (SGD + CV on globals)")
