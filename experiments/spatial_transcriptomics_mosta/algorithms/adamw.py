"""
AdamW baseline for spatial GMM clustering.
"""
import torch
import torch.nn as nn
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
    alg_cfg = cfg['algorithms']['adamw']

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
    beta1 = alg_cfg['beta1']
    beta2 = alg_cfg['beta2']
    adam_eps = alg_cfg['adam_eps']
    wd_alpha = alg_cfg['wd_alpha']
    wd_m0 = alg_cfg['wd_m0']
    wd_rho0 = alg_cfg['wd_rho0']

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

    output_folder = os.path.join(cfg['data']['output_dir'], f"adamw_baseline_{max_iter}_seed_{seed}")

    alpha_init = torch.full((n, K), 0.5 / (K - 1), device=device)
    alpha_init.scatter_(1, initial_labels.to(device).unsqueeze(1), 0.5)
    alpha_init += 5e-2 * torch.randn(n, K, device=device)
    alpha_all = nn.Parameter(alpha_init)
    m0 = nn.Parameter(mean_k.clone())
    rho0 = nn.Parameter(torch.log(var_k + 1e-12).clone())

    optimizer = torch.optim.AdamW(
        [
            {"params": [alpha_all], "lr": lr_alpha, "weight_decay": wd_alpha},
            {"params": [m0], "lr": lr_m0, "weight_decay": wd_m0},
            {"params": [rho0], "lr": lr_rho0, "weight_decay": wd_rho0},
        ],
        betas=(beta1, beta2),
        eps=adam_eps,
    )

    log_iter, log_objective, log_ari, log_time_passed, log_grad_norm = [], [], [], [], []
    final_z_pred = None
    time_start = time.time()

    for it in range(max_iter + 1):
        t_anneal = min(1.0, it / anneal_iters)
        T_cur = T_init * ((T / T_init) ** t_anneal)

        b = torch.randint(0, B, (1,), device=device).item()
        St = patch_indices[b]
        x_b = data_pca[St]
        edge_index = patch_edge_index_local[b]
        edge_weight = patch_edge_weight[b]

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
                grad_norm = compute_grad_norm_wo_graph(data_pca, alpha_all, m0, rho0, sigma0_2, sigma1_2, T_cur, eps)
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

        optimizer.zero_grad(set_to_none=True)
        alpha_b = alpha_all[St]

        loss = patch_neg_elbo_with_graph(
            x_b=x_b, alpha_b=alpha_b, m0=m0, rho0=rho0,
            sigma0_2=sigma0_2, sigma1_2=sigma1_2, n=n,
            edge_index=edge_index, edge_weight=edge_weight, T=T_cur, eps=eps,
        )
        loss.backward()
        optimizer.step()

    if final_z_pred is None:
        phi_all = torch.softmax(alpha_all / float(T_cur), dim=-1)
        final_z_pred = torch.argmax(phi_all, dim=-1)

    results = {
        "_save_name": f"results_adamw_{max_iter}_seed_{seed}_lrA_{lr_alpha:.0e}_lrM_{lr_m0:.0e}_lrR_{lr_rho0:.0e}.pt",
        "algorithm": "adamw",
        "nx": int(nx), "ny": int(ny), "B": int(B),
        "max_iter": int(max_iter),
        "neighbors_num": int(neighbors_num), "lambda_similarity": float(lambda_similarity),
        "weights_scale": float(weights_scale),
        "T": float(T), "T_init": float(T_init), "anneal_iters": int(anneal_iters),
        "print_every": int(print_every), "grad_every": int(grad_every),
        "lr_alpha": float(lr_alpha), "lr_m0": float(lr_m0), "lr_rho0": float(lr_rho0),
        "adam_beta1": float(beta1), "adam_beta2": float(beta2), "adam_eps": float(adam_eps),
        "wd_alpha": float(wd_alpha), "wd_m0": float(wd_m0), "wd_rho0": float(wd_rho0),
        "data_path": str(data_path), "graph_path": str(graph_path),
        "log_iter": log_iter, "objective": log_objective, "ari": log_ari,
        "time_passed": log_time_passed,
        "grad_norm_wo_graph_wrt_m0_rho0": log_grad_norm,
        "final_z_pred": final_z_pred.detach().cpu(),
        "final_m0": m0.detach().cpu(),
        "final_rho0": rho0.detach().cpu(),
    }

    save_and_plot(results, coords, final_z_pred, K, output_folder,
                  "final_labels.png", "Predicted labels (AdamW baseline)")
