"""
Unified driver for the moon clustering experiment (P^2D-VI vs SVI vs PD-VI vs AdamW).

Model (identical across all methods, ported from
    /home/fzd2816/feddyn/mfvi_syn_final/main.py
    /home/fzd2816/feddyn/moon_test/kmeanspp_test/{ours_kpp.py, svi_gmm_kpp.py}):

    x_n | z_n=k  ~  N(mu_k, diag(sigma0^2))     shared, known sigma0
    mu_k         ~  N(0,    diag(sigma1^2))     Gaussian prior on means
    pi_k = 1/K    (not learned)
    q(mu_k) = N(m_k, diag(exp(rho_k)))          diagonal-Gaussian variational
    q(z_n)  = Cat(softmax(alpha_n))

Initialization (all methods, all seeds): k-means++ on the data.
    m0      = k-means++ centers
    rho0    = log(Var[y])               (broadcast over clusters)
    alpha   = +2 on the k-means++ label, -2 elsewhere
    sigma0  = mean over k-means++ clusters of within-cluster std
    sigma1  = std of the k-means++ centers          (clamped >= 0.1)

Every method records, every LOG_EVERY iterations:
    iter, true ELBO (higher is better), ARI, wall-clock time, and the global
    cluster means m0 (K, d) so the Wasserstein trajectory can be reconstructed.

AdamW note: the reference Adam/SGD/RMSprop block in main.py reverted the
optimizer's update to `alpha` via `alpha_all[St] = alpha_batch` after
`optimizer.step()`, which froze the assignment logits. We drop that line so
AdamW genuinely optimizes (alpha, m0, rho0) jointly. weight_decay=0.01 is
applied to all three parameter groups (the user-selected default); this also
decays the logits / log-variances, which is documented in TUNING.md.
"""
import os
import math
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

import pavi

# ----------------------------------------------------------------------------
# paths / global config
# ----------------------------------------------------------------------------
import os as _os_for_root
ROOT          = _os_for_root.path.dirname(_os_for_root.path.abspath(__file__))
DATA_FILE     = f"{ROOT}/data/moon_data.pt"
TRUE_GMM_FILE = f"{ROOT}/data/cluster_gaussians.pt"
OUTPUT_DIR    = f"{ROOT}/output"

SEEDS        = [42, 1, 2, 3, 4]   # 42 reproduces the existing single-seed moon run
SAMPLE_SIZE  = 500
MAX_ITER     = int(os.environ.get("MAX_ITER", "10000"))   # env override for smoke tests
T            = 1.0
LOG_EVERY    = 10
RESTART_EVERY = 100
# numerical guard on the variational log-variance for the gradient baselines.
# Plain SGD/CV with a single lr push rho0 to +inf (the variance term self-amplifies);
# this clamp keeps ELBO finite and comparable. AdamW/RMSProp stay well inside it.
RHO_MIN, RHO_MAX = math.log(1e-10), math.log(1e2)
GUARD_NTRY = 25       # random_guarded init: candidate draws kept for best separation
INNER_STEPS  = 10
NEWTON_STEPS = 5

# tuned hyper-parameters carried over from moon_test/kmeanspp_test (do not retune)
P2D_ETA_M = 1e-3      # ours_precondition, mean step
P2D_ETA_S = 1e0       # ours_precondition, log-var step
SVI_TAU   = 1.0       # SVI step-size schedule rho_t = (t + tau)^(-kappa) * eta_svi
SVI_KAPPA = 0.9
SVI_ETA   = 1.0

# PAVI (particle mean-field VI) baseline: particles per factor, context batch, and
# the Euler-Langevin step-size grid auto-tuned on seed 42 per (init, data_mode),
# cached so later seeds reuse the same step (mirrors the lr sweep of the other
# gradient baselines). See pavi.py for the algorithm.
PAVI_P, PAVI_B = 64, 16
PAVI_H_GRID = (1e-6, 3e-6, 1e-5, 3e-5)
_PAVI_H_CACHE = {}

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# initialization mode for the whole run, selected via env var so the same
# scripts produce either the k-means++ or the random-init variant.
#   INIT_MODE=kpp     (default) -> outputs use no suffix
#   INIT_MODE=random            -> outputs use the "_random" suffix
INIT_MODE = os.environ.get("INIT_MODE", "kpp")
SUFFIX    = "" if INIT_MODE == "kpp" else f"_{INIT_MODE}"

# ----------------------------------------------------------------------------
# data (fixed across seeds)
# ----------------------------------------------------------------------------
_data   = torch.load(DATA_FILE)
Y       = _data["X"].to(DEVICE).double()
Z_TRUE  = _data["label_cluster"].to(DEVICE)
N, D    = Y.shape
K       = int(Z_TRUE.max().item()) + 1

_truth      = torch.load(TRUE_GMM_FILE)
TRUE_MEANS  = _truth["means"].double().cpu().numpy()       # (K, d)
TRUE_COVS   = _truth["covs"].double().cpu().numpy()        # (K, d, d)
TRUE_WEIGHTS = _truth["weights"].double().cpu().numpy()    # (K,)

Y_MEAN = Y.mean(dim=0)
Y_STD  = Y.std(dim=0)
Y_VAR  = Y.var(dim=0)


# ============================================================================
# model helpers (verbatim port; shared by all methods)
# ============================================================================
def make_fixed_batches_stratified(z_true, batch_size, seed=0):
    dev = z_true.device
    n_  = z_true.numel()
    assert n_ % batch_size == 0
    num_batches_ = n_ // batch_size
    g = torch.Generator(device=dev); g.manual_seed(seed)
    K_ = int(z_true.max().item()) + 1
    chunks = []
    for k in range(K_):
        idx_k = torch.nonzero(z_true == k, as_tuple=False).view(-1)
        idx_k = idx_k[torch.randperm(idx_k.numel(), generator=g, device=dev)]
        chunks.append(idx_k)
    all_idx = torch.cat(chunks, dim=0).view(num_batches_, batch_size)
    return [all_idx[b].clone() for b in range(num_batches_)]


def unpack_lambda(lambda_0, K, d):
    m0   = lambda_0[:K * d].view(K, d)
    rho0 = lambda_0[K * d:].view(K, d)
    return m0, rho0


def newton_rho(rho, A, gamma, rho0, inv_eta, n_total, num_newton=5):
    c = gamma - 0.5 / n_total
    for _ in range(num_newton):
        exp_rho = torch.exp(rho)
        f  = A * exp_rho + c + inv_eta * (rho - rho0)
        fp = A * exp_rho + inv_eta
        rho = rho - f / fp
    return rho


def compute_const_term_diag(y, sigma0_diag, sigma1_diag, K):
    n_, d_ = y.shape
    inv_sigma0_2      = 1.0 / (sigma0_diag ** 2)
    y_sq_weighted_sum = (y ** 2 * inv_sigma0_2.view(1, d_)).sum().item()
    logdet_sigma0_2   = (2.0 * torch.log(sigma0_diag)).sum().item()
    logdet_sigma1_2   = (2.0 * torch.log(sigma1_diag)).sum().item()
    const1 = n_ * (0.5 * d_ * math.log(2.0 * math.pi) + 0.5 * logdet_sigma0_2 + math.log(K))
    const2 = 0.5 * y_sq_weighted_sum
    const3 = (K / 2.0) * (logdet_sigma1_2 - d_)
    return const1 + const2 + const3


def elbo_batch_diag(alpha_batch, lambda_0, y, sigma0_diag, sigma1_diag, n_total):
    """Returns the NEGATIVE-ELBO surrogate; true ELBO = -(this + const_term)."""
    B, K_ = alpha_batch.shape
    d_    = y.shape[1]
    m0, rho0 = unpack_lambda(lambda_0, K_, d_)
    s2       = torch.exp(rho0)
    phi      = torch.softmax(alpha_batch, dim=-1)
    log_phi  = torch.log_softmax(alpha_batch, dim=-1)

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d_)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, d_)

    y_  = y.view(B, 1, d_)
    m_  = m0.view(1, K_, d_)
    s2_ = s2.view(1, K_, d_)

    data_term = 0.5 * ((m_ * m_ + s2_ - 2.0 * y_ * m_) * inv_sigma0_2).sum(dim=-1)
    vi_term   = (-0.5 / n_total) * rho0.sum(dim=-1) \
              + (0.5 / n_total) * ((m0 * m0 + s2) * inv_sigma1_2).sum(dim=-1)
    vi_term   = vi_term.view(1, K_)

    inside = phi * (log_phi + data_term) + vi_term
    return inside.sum(dim=-1).sum()


def local_update_diag(mu_batch, y_batch, alpha_batch, lambda_0,
                      B, K, d_lambda, eta_m, eta_s, n_total, device,
                      sigma0_diag, sigma1_diag, T,
                      num_steps=10, newton_steps=5):
    """Primal-dual local inner solve (shared by PD-VI and P^2D-VI)."""
    d_ = y_batch.shape[1]
    assert d_lambda == 2 * K * d_

    mu_m  = mu_batch[:, :K * d_].view(B, K, d_)
    gamma = mu_batch[:, K * d_:].view(B, K, d_)

    m0, rho0 = unpack_lambda(lambda_0, K, d_)
    m   = m0.view(1, K, d_).expand(B, K, d_).clone()
    rho = rho0.view(1, K, d_).expand(B, K, d_).clone()
    alpha = alpha_batch.to(device).clone()

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d_)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, 1, d_)
    inv_eta_m    = 1.0 / eta_m
    inv_eta_s    = 1.0 / eta_s

    y_ = y_batch.view(B, 1, d_)

    for _ in range(num_steps):
        exp_rho = torch.exp(rho)
        C   = 0.5 * ((m * m + exp_rho - 2.0 * y_ * m) * inv_sigma0_2).sum(dim=-1)
        alpha = -C
        phi   = torch.softmax(alpha / T, dim=-1)

        phi_     = phi.unsqueeze(-1)
        denom_m  = phi_ * inv_sigma0_2 + (1.0 / n_total) * inv_sigma1_2 + inv_eta_m
        numer_m  = phi_ * (y_ * inv_sigma0_2) + (m0.view(1, K, d_) * inv_eta_m) - mu_m
        m        = numer_m / denom_m

        A   = phi_ * (0.5 * inv_sigma0_2) + 0.5 * (1.0 / n_total) * inv_sigma1_2
        rho = newton_rho(rho, A, gamma, rho0.view(1, K, d_), inv_eta_s, n_total, num_newton=newton_steps)

    lambda_batch_out = torch.cat([m.reshape(B, K * d_), rho.reshape(B, K * d_)], dim=1)
    return alpha, lambda_batch_out


def svi_minibatch_update_diag(y_batch, alpha_batch_old, m0, rho0,
                              sigma0_diag, sigma1_diag, n_total, T, eps=1e-12):
    dtype = m0.dtype
    y_batch = y_batch.to(dtype=dtype)
    B, d_ = y_batch.shape
    K_    = m0.shape[0]
    T     = float(T)

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d_)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, d_)

    y_  = y_batch.view(B, 1, d_)
    m_  = m0.view(1, K_, d_)
    s2_ = torch.exp(rho0).view(1, K_, d_)

    C      = 0.5 * ((m_ * m_ + s2_ - 2.0 * y_ * m_) * inv_sigma0_2).sum(dim=-1)
    logits = -C
    phi    = torch.softmax(logits / T, dim=-1)
    phi_safe        = phi.clamp_min(eps)
    alpha_batch_new = T * phi_safe.log()

    scale = float(n_total) / float(max(B, 1))
    S_k   = (phi.sum(dim=0) * scale).clamp_min(eps)
    T_kd  = (phi.t() @ y_batch) * scale

    inv_sigma0_2_kd = inv_sigma0_2.view(1, d_)
    denom = (S_k.view(K_, 1) * inv_sigma0_2_kd) + inv_sigma1_2
    numer = T_kd * inv_sigma0_2_kd
    m_intermediate  = numer / denom
    s2_intermediate = 1.0 / denom
    return alpha_batch_new, m_intermediate, s2_intermediate


def _pavi_synth_alpha(m0, rho0, y, sigma0_diag):
    """Optimal q(z) logits for the parametric model given (m0, rho0).

    PAVI marginalizes z, so it has no q(z). To report the SAME ELBO and ARI as
    the other methods, we project the particle clouds onto the parametric family:
    m0 = particle mean, rho0 = log particle variance, and the q(z) logits are the
    optimal responsibilities at that point, alpha = -C. C is identical to the one
    in svi_minibatch_update_diag, so the projected ELBO is on the same scale as
    every other method's ELBO."""
    K_, d_ = m0.shape
    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d_)
    y_  = y.view(-1, 1, d_)
    m_  = m0.view(1, K_, d_)
    s2_ = torch.exp(rho0).view(1, K_, d_)
    C = 0.5 * ((m_ * m_ + s2_ - 2.0 * y_ * m_) * inv_sigma0_2).sum(dim=-1)
    return -C


# ============================================================================
# per-seed k-means++ initialization
# ============================================================================
def build_init(seed, init_mode="kpp", sigma_source="kpp"):
    """Per-seed initialization shared by every method.

    sigma0/sigma1 always come from k-means++ (computed below regardless of
    init_mode). The starting point (m0, rho0, alpha) depends on init_mode:
      'kpp'    : m0 = k-means++ centers, alpha = +-2 on the k-means++ labels.
      'random' : m0 = y_mean + 0.5 y_std * N(0,1), alpha = 1e-2 * N(0,1)
                 (the standard random init from mfvi_syn_final/main.py); only the
                 sigma scales are borrowed from k-means++, not the cluster labels.
    For 'random' the torch RNG must be seeded by the caller before this call.
    """
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed).fit(Y.cpu().numpy())
    kpp_labels  = torch.from_numpy(km.labels_).to(DEVICE)
    kpp_centers = torch.from_numpy(km.cluster_centers_).to(DEVICE).double()

    if sigma_source == "kpp":
        cluster_stds = torch.stack([Y[kpp_labels == k].std(dim=0) for k in range(K)])
        sigma0_diag  = cluster_stds.mean(dim=0).clamp(min=1e-6)
        sigma1_diag  = kpp_centers.std(dim=0).clamp(min=0.1)
    elif sigma_source == "gt":
        # ground-truth scales: sigma0 = pooled per-cluster std (sqrt of the mean
        # diagonal covariance), sigma1 = std of the true means across clusters.
        tc = torch.as_tensor(TRUE_COVS, device=DEVICE)               # (K, D, D)
        tm = torch.as_tensor(TRUE_MEANS, device=DEVICE)              # (K, D)
        sigma0_diag = tc.diagonal(dim1=-2, dim2=-1).mean(dim=0).clamp(min=1e-12).sqrt()
        sigma1_diag = tm.std(dim=0).clamp(min=0.1)
    else:
        raise ValueError(f"unknown sigma_source {sigma_source}")

    if init_mode == "kpp":
        m0_init   = kpp_centers.clone()
        rho0_init = torch.log(Y_VAR + 1e-12).view(1, D).expand(K, D).clone()
        alpha_init = torch.full((N, K), -2.0, device=DEVICE)
        alpha_init[torch.arange(N, device=DEVICE), kpp_labels] = 2.0
    elif init_mode == "random":
        m0_init   = Y_MEAN.view(1, D) + 0.5 * Y_STD.view(1, D) * torch.randn(K, D, device=DEVICE)
        rho0_init = torch.log(Y_VAR + 1e-12).view(1, D) + 1e-2 * torch.randn(K, D, device=DEVICE)
        alpha_init = 1e-2 * torch.randn(N, K, device=DEVICE)
    elif init_mode == "random_guarded":
        # still a random Gaussian init, but rejection-sampled for separation: out
        # of GUARD_NTRY draws keep the one whose closest pair of means is farthest
        # apart, so no two clusters start on top of each other. This removes the
        # collapsed-mean bad basins that make plain random init bimodal.
        best_m0, best_sep = None, -1.0
        for _ in range(GUARD_NTRY):
            cand = Y_MEAN.view(1, D) + 0.5 * Y_STD.view(1, D) * torch.randn(K, D, device=DEVICE)
            dd = torch.cdist(cand, cand) + torch.eye(K, device=DEVICE) * 1e9
            sep = dd.min().item()
            if sep > best_sep:
                best_sep, best_m0 = sep, cand
        m0_init   = best_m0
        rho0_init = torch.log(Y_VAR + 1e-12).view(1, D) + 1e-2 * torch.randn(K, D, device=DEVICE)
        alpha_init = 1e-2 * torch.randn(N, K, device=DEVICE)
    else:
        raise ValueError(f"unknown init_mode {init_mode}")

    return dict(
        kpp_labels=kpp_labels, kpp_centers=kpp_centers,
        sigma0_diag=sigma0_diag, sigma1_diag=sigma1_diag,
        m0_init=m0_init, rho0_init=rho0_init, alpha_init=alpha_init,
    )


# ============================================================================
# unified runner
# ============================================================================
def _plugin_assign(means, sigma0_diag):
    """Unified ARI assignment rule for every method: plug-in nearest centre under
    the shared likelihood covariance, z_s = argmax_k N(x_s; m_k, diag(sigma0^2))
    = argmin_k sum_d (x_sd - m_kd)^2 / sigma0_d^2. It uses ONLY the means, so no
    method's own q(z) (PAVI's synth-alpha vs the others' iterated alpha) and no
    q(mu) variance enters; every method is scored by the same rule, so the ARI
    curves share one start point at the common init."""
    inv = (1.0 / (sigma0_diag ** 2)).view(1, 1, D)
    d2 = ((Y.view(N, 1, D) - means.view(1, K, D)) ** 2 * inv).sum(dim=-1)   # (N, K)
    return torch.argmin(d2, dim=-1)


def _eval_log(alpha_all, lambda_0, sigma0_diag, sigma1_diag, const_term):
    neg_elbo = elbo_batch_diag(alpha_all, lambda_0, Y, sigma0_diag, sigma1_diag, N) + const_term
    elbo     = -neg_elbo.item()
    m0       = lambda_0[:K * D].view(K, D)
    z_pred   = _plugin_assign(m0, sigma0_diag)
    ari      = adjusted_rand_score(Z_TRUE.cpu().numpy(), z_pred.cpu().numpy())
    return elbo, ari


def _finalize(alpha_all, m0_final, sigma0_diag, sigma1_diag):
    # a diverged run (NaN/Inf in the means) is reported as NaN rather than
    # crashing the Hungarian matching, so a sweep over a bad lr still completes.
    m_np = m0_final.detach().cpu().numpy()
    if not np.isfinite(m_np).all():
        return float("nan"), float("nan"), np.zeros(N, dtype=int)
    # Unified assignment rule, identical to _eval_log: plug-in nearest centre on
    # the means only, so ARI/NMI and the panel labels never depend on a method's
    # own q(z) (alpha_all is kept only for the call signature / divergence guard).
    y_pred    = _plugin_assign(m0_final, sigma0_diag).cpu().numpy()
    y_true_np = Z_TRUE.cpu().numpy()
    ari = adjusted_rand_score(y_true_np, y_pred)
    nmi = normalized_mutual_info_score(y_true_np, y_pred)

    cost = np.linalg.norm(TRUE_MEANS[:, None] - m_np[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    pred_to_true = np.zeros(K, dtype=int)
    pred_to_true[col_ind] = row_ind
    y_pred_aligned = pred_to_true[y_pred]
    return ari, nmi, y_pred_aligned


def _epoch_order(batch_seed, ep, num_batches):
    """Per-epoch permutation of the batch visiting order (port from
    industry_data/tep_GMM_final/compare_ours_svi.py:shuf_order). The batch
    partition is fixed; only the order in which the fixed batches are visited is
    reshuffled each epoch, controlled by a seed that is separate from the model
    seed. This breaks the fixed cluster-aligned cyclic order, which otherwise
    biases the primal-dual methods."""
    return np.random.default_rng(int(batch_seed) + int(ep)).permutation(num_batches)


def _batch_sequence(batch_mode, batch_seed, max_iter, num_batches):
    """Precompute the batch index visited at every iteration.

      'fixed'   : cluster-sorted cyclic order  it % num_batches  (batch_seed ignored)
      'shuffle' : a fresh permutation each epoch (without replacement within an epoch)
      'random'  : an independent random draw each iteration, with replacement
                  (the bio_pipeline baseline form: b = randint(0, B) per step)
    The batch *contents* are always the fixed class-stratified partition; only the
    visiting sequence changes."""
    n = max_iter + 1
    if batch_mode == "fixed":
        return np.arange(n) % num_batches
    rng = np.random.default_rng(0 if batch_seed is None else int(batch_seed))
    if batch_mode == "random":
        return rng.integers(0, num_batches, size=n)
    if batch_mode == "shuffle":
        seq = np.empty(n, dtype=np.int64)
        for start in range(0, n, num_batches):
            perm = rng.permutation(num_batches)
            seq[start:min(start + num_batches, n)] = perm[:min(num_batches, n - start)]
        return seq
    raise ValueError(f"unknown batch_mode {batch_mode}")


def run_method(method, seed, *, eta=None, lr=None, weight_decay=0.01,
               max_iter=MAX_ITER, sample_size=SAMPLE_SIZE, log_every=LOG_EVERY,
               init_mode=INIT_MODE, sigma_source="kpp", batch_seed=None, batch_mode="fixed",
               p2d_eta_m=None, p2d_eta_s=None, restart_every=RESTART_EVERY, dead_frac=0.05,
               pavi_P=PAVI_P, pavi_B=PAVI_B, pavi_h=None, pavi_M=SAMPLE_SIZE, pavi_T=None,
               save=True, tag=None, verbose=True):
    """Run one (method, seed). method in {ours_precondition, SVI, ours_constant, AdamW}.

    batch_seed controls the per-epoch batch-order shuffle, separately from the
    model seed (which controls init and the batch partition). Defaults to `seed`.

    Returns a dict and optionally saves it to
    OUTPUT_DIR/{tag or method}{suffix}_seed{seed}.npz, where suffix is "" for
    kpp init and "_random" otherwise.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # batch_mode selects how the fixed class-stratified batches are visited:
    # 'fixed' (cyclic), 'shuffle' (per-epoch permutation), or 'random' (per-iter
    # draw with replacement, the bio baseline form). batch_seed seeds the latter two.

    init = build_init(seed, init_mode, sigma_source)
    sigma0_diag = init["sigma0_diag"]
    sigma1_diag = init["sigma1_diag"]
    const_term  = compute_const_term_diag(Y, sigma0_diag, sigma1_diag, K)
    batches     = make_fixed_batches_stratified(Z_TRUE, batch_size=sample_size, seed=seed)
    num_batches = len(batches)
    batch_seq   = _batch_sequence(batch_mode, batch_seed, max_iter, num_batches)

    iters, elbo_list, ari_list, time_list, m_traj = [], [], [], [], []
    t_start = time.time()

    def log_step(it, alpha_all, lambda_0_or_m0, m0_snapshot):
        elbo, ari = _eval_log(alpha_all, lambda_0_or_m0, sigma0_diag, sigma1_diag, const_term)
        iters.append(it); elbo_list.append(elbo); ari_list.append(ari)
        time_list.append(time.time() - t_start)
        m_traj.append(m0_snapshot.detach().cpu().numpy().copy())
        if verbose and it % 1000 == 0:
            print(f"[{method}][seed={seed}] iter {it:5d}/{max_iter}  ELBO={elbo:.4e}  ARI={ari:.4f}  t={time_list[-1]:.1f}s")

    # ----------------------------------------------------------------- ours
    if method in ("ours_precondition", "ours_constant"):
        if method == "ours_constant":          # PD-VI: single step size = eta
            assert eta is not None, "PD-VI needs eta"
            eta_m = eta_s = eta
        else:                                   # P^2D-VI: preconditioned
            eta_m = p2d_eta_m if p2d_eta_m is not None else P2D_ETA_M
            eta_s = p2d_eta_s if p2d_eta_s is not None else P2D_ETA_S

        m0   = init["m0_init"].clone()
        rho0 = init["rho0_init"].clone()
        alpha_all = init["alpha_init"].clone()
        lambda_0  = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)
        d_lambda  = 2 * K * D
        mu  = torch.zeros(N, d_lambda, device=DEVICE)
        h_t = torch.zeros(d_lambda, device=DEVICE)

        for it in range(max_iter + 1):
            if it % log_every == 0:
                with torch.no_grad():
                    m0_snap = lambda_0[:K * D].view(K, D)
                    log_step(it, alpha_all, lambda_0, m0_snap)

            bi = int(batch_seq[it])
            St = batches[bi]
            alpha_b, lambda_b = local_update_diag(
                mu[St], Y[St], alpha_all[St], lambda_0,
                sample_size, K, d_lambda, eta_m, eta_s, N, DEVICE,
                sigma0_diag, sigma1_diag, T,
                num_steps=INNER_STEPS, newton_steps=NEWTON_STEPS)
            alpha_all[St] = alpha_b

            if it > 0 and restart_every > 0 and it % restart_every == 0:
                _cluster_restart(alpha_all, lambda_0, mu, h_t, dead_frac)

            lambda_diff = lambda_b - lambda_0.view(1, -1)
            mu[St, :K * D] += lambda_diff[:, :K * D] / eta_m
            mu[St, K * D:] += lambda_diff[:, K * D:] / eta_s
            h_t = h_t + lambda_diff.sum(dim=0) / N
            lambda_0 = lambda_b.mean(dim=0) + h_t

        m0_final = lambda_0[:K * D].view(K, D)

    # ----------------------------------------------------------------- SVI
    elif method == "SVI":
        m0   = init["m0_init"].clone()
        rho0 = init["rho0_init"].clone()
        alpha_all = init["alpha_init"].clone()
        lambda_0  = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)
        eps = 1e-12

        for it in range(max_iter + 1):
            if it % log_every == 0:
                with torch.no_grad():
                    log_step(it, alpha_all, lambda_0, m0)

            step = float((it + SVI_TAU) ** (-SVI_KAPPA)) * SVI_ETA
            bi = int(batch_seq[it])
            St = batches[bi]
            alpha_b, m_int, s2_int = svi_minibatch_update_diag(
                Y[St], alpha_all[St], m0, rho0,
                sigma0_diag, sigma1_diag, N, T, eps=eps)
            alpha_all[St] = alpha_b
            with torch.no_grad():
                m0 = (1.0 - step) * m0 + step * m_int
                s2 = (1.0 - step) * torch.exp(rho0) + step * s2_int
                rho0 = torch.log(s2.clamp_min(eps))
                lambda_0 = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)
        m0_final = m0

    # ----------------------------------------------- AdamW / SGD / RMSProp
    elif method in ("AdamW", "SGD", "RMSProp"):
        assert lr is not None, f"{method} needs lr"
        alpha_all = torch.nn.Parameter(init["alpha_init"].clone())
        m0   = torch.nn.Parameter(init["m0_init"].clone())
        rho0 = torch.nn.Parameter(init["rho0_init"].clone())
        pg = [{"params": [alpha_all], "lr": lr},
              {"params": [m0],        "lr": lr},
              {"params": [rho0],      "lr": lr}]
        if method == "AdamW":
            optimizer = torch.optim.AdamW(pg, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        elif method == "SGD":           # plain SGD, no momentum (matches bio sgd.py)
            optimizer = torch.optim.SGD(pg, momentum=0.0)
        else:                           # RMSProp, torch defaults (matches bio rmsprop.py)
            optimizer = torch.optim.RMSprop(pg, alpha=0.99, eps=1e-8, momentum=0.0, centered=False)

        B = sample_size
        for it in range(max_iter + 1):
            if it % log_every == 0:
                with torch.no_grad():
                    lambda_0 = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)
                    log_step(it, alpha_all.detach(), lambda_0, m0.detach())

            bi = int(batch_seq[it])
            St = batches[bi]
            optimizer.zero_grad(set_to_none=True)
            lambda_0_t = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)
            # minibatch negative-ELBO surrogate (optimizer minimizes -> maximizes ELBO)
            loss = (N / B) * elbo_batch_diag(alpha_all[St], lambda_0_t, Y[St],
                                             sigma0_diag, sigma1_diag, N) + const_term
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rho0.clamp_(RHO_MIN, RHO_MAX)
        m0_final = m0.detach()
        alpha_all = alpha_all.detach()

    # ----------------------------------- CV (SGD + SAGA-style control variates)
    elif method == "CV":
        assert lr is not None, "CV needs lr"
        # port of bio_pipeline/algorithms/cv.py: SAGA control variates on the
        # global parameters (m0, rho0); alpha uses plain SGD. Single lr per block.
        alpha_all = init["alpha_init"].clone()
        m0   = init["m0_init"].clone()
        rho0 = init["rho0_init"].clone()
        B = sample_size

        def _cv_grads(St, m0_c, rho0_c, alpha_c, want_alpha):
            m0_g   = m0_c.detach().clone().requires_grad_(True)
            rho0_g = rho0_c.detach().clone().requires_grad_(True)
            a_g = (alpha_c[St].detach().clone().requires_grad_(True)
                   if want_alpha else alpha_c[St].detach())
            lam = torch.cat([m0_g.reshape(-1), rho0_g.reshape(-1)], dim=0)
            loss = (N / B) * elbo_batch_diag(a_g, lam, Y[St], sigma0_diag, sigma1_diag, N)
            if want_alpha:
                g_a, g_m, g_r = torch.autograd.grad(loss, [a_g, m0_g, rho0_g])
                return g_a.detach(), g_m.detach(), g_r.detach()
            g_m, g_r = torch.autograd.grad(loss, [m0_g, rho0_g])
            return g_m.detach(), g_r.detach()

        # SAGA gradient table over all batches (alpha fixed at init)
        gm_table = torch.zeros(num_batches, K, D, device=DEVICE)
        gr_table = torch.zeros(num_batches, K, D, device=DEVICE)
        for b in range(num_batches):
            gm_table[b], gr_table[b] = _cv_grads(batches[b], m0, rho0, alpha_all, False)
        Gm = gm_table.mean(dim=0); Gr = gr_table.mean(dim=0)

        for it in range(max_iter + 1):
            if it % log_every == 0:
                with torch.no_grad():
                    lambda_0 = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)
                    log_step(it, alpha_all, lambda_0, m0)

            bi = int(batch_seq[it])
            St = batches[bi]
            g_a, g_m, g_r = _cv_grads(St, m0, rho0, alpha_all, True)
            with torch.no_grad():
                alpha_all[St] = alpha_all[St] - lr * g_a
                g_hat_m = g_m - gm_table[bi] + Gm
                g_hat_r = g_r - gr_table[bi] + Gr
                m0   = m0   - lr * g_hat_m
                rho0 = (rho0 - lr * g_hat_r).clamp_(RHO_MIN, RHO_MAX)
                Gm = Gm + (g_m - gm_table[bi]) / float(num_batches)
                Gr = Gr + (g_r - gr_table[bi]) / float(num_batches)
                gm_table[bi] = g_m; gr_table[bi] = g_r
        m0_final = m0

    # ----------------------------------- PAVI (particle mean-field VI baseline)
    elif method in ("PAVI", "PAVI_mb"):
        # Mean field q = prod_k q_{mu_k}, each factor a P-particle cloud; z is
        # marginalized and pi fixed at 1/K (no eta block), so PAVI sees the same
        # diagonal model as every other method. The Euler-Langevin flow targets
        # the mean-field optimum argmin KL(q||p), NOT the full posterior. ELBO/ARI
        # are a post-hoc projection of the clouds onto the parametric family via
        # _pavi_synth_alpha, so they share the other methods' scale.
        data_mode = "mb" if method == "PAVI_mb" else "full"
        M = pavi_M if data_mode == "mb" else N
        T_pavi = pavi_T if pavi_T is not None else max_iter   # local; do NOT shadow module-level T
        inv_s0 = 1.0 / (sigma0_diag ** 2)
        inv_s1 = 1.0 / (sigma1_diag ** 2)
        prior_mean = torch.zeros(D, device=DEVICE)
        m0_init = init["m0_init"].clone()
        lik_scale = (float(N) / float(M)) if data_mode == "mb" else 1.0

        if pavi_h is None:                  # auto-tune once per (init, data_mode), cache
            ck = (init_mode, sigma_source, data_mode, M, pavi_P, pavi_B)
            if ck not in _PAVI_H_CACHE:
                _PAVI_H_CACHE[ck] = pavi.auto_tune_h(
                    Y, K, sigma0_diag, sigma1_diag, m0_init, P=pavi_P, B=pavi_B,
                    data_mode=data_mode, M=M, h_grid=PAVI_H_GRID, seed=seed,
                    prior_mean=prior_mean, verbose=verbose)
            pavi_h = _PAVI_H_CACHE[ck]

        gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
        particles = pavi.init_particles(m0_init, pavi_P, sigma1_diag, gen)
        m0_final  = init["m0_init"].clone()
        alpha_all = init["alpha_init"].clone()

        for it in range(T_pavi + 1):
            if it % log_every == 0:
                with torch.no_grad():
                    m_pe, rho_pe = pavi.point_estimate(particles)
                    lam = torch.cat([m_pe.reshape(-1), rho_pe.reshape(-1)], dim=0)
                    alpha_all = _pavi_synth_alpha(m_pe, rho_pe, Y, sigma0_diag)
                    m0_final = m_pe
                    log_step(it, alpha_all, lam, m_pe)
            if data_mode == "mb":
                # PAVI follows the SAME fixed batches as every other method (one
                # batch per step) instead of resampling M points uniformly, so all
                # six methods share one batch protocol. batch_size == pavi_M.
                Xb = Y[batches[int(batch_seq[it])]]
            else:
                Xb = Y
            ctx  = pavi.sample_contexts(particles, pavi_B, gen)
            grad = pavi.pavi_block_gradients(particles, ctx, Xb, inv_s0, inv_s1,
                                             prior_mean, lik_scale)
            particles = pavi.pavi_step(particles, grad, pavi_h, gen)

        with torch.no_grad():
            m_pe, rho_pe = pavi.point_estimate(particles)
            alpha_all = _pavi_synth_alpha(m_pe, rho_pe, Y, sigma0_diag)
            m0_final = m_pe

    else:
        raise ValueError(f"unknown method {method}")

    ari, nmi, y_pred_aligned = _finalize(alpha_all, m0_final, sigma0_diag, sigma1_diag)
    if verbose:
        print(f"[{method}][seed={seed}] FINAL ARI={ari:.4f} NMI={nmi:.4f} "
              f"wall={time.time()-t_start:.1f}s")

    result = dict(
        method=method, seed=seed,
        eta=(np.nan if eta is None else eta),
        lr=(np.nan if lr is None else lr),
        pavi_h=(np.nan if pavi_h is None else float(pavi_h)),
        weight_decay=weight_decay,
        iters=np.array(iters), elbo=np.array(elbo_list), ari=np.array(ari_list),
        times=np.array(time_list), m_traj=np.array(m_traj),       # (L, K, d)
        m_final=m0_final.detach().cpu().numpy(),
        labels_aligned=y_pred_aligned, ari_final=float(ari), nmi_final=float(nmi),
        sigma0=sigma0_diag.cpu().numpy(), sigma1=sigma1_diag.cpu().numpy(),
    )
    result["init_mode"] = init_mode
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        suffix = "" if init_mode == "kpp" else f"_{init_mode}"
        name = tag or method
        np.savez(f"{OUTPUT_DIR}/{name}{suffix}_seed{seed}.npz", **result)
        if verbose:
            print(f"  saved -> {OUTPUT_DIR}/{name}{suffix}_seed{seed}.npz")
    return result


def _cluster_restart(alpha_all, lambda_0, mu, h_t, dead_frac=0.05):
    """Revive dead clusters (port from main.py / ours_kpp.py).

    A cluster is dead when its soft mass falls below dead_frac * N. The default
    0.05 matches the original moon code; tep_GMM_final uses 0.005."""
    with torch.no_grad():
        phi_all       = torch.softmax(alpha_all, dim=-1)
        cluster_sizes = phi_all.sum(dim=0)
        dead = (cluster_sizes < N * dead_frac).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return
        m0_cur   = lambda_0[:K * D].reshape(K, D)
        rho0_cur = lambda_0[K * D:].reshape(K, D)
        for k_dead in dead:
            k_rich = cluster_sizes.argmax()
            rich_mask = (phi_all[:, k_rich] > 0.5).nonzero(as_tuple=True)[0]
            if len(rich_mask) == 0:
                rich_mask = torch.arange(N, device=DEVICE)
            idx = rich_mask[torch.randint(len(rich_mask), (1,), device=DEVICE)]
            m0_cur[k_dead]   = Y[idx].squeeze() + 0.1 * Y_STD * torch.randn(D, device=DEVICE)
            rho0_cur[k_dead] = rho0_cur[k_rich].clone()
            alpha_all[:, k_dead] = -5.0
            mu[:, k_dead * D:(k_dead + 1) * D] = 0.0
            mu[:, K * D + k_dead * D:K * D + (k_dead + 1) * D] = 0.0
            h_t[k_dead * D:(k_dead + 1) * D] = 0.0
            h_t[K * D + k_dead * D:K * D + (k_dead + 1) * D] = 0.0
            cluster_sizes[k_dead] = cluster_sizes[k_rich] / 2
            cluster_sizes[k_rich] = cluster_sizes[k_rich] / 2
        lambda_0.copy_(torch.cat([m0_cur.reshape(-1), rho0_cur.reshape(-1)]))


if __name__ == "__main__":
    print(f"N={N}, K={K}, d={D}, device={DEVICE}, seeds={SEEDS}")
    init = build_init(42)
    print("sigma0_diag(seed42) =", init["sigma0_diag"].tolist())
    print("sigma1_diag(seed42) =", init["sigma1_diag"].tolist())
