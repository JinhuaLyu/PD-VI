"""PAVI (Particle Algorithm for Mean-Field Variational Inference) for a Bayesian GMM.

Adaptation of the PAVI scheme from "A Particle Algorithm for Mean-Field
Variational Inference" to the moon GMM used by common.py, under the settings
chosen for this baseline:

  * mixing weights FIXED at pi_k = 1/K (no eta block, matching the other 7
    methods in compare_config.METHODS);
  * latent assignments z are MARGINALIZED OUT (PAVI is not applied to the
    discrete z directly);
  * diagonal likelihood covariance Sigma0 = diag(sigma0^2) and Gaussian prior
    mu_k ~ N(0, Sigma1), Sigma1 = diag(sigma1^2), both reused from the same
    k-means++ estimates as the other methods (common.build_init);
  * every variational factor q_{mu_k} is represented nonparametrically by P
    particles; the mean field is q = prod_k q_{mu_k}.

Model (z integrated out):
    p(X | mu_{1:K}) = prod_s sum_k (1/K) N(x_s; mu_k, Sigma0),   mu_k ~ N(0, Sigma1).

Potential (negative log posterior up to a constant):
    V(mu) = sum_k 0.5 * mu_k^T Sigma1^{-1} mu_k
            - sum_s log sum_k exp(-0.5 * q_{sk}),
    q_{sk} = (x_s - mu_k)^T Sigma0^{-1} (x_s - mu_k).

Responsibilities (pi uniform, so pi and the shared Gaussian normalizer cancel):
    r_{sk} = softmax_k(-0.5 * q_{sk}).

Block gradient:
    grad_{mu_a} V = Sigma1^{-1} mu_a - sum_s r_{sa} Sigma0^{-1} (x_s - mu_a).

PAVI update. Blocks theta_k = mu_k, k = 1..K. The variational distribution is the
product empirical measure q_P = prod_k (1/P) sum_j delta_{mu_k^(j)}. At each
iteration we draw B contexts (one particle index per factor); for each block a and
each particle mu_a^(j) we average grad_{mu_a} V over the B contexts, with the other
factors supplied by the context and slot a set to mu_a^(j); then take one
Euler-Langevin step

    mu_a^(j) <- mu_a^(j) - h * g^a(mu_a^(j)) + sqrt(2h) * xi,   xi ~ N(0, I_d).

This targets the mean-field optimum argmin_{q = prod_k q_{mu_k}} KL(q || p). It is
NOT joint Langevin Monte Carlo on the full posterior: for block a the other blocks
enter only through the current empirical measure (the B-context average), which is
exactly the mean-field coupling.

The module is usable standalone (see __main__, which validates the implementation
on a synthetic GMM with known means) and exposes vectorized primitives that
common.py's run_method("PAVI") branch calls to integrate PAVI into the moon
comparison.
"""
import math

import numpy as np
import torch

LOG_2PI = math.log(2.0 * math.pi)


# ============================================================================
# vectorized PAVI primitives (no enumeration of P^K particle combinations)
# ============================================================================
def init_particles(m0_init, P, sigma1_diag, gen, spread_frac=0.1):
    """Initialize P particles per factor around the (k-means++) centers m0_init.

    m0_init : (K, d) starting centers.  Returns particles (K, P, d).
    Particles start as m0_init[k] + spread_frac * sigma1 * N(0, I); the Langevin
    dynamics set the stationary spread, so this only affects the early iterations.
    """
    K, d = m0_init.shape
    noise = torch.randn(K, P, d, generator=gen, device=m0_init.device, dtype=m0_init.dtype)
    spread = (spread_frac * sigma1_diag).view(1, 1, d)
    return m0_init.view(K, 1, d) + spread * noise


def sample_contexts(particles, B, gen):
    """Draw B mean-field contexts from the product empirical measure.

    particles : (K, P, d).  Returns ctx (B, K, d): for each context b and factor
    k, one particle index is drawn uniformly and independently.
    """
    K, P, d = particles.shape
    idx = torch.randint(0, P, (B, K), generator=gen, device=particles.device)
    kk = torch.arange(K, device=particles.device).view(1, K).expand(B, K)
    return particles[kk, idx]                                   # (B, K, d)


def pavi_block_gradients(particles, ctx, X, inv_s0, inv_s1, prior_mean, lik_scale):
    """Stochastic mean-field gradient g^a(mu_a^(j)) for every block a and particle j.

    particles : (K, P, d)   current particles for all factors.
    ctx       : (B, K, d)    sampled contexts (column a is ignored for block a).
    X         : (S, d)       data (full set, or a minibatch when lik_scale = S/M).
    inv_s0    : (d,)         1 / sigma0^2 (diagonal Sigma0^{-1}).
    inv_s1    : (d,)         1 / sigma1^2 (diagonal Sigma1^{-1}).
    prior_mean: (d,)         prior mean m_0 (zeros here).
    lik_scale : float        S / M for a minibatch of size M (1.0 for full data).

    Returns grad (K, P, d) with
        grad[a, j] = Sigma1^{-1}(mu_a^(j) - m_0)
                     - (1/B) sum_b sum_s r_{s,a}^{(j,b)} Sigma0^{-1}(x_s - mu_a^(j)).
    The sum over s is computed exactly; only the average over the B contexts and
    the product over the other K-1 factors are Monte-Carlo (the PAVI batch).
    """
    K, P, d = particles.shape
    B = ctx.shape[0]
    S = X.shape[0]
    neg_inf = torch.tensor(float("-inf"), device=X.device, dtype=X.dtype)

    Xw = X * inv_s0.view(1, d)                                  # (S, d)
    sumsq_x = (X * Xw).sum(dim=1)                               # (S,)  sum_d x^2/sigma0^2

    # log-likelihood (up to the shared normalizer) of every datum under every
    # context mean: lq_ctx[s, b, l] = -0.5 * q(x_s, ctx[b, l]).
    ctx_flat = ctx.reshape(B * K, d)                            # (B*K, d)
    sumsq_ctx = (ctx_flat * ctx_flat * inv_s0.view(1, d)).sum(dim=1)        # (B*K,)
    cross_ctx = Xw @ ctx_flat.t()                              # (S, B*K)
    q_ctx = sumsq_x.view(S, 1) - 2.0 * cross_ctx + sumsq_ctx.view(1, B * K)
    lq_ctx = (-0.5 * q_ctx).reshape(S, B, K)                    # (S, B, K)

    grad = torch.empty_like(particles)
    for a in range(K):
        Pa = particles[a]                                      # (P, d)
        sumsq_Pa = (Pa * Pa * inv_s0.view(1, d)).sum(dim=1)    # (P,)
        cross_a = Xw @ Pa.t()                                 # (S, P)
        q_a = sumsq_x.view(S, 1) - 2.0 * cross_a + sumsq_Pa.view(1, P)
        lq_a = -0.5 * q_a                                      # (S, P)

        # log sum over the OTHER factors l != a, per (s, b).
        masked = lq_ctx.clone()
        masked[:, :, a] = neg_inf
        M = torch.logsumexp(masked, dim=2)                     # (S, B)

        # responsibility of factor a with candidate j under context b.
        log_denom = torch.logaddexp(lq_a.unsqueeze(2), M.unsqueeze(1))      # (S, P, B)
        r = torch.exp(lq_a.unsqueeze(2) - log_denom)           # (S, P, B)

        R = r.sum(dim=0)                                       # (P, B)  sum_s r
        r2 = r.reshape(S, P * B)
        wx = (X.t() @ r2).reshape(d, P, B).permute(1, 2, 0)    # (P, B, d)  sum_s r x_s

        # sum_s r * Sigma0^{-1}(x_s - mu_a^(j)), scaled by S/M for a minibatch.
        glik = inv_s0.view(1, 1, d) * (wx - Pa.unsqueeze(1) * R.unsqueeze(2))
        glik = (lik_scale * glik).mean(dim=1)                  # (P, d)  average over B
        g_prior = inv_s1.view(1, d) * (Pa - prior_mean.view(1, d))          # (P, d)
        grad[a] = g_prior - glik
    return grad


def pavi_step(particles, grad, h, gen):
    """One Euler-Langevin step: mu <- mu - h*grad + sqrt(2h)*N(0,I)."""
    if gen is None:
        noise = torch.randn_like(particles)
    else:
        noise = torch.randn(particles.shape, generator=gen,
                            device=particles.device, dtype=particles.dtype)
    return particles - h * grad + math.sqrt(2.0 * h) * noise


def point_estimate(particles, rho_min=-30.0, rho_max=20.0):
    """Collapse each particle cloud to (mean, log-variance) per dimension.

    Returns m (K, d) and rho (K, d) with rho = log Var_j[mu_k^(j)]. Used both for
    diagnostics and to project PAVI onto the parametric family for a comparable
    ELBO (see common.run_method's PAVI branch).
    """
    m = particles.mean(dim=1)                                  # (K, d)
    var = particles.var(dim=1, unbiased=False)                 # (K, d)
    rho = torch.log(var.clamp_min(math.exp(rho_min))).clamp(rho_min, rho_max)
    return m, rho


# ============================================================================
# diagnostics
# ============================================================================
def neg_log_posterior(means, X, inv_s0, inv_s1, prior_mean):
    """Potential V evaluated at a single set of means (K, d): a scalar diagnostic.

    V = sum_k 0.5 mu_k^T Sigma1^{-1} mu_k - sum_s log sum_k exp(-0.5 q_sk).
    Constants (the 1/K and Gaussian normalizers) are dropped, matching the
    definition in this module's header.
    """
    K, d = means.shape
    Xw = X * inv_s0.view(1, d)
    sumsq_x = (X * Xw).sum(dim=1)                              # (S,)
    sumsq_m = (means * means * inv_s0.view(1, d)).sum(dim=1)   # (K,)
    cross = Xw @ means.t()                                     # (S, K)
    q = sumsq_x.view(-1, 1) - 2.0 * cross + sumsq_m.view(1, K)
    data_term = -torch.logsumexp(-0.5 * q, dim=1).sum()
    prior_term = 0.5 * (inv_s1.view(1, d) * (means - prior_mean.view(1, d)) ** 2).sum()
    return (prior_term + data_term).item()


def _log_normal_diag(X, means, sigma0_diag):
    """log N(x; mu, diag(sigma0^2)) for all (x in X, mu in means).

    X (Q, d), means (M, d) -> (Q, M). Keeps the Gaussian normalizer (this is a
    genuine density, unlike the responsibility computation which drops it).
    """
    d = X.shape[1]
    inv_s0 = 1.0 / (sigma0_diag ** 2)
    norm = -0.5 * (d * LOG_2PI + (2.0 * torch.log(sigma0_diag)).sum())
    Xw = X * inv_s0.view(1, d)
    sumsq_x = (X * Xw).sum(dim=1).view(-1, 1)
    sumsq_m = (means * means * inv_s0.view(1, d)).sum(dim=1).view(1, -1)
    cross = Xw @ means.t()
    q = sumsq_x - 2.0 * cross + sumsq_m
    return norm - 0.5 * q


def posterior_predictive_loglik(X_eval, particles, sigma0_diag, pi=None):
    """Mean-field posterior predictive log-likelihood, averaged over X_eval.

    Under the mean field q = prod_k q_{mu_k},
        p(x | X) ~= E_q[sum_k pi_k N(x; mu_k, Sigma0)]
                  = sum_k pi_k (1/P) sum_j N(x; mu_k^(j), Sigma0).
    pi defaults to uniform 1/K. Returns the average over X_eval as a float.
    """
    K, P, d = particles.shape
    if pi is None:
        pi = torch.full((K,), 1.0 / K, device=particles.device, dtype=particles.dtype)
    logN = _log_normal_diag(X_eval, particles.reshape(K * P, d), sigma0_diag)   # (Q, K*P)
    logN = logN.reshape(X_eval.shape[0], K, P)
    log_w = (torch.log(pi).view(1, K, 1) - math.log(P))        # (1, K, 1)
    logpx = torch.logsumexp(logN + log_w, dim=(1, 2))          # (Q,)
    return logpx.mean().item()


def mc_assignment_probs(X, particles, sigma0_diag, G=64, gen=None, pi=None):
    """Monte-Carlo estimate of the posterior assignment probabilities P(z_s=k | X).

    Averages the responsibilities r_{sk}(theta) over G contexts theta ~ q_P, where
    each context draws one particle per factor. Returns (S, K).
    """
    K, P, d = particles.shape
    inv_s0 = 1.0 / (sigma0_diag ** 2)
    if pi is None:
        log_pi = torch.zeros(K, device=particles.device, dtype=particles.dtype)
    else:
        log_pi = torch.log(pi)
    Xw = X * inv_s0.view(1, d)
    sumsq_x = (X * Xw).sum(dim=1).view(-1, 1)                  # (S, 1)
    probs = torch.zeros(X.shape[0], K, device=particles.device, dtype=particles.dtype)
    for _ in range(G):
        ctx = sample_contexts(particles, 1, gen)[0]            # (K, d)
        sumsq_c = (ctx * ctx * inv_s0.view(1, d)).sum(dim=1).view(1, K)
        cross = Xw @ ctx.t()                                  # (S, K)
        q = sumsq_x - 2.0 * cross + sumsq_c
        probs += torch.softmax(-0.5 * q + log_pi.view(1, K), dim=1)
    return probs / G


# ============================================================================
# standalone driver + step-size tuning
# ============================================================================
def run_pavi(X, K, sigma0_diag, sigma1_diag, m0_init, *, P=64, B=16, T=3000, h=None,
             data_mode="full", M=None, seed=0, log_every=10, prior_mean=None,
             h_grid=(1e-6, 3e-6, 1e-5, 3e-5), verbose=False, device=None):
    """Run PAVI and return a history dict. Used by the self-check and reusable by
    callers that do not need the harness npz format.

    data_mode in {"full", "mb"}: "mb" draws a fresh random minibatch of size M each
    iteration and scales the likelihood gradient by S/M. If h is None it is tuned
    once on this seed over h_grid (largest grid value with finite, lowest
    negative-log-posterior pilot).
    """
    device = device or X.device
    X = X.to(device)
    sigma0_diag = sigma0_diag.to(device)
    sigma1_diag = sigma1_diag.to(device)
    m0_init = m0_init.to(device)
    S, d = X.shape
    inv_s0 = 1.0 / (sigma0_diag ** 2)
    inv_s1 = 1.0 / (sigma1_diag ** 2)
    if prior_mean is None:
        prior_mean = torch.zeros(d, device=device, dtype=X.dtype)
    if data_mode == "mb":
        assert M is not None and M <= S
        lik_scale = float(S) / float(M)
    else:
        M, lik_scale = S, 1.0

    if h is None:
        h = auto_tune_h(X, K, sigma0_diag, sigma1_diag, m0_init, P=P, B=B,
                        data_mode=data_mode, M=M, h_grid=h_grid, seed=seed,
                        prior_mean=prior_mean, verbose=verbose)

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    particles = init_particles(m0_init, P, sigma1_diag, gen)

    hist = {"iters": [], "neg_log_post": [], "pred_ll": [], "m_traj": [], "h": h}
    for it in range(T + 1):
        if it % log_every == 0:
            with torch.no_grad():
                m, _ = point_estimate(particles)
                hist["iters"].append(it)
                hist["neg_log_post"].append(
                    neg_log_posterior(m, X, inv_s0, inv_s1, prior_mean))
                hist["pred_ll"].append(
                    posterior_predictive_loglik(X, particles, sigma0_diag))
                hist["m_traj"].append(m.detach().cpu().numpy().copy())
                if verbose and it % max(log_every, T // 10 or 1) == 0:
                    print(f"  [pavi] it {it:5d}/{T}  V={hist['neg_log_post'][-1]:.4e}"
                          f"  predLL={hist['pred_ll'][-1]:.4f}")
        if data_mode == "mb":
            sel = torch.randint(0, S, (M,), generator=gen, device=device)
            Xb = X[sel]
        else:
            Xb = X
        ctx = sample_contexts(particles, B, gen)
        grad = pavi_block_gradients(particles, ctx, Xb, inv_s0, inv_s1,
                                    prior_mean, lik_scale)
        particles = pavi_step(particles, grad, h, gen)

    hist["particles"] = particles.detach()
    hist["m_final"], hist["rho_final"] = point_estimate(particles)
    return hist


def auto_tune_h(X, K, sigma0_diag, sigma1_diag, m0_init, *, P, B, data_mode, M,
                h_grid, seed, prior_mean, pilot_iters=300, verbose=False):
    """Pick the largest step size in h_grid that stays finite and reaches the
    lowest negative log posterior on a short pilot (mirrors the lr sweep the other
    baselines use; largest stable step is preferred)."""
    device = X.device
    inv_s0 = 1.0 / (sigma0_diag ** 2)
    inv_s1 = 1.0 / (sigma1_diag ** 2)
    S = X.shape[0]
    lik_scale = (float(S) / float(M)) if data_mode == "mb" else 1.0
    stable = []
    for h in sorted(h_grid):
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        particles = init_particles(m0_init, P, sigma1_diag, gen)
        diverged = False
        for _ in range(pilot_iters):
            if data_mode == "mb":
                Xb = X[torch.randint(0, S, (M,), generator=gen, device=device)]
            else:
                Xb = X
            ctx = sample_contexts(particles, B, gen)
            grad = pavi_block_gradients(particles, ctx, Xb, inv_s0, inv_s1,
                                        prior_mean, lik_scale)
            particles = pavi_step(particles, grad, h, gen)
            if not torch.isfinite(particles).all():
                diverged = True
                break
        if diverged:
            if verbose:
                print(f"  [tune_h] h={h:g} diverged")
            continue
        m, _ = point_estimate(particles)
        V = neg_log_posterior(m, X, inv_s0, inv_s1, prior_mean)
        stable.append((h, V))
        if verbose:
            print(f"  [tune_h] h={h:g}  V={V:.4e}")
    if not stable:
        best_h = min(h_grid)
        if verbose:
            print(f"  [tune_h] all candidates diverged; falling back to h={best_h:g}")
        return best_h
    # prefer the LARGEST stable step whose pilot V is within tolerance of the best
    # (fastest still-stable step), matching the lr-tie rule used by the other
    # baselines in tune_compare.py.
    V_best = min(v for _, v in stable)
    tol = 1e-3 * abs(V_best) + 1e-6
    best_h = max(h for h, v in stable if v <= V_best + tol)
    if verbose:
        print(f"  [tune_h] selected h={best_h:g} (best pilot V={V_best:.4e})")
    return best_h


# ============================================================================
# synthetic data + self-check
# ============================================================================
def sample_synthetic_gmm(S, K, d, sigma=0.15, sep=2.0, seed=0, device="cpu"):
    """Draw a balanced, well-separated synthetic GMM with known means.

    Returns X (S, d), z (S,), true_means (K, d). Means are placed on a grid scaled
    by `sep`; each component is isotropic N(mu_k, sigma^2 I) with weight 1/K.
    """
    g = torch.Generator(device=device); g.manual_seed(seed)
    side = int(math.ceil(K ** (1.0 / d)))
    coords = torch.arange(side, dtype=torch.get_default_dtype(), device=device) * sep
    grid = torch.cartesian_prod(*[coords] * d) if d > 1 else coords.view(-1, 1)
    true_means = grid[:K].clone()
    true_means = true_means - true_means.mean(dim=0, keepdim=True)
    z = torch.randint(0, K, (S,), generator=g, device=device)
    X = true_means[z] + sigma * torch.randn(S, d, generator=g, device=device)
    return X, z, true_means


def _match_means(est, true):
    """Hungarian match estimated to true means; return mean matched L2 distance."""
    from scipy.optimize import linear_sum_assignment
    est = est.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    cost = np.linalg.norm(true[:, None] - est[None, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def _self_check():
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, d, S = 4, 2, 2000
    X, z, true_means = sample_synthetic_gmm(S, K, d, sigma=0.15, sep=2.0,
                                            seed=0, device=device)
    # known scales (the standalone check does not need k-means): sigma0 = true
    # component std, sigma1 = spread of the true means.
    sigma0_diag = torch.full((d,), 0.15, device=device)
    sigma1_diag = X.std(dim=0).clamp_min(0.5)
    # k-means++ init for the starting centers (as in the moon harness).
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X.cpu().numpy())
    m0_init = torch.tensor(km.cluster_centers_, device=device, dtype=X.dtype)

    print("Running PAVI self-check on synthetic GMM "
          f"(K={K}, d={d}, S={S}, device={device}) ...")
    hist = run_pavi(X, K, sigma0_diag, sigma1_diag, m0_init, P=64, B=16, T=2000,
                    data_mode="full", seed=0, log_every=50, verbose=True)
    err = _match_means(hist["m_final"], true_means)
    base = _match_means(m0_init, true_means)
    print(f"chosen h            : {hist['h']:g}")
    print(f"k-means++ mean error: {base:.4f}")
    print(f"PAVI   mean error   : {err:.4f}  (true component std = 0.15)")
    print(f"final neg-log-post  : {hist['neg_log_post'][-1]:.4e}")
    print(f"final pred. LL      : {hist['pred_ll'][-1]:.4f}")
    # particle spread should be on the order of the posterior std ~ sigma0/sqrt(S/K).
    spread = hist["particles"].std(dim=1).mean().item()
    post_std = 0.15 / math.sqrt(S / K)
    print(f"mean particle spread: {spread:.4f}  (posterior std ~ {post_std:.4f})")
    ok = err < 0.1 and math.isfinite(hist["neg_log_post"][-1])
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _self_check()
