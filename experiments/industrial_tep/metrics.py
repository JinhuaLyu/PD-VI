"""
Closed-form 2-Wasserstein distance between the estimated and the true GMM.

The estimated data distribution for cluster k is  N(m_k, diag(sigma0^2)),
where sigma0 is the fixed/known data noise from the k-means++ init. The true
cluster k is N(mu_k^true, Sigma_k^true) with a full 2x2 covariance.

For two Gaussians the 2-Wasserstein distance has the closed form
    W2^2(N(m1,C1), N(m2,C2)) = ||m1 - m2||^2 + B^2(C1, C2),
    B^2(C1,C2) = tr(C1) + tr(C2) - 2 tr( (C1^{1/2} C2 C1^{1/2})^{1/2} )   (Bures).

We match estimated components to true components with the Hungarian algorithm
on the W2^2 cost matrix, then report the mean per-component W2 distance
    W2_mean = (1/K) sum_k W2(est_{i_k}, true_k).

For a trajectory the permutation is fixed once from the final means (the
estimated covariance is constant, so the matching is stable) and applied to
every logged step, exactly like mfvi_syn_final/plot_wasserstein.py.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _sqrtm_sym_psd(A):
    """Symmetric PSD matrix square root via eigendecomposition (stable for 2x2)."""
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def gaussian_w2_sq(m1, cov1, m2, cov2):
    """Squared 2-Wasserstein between N(m1,cov1) and N(m2,cov2)."""
    diff = np.asarray(m1, float) - np.asarray(m2, float)
    mean_term = float(diff @ diff)
    s1 = _sqrtm_sym_psd(cov1)
    inner = _sqrtm_sym_psd(s1 @ cov2 @ s1)
    bures = float(np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(inner))
    return mean_term + max(bures, 0.0)


def _cost_matrix(m_est, cov_est, true_means, true_covs):
    K = true_means.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            C[i, j] = gaussian_w2_sq(m_est[i], cov_est, true_means[j], true_covs[j])
    return C


def match_components(m_est, sigma0, true_means, true_covs):
    """Hungarian match (est -> true) on the W2^2 cost using the given means.

    Returns (perm, w2_per_true) where perm[t] = estimated index assigned to
    true cluster t, and w2_per_true[t] = W2 distance for that pair.
    """
    cov_est = np.diag(np.asarray(sigma0, float) ** 2)
    C = _cost_matrix(m_est, cov_est, true_means, true_covs)
    row_ind, col_ind = linear_sum_assignment(C)   # row=est, col=true
    perm = np.zeros(true_means.shape[0], dtype=int)
    w2_per_true = np.zeros(true_means.shape[0], dtype=float)
    for est_i, true_j in zip(row_ind, col_ind):
        perm[true_j] = est_i
        w2_per_true[true_j] = np.sqrt(max(C[est_i, true_j], 0.0))
    return perm, w2_per_true


def w2_trajectory(m_traj, sigma0, true_means, true_covs):
    """Mean per-component W2 for every logged step.

    m_traj : (L, K, d) estimated means over the logged iterations.
    Returns (w2_mean (L,), perm) with the permutation fixed from the final step.
    """
    m_traj = np.asarray(m_traj, float)
    L, K, d = m_traj.shape
    cov_est = np.diag(np.asarray(sigma0, float) ** 2)

    perm, _ = match_components(m_traj[-1], sigma0, true_means, true_covs)

    w2_mean = np.zeros(L, dtype=float)
    for t in range(L):
        d_sum = 0.0
        for true_j in range(K):
            est_i = perm[true_j]
            d_sum += np.sqrt(gaussian_w2_sq(m_traj[t, est_i], cov_est,
                                            true_means[true_j], true_covs[true_j]))
        w2_mean[t] = d_sum / K
    return w2_mean, perm


if __name__ == "__main__":
    # sanity: W2 between identical Gaussians is 0; pure mean shift recovers the shift.
    m = np.array([1.0, 2.0]); C = np.array([[0.5, 0.1], [0.1, 0.3]])
    assert abs(gaussian_w2_sq(m, C, m, C)) < 1e-9
    d2 = gaussian_w2_sq(np.array([0.0, 0.0]), C, np.array([3.0, 4.0]), C)
    assert abs(d2 - 25.0) < 1e-6, d2
    print("metrics.py self-test passed (W2 identical=0, mean-shift=5).")
