"""Config for the TEP (Tennessee Eastman) comparison.

Data: 86,000 SensorSCAN 64-d embeddings (K=21 fault classes), kept in (run, time)
order. Unified setting for every algorithm:
  k-means++ init + k-means++ sigma + time-contiguous batches (consecutive chunks
  of SAMPLE_SIZE points, near single-class / federated) visited by per-iteration
  random draw with replacement.

Methods: P2D-VI (main) and PD-VI (auxiliary) plus SVI, AdamW, CV, PAVI.
Figures: UMAP panels, t-SNE panels, and ELBO/ARI curves vs iteration & wall-clock.
Everything is saved to output/, so figures re-render from disk without re-running.
"""
import os

# (method arg for run_method, display tag, label, color, linestyle, lw, is_ours)
METHODS = [
    ("ours_precondition", "P2DVI", "P$^2$D-VI", "#E06B5A", "-",  2.6, True),
    ("ours_constant",     "PDVI",  "PD-VI",     "#4FA3A5", "-",  2.0, True),
    ("SVI",               "SVI",   "SVI",       "#9B7BB8", "--", 2.0, False),
    ("Adam",              "Adam",  "Adam",      "#6A9FD4", "--", 2.0, False),
    ("CV",                "CV",    "CV",        "#B07AA1", ":",  2.0, False),
    ("PAVI_mb",           "PAVImb", "PAVI",     "#E0A458", "-.", 2.0, False),
]
PANEL_ORDER = ["SVI", "Adam", "CV", "PAVImb", "PDVI", "P2DVI"]
SEEDS = [42, 1, 2, 3, 4]
PANEL_SEED = 42

# unified setting (all algorithms). INIT_MODE is env-selectable so the same
# scripts produce the kpp variant (no suffix) or the random_guarded variant
# (outputs carry the "_random_guarded" suffix and never overwrite the kpp run).
INIT_MODE  = os.environ.get("INIT_MODE", "kpp")
SIGMA      = "kpp"
BATCH_MODE = "random"          # per-iter draw with replacement; batches are time-contiguous
RESTART    = 100
# K=21, so the moon default dead_frac=0.05 exceeds the mean cluster mass 1/21=4.76%
# and fires the restart on nearly every cluster every period, destroying the
# k-means++ structure. Use 0.005 (the tep_GMM_final value) for the primal-dual methods.
DEAD_FRAC  = 0.005

# tuning grids -- ours is tuned heavily (the request), baselines coarsely.
# Sweeps run at TUNE_ITER iters (cheaper); the final run uses common.MAX_ITER.
# eta_m must reach PD-VI's optimal scalar penalty (0.01) so the grid covers the
# PD-VI diagonal (eta_m=eta_s) and the superset relation P2D-VI >= PD-VI holds.
P2D_ETA_M = [3e-4, 1e-3, 3e-3, 1e-2]
P2D_ETA_S = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
PDVI_ETA  = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
ADAMW_LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
CV_LRS    = [1e-7, 1e-6, 1e-5, 1e-4]
P2D_TUNE_SEEDS = [42, 1]       # ours is init/order-sensitive: score on 2 seeds
PAVI_T    = 15000              # particle-Langevin iteration budget (h auto-tuned)
TUNE_ITER = 6000               # iters used for tuning sweeps (full run uses MAX_ITER)

import os as _os
if _os.environ.get("SMOKE"):
    SEEDS = [42, 1]
    P2D_ETA_M = [1e-3]; P2D_ETA_S = [1e-2]
    PDVI_ETA = [1e-3]; ADAMW_LRS = [1e-2]; CV_LRS = [1e-5]
    P2D_TUNE_SEEDS = [42]; PAVI_T = 200; TUNE_ITER = 300


def run_kwargs(method_arg, seed, tuned):
    """run_method kwargs for this method under the unified TEP setting."""
    kw = dict(init_mode=INIT_MODE, sigma_source=SIGMA, batch_mode=BATCH_MODE,
              batch_seed=seed, restart_every=RESTART, dead_frac=DEAD_FRAC)
    if method_arg == "ours_precondition":
        kw.update(p2d_eta_m=tuned["P2DVI"]["eta_m"], p2d_eta_s=tuned["P2DVI"]["eta_s"])
    elif method_arg == "ours_constant":
        kw.update(eta=tuned["PDVI"]["eta"])
    elif method_arg in ("SGD", "CV", "AdamW", "Adam", "RMSProp"):
        kw.update(lr=tuned[method_arg]["lr"])
    elif method_arg in ("PAVI", "PAVI_mb"):
        kw.update(pavi_T=PAVI_T)
    return kw
