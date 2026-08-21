"""Final synthetic-moon comparison config.

One unified setting for every algorithm:
  guarded random init + k-means++ sigma + per-iter random batch (with replacement),
  cluster-stratified batches, 5 seeds.

Methods: P2D-VI (main) and PD-VI (auxiliary) plus SVI, SGD, CV (gradient-descent
control variates). A particle method (PAVI) is left out for now; add a tuple to
METHODS to include it later (the pipeline is method-list driven).

All trajectories and summaries are saved under output/, so the figures can be
re-rendered from disk without re-running any experiment.
"""

# (method arg for run_method, display tag, latex label, color, linestyle, lw, is_ours)
METHODS = [
    ("ours_precondition", "P2DVI", "P$^2$D-VI", "#E06B5A", "-",  2.6, True),
    ("ours_constant",     "PDVI",  "PD-VI",     "#4FA3A5", "-",  2.0, True),
    ("SVI",               "SVI",   "SVI",       "#9B7BB8", "--", 2.0, False),
    ("AdamW",             "AdamW", "Adam",      "#6A9FD4", "--", 2.0, False),
    ("CV",                "CV",    "CV",        "#B07AA1", ":",  2.0, False),
    ("PAVI_mb",           "PAVImb", "PAVI",     "#E0A458", "-.", 2.0, False),
]
PANEL_ORDER = ["SVI", "AdamW", "CV", "PAVImb", "PDVI", "P2DVI"]   # left-to-right in the panel/legend
SEEDS = [42, 1, 2, 3, 4]
PANEL_SEED = 42

# unified setting for ALL algorithms
INIT_MODE  = "random_guarded"
SIGMA      = "kpp"
BATCH_MODE = "random"          # per-iter draw with replacement (the bio form)
RESTART    = 100               # cluster-restart period for the primal-dual methods

# tuning grids (lr / eta tuned on seed 42 -> tuning_exp.json; SVI is fixed)
P2D_ETA_M = [3e-4, 1e-3, 3e-3]
P2D_ETA_S = [3e-3, 1e-2, 3e-2]
PDVI_ETA  = [3e-4, 1e-3, 3e-3, 1e-2]
SGD_LRS   = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5]
CV_LRS    = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5]
P2D_TUNE_SEEDS = [42, 1]       # P2D-VI is init-sensitive: score on 2 seeds, not 1
PAVI_T    = 10000              # particle-Langevin iteration budget (step size h auto-tuned in run_method)

import os as _os
if _os.environ.get("SMOKE"):
    SEEDS = [42, 1]
    P2D_ETA_M = [1e-3]; P2D_ETA_S = [1e-2]
    PDVI_ETA = [1e-3]; SGD_LRS = [3e-6]; CV_LRS = [1e-6]
    P2D_TUNE_SEEDS = [42]
    PAVI_T = 200


def run_kwargs(method_arg, seed, tuned):
    """Common run_method kwargs for this method under the unified setting."""
    kw = dict(init_mode=INIT_MODE, sigma_source=SIGMA, batch_mode=BATCH_MODE,
              batch_seed=seed, restart_every=RESTART)
    if method_arg == "ours_precondition":
        kw.update(p2d_eta_m=tuned["P2DVI"]["eta_m"], p2d_eta_s=tuned["P2DVI"]["eta_s"])
    elif method_arg == "ours_constant":
        kw.update(eta=tuned["PDVI"]["eta"])
    elif method_arg in ("SGD", "CV", "AdamW", "RMSProp"):
        kw.update(lr=tuned[method_arg]["lr"])
    elif method_arg in ("PAVI", "PAVI_mb"):
        # particle VI: step size h is auto-tuned on seed 42 inside run_method and
        # cached; batch_mode/batch_seed/restart_every are passed but ignored by it.
        kw.update(pavi_T=PAVI_T)
    # SVI: no extra hyper-parameters (fixed tau/kappa schedule in common.py)
    return kw
