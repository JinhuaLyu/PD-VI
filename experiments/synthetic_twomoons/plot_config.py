"""Shared style for all moon_data_final figures."""
import os
import matplotlib

# init-mode suffix, mirrors common.py (set via INIT_MODE env var)
INIT_MODE = os.environ.get("INIT_MODE", "kpp")
SUFFIX    = "" if INIT_MODE == "kpp" else f"_{INIT_MODE}"

# tag -> (display label, color, linestyle, linewidth)
METHODS = {
    "P2DVI": ("P$^2$D-VI", "#E06B5A", "-",  2.6),   # ours, highlight
    "SVI":   ("SVI",       "#9B7BB8", "--", 2.0),
    "PDVI":  ("PD-VI",     "#4FA3A5", "-",  2.0),
    "AdamW": ("AdamW",     "#6A9FD4", "--", 2.0),
}

# left-to-right order in the multi-panel scatter and in the trajectory legends
PANEL_ORDER = ["SVI", "PDVI", "AdamW", "P2DVI"]

# two-moon palettes reused from moon_test/code/plot_three_panel_comparison.py
MOON_PALETTES = [
    ['#e2f0f2', '#a9d1da', '#63a3b5', '#267189', '#063747'],
    ['#f7e2e9', '#e4afc2', '#c56f96', '#913d70', '#551b45'],
]

RC = {
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
}


def rolling_median(x, w=9):
    """Robust despike: rolling median with a centered window of w points.

    The primal-dual methods emit brief upward transients right after a cluster
    restart (every 100 iterations). A rolling median removes these 1-2 point
    spikes while preserving the convergence trend and the converged value.

    The window radius shrinks symmetrically at both ends (radius =
    min(h, i, n-1-i)), so an endpoint is never pulled toward fast-moving
    neighbours on one side only. In particular out[0] == x[0]: every method's
    plotted curve starts at its true iteration-0 value instead of the median of
    the first few (already-diverged) points, which previously made W2 start
    points differ across methods even though the raw it=0 value is identical.
    """
    import numpy as np
    x = np.asarray(x, float)
    n = len(x); h = w // 2
    out = np.empty(n)
    for i in range(n):
        r = min(h, i, n - 1 - i)
        out[i] = np.median(x[i - r:i + r + 1])
    return out


def cluster_colors(X, y_true, K):
    """Assign a within-group shade to each of the K true clusters by x-position."""
    import numpy as np
    colors = [None] * K
    groups = [range(0, 5), range(5, 10)] if K == 10 else [range(K)]
    for gid, grp in enumerate(groups):
        grp = list(grp)
        centers = np.array([X[y_true == k].mean(axis=0) for k in grp])
        order = np.argsort(centers[:, 0])
        for shade, pos in enumerate(order):
            colors[grp[pos]] = matplotlib.colors.to_rgba(MOON_PALETTES[gid % 2][shade % 5])
    return colors
