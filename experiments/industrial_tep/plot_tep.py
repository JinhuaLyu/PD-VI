"""TEP figures, all rendered from saved trajectories / cached 2-D embeddings.

  images/panels_umap.png   ground truth + each method (UMAP 2-D, median-ARI seed)
  images/panels_tsne.png   same on t-SNE 2-D
  images/curves.png        2x2 grid: {ELBO, ARI} x {iteration, wall-clock}

The UMAP and t-SNE 2-D coordinates of all 86,000 points are computed once and
cached to output/{umap,tsne}_2d.npy, so re-styling the figures never recomputes
them. Cluster colours are a fixed 21-colour palette; each method's clusters are
mapped to the true classes by contingency-overlap Hungarian (stable across
methods). Run with REDO_EMB=1 to force recomputation of the embeddings.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import linear_sum_assignment

import common as C
import tep_config as E
import plot_config as P

plt.rcParams.update(P.RC)
OUT = Path(C.OUTPUT_DIR)
IMG = Path(C.ROOT) / "images"; IMG.mkdir(exist_ok=True)
STYLE = {d: (lab, col, ls, lw) for (_, d, lab, col, ls, lw, _) in E.METHODS}
COLORS = (list(cm.tab20.colors) + list(cm.tab20b.colors))[:C.K]   # 21 distinct colours


def sanitize(a):
    a = np.asarray(a, float).copy()
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e15] = np.nan
    return a


def mask_restart_spikes(e, factor=6.0):
    """Mask cluster-restart transient spikes in an ELBO series.

    A restart (P^2D-VI / PD-VI) makes the restarted cluster's mean jump for a
    single step, so the ELBO plunges that step and recovers the next. Such a
    point sits far below the run's typical ELBO. Use the per-series median as
    the typical level -- it is robust to the <15% spike fraction and adapts to
    each method's own scale -- and drop points more than `factor`x below it
    (ELBO is negative, so factor>1 is a more-negative threshold). Early
    convergence and the steady state both stay above the threshold, so the
    non-restart trajectory is preserved; only the deep single-step plunges go."""
    e = np.asarray(e, float).copy()
    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return e
    base = float(np.median(finite))
    with np.errstate(invalid="ignore"):
        spike = e < factor * base
    e[spike] = np.nan
    return e


def rolling_median(x, w=11):
    """Rolling median (window w) over a 1-D series; NaN-aware. Smooths out the few
    restart-transient spikes (a minority inside any window) while tracking the
    genuine trajectory. The window radius shrinks symmetrically at both ends
    (radius = min(h, i, n-1-i)), so out[0] == x[0]: every method's curve starts at
    its true iteration-0 value instead of the median of the first few (already
    diverged) points, which otherwise makes the start points differ across methods
    even though the shared initialization is identical."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return x
    h = w // 2
    out = np.empty(n)
    for i in range(n):
        r = min(h, i, n - 1 - i)
        seg = x[i - r:i + r + 1]
        out[i] = np.nanmedian(seg) if np.isfinite(seg).any() else np.nan
    return out


def recolor_by_overlap(labels, y_true, K):
    """Map each predicted cluster to the true class it overlaps most (Hungarian on
    the contingency table). Stable across methods, so colours are comparable."""
    labels = np.asarray(labels)
    M = np.zeros((K, K))
    np.add.at(M, (y_true, labels), 1)
    r, c = linear_sum_assignment(-M)
    mp = np.arange(K); mp[c] = r
    return mp[labels]


def embedding(kind):
    """Cached 2-D embedding of all points (kind in {'umap','tsne'})."""
    f = OUT / f"{kind}_2d.npy"
    if f.exists() and not os.environ.get("REDO_EMB"):
        return np.load(f)
    X = C.Y.cpu().numpy()
    print(f"computing {kind} on {X.shape[0]} points (cached afterwards)...")
    if kind == "umap":
        import umap
        Z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    np.save(f, Z)
    print(f"  saved {f}")
    return Z


# ----------------------------------------------------------------- panels
def panels_figure(coords, kind):
    y_true = C.Z_TRUE.cpu().numpy()
    summ = json.load(open(OUT / f"summary_tep{C.SUFFIX}.json"))
    order = E.PANEL_ORDER

    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(4, 5, width_ratios=[1.0, 0.15, 1.0, 1.0, 1.0], hspace=0.42, wspace=0.10)
    ax_gt = fig.add_subplot(gs[1:3, 0])
    m_axes = [fig.add_subplot(gs[0:2, c]) for c in (2, 3, 4)] + \
             [fig.add_subplot(gs[2:4, c]) for c in (2, 3, 4)]

    def scatter(ax, labels, title):
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=np.array([COLORS[int(l)] for l in labels]), s=1.0, lw=0)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#666"); sp.set_linewidth(0.8)

    scatter(ax_gt, y_true, "Ground truth")
    for ax, display in zip(m_axes, order):
        lab = STYLE[display][0]
        ps = np.array(summ[display]["ari_per_seed"], float) if display in summ else None
        med_seed = E.SEEDS[int(np.nanargmin(np.abs(ps - np.nanmedian(ps))))] \
            if (ps is not None and np.isfinite(ps).any()) else E.PANEL_SEED
        f = OUT / f"{display}{C.SUFFIX}_seed{med_seed}.npz"
        if not f.exists():
            ax.set_title(f"{lab} (missing)", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); continue
        d = np.load(f); ari = float(d["ari_final"])
        if not np.isfinite(ari):
            ax.text(0.5, 0.5, "diverged", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lab, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        else:
            scatter(ax, recolor_by_overlap(d["labels_aligned"], y_true, C.K), f"{lab} (ARI={ari:.3f})")
    for ext in ("png", "pdf"):
        fig.savefig(IMG / f"panels_{kind}{C.SUFFIX}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {IMG}/panels_{kind}{C.SUFFIX}.png")


# ----------------------------------------------------------------- curves
def load_metric(display, metric, smooth=False):
    series, times, it_keep = [], [], None
    for seed in E.SEEDS:
        f = OUT / f"{display}{C.SUFFIX}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        s = sanitize(d[metric])
        if smooth:
            # Per-seed clean trajectory: ELBO first drops its deep restart
            # plunges (mask), then both metrics are rolling-median smoothed into
            # a continuous curve. The band below is the std OF these smoothed
            # per-seed trajectories, i.e. std computed from spike-free data.
            if metric == "elbo":
                s = mask_restart_spikes(s)
            s = rolling_median(s, 11)
        series.append(s); times.append(d["times"]); it_keep = d["iters"]
    if not series:
        return None
    L = min(len(s) for s in series)
    S = np.stack([s[:L] for s in series], 0); T = np.stack([t[:L] for t in times], 0)
    with np.errstate(all="ignore"):
        # mean/std across the smoothed, spike-free per-seed curves; a light pass
        # (window 9) over both removes residual early spikes caused by seeds
        # sitting at different convergence stages on a given iteration.
        mean = rolling_median(np.nanmean(S, 0), 9)
        std = rolling_median(np.nanstd(S, 0), 9)
        return it_keep[:L], np.nanmean(T, 0), mean, std


def finite_ylim(curves, pad=0.06):
    v = np.concatenate([c[np.isfinite(c)] for c in curves if c is not None and np.isfinite(c).any()])
    lo, hi = v.min(), v.max(); m = (hi - lo) * pad + 1e-9
    return lo - m, hi + m


def curves_figure():
    SHOW_BAND = True    # shaded band = mean +/-1 s.d. across seeds
    metrics = [("elbo", "ELBO", "linear", (-2e7, 0.0)), ("ari", "ARI", "linear", (-0.02, 1.02))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col")
    for row, (key, ylabel, yscale, ylim) in enumerate(metrics):
        curves = []
        for _arg, display, lab, col, ls, lw, _o in E.METHODS:
            res = load_metric(display, key, smooth=True)
            if res is None:
                continue
            it, tm, mean, std = res
            curves.append(mean)
            for ci, x in enumerate([it, tm]):
                axes[row, ci].plot(x, mean, color=col, ls=ls, lw=lw, label=lab)
                if SHOW_BAND:
                    axes[row, ci].fill_between(x, mean - std, mean + std, color=col, alpha=0.12, lw=0)
        for ci in range(2):
            ax = axes[row, ci]
            ax.set_ylabel(ylabel if ci == 0 else "")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Iteration"); axes[1, 1].set_xlabel("Wall-clock time (s)")
    axes[0, 0].set_title("versus iteration", fontsize=10)
    axes[0, 1].set_title("versus wall-clock time", fontsize=10)
    handles = [plt.Line2D([], [], color=c, ls=l, lw=w) for (_, _, _, c, l, w, _) in E.METHODS]
    labels = [m[2] for m in E.METHODS]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.text(0.5, -0.02, "Curves are seed means (each rolling-median smoothed, window 11); "
             "shaded band is mean +/-1 s.d. across 5 seeds.", ha="center", fontsize=8, style="italic")
    for ext in ("png", "pdf"):
        fig.savefig(IMG / f"curves{C.SUFFIX}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {IMG}/curves{C.SUFFIX}.png")


if __name__ == "__main__":
    curves_figure()
    panels_figure(embedding("umap"), "umap")
    panels_figure(embedding("tsne"), "tsne")
