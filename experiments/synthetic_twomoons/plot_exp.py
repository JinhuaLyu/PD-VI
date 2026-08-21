"""Render the two final figures from the saved trajectories (no re-running).

  figure 1  images/panels.png      ground truth + each method's clustering (seed 42)
  figure 2  images/curves.png      3x2 grid: {ELBO, ARI, W2} x {iteration, wall-clock}

Colours / line styles come from exp_config.METHODS. Everything is read from
output/*.npz and output/summary_exp.json, so tweaking the figures never needs a
re-run.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

import common as C
import exp_config as E
import plot_config as P
from metrics import w2_trajectory


def recolor_by_overlap(labels, y_true, K):
    """Map each predicted cluster to the true cluster it overlaps most (Hungarian
    on the contingency table, 1-to-1). More stable / interpretable than the
    mean-distance alignment when a method merges clusters: the merged blob is
    coloured by its dominant true membership, so colours are comparable across
    methods. Operates on the saved partition, so no re-run is needed."""
    labels = np.asarray(labels)
    M = np.zeros((K, K))
    np.add.at(M, (y_true, labels), 1)          # M[true, predicted] overlap counts
    r, c = linear_sum_assignment(-M)            # maximize overlap
    mp = np.arange(K)
    mp[c] = r                                   # predicted cluster c -> true cluster r
    return mp[labels]

plt.rcParams.update(P.RC)
OUT = Path(C.OUTPUT_DIR)
IMG = Path(C.ROOT) / "images"; IMG.mkdir(exist_ok=True)
NB = C.N // C.SAMPLE_SIZE
SUFFIX = "" if E.INIT_MODE == "kpp" else f"_{E.INIT_MODE}"
STYLE = {d: (lab, col, ls, lw) for (_, d, lab, col, ls, lw, _) in E.METHODS}


def stem(display):
    return f"{display}{SUFFIX}"


def sanitize(a):
    a = np.asarray(a, float).copy()
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e15] = np.nan
    return a


def load_metric(display, metric):
    """(iters, times_mean, mean, std) over seeds, epoch-sampled; W2 despiked."""
    series, times, it_keep = [], [], None
    for seed in E.SEEDS:
        f = OUT / f"{stem(display)}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        keep = (d["iters"] % NB == 0)
        if metric == "w2":
            w2, _ = w2_trajectory(d["m_traj"], d["sigma0"], C.TRUE_MEANS, C.TRUE_COVS)
            y = P.rolling_median(sanitize(w2)[keep], w=9)
        else:
            y = sanitize(d[metric])[keep]
        series.append(y); times.append(d["times"][keep]); it_keep = d["iters"][keep]
    if not series:
        return None
    L = min(len(s) for s in series)
    S = np.stack([s[:L] for s in series], 0); T = np.stack([t[:L] for t in times], 0)
    with np.errstate(all="ignore"):
        return it_keep[:L], np.nanmean(T, 0), np.nanmean(S, 0), np.nanstd(S, 0)


def finite_ylim(curves, pad=0.06):
    v = np.concatenate([c[np.isfinite(c)] for c in curves if c is not None and np.isfinite(c).any()])
    lo, hi = v.min(), v.max(); m = (hi - lo) * pad + 1e-9
    return lo - m, hi + m


# ----------------------------------------------------------------- figure 2
def curves_figure():
    # ELBO spans ~4 orders (best -1.3e4, worst -2.3e5), so use symlog to keep the
    # top methods (P2D-VI vs PD-VI) distinguishable while still showing the rest.
    metrics = [("elbo", "ELBO", None, "symlog"), ("ari", "ARI", (-0.02, 1.02), "linear"),
               ("w2", r"Wasserstein $W_2$", None, "linear")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.2), sharex="col")
    handles = None
    for row, (key, ylabel, ylim, yscale) in enumerate(metrics):
        curves = []
        for _arg, display, lab, col, ls, lw, _o in E.METHODS:
            res = load_metric(display, key)
            if res is None:
                continue
            it, tm, mean, std = res
            curves.append(mean)
            for col_i, x in enumerate([it, tm]):
                ln, = axes[row, col_i].plot(x, mean, color=col, ls=ls, lw=lw, label=lab)
                axes[row, col_i].fill_between(x, mean - std, mean + std, color=col, alpha=0.15, lw=0)
        if row == 0:
            handles = [plt.Line2D([], [], color=c, ls=l, lw=w)
                       for (_, _, _, c, l, w, _) in E.METHODS]
        for col_i in range(2):
            ax = axes[row, col_i]
            ax.set_ylabel(ylabel if col_i == 0 else "")
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=1e4)
                ax.set_ylim(top=0)   # data is all negative; drop the empty positive half
            elif ylim is not None:
                ax.set_ylim(*ylim)
            elif curves:
                try:
                    ax.set_ylim(*finite_ylim(curves))
                except ValueError:
                    pass
            ax.grid(True, alpha=0.25)
    axes[2, 0].set_xlabel("Iteration"); axes[2, 1].set_xlabel("Wall-clock time (s)")
    axes[0, 0].set_title("versus iteration", fontsize=10)
    axes[0, 1].set_title("versus wall-clock time", fontsize=10)
    labels = [m[2] for m in E.METHODS]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(IMG / f"curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {IMG}/curves.png")


# ----------------------------------------------------------------- figure 1
def panels_figure():
    X = C.Y.cpu().numpy(); y_true = C.Z_TRUE.cpu().numpy()
    colors = P.cluster_colors(X, y_true, C.K)
    summ = json.load(open(OUT / "summary_exp.json"))
    order = E.PANEL_ORDER

    # two rows: ground truth alone on the left (vertically centred), the six
    # methods in a 2x3 grid on the right, with a spacer column between them.
    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(4, 5, width_ratios=[1.0, 0.15, 1.0, 1.0, 1.0],
                          hspace=0.42, wspace=0.10)
    ax_gt = fig.add_subplot(gs[1:3, 0])                                  # centred over both rows
    m_axes = [fig.add_subplot(gs[0:2, c]) for c in (2, 3, 4)] + \
             [fig.add_subplot(gs[2:4, c]) for c in (2, 3, 4)]

    def scatter(ax, labels, title):
        ax.scatter(X[:, 0], X[:, 1], c=np.array([colors[int(l)] for l in labels]), s=2, lw=0)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#666"); sp.set_linewidth(0.8)

    scatter(ax_gt, y_true, "Ground truth")
    for ax, display in zip(m_axes, order):
        lab = STYLE[display][0]
        # show the seed whose final ARI is the MEDIAN over the 5 seeds (representative);
        # a fixed seed can be unrepresentative for bimodal methods (e.g. AdamW seed 42).
        ps = np.array(summ[display]["ari_per_seed"], float) if display in summ else None
        if ps is not None and np.isfinite(ps).any():
            med_seed = E.SEEDS[int(np.nanargmin(np.abs(ps - np.nanmedian(ps))))]
        else:
            med_seed = E.PANEL_SEED
        f = OUT / f"{stem(display)}_seed{med_seed}.npz"
        if not f.exists():
            ax.set_title(f"{lab} (missing)", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); continue
        d = np.load(f); ari = float(d["ari_final"])
        if not np.isfinite(ari):
            ax.text(0.5, 0.5, "diverged", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lab, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        else:
            scatter(ax, recolor_by_overlap(d["labels_aligned"], y_true, C.K), f"{lab} (ARI={ari:.3f})")
    for ext in ("png", "pdf"):
        fig.savefig(IMG / f"panels.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {IMG}/panels.png")


if __name__ == "__main__":
    panels_figure()
    curves_figure()
