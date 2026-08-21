"""Main comparison figure: ELBO and ARI, each vs iteration and vs wall-clock time
(2x2). Reads the saved npz only; does not re-run anything.

Row-specific choices:
  * ELBO row EXCLUDES PAVI (its ELBO is a synth-alpha profile, not comparable to
    the others' joint ELBO). It uses per-seed rolling-median despiking (the
    primal-dual cluster-restart transients reach 1e5-1e6 single-step jumps) and a
    median + IQR(25-75%) band, so outlier seeds and residual spikes do not blow
    up the band.
  * ARI row INCLUDES PAVI: ARI is the unified plug-in nearest-centre rule for
    every method, so PAVI is scored on the same footing. This row uses raw
    mean +- std (no despiking), matching curves.py; AdamW's wide band is its real
    bimodal seed spread and is kept.
The legend keeps all methods (PAVI appears in the ARI row).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
import exp_config as E
import plot_config as P

OUT = Path(C.OUTPUT_DIR)
IMG = Path(C.ROOT) / "images"; IMG.mkdir(exist_ok=True)
SUF = "" if E.INIT_MODE == "kpp" else f"_{E.INIT_MODE}"
NB = C.N // C.SAMPLE_SIZE
plt.rcParams.update(P.RC)

METHODS_ALL = list(E.METHODS)                                  # ARI row + legend
METHODS_ELBO = [m for m in E.METHODS if m[1] != "PAVImb"]      # ELBO row drops PAVI


def sanitize(a):
    a = np.asarray(a, float).copy()
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e15] = np.nan
    return a


def load(display, metric, despike_w, agg):
    """Per-seed epoch-sampled (+ optional rolling-median despike), then aggregate
    across seeds as median+IQR (agg='median_iqr') or mean+-std (agg='mean_std').
    Returns (iters, times, centre, lo, hi)."""
    series, times, it_keep = [], [], None
    for seed in E.SEEDS:
        f = OUT / f"{display}{SUF}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        keep = (d["iters"] % NB == 0)
        y = sanitize(d[metric])[keep]
        if despike_w:
            y = P.rolling_median(y, w=despike_w)
        series.append(y); times.append(d["times"][keep]); it_keep = d["iters"][keep]
    if not series:
        return None
    L = min(len(s) for s in series)
    S = np.stack([s[:L] for s in series], 0); T = np.stack([t[:L] for t in times], 0)
    with np.errstate(all="ignore"):
        if agg == "median_iqr":
            centre = np.nanmedian(S, 0)
            lo = np.nanpercentile(S, 25, axis=0); hi = np.nanpercentile(S, 75, axis=0)
        else:                                       # mean_std
            m = np.nanmean(S, 0); s = np.nanstd(S, 0)
            centre, lo, hi = m, m - s, m + s
        tmid = np.nanmedian(T, 0)
    return it_keep[:L], tmid, centre, lo, hi


def main():
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))   # independent axes: zoom each panel
    # (key, ylabel, yscale, methods, despike_w, aggregation)
    rows = [("elbo", "ELBO", "symlog", METHODS_ELBO, 11, "median_iqr"),
            ("ari", "ARI", "linear", METHODS_ALL, 0, "mean_std")]
    for r, (key, ylabel, yscale, methods, dw, agg) in enumerate(rows):
        col_xy = [[], []]            # per column: (x, band-lo, band-hi) of each method
        for _arg, disp, lab, col, ls, lw, _o in methods:
            res = load(disp, key, despike_w=dw, agg=agg)
            if res is None:
                continue
            it, tm, centre, lo, hi = res
            zo = 6 if _o else 3                          # our methods (P2D-VI/PD-VI) drawn on top
            for c, x in enumerate([it, tm]):
                axes[r, c].plot(x, centre, color=col, ls=ls, lw=lw, label=lab, zorder=zo)
                axes[r, c].fill_between(x, lo, hi, color=col, alpha=0.15, lw=0, zorder=zo - 1)
                col_xy[c].append((np.asarray(x, float), np.asarray(lo, float), np.asarray(hi, float)))
        for c in range(2):
            ax = axes[r, c]
            ax.set_ylabel(ylabel if c == 0 else "")
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=1e4)
            if col_xy[c]:                              # tight x/y limits so the curves fill the panel
                xs = np.concatenate([d[0] for d in col_xy[c]])
                los = np.concatenate([d[1] for d in col_xy[c]])
                his = np.concatenate([d[2] for d in col_xy[c]])
                xmin, xmax = np.nanmin(xs), np.nanmax(xs)
                if c == 1:                            # wall-clock column: cap the x-axis at 60 s
                    xmax = 60.0
                ax.set_xlim(xmin - 0.025 * (xmax - xmin), xmax)   # left margin only; right tight/capped
                ymin, ymax = np.nanmin(los), np.nanmax(his)
                if yscale == "symlog":                # negative data: tight bottom, fixed top -10^4
                    ax.set_ylim(ymin * 1.08, -1e4)
                else:
                    pad = (ymax - ymin) * 0.05
                    ax.set_ylim(ymin - pad, ymax + pad)
            ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Iteration"); axes[1, 1].set_xlabel("Wall-clock time (s)")
    handles = [plt.Line2D([], [], color=c, ls=l, lw=w) for (_, _, _, c, l, w, _) in METHODS_ALL]
    labels = [m[2] for m in METHODS_ALL]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(IMG / f"curves_main.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {IMG}/curves_main.png  (ELBO: no PAVI, median/IQR despiked; ARI: all methods, mean+-std)")


if __name__ == "__main__":
    main()
