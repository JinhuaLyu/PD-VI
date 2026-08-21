"""TEP main comparison figure in the moon curves_main style: ELBO and ARI, each
vs iteration and wall-clock. Every panel has an EXTERNAL zoom panel on its right
that magnifies the late-iteration converged region, so methods that overlap at
full scale separate. Layout: 2 rows x {main, zoom} x {iteration, wall-clock}.

  * ELBO row EXCLUDES PAVI (its ELBO is a synth-alpha profile, not comparable to
    the others' joint ELBO). Restart spikes are masked, per-seed rolling-median
    smoothed; band is median + IQR(25-75%).
  * ARI row INCLUDES PAVI: ARI is the unified plug-in nearest-centre rule, so
    PAVI is scored on the same footing. Per-seed rolling-median; band mean +- std.
  * Each zoom panel's y-range focuses on the upper cluster (final value above the
    mid-point), so the well-converged methods fill it and the poor ones (SVI/CV)
    fall outside.

Renders images/curves{SUFFIX}.png (use INIT_MODE=random_guarded for the rg run).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
import tep_config as E
import plot_config as P
from plot_tep import mask_restart_spikes, rolling_median   # reuse TEP smoothing

OUT = Path(C.OUTPUT_DIR)
IMG = Path(C.ROOT) / "images"; IMG.mkdir(exist_ok=True)
SUF = C.SUFFIX
CUT_ITER = 15000                                  # cut every curve at this many iterations
plt.rcParams.update(P.RC)
METHODS_ALL = list(E.METHODS)
METHODS_ELBO = [m for m in E.METHODS if m[1] != "PAVImb"]


def sanitize(a):
    a = np.asarray(a, float).copy()
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e15] = np.nan
    return a


def load(display, metric, agg):
    series, times, it_keep = [], [], None
    for seed in E.SEEDS:
        f = OUT / f"{display}{SUF}_seed{seed}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        s = sanitize(d[metric])
        if metric == "elbo":
            s = mask_restart_spikes(s)
        s = rolling_median(s, 11)
        series.append(s); times.append(d["times"]); it_keep = d["iters"]
    if not series:
        return None
    L = min(len(s) for s in series)
    S = np.stack([s[:L] for s in series], 0); T = np.stack([t[:L] for t in times], 0)
    with np.errstate(all="ignore"):
        if agg == "median_iqr":
            centre = rolling_median(np.nanmedian(S, 0), 9)
            lo = rolling_median(np.nanpercentile(S, 25, 0), 9)
            hi = rolling_median(np.nanpercentile(S, 75, 0), 9)
        else:                                       # mean_std
            m = rolling_median(np.nanmean(S, 0), 9); sd = rolling_median(np.nanstd(S, 0), 9)
            centre, lo, hi = m, m - sd, m + sd
        tmid = np.nanmean(T, 0)
    it_keep = it_keep[:L]
    keep = it_keep <= CUT_ITER                          # cut every curve at CUT_ITER iterations
    return it_keep[keep], tmid[keep], centre[keep], lo[keep], hi[keep]


def main():
    fig = plt.figure(figsize=(12.5, 5))
    # main | gap | zoom | spacer | main | gap | zoom  (gap cols widen the main->zoom distance)
    gs = fig.add_gridspec(2, 7, width_ratios=[1.0, 0.06, 0.42, 0.10, 1.0, 0.06, 0.42],
                          wspace=0.08, hspace=0.36, top=0.87, bottom=0.12, left=0.065, right=0.985)
    main_col = [0, 4]; zoom_col = [2, 6]
    xlabels = ["Iteration", "Wall-clock time (s)"]
    rows = [("elbo", "ELBO", METHODS_ELBO, "median_iqr"),
            ("ari", "ARI", METHODS_ALL, "mean_std")]
    for r, (key, ylabel, methods, agg) in enumerate(rows):
        data = []
        for _arg, disp, lab, col, ls, lw, _o in methods:
            res = load(disp, key, agg)
            if res is None:
                continue
            it, tm, centre, lo, hi = res
            data.append((np.asarray(it, float), np.asarray(tm, float), np.asarray(centre, float),
                         np.asarray(lo, float), np.asarray(hi, float), col, ls, lw))
        finals = [c[2][np.isfinite(c[2])][-1] if np.isfinite(c[2]).any() else np.nan for c in data]
        mid = 0.5 * (np.nanmax(finals) + np.nanmin(finals))
        for view in range(2):                              # 0 = iteration, 1 = wall-clock
            ax = fig.add_subplot(gs[r, main_col[view]])
            axz = fig.add_subplot(gs[r, zoom_col[view]])
            xmax_glob = max(np.nanmax(c[0] if view == 0 else c[1]) for c in data)
            xthr = 0.4 * xmax_glob
            xs_all, los_all, his_all, yv = [], [], [], []
            for (it, tm, centre, lo, hi, col, ls, lw), fin in zip(data, finals):
                x = it if view == 0 else tm
                ax.plot(x, centre, color=col, ls=ls, lw=lw)
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
                axz.plot(x, centre, color=col, ls=ls, lw=lw)
                xs_all.append(x); los_all.append(lo); his_all.append(hi)
                if np.isfinite(fin) and fin >= mid:
                    yv.append(centre[x >= xthr])
            # main: tight axes
            xs = np.concatenate(xs_all); los = np.concatenate(los_all); his = np.concatenate(his_all)
            xmin, xmx = np.nanmin(xs), np.nanmax(xs)
            ax.set_xlim(xmin - 0.025 * (xmx - xmin), xmx)
            ymin, ymax = np.nanmin(los), np.nanmax(his); pad = (ymax - ymin) * 0.05
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_ylabel(ylabel if view == 0 else ""); ax.grid(True, alpha=0.25)
            # zoom: late region, y-range from the upper cluster
            axz.set_xlim(xthr, xmx)
            ybox = None
            if yv:
                gv = np.concatenate(yv); gv = gv[np.isfinite(gv)]
                lo2, hi2 = float(gv.min()), float(gv.max()); p2 = (hi2 - lo2) * 0.3 + 1e-12
                axz.set_ylim(lo2 - p2, hi2 + p2); ybox = (lo2 - p2, hi2 + p2)
            axz.grid(True, alpha=0.25); axz.tick_params(labelsize=6)
            axz.set_title("zoom", fontsize=7)
            if ybox is not None:                       # box on the main panel + connectors to the zoom
                indicator = ax.indicate_inset([xthr, ybox[0], xmx - xthr, ybox[1] - ybox[0]],
                                              inset_ax=axz, edgecolor="#777", lw=0.9, alpha=0.65)
                # connector order is (LL, UL, LR, UR); keep only the box's right
                # corners (lower-right, upper-right) on every panel for consistency
                for i, cp in enumerate(indicator.connectors):
                    cp.set_visible(i in (2, 3))
            if r == 1:
                ax.set_xlabel(xlabels[view])
    handles = [plt.Line2D([], [], color=c, ls=l, lw=w) for (_, _, _, c, l, w, _) in METHODS_ALL]
    labels = [m[2] for m in METHODS_ALL]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 0.99))
    for ext in ("png", "pdf"):
        fig.savefig(IMG / f"curves{SUF}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {IMG}/curves{SUF}.png  (each panel + external right-side zoom on the converged region)")


if __name__ == "__main__":
    main()
