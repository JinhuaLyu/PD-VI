"""Run the final comparison: 5 methods x 5 seeds under the unified setting.

Reads tuned hyper-parameters from output/tuning_exp.json. Every run's full
trajectory is saved to output/{display}_seed{seed}.npz, and a summary (final ARI
/ ELBO / W2, mean +- std) to output/summary_exp.json, so the figures can be
re-rendered from disk without re-running anything.
"""
import json
import time

import numpy as np
import common as C
import exp_config as E
from metrics import w2_trajectory

with open(f"{C.OUTPUT_DIR}/tuning_exp.json") as f:
    TUNED = json.load(f)

import os
SUFFIX = "" if E.INIT_MODE == "kpp" else f"_{E.INIT_MODE}"


def _w2(d):
    try:
        return float(w2_trajectory(d["m_traj"], d["sigma0"], C.TRUE_MEANS, C.TRUE_COVS)[0][-1])
    except Exception:
        return float("nan")


t_all = time.time()
summary = {}
for method_arg, display, *_ in E.METHODS:
    aris, elbos, w2s = [], [], []
    for seed in E.SEEDS:
        npz = f"{C.OUTPUT_DIR}/{display}{SUFFIX}_seed{seed}.npz"
        try:
            if os.path.exists(npz):            # incremental: reuse a saved run
                d = np.load(npz)
                a = float(d["ari_final"]); e = float(d["elbo"][-1]); w2 = _w2(d)
            else:
                kw = E.run_kwargs(method_arg, seed, TUNED)
                r = C.run_method(method_arg, seed, tag=display, save=True, verbose=False, **kw)
                a = float(r["ari_final"]); e = float(r["elbo"][-1]); w2 = _w2(r)
        except Exception as ex:
            print(f"  !! {display} seed{seed} FAILED: {type(ex).__name__}: {ex}")
            a = e = w2 = float("nan")
        aris.append(a); elbos.append(e); w2s.append(w2)
    arr = np.array(aris)
    summary[display] = {
        "ari_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
        "ari_std":  float(np.nanstd(arr))  if np.isfinite(arr).any() else float("nan"),
        "ari_per_seed": aris, "elbo_final": elbos, "w2_final": w2s,
    }
    m = summary[display]
    print(f"  {display:8s} ARI={m['ari_mean']:.4f}+-{m['ari_std']:.4f}  per-seed={[round(x,3) for x in aris]}")

with open(f"{C.OUTPUT_DIR}/summary_exp.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDONE in {(time.time()-t_all)/60:.1f} min -> {C.OUTPUT_DIR}/summary_exp.json")
