"""Run the TEP comparison: 6 methods x 5 seeds at common.MAX_ITER.

Reads tuned hyper-parameters from output/tuning_tep.json. Each run's full
trajectory is saved to output/{display}_seed{seed}.npz, and a summary (final ARI
and ELBO, mean +- std) to output/summary_tep.json. Incremental: a (method, seed)
whose npz already exists is loaded, not re-run.
"""
import json
import os
import time

import numpy as np
import common as C
import tep_config as E

with open(f"{C.OUTPUT_DIR}/tuning_tep{C.SUFFIX}.json") as f:
    TUNED = json.load(f)

t_all = time.time()
summary = {}
for method_arg, display, *_ in E.METHODS:
    aris, elbos = [], []
    for seed in E.SEEDS:
        npz = f"{C.OUTPUT_DIR}/{display}{C.SUFFIX}_seed{seed}.npz"
        try:
            if os.path.exists(npz):                       # incremental: reuse a saved run
                d = np.load(npz)
                a = float(d["ari_final"]); e = float(d["elbo"][-1])
            else:
                kw = E.run_kwargs(method_arg, seed, TUNED)
                r = C.run_method(method_arg, seed, tag=display, save=True, verbose=False, **kw)
                a = float(r["ari_final"]); e = float(r["elbo"][-1])
        except Exception as ex:
            print(f"  !! {display} seed{seed} FAILED: {type(ex).__name__}: {ex}")
            a = e = float("nan")
        aris.append(a); elbos.append(e)
        print(f"  {display:7s} seed{seed}: ARI={a:.4f} ELBO={e:.3e}")
    arr = np.array(aris)
    summary[display] = {
        "ari_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
        "ari_std":  float(np.nanstd(arr))  if np.isfinite(arr).any() else float("nan"),
        "ari_per_seed": aris, "elbo_final": elbos,
    }
    m = summary[display]
    print(f"  -> {display:7s} ARI={m['ari_mean']:.4f}+-{m['ari_std']:.4f}")

with open(f"{C.OUTPUT_DIR}/summary_tep{C.SUFFIX}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDONE in {(time.time()-t_all)/60:.1f} min -> {C.OUTPUT_DIR}/summary_tep{C.SUFFIX}.json")
