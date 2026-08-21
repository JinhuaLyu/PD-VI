#!/usr/bin/env python3
"""
Reproduce the Two Moons synthetic clustering experiment.

This script runs the full Two Moons experiment pipeline:
  1. Tune hyper-parameters on seed 42
  2. Run all methods x all seeds with tuned hyper-parameters
  3. Generate result summary

Usage:
  cd experiments/synthetic_twomoons
  python run.py                 # Run full pipeline
  python run.py --tune-only     # Only run tuning
  python run.py --run-only      # Only run experiments (requires tuning results)

Output:
  - output/{METHOD}_seed{SEED}.npz  : Full trajectories (ELBO, ARI, etc.)
  - output/summary_exp.json          : Summary statistics (mean +- std)
  - output/tuning_exp.json           : Tuned hyper-parameters
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import common as C
import config as E
from metrics import w2_trajectory


def run_tuning():
    """Hyper-parameter tuning on seed 42."""
    print("[*] Running hyper-parameter tuning (seed 42 only)...")
    output_file = f"{C.OUTPUT_DIR}/tuning_exp.json"
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    tuned = {}

    # Tune P2D-VI (preconditioned)
    print("  Tuning P2D-VI...")
    best_ari = -1
    best_eta_m, best_eta_s = None, None
    for eta_m in E.P2D_ETA_M:
        for eta_s in E.P2D_ETA_S:
            try:
                r = C.run_method("ours_precondition", 42, p2d_eta_m=eta_m, p2d_eta_s=eta_s,
                                init_mode=E.INIT_MODE, sigma_source=E.SIGMA,
                                batch_mode=E.BATCH_MODE, batch_seed=42,
                                restart_every=E.RESTART, verbose=False)
                ari = float(r["ari_final"])
                if ari > best_ari:
                    best_ari = ari
                    best_eta_m, best_eta_s = eta_m, eta_s
                print(f"    eta_m={eta_m:.1e}, eta_s={eta_s:.1e}: ARI={ari:.4f}")
            except Exception as e:
                print(f"    eta_m={eta_m:.1e}, eta_s={eta_s:.1e}: FAILED ({e})")
    tuned["P2DVI"] = {"eta_m": best_eta_m, "eta_s": best_eta_s}
    print(f"  -> Best P2D-VI: eta_m={best_eta_m:.1e}, eta_s={best_eta_s:.1e}, ARI={best_ari:.4f}")

    # Tune PD-VI (constant step size)
    print("  Tuning PD-VI...")
    best_ari = -1
    best_eta = None
    for eta in E.PDVI_ETA:
        try:
            r = C.run_method("ours_constant", 42, eta=eta,
                            init_mode=E.INIT_MODE, sigma_source=E.SIGMA,
                            batch_mode=E.BATCH_MODE, batch_seed=42,
                            restart_every=E.RESTART, verbose=False)
            ari = float(r["ari_final"])
            if ari > best_ari:
                best_ari = ari
                best_eta = eta
            print(f"    eta={eta:.1e}: ARI={ari:.4f}")
        except Exception as e:
            print(f"    eta={eta:.1e}: FAILED ({e})")
    tuned["PDVI"] = {"eta": best_eta}
    print(f"  -> Best PD-VI: eta={best_eta:.1e}, ARI={best_ari:.4f}")

    # AdamW (lr tuning)
    print("  Tuning AdamW...")
    best_ari = -1
    best_lr = None
    for lr in E.SGD_LRS:  # Use same LR grid
        try:
            r = C.run_method("AdamW", 42, lr=lr,
                            init_mode=E.INIT_MODE, sigma_source=E.SIGMA,
                            batch_mode=E.BATCH_MODE, batch_seed=42,
                            restart_every=E.RESTART, verbose=False)
            ari = float(r["ari_final"])
            if ari > best_ari:
                best_ari = ari
                best_lr = lr
            print(f"    lr={lr:.1e}: ARI={ari:.4f}")
        except Exception as e:
            print(f"    lr={lr:.1e}: FAILED ({e})")
    tuned["AdamW"] = {"lr": best_lr}
    print(f"  -> Best AdamW: lr={best_lr:.1e}, ARI={best_ari:.4f}")

    # CV (gradient descent with control variates)
    print("  Tuning CV...")
    best_ari = -1
    best_lr = None
    for lr in E.CV_LRS:
        try:
            r = C.run_method("CV", 42, lr=lr,
                            init_mode=E.INIT_MODE, sigma_source=E.SIGMA,
                            batch_mode=E.BATCH_MODE, batch_seed=42,
                            restart_every=E.RESTART, verbose=False)
            ari = float(r["ari_final"])
            if ari > best_ari:
                best_ari = ari
                best_lr = lr
            print(f"    lr={lr:.1e}: ARI={ari:.4f}")
        except Exception as e:
            print(f"    lr={lr:.1e}: FAILED ({e})")
    tuned["CV"] = {"lr": best_lr}
    print(f"  -> Best CV: lr={best_lr:.1e}, ARI={best_ari:.4f}")

    # SVI and PAVI are fixed (no tuning)
    tuned["SVI"] = {}
    tuned["PAVI_mb"] = {}

    # Save tuning results
    with open(output_file, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"[✓] Tuning results saved to {output_file}")
    return tuned


def run_experiments(tuned):
    """Run all methods x all seeds with tuned hyper-parameters."""
    print("[*] Running full experiment (all methods x all seeds)...")
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    summary = {}
    for method_arg, display, *_ in E.METHODS:
        aris, elbos, w2s = [], [], []
        print(f"\n  {display}:")
        for seed in E.SEEDS:
            npz = f"{C.OUTPUT_DIR}/{display}{C.SUFFIX}_seed{seed}.npz"
            try:
                if os.path.exists(npz):
                    # Incremental: reuse saved run
                    d = np.load(npz)
                    a = float(d["ari_final"])
                    e = float(d["elbo"][-1])
                    try:
                        w2 = float(w2_trajectory(d["m_traj"], d["sigma0"], C.TRUE_MEANS, C.TRUE_COVS)[0][-1])
                    except Exception:
                        w2 = float("nan")
                    print(f"    seed {seed}: ARI={a:.4f} (loaded from {npz})")
                else:
                    # Run method
                    kw = E.run_kwargs(method_arg, seed, tuned)
                    r = C.run_method(method_arg, seed, tag=display, save=True, verbose=False, **kw)
                    a = float(r["ari_final"])
                    e = float(r["elbo"][-1])
                    try:
                        w2 = float(w2_trajectory(r["m_traj"], r["sigma0"], C.TRUE_MEANS, C.TRUE_COVS)[0][-1])
                    except Exception:
                        w2 = float("nan")
                    print(f"    seed {seed}: ARI={a:.4f}")
            except Exception as ex:
                print(f"    seed {seed}: FAILED ({type(ex).__name__})")
                a = e = w2 = float("nan")
            aris.append(a)
            elbos.append(e)
            w2s.append(w2)

        arr = np.array(aris)
        summary[display] = {
            "ari_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
            "ari_std": float(np.nanstd(arr)) if np.isfinite(arr).any() else float("nan"),
            "elbo_mean": float(np.nanmean(elbos)) if np.isfinite(elbos).any() else float("nan"),
            "elbo_std": float(np.nanstd(elbos)) if np.isfinite(elbos).any() else float("nan"),
            "w2_mean": float(np.nanmean(w2s)) if np.isfinite(w2s).any() else float("nan"),
            "w2_std": float(np.nanstd(w2s)) if np.isfinite(w2s).any() else float("nan"),
        }

    # Save summary
    summary_file = f"{C.OUTPUT_DIR}/summary_exp.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Full experiment results saved to {summary_file}")
    print("\nSummary:")
    print("Method       | ARI (mean±std)     | ELBO (mean±std)       | W2 (mean±std)")
    print("-" * 80)
    for display, stats in summary.items():
        ari_str = f"{stats['ari_mean']:.3f}±{stats['ari_std']:.3f}"
        elbo_str = f"{stats['elbo_mean']:.2e}±{stats['elbo_std']:.2e}"
        w2_str = f"{stats['w2_mean']:.3f}±{stats['w2_std']:.3f}"
        print(f"{display:12} | {ari_str:18} | {elbo_str:21} | {w2_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Two Moons synthetic clustering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--tune-only", action="store_true",
                       help="Only run hyper-parameter tuning")
    parser.add_argument("--run-only", action="store_true",
                       help="Only run experiments (requires existing tuning results)")
    args = parser.parse_args()

    t_start = time.time()

    if args.run_only:
        # Load existing tuning results
        tuning_file = f"{C.OUTPUT_DIR}/tuning_exp.json"
        if not os.path.exists(tuning_file):
            print(f"Error: Tuning results not found at {tuning_file}")
            print("Please run with --tune-only first, or without flags to run the full pipeline.")
            sys.exit(1)
        with open(tuning_file) as f:
            tuned = json.load(f)
        run_experiments(tuned)
    elif args.tune_only:
        run_tuning()
    else:
        # Full pipeline: tune -> run -> figures
        tuned = run_tuning()
        run_experiments(tuned)

    elapsed = time.time() - t_start
    print(f"\n[✓] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
