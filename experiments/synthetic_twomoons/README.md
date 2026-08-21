# Two Moons Synthetic Clustering Experiment

This directory contains the code to reproduce the Two Moons synthetic clustering experiment from the paper.

## Quick Start

```bash
# From this directory
python run.py                 # Run full pipeline (tune -> run -> summarize)
python run.py --tune-only     # Only run hyper-parameter tuning
python run.py --run-only      # Only run experiments (requires tuning results)
```

## Overview

The Two Moons experiment compares six clustering methods on a synthetic Gaussian Mixture Model with 2 clusters:

| Method | Type | Role |
|--------|------|------|
| **P²D-VI** (ours, main) | Primal-dual | Main algorithm with preconditioned step sizes (η_m, η_s) |
| **PD-VI** (ours, ablation) | Primal-dual | Constant step size version (single η) |
| **SVI** | Stochastic VI | Baseline with fixed schedule τ/(t+1)^κ |
| **AdamW** | Adaptive | Standard Adam with weight decay |
| **CV** | Control variates | Gradient descent with variance reduction |
| **PAVI** | Particle | Particle-based mean-field VI |

## Unified Experimental Setting

All methods use:
- **Initialization**: Random guarded (random Gaussian means, rejection-sampled for separation)
- **Scales (σ₀, σ₁)**: k-means++ initialization
- **Batching**: Class-stratified, randomly drawn with replacement
- **Cluster restart**: Every 100 iterations (PD-VI methods only)
- **Total iterations**: 10,000
- **Random seeds**: [42, 1, 2, 3, 4]

## Workflow

### 1. Hyper-parameter Tuning

The script tunes each method on seed 42 to find the best learning rate / step size:

- **P²D-VI**: Grid search over (η_m, η_s) ∈ [3e-4, 1e-3, 3e-3] × [3e-3, 1e-2, 3e-2]
- **PD-VI**: Single η ∈ [3e-4, 1e-3, 3e-3, 1e-2]
- **AdamW, CV**: Learning rate ∈ [1e-7, 3e-7, 1e-6, 3e-6, 1e-5]
- **SVI**: Fixed schedule (no tuning)
- **PAVI**: Step size auto-tuned, cached for all seeds

### 2. Full Experiment

Once tuning is complete, the script runs all methods on all 5 seeds with tuned hyper-parameters.

### 3. Results Summary

Results are saved in `output/`:

- `tuning_exp.json`: Tuned hyper-parameters and seed-42 metrics
- `summary_exp.json`: Mean ± std statistics across all 5 seeds
- `{METHOD}_seed{SEED}.npz`: Full trajectory for each (method, seed) pair
  - Contains: ELBO, ARI, wall-clock time, cluster means trajectory, etc.

## Results

Final results (5 seeds, mean ± std):

| Method | ARI | ELBO (median) | W₂ (median) |
|--------|-----|-----------------|-------------|
| **P²D-VI** | 0.916 ± 0.005 | **-1.33e4** | **0.069** |
| **PD-VI** | 0.918 ± 0.014 | -1.89e4 | 0.070 |
| AdamW | 0.716 ± 0.142 | -1.52e4 | 0.266 |
| PAVI | 0.680 ± 0.112 | -1.45e4 | 0.257 |
| SVI | 0.315 ± 0.065 | -1.71e5 | 0.837 |
| CV | 0.250 ± 0.043 | -2.23e5 | 0.913 |

**Key findings**:
- P²D-VI and PD-VI are tied on ARI (both at achievable ceiling) but P²D-VI wins decisively on ELBO (its objective)
- P²D-VI is the most stable (std 0.005)
- AdamW (tuned, adaptive) reaches only 0.716 ARI due to lack of cluster restart
- SVI and CV collapse; CV's shared nearest-centre rule recovers only weak structure

## File Reference

| File | Purpose |
|------|---------|
| `run.py` | Main driver: tuning, full experiment, results summary |
| `common.py` | Unified simulation engine (all methods share this) |
| `config.py` | Method grid, hyperparameter ranges, tuning setup |
| `metrics.py` | ARI, ELBO, Wasserstein-2 evaluation |
| `pavi.py` | Particle mean-field VI implementation |
| `plot_exp.py` | Generate manuscript figures (requires results in `output/`) |
| `plot_config.py` | Figure styling (colors, line styles, panel order) |
| `curves_main.py` | Generate learning curves |
| `data/` | Synthetic data files (moon_data.pt, cluster_gaussians.pt) |
| `output/` | Results (tuning, experiments, trajectories) |

## Reproducing the Paper Figures

After running the full experiment, generate the manuscript figures:

```bash
python plot_exp.py
# Outputs: figures/{panels.png, panels.pdf, curves.png, curves.pdf}
```

The figure styling (colors, line styles, panel order, legend) is configured in `config.py`.

## Environment

Required packages (see main repository README for full installation):

```
numpy
scipy
torch
scikit-learn
matplotlib
```

## Notes

- All ARI scores use a unified plug-in nearest-centre rule (independent of each method's q(z))
- ELBO uses `symlog` scale in plots (spans ~4 orders of magnitude)
- Particle method (PAVI) step size is auto-tuned on seed 42 and cached for subsequent seeds
- Non-IID batch mode: stratified batches are visited randomly with replacement
