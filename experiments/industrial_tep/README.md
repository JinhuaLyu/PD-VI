# Tennessee Eastman Process (TEP) Industrial Experiment

This directory contains the code to reproduce the Tennessee Eastman Process (TEP) industrial fault detection experiment from the paper.

## Quick Start

```bash
# From this directory
python run.py                 # Run full pipeline (tune -> run -> summarize)
python run.py --tune-only     # Only run hyper-parameter tuning
python run.py --run-only      # Only run experiments (requires tuning results)
```

## Overview

The TEP experiment applies Gaussian Mixture Model clustering to industrial fault detection data from the Tennessee Eastman Process simulator. The dataset contains:

- **1298 samples** (fault-free and multiple fault types)
- **52 features** (after dimensionality reduction)
- **Embedded features** (tep_embeddings.npz)
- **Number of clusters**: 6

The experiment compares six clustering methods on this real industrial dataset.

## Experimental Setting

All methods use the same configuration:

- **Initialization**: k-means++ for cluster centers
- **Scales (σ₀, σ₁)**: Computed from data statistics
- **Batching**: Class-stratified, randomly drawn with replacement
- **Cluster restart**: Every 100 iterations (PD-VI methods only)
- **Total iterations**: 20,000 (~116 epochs over 172 time batches)
- **Random seeds**: [42, 1, 2, 3, 4]
- **Sample batch size**: 500

## Workflow

### 1. Hyper-parameter Tuning

The script tunes each method on seed 42 to find the best learning rate / step size:

- **P²D-VI**: Grid search over (η_m, η_s)
- **PD-VI**: Single η
- **AdamW**: Learning rate tuning
- **CV**: Control variate learning rate tuning
- **SVI**: Fixed schedule (no tuning)
- **PAVI**: Step size auto-tuned and cached

### 2. Full Experiment

Runs all methods on all 5 seeds with tuned hyper-parameters.

### 3. Results Summary

Results are saved in `output/`:

- `tuning_exp.json`: Tuned hyper-parameters and seed-42 metrics
- `summary_exp.json`: Mean ± std statistics across all 5 seeds
- `{METHOD}_seed{SEED}.npz`: Full trajectory for each (method, seed) pair

## Results

The TEP experiment demonstrates the algorithm's effectiveness on a real industrial dataset with multiple fault types.

## File Reference

| File | Purpose |
|------|---------|
| `run.py` | Main driver: tuning, full experiment, results summary |
| `common.py` | Unified simulation engine for TEP data |
| `config.py` | Method grid and hyperparameter ranges |
| `metrics.py` | ARI, ELBO, Wasserstein-2 evaluation |
| `pavi.py` | Particle mean-field VI implementation |
| `plot_tep.py` | Generate manuscript figures |
| `plot_config.py` | Figure styling configuration |
| `curves_main_tep.py` | Generate learning curves |
| `data/` | TEP embedding data (tep_embeddings.npz) |
| `output/` | Results (tuning, experiments, trajectories) |

## Data Format

The `tep_embeddings.npz` file contains:

- `X`: (1298, 52) embedded features
- `label_cluster`: (1298,) cluster labels for evaluation (ARI computation)

## Notes

- TEP uses 20,000 iterations (vs 10,000 for synthetic Moon) due to the larger dataset
- All ARI scores use a unified plug-in nearest-centre rule
- Non-IID batch mode: stratified batches are visited randomly with replacement
- Results on real industrial data show algorithm robustness beyond synthetic settings
