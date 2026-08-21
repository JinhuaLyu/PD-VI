# Primal-Dual Variational Inference (PD-VI & P²D-VI)

This repository contains code to reproduce all experiments from the paper on primal-dual variational inference methods for Gaussian mixture model (GMM) clustering.

## Paper Overview

The paper introduces two primal-dual variational inference methods:

1. **PD-VI**: Primal-dual VI with a single constant step size η
2. **P²D-VI**: Preconditioned version with separate step sizes for means (η_m) and log-variances (η_s)

Both methods use:
- Primal-dual algorithm updates with dual averaging
- Local inner solver for batch-level optimization
- Cluster restart for collapsed cluster recovery
- Non-IID stratified batching

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Two Moons synthetic experiment
cd experiments/synthetic_twomoons
python run.py

# Tennessee Eastman Process (TEP)
cd ../industrial_tep
python run.py

# Spatial Transcriptomics (MOSTA)
cd ../spatial_transcriptomics_mosta
python run_merfish_official.py --n_gpus 5
```

## Experiments

### 1. Synthetic Two Moons (`experiments/synthetic_twomoons/`)

Binary clustering on 500 samples from two half-moons.

- **Methods**: P²D-VI, PD-VI, SVI, AdamW, CV, PAVI
- **Seeds**: [42, 1, 2, 3, 4]
- **Iterations**: 10,000
- **Results**: ARI, ELBO, Wasserstein-2

```bash
cd experiments/synthetic_twomoons
python run.py                 # Full pipeline
python run.py --tune-only     # Tuning only
python plot_exp.py            # Generate figures
```

### 2. Industrial TEP (`experiments/industrial_tep/`)

Fault detection on 1298 samples from Tennessee Eastman Process simulator.

- **Dataset**: 52-dimensional real data, 6 clusters (fault types)
- **Methods**: Same as Two Moons
- **Iterations**: 20,000

```bash
cd experiments/industrial_tep
python run.py
python plot_tep.py            # Generate figures
```

### 3. Spatial Transcriptomics (`experiments/spatial_transcriptomics_mosta/`)

Gene expression clustering with spatial constraints on MOSTA/MERFISH data.

- **Data**: Multiple tissue types and MERFISH slices
- **Graph**: Spatial neighborhood with similarity weighting
- **Parallel**: Multi-GPU execution (5 seeds on 5 GPUs)

```bash
cd experiments/spatial_transcriptomics_mosta
python run_merfish_official.py --n_gpus 5
```

## Repository Structure

```
PD-VI_repo/
├── README.md
├── requirements.txt
├── experiments/
│   ├── synthetic_twomoons/      # Two Moons experiment
│   │   ├── run.py               # Main driver
│   │   ├── common.py            # Unified engine
│   │   ├── config.py            # Hyperparameter grids
│   │   ├── data/                # Synthetic data
│   │   └── output/              # Results
│   ├── industrial_tep/          # TEP experiment
│   │   ├── run.py
│   │   ├── common.py
│   │   ├── data/                # TEP embeddings
│   │   └── output/              # Results
│   └── spatial_transcriptomics_mosta/  # MOSTA experiment
│       ├── run_merfish_official.py
│       ├── config.yaml
│       ├── algorithms/
│       └── data_processing/
└── outputs/                     # Example results
```

## Hyperparameter Tuning

Each method is tuned on seed 42 via grid search and applied to all remaining seeds.

**P²D-VI** (ours): Grid over (η_m, η_s) ∈ [3e-4, 1e-3, 3e-3] × [3e-3, 1e-2, 3e-2]

**PD-VI** (ours): Single η ∈ [3e-4, 1e-3, 3e-3, 1e-2]

**Baselines**: Learning rates tuned individually; SVI uses fixed schedule

## Results

### Two Moons (5 seeds, mean ± std)

| Method | ARI | ELBO (median) | W₂ (median) |
|--------|-----|---------------|-------------|
| **P²D-VI** | 0.916 ± 0.005 | **-1.33e4** | **0.069** |
| **PD-VI** | 0.918 ± 0.014 | -1.89e4 | 0.070 |
| AdamW | 0.716 ± 0.142 | -1.52e4 | 0.266 |
| PAVI | 0.680 ± 0.112 | -1.45e4 | 0.257 |
| SVI | 0.315 ± 0.065 | -1.71e5 | 0.837 |
| CV | 0.250 ± 0.043 | -2.23e5 | 0.913 |

## Evaluation Metrics

- **ARI** (Adjusted Rand Index): Clustering agreement [−1, 1]
- **ELBO**: Variational objective (higher is better)
- **W₂**: Wasserstein-2 distance from learned to true distribution

## Notes

- All batch modes: Stratified (class-balanced), randomly drawn with replacement
- Cluster restart: Every 100 iterations (PD-VI methods only)
- Numerical safety: Log-variances clamped to [log(1e-10), log(1e2)]
- Reproducibility: Set seeds explicitly; GPU operations are deterministic with `torch.manual_seed()`

## For More Details

See experiment-specific README files:
- `experiments/synthetic_twomoons/README.md`
- `experiments/industrial_tep/README.md`
- `experiments/spatial_transcriptomics_mosta/README.md`
