# Spatial Transcriptomics Experiment (MOSTA Dataset)

This directory contains the code to reproduce the spatial transcriptomics (MOSTA - Mouse Organogenesis Spatiotemporal transcriptomics atlas) clustering experiment from the paper.

## Overview

This experiment applies the P²D-VI algorithm to spatial transcriptomics data from the MOSTA dataset, including:

- **Data**: Mouse organogenesis spatiotemporal atlas (multiple tissue types)
- **Task**: Spatial clustering with constraint propagation
- **Graph**: Constructed from spatial coordinates with similarity-weighted edges
- **Results**: ARI and ELBO evaluation on multiple datasets

## Quick Start

The main entry point is `run_merfish_official.py`, which orchestrates multi-GPU parallel runs:

```bash
# Run official MERFISH results (5 seeds on multiple GPUs)
python run_merfish_official.py [--n_gpus 5] [--config config.yaml]

# Or on a single GPU
CUDA_VISIBLE_DEVICES=0 python run_merfish_official.py --n_gpus 1
```

## Workflow

### 1. Data Preparation

Before running the algorithm, you need to:

1. Obtain the raw MOSTA/MERFISH data (see data section below)
2. Preprocess and embed the data:

```bash
python data_processing/data_preprocess.py --config config.yaml \
    --raw_file MERFISH_0.04.h5ad --dataset_id MERFISH_0.04

python data_processing/prepare_data.py --config config.yaml \
    --dataset_id MERFISH_0.04 --seed 1
```

### 2. Graph Construction

The experiment constructs spatial graphs from coordinates:

- **Tangent computation**: Local Fisher information matrix for initial tangent estimation
- **Circular neighbor graph**: Builds weighted graph based on spatial proximity and cluster similarity
- **Lambda scheduling**: Similarity weighting dynamically updated during training

### 3. Algorithm Execution

The `run_merfish_official.py` script:

1. Rebuilds graph files to ensure consistent neighbor configuration
2. Spawns parallel runs across multiple GPUs (default: 5 seeds on 5 GPUs)
3. Collects ARI and ELBO metrics from each run
4. Generates summary statistics

## Configuration

The `config.yaml` file controls:

- **Data paths**: Base directory, intermediate files, output location
- **Preprocessing**: Gene selection, PCA, normalization
- **Graph construction**: Neighbor parameters, radius, lambda weighting
- **Algorithm**: Method selection, hyperparameters, temperature schedule
- **Training**: Iterations, device, scheduling

Example section for MERFISH:

```yaml
data:
  dataset_id: MERFISH_0.04
  base_dir: ../data/MERFISH_hypothalamic
  output_dir: ../output/MERFISH

graph:
  neighbors:
    radius: 43            # spatial distance threshold
    num: 25               # target number of neighbors
  weights:
    lambda_similarity: 1.0

training:
  algorithm: ours        # use P²D-VI
  seed: 1
  max_iter: 20000
```

## File Reference

| File | Purpose |
|------|---------|
| `run_merfish_official.py` | Main orchestrator for official results (multi-GPU parallel) |
| `config.yaml` | Configuration for MERFISH dataset (data paths, hyperparams) |
| `algorithms/` | Algorithm implementations (ours, sgd, adamw, etc.) |
| `data_processing/` | Scripts for data preparation and graph construction |
| `data_processing/data_preprocess.py` | Raw data preprocessing (normalization, HVG, PCA) |
| `data_processing/prepare_data.py` | Prepare training data and graph files |

## Algorithm Details

The spatial transcriptomics setting extends the base GMM clustering:

- **Data**: Gene expression vectors with spatial coordinates
- **Graph**: Constraint propagation via spatial neighborhood
- **Constraint**: Similarity-weighted edges encourage neighboring points to share clusters
- **Optimization**: Local inner solver + global dual averaging
- **Parallel**: Multiple seeds run in parallel on separate GPUs

## Data Access

To obtain the MOSTA/MERFISH data:

1. **MOSTA**: Available from the spatial-transcriptomics-atlas repositories
2. **MERFISH**: Hypothalamic preoptic region data (5 slices)
3. **Format**: h5ad (AnnData HDF5 format) or equivalent expression matrices

The `data_processing/` scripts assume data is in `../data/MERFISH_hypothalamic/` with filenames like `MERFISH_0.04.h5ad`.

## Results

The experiment reports:

- **ARI**: Adjusted Rand Index for cluster quality (compared to ground-truth domains)
- **ELBO**: Evidence lower bound (objective being optimized)
- **W₂**: Wasserstein-2 distance between learned and true distributions (when applicable)

Run across 5 random seeds to assess stability.

## GPU Requirements

- **Single GPU**: Run with `--n_gpus 1` (requires ~12 GB VRAM)
- **Multi-GPU**: Default is 5 GPUs in parallel for 5 seeds
- **CPU mode**: Not recommended (substantially slower)

## Notes

- The official run uses circular neighbor graphs (ignores tangent direction)
- Graph construction is deterministic once data is prepared
- Hyperparameters (η_m, η_s, lambda_similarity) are pre-tuned in config.yaml
- Results are saved in `output/` with metadata (ARI, ELBO, runtime)
