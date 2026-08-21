# Data processing scripts for the bio pipeline.
# Run each script independently with --config path/to/config.yaml
# Pipeline order:
#   1. data_preprocess.py
#   2. pca_data.py       (requires --seed)
#   3. compute_tangent.py  (requires --seed)
#   4. compute_neighbors.py
#   5. compute_weight.py  (requires --seed)
