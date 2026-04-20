#!/usr/bin/env bash

SEEDS=(1 16 32)

OBS_MODES=(
  "graph_delaunay"
  "graph_knn"  
  "scalars_cells_substrates"
  "img_mc_cells_substrates"
  "img_mc_cells"
  "img_mc_substrates"
  "scalars_cells"
  "scalars_substrates")


for seed in "${SEEDS[@]}"; do
  for obs in "${OBS_MODES[@]}"; do
    echo "Running seed=${seed}, observation_mode=${obs}"
    python custom_modules/physigym/physigym/envs/run_physigym_tip_async_sac.py \
      --seed "${seed}" \
      --observation_mode "${obs}"
  done
done
