#!/usr/bin/env bash
# 1. Record: 50 scripted demos with full simulator state (failed attempts are regenerated), exported to rerun/demos.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
D="$ACT_DATA_DIR/sim_transfer_cube_scripted"
if ls "$D"/episode_*.hdf5 >/dev/null 2>&1; then echo "$D already has episodes, skipping recording"; else
  mkdir -p "$ROOT/results"
  python record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir "$D" --num_episodes 50 --seed 0 \
    --success_log "$ROOT/results/demo_success.csv" | grep --line-buffered -E "Successful|Failed|^Success"
fi
python -m cfnbc.export_demos --dataset_dir "$D"
python -m cfnbc.training_layers          # train_pad layer: ACT's zero padding past the episode end, for the Hub loaders
