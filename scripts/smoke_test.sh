#!/usr/bin/env bash
# Tiny end-to-end check: 3 demos, 5 training epochs, 1 eval rollout, export to Rerun. Writes to data_smoke/, rerun_smoke/.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
export ACT_DATA_DIR="$ROOT/data_smoke" ACT_RERUN_DIR="$ROOT/rerun_smoke" ACT_RUN_NAME=smoke
D="$ACT_DATA_DIR/sim_transfer_cube_scripted"
python record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir "$D" --num_episodes 3 --seed 123
for i in $(seq 3 49); do ln -sf "$D/episode_$((i%3)).hdf5" "$D/episode_$i.hdf5"; done   # ACT's loader expects 50
ARGS=(--task_name sim_transfer_cube_scripted --ckpt_dir "$ROOT/ckpts/smoke" --policy_class ACT --kl_weight 10
      --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --lr 1e-5 --seed 0)
mkdir -p "$ROOT/ckpts/smoke"
python imitate_episodes.py "${ARGS[@]}" --num_epochs 5 | grep -E "^(Val|Train) loss|Best"
python imitate_episodes.py "${ARGS[@]}" --num_epochs 5 --eval --temporal_agg --num_rollouts 1 \
  --eval_ckpt_names policy_last.ckpt --no_save_video
python -m cfnbc.export_demos --dataset_dir "$D"
echo "smoke test ok: recordings in $ACT_RERUN_DIR"
