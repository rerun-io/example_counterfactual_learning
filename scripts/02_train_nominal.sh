#!/usr/bin/env bash
# 1b. Train the nominal ACT policy (2000 epochs, ACT's README hyperparameters) for each seed in $SEEDS (default 0).
# Training runs are logged to rerun/training. Resumable: seeds with a finished checkpoint are skipped.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
for SEED in "${SEEDS[@]}"; do
  CK="$ROOT/ckpts/seed$SEED"
  [[ -f "$CK/policy_last.ckpt" ]] && { echo "seed $SEED already trained"; continue; }
  mkdir -p "$CK"; echo "== training seed $SEED ($(date +%H:%M))"
  ACT_RUN_NAME="seed$SEED" python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir "$CK" \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000 --lr 1e-5 --seed "$SEED" > "$CK/train.log" 2>&1
  grep "Best ckpt" "$CK/train.log"
done
