#!/usr/bin/env bash
# 3-4. Counterfactuals: check the nuisances preserve the task (3D scene layers + geometry checks, CPU), evaluate the
# nominal policy under each condition (GPU), and measure action drift on paired clean/counterfactual frames with the
# frames and predictions stored as layers (GPU).
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
require_server
CONDS=(appearance_L0 appearance_L1 appearance_L2 appearance_L3 distractor_L0 distractor_L1 distractor_L2 distractor_L3
       lighting_L3 support_L3 combo_app3_dis3_lit3_sup3)
python -m cfnbc.scene3d --episodes $(seq 0 49)                 # robot + cube scene from recorded state, per demo
python -m cfnbc.task_preservation --scene_layers               # clearance / occlusion checks -> table task_preservation
for SEED in "${SEEDS[@]}"; do
  for C in clean "${CONDS[@]}"; do
    TAG=""; ARGS=(); [[ "$C" != clean ]] && TAG="_$C" && ARGS=(--nuisance "$C")
    OUT="$ROOT/ckpts/seed$SEED/rollouts_policy_last_tagg_seed1000_n100$TAG.csv"
    [[ -f "$OUT" ]] && continue
    echo "== eval seed $SEED $C ($(date +%H:%M))"
    python -m cfnbc.fast_eval --seed "$SEED" ${ARGS[@]+"${ARGS[@]}"} --num_envs 16 > "$ROOT/ckpts/seed$SEED/eval_$C.log" 2>&1
    grep "success_rate=" "$ROOT/ckpts/seed$SEED/eval_$C.log"
  done
done
python -m cfnbc.compute_drift --seeds "${SEEDS[@]}" --conditions "${CONDS[@]}" --rerun_layers --device "$ACT_DEVICE" | grep -E "candidates|table"
python -m cfnbc.qa --rollouts | tail -1                        # stable stage, final cube height, strict success
python -m cfnbc.scene3d --all_rollouts                         # 3D scene for every rollout
python -m cfnbc.hub sync
python -m cfnbc.eval_table | tail -1
for SEED in "${SEEDS[@]}"; do python -m cfnbc.drift_query --seed "$SEED" | tail -3; done
