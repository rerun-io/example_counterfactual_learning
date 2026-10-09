#!/usr/bin/env bash
# 6-7. Fine-tune the nominal policy on clean demos + each selection, streamed from the catalog (clean demos with
# RerunIterableDataset, repair set with RerunMapDataset), then evaluate each repaired policy. ~1 h per fine-tune at
# 800 epochs on an Apple M-series GPU; EPOCHS=400 halves it.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
require_server
EPOCHS="${EPOCHS:-800}"
S0="${SEEDS[0]}"
CONDS=(appearance_L0 lighting_L3 combo_app3_dis3_lit3_sup3)
for SEL in "s${S0}_random_k20_v1" "s${S0}_drift_weighted_k20_v1"; do
  CK="$ROOT/ckpts/repair_${SEL}_hubclean"
  if [[ ! -f "$CK/policy_last.ckpt" ]]; then
    mkdir -p "$CK"; echo "== fine-tune $SEL ($EPOCHS epochs, $(date +%H:%M))"
    python -m cfnbc.finetune --selection "$SEL" --seed "$S0" --epochs "$EPOCHS" --clean_source hub \
      | tee "$CK/train.log" | grep --line-buffered -E "repair set|clean demos|^epoch|saved"
  fi
  for C in clean "${CONDS[@]}"; do
    TAG=""; ARGS=(); [[ "$C" != clean ]] && TAG="_$C" && ARGS=(--nuisance "$C")
    [[ -f "$CK/rollouts_policy_last_tagg_seed1000_n100$TAG.csv" ]] && continue
    python -m cfnbc.fast_eval --seed "$S0" --ckpt_dir "$CK" ${ARGS[@]+"${ARGS[@]}"} > "$CK/eval_$C.log" 2>&1
    echo "$SEL $C: $(grep 'success_rate=' "$CK/eval_$C.log")"
  done
done
python -m cfnbc.qa --rollouts | tail -1
python -m cfnbc.scene3d --all_rollouts
python -m cfnbc.hub sync
