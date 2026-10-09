#!/usr/bin/env bash
# The same workflow on real teleop data (lerobot/aloha_static_cups_open, MIT): ingest, QA, counterfactual edits with
# masks, drift, selection, repair fine-tunes and an offline held-out evaluation. ~4.5 h, mostly training.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
require_server
EPOCHS="${EPOCHS:-800}"
DS="$ROOT/data_real/aloha_static_cups_open"
if [[ ! -d "$DS/meta" ]]; then
  hf download lerobot/aloha_static_cups_open --repo-type dataset --revision d793c969cf716001dcca18a0842c3d7e9de9e41b \
    --local-dir "$DS"
fi
[[ -d "$ACT_RERUN_DIR/real_demos" ]] || python -m cfnbc.real.ingest --dataset "$DS"
python -m cfnbc.real.qa_real
python -m cfnbc.real.repair_real --pad
python -m cfnbc.hub sync --datasets real_demos
[[ -d "$ROOT/data_real/act_cups_open" ]] || python -m cfnbc.real.to_hdf5          # ACT's HDF5 layout, for its own trainer
if [[ ! -f "$ROOT/ckpts/real_cups_s0/policy_last.ckpt" ]]; then
  mkdir -p "$ROOT/ckpts/real_cups_s0"; echo "== train real ACT ($(date +%H:%M))"
  python -m cfnbc.real.train_real --seed 0 --epochs 2000 > "$ROOT/ckpts/real_cups_s0/train.log" 2>&1
fi
python -m cfnbc.real.compute_drift_real | grep -E "violation|table"
python -m cfnbc.hub sync --datasets real_demos --replace
python -m cfnbc.selection --table real_drift_candidates --prefix real --k 20 \
  --methods random top_drift drift_weighted --exclude_heldout
for SEL in real_random_k20_v1 real_drift_weighted_k20_v1; do
  CK="$ROOT/ckpts/repair_$SEL"
  [[ -f "$CK/policy_last.ckpt" ]] && continue
  mkdir -p "$CK"; echo "== fine-tune $SEL ($(date +%H:%M))"
  python -m cfnbc.finetune --real --selection "$SEL" --epochs "$EPOCHS" | tee "$CK/train.log" | grep -E "repair set|saved"
done
python -m cfnbc.real.offline_eval --stride 20 --device "$ACT_DEVICE" \
  --policies real_cups_s0 repair_real_random_k20_v1 repair_real_drift_weighted_k20_v1
python -m cfnbc.hub sync
