#!/usr/bin/env bash
# 5. Select K=20 repair candidates per method from the drift_candidates table -> table repair_selections. Then build
# ghost arms (the policy's intended pose, clean vs counterfactual) for the top drift episodes.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
require_server
python -m cfnbc.selection --seed "${SEEDS[0]}" --k 20 --methods random top_drift drift_weighted
TOP=$(python - <<'PY'
import os, rerun as rr
from datafusion import col, lit
t = rr.catalog.CatalogClient(os.environ["CFNBC_CATALOG_URL"]).get_table("drift_candidates").reader()
t = t.filter((col("condition") == lit("lighting_L3")) & (col("seed") == lit(int(os.environ.get("SEED0", "0")))))
print(" ".join(t.sort(col("drift").sort(ascending=False)).limit(2).to_pydict()["segment_id"]))
PY
)
SEED0="${SEEDS[0]}" python -m cfnbc.ghost --segments $TOP --condition lighting_L3 --seed "${SEEDS[0]}"
python -m cfnbc.hub sync --datasets cube_demos
