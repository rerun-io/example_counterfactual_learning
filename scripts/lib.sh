# shared helpers for the pipeline scripts
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"
SEEDS=(${SEEDS:-0})   # nominal policy seeds; the post used seed 0 (the paper uses 3: SEEDS="0 1 2")

require_server() {
  python - <<'PY' || { echo "No catalog at $CFNBC_CATALOG_URL. Start it in another terminal: bash scripts/serve.sh"; exit 1; }
import os, rerun as rr
rr.catalog.CatalogClient(os.environ["CFNBC_CATALOG_URL"]).dataset_names()
PY
}
