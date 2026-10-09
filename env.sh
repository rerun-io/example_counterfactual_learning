# Source this before running anything:  source env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
source "$ROOT/.venv/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"     # macOS: glfw; Linux without a display: egl
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACT_DEVICE="${ACT_DEVICE:-mps}"     # mps (Apple silicon), cuda, or cpu
export ACT_DATA_DIR="$ROOT/data"
export TORCH_HOME="$ROOT/.cache/torch"
export MPLBACKEND=Agg MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export PYTHONUNBUFFERED=1
export ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-0}"   # DataLoader workers for ACT's own loader
export ACT_RERUN="${ACT_RERUN:-1}"               # log training runs and eval rollouts to Rerun
export ACT_RERUN_DIR="${ACT_RERUN_DIR:-$ROOT/rerun}"
# The catalog server started by scripts/serve.sh. Point this at a Rerun Hub instead (and set CFNBC_HUB=1) if you have one.
export CFNBC_CATALOG_URL="${CFNBC_CATALOG_URL:-rerun+http://127.0.0.1:51234}"
cd "$ROOT/act"
