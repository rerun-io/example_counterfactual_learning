#!/usr/bin/env bash
# Create the Python environment, fetch ACT at the pinned upstream commit and apply our patch.
#   bash scripts/setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ACT_COMMIT=742c753c0d4a5d87076c8f69e5628c79a8cc5488   # tonyzhaozh/act "Update README.md"
if [[ ! -d act ]]; then
  git clone https://github.com/tonyzhaozh/act.git act
  git -C act checkout -q "$ACT_COMMIT"
  git -C act apply "$ROOT/act.patch"
  echo "act/ at $ACT_COMMIT with act.patch applied"
fi
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
uv python install 3.11
uv venv --python-preference only-managed --python 3.11 .venv
source .venv/bin/activate
# ACT's README pins mujoco 2.3.7 and dm_control 1.0.14; rerun-sdk 0.38 needs Python >= 3.10 and numpy 2.
uv pip install torch==2.8.0 torchvision==0.23.0 mujoco==2.3.7 dm_control==1.0.14 pyquaternion pyyaml einops h5py \
  opencv-python-headless matplotlib packaging tqdm scipy pandas "huggingface_hub[cli]" \
  "rerun-sdk[all,dataloader]==0.38.1"
uv pip install -e act/detr -e cfnbc
if [[ "$(uname)" == Darwin ]]; then
  # mujoco 2.3.7 on recent macOS: mujoco/cgl dlopens a path that no longer resolves. Point it at the real framework.
  CGL="$(python -c 'import site; print(site.getsitepackages()[0])')/mujoco/cgl/cgl.py"
  sed -i '' "s#'/System/Library/OpenGL.framework/OpenGL'#'/System/Library/Frameworks/OpenGL.framework/OpenGL'#" "$CGL"
fi
echo "done. Next: bash scripts/check_env.sh"
