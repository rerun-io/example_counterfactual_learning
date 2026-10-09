#!/usr/bin/env bash
# Check the GPU and MuJoCo rendering work (run in a normal terminal session: rendering needs a display on macOS).
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
python - <<'PY'
import os, torch, rerun
from dm_control import mujoco
dev = os.environ.get("ACT_DEVICE", "cpu")
x = torch.randn(256, 256, device=dev); print(f"torch on {dev} ok:", bool((x @ x).sum() != 0))
p = mujoco.Physics.from_xml_path("assets/bimanual_viperx_transfer_cube.xml")
img = p.render(height=480, width=640, camera_id="top"); print("mujoco render ok", img.shape)
print("rerun", rerun.__version__)
PY
