"""Ghost arms: the policy's intended next pose under the clean frame and under a counterfactual, as 3D robot meshes.

Drift is open-loop: the clean and counterfactual frames show the same recorded state, so the cameras can't show it. This
layer makes it visible. For each step with predictions (layers `clean_pred` and `<condition>` on cube_demos), the
predicted first action is un-normalized, mapped onto the robot's joints exactly as sim_env.before_step does (arm joints
direct, normalized gripper -> finger joints (+g, -g)), and the arm meshes are posed by forward kinematics. Two ghosts are
logged under /world/scene, next to the recorded robot of the `scene3d` layer:

    /world/scene/ghost_clean/...        what the policy commands from the clean frame
    /world/scene/ghost_<condition>/...  what it commands from the counterfactual frame

    python -m cfnbc.ghost --segments demo_e032 demo_e011 --condition lighting_L3 [--seed 0]

Writes <rerun_root>/ghost/ghost_<condition>/<segment>.rrd (layer `ghost_<condition>`); `python -m cfnbc.hub sync`
registers it. Predictions exist every 10th step, so a ghost holds its pose until the next prediction.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

from cfnbc import hub
from cfnbc import scene3d as SC
from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[2]
GRIPPER_OPEN, GRIPPER_CLOSE = 0.05800, 0.01844   # act/constants.py PUPPET_GRIPPER_POSITION_{OPEN,CLOSE}
GHOST_RGBA = {"clean": (42, 120, 214, 110), "counterfactual": (235, 104, 52, 110)}


def action_to_qpos(a: np.ndarray) -> np.ndarray:
    """14-dim ACT action (un-normalized) -> the robot's 16 joint positions, as sim_env.BimanualViperXTask.before_step."""
    g = lambda x: x * (GRIPPER_OPEN - GRIPPER_CLOSE) + GRIPPER_CLOSE
    gl, gr = g(a[6]), g(a[13])
    return np.concatenate([a[:6], [gl, -gl], a[7:13], [gr, -gr]])


def robot_geoms(m) -> list[int]:
    return [g for g in SC.visible_geoms(m) if (m.id2name(int(m.geom_bodyid[g]), "body") or "").startswith("vx300s")]


def predictions(client, segment: str, condition: str, seed: int, stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from datafusion import col
    c, n = f"/policy/seed{seed}/clean_pred/action0", f"/policy/seed{seed}/{condition}/action0"
    t = (client.get_dataset(S.DATASET_DEMOS).filter_segments([segment]).filter_contents([c, n]).reader(index="step")
         .filter(col(f"{c}:Scalars:scalars").is_not_null() & col(f"{n}:Scalars:scalars").is_not_null())
         .sort(col("step")).to_pandas())
    unnorm = lambda v: np.stack(v.map(np.asarray)) * stats["action_std"] + stats["action_mean"]
    return t["step"].to_numpy(), unnorm(t[f"{c}:Scalars:scalars"]), unnorm(t[f"{n}:Scalars:scalars"])


def write_ghosts(client, segment: str, condition: str, seed: int = 0) -> Path:
    with open(ROOT / "ckpts" / f"seed{seed}" / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    steps, a_clean, a_cf = predictions(client, segment, condition, seed, stats)
    physics = SC.load_physics()
    m = physics.model
    geoms = robot_geoms(m)
    out = S.rerun_root() / "ghost" / f"ghost_{condition}" / f"{segment}.rrd"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(S.APP_ID, recording_id=segment)
    rec.save(out)
    rr.send_property("ghost", rr.AnyValues(condition=condition, policy_seed=seed), recording=rec)
    idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]
    for name, actions, rgba in (("clean", a_clean, GHOST_RGBA["clean"]), (condition, a_cf, GHOST_RGBA["counterfactual"])):
        root = f"{SC.SCENE_ROOT}/ghost_{name}"
        pos = np.empty((len(steps), len(geoms), 3))
        quat = np.empty((len(steps), len(geoms), 4))
        q = np.empty(4)
        for t, a in enumerate(actions):
            physics.data.qpos[:16] = action_to_qpos(a)
            physics.forward()
            for k, g in enumerate(geoms):
                pos[t, k] = physics.data.geom_xpos[g]
                mujoco.mju_mat2Quat(q, physics.data.geom_xmat[g])
                quat[t, k] = (q[1], q[2], q[3], q[0])
        for k, g in enumerate(geoms):
            path = f"{root}/{(m.id2name(g, 'geom') or f'geom_{g}').replace('/', '_')}"
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = int(m.geom_dataid[g])
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            rr.log(path, rr.Mesh3D(vertex_positions=m.mesh_vert[v0:v0 + nv], triangle_indices=m.mesh_face[f0:f0 + nf],
                                   albedo_factor=rgba), static=True, recording=rec)
            rr.send_columns(path, indexes=idx, recording=rec,
                            columns=rr.Transform3D.columns(translation=pos[:, k], quaternion=quat[:, k]))
    rec.flush()
    rec.disconnect()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--segments", nargs="+", required=True)
    p.add_argument("--condition", default="lighting_L3")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    with hub.open_catalog() as client:
        for seg in a.segments:
            print("wrote", write_ghosts(client, seg, a.condition, a.seed))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "disable")
    main()
