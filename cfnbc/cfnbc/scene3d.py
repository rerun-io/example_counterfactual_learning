"""3D scene-from-state: log the MuJoCo scene (robot, table, cube, nuisance geometry) from recorded states.

Simulator-agnostic pattern: static meshes + per-step poses, no physics and no rendering. Meshes come from the
*compiled* model (MuJoCo recenters mesh vertices into the geom frame), poses from data.geom_xpos/geom_xmat after
mj_forward on each recorded qpos.

    python -m cfnbc.scene3d --episodes 0 49 [--condition distractor_L3] [--stride 2]

Writes <rerun_root>/scene3d/demo_eNNN.rrd (layer `scene3d`, clean) or
<rerun_root>/scene3d_nuisance/scene3d_<condition>/demo_eNNN.rrd (layer `scene3d_<condition>`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import mujoco
import numpy as np
import rerun as rr
from dm_control import mujoco as dmj

from cfnbc import nuisance as N
from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[2]
SCENE_ROOT = "/world/scene"
HIDDEN = N.HIDDEN_GROUP


def load_physics() -> dmj.Physics:
    return dmj.Physics.from_xml_path(str(ROOT / "act" / "assets" / N.NUISANCE_XML))


def geom_entity(m, g: int) -> str:
    return f"{SCENE_ROOT}/{(m.id2name(g, 'geom') or f'geom_{g}').replace('/', '_')}"


def visible_geoms(m) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_group[g] != HIDDEN]


def log_static_geometry(rec, physics, only_nuisance: bool = False) -> list[int]:
    """Log every visible geom's shape once (static). Returns the geom ids logged."""
    m = physics.model
    geoms = visible_geoms(m)
    if only_nuisance:
        geoms = [g for g in geoms if (m.id2name(g, "geom") or "").startswith("nuis_")]
    for g in geoms:
        rgba = (np.clip(m.geom_rgba[g], 0, 1) * 255).astype(np.uint8)
        path = geom_entity(m, g)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            rr.log(path, rr.Mesh3D(vertex_positions=m.mesh_vert[v0:v0 + nv], triangle_indices=m.mesh_face[f0:f0 + nf],
                                   albedo_factor=rgba), static=True, recording=rec)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            rr.log(path, rr.Boxes3D(half_sizes=[m.geom_size[g]], colors=[rgba], fill_mode="solid"), static=True,
                   recording=rec)
    return geoms


def geom_poses(physics, sim_qpos: np.ndarray, geoms: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(T, G, 3) positions and (T, G, 4) xyzw quaternions of `geoms` for each recorded qpos."""
    pos = np.empty((len(sim_qpos), len(geoms), 3))
    quat = np.empty((len(sim_qpos), len(geoms), 4))
    q = np.empty(4)
    for t, qp in enumerate(sim_qpos):
        physics.data.qpos[:] = qp
        physics.forward()
        for k, g in enumerate(geoms):
            pos[t, k] = physics.data.geom_xpos[g]
            mujoco.mju_mat2Quat(q, physics.data.geom_xmat[g])
            quat[t, k] = (q[1], q[2], q[3], q[0])
    return pos, quat


def log_scene(rec, sim_q: np.ndarray, condition: str, cube_init_pose, stride: int = 1, only_nuisance: bool = False) -> None:
    """Log static geometry + per-step geom poses for a recorded qpos sequence under `condition` into `rec`."""
    physics = load_physics()
    N.apply(physics, condition, cube_init_pose)
    rr.send_property("scene3d", rr.AnyValues(condition=condition, stride=stride), recording=rec)
    rr.log(SCENE_ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=rec)
    geoms = log_static_geometry(rec, physics, only_nuisance=only_nuisance)
    if not geoms and only_nuisance:
        raise ValueError(f"{condition} adds no geometry (appearance/lighting are not 3D-representable)")
    steps = np.arange(0, len(sim_q), stride)
    pos, quat = geom_poses(physics, sim_q[steps], geoms)
    idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]
    m = physics.model
    for k, g in enumerate(geoms):
        if m.geom_bodyid[g] == 0 or (np.ptp(pos[:, k], 0).max() == 0 and np.ptp(quat[:, k], 0).max() == 0):
            rr.log(geom_entity(m, g), rr.Transform3D(translation=pos[0, k], quaternion=quat[0, k]), static=True,
                   recording=rec)  # static geometry: one pose
        else:
            rr.send_columns(geom_entity(m, g), indexes=idx, recording=rec,
                            columns=rr.Transform3D.columns(translation=pos[:, k], quaternion=quat[:, k]))


def write_scene(e: int, condition: str = "clean", stride: int = 1, extra=None) -> Path:
    """Write the scene layer for demo episode e under `condition`. `extra(rec)` may log more (e.g. check results).

    clean: full scene (robot, table, cube). Other conditions: only the nuisance geometry (static), which overlays the
    clean `scene3d` layer when the segment is opened from the catalog.
    """
    with h5py.File(S.sim_data_dir() / f"episode_{e}.hdf5") as f:
        sim_q = f["sim_state/qpos"][()]
    seg = S.demo_segment_id(e)
    if condition == "clean":
        out_dir, layer = S.rerun_root() / "scene3d", "scene3d"
    else:
        layer = f"scene3d_{condition}"
        out_dir = S.rerun_root() / "scene3d_nuisance" / layer
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seg}.rrd"
    rec = rr.RecordingStream(S.APP_ID, recording_id=seg)
    rec.save(path)
    log_scene(rec, sim_q, condition, sim_q[0, 16:23], stride, only_nuisance=condition != "clean")
    if extra is not None:
        extra(rec)
    rec.flush()
    rec.disconnect()
    return path


def write_rollout_scenes(segments: list[str] | None = None, stride: int = 1) -> int:
    """3D scene layer for eval rollouts (layer `scene3d` on cube_rollouts), from each rollout's recorded /sim/qpos,
    with its nuisance geometry placed from the rollout's initial cube pose. Default selection: the rollouts worth
    looking at in 3D, i.e. dropped handoffs (upstream success, strict failure) and clean failures."""
    import gc
    from cfnbc import hub, query as Q
    out_dir = S.rerun_root() / "rollout_scene3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    with hub.open_catalog() as client:
        props = Q.rollout_table(client).merge(Q._props(client, S.DATASET_ROLLOUTS, "qa"), on="rerun_segment_id")
        if segments is None:
            pick = ((props.success == 1) & (props.success_strict == 0)) | ((props.nuisance == "clean") & (props.success == 0))
            segments = sorted(props[pick].rerun_segment_id)
        props = props.set_index("rerun_segment_id")
        ds = client.get_dataset(S.DATASET_ROLLOUTS)
        n = 0
        for seg in segments:
            t = ds.filter_segments(seg).filter_contents([S.SIM_QPOS]).reader(index=S.TIMELINE_STEP).to_arrow_table().sort_by(S.TIMELINE_STEP)
            sim_q = np.asarray([q for q in t[f"{S.SIM_QPOS}:Scalars:scalars"].to_pylist() if q])
            cond = props.loc[seg, "nuisance"]
            rec = rr.RecordingStream(S.APP_ID, recording_id=seg)
            rec.save(out_dir / f"{seg}.rrd")
            log_scene(rec, sim_q, cond, np.asarray(props.loc[seg, "init_pose"], dtype=np.float64), stride)
            rec.flush(); rec.disconnect(); del rec
            n += 1
            if n % 100 == 0:
                gc.collect()
    return n


def write_all_rollout_scenes(stride: int = 1, batch: int = 800) -> int:
    """scene3d layers for every rollout that doesn't have one yet, in subprocess batches (one process can only create
    a few thousand RecordingStreams before running out of threads)."""
    import subprocess
    import sys
    from cfnbc import hub
    done = {p.stem for p in (S.rerun_root() / "rollout_scene3d").glob("*.rrd")}
    with hub.open_catalog() as client:
        segs = client.get_dataset(S.DATASET_ROLLOUTS).segment_table().to_pandas().rerun_segment_id
    todo = sorted(set(segs) - done)
    print(f"{len(todo)} rollouts without a scene3d layer ({len(done)} already written)", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        subprocess.run([sys.executable, "-m", "cfnbc.scene3d", "--stride", str(stride), "--rollouts", *chunk], check=True)
        print(f"  {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rollouts", nargs="*", default=None,
                        help="write rollout scene layers instead (no ids: dropped handoffs + clean failures)")
    parser.add_argument("--all_rollouts", action="store_true", help="write the layer for every rollout still missing it")
    args = parser.parse_args()
    if args.all_rollouts:
        print(f"rollout scene3d layers written: {write_all_rollout_scenes(args.stride)}")
        return
    if args.rollouts is not None:
        n = write_rollout_scenes(args.rollouts or None, args.stride)
        print(f"rollout scene3d layers: {n}")
        return
    for e in args.episodes:
        p = write_scene(e, args.condition, args.stride)
        print(f"{p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
