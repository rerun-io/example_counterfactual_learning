"""Geometric task-preservation checks for nuisance conditions that add geometry (distractor, support, combos).

The paper assumes each nuisance preserves the task and the expert action, and concedes not every distractor or
support change does. Our distractors/patch have no collision geometry, so physics is unchanged by construction
(verified bit-identical). What can still break the assumption is *visual*: the arm visibly passing through a
distractor, or nuisance geometry hiding the target from the policy camera. Per (demo episode, condition):

  min_clearance_m      min signed distance over the demo between gripper-region vertices (lower forearm .. fingers,
                       both arms; compiled mesh verts) and any distractor box. <0 means visual interpenetration.
  penetration_steps    steps with min_clearance < 0
  cube_distractor_m    min signed distance cube corners -> distractor boxes over the demo (target/distractor overlap)
  occl_nuisance_max    max over steps of the fraction of the cube's camera-facing surface samples whose ray to the
                       top camera hits a nuisance geom first (mj_ray, hidden geoms excluded)
  occl_nuisance_mean   mean of the same over steps
  occl_robot_mean      same for robot geoms (context: occlusion that also exists in the clean scene)
  support_cube_overlap fraction of the cube's initial footprint covered by the support patch
  clearance_step / cube_distractor_step   step of the respective minimum (for review deep links)

    python -m cfnbc.task_preservation [--episodes ...] [--conditions ...] [--stride 2] [--scene_layers]

Writes results/task_preservation/checks.parquet (reference) and table `task_preservation` in the catalog.
--scene_layers writes a scene3d_<condition> layer (nuisance geometry + per-step check scalars) for every flagged
(episode, condition) and for episode 0 of every condition.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pandas as pd
import rerun as rr

from cfnbc import nuisance as N
from cfnbc import schema as S
from cfnbc.scene3d import load_physics, write_scene

ROOT = Path(__file__).resolve().parents[2]
GRIPPER_PARTS = ("5_lower_forearm", "6_wrist", "7_gripper", "8_gripper_prop", "9_gripper_bar",
                 "10_left_gripper_finger", "10_right_gripper_finger")
GEOM_CONDITIONS = [c for c, sp in N.CONDITIONS.items() if sp.distractor is not None or sp.support is not None]
OCCL_FLAG = 0.10  # flag if >10% of the cube's visible surface is hidden by nuisance geometry at some step


def point_box_sdist(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance from points (N,3) to an axis-aligned box. Nuisance boxes are unrotated."""
    d = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(d, 0), axis=-1)
    inside = np.minimum(d.max(axis=-1), 0)
    return outside + inside


class Checker:
    def __init__(self, vert_stride: int = 3, face_samples: int = 4):
        self.physics = load_physics()
        m = self.physics.model
        self.m = m
        self.gripper_geoms = [m.name2id(f"vx300s_{side}/{part}", "geom") for side in ("left", "right") for part in GRIPPER_PARTS]
        self.local_verts = []
        for g in self.gripper_geoms:
            mid = int(m.geom_dataid[g])
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            self.local_verts.append(m.mesh_vert[v0:v0 + nv:vert_stride].copy())
        self.cube = m.name2id("red_box", "geom")
        self.cam = m.name2id("top", "camera")
        self.distractors = [m.name2id(f"nuis_distractor_{k}", "geom") for k in range(4)]
        self.support = m.name2id("nuis_support", "geom")
        self.nuis_geoms = set(self.distractors + [self.support])
        self.robot_bodies_start = m.name2id("vx300s_left", "body")
        # cube surface samples in the cube frame: face_samples^2 points per face, with outward normals
        u = (np.arange(face_samples) + 0.5) / face_samples * 2 - 1
        uu, vv = np.meshgrid(u, u)
        pts, normals = [], []
        for axis in range(3):
            for sign in (-1, 1):
                p = np.zeros((uu.size, 3))
                o = [a for a in range(3) if a != axis]
                p[:, axis] = sign
                p[:, o[0]], p[:, o[1]] = uu.ravel(), vv.ravel()
                n = np.zeros(3)
                n[axis] = sign
                pts.append(p)
                normals.append(np.repeat(n[None], len(p), 0))
        self.cube_pts_unit = np.concatenate(pts) * 0.999  # just inside the surface so rays leave the cube cleanly
        self.cube_normals = np.concatenate(normals)
        self.geomgroup = np.array([1, 1, 1, 1, 0, 0], np.uint8)  # exclude hidden group 4
        self._gid = np.zeros(1, np.int32)

    def _visible_boxes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        m, d = self.m, self.physics.data
        return [(d.geom_xpos[g].copy(), m.geom_size[g].copy()) for g in self.distractors if m.geom_group[g] != N.HIDDEN_GROUP]

    def occlusion(self) -> tuple[float, float]:
        """Fractions of the cube's camera-facing samples occluded by nuisance / robot geoms at the current state."""
        m, d = self.m, self.physics.data
        R = d.geom_xmat[self.cube].reshape(3, 3)
        c = d.geom_xpos[self.cube]
        half = m.geom_size[self.cube]
        cam = d.cam_xpos[self.cam]
        pts = c + (self.cube_pts_unit * half) @ R.T
        nrm = self.cube_normals @ R.T
        facing = ((cam - pts) * nrm).sum(1) > 0
        n_nuis = n_robot = 0
        sel = np.nonzero(facing)[0]
        for i in sel:
            vec = pts[i] - cam
            dist_pt = np.linalg.norm(vec)
            hit = mujoco.mj_ray(m.ptr, d.ptr, cam, vec / dist_pt, self.geomgroup, 1, -1, self._gid)
            g = int(self._gid[0])
            if hit < 0 or g == self.cube or hit >= dist_pt - 1e-4:
                continue
            if g in self.nuis_geoms:
                n_nuis += 1
            elif m.geom_bodyid[g] >= self.robot_bodies_start and g != self.cube:
                n_robot += 1
        n = max(len(sel), 1)
        return n_nuis / n, n_robot / n

    def check(self, e: int, condition: str, stride: int = 2) -> tuple[dict, dict[str, np.ndarray]]:
        with h5py.File(S.sim_data_dir() / f"episode_{e}.hdf5") as f:
            sim_q = f["sim_state/qpos"][()]
        p, m = self.physics, self.m
        N.apply(p, condition, sim_q[0, 16:23])
        boxes = self._visible_boxes()
        steps = np.arange(0, len(sim_q), stride)
        clearance = np.full(len(steps), np.inf)
        cube_dist = np.full(len(steps), np.inf)
        occl_n = np.zeros(len(steps))
        occl_r = np.zeros(len(steps))
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        for k, t in enumerate(steps):
            p.data.qpos[:] = sim_q[t]
            p.forward()
            if boxes:
                world = np.concatenate([p.data.geom_xpos[g] + v @ p.data.geom_xmat[g].reshape(3, 3).T
                                        for g, v in zip(self.gripper_geoms, self.local_verts)])
                clearance[k] = min(point_box_sdist(world, c, h).min() for c, h in boxes)
                R = p.data.geom_xmat[self.cube].reshape(3, 3)
                cc = p.data.geom_xpos[self.cube] + (corners * m.geom_size[self.cube]) @ R.T
                cube_dist[k] = min(point_box_sdist(cc, c, h).min() for c, h in boxes)
            occl_n[k], occl_r[k] = self.occlusion()
        sup = N.CONDITIONS[condition].support
        overlap = 0.0
        if sup is not None:
            cs, hs = sim_q[0, 16:18], m.geom_size[self.cube][:2]
            ps, hp = p.data.geom_xpos[self.support][:2], m.geom_size[self.support][:2]
            inter = np.clip(np.minimum(cs + hs, ps + hp) - np.maximum(cs - hs, ps - hp), 0, None)
            overlap = float(inter.prod() / (4 * hs.prod()))
        row = dict(episode=e, segment_id=S.demo_segment_id(e), condition=condition, family=N.FAMILY_OF[condition],
                   min_clearance_m=float(clearance.min()) if boxes else np.nan,
                   penetration_steps=int(((clearance < 0).sum()) * stride) if boxes else 0,
                   cube_distractor_m=float(cube_dist.min()) if boxes else np.nan,
                   clearance_step=int(steps[np.argmin(clearance)]) if boxes else -1,
                   cube_distractor_step=int(steps[np.argmin(cube_dist)]) if boxes else -1,
                   occl_nuisance_max=float(occl_n.max()), occl_nuisance_mean=float(occl_n.mean()),
                   occl_robot_mean=float(occl_r.mean()), support_cube_overlap=overlap)
        row["flag"] = bool(row["penetration_steps"] > 0 or row["occl_nuisance_max"] > OCCL_FLAG
                           or (boxes and row["cube_distractor_m"] < 0))
        series = dict(steps=steps, min_clearance=np.where(np.isfinite(clearance), clearance, np.nan),
                      occl_nuisance=occl_n, occl_robot=occl_r)
        return row, series


def log_checks(series: dict[str, np.ndarray]):
    def _log(rec):
        steps = series["steps"]
        idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]
        for name in ("min_clearance", "occl_nuisance", "occl_robot"):
            rr.send_columns(f"/checks/{name}", indexes=idx, recording=rec,
                            columns=rr.Scalars.columns(scalars=series[name]))
    return _log


def table_rows(df: pd.DataFrame):
    import pyarrow as pa
    return pa.Table.from_pandas(df.drop(columns=["flag"]).assign(flag=df.flag.astype(bool)), preserve_index=False)


def write_table(df: pd.DataFrame) -> None:
    from cfnbc import hub
    tbl = table_rows(df)
    with hub.open_catalog() as client:
        if "task_preservation" in client.table_names():
            t = client.get_table("task_preservation")
            t.overwrite(tbl.to_batches())
        else:
            t = hub.ensure_table(client, "task_preservation", tbl.schema)
            t.append(tbl.to_batches())
        print(f"table task_preservation: {len(df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--scene_layers", action="store_true")
    parser.add_argument("--no_table", action="store_true")
    args = parser.parse_args()
    episodes = args.episodes if args.episodes is not None else list(range(50))
    conditions = args.conditions or GEOM_CONDITIONS
    chk = Checker()
    rows, t0 = [], time.time()
    for e in episodes:
        for c in conditions:
            row, series = chk.check(e, c, args.stride)
            rows.append(row)
            if args.scene_layers and (row["flag"] or e == episodes[0]):
                write_scene(e, c, args.stride, extra=log_checks(series))
        print(f"episode {e}: {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "task_preservation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "checks.parquet")
    summ = df.groupby("condition").agg(min_clearance_cm=("min_clearance_m", lambda s: 100 * s.min()),
                                       episodes_penetrating=("penetration_steps", lambda s: int((s > 0).sum())),
                                       cube_distractor_cm=("cube_distractor_m", lambda s: 100 * s.min()),
                                       occl_nuisance_max=("occl_nuisance_max", "max"),
                                       occl_robot_mean=("occl_robot_mean", "mean"),
                                       support_overlap=("support_cube_overlap", "mean"),
                                       flagged=("flag", "sum"))
    summ.round(3).to_csv(out / "summary.csv")
    with pd.option_context("display.width", 200):
        print(summ.round(3))
    if not args.no_table:
        write_table(df)


if __name__ == "__main__":
    main()
