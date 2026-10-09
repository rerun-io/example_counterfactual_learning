"""Per-demo QA metrics, written as a `qa` layer on cube_demos plus a named dataset definition table.

    python -m cfnbc.qa [--episodes ...] [--define A50 --version 1]
    python -m cfnbc.qa --rollouts          # stable-stage QA layer for every cube_rollouts segment

Metrics are computed from the HDF5 (actions, images) and the recorded sim_state (contacts via mj_forward,
reward stages via upstream's TransferCubeTask.get_reward), with no rendering and no rollout:
  per step: reward_stage (0-4, upstream raw), stable_stage (flicker-filtered, see stable_stages), action_jerk (L2 of 3rd difference over arm joints), cube_height,
            joint_limit_margin (min distance of any arm joint to its range limit, rad)
  summary (segment properties): first step of each reward stage, handoff duration, max jerk, gripper chatter
            (gripper-command direction reversals per arm), cube max height, min joint-limit margin,
            blank frames, duplicate consecutive frames, success
Writes <rerun_root>/qa/demo_eNNN.rrd and results/qa/demo_qa.csv (reference copy).
--define writes rows (name, version, segment_id, sha256 of each HDF5 episode) to the
`dataset_defs` table.
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rerun as rr

from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "act"
DATA = S.sim_data_dir()
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # 14-d action minus the two gripper dims
GRIPPERS = (6, 13)


def _task_and_physics():
    if str(ACT_DIR) not in sys.path:
        sys.path.insert(0, str(ACT_DIR))
    from dm_control import mujoco
    import sim_env
    physics = mujoco.Physics.from_xml_path(str(ACT_DIR / "assets" / "bimanual_viperx_transfer_cube.xml"))
    return sim_env.TransferCubeTask(random=False), physics


def reward_stages(task, physics, sim_qpos: np.ndarray) -> np.ndarray:
    out = np.empty(len(sim_qpos), dtype=np.int64)
    for t, q in enumerate(sim_qpos):
        physics.data.qpos[:] = q
        physics.forward()
        out[t] = task.get_reward(physics)
    return out


CUBE_REST_Z = 0.02     # cube half-size: centre height when resting on the table
STABLE_HOLD = 5        # steps a stage must persist
STABLE_RISE = 0.005    # m above resting height required for the "not touching table" stages (2, 4)
STRICT_FINAL_Z = 0.03  # m; cube centre above this at the last step = still held (resting height is 0.02)


def stable_stages(raw: np.ndarray, cube_z: np.ndarray, hold: int = STABLE_HOLD, rise: float = STABLE_RISE) -> np.ndarray:
    """Upstream's per-step reward stage with contact flicker removed.

    Stage s counts at step t only if the raw stage was >= s for the last `hold` steps, and, for the lift stages
    (2: right gripper holds cube off the table, 4: left gripper holds it off the table), the cube is at least
    `rise` above its resting height. Otherwise 2 -> 1 and 4 -> 3.
    """
    raw = np.asarray(raw, dtype=np.int64)
    held = np.array([raw[max(0, t - hold + 1):t + 1].min() if t >= hold - 1 else 0 for t in range(len(raw))])
    lifted = (np.asarray(cube_z) - CUBE_REST_Z) >= rise
    out = held.copy()
    out[(held == 2) & ~lifted] = 1
    out[(held == 4) & ~lifted] = 3
    return out


def first_at_least(stages: np.ndarray, stage: int) -> int:
    hit = np.nonzero(stages >= stage)[0]
    return int(hit[0]) if len(hit) else -1


def joint_limit_margin(physics, sim_qpos: np.ndarray) -> np.ndarray:
    m = physics.model
    idx = [m.name2id(f"vx300s_{side}/{j}", "joint") for side in ("left", "right")
           for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
    lo, hi = m.jnt_range[idx, 0], m.jnt_range[idx, 1]
    q = sim_qpos[:, [m.jnt_qposadr[i] for i in idx]]
    return np.minimum(q - lo, hi - q).min(axis=1)


def episode_metrics(e: int, task, physics) -> tuple[dict, dict[str, np.ndarray]]:
    with h5py.File(DATA / f"episode_{e}.hdf5") as f:
        action = f["action"][()].astype(np.float64)
        sim_q = f["sim_state/qpos"][()]
        imgs = f["observations/images/top"]
        frame_std = np.array([imgs[t].std() for t in range(len(imgs))])
        dup = sum(np.array_equal(imgs[t], imgs[t - 1]) for t in range(1, len(imgs)))
        success = int(f.attrs.get("success", -1))
    stages = reward_stages(task, physics, sim_q)
    stable = stable_stages(stages, sim_q[:, 18])
    jerk = np.zeros(len(action))
    jerk[3:] = np.linalg.norm(np.diff(action[:, ARM], n=3, axis=0), axis=1)
    cube_h = sim_q[:, 18]
    margin = joint_limit_margin(physics, sim_q)

    def first(stage):
        hit = np.nonzero(stages >= stage)[0]
        return int(hit[0]) if len(hit) else -1

    chatter = {}
    for name, d in zip(("left", "right"), GRIPPERS):
        v = np.sign(np.diff(action[:, d]))
        v = v[v != 0]
        chatter[name] = int((v[1:] != v[:-1]).sum())
    summary = dict(episode=e, segment_id=S.demo_segment_id(e), success=success,
                   first_touch=first(1), first_lift=first(2), first_transfer_attempt=first(3),
                   first_transfer=first(4), max_stage=int(stages.max()),
                   handoff_duration=(first(4) - first(3)) if first(4) >= 0 and first(3) >= 0 else -1,
                   max_jerk=float(jerk.max()), mean_jerk=float(jerk[3:].mean()),
                   gripper_chatter_left=chatter["left"], gripper_chatter_right=chatter["right"],
                   stable_first_touch=first_at_least(stable, 1), stable_first_lift=first_at_least(stable, 2),
                   stable_first_transfer=first_at_least(stable, 4), stable_max_stage=int(stable.max()),
                   cube_max_height=float(cube_h.max()), min_joint_limit_margin=float(margin.min()),
                   final_cube_z=float(cube_h[-1]),
                   success_strict=int(stable.max() == 4 and cube_h[-1] > STRICT_FINAL_Z),
                   lift_gap=(first_at_least(stable, 2) - first(2)) if first(2) >= 0 and first_at_least(stable, 2) >= 0 else -1,
                   blank_frames=int((frame_std < 1.0).sum()), duplicate_frames=int(dup))
    series = dict(reward_stage=stages.astype(np.float64), stable_stage=stable.astype(np.float64),
                  action_jerk=jerk, cube_height=cube_h,
                  joint_limit_margin=margin)
    return summary, series


def write_layer(summary: dict, series: dict[str, np.ndarray], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = summary["segment_id"]
    path = out_dir / f"{seg}.rrd"
    rec = rr.RecordingStream(S.APP_ID, recording_id=seg)
    rec.save(path)
    rr.send_property("qa", rr.AnyValues(**{k: v for k, v in summary.items() if k != "segment_id"}), recording=rec)
    n = len(series["reward_stage"])
    steps = np.arange(n)
    idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]
    for name, values in series.items():
        rr.send_columns(f"/qa/{name}", indexes=idx, columns=rr.Scalars.columns(scalars=values), recording=rec)
    rec.flush()
    rec.disconnect()
    return path


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def dataset_def_rows(name: str, version: int, episodes: list[int]):
    """pyarrow Table of `dataset_defs` rows for a named dataset definition."""
    import pyarrow as pa
    schema = pa.schema([("name", pa.string()), ("version", pa.int32()), ("segment_id", pa.string()),
                        ("source", pa.string()), ("sha256", pa.string()), ("created_at", pa.string())])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = dict(name=[], version=[], segment_id=[], source=[], sha256=[], created_at=[])
    for e in episodes:
        src = f"data/sim_transfer_cube_scripted/episode_{e}.hdf5"
        rows["name"].append(name); rows["version"].append(version); rows["segment_id"].append(S.demo_segment_id(e))
        rows["source"].append(src); rows["sha256"].append(file_sha256(DATA / Path(src).name)); rows["created_at"].append(now)
    return pa.table(rows, schema=schema)


def define_dataset(name: str, version: int, episodes: list[int]) -> None:
    from cfnbc import hub
    tbl = dataset_def_rows(name, version, episodes)
    with hub.open_catalog() as client:
        t = hub.ensure_table(client, "dataset_defs", tbl.schema)
        existing = t.reader().to_pandas()
        if ((existing["name"] == name) & (existing["version"] == version)).any():
            print(f"{name} v{version} already defined (table dataset_defs); keeping it")
            return
        t.append(tbl.to_batches())
        print(f"defined {name} v{version}: {len(episodes)} segments (table {t.name})")


def rollout_qa(segments: list[str] | None = None, force: bool = False, run_prefix: str | None = None) -> pd.DataFrame:
    """Stable-stage QA for eval rollouts, from the catalog (`/reward` = upstream raw stage, `/sim/qpos`).

    Writes a `qa` layer per rollout segment (<rerun_root>/rollout_qa/<seg>.rrd: /qa/stable_stage + properties
    raw_max_stage, stable_max_stage, first stable touch/lift/transfer) and results/qa/rollout_qa.csv.
    """
    from cfnbc import hub
    with hub.open_catalog() as client:
        ds = client.get_dataset(S.DATASET_ROLLOUTS)
        if segments:
            ds = ds.filter_segments(segments)
        t = ds.filter_contents([S.REWARD, S.SIM_QPOS]).reader(index=S.TIMELINE_STEP).to_arrow_table()
    df = pd.DataFrame({"seg": t["rerun_segment_id"].to_pylist(), "step": t[S.TIMELINE_STEP].to_pylist(),
                       "reward": [r[0] if r else np.nan for r in t[f"{S.REWARD}:Scalars:scalars"].to_pylist()],
                       "cube_z": [q[18] if q else np.nan for q in t[f"{S.SIM_QPOS}:Scalars:scalars"].to_pylist()]})
    if run_prefix:  # process one run per call: thousands of RecordingStreams in one process exhaust OS threads
        df = df[df.seg.str.startswith(run_prefix)]
    out_dir = S.rerun_root() / "rollout_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, written = [], 0
    for seg, g in df.sort_values("step").groupby("seg"):
        g = g.dropna()
        raw, z, steps = g.reward.to_numpy().astype(np.int64), g.cube_z.to_numpy(), g.step.to_numpy()
        stable = stable_stages(raw, z)
        row = dict(segment_id=seg, raw_max_stage=int(raw.max()), stable_max_stage=int(stable.max()),
                   raw_first_lift=first_at_least(raw, 2), stable_first_touch=first_at_least(stable, 1),
                   stable_first_lift=first_at_least(stable, 2), stable_first_transfer=first_at_least(stable, 4),
                   cube_max_height=float(z[steps >= 20].max()),  # skip the spawn drop (cube starts at z=0.05)
                   final_cube_z=float(z[-1]),
                   # strict success: transferred (stable stage 4) AND still held at the end (demos keep the cube
                   # at ~21.6 cm through step 399). Upstream counts any single step at stage 4.
                   success_strict=int(stable.max() == 4 and z[-1] > STRICT_FINAL_Z))
        rows.append(row)
        dst, src = out_dir / f"{seg}.rrd", S.rerun_root() / "rollouts" / f"{seg}.rrd"
        if not force and dst.exists() and (not src.exists() or dst.stat().st_mtime >= src.stat().st_mtime):
            continue  # layer is up to date: only (re)write QA for new or re-run rollouts
        rec = rr.RecordingStream(S.APP_ID, recording_id=seg)
        rec.save(dst)
        rr.send_property("qa", rr.AnyValues(**{k: v for k, v in row.items() if k != "segment_id"}), recording=rec)
        idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]
        rr.send_columns("/qa/stable_stage", indexes=idx, columns=rr.Scalars.columns(scalars=stable.astype(np.float64)),
                        recording=rec)
        rr.send_columns("/qa/cube_height", indexes=idx, columns=rr.Scalars.columns(scalars=z), recording=rec)
        rec.flush()
        rec.disconnect()
        del rec  # each RecordingStream owns background threads; release them now (thousands of rollouts -> EAGAIN)
        written += 1
        if written % 200 == 0:
            gc.collect()
    print(f"rollout qa: {len(rows)} rollouts, {written} layer files (re)written")
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "qa"
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "rollout_qa.csv"
    if run_prefix and csv.exists():  # merge this run's rows into the full table
        old = pd.read_csv(csv)
        res = pd.concat([old[~old.segment_id.str.startswith(run_prefix)], res], ignore_index=True)
    res.to_csv(csv, index=False)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--define", default=None, help="also write a named dataset definition, e.g. A50")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--rollouts", action="store_true", help="stable-stage QA for cube_rollouts instead of demos")
    parser.add_argument("--force", action="store_true", help="rewrite every rollout qa layer (e.g. after adding a series)")
    parser.add_argument("--run_prefix", default=None, help="only rollouts whose segment id starts with this (one run)")
    args = parser.parse_args()
    if args.rollouts:
        if args.force and not args.run_prefix:  # one subprocess per run keeps the thread count bounded
            import subprocess
            from cfnbc import hub as _hub
            with _hub.open_catalog() as c:
                runs = sorted({s.split("__")[0] for s in c.get_dataset(S.DATASET_ROLLOUTS).segment_ids()})
            for run in runs:
                subprocess.run([sys.executable, "-m", "cfnbc.qa", "--rollouts", "--force", "--run_prefix", f"{run}__"],
                               check=True)
            res = pd.read_csv(ROOT / "results" / "qa" / "rollout_qa.csv")
        else:
            res = rollout_qa(force=args.force, run_prefix=args.run_prefix)
        print(pd.crosstab(res.raw_max_stage, res.stable_max_stage, rownames=["raw max stage"],
                          colnames=["stable max stage"]))
        print(f"strict successes: {int(res.success_strict.sum())} of {int((res.raw_max_stage == 4).sum())} upstream successes")
        return
    episodes = args.episodes if args.episodes is not None else list(range(50))
    task, physics = _task_and_physics()
    rows = []
    for e in episodes:
        summary, series = episode_metrics(e, task, physics)
        write_layer(summary, series, S.rerun_root() / "qa")
        rows.append(summary)
        print(f"{summary['segment_id']}: stage {summary['max_stage']} transfer@{summary['first_transfer']} "
              f"jerk {summary['max_jerk']:.3f} chatter {summary['gripper_chatter_left']}/{summary['gripper_chatter_right']}",
              flush=True)
    out = ROOT / "results" / "qa"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "demo_qa.csv", index=False)
    print(df.describe().T[["mean", "min", "max"]].round(3).to_string())
    if args.define:
        define_dataset(args.define, args.version, episodes)


if __name__ == "__main__":
    main()
