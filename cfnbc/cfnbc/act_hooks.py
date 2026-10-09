"""Optional Rerun logging hooks called from act/imitate_episodes.py.

Disabled (no-op) unless env ACT_RERUN is set to 1. Output goes to <rerun_root>/{training,rollouts}/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rerun as rr

from cfnbc import schema as S


def enabled() -> bool:
    return os.environ.get("ACT_RERUN", "0") == "1"


def _run_name(config) -> str:
    return os.environ.get("ACT_RUN_NAME") or Path(config["ckpt_dir"]).resolve().name


def _scalar_dict(summary) -> dict[str, float]:
    return {k: float(v.item() if hasattr(v, "item") else v) for k, v in summary.items()}


class _NoOp:
    def __getattr__(self, name):
        return lambda *a, **k: None


class TrainLogger:
    """One recording per training run. Timelines: `epoch` (train/val epoch means), `step` (per-batch train)."""

    def __init__(self, config, extra_props: dict | None = None):
        run = _run_name(config)
        out_dir = S.rerun_root() / "training"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.rec = rr.RecordingStream(S.APP_ID, recording_id=f"train__{run}")
        self.rec.save(out_dir / f"train__{run}.rrd")
        pc = config["policy_config"]
        props = dict(run=run, seed=int(config["seed"]), num_epochs=int(config["num_epochs"]), lr=float(config["lr"]),
                     policy_class=str(config["policy_class"]), task_name=str(config["task_name"]),
                     device=str(config["device"]), num_workers=int(os.environ.get("ACT_NUM_WORKERS", 1)),
                     camera_names=",".join(config["camera_names"]),
                     **{k: v for k, v in pc.items() if isinstance(v, (int, float, str)) and k != "lr"})
        props.update(extra_props or {})
        rr.send_property("train", rr.AnyValues(**props), recording=self.rec)
        self.step = 0

    def log_epoch(self, epoch: int, split: str, summary) -> None:
        rr.set_time("epoch", sequence=epoch, recording=self.rec)
        for k, v in _scalar_dict(summary).items():
            rr.log(f"/{split}/{k}", rr.Scalars(v), recording=self.rec)

    def log_batch(self, summary) -> None:
        rr.set_time("step", sequence=self.step, recording=self.rec)
        for k, v in _scalar_dict(summary).items():
            rr.log(f"/train_batch/{k}", rr.Scalars(v), recording=self.rec)
        self.step += 1

    def log_best(self, epoch: int, val_loss: float) -> None:
        rr.send_property("result", rr.AnyValues(best_epoch=int(epoch), min_val_loss=float(val_loss)),
                         recording=self.rec)

    def close(self) -> None:
        self.rec.flush()
        self.rec.disconnect()


class RolloutLogger:
    """One recording (segment) per eval rollout, same entity layout as the demos plus policy outputs."""

    def __init__(self, config, ckpt_name: str, nuisance: str = "clean"):
        self.run = _run_name(config)
        self.config, self.ckpt_name, self.nuisance = config, ckpt_name, nuisance
        self.out_dir = S.rerun_root() / "rollouts"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rec = None

    def start(self, rollout_id: int, init_pose: np.ndarray) -> None:
        c = self.config
        self.rollout_id, self.init_pose = rollout_id, np.asarray(init_pose, dtype=np.float64)
        seg = S.rollout_segment_id(self.run, self.ckpt_name, c["temporal_agg"], c["eval_seed"], rollout_id,
                                   self.nuisance)
        self.rec = rr.RecordingStream(S.APP_ID, recording_id=seg)
        self.rec.save(self.out_dir / f"{seg}.rrd")
        S.log_static_series_names(self.rec)
        self.video = None
        self.rows = {S.QPOS: [], S.ACTION: [], S.SIM_QPOS: [], S.SIM_QVEL: [], S.REWARD: []}

    def step(self, t: int, obs, physics, action: np.ndarray, chunk, reward) -> None:
        img = obs["images"]["top"]
        if self.video is None:
            self.video = S.H264StreamLogger(self.rec, S.CAMERA_TOP, img.shape[1], img.shape[0], fps=int(round(1 / S.DT)))
        self.video.log_frame(img)
        self.rows[S.QPOS].append(np.asarray(obs["qpos"], dtype=np.float64))
        self.rows[S.ACTION].append(np.asarray(action, dtype=np.float64))
        self.rows[S.SIM_QPOS].append(physics.data.qpos.copy())
        self.rows[S.SIM_QVEL].append(physics.data.qvel.copy())
        self.rows[S.REWARD].append(float(reward) if reward is not None else np.nan)
        if chunk is not None:  # (chunk_size, 14) denormalized prediction made at this step
            rr.set_time(S.TIMELINE_STEP, sequence=t, recording=self.rec)
            rr.set_time(S.TIMELINE_TIME, duration=t * S.DT, recording=self.rec)
            rr.log(S.POLICY_CHUNK, rr.Tensor(np.asarray(chunk, dtype=np.float32), dim_names=["k", "joint"]),
                   recording=self.rec)

    def end(self, highest_reward, success: bool) -> None:
        n = len(self.rows[S.QPOS])
        steps = np.arange(n)
        idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]
        for path, rows in self.rows.items():
            rr.send_columns(path, indexes=idx, columns=rr.Scalars.columns(scalars=np.asarray(rows)), recording=self.rec)
        sim_qpos = np.asarray(self.rows[S.SIM_QPOS])
        rr.send_columns(S.CUBE, indexes=idx, recording=self.rec,
                        columns=rr.Transform3D.columns(translation=sim_qpos[:, 16:19],
                                                       quaternion=sim_qpos[:, [20, 21, 22, 19]]))
        if self.video is not None:
            self.video.close()
        c = self.config
        rr.send_property("rollout", rr.AnyValues(
            run=self.run, ckpt_name=self.ckpt_name, temporal_agg=int(bool(c["temporal_agg"])),
            eval_seed=int(c["eval_seed"]), rollout_id=int(self.rollout_id), nuisance=self.nuisance,
            init_pose=self.init_pose, highest_reward=int(highest_reward), success=int(bool(success)),
            episode_return=float(np.nansum(self.rows[S.REWARD])),
        ), recording=self.rec)
        self.rec.flush()
        self.rec.disconnect()
        self.rec = None


def make_train_logger(config):
    return TrainLogger(config) if enabled() else _NoOp()


def make_rollout_logger(config, ckpt_name: str):
    nuisance = os.environ.get("ACT_NUISANCE") or "clean"
    return RolloutLogger(config, ckpt_name, nuisance=nuisance) if enabled() else _NoOp()
