"""Shared Rerun layout for demos, eval rollouts and training runs.

Demos and rollouts use the same entity paths, so the same dataframe query works on both datasets.
Timestep convention (matches the ACT HDF5): row `step=t` holds the observation at t and the action
commanded at t. In rollouts, `/reward` at t is the reward returned by `env.step(action_t)`.
"""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import rerun as rr

DT = 0.02  # constants.DT
APP_ID = "cfnbc_cube_transfer"

DATASET_DEMOS = "cube_demos"
DATASET_ROLLOUTS = "cube_rollouts"
DATASET_TRAINING = "cube_training"

# entity paths
CAMERA_TOP = "/camera/top"
QPOS = "/robot/qpos"           # 14-d observation qpos (normalized grippers)
QVEL = "/robot/qvel"
ACTION = "/robot/action"       # 14-d commanded joint action
SIM_QPOS = "/sim/qpos"         # full MuJoCo qpos (23), cube free joint at [16:23]
SIM_QVEL = "/sim/qvel"         # full MuJoCo qvel (22)
CUBE = "/world/cube"
POLICY_CHUNK = "/policy/chunk" # rollouts: (chunk_size, 14) predicted action chunk when the policy was queried
REWARD = "/reward"

TIMELINE_STEP = "step"
TIMELINE_TIME = "sim_time"

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]
JOINT_NAMES_14 = [f"left_{j}" for j in ARM_JOINTS] + [f"right_{j}" for j in ARM_JOINTS]
# fixed per-joint colours (tab20 pairs: one hue per joint, dark = left arm, light = right arm), so the same joint has
# the same colour in every plot and every view
_TAB20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138),
          (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
          (227, 119, 194), (247, 182, 210)]
JOINT_COLORS_14 = _TAB20[0::2] + _TAB20[1::2]
CUBE_HALF_SIZE = 0.02  # bimanual_viperx_transfer_cube.xml red_box size


def sim_data_dir() -> Path:
    """ACT's scripted cube-transfer demos (HDF5): $ACT_DATA_DIR/sim_transfer_cube_scripted, default <repo>/data/..."""
    return Path(os.environ.get("ACT_DATA_DIR", Path(__file__).resolve().parents[2] / "data")) / "sim_transfer_cube_scripted"


def real_data_dir() -> Path:
    """Real LeRobot dataset and its ACT-format export: $CFNBC_REAL_DATA_DIR, default <repo>/data_real."""
    return Path(os.environ.get("CFNBC_REAL_DATA_DIR", Path(__file__).resolve().parents[2] / "data_real"))


def rerun_root() -> Path:
    """Root dir for .rrd files: $ACT_RERUN_DIR, else <workspace>/rerun."""
    return Path(os.environ.get("ACT_RERUN_DIR", Path(__file__).resolve().parents[2] / "rerun"))


def demo_segment_id(episode_idx: int) -> str:
    return f"demo_e{episode_idx:03d}"


def rollout_segment_id(run: str, ckpt_name: str, temporal_agg: bool, eval_seed: int, rollout_id: int,
                       nuisance: str = "clean") -> str:
    ckpt = ckpt_name.removesuffix(".ckpt").removeprefix("policy_")
    tagg = "tagg" if temporal_agg else "notagg"
    return f"{run}__{ckpt}__{tagg}__es{eval_seed}__{nuisance}__r{rollout_id:03d}"


def log_static_series_names(rec: rr.RecordingStream) -> None:
    for path in (QPOS, QVEL, ACTION):
        rr.log(path, rr.SeriesLines(names=JOINT_NAMES_14, colors=JOINT_COLORS_14), static=True, recording=rec)
    rr.log(CUBE, rr.Boxes3D(half_sizes=[CUBE_HALF_SIZE] * 3, colors=[220, 30, 30]), static=True, recording=rec)


def cube_transform(sim_qpos: np.ndarray) -> rr.Transform3D:
    pos, wxyz = sim_qpos[16:19], sim_qpos[19:23]
    return rr.Transform3D(translation=pos, quaternion=rr.Quaternion(xyzw=[wxyz[1], wxyz[2], wxyz[3], wxyz[0]]))


class H264StreamLogger:
    """Encode frames to H.264 (Annex B, no B-frames) and log them as rr.VideoStream samples."""

    def __init__(self, rec: rr.RecordingStream, entity_path: str, width: int, height: int, fps: int = 50,
                 crf: int = 18):
        self.rec, self.entity_path = rec, entity_path
        self.codec = av.CodecContext.create("libx264", "w")
        self.codec.width, self.codec.height = width, height
        self.codec.pix_fmt = "yuv420p"
        self.codec.time_base = Fraction(1, fps)
        self.codec.options = {"tune": "zerolatency", "preset": "veryfast", "crf": str(crf), "bf": "0"}
        rr.log(entity_path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True, recording=rec)
        self.pts = 0

    def log_frame(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.pts
        self.pts += 1
        self._send(self.codec.encode(frame))

    def close(self) -> None:
        self._send(self.codec.encode(None))  # flush

    def _send(self, packets) -> None:
        for p in packets:
            if p.pts is None:
                continue
            # zerolatency + no B-frames: packet pts == frame index == step
            rr.set_time(TIMELINE_STEP, sequence=int(p.pts), recording=self.rec)
            rr.set_time(TIMELINE_TIME, duration=int(p.pts) * DT, recording=self.rec)
            rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(p), is_keyframe=p.is_keyframe),
                   recording=self.rec)
