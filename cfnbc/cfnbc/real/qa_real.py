"""QA for human teleop demos (what scripted sim demos never need), written as a `qa` layer on real_demos.

    python -m cfnbc.real.qa_real

Per step (from the parquet state/action at 50 Hz): arm joint speed (rad/s, follower state), idle flag (speed below
threshold), leader-follower tracking error (|action_t - state_{t+lag}|), action jerk.
Per episode (segment properties): active start/end (first/last step with motion), idle fraction, longest pause inside
the active window, leader->follower lag (steps, best alignment), mean tracking error, max jerk, gripper reversals.
Also writes results/real/qa_real.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rerun as rr

from cfnbc import schema as S
from cfnbc.real.ingest import APP_ID_REAL, ROOT, segment_id

SRC = S.real_data_dir() / "aloha_static_cups_open"
FPS = 50
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = (6, 13)
IDLE_RAD_S = 0.25  # above the encoder-quantization floor (0.00153 rad/tick -> ~0.08 rad/s jitter)


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def episode_qa(state: np.ndarray, action: np.ndarray) -> tuple[dict, dict]:
    speed = np.zeros(len(state))
    speed[1:] = np.linalg.norm(np.diff(state[:, ARM], axis=0), axis=1) * FPS
    idle = speed < IDLE_RAD_S
    moving = np.nonzero(~idle)[0]
    start, end = (int(moving[0]), int(moving[-1])) if len(moving) else (0, len(state) - 1)
    lags = range(0, 15)
    err = [np.abs(action[: len(action) - k, ARM] - state[k:, ARM]).mean() for k in lags]
    lag = int(np.argmin(err))
    track = np.zeros(len(state))
    track[: len(state) - lag] = np.linalg.norm(action[: len(action) - lag, ARM] - state[lag:, ARM], axis=1)
    jerk = np.zeros(len(action))
    jerk[3:] = np.linalg.norm(np.diff(action[:, ARM], n=3, axis=0), axis=1) * FPS ** 3 / 1e3  # krad/s^3
    rev = 0
    for g in GRIP:
        s_ = np.sign(np.diff(action[:, g]))
        s_ = s_[np.abs(np.diff(action[:, g])) > 1e-3]
        rev += int((s_[1:] != s_[:-1]).sum())
    tail = speed[-20:].mean()  # still moving when the fixed-length recording stopped?
    summary = dict(active_start=start, active_end=end, active_s=round((end - start) / FPS, 2),
                   end_speed_rad_s=float(tail), truncated_mid_motion=int(tail > IDLE_RAD_S),
                   idle_fraction=float(idle.mean()), longest_pause_s=longest_run(idle[start:end + 1]) / FPS,
                   leader_follower_lag_steps=lag, tracking_error_rad=float(track[start:end + 1].mean()),
                   max_jerk=float(jerk.max()), gripper_reversals=rev)
    return summary, dict(joint_speed=speed, idle=idle.astype(float), tracking_error=track, action_jerk=jerk)


def main() -> None:
    df = pd.read_parquet(SRC / "data" / "chunk-000" / "file-000.parquet")
    out = S.rerun_root() / "real_qa"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        state, action = np.stack(g["observation.state"]), np.stack(g["action"])
        summ, series = episode_qa(state, action)
        seg = segment_id(int(ep))
        rows.append(dict(segment_id=seg, episode=int(ep), **summ))
        rec = rr.RecordingStream(APP_ID_REAL, recording_id=seg)
        rec.save(out / f"{seg}.rrd")
        rr.send_property("qa", rr.AnyValues(**summ), recording=rec)
        steps = g["frame_index"].to_numpy()
        idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps)]
        for name, v in series.items():
            rr.send_columns(f"/qa/{name}", indexes=idx, columns=rr.Scalars.columns(scalars=v), recording=rec)
        rec.flush()
        rec.disconnect()
    res = pd.DataFrame(rows)
    (ROOT / "results" / "real").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "results" / "real" / "qa_real.csv", index=False)
    print(res.describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
