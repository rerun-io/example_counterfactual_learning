"""Ingest a real LeRobot teleop dataset (ALOHA static) into the catalog: one segment per episode.

    python -m cfnbc.real.ingest [--dataset data_real/aloha_static_cups_open]

Uses rerun.experimental.LeRobotReader: 4 AV1 camera streams stay native (VideoStream samples), state/action become
Scalars on a `step` timeline, the task string a TextDocument. A small `provenance` layer adds segment properties
(source dataset + revision, robot, fps, episode length, per-episode cam_high brightness from the dataset's own stats).
Writes <rerun_root>/real_demos/cups_eNNN.rrd (layer base) and <rerun_root>/real_provenance/ (layer provenance).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.experimental as rx

from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[3]
DATASET_REAL = "real_demos"
APP_ID_REAL = "cfnbc_aloha_cups_open"
REVISION = "v3.0"


def segment_id(ep: int) -> str:
    return f"cups_e{ep:03d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(S.real_data_dir() / "aloha_static_cups_open"))
    args = p.parse_args()
    src = Path(args.dataset)
    info = json.loads((src / "meta" / "info.json").read_text())
    eps = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    reader = rx.LeRobotReader(src)
    out, prov = S.rerun_root() / "real_demos", S.rerun_root() / "real_provenance"
    out.mkdir(parents=True, exist_ok=True)
    prov.mkdir(parents=True, exist_ok=True)
    for ep in reader.episodes():
        seg = segment_id(ep)
        reader.stream(ep, timeline=S.TIMELINE_STEP).write_rrd(str(out / f"{seg}.rrd"), application_id=APP_ID_REAL,
                                                                recording_id=seg)
        row = eps[eps.episode_index == ep].iloc[0]
        mean_rgb = np.asarray([np.asarray(c).ravel()[0] for c in row["stats/observation.images.cam_high/mean"]])
        rec = rr.RecordingStream(APP_ID_REAL, recording_id=seg)
        rec.save(prov / f"{seg}.rrd")
        rr.send_property("provenance", rr.AnyValues(
            source=f"huggingface:lerobot/{src.name}@{REVISION}", robot=info.get("robot_type", ""), fps=int(info["fps"]),
            episode_index=int(ep), length=int(row["length"]), operator="human teleop (ALOHA leader arms)",
            cam_high_mean_brightness=float(mean_rgb.mean()), cameras=",".join(k.split(".")[-1] for k in info["features"] if "images" in k),
        ), recording=rec)
        rec.flush()
        rec.disconnect()
    print(f"ingested {len(reader.episodes())} episodes -> {out}")


if __name__ == "__main__":
    main()
