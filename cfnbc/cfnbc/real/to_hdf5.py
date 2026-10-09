"""Convert the LeRobot ALOHA dataset to upstream ACT's HDF5 episode format (training reference path).

    python -m cfnbc.real.to_hdf5 [--cams cam_high]

Writes data_real/act_cups_open/episode_{i}.hdf5 with /observations/qpos (14, follower state), /observations/qvel
(finite difference, unused by ACT), /observations/images/<cam> (480x640x3 uint8, decoded once from the AV1 stream),
/action (14, leader arms) and attrs sim=False, so EpisodicDataset applies its real-data alignment
(action[t-1:] for observation t, upstream's handling of leader/follower timing).
"""
from __future__ import annotations

import argparse
import json

import av
import h5py
import numpy as np
import pandas as pd

from cfnbc.real.ingest import ROOT
from cfnbc import schema as S

SRC = S.real_data_dir() / "aloha_static_cups_open"
OUT = S.real_data_dir() / "act_cups_open"
FPS = 50


def decode_all(cam: str, n_frames: int) -> np.ndarray:
    c = av.open(str(SRC / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"))
    out = np.empty((n_frames, 480, 640, 3), np.uint8)
    for i, fr in enumerate(c.decode(video=0)):
        if i >= n_frames:
            break
        out[i] = fr.to_ndarray(format="rgb24")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cams", nargs="+", default=["cam_high"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC / "data" / "chunk-000" / "file-000.parquet").sort_values("index")
    eps = pd.read_parquet(SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = json.loads((SRC / "meta" / "info.json").read_text())
    frames = {cam: decode_all(cam, len(df)) for cam in args.cams}
    for _, e in eps.iterrows():
        ep = int(e.episode_index)
        a, b = int(e["dataset_from_index"]), int(e["dataset_to_index"])
        g = df.iloc[a:b]
        qpos = np.stack(g["observation.state"]).astype(np.float32)  # float32 like upstream's recorder
        qvel = np.vstack([np.zeros((1, 14)), np.diff(qpos, axis=0) * FPS])
        # video frames are time-aligned with parquet rows (episode windows from_timestamp/to_timestamp at 50 fps)
        f0 = int(round(float(e[f"videos/observation.images.{args.cams[0]}/from_timestamp"]) * FPS))
        with h5py.File(OUT / f"episode_{ep}.hdf5", "w") as h:
            h.attrs["sim"] = False
            h.attrs["source"] = f"lerobot/aloha_static_cups_open@v3.0 episode {ep}"
            h.create_dataset("observations/qpos", data=qpos)
            h.create_dataset("observations/qvel", data=qvel.astype(np.float32))
            h.create_dataset("action", data=np.stack(g["action"]).astype(np.float32))
            for cam in args.cams:
                h.create_dataset(f"observations/images/{cam}", data=frames[cam][f0:f0 + len(g)], chunks=(1, 480, 640, 3))
    print(f"wrote {len(eps)} episodes ({info['total_frames']} frames, cams {args.cams}) -> {OUT}")


if __name__ == "__main__":
    main()
