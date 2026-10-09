"""Export ACT HDF5 demos (with the cfnbc sim_state/ group) to one .rrd per episode.

    python -m cfnbc.export_demos --dataset_dir $ACT_DATA_DIR/sim_transfer_cube_scripted [--out_dir rerun/demos]

Images are stored raw (lossless) so a Rerun-backed dataloader sees the same pixels as HDF5 training.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import rerun as rr

from cfnbc import schema as S


def export_episode(h5_path: Path, out_dir: Path) -> Path:
    episode_idx = int(re.search(r"episode_(\d+)\.hdf5$", h5_path.name).group(1))
    segment_id = S.demo_segment_id(episode_idx)
    out_path = out_dir / f"{segment_id}.rrd"

    rec = rr.RecordingStream(S.APP_ID, recording_id=segment_id)
    rec.save(out_path)
    with h5py.File(h5_path, "r") as f:
        n = f["action"].shape[0]
        steps = np.arange(n)
        indexes = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps),
                   rr.TimeColumn(S.TIMELINE_TIME, duration=steps * S.DT)]

        attrs = dict(f.attrs)
        rr.send_property("demo", rr.AnyValues(
            episode_idx=episode_idx,
            success=int(attrs.get("success", -1)),
            record_seed=int(attrs.get("record_seed", -1)),
            attempt_idx=int(attrs.get("attempt_idx", -1)),
            mujoco_version=str(attrs.get("mujoco_version", "")),
            init_env_state=np.asarray(attrs.get("init_env_state", []), dtype=np.float64),
            source=str(h5_path),
        ), recording=rec)
        S.log_static_series_names(rec)

        for path, key in ((S.QPOS, "observations/qpos"), (S.QVEL, "observations/qvel"), (S.ACTION, "action")):
            rr.send_columns(path, indexes=indexes, columns=rr.Scalars.columns(scalars=f[key][()]), recording=rec)

        if "sim_state" in f:
            sim_qpos = f["sim_state/qpos"][()]
            rr.send_columns(S.SIM_QPOS, indexes=indexes, columns=rr.Scalars.columns(scalars=sim_qpos), recording=rec)
            rr.send_columns(S.SIM_QVEL, indexes=indexes, columns=rr.Scalars.columns(scalars=f["sim_state/qvel"][()]),
                            recording=rec)
            quat_xyzw = sim_qpos[:, [20, 21, 22, 19]]
            rr.send_columns(S.CUBE, indexes=indexes,
                            columns=rr.Transform3D.columns(translation=sim_qpos[:, 16:19], quaternion=quat_xyzw),
                            recording=rec)

        images = f["observations/images/top"]
        for t in range(n):
            rr.set_time(S.TIMELINE_STEP, sequence=t, recording=rec)
            rr.set_time(S.TIMELINE_TIME, duration=t * S.DT, recording=rec)
            rr.log(S.CAMERA_TOP, rr.Image(images[t], color_model="RGB"), recording=rec)
    rec.flush()
    rec.disconnect()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", default=None, help="default: <rerun_root>/demos")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="subset of episode indices")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else S.rerun_root() / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.dataset_dir).glob("episode_*.hdf5"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    paths = [p for p in paths if not p.is_symlink()]
    if args.episodes is not None:
        keep = {f"episode_{i}.hdf5" for i in args.episodes}
        paths = [p for p in paths if p.name in keep]
    for p in paths:
        out = export_episode(p, out_dir)
        print(f"{p.name} -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
