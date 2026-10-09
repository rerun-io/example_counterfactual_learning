"""Training-view layer for ACT: action padding as data, so a windowed Rerun loader matches EpisodicDataset.

ACT's EpisodicDataset samples any start step t in [0, 400) and takes the action chunk a[t : t+100], zero-padded
past the episode end with an `is_pad` mask. A Rerun `Field(window=range(100))` only yields samples whose whole window
exists, which would drop t > 300. The `train_pad` layer appends zero actions at steps 400..498 and an `/train/is_pad`
scalar over 0..498 (0 for real steps, 1 for padding), so every real step has a complete window and the mask comes from
the same query. Samples at t >= 400 remain incomplete (their windows run past 498) and are excluded by the loader.

    python -m cfnbc.training_layers            # writes <rerun_root>/train_pad/demo_eNNN.rrd for all demos
"""
from __future__ import annotations

import argparse

import numpy as np
import rerun as rr

from cfnbc import schema as S

EPISODE_LEN, CHUNK = 400, 100


def write_pad_layer(e: int) -> None:
    seg = S.demo_segment_id(e)
    out = S.rerun_root() / "train_pad"
    out.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(S.APP_ID, recording_id=seg)
    rec.save(out / f"{seg}.rrd")
    rr.send_property("train_pad", rr.AnyValues(episode_len=EPISODE_LEN, chunk=CHUNK), recording=rec)
    pad_steps = np.arange(EPISODE_LEN, EPISODE_LEN + CHUNK - 1)
    idx = lambda st: [rr.TimeColumn(S.TIMELINE_STEP, sequence=st), rr.TimeColumn(S.TIMELINE_TIME, duration=st * S.DT)]
    rr.send_columns(S.ACTION, indexes=idx(pad_steps), columns=rr.Scalars.columns(scalars=np.zeros((len(pad_steps), 14))),
                    recording=rec)
    all_steps = np.arange(EPISODE_LEN + CHUNK - 1)
    rr.send_columns("/train/is_pad", indexes=idx(all_steps),
                    columns=rr.Scalars.columns(scalars=(all_steps >= EPISODE_LEN).astype(np.float64)), recording=rec)
    rec.flush()
    rec.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="*", default=list(range(50)))
    args = parser.parse_args()
    for e in args.episodes:
        write_pad_layer(e)
    print(f"train_pad layer written for {len(args.episodes)} demos")


if __name__ == "__main__":
    main()
