"""Clean ACT training samples streamed from the catalog with RerunIterableDataset.

The clean demos are dense: every step 0..399 of every episode is a valid training sample (image, qpos, the next 100
actions, padding mask), so they suit the iterable dataset, which streams the whole index in fetch blocks, shuffles
internally and shards across DataLoader workers. Repair sets stay map-style (cfnbc.repair_data): they are a short list
of exact (episode, step) samples from sparse counterfactual layers.

Windowed fields (action, is_pad over steps t..t+99, with the `train_pad` layer supplying zeros past step 399) make
the index exactly steps 0..399 per episode. Output matches act/utils.py:EpisodicDataset for the same (episode, step).

    python -m cfnbc.clean_stream --check     # all 400 samples of two episodes vs EpisodicDataset math on HDF5
    python -m cfnbc.clean_stream --bench     # throughput, shuffled, over the 40 training episodes
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import rerun.experimental.dataloader as dl

from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[2]
H, W, CHUNK = 480, 640, 100


def fields(image_entity: str = S.CAMERA_TOP) -> dict[str, dl.Field]:
    return {
        "image": dl.Field(f"{image_entity}:Image:buffer", decode=dl.NumericDecoder()),
        "qpos": dl.Field(f"{S.QPOS}:Scalars:scalars", decode=dl.NumericDecoder()),
        "action": dl.Field(f"{S.ACTION}:Scalars:scalars", decode=dl.NumericDecoder(), window=tuple(range(CHUNK))),
        "is_pad": dl.Field("/train/is_pad:Scalars:scalars", decode=dl.NumericDecoder(), window=tuple(range(CHUNK))),
    }


def train_segments(num_episodes: int = 50, train_ratio: float = 0.8) -> list[str]:
    """Upstream's training split: set_seed(1) leaves np.random in seed-1 state before load_data's permutation."""
    perm = np.random.RandomState(1).permutation(num_episodes)
    return [S.demo_segment_id(int(e)) for e in perm[:int(train_ratio * num_episodes)]]


class CleanStream(torch.utils.data.IterableDataset):
    """Normalized (image (1,3,H,W), qpos, action chunk (100,14), is_pad (100,)) samples, like EpisodicDataset."""

    def __init__(self, ds_entry, segments: list[str], stats: dict, shuffle: dl.ShuffleStrategy | None = None,
                 fetch_block_size: int = 128):
        self.inner = dl.RerunIterableDataset(dl.DataSource(ds_entry, segments=segments), S.TIMELINE_STEP, fields(),
                                             shuffle_strategy=shuffle, fetch_block_size=fetch_block_size)
        self.stats = {k: torch.as_tensor(np.asarray(stats[k]), dtype=torch.float32)
                      for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")}

    def set_epoch(self, epoch: int) -> None:
        self.inner.set_epoch(epoch)

    def __iter__(self):
        s = self.stats
        for x in self.inner:
            image = x["image"].reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            qpos = (x["qpos"].float() - s["qpos_mean"]) / s["qpos_std"]
            action = (x["action"].float() - s["action_mean"]) / s["action_std"]
            yield image, qpos, action, x["is_pad"].reshape(-1).bool()


def _stats():
    with open(ROOT / "ckpts" / "seed0" / "dataset_stats.pkl", "rb") as f:
        return pickle.load(f)


def check(episodes=(3, 17)) -> None:
    import h5py
    from cfnbc import hub
    stats = _stats()
    am, asd = torch.as_tensor(stats["action_mean"]).float(), torch.as_tensor(stats["action_std"]).float()
    qm, qsd = torch.as_tensor(stats["qpos_mean"]).float(), torch.as_tensor(stats["qpos_std"]).float()
    with hub.open_catalog() as client:
        demos = client.get_dataset(S.DATASET_DEMOS)
        for e in episodes:
            with h5py.File(S.sim_data_dir() / f"episode_{e}.hdf5") as f:
                A, Q, I = f["action"][()], f["observations/qpos"][()], f["observations/images/top"][()]
            n, bad = 0, 0
            for t, (image, qpos, action, is_pad) in enumerate(CleanStream(demos, [S.demo_segment_id(e)], stats,
                                                                          shuffle=dl.NoShuffle())):
                pad = np.zeros((400, 14), np.float32); pad[:400 - t] = A[t:]
                ref_a = (torch.from_numpy(pad[:CHUNK]) - am) / asd
                ref_q = (torch.from_numpy(Q[t]).float() - qm) / qsd
                ref_i = torch.from_numpy(I[t]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                ref_p = torch.zeros(400, dtype=torch.bool); ref_p[400 - t:] = True
                ok = (torch.equal(image, ref_i) and torch.allclose(qpos, ref_q, atol=1e-6)
                      and torch.allclose(action, ref_a, atol=1e-5) and torch.equal(is_pad, ref_p[:CHUNK]))
                bad += not ok
                n += 1
            print(f"episode {e}: {n} samples streamed, {n - bad} identical to EpisodicDataset, {bad} different")


def bench(samples: int = 2000, batch_size: int = 8) -> None:
    from cfnbc import hub
    stats = _stats()
    with hub.open_catalog() as client:
        demos = client.get_dataset(S.DATASET_DEMOS)
        for name, shuffle in (("SampleShuffle", dl.SampleShuffle()), ("BlockShuffle(buffer 512)", dl.BlockShuffle(buffer_size=512))):
            ds = CleanStream(demos, train_segments(), stats, shuffle=shuffle)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)
            t0, n = time.time(), 0
            for batch in loader:
                n += len(batch[0])
                if n >= samples:
                    break
            dt = time.time() - t0
            print(f"{name:26s} {n} samples in {dt:.1f}s = {n / dt:.0f} samples/s (incl. first fetch)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--bench", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    if a.bench:
        bench()
