"""ACT training samples streamed from the Rerun catalog (rerun.experimental.dataloader).

A repair selection (table `repair_selections`) names (demo segment, condition) candidates. Each candidate contributes
the counterfactual frames at its sampled steps T_i (every 10th step), paired with the *clean* demo's qpos and expert
action chunk (paper Eq. 5): image from layer `<condition>` (/camera/top_<condition>), qpos/action from `base`, padding
mask from `train_pad`. One RerunMapDataset per segment (index == step; verified exact), samples cached after the first
fetch (the repair set is small: K=20 x 40 steps).

Returns the same tensors as act/utils.py:EpisodicDataset: (image (1,3,H,W) float in [0,1], qpos normalized,
action chunk normalized (100,14), is_pad (100,) bool).

    python -m cfnbc.repair_data --check      # equivalence vs HDF5 + EpisodicDataset math on clean frames
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import rerun.experimental.dataloader as dl

from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[2]
H, W, CHUNK = 480, 640, 100


def segment_dataset(ds_entry, segment: str, image_entity: str) -> dl.RerunMapDataset:
    return dl.RerunMapDataset(dl.DataSource(ds_entry, segments=[segment]), S.TIMELINE_STEP, {
        "image": dl.Field(f"{image_entity}:Image:buffer", decode=dl.NumericDecoder()),
        "qpos": dl.Field(f"{S.QPOS}:Scalars:scalars", decode=dl.NumericDecoder()),
        "action": dl.Field(f"{S.ACTION}:Scalars:scalars", decode=dl.NumericDecoder(), window=tuple(range(CHUNK))),
        "is_pad": dl.Field("/train/is_pad:Scalars:scalars", decode=dl.NumericDecoder(), window=tuple(range(CHUNK))),
    })


class HubSamples(torch.utils.data.Dataset):
    """Samples (segment, step, image_entity) read from the catalog, normalized like EpisodicDataset."""

    def __init__(self, client, items: list[tuple[str, int, str]], stats: dict, cache: bool = True):
        self.items = items
        self.stats = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in stats.items()
                      if k in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
        ds_entry = client.get_dataset(S.DATASET_DEMOS)
        self._ds = {}
        for seg, _, ent in items:
            if (seg, ent) not in self._ds:
                self._ds[(seg, ent)] = segment_dataset(ds_entry, seg, ent)
        self._cache: dict[int, tuple] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        seg, step, ent = self.items[i]
        x = self._ds[(seg, ent)][step]
        image = x["image"].reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.0   # (1 cam, 3, H, W)
        qpos = (x["qpos"].float() - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        action = (x["action"].float() - self.stats["action_mean"]) / self.stats["action_std"]
        is_pad = x["is_pad"].reshape(-1).bool()
        # EpisodicDataset normalizes the zero padding too; keep identical semantics (padded rows are masked in the loss)
        out = (image, qpos, action, is_pad)
        if self._cache is not None:
            self._cache[i] = out
        return out


def selection_items(client, selection: str, stride: int = 10) -> list[tuple[str, int, str]]:
    sel = client.get_table("repair_selections").reader().to_pandas()
    sel = sel[sel.selection == selection]
    if sel.empty:
        raise SystemExit(f"selection {selection} not found in table repair_selections")
    return [(r.segment_id, t, f"/camera/top_{r.condition}") for r in sel.itertuples() for t in range(0, 400, stride)]


def load_stats(ckpt_dir: Path) -> dict:
    with open(ckpt_dir / "dataset_stats.pkl", "rb") as f:
        return pickle.load(f)


def check() -> None:
    """Clean-frame samples from the catalog must equal EpisodicDataset's math on the HDF5 episode."""
    import h5py
    from cfnbc import hub
    stats = load_stats(ROOT / "ckpts" / "seed0")
    rng = np.random.default_rng(0)
    probes = [(int(e), int(t)) for e, t in zip(rng.integers(0, 50, 6), rng.integers(0, 400, 6))] + [(3, 399), (11, 350)]
    with hub.open_catalog() as client:
        ds = HubSamples(client, [(S.demo_segment_id(e), t, S.CAMERA_TOP) for e, t in probes], stats, cache=False)
        t0 = time.time()
        for k, (e, t) in enumerate(probes):
            image, qpos, action, is_pad = ds[k]
            with h5py.File(S.sim_data_dir() / f"episode_{e}.hdf5") as f:
                A, Q, I = f["action"][()], f["observations/qpos"][t], f["observations/images/top"][t]
            pad = np.zeros((400, 14), np.float32); pad[:400 - t] = A[t:]
            ref_action = (torch.from_numpy(pad) - torch.as_tensor(stats["action_mean"]).float()) / torch.as_tensor(stats["action_std"]).float()
            ref_qpos = (torch.from_numpy(Q).float() - torch.as_tensor(stats["qpos_mean"]).float()) / torch.as_tensor(stats["qpos_std"]).float()
            ref_img = torch.einsum("hwc->chw", torch.from_numpy(I)).unsqueeze(0).float() / 255.0
            ref_pad = torch.zeros(400, dtype=torch.bool); ref_pad[400 - t:] = True
            ok = (torch.equal(image, ref_img), torch.allclose(qpos, ref_qpos, atol=1e-6),
                  torch.allclose(action, ref_action[:CHUNK], atol=1e-5), torch.equal(is_pad, ref_pad[:CHUNK]))
            print(f"episode {e:2d} step {t:3d}: image {ok[0]}, qpos {ok[1]}, action chunk {ok[2]}, is_pad {ok[3]}")
        print(f"{(time.time() - t0) / len(probes) * 1000:.0f} ms/sample (uncached)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    if p.parse_args().check:
        check()
