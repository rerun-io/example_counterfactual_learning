"""Real-data training samples from the catalog: counterfactual JPEG frames + recorded teleop actions.

Layout on `real_demos` (LeRobot ingest): /observation.state (qpos), /action (leader arms), plus our layers
`<condition>` (/camera/cam_high_<condition>, EncodedImage JPEG), `clean_pred` (/camera/cam_high_clean) and
`train_pad` (zero actions past the episode end + /train/is_pad). Matches act/utils.py:EpisodicDataset for real data
(attrs sim=False): observation t is paired with action[t-1 : t+99] (upstream's leader/follower alignment), i.e. a
Rerun window of offsets -1..98. Steps t=0 are skipped (no t-1).

    python -m cfnbc.real.repair_real --pad        # write the train_pad layer for real_demos
    python -m cfnbc.real.repair_real --check      # equivalence vs HDF5 + EpisodicDataset math (clean frames)
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.experimental.dataloader as dl
import torch

from cfnbc import hub
from cfnbc import schema as S
from cfnbc.real.ingest import APP_ID_REAL, DATASET_REAL, ROOT, segment_id

CHUNK, EP_LEN = 100, 400
QPOS, ACTION = "/observation.state", "/action"
OFFSETS = tuple(range(-1, CHUNK - 1))


def write_pad_layer() -> None:
    out = S.rerun_root() / "real_train_pad"
    out.mkdir(parents=True, exist_ok=True)
    pad = np.arange(EP_LEN, EP_LEN + CHUNK - 1)
    allst = np.arange(EP_LEN + CHUNK - 1)
    for ep in range(50):
        seg = segment_id(ep)
        rec = rr.RecordingStream(APP_ID_REAL, recording_id=seg)
        rec.save(out / f"{seg}.rrd")
        rr.send_columns(ACTION, indexes=[rr.TimeColumn(S.TIMELINE_STEP, sequence=pad)],
                        columns=rr.Scalars.columns(scalars=np.zeros((len(pad), 14))), recording=rec)
        rr.send_columns("/train/is_pad", indexes=[rr.TimeColumn(S.TIMELINE_STEP, sequence=allst)],
                        columns=rr.Scalars.columns(scalars=(allst >= EP_LEN).astype(np.float64)), recording=rec)
        rec.flush(); rec.disconnect()
    print(f"real train_pad layer -> {out}")


def segment_dataset(ds_entry, seg: str, image_entity: str) -> dl.RerunMapDataset:
    return dl.RerunMapDataset(dl.DataSource(ds_entry, segments=[seg]), S.TIMELINE_STEP, {
        "image": dl.Field(f"{image_entity}:EncodedImage:blob", decode=dl.ImageDecoder()),
        "qpos": dl.Field(f"{QPOS}:Scalars:scalars", decode=dl.NumericDecoder()),
        "action": dl.Field(f"{ACTION}:Scalars:scalars", decode=dl.NumericDecoder(), window=OFFSETS),
        "is_pad": dl.Field("/train/is_pad:Scalars:scalars", decode=dl.NumericDecoder(), window=OFFSETS),
    })


class RealHubSamples(torch.utils.data.Dataset):
    """(segment, step, image_entity) -> (image (1,3,H,W) in [0,1], qpos norm, action chunk norm (100,14), is_pad)."""

    def __init__(self, client, items: list[tuple[str, int, str]], stats: dict, cache: bool = True):
        self.items = [(s, t, e) for s, t, e in items if t >= 1]
        self.stats = {k: torch.as_tensor(np.asarray(stats[k]), dtype=torch.float32)
                      for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
        ds_entry = client.get_dataset(DATASET_REAL)
        self._ds = {}
        for seg, _, ent in self.items:
            if (seg, ent) not in self._ds:
                self._ds[(seg, ent)] = segment_dataset(ds_entry, seg, ent)
        self._cache: dict | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.items)

    def _index(self, d: dl.RerunMapDataset, step: int) -> int:
        # samples exist where the -1..98 window is complete: steps 1..399 -> index step-1 (verified in --check)
        return step - 1

    def __getitem__(self, i: int):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        seg, step, ent = self.items[i]
        x = self._ds[(seg, ent)][self._index(self._ds[(seg, ent)], step)]
        image = x["image"].unsqueeze(0).float() / 255.0
        qpos = (x["qpos"].float() - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        action = (x["action"].float() - self.stats["action_mean"]) / self.stats["action_std"]
        out = (image, qpos, action, x["is_pad"].reshape(-1).bool())
        if self._cache is not None:
            self._cache[i] = out
        return out


def check(stats_path: Path | None = None, probes=None) -> None:
    import h5py
    from cfnbc.real.edits import decode_jpeg
    with open(stats_path or ROOT / "ckpts" / "real_cups_s0" / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    probes = probes or [(3, 10), (3, 200), (17, 390), (42, 1), (42, 350)]
    with hub.open_catalog() as client:
        ds = RealHubSamples(client, [(segment_id(e), t, "/camera/cam_high_clean") for e, t in probes], stats, cache=False)
        jp = client.get_dataset(DATASET_REAL)
        t0 = time.time()
        for k, (e, t) in enumerate(probes):
            image, qpos, action, is_pad = ds[k]
            with h5py.File(S.real_data_dir() / "act_cups_open" / f"episode_{e}.hdf5") as f:
                A, Q = f["action"][()], f["observations/qpos"][t]
            s0 = max(0, t - 1)
            pad = np.zeros((EP_LEN, 14), np.float32); pad[:EP_LEN - s0] = A[s0:]
            ref_a = (torch.from_numpy(pad[:CHUNK]) - torch.as_tensor(stats["action_mean"]).float()) / torch.as_tensor(stats["action_std"]).float()
            ref_q = (torch.from_numpy(Q).float() - torch.as_tensor(stats["qpos_mean"]).float()) / torch.as_tensor(stats["qpos_std"]).float()
            ref_p = torch.zeros(EP_LEN, dtype=torch.bool); ref_p[EP_LEN - s0:] = True
            tb = jp.filter_segments(segment_id(e)).filter_contents(["/camera/cam_high_clean"]).reader(index="step").to_arrow_table()
            blob = {st: b for st, b in zip(tb["step"].to_pylist(), tb["/camera/cam_high_clean:EncodedImage:blob"].to_pylist()) if b}
            ref_img = torch.from_numpy(decode_jpeg(bytes(blob[t][0])).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            print(f"episode {e:2d} step {t:3d}: image {torch.equal(image, ref_img)}, qpos {torch.allclose(qpos, ref_q, atol=1e-6)}, "
                  f"action chunk {torch.allclose(action, ref_a, atol=1e-5)}, is_pad {torch.equal(is_pad, ref_p[:CHUNK])}")
        print(f"{(time.time() - t0) / len(probes) * 1000:.0f} ms/sample")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pad", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.pad:
        write_pad_layer()
    if a.check:
        check()
