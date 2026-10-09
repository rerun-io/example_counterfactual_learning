"""Counterfactual action drift on real teleop data: edited real frames instead of re-rendered sim states.

    python -m cfnbc.real.compute_drift_real [--ckpt_dir ckpts/real_cups_s0] [--episodes ...] [--conditions ...]

For each episode and every `stride`-th step: clean frame (cam_high, JPEG round trip) and each edited frame
(cfnbc.real.edits, JPEG round trip) go through the real-data ACT with the same qpos; drift = mean ||Δa|| of the
first predicted action (normalized), response feature as in the sim audit. Writes to the catalog (real_demos):
  layer `masks`               /masks/keep_out (SegmentationImage at sampled steps), /masks/swept (static)
  layer `clean_pred`          /camera/cam_high_clean (EncodedImage, what the policy saw), /policy/real/clean_pred/action0
  layer `<condition>`         /camera/cam_high_<condition> (EncodedImage), /policy/real/<condition>/action0, .../delta_norm
  tables `real_drift_candidates` (one row per episode x condition) and `real_edit_checks` (mask-violation checks)
and results/real/drift/candidates.parquet (reference).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rerun as rr

from cfnbc import hub
from cfnbc import schema as S
from cfnbc.compute_drift import response_feature
from cfnbc.policy_io import ACTRunner
from cfnbc.real import edits as E
from cfnbc.real.ingest import APP_ID_REAL, DATASET_REAL, ROOT, segment_id

DATA = S.real_data_dir() / "act_cups_open"
OUT_ROOT = lambda: S.rerun_root() / "real_counterfactual"


def _rec(layer: str, seg: str) -> rr.RecordingStream:
    out = (S.rerun_root() / "real_masks") if layer == "masks" else OUT_ROOT() / layer
    out.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(APP_ID_REAL, recording_id=seg)
    rec.save(out / f"{seg}.rrd")
    return rec


def _idx(steps):
    return [rr.TimeColumn(S.TIMELINE_STEP, sequence=np.asarray(steps))]


def write_masks(seg: str, steps, keep_outs, ctx: E.EpisodeContext) -> None:
    rec = _rec("masks", seg)
    rr.log("/masks", rr.AnnotationContext([(0, "editable table", (0, 0, 0, 0)), (1, "keep-out", (255, 40, 40, 140)),
                                           (2, "swept by arms/objects", (255, 200, 0, 90))]), static=True, recording=rec)
    rr.log("/masks/swept", rr.SegmentationImage(np.where(ctx.swept, 2, 0).astype(np.uint8)), static=True, recording=rec)
    for t, ko in zip(steps, keep_outs):
        rr.set_time(S.TIMELINE_STEP, sequence=int(t), recording=rec)
        rr.log("/masks/keep_out", rr.SegmentationImage(ko.astype(np.uint8)), recording=rec)
    rec.flush(); rec.disconnect()


def write_layer(seg: str, layer: str, steps, jpegs: list[bytes], pred: np.ndarray, delta: np.ndarray | None,
                props: dict) -> None:
    rec = _rec(layer, seg)
    rr.send_property("counterfactual", rr.AnyValues(**props), recording=rec)
    cam = f"/camera/cam_high_{'clean' if layer == 'clean_pred' else layer}"
    ent = f"/policy/real/{layer}"
    if not np.isnan(pred).all():
        rr.log(f"{ent}/action0", rr.SeriesLines(names=S.JOINT_NAMES_14, colors=S.JOINT_COLORS_14), static=True, recording=rec)
        rr.send_columns(f"{ent}/action0", indexes=_idx(steps), columns=rr.Scalars.columns(scalars=pred), recording=rec)
    if delta is not None and not np.isnan(delta).all():
        rr.send_columns(f"{ent}/delta_norm", indexes=_idx(steps), columns=rr.Scalars.columns(scalars=delta), recording=rec)
    for t, jb in zip(steps, jpegs):
        rr.set_time(S.TIMELINE_STEP, sequence=int(t), recording=rec)
        rr.log(cam, rr.EncodedImage(contents=jb, media_type="image/jpeg"), recording=rec)
    rec.flush(); rec.disconnect()


def write_tables(cands: pd.DataFrame, checks: pd.DataFrame) -> None:
    import pyarrow as pa
    with hub.open_catalog() as client:
        for name, df in (("real_drift_candidates", cands), ("real_edit_checks", checks)):
            df = df.copy()
            if "response" in df:
                df["response"] = df.response.map(lambda r: np.asarray(r, dtype=np.float64).tolist())
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if name in client.table_names():
                client.get_table(name).overwrite(tbl.to_batches())
            else:
                hub.ensure_table(client, name, tbl.schema).append(tbl.to_batches())
            print(f"table {name}: {len(df)} rows")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=str(ROOT / "ckpts" / "real_cups_s0"))
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--device", default=os.environ.get("ACT_DEVICE", "mps"))
    p.add_argument("--out_dir", default=str(ROOT / "results" / "real" / "drift"))
    p.add_argument("--no_table", action="store_true")
    p.add_argument("--frames_only", action="store_true",
                   help="write masks + clean/edited frame layers without a policy (no predictions, no drift tables)")
    args = p.parse_args()
    episodes = args.episodes if args.episodes is not None else list(range(50))
    conds = args.conditions or E.DEFAULT_CONDITIONS
    runner = None if args.frames_only else ACTRunner(args.ckpt_dir, device=args.device)
    rows, checks, t0 = [], [], time.time()
    for k, ep in enumerate(episodes):
        seg = segment_id(ep)
        with h5py.File(DATA / f"episode_{ep}.hdf5") as f:
            steps = np.arange(0, f["action"].shape[0], args.stride)
            frames = f["observations/images/cam_high"][steps]
            qpos = f["observations/qpos"][steps].astype(np.float64)
        ctx = E.EpisodeContext.build(frames)
        masks = [E.frame_masks(fr, ctx) for fr in frames]
        write_masks(seg, steps, [m[1] for m in masks], ctx)
        clean_jpg = [E.encode_jpeg(fr) for fr in frames]   # stored bytes == what the policy sees (decoded below)
        a_clean = (runner.predict_normalized(qpos, np.stack([E.decode_jpeg(b) for b in clean_jpg])) if runner
                   else np.full((len(steps), 1, 14), np.nan))
        write_layer(seg, "clean_pred", steps, clean_jpg, a_clean[:, 0], None,
                    dict(condition="clean", family="clean", stride=args.stride))
        for c in conds:
            outs = [E.apply(fr, c, ctx, m, roundtrip=False) for fr, m in zip(frames, masks)]
            jpgs = [E.encode_jpeg(o[0]) for o in outs]
            a_n = (runner.predict_normalized(qpos, np.stack([E.decode_jpeg(b) for b in jpgs])) if runner
                   else np.full_like(a_clean, np.nan))
            d = a_n[:, 0] - a_clean[:, 0]
            dn = np.linalg.norm(d, axis=1)
            fam = E.CONDITIONS[c][0]
            rows.append(dict(episode=ep, segment_id=seg, condition=c, family=fam, n_steps=len(steps),
                             drift=float(dn.mean()), peak_step=int(steps[np.argmax(dn)]), response=response_feature(d)))
            ch = pd.DataFrame([o[1] for o in outs])
            checks.append(dict(segment_id=seg, condition=c, family=fam, swept_frac=float(ctx.swept.mean()),
                               keepout_violation_max=float(ch.keepout_violation_frac.max()),
                               changed_frac_mean=float(ch.changed_frac.mean()),
                               distractor_clearance_min_px=float(ch["distractor_clearance_px"].min()) if "distractor_clearance_px" in ch else float("nan")))
            write_layer(seg, c, steps, jpgs, a_n[:, 0], dn,
                        dict(condition=c, family=fam, stride=args.stride, drift=float(dn.mean())))
        el = time.time() - t0
        print(f"episode {ep} ({k + 1}/{len(episodes)}) {el:.0f}s, ~{el / (k + 1) * (len(episodes) - k - 1) / 60:.0f} min left", flush=True)
    if args.frames_only:
        chk = pd.DataFrame(checks)
        print("frames-only: max keep-out violation", chk.keepout_violation_max.max(), "| min distractor clearance px",
              chk.distractor_clearance_min_px.min())
        return
    cands, chk = pd.DataFrame(rows), pd.DataFrame(checks)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cands.to_parquet(out / "candidates.parquet"); chk.to_csv(out / "edit_checks.csv", index=False)
    print(cands.groupby("condition").drift.mean().sort_values(ascending=False).round(4).to_string())
    print("max keep-out violation:", chk.keepout_violation_max.max(), "| min distractor clearance px:", chk.distractor_clearance_min_px.min())
    if not args.no_table:
        write_tables(cands, chk)


if __name__ == "__main__":
    main()
