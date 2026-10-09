"""Paired counterfactual action drift for the nominal policies (paper Eq. 2, App. A).

Candidate c_i = (demo episode, nuisance condition); timesteps T_i = every `stride`-th step (paper: 50 episodes
-> 44,000 samples = 50 x 22 x 40). For each t: clean obs = recorded state re-rendered with condition 'clean',
nuisance obs = same state under the condition, same qpos. Policy outputs are already normalized (N(a)).

    python -m cfnbc.compute_drift [--seeds 0 1 2] [--stride 10] [--rerun_layers]

Writes results/drift/seed{S}/candidates.parquet (one row per candidate: episode, condition, family, drift d_i,
56-dim response feature r_i) and deltas.npz (per-sample Δa of the first chunk action and of the chunk mean).
--rerun_layers writes, per demo segment, a layer `<condition>` (nuisance frames /camera/top_<condition>, each seed's
predicted first action /policy/seed{S}/<condition>/action0 and ||Δa|| /policy/seed{S}/<condition>/delta_norm) and a
layer `clean_pred` (/policy/seed{S}/clean_pred/action0) under <rerun_root>/counterfactual/. Candidates of all seeds go
to the catalog table `drift_candidates` (unless --no_table). Drift is then reproducible as a query (cfnbc.drift_query).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cfnbc import nuisance as N
from cfnbc import schema as S
from cfnbc.counterfactual import StateRenderer
from cfnbc.policy_io import ACTRunner

ROOT = Path(__file__).resolve().parents[2]


def response_feature(delta: np.ndarray) -> np.ndarray:
    """delta (T,14) -> concat(mean, std, mean|.|, max|.|) over timesteps per action dim (App. A) = (56,)."""
    return np.concatenate([delta.mean(0), delta.std(0), np.abs(delta).mean(0), np.abs(delta).max(0)])


def write_layer(segment_id: str, layer: str, steps, frames: np.ndarray | None, preds: dict[str, np.ndarray],
                deltas: dict[str, np.ndarray], props: dict, out_root: Path) -> None:
    """One layer recording on the demo segment: counterfactual frames (optional) + per-seed predictions.

    preds:  {"seed0": (T, 14) first predicted action, normalized}  -> /policy/<seed>/<layer>/action0
    deltas: {"seed0": (T,) ||Δa|| vs clean}                         -> /policy/<seed>/<layer>/delta_norm
    """
    import rerun as rr
    out = out_root / layer
    out.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(S.APP_ID, recording_id=segment_id)
    rec.save(out / f"{segment_id}.rrd")
    rr.send_property("counterfactual", rr.AnyValues(**props), recording=rec)
    idx = [rr.TimeColumn(S.TIMELINE_STEP, sequence=steps), rr.TimeColumn(S.TIMELINE_TIME, duration=np.asarray(steps) * S.DT)]
    for seed, a in preds.items():
        rr.log(f"/policy/{seed}/{layer}/action0", rr.SeriesLines(names=S.JOINT_NAMES_14, colors=S.JOINT_COLORS_14), static=True, recording=rec)
        rr.send_columns(f"/policy/{seed}/{layer}/action0", indexes=idx, columns=rr.Scalars.columns(scalars=a),
                        recording=rec)
    for seed, d in deltas.items():
        rr.send_columns(f"/policy/{seed}/{layer}/delta_norm", indexes=idx, columns=rr.Scalars.columns(scalars=d),
                        recording=rec)
    if frames is not None:
        for t, img in zip(steps, frames):
            rr.set_time(S.TIMELINE_STEP, sequence=int(t), recording=rec)
            rr.set_time(S.TIMELINE_TIME, duration=int(t) * S.DT, recording=rec)
            rr.log(f"/camera/top_{layer}", rr.Image(img, color_model="RGB"), recording=rec)
    rec.flush()
    rec.disconnect()


def write_table(frames: dict[int, pd.DataFrame]) -> None:
    """Candidates of all seeds -> catalog table `drift_candidates` (overwritten)."""
    import pyarrow as pa
    from cfnbc import hub
    df = pd.concat([d.assign(seed=s) for s, d in frames.items()], ignore_index=True)
    df["response"] = df["response"].map(lambda r: np.asarray(r, dtype=np.float64).tolist())
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    with hub.open_catalog() as client:
        if "drift_candidates" in client.table_names():
            client.get_table("drift_candidates").overwrite(tbl.to_batches())
        else:
            hub.ensure_table(client, "drift_candidates", tbl.schema).append(tbl.to_batches())
    print(f"table drift_candidates: {len(df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--device", default=os.environ.get("ACT_DEVICE", "mps"))
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "drift"))
    parser.add_argument("--rerun_layers", action="store_true")
    parser.add_argument("--no_table", action="store_true")
    args = parser.parse_args()

    episodes = args.episodes if args.episodes is not None else list(range(50))
    conditions = args.conditions or N.NUISANCE_CONDITIONS
    runners = {s: ACTRunner(ROOT / "ckpts" / f"seed{s}", device=args.device) for s in args.seeds}
    renderer = StateRenderer()
    rows = {s: [] for s in args.seeds}
    d_first = {s: [] for s in args.seeds}
    d_mean = {s: [] for s in args.seeds}
    t0 = time.time()
    for ei, e in enumerate(episodes):
        with h5py.File(S.sim_data_dir() / f"episode_{e}.hdf5") as f:
            sim_q = f["sim_state/qpos"][()]
            obs_q = f["observations/qpos"][()]
        steps = np.arange(0, len(obs_q), args.stride)
        qpos = obs_q[steps]
        seg = S.demo_segment_id(e)
        cf_root = S.rerun_root() / "counterfactual"
        clean = renderer.render_episode(sim_q, "clean", steps)
        a_clean = {s: r.predict_normalized(qpos, clean) for s, r in runners.items()}
        if args.rerun_layers:  # clean predictions; clean pixels are the base layer's own frames (verified identical)
            write_layer(seg, "clean_pred", steps, None, {f"seed{s}": a[:, 0] for s, a in a_clean.items()}, {},
                        dict(condition="clean", family="clean", stride=args.stride), cf_root)
        for c in conditions:
            frames = renderer.render_episode(sim_q, c, steps)
            preds, deltas, props = {}, {}, dict(condition=c, family=N.FAMILY_OF[c], stride=args.stride)
            for s, r in runners.items():
                a_n = r.predict_normalized(qpos, frames)
                delta = a_n - a_clean[s]                                 # (T, chunk, 14)
                df, dm = delta[:, 0], delta.mean(1)                     # first action; chunk mean
                dnorm = np.linalg.norm(df, axis=1)
                rows[s].append(dict(episode=e, segment_id=seg, condition=c, family=N.FAMILY_OF[c],
                                    n_steps=len(steps), drift=float(dnorm.mean()),
                                    drift_chunk_mean=float(np.linalg.norm(dm, axis=1).mean()),
                                    peak_step=int(steps[np.argmax(dnorm)]), response=response_feature(df)))
                d_first[s].append(df.astype(np.float32))
                d_mean[s].append(dm.astype(np.float32))
                preds[f"seed{s}"], deltas[f"seed{s}"] = a_n[:, 0], dnorm
                props[f"drift_seed{s}"] = float(dnorm.mean())
            if args.rerun_layers:
                write_layer(seg, c, steps, frames, preds, deltas, props, cf_root)
        el = time.time() - t0
        print(f"episode {e} ({ei + 1}/{len(episodes)}) {el:.0f}s elapsed, "
              f"~{el / (ei + 1) * (len(episodes) - ei - 1):.0f}s left", flush=True)

    for s in args.seeds:
        out = Path(args.out_dir) / f"seed{s}"
        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows[s])
        df.to_parquet(out / "candidates.parquet")
        np.savez_compressed(out / "deltas.npz", first=np.stack(d_first[s]), chunk_mean=np.stack(d_mean[s]),
                            episode=df.episode.values, condition=df.condition.values, stride=args.stride)
        print(f"seed{s}: {len(df)} candidates -> {out}")
        print(df.groupby("condition").drift.mean().sort_values(ascending=False).round(4).to_string())
    if not args.no_table:
        write_table({s: pd.DataFrame(rows[s]) for s in args.seeds})


if __name__ == "__main__":
    main()
