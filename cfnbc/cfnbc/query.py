"""Analysis queries over the experiment catalog.

    from cfnbc import query as Q
    srv = Q.local_server()               # or client = Q.connect("rerun+http://...")
    client = srv.client()
    Q.eval_summary(client)               # per (run, ckpt, temporal_agg, eval_seed, nuisance) success rate
    Q.training_curves(client)            # long-form epoch metrics for every run
"""
from __future__ import annotations

import pandas as pd
import rerun as rr

from cfnbc import schema as S
from cfnbc.serve import start as local_server  # noqa: F401  (re-export)


def connect(url: str) -> rr.catalog.CatalogClient:
    return rr.catalog.CatalogClient(url)


def _props(client, dataset: str, prefix: str) -> pd.DataFrame:
    df = client.get_dataset(dataset).segment_table().to_pandas()
    cols = {c: c.split(":")[-1] for c in df.columns if c.startswith(f"property:{prefix}:")}
    out = df[["rerun_segment_id", *cols]].rename(columns=cols)
    for c in cols.values():  # property components are batches; unwrap single values, keep vectors as arrays
        out[c] = out[c].map(lambda v: v[0] if v is not None and len(v) == 1 else v)
    return out


def rollout_table(client) -> pd.DataFrame:
    """One row per eval rollout segment with its properties."""
    return _props(client, S.DATASET_ROLLOUTS, "rollout")


def eval_summary(client) -> pd.DataFrame:
    keys = ["run", "ckpt_name", "temporal_agg", "eval_seed", "nuisance"]
    df = rollout_table(client)
    return (df.groupby(keys).agg(n=("success", "size"), success_rate=("success", "mean"),
                                 mean_highest_reward=("highest_reward", "mean")).reset_index())


def demo_table(client) -> pd.DataFrame:
    return _props(client, S.DATASET_DEMOS, "demo")


def training_runs(client) -> pd.DataFrame:
    return _props(client, S.DATASET_TRAINING, "train").merge(
        _props(client, S.DATASET_TRAINING, "result"), on="rerun_segment_id", how="left")


def training_curves(client, splits=("train", "val")) -> pd.DataFrame:
    """Long-form: segment, epoch, split, metric, value."""
    ds = client.get_dataset(S.DATASET_TRAINING)
    df = ds.reader(index="epoch").to_pandas()
    out = []
    for col in df.columns:
        parts = col.split(":")
        if len(parts) != 3 or parts[1] != "Scalars":
            continue
        split, metric = parts[0].strip("/").split("/", 1)
        if split not in splits:
            continue
        vals = df[col].map(lambda v: v[0] if v is not None and len(v) else None)
        out.append(pd.DataFrame({"segment": df["rerun_segment_id"], "epoch": df["epoch"], "split": split,
                                 "metric": metric, "value": vals}))
    return pd.concat(out).dropna(subset=["value"]).sort_values(["segment", "split", "metric", "epoch"])


def segment_scalars(client, dataset: str, segment_id: str, entity: str) -> pd.DataFrame:
    """(step x 14|23) array for one segment's Scalars entity, e.g. S.ACTION, as a wide DataFrame."""
    ds = client.get_dataset(dataset).filter_segments(segment_id).filter_contents(entity)
    df = ds.reader(index=S.TIMELINE_STEP).to_pandas()
    col = f"{entity}:Scalars:scalars"
    wide = pd.DataFrame(df[col].tolist(), index=df[S.TIMELINE_STEP])
    if wide.shape[1] == 14:
        wide.columns = S.JOINT_NAMES_14
    return wide.sort_index()
