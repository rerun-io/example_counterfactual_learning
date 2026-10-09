"""Action drift recomputed from the catalog: join prediction layers on (segment, step).

The demo segment's layers `clean_pred` and `<condition>` (written by cfnbc.compute_drift --rerun_layers) hold each
seed's predicted first action at the sampled steps. Drift per candidate = mean over steps of
||a_condition - a_clean||_2 (paper Eq. 2; predictions are already normalized). This module reproduces it with one
query per (seed, condition set) and cross-checks the numpy reference (results/drift/seed{S}/candidates.parquet).

    python -m cfnbc.drift_query --seed 0 [--conditions ...]      # prints per-condition drift + cross-check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cfnbc import hub
from cfnbc import schema as S

ROOT = Path(__file__).resolve().parents[2]


def _col(layer: str, seed: int) -> str:
    return f"/policy/seed{seed}/{layer}/action0:Scalars:scalars"


def drift_from_catalog(client, seed: int, conditions: list[str], segments: list[str] | None = None) -> pd.DataFrame:
    """One row per (segment, condition): drift, peak step, n_steps, all from the catalog."""
    ds = client.get_dataset(S.DATASET_DEMOS)
    if segments:
        ds = ds.filter_segments(segments)
    entities = [f"/policy/seed{seed}/clean_pred/action0"] + [f"/policy/seed{seed}/{c}/action0" for c in conditions]
    t = ds.filter_contents(entities).reader(index=S.TIMELINE_STEP).to_arrow_table()
    seg = np.asarray(t["rerun_segment_id"].to_pylist())
    step = np.asarray(t[S.TIMELINE_STEP].to_pylist())
    clean = t[_col("clean_pred", seed)].to_pylist()
    rows = []
    for c in conditions:
        if _col(c, seed) not in t.column_names:
            continue
        cond = t[_col(c, seed)].to_pylist()
        ok = np.array([a is not None and b is not None for a, b in zip(clean, cond)])
        if not ok.any():
            continue
        d = np.linalg.norm(np.asarray([cond[i] for i in np.nonzero(ok)[0]]) -
                           np.asarray([clean[i] for i in np.nonzero(ok)[0]]), axis=1)
        df = pd.DataFrame({"segment_id": seg[ok], "step": step[ok], "delta": d})
        g = df.groupby("segment_id")
        out = g.delta.mean().rename("drift").to_frame()
        out["n_steps"] = g.size()
        out["peak_step"] = df.loc[g.delta.idxmax(), ["segment_id", "step"]].set_index("segment_id").step
        rows.append(out.reset_index().assign(condition=c))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cross_check(q: pd.DataFrame, seed: int, ref_dir: Path | None = None) -> bool:
    ref = pd.read_parquet((ref_dir or ROOT / "results" / "drift") / f"seed{seed}" / "candidates.parquet")
    if "segment_id" not in ref:
        ref["segment_id"] = ref.episode.map(S.demo_segment_id)
    m = ref.merge(q, on=["segment_id", "condition"], suffixes=("_ref", "_query"))
    ok = len(m) == len(q) and np.allclose(m.drift_ref, m.drift_query, atol=1e-5)
    print(f"drift query vs numpy reference: {len(m)} candidates, max |diff| "
          f"{np.abs(m.drift_ref - m.drift_query).max():.2e} -> {'match' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--ref_dir", default=None)
    args = parser.parse_args()
    with hub.open_catalog() as client:
        if args.conditions:
            conds = args.conditions
        else:
            st = client.get_dataset(S.DATASET_DEMOS).segment_table().to_pandas()
            conds = sorted({l for ls in st.rerun_layer_names for l in ls} -
                           {"base", "qa", "scene3d", "clean_pred"} - {l for ls in st.rerun_layer_names for l in ls if l.startswith("scene3d_")})
        q = drift_from_catalog(client, args.seed, conds)
    print(q.groupby("condition").drift.mean().sort_values(ascending=False).round(4).to_string())
    cross_check(q, args.seed, Path(args.ref_dir) if args.ref_dir else None)


if __name__ == "__main__":
    main()
