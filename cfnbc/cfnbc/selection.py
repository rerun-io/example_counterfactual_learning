"""Repair-set selection over the drift candidate pool (paper §III-C, App. A), stored as named catalog selections.

Candidates come from the catalog table `drift_candidates` (one row per (seed, demo segment, condition): drift d_i and
56-dim response feature r_i). Methods:
  random            K candidates uniformly at random (rng seed recorded)
  top_drift         the K highest-drift candidates
  drift_weighted    greedy maximization of drift-weighted response coverage:
                      r~ = standardized r;  A_ij = exp(-||r~_i - r~_j||^2 / (2 sigma^2)), sigma = median non-zero
                      pairwise distance;  d~ = d / mean(d);  m_j = max_{l in R} A_lj (0 for empty R);
                      g(i | R) = d~_i * sum_j d~_j * (max(m_j, A_ij) - m_j);  add argmax_i g until |R| = K.

    python -m cfnbc.selection --seed 0 --k 20 --methods random drift_weighted [--rng 0]

Each selection is appended to the catalog table `repair_selections` (selection, method, k, policy seed, rank, segment,
condition, drift, params, pool size, created_at) and written to results/selection/<selection>.csv.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cfnbc import hub

ROOT = Path(__file__).resolve().parents[2]


def load_pool(client, seed: int | None, conditions: list[str] | None = None, table: str = "drift_candidates",
              exclude_segments: list[str] | None = None) -> pd.DataFrame:
    df = client.get_table(table).reader().to_pandas()
    if seed is not None and "seed" in df:
        df = df[df.seed == seed]
    if exclude_segments:
        df = df[~df.segment_id.isin(exclude_segments)]
    if conditions:
        df = df[df.condition.isin(conditions)]
    return df.reset_index(drop=True)


def select_random(pool: pd.DataFrame, k: int, rng: int) -> list[int]:
    return list(np.random.default_rng(rng).choice(len(pool), size=k, replace=False))


def select_top_drift(pool: pd.DataFrame, k: int) -> list[int]:
    return list(np.argsort(-pool.drift.to_numpy())[:k])


def affinity(responses: np.ndarray) -> tuple[np.ndarray, float]:
    r = np.stack(responses).astype(np.float64)
    r = (r - r.mean(0)) / np.where(r.std(0) > 0, r.std(0), 1.0)
    sq = ((r[:, None, :] - r[None, :, :]) ** 2).sum(-1)
    dist = np.sqrt(sq)
    sigma = float(np.median(dist[dist > 0]))
    return np.exp(-sq / (2 * sigma ** 2)), sigma


def select_drift_weighted(pool: pd.DataFrame, k: int) -> tuple[list[int], dict]:
    A, sigma = affinity(pool.response.to_list())
    d = pool.drift.to_numpy()
    dn = d / d.mean()
    m = np.zeros(len(pool))
    chosen: list[int] = []
    for _ in range(k):
        gain = dn * ((np.maximum(m[None, :], A) - m[None, :]) * dn[None, :]).sum(1)
        gain[chosen] = -np.inf
        i = int(np.argmax(gain))
        chosen.append(i)
        m = np.maximum(m, A[i])
    coverage = float((dn * m).sum() / dn.sum())
    return chosen, dict(sigma=sigma, weighted_coverage=coverage)


def coverage_of(pool: pd.DataFrame, idx: list[int]) -> float:
    A, _ = affinity(pool.response.to_list())
    dn = pool.drift.to_numpy() / pool.drift.mean()
    return float((dn * A[idx].max(0)).sum() / dn.sum())


def store(client, name: str, method: str, k: int, seed: int, pool: pd.DataFrame, idx: list[int], params: dict) -> pd.DataFrame:
    import pyarrow as pa
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sel = pool.iloc[idx]
    df = pd.DataFrame(dict(selection=name, method=method, k=k, policy_seed=seed, rank=np.arange(len(idx)),
                           segment_id=sel.segment_id.to_numpy(), condition=sel.condition.to_numpy(),
                           family=sel.family.to_numpy(), drift=sel.drift.to_numpy(), peak_step=sel.peak_step.to_numpy(),
                           params=json.dumps(params), pool_size=len(pool), created_at=now))
    out = ROOT / "results" / "selection"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    t = hub.ensure_table(client, "repair_selections", tbl.schema)
    existing = t.reader().to_pandas()
    if (existing.get("selection", pd.Series(dtype=str)) == name).any():
        raise SystemExit(f"selection {name} already stored; choose a new name")
    t.append(tbl.to_batches())
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="nominal policy seed whose drift pool to use")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=["random", "drift_weighted"])
    parser.add_argument("--rng", type=int, default=0)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--table", default="drift_candidates", help="candidate table (real data: real_drift_candidates)")
    parser.add_argument("--prefix", default=None, help="selection name prefix (default s<seed>)")
    parser.add_argument("--exclude_heldout", action="store_true",
                        help="drop the 10 validation episodes of upstream's split (set_seed(1)) from the pool")
    args = parser.parse_args()
    exclude = None
    if args.exclude_heldout:
        from cfnbc.real.offline_eval import heldout_segments
        exclude = heldout_segments()
    with hub.open_catalog() as client:
        pool = load_pool(client, args.seed, args.conditions, args.table, exclude)
        print(f"pool: {len(pool)} candidates (seed {args.seed}), mean drift {pool.drift.mean():.3f}")
        for method in args.methods:
            if method == "random":
                idx, params = select_random(pool, args.k, args.rng), dict(rng=args.rng)
            elif method == "top_drift":
                idx, params = select_top_drift(pool, args.k), {}
            elif method == "drift_weighted":
                idx, params = select_drift_weighted(pool, args.k)
            else:
                raise SystemExit(f"unknown method {method}")
            params["weighted_coverage"] = coverage_of(pool, idx)
            name = f"{args.prefix or f's{args.seed}'}_{method}_k{args.k}_{args.tag}"
            df = store(client, name, method, args.k, args.seed, pool, idx, params)
            print(f"{name}: mean drift {df.drift.mean():.3f}, coverage {params['weighted_coverage']:.3f}, "
                  f"families {df.family.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
