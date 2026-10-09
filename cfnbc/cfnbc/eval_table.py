"""Aggregate per-rollout eval CSVs into results/eval.csv (per-seed rates + 3-seed mean per config).

    python -m cfnbc.eval_table [--ckpts_root ckpts] [--out results/eval.csv] [--check_rerun]

Source of truth is ckpts/seed*/rollouts_<ckpt>_<tagg|notagg>_seed<S>_n<N>[_<nuisance>].csv written by
imitate_episodes.py (upstream success: max reward reached at any step). success_rate_strict comes from
results/qa/rollout_qa.csv (stable stage 4 and cube still held at the end; see cfnbc.qa).
--check_rerun starts a local server over <rerun_root> and asserts the cube_rollouts dataset gives the same rates.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PAT = re.compile(r"rollouts_(?P<ckpt>.+?)_(?P<tagg>tagg|notagg)_seed(?P<eval_seed>\d+)_n(?P<n>\d+)(?:_(?P<nuisance>\w+))?\.csv$")
TARGET = 0.96  # paper Table I, A50 clean success


_ROLLOUT_QA: pd.DataFrame | None = None


def _strict_rates(run: str, m) -> float:
    """Strict success rate (cfnbc.qa: stable stage 4 and cube still held at the end) from results/qa/rollout_qa.csv;
    NaN if QA hasn't been run for these rollouts (`python -m cfnbc.qa --rollouts`)."""
    global _ROLLOUT_QA
    if _ROLLOUT_QA is None:
        p = Path(__file__).resolve().parents[2] / "results" / "qa" / "rollout_qa.csv"
        _ROLLOUT_QA = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["segment_id", "success_strict"])
    tagg = m["tagg"]
    prefix = f"{run}__{m['ckpt'].removeprefix('policy_')}__{tagg}__es{m['eval_seed']}__{m['nuisance'] or 'clean'}__r"
    sel = _ROLLOUT_QA[_ROLLOUT_QA.segment_id.str.startswith(prefix)] if "success_strict" in _ROLLOUT_QA else []
    return float(sel.success_strict.mean()) if len(sel) == int(m["n"]) else float("nan")


def per_seed(ckpts_root: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(ckpts_root.glob("seed*/rollouts_*.csv")):
        m = PAT.search(csv.name)
        if not m:
            continue
        df = pd.read_csv(csv)
        strict = _strict_rates(csv.parent.name, m)
        rows.append(dict(train_seed=int(csv.parent.name.removeprefix("seed")), ckpt=m["ckpt"], nuisance=m["nuisance"] or "clean",
                         temporal_agg=int(m["tagg"] == "tagg"), eval_seed=int(m["eval_seed"]), n=len(df),
                         successes=int(df["success"].sum()), success_rate=df["success"].mean(),
                         success_rate_strict=strict))
    return pd.DataFrame(rows)


def summarize(ps: pd.DataFrame) -> pd.DataFrame:
    keys = ["ckpt", "temporal_agg", "eval_seed", "n", "nuisance"]
    agg = (ps.groupby(keys).agg(num_seeds=("train_seed", "nunique"), success_rate=("success_rate", "mean"),
                                std=("success_rate", "std"), success_rate_strict=("success_rate_strict", "mean"))
           .reset_index())
    agg["train_seed"] = "mean"
    agg["target"] = agg["nuisance"].map(lambda n: TARGET if n == "clean" else float("nan"))  # Table I clean only
    agg["delta_vs_target"] = agg["success_rate"] - agg["target"]
    out = pd.concat([ps.assign(train_seed=ps["train_seed"].astype(str)), agg], ignore_index=True)
    return out.sort_values(keys + ["train_seed"]).reset_index(drop=True)


def check_rerun(ps: pd.DataFrame) -> None:
    from cfnbc import query as Q
    with Q.local_server(port=0) as srv:
        rt = Q.rollout_table(srv.client())
    rt = rt[rt["run"].str.match(r"seed\d+$")]
    rt = rt.assign(train_seed=rt["run"].str.removeprefix("seed").astype(int),
                   ckpt=rt["ckpt_name"].str.removesuffix(".ckpt"))
    rr_rates = rt.groupby(["train_seed", "ckpt", "temporal_agg", "eval_seed", "nuisance"])["success"].agg(["mean", "size"])
    merged = ps.set_index(["train_seed", "ckpt", "temporal_agg", "eval_seed", "nuisance"]).join(rr_rates, how="left")
    bad = merged[(merged["mean"] != merged["success_rate"]) | (merged["size"] != merged["n"])]
    print(merged[["n", "success_rate", "size", "mean"]].rename(columns={"size": "rerun_n", "mean": "rerun_rate"}))
    print("Rerun catalog matches CSVs" if bad.empty else f"MISMATCH in {len(bad)} configs")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts_root", default=str(root / "ckpts"))
    parser.add_argument("--out", default=str(root / "results" / "eval.csv"))
    parser.add_argument("--check_rerun", action="store_true")
    args = parser.parse_args()
    ps = per_seed(Path(args.ckpts_root))
    if ps.empty:
        raise SystemExit(f"no rollouts_*.csv under {args.ckpts_root}/seed*/")
    out = summarize(ps)
    out.to_csv(args.out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out)
    print(f"wrote {args.out}")
    if args.check_rerun:
        check_rerun(ps)


if __name__ == "__main__":
    main()
