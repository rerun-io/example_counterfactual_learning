"""Offline robustness evaluation for real-data policies (no robot): held-out BC error on clean vs edited frames.

Closed-loop success needs the robot. Offline, the best available signal is how well a policy reproduces the human's
action chunk on *held-out* episodes, from the clean frame and from each task-preserving edit of it (the expert action is
unchanged by construction). A robust policy has a small gap between clean and edited error.

    python -m cfnbc.real.offline_eval --policies real_cups_s0 repair_real_drift_weighted_k20_v1 ...

Held-out = upstream's 10 validation episodes (set_seed(1) then np.random.permutation(50)[40:], as in load_data); they
are excluded from repair selection. Frames and actions are read from the catalog (cfnbc.real.repair_real). Writes the
table `real_offline_eval` and results/real/offline_eval.csv: per (policy, condition): masked L1 (normalized), and the
gap to the clean L1, with a paired bootstrap 95% CI on the gap (per-sample errors in
results/real/offline_eval_samples.csv).
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch

from cfnbc import hub
from cfnbc.policy_io import build_act_policy
from cfnbc.real.edits import DEFAULT_CONDITIONS
from cfnbc.real.ingest import ROOT, segment_id
from cfnbc.real.repair_real import RealHubSamples


def heldout_segments() -> list[str]:
    rng = np.random.RandomState(1)  # np.random.seed(1) state, as set_seed(1) leaves it before load_data's permutation
    return [segment_id(int(e)) for e in rng.permutation(50)[40:]]


def evaluate(client, ckpt_dir, stats, conditions, steps, device) -> tuple[list[dict], dict]:
    policy = build_act_policy(ckpt_dir, device)
    policy.load_state_dict(torch.load(ckpt_dir / "policy_last.ckpt", map_location=device))
    policy.to(device).eval()
    rows, per_sample = [], {}
    for cond in ["clean"] + conditions:
        ent = "/camera/cam_high_clean" if cond == "clean" else f"/camera/cam_high_{cond}"
        ds = RealHubSamples(client, [(s, t, ent) for s in heldout_segments() for t in steps], stats, cache=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False, num_workers=0)
        errs = []
        with torch.inference_mode():
            for img, qpos, action, is_pad in loader:
                a_hat = policy(qpos.to(device), img.to(device))                      # (B, 100, 14), z = prior mean
                m = (~is_pad).unsqueeze(-1).float()
                errs += ((torch.abs(a_hat.cpu() - action) * m).sum((1, 2)) / m.sum((1, 2)) / 14).tolist()
        per_sample[cond] = np.asarray(errs)
        rows.append(dict(condition=cond, l1=float(np.mean(errs)), samples=len(errs)))
    clean = rows[0]["l1"]
    rng = np.random.default_rng(0)
    for r in rows:
        r["gap_vs_clean"] = r["l1"] - clean
        d = per_sample[r["condition"]] - per_sample["clean"]          # paired: same held-out samples
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        r["gap_ci_lo"], r["gap_ci_hi"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return rows, per_sample


def main() -> None:
    import pyarrow as pa
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True, help="checkpoint dir names under ckpts/")
    p.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--device", default=os.environ.get("ACT_DEVICE", "mps"))
    args = p.parse_args()
    steps = list(range(args.stride, 400, args.stride))
    with open(ROOT / "ckpts" / "real_cups_s0" / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)   # all real policies share theta_0's normalization
    out, samples = [], []
    with hub.open_catalog() as client:
        for name in args.policies:
            rows, per_sample = evaluate(client, ROOT / "ckpts" / name, stats, args.conditions, steps, args.device)
            out += [dict(policy=name, device=args.device, **r) for r in rows]
            samples += [dict(policy=name, condition=c, sample=i, l1=float(v)) for c, e in per_sample.items() for i, v in enumerate(e)]
            print(pd.DataFrame([o for o in out if o["policy"] == name]).round(4).to_string(index=False), flush=True)
        df = pd.DataFrame(out)
        (ROOT / "results" / "real").mkdir(parents=True, exist_ok=True)
        df.to_csv(ROOT / "results" / "real" / "offline_eval.csv", index=False)
        pd.DataFrame(samples).to_csv(ROOT / "results" / "real" / "offline_eval_samples.csv", index=False)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if "real_offline_eval" in client.table_names():
            client.get_table("real_offline_eval").overwrite(tbl.to_batches())
        else:
            hub.ensure_table(client, "real_offline_eval", tbl.schema).append(tbl.to_batches())
    print(df.pivot_table(index="condition", columns="policy", values="l1").round(4))


if __name__ == "__main__":
    main()
