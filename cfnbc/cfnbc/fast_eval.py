"""Vectorized ACT evaluation: N environments in lockstep, one batched policy forward pass per step.

Reproduces act/imitate_episodes.py:eval_bc semantics per environment:
  - initial cube poses: set_seed(eval_seed), then sample_box_pose() for all rollouts in order (same sequence)
  - qpos/image preprocessing, temporal ensembling (k=0.01, oldest prediction weighted highest; query every step)
    or chunk replay (query every chunk_size steps), post-processing with dataset_stats.pkl
  - success = max reward over the episode == env max reward (upstream metric), episode_return = sum of rewards
Outputs the same rollouts_<tag>.csv / result_<tag>.txt as upstream, and Rerun rollout recordings (ACT_RERUN=1).

    python -m cfnbc.fast_eval --seed 0 [--nuisance distractor_L2] [--num_envs 16] [--amp none|fp16|bf16]
                              [--num_rollouts 100] [--no_temporal_agg] [--out_dir DIR] [--run_name NAME]

Defaults write next to the checkpoint (ckpts/seed{S}/) with run name seed{S}, like scripts/eval_nuisance.sh.
Numerics: batching changes float summation order (≈1e-6), and --amp changes precision. Validate against the frozen
upstream results with --validate_against before trusting a configuration (pose-by-pose outcome agreement).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "act"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="training seed -> ckpts/seed{S}")
    parser.add_argument("--ckpt_dir", default=None, help="checkpoint dir (default ckpts/seed{S}), e.g. a repaired policy")
    parser.add_argument("--ckpt_name", default="policy_last.ckpt")
    parser.add_argument("--nuisance", default=None)
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--no_temporal_agg", action="store_true")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
    parser.add_argument("--device", default=os.environ.get("ACT_DEVICE", "mps"))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--validate_against", default=None, help="upstream rollouts_*.csv to compare outcomes with")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else ROOT / "ckpts" / f"seed{args.seed}"
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (ckpt_dir.name if args.ckpt_dir else f"seed{args.seed}")
    temporal_agg = not args.no_temporal_agg
    os.environ["ACT_RUN_NAME"] = run_name
    os.environ.setdefault("ACT_RENDER_CAMS", "top")
    if args.nuisance:
        os.environ["ACT_NUISANCE"] = args.nuisance
    else:
        os.environ.pop("ACT_NUISANCE", None)
    if str(ACT_DIR) not in sys.path:
        sys.path.insert(0, str(ACT_DIR))
    import sim_env
    from sim_env import BOX_POSE, make_sim_env
    from utils import sample_box_pose, set_seed
    from cfnbc.act_hooks import RolloutLogger, enabled as rerun_enabled
    from cfnbc.policy_io import build_act_policy

    device = torch.device(args.device)
    # same RNG order as eval_bc: set_seed, build/load policy, then draw all init poses
    set_seed(args.eval_seed)
    policy = build_act_policy(ckpt_dir, args.device)
    print(policy.load_state_dict(torch.load(ckpt_dir / args.ckpt_name, map_location=device)))
    policy.to(device).eval()
    with open(ckpt_dir / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    init_poses = [sample_box_pose() for _ in range(args.num_rollouts)]

    num_queries, state_dim, max_t = 100, 14, 400
    query_frequency = 1 if temporal_agg else num_queries
    qpos_mean, qpos_std = stats["qpos_mean"], stats["qpos_std"]
    a_mean = torch.tensor(stats["action_mean"], device=device, dtype=torch.float32)
    a_std = torch.tensor(stats["action_std"], device=device, dtype=torch.float32)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)
    config = dict(ckpt_dir=str(ckpt_dir), temporal_agg=temporal_agg, eval_seed=args.eval_seed)

    n_envs = min(args.num_envs, args.num_rollouts)
    envs = [make_sim_env("sim_transfer_cube_scripted") for _ in range(n_envs)]
    env_max_reward = envs[0].task.max_reward
    results = [None] * args.num_rollouts
    t_start = time.time()

    for b0 in range(0, args.num_rollouts, n_envs):
        ids = list(range(b0, min(b0 + n_envs, args.num_rollouts)))
        n = len(ids)
        ts_list, loggers = [], []
        for k, rid in enumerate(ids):
            BOX_POSE[0] = init_poses[rid]
            ts_list.append(envs[k].reset())
            lg = RolloutLogger(config, args.ckpt_name, nuisance=args.nuisance or "clean") if rerun_enabled() else None
            if lg:
                lg.start(rid, init_poses[rid])
            loggers.append(lg)
        all_time = torch.zeros([n, max_t, max_t + num_queries, state_dim], device=device) if temporal_agg else None
        rewards = np.zeros((n, max_t))
        with torch.inference_mode():
            for t in range(max_t):
                obs = [ts.observation for ts in ts_list]
                qpos_np = np.stack([np.asarray(o["qpos"]) for o in obs])
                qpos = torch.from_numpy((qpos_np - qpos_mean) / qpos_std).float().to(device)
                img = torch.from_numpy(np.stack([o["images"]["top"] for o in obs])).to(device)
                img = img.permute(0, 3, 1, 2).float().div(255.0).unsqueeze(1)  # (n, 1 cam, 3, H, W)
                chunk_cpu = None
                if t % query_frequency == 0:
                    if amp_dtype is not None:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            all_actions = policy(qpos, img).float()
                    else:
                        all_actions = policy(qpos, img)
                    if any(loggers):
                        chunk_cpu = (all_actions * a_std + a_mean).cpu().numpy()
                if temporal_agg:
                    all_time[:, t, t:t + num_queries] = all_actions
                    cand = all_time[:, :, t]                                   # (n, max_t, 14)
                    populated = torch.all(cand != 0, dim=-1)                   # (n, max_t)
                    order = torch.cumsum(populated.to(torch.int64), dim=1) - 1  # index among populated rows
                    w = torch.exp(-0.01 * order.to(torch.float32)) * populated
                    w = w / w.sum(dim=1, keepdim=True)
                    raw = (cand * w.unsqueeze(-1)).sum(dim=1)                  # (n, 14)
                else:
                    raw = all_actions[:, t % query_frequency]
                actions = (raw * a_std + a_mean).cpu().numpy()
                new_ts = []
                for k in range(n):
                    ts = envs[k].step(actions[k])
                    rewards[k, t] = ts.reward
                    if loggers[k]:
                        loggers[k].step(t, obs[k], envs[k].physics, actions[k],
                                        chunk_cpu[k] if chunk_cpu is not None else None, ts.reward)
                    new_ts.append(ts)
                ts_list = new_ts
        for k, rid in enumerate(ids):
            hr = int(rewards[k].max())
            results[rid] = dict(rollout_id=rid, init_pose=" ".join(f"{x:.8f}" for x in init_poses[rid]),
                                episode_return=float(rewards[k].sum()), highest_reward=hr,
                                success=int(hr == env_max_reward))
            if loggers[k]:
                loggers[k].end(hr, hr == env_max_reward)
        done = ids[-1] + 1
        el = time.time() - t_start
        print(f"rollouts {done}/{args.num_rollouts}  {el:.0f}s  ({el / done:.2f} s/rollout)  "
              f"success so far {np.mean([r['success'] for r in results[:done]]):.3f}", flush=True)

    df = pd.DataFrame(results)
    tag = (args.ckpt_name.split(".")[0] + ("_tagg" if temporal_agg else "_notagg")
           + f"_seed{args.eval_seed}_n{args.num_rollouts}" + (f"_{args.nuisance}" if args.nuisance else ""))
    df.to_csv(out_dir / f"rollouts_{tag}.csv", index=False, float_format="%g")
    sr = df.success.mean()
    with open(out_dir / f"result_{tag}.txt", "w") as f:
        f.write(f"\nSuccess rate: {sr}\nAverage return: {df.episode_return.mean()}\n\n")
        for r in range(env_max_reward + 1):
            c = int((df.highest_reward >= r).sum())
            f.write(f"Reward >= {r}: {c}/{len(df)} = {100 * c / len(df)}%\n")
        f.write(f"\nfast_eval: num_envs={n_envs} amp={args.amp} device={args.device} "
                f"wall={time.time() - t_start:.0f}s\n")
    print(f"{args.ckpt_name}: success_rate={sr} ({time.time() - t_start:.0f}s) -> {out_dir}/rollouts_{tag}.csv")

    if args.validate_against:
        ref = pd.read_csv(args.validate_against)
        same_pose = (ref.init_pose.str.split().map(lambda v: np.array(v, float)).apply(np.round, args=(6,)).map(tuple)
                     == df.init_pose.str.split().map(lambda v: np.array(v, float)).apply(np.round, args=(6,)).map(tuple))
        agree = (ref.success.values == df.success.values)
        print(f"validation vs {args.validate_against}: init poses identical {bool(same_pose.all())}, "
              f"success agree {int(agree.sum())}/{len(df)}, highest_reward agree "
              f"{int((ref.highest_reward.values == df.highest_reward.values).sum())}/{len(df)}, "
              f"rate ref {ref.success.mean():.3f} vs fast {sr:.3f}")


if __name__ == "__main__":
    main()
