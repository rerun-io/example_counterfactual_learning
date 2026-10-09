"""Train ACT on the real ALOHA teleop demos with upstream ACT's training loop (same recipe as the sim baseline).

    python -m cfnbc.real.train_real [--seed 0] [--epochs 2000]

Dataset: data_real/act_cups_open (cfnbc.real.to_hdf5), camera cam_high only (as in our sim setup), 50 episodes x 400
steps, README hyperparameters (chunk 100, kl 10, hidden 512, ff 3200, lr 1e-5, batch 8). Train/val split as upstream
(set_seed(1) before load_data). Checkpoints in ckpts/real_cups_s{seed}/; scalars to cube_training (run real_cups_s{seed}).
No closed-loop eval: there is no robot, so the offline audit and held-out losses stand in.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch

from cfnbc.policy_io import ACT_DIR, POLICY_CONFIG
from cfnbc import schema as S
from cfnbc.real.ingest import ROOT

DATA = S.real_data_dir() / "act_cups_open"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--device", default=os.environ.get("ACT_DEVICE", "mps"))
    args = p.parse_args()
    ckpt_dir = ROOT / "ckpts" / f"real_cups_s{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ACT_RUN_NAME"] = f"real_cups_s{args.seed}"
    if str(ACT_DIR) not in sys.path:
        sys.path.insert(0, str(ACT_DIR))
    argv = sys.argv  # detr's parser re-parses sys.argv (required flags)
    sys.argv = [argv[0], "--ckpt_dir", str(ckpt_dir), "--policy_class", "ACT", "--task_name", "real_cups_open",
                "--seed", str(args.seed), "--num_epochs", str(args.epochs), "--device", args.device]
    import imitate_episodes as ie
    from utils import load_data, set_seed

    policy_config = dict(POLICY_CONFIG, camera_names=["cam_high"])
    config = dict(num_epochs=args.epochs, ckpt_dir=str(ckpt_dir), episode_len=400, state_dim=14, lr=POLICY_CONFIG["lr"],
                  policy_class="ACT", onscreen_render=False, policy_config=policy_config, task_name="real_cups_open",
                  seed=args.seed, temporal_agg=True, camera_names=["cam_high"], real_robot=True,
                  device=torch.device(args.device))
    set_seed(1)
    train_loader, val_loader, stats, _ = load_data(str(DATA), 50, ["cam_high"], 8, 8)
    with open(ckpt_dir / "dataset_stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    best_epoch, min_val, best_state = ie.train_bc(train_loader, val_loader, config)
    torch.save(best_state, ckpt_dir / "policy_best.ckpt")
    sys.argv = argv
    print(f"done: best val {min_val:.5f} @ epoch {best_epoch} -> {ckpt_dir}")


if __name__ == "__main__":
    main()
