"""Load a trained ACT policy outside imitate_episodes.py and query it in normalized action space."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ACT_DIR = Path(__file__).resolve().parents[2] / "act"

# README / train_seeds.sh settings; enc/dec/nheads/backbone are fixed in imitate_episodes.main
POLICY_CONFIG = dict(lr=1e-5, num_queries=100, kl_weight=10, hidden_dim=512, dim_feedforward=3200,
                     lr_backbone=1e-5, backbone="resnet18", enc_layers=4, dec_layers=7, nheads=8,
                     camera_names=["top"])


def build_act_policy(ckpt_dir: str | Path, device: str = "mps"):
    """Construct an untrained ACTPolicy. detr's arg parser re-parses sys.argv and has required flags."""
    if str(ACT_DIR) not in sys.path:
        sys.path.insert(0, str(ACT_DIR))
    from policy import ACTPolicy
    argv = sys.argv
    sys.argv = [argv[0], "--ckpt_dir", str(ckpt_dir), "--policy_class", "ACT", "--task_name",
                "sim_transfer_cube_scripted", "--seed", "0", "--num_epochs", "1", "--device", device]
    try:
        return ACTPolicy(dict(POLICY_CONFIG))
    finally:
        sys.argv = argv


class ACTRunner:
    def __init__(self, ckpt_dir: str | Path, ckpt_name: str = "policy_last.ckpt", device: str = "mps"):
        self.policy = build_act_policy(ckpt_dir, device)
        self.device = torch.device(device)
        state = torch.load(Path(ckpt_dir) / ckpt_name, map_location=self.device)
        print(self.policy.load_state_dict(state))
        self.policy.to(self.device).eval()
        with open(Path(ckpt_dir) / "dataset_stats.pkl", "rb") as f:
            self.stats = pickle.load(f)

    @torch.inference_mode()
    def predict_normalized(self, qpos: np.ndarray, images: np.ndarray) -> np.ndarray:
        """qpos (B,14) raw, images (B,H,W,3) uint8 top camera -> (B, chunk, 14) normalized actions N(a)."""
        q = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        q = torch.from_numpy(q).float().to(self.device)
        im = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float().div(255.0).unsqueeze(1)
        return self.policy(q, im).float().cpu().numpy()
