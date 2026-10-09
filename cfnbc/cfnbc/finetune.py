"""Counterfactual repair fine-tuning (paper Eq. 6) with the repair set streamed from the Rerun catalog.

    theta_repair = argmin  L_BC(theta; D_clean) + lambda_cf * L_BC(theta; D_cf(R_K))

Starts from the nominal checkpoint theta_0 (ckpts/seed{S}/policy_last.ckpt) and keeps its normalization stats.
D_clean: upstream's EpisodicDataset over the same 40 training demos (same split: set_seed(1) before load_data, as in
imitate_episodes.main). D_cf: the selection's counterfactual frames + clean expert actions from the catalog
(cfnbc.repair_data.HubSamples). Each optimizer step uses one clean batch and one counterfactual batch of the same size.

    python -m cfnbc.finetune --selection s0_drift_weighted_k20_v1 [--epochs 800] [--lambda_cf 0.5]

Writes ckpts/repair_<selection>/{policy_last.ckpt, dataset_stats.pkl, train.log} and logs scalars to cube_training
(run repair_<selection>) when ACT_RERUN=1.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cfnbc import hub
from cfnbc import schema as S
from cfnbc.policy_io import ACT_DIR, POLICY_CONFIG, build_act_policy
from cfnbc.repair_data import HubSamples, load_stats, selection_items

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--seed", type=int, default=0, help="nominal policy seed (theta_0) and fine-tune seed")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lambda_cf", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=os.environ.get("ACT_DEVICE", "mps"))
    parser.add_argument("--real", action="store_true", help="real teleop data: theta_0 = ckpts/real_cups_s{seed}, "
                        "clean demos from data_real/act_cups_open (cam_high), repair set from real_demos layers")
    parser.add_argument("--clean_source", choices=["hdf5", "hub"], default="hdf5",
                        help="hub: stream the clean demos from cube_demos with RerunIterableDataset (cfnbc.clean_stream) "
                             "instead of ACT's HDF5 loader; each epoch draws as many clean batches as the HDF5 loader has")
    args = parser.parse_args()

    if str(ACT_DIR) not in sys.path:
        sys.path.insert(0, str(ACT_DIR))
    from utils import load_data, set_seed, compute_dict_mean, detach_dict
    from cfnbc.act_hooks import TrainLogger, _NoOp, enabled as rerun_enabled

    base = ROOT / "ckpts" / (f"real_cups_s{args.seed}" if args.real else f"seed{args.seed}")
    if args.real and args.clean_source == "hub":
        raise SystemExit("--clean_source hub is sim-only for now (real_demos stores cam_high as AV1 video)")
    out = ROOT / "ckpts" / (f"repair_{args.selection}" + ("_hubclean" if args.clean_source == "hub" else ""))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / "dataset_stats.pkl", out / "dataset_stats.pkl")
    stats = load_stats(base)
    device = torch.device(args.device)

    # clean stream: identical split to the nominal training run
    set_seed(1)
    if args.real:
        cfg = dict(dataset_dir=str(S.real_data_dir() / "act_cups_open"), num_episodes=50, camera_names=["cam_high"])
    else:
        from constants import SIM_TASK_CONFIGS
        cfg = SIM_TASK_CONFIGS["sim_transfer_cube_scripted"]
    train_loader, val_loader, _, _ = load_data(cfg["dataset_dir"], cfg["num_episodes"], cfg["camera_names"],
                                               args.batch_size, args.batch_size)

    stack = contextlib.ExitStack()
    client = stack.enter_context(hub.open_catalog())   # kept open for the whole run: lazy catalog streams need the client
    items = selection_items(client, args.selection)
    if args.real:
        from cfnbc.real.repair_real import RealHubSamples
        items = [(s, t, e.replace("/camera/top_", "/camera/cam_high_")) for s, t, e in items]
        cf_ds = RealHubSamples(client, items, stats, cache=True)
    else:
        cf_ds = HubSamples(client, items, stats, cache=True)
    t0 = time.time()
    for i in range(len(cf_ds)):  # stream the repair set from the catalog once
        cf_ds[i]
    print(f"repair set: {len(cf_ds)} samples from {len({(s, e) for s, _, e in items})} candidates, "
          f"streamed from the catalog in {time.time() - t0:.0f}s", flush=True)

    set_seed(args.seed)
    cf_loader = torch.utils.data.DataLoader(cf_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    cf_iter = itertools.cycle(cf_loader)
    clean_batches_per_epoch = len(train_loader)
    if args.clean_source == "hub":
        import rerun.experimental.dataloader as dl
        from cfnbc.clean_stream import CleanStream, train_segments
        clean_ds = CleanStream(client.get_dataset(S.DATASET_DEMOS), train_segments(), stats,
                               shuffle=dl.BlockShuffle(buffer_size=512))
        clean_loader = torch.utils.data.DataLoader(clean_ds, batch_size=args.batch_size, num_workers=0)

        def clean_cycle():
            ep = 0
            while True:
                clean_ds.set_epoch(ep)
                yield from clean_loader
                ep += 1
        clean_iter = clean_cycle()
        print(f"clean demos: streamed from the catalog ({len(train_segments())} episodes x 400 steps, "
              f"{clean_batches_per_epoch} batches per epoch)", flush=True)

    policy = build_act_policy(base, args.device)
    print(policy.load_state_dict(torch.load(base / "policy_last.ckpt", map_location=device)))
    policy.to(device)
    optimizer = policy.configure_optimizers()
    config = dict(ckpt_dir=str(out), seed=args.seed, num_epochs=args.epochs, lr=POLICY_CONFIG["lr"],
                  policy_class="ACT", task_name="sim_transfer_cube_scripted", device=args.device,
                  camera_names=POLICY_CONFIG["camera_names"], policy_config=POLICY_CONFIG)
    os.environ["ACT_RUN_NAME"] = f"repair_{args.selection}"
    tlog = (TrainLogger(config, extra_props=dict(selection=args.selection, lambda_cf=args.lambda_cf, base=str(base),
                                                 cf_samples=len(cf_ds))) if rerun_enabled() else _NoOp())

    def fwd(batch):
        img, qpos, action, is_pad = (x.to(device) for x in batch)
        return policy(qpos, img, action, is_pad)

    t0 = time.time()
    for epoch in range(args.epochs):
        with torch.inference_mode():
            policy.eval()
            val = compute_dict_mean([fwd(b) for b in val_loader])
        tlog.log_epoch(epoch, "val", val)
        policy.train()
        hist = []
        clean_batches = (train_loader if args.clean_source == "hdf5"
                         else (next(clean_iter) for _ in range(clean_batches_per_epoch)))
        for clean_batch in clean_batches:
            lc = fwd(clean_batch)
            lcf = fwd(next(cf_iter))
            loss = lc["loss"] + args.lambda_cf * lcf["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rec = {**{f"clean_{k}": v for k, v in detach_dict(lc).items()},
                   **{f"cf_{k}": v for k, v in detach_dict(lcf).items()}, "loss": loss.detach()}
            hist.append(rec)
            tlog.log_batch(rec)
        summ = compute_dict_mean(hist)
        tlog.log_epoch(epoch, "train", summ)
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            el = time.time() - t0
            print(f"epoch {epoch}: loss {summ['loss']:.4f} (clean {summ['clean_loss']:.4f}, cf {summ['cf_loss']:.4f}), "
                  f"val {val['loss']:.4f}  {el / (epoch + 1):.1f}s/epoch, ~{el / (epoch + 1) * (args.epochs - epoch - 1) / 60:.0f} min left",
                  flush=True)
    torch.save(policy.state_dict(), out / "policy_last.ckpt")
    tlog.log_best(args.epochs - 1, float(val["loss"]))
    tlog.close()
    print(f"saved {out / 'policy_last.ckpt'}")
    stack.close()


if __name__ == "__main__":
    main()
