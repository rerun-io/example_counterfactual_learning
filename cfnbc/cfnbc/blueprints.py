"""Viewer blueprints for the experiment datasets, saved as .rbl under <rerun_root>/blueprints/ and registered by
cfnbc.serve as each dataset's default.

    python -m cfnbc.blueprints        # (re)generate all .rbl files
"""
from __future__ import annotations

from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from cfnbc import schema as S

OUT = S.rerun_root() / "blueprints"


def joint_plot(**kw) -> rrb.TimeSeriesView:
    """Time series of 14 joint values with the fixed per-joint colours as view defaults (applies to recordings that
    were logged without colours, too)."""
    return rrb.TimeSeriesView(defaults=[rr.SeriesLines.from_fields(colors=S.JOINT_COLORS_14)], **kw)


def demo_review() -> rrb.Blueprint:
    """Demo review: top camera | joint plots | QA scalars | cube pose."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=S.CAMERA_TOP, name="top camera"),
                rrb.Spatial3DView(origin="/world", name="cube pose"),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                joint_plot(origin=S.ACTION, name="action (14)"),
                joint_plot(origin=S.QPOS, name="qpos (14)"),
                rrb.TimeSeriesView(origin="/qa/reward_stage", name="reward stage"),
                rrb.TimeSeriesView(origin="/qa/action_jerk", name="action jerk"),
                rrb.TimeSeriesView(origin="/qa/cube_height", name="cube height"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def rollout_review() -> rrb.Blueprint:
    """Eval rollout: top camera + 3D scene from recorded state (scene3d layer, where written), reward (upstream raw
    stage) vs stable stage, cube height (rollout qa layer), commanded action."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=S.CAMERA_TOP, name="top camera (what the policy saw)"),
                rrb.Spatial3DView(origin="/world/scene", name="3D scene from recorded state (what happened)"),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/", contents=[S.REWARD, "/qa/stable_stage"], name="reward stage: upstream vs stable",
                                   axis_y=rrb.ScalarAxis(range=(-0.2, 4.2))),
                rrb.TimeSeriesView(origin="/qa/cube_height", name="cube height (m)", axis_y=rrb.ScalarAxis(range=(0.0, 0.3))),
                joint_plot(origin=S.ACTION, name="commanded action (14)"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def training_review() -> rrb.Blueprint:
    """Training run; y-ranges pinned so the epoch-0 spike (loss ~70) doesn't flatten the curves."""
    return rrb.Blueprint(
        rrb.Grid(
            rrb.TimeSeriesView(origin="/", contents=["/train/loss", "/val/loss"], name="loss", axis_y=rrb.ScalarAxis(range=(0.0, 1.0))),
            rrb.TimeSeriesView(origin="/", contents=["/train/l1", "/val/l1"], name="l1", axis_y=rrb.ScalarAxis(range=(0.0, 0.4))),
            rrb.TimeSeriesView(origin="/", contents=["/train/kl", "/val/kl"], name="kl", axis_y=rrb.ScalarAxis(range=(0.0, 0.3))),
            rrb.TimeSeriesView(origin="/train_batch", name="per-batch", axis_y=rrb.ScalarAxis(range=(0.0, 2.0))),
        ),
        collapse_panels=True,
    )


def drift_triage(condition: str, seed: int = 0) -> rrb.Blueprint:
    """Clean vs counterfactual frame side by side, the policy's predicted first action under each, and ||Δa|| per step.
    Open a demo segment of cube_demos (layers base + clean_pred + <condition>) with this blueprint."""
    pol = f"/policy/seed{seed}"
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=S.CAMERA_TOP, name="clean (base layer)"),
                rrb.Spatial2DView(origin=f"/camera/top_{condition}", name=f"{condition} (counterfactual layer)"),
            ),
            rrb.Horizontal(
                joint_plot(origin=f"{pol}/clean_pred/action0", name="predicted action, clean (normalized)"),
                joint_plot(origin=f"{pol}/{condition}/action0", name=f"predicted action, {condition}"),
                rrb.TimeSeriesView(origin=f"{pol}/{condition}/delta_norm", name="||Δa|| (drift per step)"),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def drift_ghosts(condition: str, seed: int = 0, eye_target=None, eye_offset=(0.25, -0.35, 0.25)) -> rrb.Blueprint:
    """Drift triage with the policy's intent made visible: clean | counterfactual frame, and the 3D scene with the
    recorded robot (grey) plus ghost arms posed at the predicted action under each frame (cfnbc.ghost layer)."""
    pol = f"/policy/seed{seed}"
    eye = None
    if eye_target is not None:
        t = [float(v) for v in eye_target]
        eye = rrb.EyeControls3D(position=[t[i] + eye_offset[i] for i in range(3)], look_target=t, eye_up=[0, 0, 1])
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin=S.CAMERA_TOP, name="clean frame"),
                    rrb.Spatial2DView(origin=f"/camera/top_{condition}", name=f"{condition} frame (same recorded state)"),
                ),
                rrb.Spatial3DView(origin="/world/scene", eye_controls=eye,
                                  name=f"policy intent: recorded robot (grey), from clean frame (blue), from {condition} (orange)"),
                column_shares=[1, 2],
            ),
            rrb.TimeSeriesView(origin=f"{pol}/{condition}/delta_norm", name="||Δa|| (drift per step)"),
            row_shares=[4, 1],
        ),
        collapse_panels=True,
    )


def real_demo_review() -> rrb.Blueprint:
    """Real ALOHA teleop episode: overhead + wrist cameras, gripper command vs follower, teleop QA scalars."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/observation.images.cam_high", name="cam_high"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/observation.images.cam_left_wrist", name="left wrist"),
                    rrb.Spatial2DView(origin="/observation.images.cam_right_wrist", name="right wrist"),
                ),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                joint_plot(origin="/action", name="leader command (action)"),
                joint_plot(origin="/observation.state", name="follower state"),
                rrb.TimeSeriesView(origin="/qa/joint_speed", name="arm speed (rad/s)"),
                rrb.TimeSeriesView(origin="/qa/tracking_error", name="leader-follower tracking error"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def real_edit_review(condition: str) -> rrb.Blueprint:
    """Real frame (clean JPEG) | edited frame, with the mask overlay and per-step drift."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/", contents=["/camera/cam_high_clean", "/masks/keep_out", "/masks/swept"],
                                  name="clean + masks (red keep-out, yellow swept)"),
                rrb.Spatial2DView(origin=f"/camera/cam_high_{condition}", name=f"edited: {condition}"),
            ),
            rrb.Horizontal(
                joint_plot(origin="/policy/real/clean_pred/action0", name="predicted action, clean"),
                joint_plot(origin=f"/policy/real/{condition}/action0", name="predicted action under the edit"),
                rrb.TimeSeriesView(origin=f"/policy/real/{condition}/delta_norm", name="||Δa|| (drift per step)"),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
    )


BLUEPRINTS = {S.DATASET_DEMOS: demo_review, S.DATASET_ROLLOUTS: rollout_review, S.DATASET_TRAINING: training_review,
              "real_demos": real_demo_review}


def path_for(dataset: str) -> Path:
    return OUT / f"{dataset}.rbl"


def save_all() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for dataset, fn in BLUEPRINTS.items():
        p = path_for(dataset)
        fn().save("cfnbc_aloha_cups_open" if dataset == "real_demos" else S.APP_ID, p)
        out[dataset] = p
    return out


VIEWS = {  # named layouts that can temporarily become a dataset's default (e.g. for screenshots)
    "drift_triage_lighting_L3": (S.DATASET_DEMOS, lambda: drift_triage("lighting_L3", seed=0)),
    "real_edit_appearance_L3": ("real_demos", lambda: real_edit_review("appearance_L3")),
    "real_edit_distractor_L3": ("real_demos", lambda: real_edit_review("distractor_L3")),
}


def set_default(view: str) -> None:
    """Register a named layout as its dataset's default blueprint on the catalog. In the viewer, the dataset's default wins
    over a .rbl opened next to a catalog URL, so this is how to show a custom layout for catalog segments."""
    from cfnbc import hub
    import time
    dataset, fn = VIEWS[view]
    # registered files must never change afterwards (the catalog references them): write a new, versioned file
    p = OUT / "registered" / f"{view}_{int(time.time())}.rbl"
    p.parent.mkdir(parents=True, exist_ok=True)
    fn().save(S.APP_ID, p)
    with hub.open_catalog() as client:
        client.get_dataset(dataset).register_blueprint(f"file://{p.resolve()}", set_default=True)
    print(f"{dataset}: default blueprint -> {view} (restore with: python -m cfnbc.blueprints --restore)")


def restore_defaults() -> None:
    from cfnbc import hub
    from cfnbc.serve import register_blueprints
    save_all()
    with hub.open_catalog() as client:
        register_blueprints(client)
    print("default blueprints restored:", ", ".join(BLUEPRINTS))


def _require_server() -> None:
    from cfnbc import hub
    if not hub.catalog_url():
        raise SystemExit("CFNBC_CATALOG_URL is not set: run `source env.sh` (and start scripts/serve.sh) first (otherwise this would "
                         "change a throwaway in-process server, not the running one)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-default", choices=sorted(VIEWS), help="make a named layout its dataset's default")
    ap.add_argument("--restore", action="store_true", help="re-register the standard default blueprints")
    a = ap.parse_args()
    if a.set_default:
        _require_server()
        set_default(a.set_default)
    elif a.restore:
        _require_server()
        restore_defaults()
    else:
        for d, p in save_all().items():
            print(f"{d}: {p}")
