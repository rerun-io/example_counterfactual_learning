"""Task-preserving visual nuisances for sim_transfer_cube (CFNBC paper §IV, App. B, Fig. 6/7).

Four families x four levels (L0-L3) = 16 single-factor conditions, plus 6 predefined combinations = 22.
The paper gives families, levels and figures but no numeric parameters. Every value below is an ASSUMPTION
approximated from Fig. 6 (see results/assumptions.md).

Requires the scene act/assets/bimanual_viperx_transfer_cube_nuisance.xml (original scene + hidden, collision-free
nuisance geoms; physics verified bit-identical). All edits are visual: geom rgba/group/body_pos of static
contype=0 bodies, light parameters and headlight. None of them touch dynamics.

Usage:
    spec = CONDITIONS["distractor_L2"]
    apply(physics, spec, cube_init_pose)   # after physics reset / setting box pose; calls physics.forward()
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

NUISANCE_XML = "bimanual_viperx_transfer_cube_nuisance.xml"
VISIBLE_GROUP, HIDDEN_GROUP = 1, 4

# --- family parameters (ASSUMPTION, approximated from paper Fig. 6) --------------------------------------------
TABLE_RGBA_CLEAN = (0.2, 0.2, 0.2, 1.0)
APPEARANCE_TABLE_RGBA = {  # Fig. 6 row 1: white, brown, green, magenta table
    0: (0.95, 0.95, 0.92, 1.0),
    1: (0.62, 0.40, 0.25, 1.0),
    2: (0.30, 0.60, 0.45, 1.0),
    3: (0.85, 0.22, 0.78, 1.0),
}
# Lighting: the headlight is switched off (scalar `active` flag) and replaced by per-light ambient on the first fill
# light; the shadow-casting light (index 1) is re-aimed and strengthened, the fills dimmed.
# NOTE: never write model.vis.headlight.{ambient,diffuse,specular}: in mujoco 2.3.7 / dm_control 1.0.14 their numpy
# views have stride 0, so `[:] = x` writes only the red channel (this produced a teal cast; findings-log #16).
LIGHTING = {
    0: dict(shadow_dir=(0.0, 0.3, -1.0), shadow_diffuse=0.45, fill_diffuse=0.25, ambient=0.35),
    1: dict(shadow_dir=(-0.5, 0.5, -1.0), shadow_diffuse=0.55, fill_diffuse=0.18, ambient=0.28),
    2: dict(shadow_dir=(-0.8, -0.4, -1.0), shadow_diffuse=0.65, fill_diffuse=0.12, ambient=0.20),
    3: dict(shadow_dir=(0.5, 0.3, -1.0), shadow_diffuse=0.75, fill_diffuse=0.05, ambient=0.12),  # shadows up-right onto the table (was (0.6,-0.8,-0.9): off-table)
}
# Distractors: 4 blue cubes around the target's initial position; radius shrinks with level (Fig. 6 row 3).
DISTRACTOR_RADIUS = {0: 0.13, 1: 0.10, 2: 0.07, 3: 0.055}  # L3 widened from 0.045 (Stage C clipping check)
DISTRACTOR_ANGLES = np.deg2rad([20, 110, 200, 290])
# Local support: grey patch near/under the target; offset shrinks and contrast grows with level (Fig. 6 row 4).
SUPPORT = {  # colours: subtle, slightly lighter than the table (0.2), per the paper's Fig. 6 (was 0.30-0.48)
    0: dict(offset=(-0.14, -0.06), half=0.045, rgba=(0.24, 0.24, 0.24, 1.0)),
    1: dict(offset=(-0.08, 0.03), half=0.05, rgba=(0.26, 0.26, 0.26, 1.0)),
    2: dict(offset=(-0.035, 0.02), half=0.05, rgba=(0.28, 0.28, 0.28, 1.0)),
    3: dict(offset=(0.0, 0.0), half=0.055, rgba=(0.30, 0.30, 0.30, 1.0)),
}


@dataclass(frozen=True)
class Spec:
    appearance: int | None = None
    lighting: int | None = None
    distractor: int | None = None
    support: int | None = None

    @property
    def is_clean(self) -> bool:
        return all(v is None for v in (self.appearance, self.lighting, self.distractor, self.support))


def _build_conditions() -> dict[str, Spec]:
    conds = {"clean": Spec()}
    for fam in ("appearance", "lighting", "distractor", "support"):
        for lvl in range(4):
            conds[f"{fam}_L{lvl}"] = Spec(**{fam: lvl})
    # App. B combination list
    conds["combo_app3_sup3"] = Spec(appearance=3, support=3)
    conds["combo_dis3_sup3"] = Spec(distractor=3, support=3)
    conds["combo_app2_sup2"] = Spec(appearance=2, support=2)
    conds["combo_app3_dis3_lit3_sup3"] = Spec(appearance=3, distractor=3, lighting=3, support=3)
    conds["combo_app2_dis2_lit2_sup2"] = Spec(appearance=2, distractor=2, lighting=2, support=2)
    conds["combo_app3_dis2_lit3_sup2"] = Spec(appearance=3, distractor=2, lighting=3, support=2)
    return conds


CONDITIONS: dict[str, Spec] = _build_conditions()
NUISANCE_CONDITIONS = [c for c in CONDITIONS if c != "clean"]  # the paper's 22
FAMILY_OF = {c: (c.split("_")[0] if not c.startswith("combo") else "combo") for c in NUISANCE_CONDITIONS}


def from_env() -> str | None:
    """Condition name from env ACT_NUISANCE (None/'' -> no nuisance scene)."""
    name = os.environ.get("ACT_NUISANCE", "").strip()
    if name and name not in CONDITIONS:
        raise ValueError(f"unknown ACT_NUISANCE={name!r}; choose from {list(CONDITIONS)}")
    return name or None


def apply(physics, spec: Spec | str, cube_init_pose) -> None:
    """Configure the nuisance scene for one episode. cube_init_pose: (7,) xyz+quat of the target at reset."""
    if isinstance(spec, str):
        spec = CONDITIONS[spec]
    m = physics.model
    cube_xy = np.asarray(cube_init_pose, dtype=np.float64)[:2]

    table = m.name2id("table", "geom")
    m.geom_rgba[table] = APPEARANCE_TABLE_RGBA[spec.appearance] if spec.appearance is not None else TABLE_RGBA_CLEAN

    _reset_lights(physics)
    if spec.lighting is not None:
        p = LIGHTING[spec.lighting]
        d = np.asarray(p["shadow_dir"], dtype=np.float64)
        m.light_dir[1] = d / np.linalg.norm(d)
        m.light_diffuse[1] = p["shadow_diffuse"]
        m.light_diffuse[0] = p["fill_diffuse"]
        m.light_diffuse[2] = p["fill_diffuse"]
        m.vis.headlight.active = 0
        m.light_ambient[0] = p["ambient"]

    for k in range(4):
        gid, bid = m.name2id(f"nuis_distractor_{k}", "geom"), m.name2id(f"nuis_distractor_{k}", "body")
        if spec.distractor is None:
            m.geom_group[gid] = HIDDEN_GROUP
            continue
        r, a = DISTRACTOR_RADIUS[spec.distractor], DISTRACTOR_ANGLES[k]
        m.body_pos[bid][:2] = cube_xy + r * np.array([np.cos(a), np.sin(a)])
        m.body_pos[bid][2] = 0.0
        m.geom_group[gid] = VISIBLE_GROUP

    gid, bid = m.name2id("nuis_support", "geom"), m.name2id("nuis_support", "body")
    if spec.support is None:
        m.geom_group[gid] = HIDDEN_GROUP
    else:
        p = SUPPORT[spec.support]
        m.body_pos[bid][:2] = cube_xy + np.asarray(p["offset"])
        m.body_pos[bid][2] = 0.0
        m.geom_size[gid][:2] = p["half"]
        m.geom_rgba[gid] = p["rgba"]
        m.geom_group[gid] = VISIBLE_GROUP
    physics.forward()


_LIGHT_DEFAULTS: dict[int, tuple] = {}


def _reset_lights(physics) -> None:
    """Restore the scene.xml light values (cached from the first call on a fresh model). Headlight vectors are never
    written (stride-0 views, see LIGHTING); only its scalar `active` flag is toggled."""
    m = physics.model
    key = id(m.ptr) if hasattr(m, "ptr") else id(m)
    if key not in _LIGHT_DEFAULTS:
        _LIGHT_DEFAULTS[key] = (m.light_dir.copy(), m.light_diffuse.copy(), m.light_ambient.copy(),
                                int(m.vis.headlight.active))
    d, diff, amb, active = _LIGHT_DEFAULTS[key]
    m.light_dir[:] = d
    m.light_diffuse[:] = diff
    m.light_ambient[:] = amb
    m.vis.headlight.active = active
