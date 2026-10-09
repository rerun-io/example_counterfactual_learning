"""Re-render recorded demo states under nuisance conditions (paired counterfactual observations).

Uses the per-timestep `sim_state/` saved in each demo HDF5. Replaying the stored float32 actions does NOT
reproduce the recorded trajectory (the recorder stepped with float64 actions), so states are set directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dm_control import mujoco

from cfnbc import nuisance as N

ACT_ASSETS = Path(__file__).resolve().parents[2] / "act" / "assets"


class StateRenderer:
    """One nuisance-scene Physics, reused across conditions and timesteps."""

    def __init__(self, height: int = 480, width: int = 640, camera: str = "top"):
        self.physics = mujoco.Physics.from_xml_path(str(ACT_ASSETS / N.NUISANCE_XML))
        self.height, self.width, self.camera = height, width, camera

    def render(self, sim_qpos: np.ndarray, condition: str, cube_init_pose: np.ndarray,
               sim_qvel: np.ndarray | None = None) -> np.ndarray:
        p = self.physics
        with p.reset_context():
            p.data.qpos[:] = sim_qpos
            if sim_qvel is not None:
                p.data.qvel[:] = sim_qvel
        N.apply(p, condition, cube_init_pose)  # calls forward()
        return p.render(height=self.height, width=self.width, camera_id=self.camera)

    def render_episode(self, sim_qpos: np.ndarray, condition: str, timesteps) -> np.ndarray:
        """(len(timesteps), H, W, 3) frames; nuisance placement keyed to the cube's pose at t=0."""
        cube0 = sim_qpos[0, 16:23]
        return np.stack([self.render(sim_qpos[t], condition, cube0) for t in timesteps])
