# Copyright 2026 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Pick-lift-place success metrics for the Mobile AI lift task.

Shared by ACT and Pi0 closed-loop eval (``act_rollout.py`` / ``run_play_*.sh``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

# =============================================================================
# EVAL CONTRACT (locked for ACT and Pi0 sim rollouts — change intentionally)
# -----------------------------------------------------------------------------
# Env: Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0, FPS 60, episode timeout 90 s
# Start (act_rollout): arms home, grippers open 0.044 m, warm-up hold, left fixed
# Success: clear lift (z > LIFT_CLEAR_Z) then cube_is_placed (on-table+stable+open)
# Early stop:
#   IDLE_STEPS            -> stop_reason=no_progress
#   MAX_APPROACH_STEPS    -> stop_reason=no_pick
#   MAX_STEPS_AFTER_LIFT  -> stop_reason=no_place
#   POST_SUCCESS_STEPS    -> stop_reason=success
# Metrics: overall success_rate + success_rate_by_color + per-episode cube_color
# =============================================================================

# Table top is ~0.71 m; cube spawn center ~0.745 m when resting on the table.
TABLE_SURFACE_Z = 0.71
CUBE_REST_Z = 0.745
LIFT_DELTA_Z = 0.08
PLACE_Z_TOLERANCE = 0.08
# Eval lift must clear the on-table band upper edge to avoid false positives on approach.
LIFT_CLEAR_MARGIN = 0.02
LIFT_CLEAR_Z = CUBE_REST_Z + PLACE_Z_TOLERANCE + LIFT_CLEAR_MARGIN
PLACE_VEL_THRESHOLD = 0.08
GRIPPER_OPEN_THRESHOLD = 0.02
DROP_Z = 0.65
POST_SUCCESS_STEPS = 60
# Continuous pick window (no robot reset): hard cap if the cube never lifts.
APPROACH_STEPS_PER_TRY = 500
MAX_PICK_TRIES = 3
MAX_APPROACH_STEPS = 1000
MAX_STEPS_AFTER_LIFT = 500
# Stop early if the cube stays on the table with no lift progress for this many steps.
IDLE_STEPS = 500

LEFT_CARRIAGE = "follower_left_left_carriage_joint"
RIGHT_CARRIAGE = "follower_right_left_carriage_joint"

# Discrete cube colors used by EventCfg.randomize_cube_color.
_CUBE_COLOR_NAMES: dict[tuple[float, float, float], str] = {
    (1.0, 0.0, 0.0): "red",
    (0.0, 1.0, 0.0): "green",
    (0.0, 0.0, 1.0): "blue",
}


def cube_color_name(env: ManagerBasedEnv, env_id: int = 0) -> str:
    """Return the discrete cube color name for this env (red/green/blue/unknown)."""
    colors = getattr(env, "_cube_color_rgb", None)
    if not isinstance(colors, dict):
        return "unknown"
    rgb = colors.get(int(env_id))
    if rgb is None:
        return "unknown"
    key = tuple(round(float(c), 3) for c in rgb)
    # Exact match first, then nearest named color.
    if key in _CUBE_COLOR_NAMES:
        return _CUBE_COLOR_NAMES[key]
    for named_rgb, name in _CUBE_COLOR_NAMES.items():
        if all(abs(a - b) < 0.05 for a, b in zip(key, named_rgb)):
            return name
    return "unknown"


@dataclass
class EpisodeMetrics:
    """Scalar episode outcome flags for logging and evaluation."""

    cube_lifted: bool
    cube_on_table: bool
    cube_returned_after_lift: bool
    cube_dropped: bool
    episode_success: bool
    stop_reason: str
    cube_color: str
    cube_height: float
    cube_speed: float
    left_gripper: float
    right_gripper: float


@dataclass
class EpisodeCubeTracker:
    """Track cube state over an episode for temporal success criteria."""

    ever_lifted: bool = False
    first_lift_step: int | None = None
    returned_after_lift: bool = False
    success_trigger_step: int | None = None
    post_success_steps: int = 0
    stop_reason: str = "running"
    idle_steps: int = 0

    def update(self, env: ManagerBasedEnv, step: int, env_id: int = 0) -> None:
        """Update tracker from the current env state."""
        if not self.ever_lifted and bool(cube_is_clearly_lifted(env)[env_id].item()):
            self.ever_lifted = True
            self.first_lift_step = step
            self.idle_steps = 0

        if (
            self.ever_lifted
            and self.first_lift_step is not None
            and step > self.first_lift_step
            and not self.returned_after_lift
            and bool(cube_is_placed(env)[env_id].item())
        ):
            self.returned_after_lift = True
            self.success_trigger_step = step

        if self.is_success() and self.success_trigger_step is not None:
            self.post_success_steps += 1

        # Count consecutive steps with no pick progress (cube still resting on table).
        if not self.ever_lifted:
            if bool(cube_on_table(env)[env_id].item()):
                self.idle_steps += 1
            else:
                self.idle_steps = 0

    def is_success(self) -> bool:
        """True when the cube was lifted and later released on the table."""
        return self.ever_lifted and self.returned_after_lift

    def should_stop(self, step: int) -> bool:
        """Stop after success, idle no-progress, no pick, or failed place."""
        if self.is_success() and self.post_success_steps >= POST_SUCCESS_STEPS:
            self.stop_reason = "success"
            return True
        if not self.ever_lifted and self.idle_steps >= IDLE_STEPS:
            self.stop_reason = "no_progress"
            return True
        if not self.ever_lifted and step >= MAX_APPROACH_STEPS:
            self.stop_reason = "no_pick"
            return True
        if (
            self.ever_lifted
            and self.first_lift_step is not None
            and not self.is_success()
            and step - self.first_lift_step >= MAX_STEPS_AFTER_LIFT
        ):
            self.stop_reason = "no_place"
            return True
        return False


def cube_height_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """World-frame cube center height."""
    cube: RigidObject = env.scene[asset_cfg.name]
    return cube.data.root_pos_w[:, 2]


def cube_is_lifted(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    minimal_height: float = TABLE_SURFACE_Z + LIFT_DELTA_Z,
) -> torch.Tensor:
    """True when the cube is lifted above the table."""
    return cube_height_w(env, asset_cfg) > minimal_height


def cube_is_clearly_lifted(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    minimal_height: float = LIFT_CLEAR_Z,
) -> torch.Tensor:
    """True when the cube clears the on-table height band (eval lift detection)."""
    return cube_height_w(env, asset_cfg) > minimal_height


def cube_on_table(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the cube is resting on the table with low velocity (no gripper check)."""
    cube: RigidObject = env.scene[asset_cfg.name]
    height = cube.data.root_pos_w[:, 2]
    speed = torch.linalg.norm(cube.data.root_lin_vel_w, dim=1)

    on_table = torch.abs(height - CUBE_REST_Z) < PLACE_Z_TOLERANCE
    stable = speed < PLACE_VEL_THRESHOLD
    return on_table & stable


def cube_is_placed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the cube is resting on the table with low velocity and open grippers."""
    cube: RigidObject = env.scene[asset_cfg.name]
    height = cube.data.root_pos_w[:, 2]
    speed = torch.linalg.norm(cube.data.root_lin_vel_w, dim=1)

    robot: Articulation = env.scene["robot"]
    left_idx = robot.joint_names.index(LEFT_CARRIAGE)
    right_idx = robot.joint_names.index(RIGHT_CARRIAGE)
    grippers_open = (robot.data.joint_pos[:, left_idx] > GRIPPER_OPEN_THRESHOLD) | (
        robot.data.joint_pos[:, right_idx] > GRIPPER_OPEN_THRESHOLD
    )

    on_table = torch.abs(height - CUBE_REST_Z) < PLACE_Z_TOLERANCE
    stable = speed < PLACE_VEL_THRESHOLD
    return on_table & stable & grippers_open


def cube_dropped(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    minimum_height: float = DROP_Z,
) -> torch.Tensor:
    """True when the cube falls below the table."""
    return cube_height_w(env, asset_cfg) < minimum_height


def evaluate_episode_metrics(
    env: ManagerBasedEnv,
    *,
    tracker: EpisodeCubeTracker,
    env_id: int = 0,
) -> EpisodeMetrics:
    """Compute episode-level success flags for rollout logging."""
    cube: RigidObject = env.scene["cube"]
    robot: Articulation = env.scene["robot"]

    cube_h = float(cube.data.root_pos_w[env_id, 2].item())
    cube_speed = float(torch.linalg.norm(cube.data.root_lin_vel_w[env_id]).item())
    left_idx = robot.joint_names.index(LEFT_CARRIAGE)
    right_idx = robot.joint_names.index(RIGHT_CARRIAGE)
    left_grip = float(robot.data.joint_pos[env_id, left_idx].item())
    right_grip = float(robot.data.joint_pos[env_id, right_idx].item())

    on_table = bool(cube_on_table(env)[env_id].item())
    dropped = bool(cube_dropped(env)[env_id].item())
    success = tracker.is_success()

    return EpisodeMetrics(
        cube_lifted=tracker.ever_lifted,
        cube_on_table=on_table,
        cube_returned_after_lift=tracker.returned_after_lift,
        cube_dropped=dropped,
        episode_success=success,
        stop_reason=tracker.stop_reason,
        cube_color=cube_color_name(env, env_id),
        cube_height=cube_h,
        cube_speed=cube_speed,
        left_gripper=left_grip,
        right_gripper=right_grip,
    )
