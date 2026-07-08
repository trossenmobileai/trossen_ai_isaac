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

"""Shared Mobile AI absolute-IK teleoperation utilities."""

from __future__ import annotations

import logging

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

# Action layout (16D):
#   indices  0..2   -> left  arm position    [x, y, z]  (base frame)
#   indices  3..6   -> left  arm quaternion  [w, x, y, z]  (base frame)
#   indices  7..9   -> right arm position    [x, y, z]  (base frame)
#   indices 10..13  -> right arm quaternion  [w, x, y, z]  (base frame)
#   index   14      -> left  gripper scalar  (+1 open / -1 close)
#   index   15      -> right gripper scalar  (+1 open / -1 close)
ACTION_DIM = 16
POSE_DIM = 14
LEFT_GRIP_IDX = 14
RIGHT_GRIP_IDX = 15

LEFT_ARM = "left"
RIGHT_ARM = "right"
LEFT_EE_BODY_NAME = "follower_left_link_6"
RIGHT_EE_BODY_NAME = "follower_right_link_6"

# Physics step rate for Mobile AI reach / record envs (sim.dt = 1/60).
SIM_HZ = 60

POS_SCALE = 0.05
ROT_SCALE = 0.05


def get_ee_base_pose(
    robot: Articulation, body_idx: int, env_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the EE pose of a single body in the robot base frame."""
    pos_w = robot.data.body_pos_w[env_idx, body_idx]
    quat_w = robot.data.body_quat_w[env_idx, body_idx]
    root_pos_w = robot.data.root_pos_w[env_idx]
    root_quat_w = robot.data.root_quat_w[env_idx]

    pos_b, quat_b = subtract_frame_transforms(
        root_pos_w.unsqueeze(0),
        root_quat_w.unsqueeze(0),
        pos_w.unsqueeze(0),
        quat_w.unsqueeze(0),
    )
    return pos_b.squeeze(0), quat_b.squeeze(0)


def resolve_ee_body_indices(robot: Articulation) -> dict[str, int]:
    """Map left/right arm keys to follower EE body indices."""
    left_body_idx = robot.body_names.index(LEFT_EE_BODY_NAME)
    right_body_idx = robot.body_names.index(RIGHT_EE_BODY_NAME)
    return {LEFT_ARM: left_body_idx, RIGHT_ARM: right_body_idx}


def assemble_ik_abs_action(
    l_pos: torch.Tensor,
    l_quat: torch.Tensor,
    r_pos: torch.Tensor,
    r_quat: torch.Tensor,
    l_grip: float,
    r_grip: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Concatenate a 16D absolute IK action vector."""
    return torch.cat(
        [
            l_pos,
            l_quat,
            r_pos,
            r_quat,
            torch.tensor([l_grip, r_grip], dtype=torch.float32, device=device),
        ]
    )


def broadcast_action(action_1d: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Broadcast a 1D action to all parallel envs."""
    return action_1d.unsqueeze(0).repeat(num_envs, 1)


def warn_action_dim_mismatch(env, expected: int = ACTION_DIM) -> None:
    """Log a warning when the env action width does not match IK-Abs teleop."""
    try:
        action_dim = env.action_manager.total_action_dim
    except AttributeError:
        return
    if action_dim != expected:
        logger.warning(
            f"Env action_dim={action_dim} but teleop assembles {expected}D actions. "
            "You probably selected an IK-Rel task; expect a shape mismatch on env.step(). "
            "Use one of the Isaac-Reach-MobileAI-IK-Abs-* tasks."
        )


def make_env_cfg(task: str, device: str, num_envs: int, fps: int | None = None):
    """Parse env cfg, disable timeout, and optionally set decimation from fps."""
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.env_name = task
    # Recording tasks must keep the three camera sensors enabled.
    if "Record" in task:
        expected_cams = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        scene_cfg = getattr(env_cfg, "scene", None)
        if scene_cfg is not None:
            missing = [name for name in expected_cams if not hasattr(scene_cfg, name)]
            if missing:
                logger.warning("Env cfg is missing expected camera configs: %s", missing)
    env_cfg.terminations.time_out = None

    if fps is not None:
        decimation = max(1, round(SIM_HZ / fps))
        env_cfg.decimation = decimation
        env_cfg.sim.render_interval = decimation

    return env_cfg
