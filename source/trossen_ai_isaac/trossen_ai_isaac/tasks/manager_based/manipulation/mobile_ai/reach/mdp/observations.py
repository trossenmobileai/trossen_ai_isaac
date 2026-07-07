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

"""Observation helpers for Mobile AI IL recording."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv

# Joint order matches `trossen_ai_2026_project` meta/info.json:
# left_joint_0..5, left_joint_6 (carriage), right_joint_0..5, right_joint_6.
RECORD_JOINT_NAMES = [
    "follower_left_joint_0",
    "follower_left_joint_1",
    "follower_left_joint_2",
    "follower_left_joint_3",
    "follower_left_joint_4",
    "follower_left_joint_5",
    "follower_left_left_carriage_joint",
    "follower_right_joint_0",
    "follower_right_joint_1",
    "follower_right_joint_2",
    "follower_right_joint_3",
    "follower_right_joint_4",
    "follower_right_joint_5",
    "follower_right_left_carriage_joint",
]

# Resolve indices once at import time — joint names are fixed on the Mobile AI USD.
_RECORD_JOINT_IDS: list[int] | None = None


def _record_joint_ids(robot: Articulation) -> list[int]:
    global _RECORD_JOINT_IDS
    if _RECORD_JOINT_IDS is None:
        _RECORD_JOINT_IDS = [robot.joint_names.index(name) for name in RECORD_JOINT_NAMES]
    return _RECORD_JOINT_IDS


def record_joint_pos_14(env: ManagerBasedEnv) -> torch.Tensor:
    """Return 14D absolute joint positions in LeRobot dataset order."""
    robot: Articulation = env.scene["robot"]
    joint_ids = _record_joint_ids(robot)
    return robot.data.joint_pos[:, joint_ids]


def record_joint_target_14(env: ManagerBasedEnv) -> torch.Tensor:
    """Return 14D commanded joint targets in LeRobot dataset order."""
    robot: Articulation = env.scene["robot"]
    joint_ids = _record_joint_ids(robot)
    return robot.data.joint_pos_target[:, joint_ids]
