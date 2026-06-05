# Copyright 2026 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

"""Absolute-IK reach environment configuration for the Mobile AI bimanual robot.

This config produces a 14D action space [left_7D | right_7D] where each 7D entry is
[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] in the robot base frame. It is
the natural action layout for VR hand-tracking teleoperation, where each hand maps
directly to one arm's end-effector pose.

The `handtracking` teleop device is wired up here so the script-side code can build
the OpenXR device + retargeters via `create_teleop_device(...)` without hardcoding
Isaac Lab classes in the teleop script.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DeviceBase
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters import (
    GripperRetargeterCfg,
    Se3AbsRetargeterCfg,
)
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from .reach_env_cfg import MobileAIReachEnvCfg


@configclass
class MobileAIReachEnvCfg_IK_ABS(MobileAIReachEnvCfg):
    """Mobile AI reach environment using absolute IK on both arms.

    Action vector layout (14D total, left then right):
        [L_pos_x, L_pos_y, L_pos_z, L_quat_w, L_quat_x, L_quat_y, L_quat_z,
         R_pos_x, R_pos_y, R_pos_z, R_quat_w, R_quat_x, R_quat_y, R_quat_z]

    All poses are in the robot base frame.
    """

    # XR anchor — defines how the OpenXR world frame is positioned relative to the
    # robot base frame. With anchor at the origin and identity rotation, the user's
    # hand poses are interpreted directly as targets in the robot base frame.
    # If the operator is standing in front of the robot, you can shift `anchor_pos`
    # in the +x direction (toward the robot) to bring the comfortable workspace closer.
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        super().__post_init__()

        # Flip both arm action terms to absolute pose mode. Each term now expects
        # a 7D pose [pos_xyz, quat_wxyz] per env step instead of a 6D delta.
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["follower_left_joint_[0-5]"],
            body_name="follower_left_link_6",
            debug_vis=True,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="pinv",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["follower_right_joint_[0-5]"],
            body_name="follower_right_link_6",
            debug_vis=True,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="pinv",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # Register the handtracking device. Retargeters are ordered such that
        # advance() returns a 1D tensor of shape [16]:
        #   indices  0..6   -> left  arm pose  (consumed by env action)
        #   indices  7..13  -> right arm pose  (consumed by env action)
        #   index    14     -> left  gripper   (no env slot yet, dropped by script)
        #   index    15     -> right gripper   (no env slot yet, dropped by script)
        self.teleop_devices.devices["handtracking"] = OpenXRDeviceCfg(
            retargeters=[
                Se3AbsRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                    zero_out_xy_rotation=False,
                    use_wrist_rotation=True,
                    use_wrist_position=True,
                    enable_visualization=True,
                    sim_device=self.sim.device,
                ),
                Se3AbsRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                    zero_out_xy_rotation=False,
                    use_wrist_rotation=True,
                    use_wrist_position=True,
                    enable_visualization=True,
                    sim_device=self.sim.device,
                ),
                GripperRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                    sim_device=self.sim.device,
                ),
                GripperRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                    sim_device=self.sim.device,
                ),
            ],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            # VR conventionally starts paused so the operator can position themselves
            # before the robot reacts; pinching to START begins teleop.
            teleoperation_active_default=False,
        )


@configclass
class MobileAIReachEnvCfg_IK_ABS_PLAY(MobileAIReachEnvCfg_IK_ABS):
    """Play/teleoperation variant: 1 env, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
