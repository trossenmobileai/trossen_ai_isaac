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

    # XR anchor — defines how the OpenXR world frame is positioned and oriented
    # relative to the robot base frame. `anchor_pos` is the OpenXR origin (under
    # the operator's feet) expressed in the robot base frame; `anchor_rot` is a
    # wxyz quaternion rotating OpenXR vectors into the robot base frame.
    #
    # Empirical OpenXR convention on this Quest 3 + ALVR + SteamVR setup:
    #   +X = operator right, +Y = operator forward, +Z = up.
    # Robot base frame convention:
    #   +X = robot forward, +Y = robot left, +Z = up.
    # The -90 deg yaw quaternion below maps OpenXR (right, forward, up)
    # onto robot (forward, left, up):
    #     openxr +Y -> robot +X (forward stays forward)
    #     openxr +X -> robot -Y (operator's right becomes robot's right)
    #     openxr +Z -> robot +Z (up stays up)
    #
    # `anchor_pos.z = -0.6` drops the OpenXR ground 60 cm below the robot base
    # so a hand at chest height (openxr.z ~ 0.8 m) lands at robot.z ~ 0.2 m,
    # squarely inside the Mobile AI arm workspace. `anchor_pos.x = -0.3` places
    # the operator 30 cm "behind" the robot so reaching forward naturally drives
    # IK targets in front of the robot rather than behind it.
    #
    # These values are best treated as a starting point — `scripts/teleoperation/
    # teleop_dual_arm_vr.py` exposes `--anchor_pos` and `--anchor_rot` flags so
    # the operator can sweep through anchors without re-editing this config.
    xr: XrCfg = XrCfg(
        anchor_pos=(-0.3, 0.0, -0.6),
        anchor_rot=(0.7071068, 0.0, 0.0, -0.7071068),
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
        #
        # NOTE: enable_visualization=False is a workaround for an Isaac Lab bug in
        # Se3AbsRetargeter._update_visualization, which calls Rotation.from_matrix()
        # on a value that is actually a (4,) quaternion. With visualization on, the
        # first frame raises:
        #     "Expected `matrix` to have shape (3, 3) or (N, 3, 3), got (4,)"
        # The retargeter itself works correctly — only the goal-frame marker is broken.
        self.teleop_devices.devices["handtracking"] = OpenXRDeviceCfg(
            retargeters=[
                Se3AbsRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                    zero_out_xy_rotation=False,
                    use_wrist_rotation=True,
                    use_wrist_position=True,
                    enable_visualization=False,
                    sim_device=self.sim.device,
                ),
                Se3AbsRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                    zero_out_xy_rotation=False,
                    use_wrist_rotation=True,
                    use_wrist_position=True,
                    enable_visualization=False,
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
