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
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
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

    # XR anchor — first-person view at the robot's head camera.
    #
    # `anchor_prim_path` parents the XRAnchor under the camera stand body
    # (`cam_high_link`) on the robot articulation. The operator's headset then tracks
    # that prim in world space, so the view sits *inside* the robot looking out
    # from its camera, regardless of how the robot is positioned in the scene.
    #
    # `anchor_pos` and `anchor_rot` are applied as a USD child transform on top
    # of the parent prim:
    #   * `anchor_pos.z = -1.7` cancels a 1.7 m tall operator's physical
    #     headset height so their eyes land at the camera height instead of
    #     ~1.7 m above it. Override via --anchor_pos if you're a different
    #     height (e.g. `--anchor_pos 0 0 -1.6` for 1.6 m, or `0 0 -1.8` for
    #     1.8 m). The hand-anchored IK math is unaffected by this offset.
    #   * `anchor_rot` is the same -90 deg yaw we've been using to align
    #     OpenXR's "forward" (+Y) with the robot's "forward" (+X). This stays
    #     useful in FOV terms -- it determines how operator head motion in
    #     the room maps to view rotation in the sim.
    #
    # anchor_rotation_mode defaults to FIXED, so this -90 deg yaw stays world-
    # constant rather than rotating with the parent prim. For mobile bases that
    # turn, you might prefer FOLLOW_PRIM_SMOOTHED so the view yaws with the
    # robot; switch at runtime via env_cfg.xr.anchor_rotation_mode if needed.
    #
    # NOTE: the prim path assumes env_0. If you bring up multiple envs, the
    # operator can only "be in" one of them; env_0 is the conventional choice.
    # If `cam_high_link` is not the right body on your URDF, override with
    # --anchor_prim_path or run with --list_bodies once to see what's available.
    xr: XrCfg = XrCfg(
        anchor_prim_path="/World/envs/env_0/Robot/cam_high_link",
        anchor_pos=(0.0, 0.0, -1.7),
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

        # Wire both grippers as binary open/close actions, driven by the pinch
        # (thumb-index distance) from each hand's GripperRetargeter. Each carriage
        # joint travels 0.0 m (closed) to 0.044 m (open); the opposite finger is a
        # mimic joint in the USD. Declared after the arm actions, so the full env
        # action vector becomes 16D: [L_pose(7), R_pose(7), L_grip(1), R_grip(1)].
        self.actions.left_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["follower_left_left_carriage_joint"],
            open_command_expr={"follower_left_left_carriage_joint": 0.044},
            close_command_expr={"follower_left_left_carriage_joint": 0.0},
        )
        self.actions.right_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["follower_right_left_carriage_joint"],
            open_command_expr={"follower_right_left_carriage_joint": 0.044},
            close_command_expr={"follower_right_left_carriage_joint": 0.0},
        )

        # Register the handtracking device. Retargeters are ordered such that
        # advance() returns a 1D tensor of shape [16]:
        #   indices  0..6   -> left  arm pose    (consumed by left_arm_action)
        #   indices  7..13  -> right arm pose    (consumed by right_arm_action)
        #   index    14     -> left  gripper     (consumed by left_gripper_action)
        #   index    15     -> right gripper     (consumed by right_gripper_action)
        # The GripperRetargeter emits +1.0 (open) / -1.0 (close); BinaryJoint-
        # PositionAction maps value > 0 -> open_command, else close_command.
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
