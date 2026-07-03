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

"""VR teleoperation constants and view presets."""

# Action layout produced by the four retargeters declared in
# MobileAIReachEnvCfg_IK_ABS.teleop_devices.devices["handtracking"]:
#   indices 0..6   -> left  arm pose [pos_xyz, quat_wxyz]
#   indices 7..13  -> right arm pose [pos_xyz, quat_wxyz]
#   index   14     -> left  gripper scalar
#   index   15     -> right gripper scalar
ACTION_DIM_PER_ARM = 7
POSE_DIM = 2 * ACTION_DIM_PER_ARM  # 14D of arm poses
ACTION_DIM_ENV = POSE_DIM + 2  # 16D goes to env.step (poses + 2 grippers)
LEFT_GRIP_IDX = 14
RIGHT_GRIP_IDX = 15

LEFT_EE_BODY_NAME = "follower_left_link_6"
RIGHT_EE_BODY_NAME = "follower_right_link_6"

# Viewpoint presets selected by --view. Each maps to an XR anchor prim, a child
# transform offset (meters, relative to that prim) and a rotation (wxyz). These
# are starting points and are expected to be fine-tuned at runtime via the
# --anchor_pos / --anchor_rot / --anchor_prim_path overrides. The -90 deg yaw
# (0.7071, 0, 0, -0.7071) aligns OpenXR "forward" (+Y) with the robot "forward"
# (+X); third_person uses the opposite yaw so the operator looks back at the arms.
_ENV0_ROBOT = "/World/envs/env_0/Robot"
VIEW_PRESETS: dict[str, dict] = {
    "first_person": {
        "anchor_prim_path": f"{_ENV0_ROBOT}/cam_high_link",
        "anchor_pos": (0.0, 0.0, -1.7),
        "anchor_rot": (0.7071068, 0.0, 0.0, -0.7071068),
    },
    "over_shoulder": {
        "anchor_prim_path": f"{_ENV0_ROBOT}/base_link",
        "anchor_pos": (-0.7, 0.0, 0.5),
        "anchor_rot": (0.7071068, 0.0, 0.0, -0.7071068),
    },
    "third_person": {
        "anchor_prim_path": f"{_ENV0_ROBOT}/base_link",
        "anchor_pos": (0.6, -1.0, 0.6),
        "anchor_rot": (0.7071068, 0.0, 0.0, 0.7071068),
    },
}

# Keypoint joints visualized per hand when markers are enabled.
HAND_KEYPOINT_JOINTS = ("wrist", "thumb_tip", "index_tip")
