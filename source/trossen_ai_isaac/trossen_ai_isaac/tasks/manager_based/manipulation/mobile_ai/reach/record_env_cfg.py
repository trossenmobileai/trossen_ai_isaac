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

"""IL recording environment for the Mobile AI bimanual robot.

Inherits absolute IK + binary grippers + handtracking from ``MobileAIReachEnvCfg_IK_ABS_PLAY``
so the same task can later support VR teleoperation with recording (Phase 2). Phase 1 only
uses the observation and camera pipeline for dataset validation.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
from .ik_abs_env_cfg import MobileAIReachEnvCfg_IK_ABS_PLAY
from .reach_env_cfg import MobileAIReachSceneCfg


@configclass
class EmptyCommandsCfg:
    """No random EE pose commands — IL recording does not use command terms."""

    pass


@configclass
class MobileAIRecordSceneCfg(MobileAIReachSceneCfg):
    """Reach scene extended with the three RGB cameras used by LeRobot datasets."""

    cam_high: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/cam_high_link/cam_high_color_frame/cam_high_color_optical_frame",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )

    cam_left_wrist: CameraCfg = CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/follower_left_camera_link/"
            "follower_left_camera_color_frame/follower_left_camera_color_optical_frame"
        ),
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )

    cam_right_wrist: CameraCfg = CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/follower_right_camera_link/"
            "follower_right_camera_color_frame/follower_right_camera_color_optical_frame"
        ),
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )


@configclass
class RecordObservationsCfg:
    """14D absolute joint positions only — matches ``trossen_ai_2026_project`` state layout."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group for IL recording."""

        joint_pos = ObsTerm(func=mdp.record_joint_pos_14)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class MobileAIReachEnvCfg_RECORD_PLAY(MobileAIReachEnvCfg_IK_ABS_PLAY):
    """Play variant for IL recording: 60 Hz, 14D joint obs, 3 RGB cameras, no EE commands."""

    scene: MobileAIRecordSceneCfg = MobileAIRecordSceneCfg(num_envs=1, env_spacing=2.5)
    observations: RecordObservationsCfg = RecordObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.commands = EmptyCommandsCfg()

        self.actions.left_arm_action.debug_vis = False
        self.actions.right_arm_action.debug_vis = False
