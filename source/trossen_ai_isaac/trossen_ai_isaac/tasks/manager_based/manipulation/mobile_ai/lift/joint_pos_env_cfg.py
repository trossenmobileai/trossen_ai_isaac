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

"""Joint-position rollout environment for ACT policy evaluation.

``MobileAILiftEnvCfg_JOINT_POS_PLAY`` accepts 14D absolute joint targets
(left arm 7D + right arm 7D).  For a 7D right-arm-only ACT checkpoint the
caller should keep the left arm at its current state — see
``trossen_ai_isaac.evaluation.act_rollout._policy_action_to_env``.
"""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.mdp.observations import (
    RECORD_JOINT_NAMES,
)
from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.record_env_cfg import (
    MobileAIRecordSceneCfg,
)

from .lift_env_cfg import MobileAILiftEnvCfg

LEFT_JOINTS = RECORD_JOINT_NAMES[:7]
RIGHT_JOINTS = RECORD_JOINT_NAMES[7:]


@configclass
class MobileAILiftEnvCfg_JOINT_POS(MobileAILiftEnvCfg):
    """Lift task controlled by absolute joint position targets (14D env actions).

    Both arms are independently commanded via joint-position control; grippers
    are included in each arm's 7D slice (index 6 = carriage joint = gripper).
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.left_arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_JOINTS,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.right_arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_JOINTS,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.left_gripper_action = None
        self.actions.right_gripper_action = None


@configclass
class MobileAILiftEnvCfg_JOINT_POS_PLAY(MobileAILiftEnvCfg_JOINT_POS):
    """Play variant for closed-loop ACT rollout with cameras (14D env actions).

    Cameras are enabled via ``MobileAIRecordSceneCfg``; the ``act_rollout``
    module reads them directly and forwards to the policy sidecar.
    """

    scene: MobileAIRecordSceneCfg = MobileAIRecordSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.observations.policy.enable_corruption = False
