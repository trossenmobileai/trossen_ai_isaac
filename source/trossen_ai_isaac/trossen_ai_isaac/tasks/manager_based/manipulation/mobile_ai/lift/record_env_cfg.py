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

"""IL recording environment for the Mobile AI lift task."""

from isaaclab.utils import configclass

from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.record_env_cfg import (
    MobileAIRecordSceneCfg,
)

from .ik_abs_env_cfg import MobileAILiftEnvCfg_IK_ABS_PLAY


@configclass
class MobileAILiftEnvCfg_RECORD_PLAY(MobileAILiftEnvCfg_IK_ABS_PLAY):
    """Record demonstrations for pick-lift-place with 14D obs and 3 RGB cameras."""

    scene: MobileAIRecordSceneCfg = MobileAIRecordSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.actions.left_arm_action.debug_vis = False
        self.actions.right_arm_action.debug_vis = False
