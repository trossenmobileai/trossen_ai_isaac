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

"""Base pick-lift-place environment for the Mobile AI bimanual robot."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.reach_env_cfg import (
    ActionsCfg,
    EventCfg,
    MobileAIReachSceneCfg,
)

from . import mdp as lift_mdp


@configclass
class EmptyCommandsCfg:
    """No random EE pose commands during IL collection or rollout."""

    pass


@configclass
class LiftObservationsCfg:
    """14D joint state for IL-compatible observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=lift_mdp.record_joint_pos_14)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LiftRewardsCfg:
    """Sparse shaping rewards for debugging; not used during IL training."""

    cube_lifted = RewTerm(func=lift_mdp.cube_is_lifted, weight=1.0)
    cube_placed = RewTerm(func=lift_mdp.cube_is_placed, weight=2.0)


@configclass
class LiftTerminationsCfg:
    """Episode termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=lift_mdp.cube_dropped)


@configclass
class MobileAILiftEnvCfg(ManagerBasedRLEnvCfg):
    """Pick up a cube, lift it, and place it back on the table."""

    scene: MobileAIReachSceneCfg = MobileAIReachSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: LiftObservationsCfg = LiftObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: EmptyCommandsCfg = EmptyCommandsCfg()
    events: EventCfg = EventCfg()
    rewards: LiftRewardsCfg = LiftRewardsCfg()
    terminations: LiftTerminationsCfg = LiftTerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 60.0
