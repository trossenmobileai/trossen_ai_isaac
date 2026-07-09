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

"""Reach environment configuration for Mobile AI dual-arm IL teleoperation."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, Usd
import random

from trossen_ai_isaac.tasks.manager_based.manipulation.assets import MOBILE_AI_HIGH_PD_CFG


##
# Scene definition
##

@configclass
class MobileAIReachSceneCfg(InteractiveSceneCfg):
    """Mobile AI reach scene: robot, ground, light, table, and interaction cube."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.99, 2.0, 0.807),  # Width, Length, Height of the table
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)), # Grey table
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.85, 0.0, 0.4035),  # Positioned right in front of the robot base
        ),
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.07),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.85, 0.0, 0.842), # Placed perfectly on the surface of the table
        ),
    )

    robot: ArticulationCfg = MOBILE_AI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=MOBILE_AI_HIGH_PD_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            )
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Goal pose commands — used as task specification in observations."""

    ee_pose_left = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="follower_left_link_6",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.15, 0.15),
            pos_z=(0.1, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )

    ee_pose_right = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="follower_right_link_6",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.15, 0.15),
            pos_z=(0.1, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications — what gets recorded during demonstrations.

    Term declaration order defines the env action-vector layout (the
    ActionManager concatenates terms in order and skips any left as None). With
    the gripper terms last, an env that wires both arms plus both grippers gets:
        [left_arm | right_arm | left_gripper | right_gripper]
    e.g. 14D absolute IK + 2D binary grippers = 16D for VR teleop.
    """

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    # Legacy single-gripper slot, kept for compatibility; unused by the dual-arm
    # configs which declare explicit left/right gripper terms below.
    gripper_action: ActionTerm | None = None
    left_gripper_action: ActionTerm | None = None
    right_gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications — what the policy sees."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        pose_command_left = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose_left"}
        )
        pose_command_right = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose_right"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()




def randomize_cube_color_discrete(env, env_ids, asset_cfg: SceneEntityCfg, colors: list[tuple[float, float, float]]):
    """Sets each cube instance to one randomly chosen solid RGB color from `colors`
    (a true discrete pick — e.g. pure red OR pure green OR pure blue, not a blend)."""
    asset = env.scene[asset_cfg.name]
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    stage = get_current_stage()

    if env_ids is None:
        env_ids = range(len(prim_paths))

    for env_id in env_ids:
        prim_path = prim_paths[int(env_id)]
        cube_prim = stage.GetPrimAtPath(prim_path)
        chosen_rgb = random.choice(colors)
        for descendant in Usd.PrimRange(cube_prim):
            attr = descendant.GetAttribute("inputs:diffuseColor")
            if attr and attr.IsValid():
                attr.Set(Gf.Vec3f(*chosen_rgb))

@configclass
class EventCfg:
    """Reset events between demonstrations."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    
    randomize_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.10, 0.0),
                "y": (-0.20, 0.20),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

    randomize_cube_color = EventTerm(
        func=randomize_cube_color_discrete,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "colors": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        },
    )



    
@configclass
class RewardsCfg:
    """Placeholder — not used for IL, required by ManagerBasedRLEnvCfg."""
    pass


@configclass
class TerminationsCfg:
    """Placeholder — not used for IL, required by ManagerBasedRLEnvCfg."""
    pass


##
# Environment configuration
##

@configclass
class MobileAIReachEnvCfg(ManagerBasedRLEnvCfg):
    """Dual-arm Mobile AI environment for IL teleoperation and data collection."""

    scene: MobileAIReachSceneCfg = MobileAIReachSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                    pos_sensitivity=0.4,
                    rot_sensitivity=0.8,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
            }
        )

        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["follower_left_joint_[0-5]"],
            body_name="follower_left_link_6",
            debug_vis=True,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="pinv",
            ),
            scale=0.05,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["follower_right_joint_[0-5]"],
            body_name="follower_right_link_6",
            debug_vis=True,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="pinv",
            ),
            scale=0.05,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )


@configclass
class MobileAIReachEnvCfg_PLAY(MobileAIReachEnvCfg):
    """Play/teleoperation config — 1 env, no noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False