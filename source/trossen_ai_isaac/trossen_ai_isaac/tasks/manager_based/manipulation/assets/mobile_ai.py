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

"""Configuration for Trossen Mobile AI dual-arm platform.

This module provides Isaac Lab ArticulationCfg configurations for:
- MOBILE_AI_CFG: Base configuration with physics simulation
- MOBILE_AI_HIGH_PD_CFG: Configuration with gravity disabled for IK control
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get the path to assets relative to this package
# Path: source/trossen_ai/trossen_ai/tasks/manager_based/manipulation/assets/
_ASSETS_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "..",
        "..",
        "assets",
        "robots",
    )
)

MOBILE_AI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSETS_ROOT, "mobile_ai", "mobile_ai.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "follower_left_joint_0": 0.0,
            "follower_left_joint_1": 0.0,
            "follower_left_joint_2": 0.0,
            "follower_left_joint_3": 0.0,
            "follower_left_joint_4": 0.0,
            "follower_left_joint_5": 0.0,
            "follower_left_left_carriage_joint": 0.0,
            "follower_right_joint_0": 0.0,
            "follower_right_joint_1": 0.0,
            "follower_right_joint_2": 0.0,
            "follower_right_joint_3": 0.0,
            "follower_right_joint_4": 0.0,
            "follower_right_joint_5": 0.0,
            "follower_right_left_carriage_joint": 0.0,
        },
    ),
    # mobile_ai.usd already has stiffness, damping and effort_limit parameters
    actuators={
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_joint_[0-5]"],
            stiffness=None,
            damping=None,
        ),
        # right_carriage_joint is a mimic joint specified in USD file
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_left_carriage_joint"],
            stiffness=None,
            damping=None,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["follower_right_joint_[0-5]"],
            stiffness=None,
            damping=None,
        ),
        # right_carriage_joint is a mimic joint specified in USD file
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_right_left_carriage_joint"],
            stiffness=None,
            damping=None,
        ),
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel", "right_wheel"],
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Configuration for Trossen WXAI robot arm with standard physics.
MOBILE_AI_HIGH_PD_CFG = MOBILE_AI_CFG.copy()
MOBILE_AI_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MOBILE_AI_HIGH_PD_CFG.actuators["left_arm"].stiffness = 400.0
MOBILE_AI_HIGH_PD_CFG.actuators["left_arm"].damping = 80.0
MOBILE_AI_HIGH_PD_CFG.actuators["right_arm"].stiffness = 400.0
MOBILE_AI_HIGH_PD_CFG.actuators["right_arm"].damping = 80.0


__all__ = ["MOBILE_AI_CFG", "MOBILE_AI_HIGH_PD_CFG"]
