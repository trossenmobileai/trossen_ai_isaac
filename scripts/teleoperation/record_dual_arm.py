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

"""Dual-arm teleoperation with LeRobot episode recording (keyboard / gamepad).

Uses ``Isaac-Reach-MobileAI-Record-Play-v0`` for 14D joint obs and three RGB cameras.
Teleop logic is shared with ``teleop_dual_arm_switch.py`` via ``trossen_ai_isaac.teleop``.
"""

import argparse
import logging

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Record Mobile AI dual-arm teleop demonstrations to a LeRobot dataset."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleop device: keyboard, gamepad, or spacemouse.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-MobileAI-Record-Play-v0",
    help="Gym task id (must include cameras and 14D joint observations).",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity scale factor.")
parser.add_argument(
    "--gamepad_dead_zone",
    type=float,
    default=0.15,
    help="Per-axis dead zone for gamepad stick drift (default 0.15).",
)
parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="LeRobot dataset repo id, e.g. YourUser/trossen_ai_sim_reach.",
)
parser.add_argument(
    "--root",
    type=str,
    default=None,
    help="Optional local root directory for the dataset (defaults to HF cache).",
)
parser.add_argument(
    "--fps",
    type=int,
    default=60,
    help="Dataset FPS; also sets env decimation (60 -> decimation 1, 30 -> decimation 2).",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="mobile_ai_reach",
    help="Natural-language task label stored in each frame.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg
from trossen_ai_isaac.teleop.se3_switch import run_se3_switch_loop

logger = logging.getLogger(__name__)


def main() -> None:
    """Run switchable teleop and write LeRobot episodes to disk."""
    env_cfg = make_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        fps=args_cli.fps,
    )
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error("Failed to create environment: %s", exc)
        simulation_app.close()
        return

    try:
        recorder = LeRobotRecorder(
            repo_id=args_cli.repo_id,
            fps=args_cli.fps,
            task=args_cli.task_description,
            root=args_cli.root,
        )
    except ImportError as exc:
        logger.error("%s", exc)
        env.close()
        simulation_app.close()
        return

    run_se3_switch_loop(simulation_app, env, env_cfg, args_cli, recorder=recorder)


if __name__ == "__main__":
    main()
    simulation_app.close()
