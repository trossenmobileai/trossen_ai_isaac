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

Recording controls (keyboard):
    N  — toggle episode recording (start / save + reset arms)
    M  — discard current episode buffer without saving
    R  — reset environment (discards in-progress recording)
"""

import argparse
import logging
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_scripts_dir = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
sys.path.insert(0, str(_scripts_dir / "lib"))
from teleop_cli_loader import load_teleop_cli

teleop_cli = load_teleop_cli()

parser = argparse.ArgumentParser(
    description="Record Mobile AI dual-arm teleop demonstrations to a LeRobot dataset."
)
teleop_cli.add_mobile_ai_teleop_args(parser)
parser.set_defaults(task="Isaac-Reach-MobileAI-Record-Play-v0")
teleop_cli.add_record_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder
from trossen_ai_isaac.recording.runtime import install_recording_signal_handlers, run_recording_session
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg
from trossen_ai_isaac.teleop.se3_switch import run_se3_switch_loop

logger = logging.getLogger(__name__)

install_recording_signal_handlers()


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
        return

    try:
        recorder = LeRobotRecorder(
            repo_id=args_cli.repo_id,
            fps=args_cli.fps,
            task=args_cli.task_description,
            root=args_cli.root,
            overwrite=args_cli.overwrite,
        )
    except (ImportError, FileExistsError) as exc:
        logger.error("%s", exc)
        env.close()
        return

    try:
        run_recording_session(
            simulation_app,
            env,
            env_cfg,
            args_cli,
            recorder,
            run_se3_switch_loop,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
