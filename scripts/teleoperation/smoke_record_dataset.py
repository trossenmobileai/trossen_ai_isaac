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

"""Automated short recording smoke test (no teleop).

Steps the record env with zero actions, saves one episode, finalizes the dataset,
and prints the dataset root for follow-up verification with verify_recorded_dataset.py.
"""

import argparse
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Automated one-episode LeRobot recording smoke test.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-MobileAI-Record-Play-v0",
    help="Gym task id for the record environment.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=60,
    help="Number of zero-action steps to record before saving the episode.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="trossen-admin/trossen_ai_sim_reach_smoke",
    help="LeRobot dataset repo id.",
)
parser.add_argument(
    "--root",
    type=str,
    default="/tmp/trossen_ai_sim_reach_smoke",
    help="Local root directory for the dataset.",
)
parser.add_argument(
    "--fps",
    type=int,
    default=60,
    help="Dataset FPS.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Remove existing dataset root before recording.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import trossen_ai_isaac.tasks  # noqa: F401
from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg


def main() -> None:
    """Record one short zero-action episode and finalize the dataset."""
    root = Path(args_cli.root).expanduser()
    if args_cli.overwrite and root.exists():
        shutil.rmtree(root)

    env_cfg = make_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        fps=args_cli.fps,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    recorder = None
    try:
        recorder = LeRobotRecorder(
            repo_id=args_cli.repo_id,
            fps=args_cli.fps,
            task="mobile_ai_reach_smoke",
            root=str(root),
        )
        env.reset()
        action_dim = env.action_manager.total_action_dim
        zero_action = torch.zeros(env.num_envs, action_dim, device=env.device)
        for _ in range(args_cli.num_steps):
            env.step(zero_action)
            recorder.on_step(env)
        recorder.save_episode()
        print(f"Recorded {args_cli.num_steps} frames to {recorder.dataset_root}")
    finally:
        if recorder is not None:
            recorder.finalize()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
