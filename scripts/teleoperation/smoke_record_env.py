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

"""Smoke test for the Mobile AI IL record environment.

Validates 14D policy observations and three 640x480 RGB camera streams without VR.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for Mobile AI record environment.")
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
    help="Number of zero-action steps to run after reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def main() -> None:
    """Create the record env, step with zero actions, and print obs/camera shapes."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    policy_obs = obs["policy"]
    print(f"obs['policy'] shape: {tuple(policy_obs.shape)} (expected: ({env.num_envs}, 14))")

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(env.num_envs, action_dim, device=env.device)

    for _ in range(args_cli.num_steps):
        obs, _, _, _, _ = env.step(zero_action)

    policy_obs = obs["policy"]
    print(f"obs['policy'] after {args_cli.num_steps} steps: {tuple(policy_obs.shape)}")

    for cam_name in CAMERA_KEYS:
        rgb = env.scene[cam_name].data.output["rgb"]
        print(f"{cam_name} rgb shape: {tuple(rgb.shape)} (expected: ({env.num_envs}, 480, 640, 3))")

    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
