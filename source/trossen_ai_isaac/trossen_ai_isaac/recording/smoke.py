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

"""Shared zero-action smoke helpers for the Mobile AI record environment."""

from __future__ import annotations

import shutil
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def run_zero_action_env_smoke(
    task: str,
    *,
    device: str,
    num_envs: int = 1,
    num_steps: int = 60,
) -> None:
    """Step the record env with zero actions and validate obs/camera shapes."""
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    policy_obs = obs["policy"]
    print(f"obs['policy'] shape: {tuple(policy_obs.shape)} (expected: ({env.num_envs}, 14))")

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(env.num_envs, action_dim, device=env.device)

    for _ in range(num_steps):
        obs, _, _, _, _ = env.step(zero_action)

    policy_obs = obs["policy"]
    print(f"obs['policy'] after {num_steps} steps: {tuple(policy_obs.shape)}")

    for cam_name in CAMERA_KEYS:
        rgb = env.scene[cam_name].data.output["rgb"]
        print(f"{cam_name} rgb shape: {tuple(rgb.shape)} (expected: ({env.num_envs}, 480, 640, 3))")
        rgb_np = rgb[0].detach().cpu().numpy()
        if rgb_np.max() <= 10:
            raise AssertionError(f"{cam_name} appears black (max={rgb_np.max()})")
        print(f"{cam_name} pixel range: {rgb_np.min()}–{rgb_np.max()}")

    env.close()
    print("Smoke test passed.")


def run_zero_action_dataset_smoke(
    task: str,
    *,
    device: str,
    num_envs: int = 1,
    num_steps: int = 60,
    repo_id: str,
    root: str | Path,
    fps: int = 60,
    task_description: str = "mobile_ai_reach_smoke",
    overwrite: bool = False,
) -> Path:
    """Record one short zero-action episode and finalize the dataset."""
    root_path = Path(root).expanduser()
    if overwrite and root_path.exists():
        shutil.rmtree(root_path)

    env_cfg = make_env_cfg(task, device=device, num_envs=num_envs, fps=fps)
    env = gym.make(task, cfg=env_cfg).unwrapped
    recorder = None
    try:
        recorder = LeRobotRecorder(
            repo_id=repo_id,
            fps=fps,
            task=task_description,
            root=str(root_path),
            overwrite=overwrite,
        )
        env.reset()
        action_dim = env.action_manager.total_action_dim
        zero_action = torch.zeros(env.num_envs, action_dim, device=env.device)
        for _ in range(num_steps):
            env.step(zero_action)
            recorder.on_step(env)
        recorder.save_episode()
        print(f"Recorded {num_steps} frames to {recorder.dataset_root}")
        return Path(recorder.dataset_root)
    finally:
        if recorder is not None:
            recorder.finalize()
        env.close()
