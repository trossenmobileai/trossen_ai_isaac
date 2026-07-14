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

"""Closed-loop ACT rollout in the Mobile AI joint-position environment.

This module is wired for the 7D right-arm checkpoint produced by the
single-arm recording pipeline (``--record_arm right``).  The policy
receives 7D state (right arm joints) + cam_high + cam_right_wrist and
outputs 7D right-arm joint targets.  The left arm is held at whatever
position it has after the env reset (it is not moved).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from trossen_ai_isaac.evaluation.policy_client import PolicySidecarClient, build_sidecar_command
from trossen_ai_isaac.recording.frame_capture import capture_images, capture_state
from trossen_ai_isaac.recording.schema import RIGHT_JOINT_NAMES, RecordingLayout
from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.lift.mdp.metrics import (
    cube_height_w,
    evaluate_episode_metrics,
)
from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.mdp.observations import (
    record_joint_pos_14,
)
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg


@dataclass
class RolloutSummary:
    episodes: int
    successes: int
    success_rate: float
    results: list[dict]


# The 7D right-arm recording layout (cam_high + cam_right_wrist, joint indices 7-13).
_RIGHT_LAYOUT = RecordingLayout.from_arm_mode("right")


def _build_policy_observation(env) -> dict[str, np.ndarray]:
    """Build the 7D right-arm observation dict expected by the ACT policy."""
    state = capture_state(env, _RIGHT_LAYOUT)        # (7,) float32
    images = capture_images(env, _RIGHT_LAYOUT)       # cam_high + cam_right_wrist
    obs: dict[str, np.ndarray] = {"observation.state": state}
    for cam_key, img in images.items():
        obs[f"observation.images.{cam_key}"] = img
    return obs


def _policy_action_to_env(action_7d: np.ndarray, env) -> torch.Tensor:
    """Map the 7D right-arm policy output to a full env action tensor.

    The env uses joint-position control over both arms (14D total).  We
    read the current 14D joint state, keep the left arm (indices 0–6) at
    its current position, and set the right arm (indices 7–13) to the
    policy target.  This prevents the uncontrolled left arm from drifting.
    """
    current_14d = record_joint_pos_14(env)[0].detach().cpu().numpy().astype(np.float32)
    full_action = current_14d.copy()
    full_action[7:14] = action_7d[:7]
    return torch.as_tensor(full_action, device=env.device, dtype=torch.float32).unsqueeze(0)


# Isaac Sim sets PYTHONPATH/PYTHONHOME for its bundled Python 3.11.  If the
# sidecar subprocess inherits those, a conda Python 3.12 hits stdlib from 3.11
# and crashes with "SRE module mismatch".
_SIDECAR_ENV_STRIP = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "CARB_APP_PATH",
    "EXP_PATH",
    "ISAAC_PATH",
    "OMNI_USER",
    "OMNI_SERVER",
)


def _sidecar_subprocess_env() -> dict[str, str]:
    """Return a copy of os.environ with Isaac Sim Python paths removed."""
    env = os.environ.copy()
    for key in _SIDECAR_ENV_STRIP:
        env.pop(key, None)
    return env


def _connect_sidecar(
    client: PolicySidecarClient,
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout_s: float = 120.0,
) -> None:
    """Connect to the sidecar once it is listening.

    Uses the rollout client directly so the sidecar's single accepted
    connection is not consumed by a probe that sends shutdown.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Policy sidecar exited early with code {proc.returncode}")
        try:
            client.connect()
            return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("Timed out waiting for policy sidecar")


def run_act_rollout(
    *,
    task: str,
    policy_path: Path | str,
    device: str,
    num_envs: int = 1,
    num_episodes: int = 10,
    fps: int = 30,
    task_description: str = "Pick up the cube, lift it, and place it back on the table",
    sidecar_python: Path | str,
    sidecar_host: str = "127.0.0.1",
    sidecar_port: int = 5555,
    output_dir: Path | str | None = None,
    simulation_app=None,
) -> RolloutSummary:
    """Roll out an ACT checkpoint in simulation via the external policy sidecar.

    The checkpoint must have been trained on right-arm-only data (7D state,
    cam_high + cam_right_wrist images).  The left arm is held at its reset
    pose for the entire episode.
    """
    policy_path = Path(policy_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve() if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    sidecar_cmd = build_sidecar_command(
        policy_path,
        python_exe=sidecar_python,
        host=sidecar_host,
        port=sidecar_port,
        device=device,
    )
    sidecar_proc = subprocess.Popen(sidecar_cmd, env=_sidecar_subprocess_env())
    client = PolicySidecarClient(host=sidecar_host, port=sidecar_port)
    results: list[dict] = []

    try:
        _connect_sidecar(client, sidecar_proc, sidecar_host, sidecar_port)

        env_cfg = make_env_cfg(
            task, device=device, num_envs=num_envs, fps=fps, enable_timeout=True
        )
        env = gym.make(task, cfg=env_cfg).unwrapped

        try:
            for episode_idx in range(num_episodes):
                env.reset()
                client.reset()
                max_cube_height = float(cube_height_w(env)[0].item())
                done = False
                steps = 0

                while simulation_app is None or simulation_app.is_running():
                    if done:
                        break
                    observation = _build_policy_observation(env)
                    action_7d = client.infer(observation, task_description)
                    env_action = _policy_action_to_env(action_7d, env)
                    _obs, _reward, terminated, truncated, _info = env.step(env_action)
                    steps += 1
                    max_cube_height = max(max_cube_height, float(cube_height_w(env)[0].item()))
                    done = bool(terminated[0].item() or truncated[0].item())
                    if simulation_app is not None:
                        simulation_app.update()

                metrics = evaluate_episode_metrics(env, max_cube_height=max_cube_height)
                episode_result = {
                    "episode": episode_idx,
                    "steps": steps,
                    **asdict(metrics),
                }
                results.append(episode_result)
                print(
                    f"[EP {episode_idx + 1}/{num_episodes}] success={metrics.episode_success} "
                    f"lifted={metrics.cube_lifted} placed={metrics.cube_placed} steps={steps}"
                )
        finally:
            env.close()
    finally:
        client.close()
        sidecar_proc.terminate()
        try:
            sidecar_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sidecar_proc.kill()

    successes = sum(1 for row in results if row["episode_success"])
    summary = RolloutSummary(
        episodes=num_episodes,
        successes=successes,
        success_rate=successes / max(num_episodes, 1),
        results=results,
    )

    if output_path is not None:
        report_path = output_path / "rollout_summary.json"
        report_path.write_text(json.dumps(asdict(summary), indent=2))
        print(f"[OK] Wrote rollout summary to {report_path}")

    print(
        f"[OK] Rollout complete: {successes}/{num_episodes} successes "
        f"({summary.success_rate * 100:.1f}%)"
    )
    return summary
