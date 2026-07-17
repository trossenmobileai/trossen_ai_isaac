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

"""Closed-loop policy rollout in the Mobile AI joint-position environment.

Shared by ACT and Pi0 via ``run_play_act.sh`` / ``run_play_pi0.sh`` and
``play_act.py``. Eval thresholds and early-stop rules are locked in
``lift/mdp/metrics.py`` (EVAL CONTRACT).

Wired for the 7D right-arm checkpoint from ``--record_arm right``: the policy
receives 7D state (right arm joints) + cam_high + cam_right_wrist and outputs
7D right-arm joint targets. The left arm is held at the fixed eval start pose
every step (not moved by the policy).
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
    EpisodeCubeTracker,
    evaluate_episode_metrics,
)
from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.mdp.observations import (
    RECORD_JOINT_NAMES,
)
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg


@dataclass
class RolloutSummary:
    episodes: int
    successes: int
    success_rate: float
    success_rate_by_color: dict[str, dict[str, float | int]]
    results: list[dict]


def _success_rate_by_color(results: list[dict]) -> dict[str, dict[str, float | int]]:
    """Aggregate per-color episode counts and success rates."""
    by_color: dict[str, dict[str, float | int]] = {}
    for row in results:
        color = str(row.get("cube_color", "unknown"))
        bucket = by_color.setdefault(color, {"episodes": 0, "successes": 0, "success_rate": 0.0})
        bucket["episodes"] = int(bucket["episodes"]) + 1
        if row.get("episode_success"):
            bucket["successes"] = int(bucket["successes"]) + 1
    for bucket in by_color.values():
        eps = int(bucket["episodes"])
        bucket["success_rate"] = float(bucket["successes"]) / max(eps, 1)
    return dict(sorted(by_color.items()))


# The 7D right-arm recording layout (cam_high + cam_right_wrist, joint indices 7-13).
_RIGHT_LAYOUT = RecordingLayout.from_arm_mode("right")

# Fixed eval start pose (EVAL CONTRACT — see lift/mdp/metrics.py): arms near home,
# grippers open (matches recorded demos).
_GRIPPER_OPEN = 0.044
# Hold start pose before policy runs (~0.5 s at 60 FPS). Not counted in episode metrics.
_START_POSE_WARMUP_STEPS = 30
_EVAL_START_POSE_14 = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _GRIPPER_OPEN,  # left gripper
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _GRIPPER_OPEN,  # right gripper
    ],
    dtype=np.float32,
)


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

    The env uses joint-position control over both arms (14D total).  The left
    arm is held at the fixed eval start pose every step (not the measured
    state), so physics coupling from the right arm cannot accumulate into a
    permanent left EE tilt.  The right arm is set to the policy target.
    """
    full_action = _EVAL_START_POSE_14.copy()
    full_action[7:14] = action_7d[:7]
    return torch.as_tensor(full_action, device=env.device, dtype=torch.float32).unsqueeze(0)


def _start_pose_action(env) -> torch.Tensor:
    """14D start-pose action for all envs."""
    pose = torch.as_tensor(_EVAL_START_POSE_14, device=env.device, dtype=torch.float32)
    return pose.unsqueeze(0).expand(env.num_envs, -1).contiguous()


def _force_eval_start_pose(env) -> None:
    """Snap both arms and grippers to a fixed start pose after ``env.reset()``.

    Writes joint state and position targets in one shot (no settle ``env.step``),
    then seeds the action manager so leftover zero actions (closed gripper /
    zero joints) are not applied on the next physics step.
    """
    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in RECORD_JOINT_NAMES]
    pose = _start_pose_action(env)
    zero_vel = torch.zeros_like(pose)

    robot.write_joint_state_to_sim(pose, zero_vel, joint_ids=joint_ids)
    robot.set_joint_position_target(pose, joint_ids=joint_ids)
    robot.write_data_to_sim()
    env.scene.write_data_to_sim()
    env.sim.forward()

    # Action manager resets to zeros (closed grippers). Seed it with the start
    # pose so the first policy step does not fight stale zero targets.
    env.action_manager.process_action(pose)
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.forward()


def _warmup_at_start_pose(
    env,
    *,
    warmup_steps: int = _START_POSE_WARMUP_STEPS,
    simulation_app=None,
) -> None:
    """Hold the fixed start pose for a few steps before policy inference.

    Lets PD settle and gives a visible stable home pose in ``--visual`` mode.
    These steps are not counted toward episode metrics / early-stop budgets.
    """
    if warmup_steps <= 0:
        return
    hold = _start_pose_action(env)
    for _ in range(warmup_steps):
        env.step(hold)
        if simulation_app is not None:
            simulation_app.update()


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
    """Roll out an ACT or Pi0 checkpoint in simulation via the policy sidecar.

    The checkpoint must have been trained on right-arm-only data (7D state,
    cam_high + cam_right_wrist images).  The left arm is held at its fixed
    eval start pose for the entire episode. Eval thresholds: see EVAL CONTRACT
    in ``lift/mdp/metrics.py``.
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
                _force_eval_start_pose(env)
                _warmup_at_start_pose(env, simulation_app=simulation_app)
                client.reset()
                tracker = EpisodeCubeTracker()
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
                    tracker.update(env, steps)
                    env_done = bool(terminated[0].item() or truncated[0].item())
                    tracker_stop = tracker.should_stop(steps)
                    done = env_done or tracker_stop
                    if env_done and tracker.stop_reason == "running":
                        tracker.stop_reason = "env_done"
                    if simulation_app is not None:
                        simulation_app.update()

                metrics = evaluate_episode_metrics(env, tracker=tracker)
                episode_result = {
                    "episode": episode_idx,
                    "steps": steps,
                    **asdict(metrics),
                }
                results.append(episode_result)
                print(
                    f"[EP {episode_idx + 1}/{num_episodes}] success={metrics.episode_success} "
                    f"color={metrics.cube_color} lifted={metrics.cube_lifted} "
                    f"returned={metrics.cube_returned_after_lift} "
                    f"on_table={metrics.cube_on_table} stop={metrics.stop_reason} steps={steps}"
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
    by_color = _success_rate_by_color(results)
    summary = RolloutSummary(
        episodes=num_episodes,
        successes=successes,
        success_rate=successes / max(num_episodes, 1),
        success_rate_by_color=by_color,
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
    for color, stats in by_color.items():
        print(
            f"  color={color}: {stats['successes']}/{stats['episodes']} "
            f"({float(stats['success_rate']) * 100:.1f}%)"
        )
    return summary
