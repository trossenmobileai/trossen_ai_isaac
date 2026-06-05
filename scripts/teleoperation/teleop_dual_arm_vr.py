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

"""VR hand-tracking teleoperation for the Mobile AI bimanual robot.

This script is purpose-built for VR. Both arms are controlled simultaneously:
    left  hand  -> left  arm  (follower_left_link_6)
    right hand  -> right arm  (follower_right_link_6)

There is no arm switching, no dead-zone, and no delta computation — each hand pose
flows straight through `Se3AbsRetargeter` into a 14D absolute pose action that the
env's differential IK solver consumes directly.

Pipeline:
    OpenXRDevice.advance()
        -> torch.Tensor of shape [16] in the order declared in MobileAIReachEnvCfg_IK_ABS:
           [L_pose(7), R_pose(7), L_grip(1), R_grip(1)]
    -> slice [:14] = [L_pose, R_pose]              (env action)
    -> broadcast to [num_envs, 14] and env.step()

The two gripper retargeter outputs are intentionally NOT fed to the env because the
env's ActionsCfg has no gripper_action term yet. They are logged for future use.

Gestures (provided by Isaac Lab's OpenXR teleop UI in the headset):
    START : begin streaming hand poses to the robot
    STOP  : pause (robot holds last commanded pose)
    RESET : reset the env to its initial pose

Prerequisites on the workstation:
    * Isaac Lab installed and `./isaaclab.sh -p ...` available
    * ALVR running, Meta Quest 3 connected, SteamVR providing the OpenXR runtime
    * Run via `./isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py ...`

Launch Isaac Sim Simulator first.
"""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="VR hand-tracking teleoperation for the Mobile AI bimanual robot."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Reach-MobileAI-IK-Abs-Play-v0",
    help="Name of the task. Must be an absolute-IK variant (14D action space).",
)
parser.add_argument(
    "--device_name",
    type=str,
    default="handtracking",
    help="Key into env_cfg.teleop_devices.devices selecting the OpenXR device entry.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force XR runtime ON before constructing AppLauncher.
app_launcher_args = vars(args_cli)
app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# --- Imports after sim launch ---
import logging

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

# Action layout produced by the four retargeters declared in
# MobileAIReachEnvCfg_IK_ABS.teleop_devices.devices["handtracking"]:
#   indices 0..6   -> left  arm pose [pos_xyz, quat_wxyz]
#   indices 7..13  -> right arm pose [pos_xyz, quat_wxyz]
#   index   14     -> left  gripper scalar
#   index   15     -> right gripper scalar
ACTION_DIM_PER_ARM = 7
ACTION_DIM_ENV = 2 * ACTION_DIM_PER_ARM  # 14D goes to env.step
RETARGETER_OUTPUT_DIM = 2 * ACTION_DIM_PER_ARM + 2  # 16D coming out of advance()
LEFT_GRIP_IDX = 14
RIGHT_GRIP_IDX = 15


def main() -> None:
    """Run dual-arm VR teleoperation."""
    # -- Environment setup --
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    # XR cannot share the render pipeline with USD cameras — strip any camera configs
    # the env may have declared, and switch to DLSS for VR-friendly anti-aliasing.
    env_cfg = remove_camera_configs(env_cfg)
    env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Validate that the selected task actually exposes a handtracking device. Failing
    # loud here is much friendlier than a cryptic AttributeError 200 lines later.
    if not hasattr(env_cfg, "teleop_devices") or args_cli.device_name not in env_cfg.teleop_devices.devices:
        logger.error(
            f"Task '{args_cli.task}' does not declare a teleop device named '{args_cli.device_name}'. "
            "Use one of the Isaac-Reach-MobileAI-IK-Abs-* tasks, which register the 'handtracking' device."
        )
        simulation_app.close()
        return

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Sanity-check the env's action width. If a relative-IK task slipped through,
    # advance() will still produce 16D output but env.step() expects 12D and will throw.
    try:
        action_dim = env.action_manager.total_action_dim
    except AttributeError:
        action_dim = None
    if action_dim is not None and action_dim != ACTION_DIM_ENV:
        logger.warning(
            f"Env action_dim={action_dim} but this script assembles {ACTION_DIM_ENV}D actions. "
            "You probably selected a non-IK-Abs task; expect a shape mismatch on env.step()."
        )

    # -- State flags --
    should_reset = False
    teleoperation_active = False  # VR convention: user pinches START to begin

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("[RESET] Environment will reset on next step")

    def start_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("[TELEOP] Activated (left+right hands -> left+right arms)")

    def stop_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("[TELEOP] Deactivated (robot holds last pose)")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "START": start_teleop,
        "STOP": stop_teleop,
        "RESET": reset_env,
    }

    # -- Build OpenXR device via the env-config-driven factory --
    try:
        teleop_interface = create_teleop_device(
            args_cli.device_name,
            env_cfg.teleop_devices.devices,
            teleoperation_callbacks,
        )
    except Exception as e:
        logger.error(f"Failed to create OpenXR teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    print(f"\nUsing teleop device: {teleop_interface}")
    print("=" * 60)
    print("VR DUAL-ARM TELEOPERATION (hand tracking)")
    print(f"  Task:        {args_cli.task}")
    print(f"  Action dim:  {ACTION_DIM_ENV} (left 7D + right 7D, absolute IK)")
    print("  Gestures:    START / STOP / RESET (pinch in the headset UI)")
    print("=" * 60)

    # -- Reset --
    env.reset()
    teleop_interface.reset()

    step_count = 0

    # -- Main loop --
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # 1D tensor of shape [16]: [L_pose(7), R_pose(7), L_grip(1), R_grip(1)]
                raw = teleop_interface.advance()

                if raw is None or raw.numel() < ACTION_DIM_ENV:
                    # No XR data yet (operator hasn't put the headset on, or tracking
                    # is briefly lost). Render to keep the window responsive and try
                    # again next frame.
                    env.sim.render()
                    step_count += 1
                    continue

                action_14d = raw[:ACTION_DIM_ENV].to(dtype=torch.float32, device=args_cli.device)

                # Gripper values are captured for logging / future wiring. They are
                # NOT inserted into the env action because ActionsCfg.gripper_action
                # is currently None.
                left_grip = float(raw[LEFT_GRIP_IDX].item()) if raw.numel() > LEFT_GRIP_IDX else 0.0
                right_grip = float(raw[RIGHT_GRIP_IDX].item()) if raw.numel() > RIGHT_GRIP_IDX else 0.0

                if teleoperation_active:
                    actions = action_14d.unsqueeze(0).repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    # Render only — the robot holds whatever pose the IK target last
                    # settled on. Without this, the viewer freezes while paused.
                    env.sim.render()

                if step_count % 60 == 0:
                    l_pos = action_14d[:3].tolist()
                    r_pos = action_14d[7:10].tolist()
                    state = "ON" if teleoperation_active else "PAUSED"
                    print(
                        f"[VR step={step_count} {state}] "
                        f"L_pos={[f'{v:+.3f}' for v in l_pos]} "
                        f"R_pos={[f'{v:+.3f}' for v in r_pos]} "
                        f"L_grip={left_grip:+.2f} R_grip={right_grip:+.2f}"
                    )
                step_count += 1

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    print("[RESET] Done")

        except Exception as e:
            logger.error(f"Simulation step error: {e}")
            break

    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
