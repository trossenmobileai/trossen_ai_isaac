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

"""Dual-arm switchable teleoperation script for the Mobile AI robot.

Controls the left and right arms one at a time using a single input device.
Toggle between arms with:
  - Keyboard: TAB key
  - Gamepad:  Y button

Action space is always 12D: [left_6D | right_6D]
The inactive arm slot is zeroed out every step.

Launch Isaac Sim Simulator first.
"""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Dual-arm switchable teleoperation for Mobile AI robot."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleop device: keyboard, spacemouse, or gamepad.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# --- Imports after sim launch ---
import logging

import carb
import gymnasium as gym
import torch
import numpy as np
import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

# Arm index constants — matches the order actions are defined in ActionsCfg:
#   left_arm_action  → indices [0:6]
#   right_arm_action → indices [6:12]
LEFT_ARM = "left"
RIGHT_ARM = "right"
ARM_SLICE = {
    LEFT_ARM:  slice(0, 6),
    RIGHT_ARM: slice(6, 12),
}


def main() -> None:
    step_count = 0
    """Run dual-arm switchable teleoperation."""
    last_action = {
        LEFT_ARM:  torch.zeros(6, dtype=torch.float32, device=args_cli.device),
        RIGHT_ARM: torch.zeros(6, dtype=torch.float32, device=args_cli.device),
    }

    # -- Environment setup --
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # -- State flags --
    active_arm = LEFT_ARM          # which arm the device currently controls
    should_reset = False
    teleoperation_active = True

    # -- Callbacks --
    def toggle_arm() -> None:
        """Switch the active arm between left and right."""
        nonlocal active_arm
        active_arm = RIGHT_ARM if active_arm == LEFT_ARM else LEFT_ARM
        print(f"[ARM SWITCH] Now controlling: {active_arm.upper()} arm")

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("[RESET] Environment will reset on next step")

    def start_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("[TELEOP] Activated")

    def stop_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("[TELEOP] Deactivated")

    # Standard callbacks passed to create_teleop_device
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R":     reset_env,
        "START": start_teleop,
        "STOP":  stop_teleop,
        "RESET": reset_env,
    }

    # -- Build teleop device --
    teleop_interface = None
    sensitivity = args_cli.sensitivity

    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device,
                env_cfg.teleop_devices.devices,
                teleoperation_callbacks,
            )
        else:
            # Fallback: build device manually
            logger.warning(f"Device '{args_cli.teleop_device}' not in env config — creating default.")
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(
                        pos_sensitivity=0.05 * sensitivity,
                        rot_sensitivity=0.05 * sensitivity,
                    )
                )
                for key, cb in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, cb)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not bind '{key}': {e}")

            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(
                        pos_sensitivity=0.1 * sensitivity,
                        rot_sensitivity=0.1 * sensitivity,
                    )
                )
                for key, cb in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, cb)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not bind '{key}': {e}")

            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(
                        pos_sensitivity=0.05 * sensitivity,
                        rot_sensitivity=0.05 * sensitivity,
                    )
                )
                for key, cb in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, cb)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not bind '{key}': {e}")
            else:
                logger.error(f"Unsupported device: {args_cli.teleop_device}")
                env.close()
                simulation_app.close()
                return

    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("No teleop interface created.")
        env.close()
        simulation_app.close()
        return

    # -- Bind arm-toggle key AFTER device is created --
    # Keyboard: TAB key
    # Gamepad:  Y button
    try:
        if isinstance(teleop_interface, Se3Keyboard):
            teleop_interface.add_callback("TAB", toggle_arm)
            print("[BINDINGS] TAB = toggle active arm")
        elif isinstance(teleop_interface, Se3Gamepad):
            teleop_interface.add_callback(carb.input.GamepadInput.Y, toggle_arm)
            print("[BINDINGS] Y button = toggle active arm")
    except Exception as e:
        logger.warning(f"Could not bind arm toggle: {e}")

    print(f"\nUsing teleop device: {teleop_interface}")
    print("=" * 50)
    print("DUAL ARM SWITCH TELEOPERATION")
    print(f"  Active arm: {active_arm.upper()} (toggle with TAB / Y button)")
    print("  R = reset environment")
    print("=" * 50)

    # -- Reset --
    env.reset()
    teleop_interface.reset()

    # -- Main loop --
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # Get 7D delta from device: [dx, dy, dz, drx, dry, drz, gripper]
                device_output = teleop_interface.advance()
                
                # Dead-zone: prevent IK drift from numerical noise
                if device_output[:6].abs().max().item() < 1e-3:
                    device_output[:6] = 0.0
                
                #print(f"device raw: {device_output[:6].tolist()}")

                if teleoperation_active:
                    full_action = torch.zeros(12, dtype=torch.float32, device=args_cli.device)

                    # Active arm: use live device input
                    delta = device_output[:6]
                    if delta.abs().max().item() > 1e-3:
                        last_action[active_arm] = delta
                    else:
                        last_action[active_arm] = torch.zeros(6, dtype=torch.float32, device=args_cli.device)

                    # Always fill both arms — idle arm gets zero delta to hold IK target
                    full_action[ARM_SLICE[LEFT_ARM]]  = last_action[LEFT_ARM]
                    full_action[ARM_SLICE[RIGHT_ARM]] = last_action[RIGHT_ARM]

                    actions = full_action.unsqueeze(0).repeat(env.num_envs, 1)

                    if step_count % 60 == 0:  # about once per second
                        print(f"[DEBUG] step {step_count}, full_action = {full_action.tolist()}")
                    
                    env.step(actions)   # always step — zero delta = hold position
                    step_count += 1
                    
                else:
                    zero_action = torch.zeros(12, dtype=torch.float32, device=args_cli.device)
                    if step_count % 60 == 0:
                        print(f"[DEBUG] step {step_count}, full_action = {zero_action.tolist()}")
                    env.step(zero_action.unsqueeze(0).repeat(env.num_envs, 1))
                    step_count += 1

                if should_reset:
                    env.reset()
                    # Debug: print initial EE pose
                    robot = env.unwrapped.scene["robot"]
                    body_idx = robot.find_bodies("follower_left_link_6")[0]
                    print("Initial EE quat:", robot.data.body_quat_w[0, body_idx].tolist())
                    teleop_interface.reset()
                    should_reset = False
                    print(f"[RESET] Done — controlling {active_arm.upper()} arm")

        except Exception as e:
            logger.error(f"Simulation step error: {e}")
            break

    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
