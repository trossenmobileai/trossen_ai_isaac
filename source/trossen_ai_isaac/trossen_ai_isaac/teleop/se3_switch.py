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

"""Keyboard / gamepad switchable dual-arm teleoperation loop for Mobile AI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import carb
import omni.appwindow
import torch
from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.utils.math import normalize, quat_from_angle_axis, quat_mul

from trossen_ai_isaac.teleop.mobile_ai_ik_abs import (
    ACTION_DIM,
    LEFT_ARM,
    POS_SCALE,
    RIGHT_ARM,
    ROT_SCALE,
    assemble_ik_abs_action,
    broadcast_action,
    get_ee_base_pose,
    resolve_ee_body_indices,
    warn_action_dim_mismatch,
)
from trossen_ai_isaac.teleop.session import TeleopSession, shutdown_requested

if TYPE_CHECKING:
    from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder

logger = logging.getLogger(__name__)

_GP_SWITCH = carb.input.GamepadInput.Y
_GP_GRIPPER = carb.input.GamepadInput.A
_GP_RESET = carb.input.GamepadInput.B
_GP_RECORD = carb.input.GamepadInput.X


def gamepad_button_value(input_iface, gamepad, btn: carb.input.GamepadInput) -> float:
    """Return the current analog value [0..1] of a gamepad button."""
    try:
        return float(input_iface.get_gamepad_value(gamepad, btn))
    except Exception:
        return 0.0


@dataclass
class Se3SwitchState:
    """Mutable teleop state for switchable dual-arm IK control."""

    active_arm: str = LEFT_ARM
    session: TeleopSession = field(default_factory=TeleopSession)
    target_pos: dict[str, torch.Tensor] = field(default_factory=dict)
    target_quat: dict[str, torch.Tensor] = field(default_factory=dict)
    grip: dict[str, float] = field(default_factory=lambda: {LEFT_ARM: 1.0, RIGHT_ARM: 1.0})


def seed_targets_from_ee(
    state: Se3SwitchState,
    robot,
    body_idx_map: dict[str, int],
    device: str | torch.device,
) -> None:
    """Seed both arm IK targets from the current articulation state."""
    for arm, idx in body_idx_map.items():
        pos, quat = get_ee_base_pose(robot, idx)
        state.target_pos[arm] = pos.to(device)
        state.target_quat[arm] = quat.to(device)
        state.grip[arm] = 1.0


def build_se3_device(
    args: Any,
    env_cfg: Any,
    keyboard_callbacks: dict[str, Callable[[], None]],
):
    """Create the Se3 teleop device (keyboard, gamepad, or spacemouse)."""
    sensitivity = args.sensitivity
    is_gamepad = args.teleop_device.lower() == "gamepad"

    if hasattr(env_cfg, "teleop_devices") and args.teleop_device in env_cfg.teleop_devices.devices:
        cbs = {} if is_gamepad else keyboard_callbacks
        return create_teleop_device(args.teleop_device, env_cfg.teleop_devices.devices, cbs)

    logger.warning("Device '%s' not in env config — creating default.", args.teleop_device)
    device_name = args.teleop_device.lower()

    if device_name == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(
                gripper_term=False,
                pos_sensitivity=0.4 * sensitivity,
                rot_sensitivity=0.8 * sensitivity,
            )
        )
        for key, cb in keyboard_callbacks.items():
            try:
                teleop_interface.add_callback(key, cb)
            except (ValueError, TypeError) as exc:
                logger.warning("Could not bind '%s': %s", key, exc)
        return teleop_interface

    if device_name == "gamepad":
        return Se3Gamepad(
            Se3GamepadCfg(
                gripper_term=False,
                pos_sensitivity=0.1 * sensitivity,
                rot_sensitivity=0.1 * sensitivity,
            )
        )

    if device_name == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                gripper_term=False,
                pos_sensitivity=0.05 * sensitivity,
                rot_sensitivity=0.05 * sensitivity,
            )
        )
        for key, cb in keyboard_callbacks.items():
            try:
                teleop_interface.add_callback(key, cb)
            except (ValueError, TypeError) as exc:
                logger.warning("Could not bind '%s': %s", key, exc)
        return teleop_interface

    raise ValueError(f"Unsupported teleop device: {args.teleop_device}")


def _apply_device_delta(
    state: Se3SwitchState,
    robot,
    body_idx_map: dict[str, int],
    device_output: torch.Tensor,
    device: str | torch.device,
) -> None:
    """Integrate a 6D device delta into the active arm IK target."""
    delta_pos = device_output[:3].to(device)
    rot_vec = device_output[3:6].to(device)

    cur_pos, cur_quat = get_ee_base_pose(robot, body_idx_map[state.active_arm])
    state.target_pos[state.active_arm] = cur_pos.to(device) + POS_SCALE * delta_pos

    angle = rot_vec.norm()
    if angle > 1e-6:
        axis = rot_vec / angle
        dq = quat_from_angle_axis(ROT_SCALE * angle, axis)
        state.target_quat[state.active_arm] = normalize(quat_mul(dq, cur_quat.to(device)))
    else:
        state.target_quat[state.active_arm] = cur_quat.to(device)


def run_se3_switch_loop(
    simulation_app,
    env,
    env_cfg: Any,
    args: Any,
    recorder: LeRobotRecorder | None = None,
) -> None:
    """Run switchable dual-arm IK teleop; optionally record LeRobot frames each step."""
    warn_action_dim_mismatch(env)

    robot = env.scene["robot"]
    try:
        body_idx_map = resolve_ee_body_indices(robot)
    except ValueError as exc:
        logger.error(
            "Could not locate EE bodies on robot (%s). Available: %s",
            exc,
            list(robot.body_names),
        )
        env.close()
        return

    dev = args.device
    state = Se3SwitchState()
    state.target_pos = {
        LEFT_ARM: torch.zeros(3, dtype=torch.float32, device=dev),
        RIGHT_ARM: torch.zeros(3, dtype=torch.float32, device=dev),
    }
    state.target_quat = {
        LEFT_ARM: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev),
        RIGHT_ARM: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev),
    }

    def toggle_arm() -> None:
        inactive = RIGHT_ARM if state.active_arm == LEFT_ARM else LEFT_ARM
        pos, quat = get_ee_base_pose(robot, body_idx_map[inactive])
        state.target_pos[inactive] = pos.to(dev)
        state.target_quat[inactive] = quat.to(dev)
        state.active_arm = inactive
        g_state = "OPEN" if state.grip[state.active_arm] > 0 else "CLOSE"
        print(f"[ARM SWITCH] Now controlling: {state.active_arm.upper()} arm  gripper={g_state}")

    def toggle_gripper() -> None:
        state.grip[state.active_arm] = -1.0 if state.grip[state.active_arm] > 0 else 1.0
        g = "OPEN" if state.grip[state.active_arm] > 0 else "CLOSE"
        print(f"[GRIPPER] {state.active_arm.upper()} gripper -> {g}")

    def reset_env() -> None:
        if recorder is not None and state.session.episode_recording_active:
            state.session.episode_recording_active = False
            recorder.discard_episode()
            print("[RECORD] Recording stopped — episode discarded before reset")
        state.session.request_reset()
        print("[RESET] Environment will reset on next step")

    def toggle_episode_recording() -> None:
        if recorder is None:
            return
        if not state.session.episode_recording_active:
            recorder.discard_episode()
            state.session.episode_recording_active = True
            print("[RECORD] Episode recording started — press N again to save and reset arms")
            return
        state.session.episode_recording_active = False
        recorder.save_episode()
        state.session.request_reset()
        print("[RECORD] Episode saved — resetting robot to initial pose")

    def discard_episode() -> None:
        if recorder is None:
            return
        state.session.episode_recording_active = False
        recorder.discard_episode()
        print("[RECORD] Episode buffer discarded — recording stopped")

    def start_teleop() -> None:
        state.session.start()
        print("[TELEOP] Activated")

    def stop_teleop() -> None:
        state.session.stop()
        print("[TELEOP] Deactivated")

    keyboard_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_teleop,
        "STOP": stop_teleop,
        "RESET": reset_env,
    }
    if recorder is not None:
        keyboard_callbacks["N"] = toggle_episode_recording
        keyboard_callbacks["M"] = discard_episode

    try:
        teleop_interface = build_se3_device(args, env_cfg, keyboard_callbacks)
    except Exception as exc:
        logger.error("Failed to create teleop device: %s", exc)
        env.close()
        return

    if isinstance(teleop_interface, Se3Gamepad):
        teleop_interface.dead_zone = args.gamepad_dead_zone

    arm_switch_key_desc = "(none)"
    gripper_key_desc = "(none)"
    reset_key_desc = "R key"
    try:
        if isinstance(teleop_interface, Se3Keyboard):
            teleop_interface.add_callback("TAB", toggle_arm)
            teleop_interface.add_callback("K", toggle_gripper)
            arm_switch_key_desc = "TAB"
            gripper_key_desc = "K"
            if recorder is not None:
                teleop_interface.add_callback("N", toggle_episode_recording)
                teleop_interface.add_callback("M", discard_episode)
        elif isinstance(teleop_interface, Se3Gamepad):
            arm_switch_key_desc = "Y button (polled)"
            gripper_key_desc = "A button (polled)"
            reset_key_desc = "B button (polled)"
        elif isinstance(teleop_interface, Se3SpaceMouse):
            teleop_interface.add_callback("L", toggle_gripper)
            gripper_key_desc = "Left button"
            arm_switch_key_desc = "(not available on SpaceMouse)"
            reset_key_desc = "Right button"
    except Exception as exc:
        logger.warning("Could not bind arm-switch/gripper controls: %s", exc)

    gp_input_iface = None
    gp_handle = None
    if isinstance(teleop_interface, Se3Gamepad):
        try:
            gp_input_iface = carb.input.acquire_input_interface()
            gp_handle = omni.appwindow.get_default_app_window().get_gamepad(0)
        except Exception as exc:
            logger.warning("Could not acquire gamepad handle for polling (%s)", exc)

    gp_prev = {_GP_SWITCH: False, _GP_GRIPPER: False, _GP_RESET: False, _GP_RECORD: False}

    print(f"\nUsing teleop device: {teleop_interface}")
    print("=" * 60)
    print("DUAL ARM SWITCH TELEOPERATION (IK-Abs, 16D)")
    print(f"  Task:        {args.task}")
    print(f"  Action dim:  {ACTION_DIM}  (left 7D + right 7D pose + 2 binary grippers)")
    print(f"  Active arm:  {state.active_arm.upper()}")
    print(f"  Arm switch:  {arm_switch_key_desc}")
    print(f"  Gripper:     {gripper_key_desc}  (toggles active arm open/close)")
    print(f"  Reset:       {reset_key_desc}")
    if recorder is not None:
        print("  Recording:   N=toggle episode (start/save+reset)  M=discard  R=reset")
        print(f"  Dataset:     {recorder.dataset_root}")
    print(f"  Pos scale:   {POS_SCALE}  (tune with --sensitivity)")
    print("=" * 60)

    env.reset()
    teleop_interface.reset()
    seed_targets_from_ee(state, robot, body_idx_map, dev)

    step_count = 0
    while simulation_app.is_running() and not shutdown_requested():
        try:
            with torch.inference_mode():
                if gp_input_iface is not None and gp_handle is not None:
                    gp_actions = [
                        (_GP_SWITCH, toggle_arm),
                        (_GP_GRIPPER, toggle_gripper),
                        (_GP_RESET, reset_env),
                    ]
                    if recorder is not None:
                        gp_actions.append((_GP_RECORD, toggle_episode_recording))
                    for btn, action_fn in gp_actions:
                        cur = gamepad_button_value(gp_input_iface, gp_handle, btn) > 0.5
                        if cur and not gp_prev[btn]:
                            action_fn()
                        gp_prev[btn] = cur

                device_output = teleop_interface.advance()
                component_threshold = (
                    args.gamepad_dead_zone if isinstance(teleop_interface, Se3Gamepad) else 1e-3
                )
                device_output = device_output.clone()
                device_output[device_output.abs() < component_threshold] = 0.0
                has_motion = device_output[:6].abs().max().item() > 0.0

                if state.session.teleoperation_active and has_motion:
                    _apply_device_delta(state, robot, body_idx_map, device_output, dev)

                action_1d = assemble_ik_abs_action(
                    state.target_pos[LEFT_ARM],
                    state.target_quat[LEFT_ARM],
                    state.target_pos[RIGHT_ARM],
                    state.target_quat[RIGHT_ARM],
                    state.grip[LEFT_ARM],
                    state.grip[RIGHT_ARM],
                    dev,
                )
                actions = broadcast_action(action_1d, env.num_envs)

                if step_count % 60 == 0:
                    l_grip = "OPEN" if state.grip[LEFT_ARM] > 0 else "CLOSE"
                    r_grip = "OPEN" if state.grip[RIGHT_ARM] > 0 else "CLOSE"
                    l_pos = [f"{v:+.3f}" for v in state.target_pos[LEFT_ARM].tolist()]
                    r_pos = [f"{v:+.3f}" for v in state.target_pos[RIGHT_ARM].tolist()]
                    mode = "ON" if state.session.teleoperation_active else "PAUSED"
                    rec = ""
                    if recorder is not None:
                        rec_state = "REC" if state.session.episode_recording_active else "idle"
                        rec = f" {rec_state} frames={recorder.frame_count}"
                    print(
                        f"[step={step_count} {mode} arm={state.active_arm.upper()} "
                        f"L_grip={l_grip} R_grip={r_grip}{rec}] "
                        f"L_pos={l_pos}  R_pos={r_pos}"
                    )

                env.step(actions)
                step_count += 1

                if recorder is not None and state.session.episode_recording_active:
                    recorder.on_step(env)

                if state.session.consume_reset():
                    env.reset()
                    teleop_interface.reset()
                    seed_targets_from_ee(state, robot, body_idx_map, dev)
                    print(f"[RESET] Done — controlling {state.active_arm.upper()} arm")

        except Exception as exc:
            logger.error("Simulation step error: %s", exc)
            break

    if shutdown_requested():
        print("[EXIT] Shutdown requested — leaving teleop loop.")
    env.close()
    print("Environment closed.")
