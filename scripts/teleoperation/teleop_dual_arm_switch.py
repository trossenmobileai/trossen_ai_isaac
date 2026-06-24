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

"""Dual-arm switchable teleoperation for the Mobile AI robot (IK-Abs, 16D).

Controls the left and right arms one at a time using a single Se3 input device.
The active arm's IK target is updated each step by offsetting the current real
EE pose by the device's 6D delta — so releasing a key/stick always brings the
arm to an immediate stop with no windup drift.  The inactive arm holds the fixed
target captured at the last switch or reset.

Toggle between arms with:
    Keyboard:    TAB key
    Gamepad:     Y button
    SpaceMouse:  (arm switching not available; only one free button)

Toggle the active arm's gripper (open/close) with:
    Keyboard:    K key
    Gamepad:     A button
    SpaceMouse:  Left button  (right button = reset)

Reset environment:
    Keyboard:    R key
    Gamepad:     B button
    SpaceMouse:  Right button

Action space is 16D absolute IK:
    [L_pos(3), L_quat(4), R_pos(3), R_quat(4), L_grip(1), R_grip(1)]

The gripper scalars follow the BinaryJointPositionAction convention:
    +1.0  -> open     -1.0  -> close

Gripper state is per-arm and independent: closing one arm does not affect the other.

Gamepad button handling uses rising-edge polling (via carb.input.get_gamepad_value)
instead of add_callback, because the Isaac Lab gamepad dispatches callbacks on both
press and release, which would double-fire toggle actions and make them appear broken.

Launch Isaac Sim Simulator first.
"""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Dual-arm switchable teleoperation for Mobile AI robot (IK-Abs, 16D)."
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
    default="Isaac-Reach-MobileAI-IK-Abs-Play-v0",
    help="Name of the task. Must be an absolute-IK variant (16D action space).",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity scale factor.")
parser.add_argument(
    "--gamepad_dead_zone",
    type=float,
    default=0.15,
    help=(
        "Dead zone applied to each gamepad stick axis individually. "
        "Any per-axis value below this threshold is treated as zero. "
        "Default 0.15 covers typical hardware stick drift (~0.03-0.05) "
        "with ample margin. Raise if drift persists; lower only if slow "
        "intentional pushes are being clipped."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# --- Imports after sim launch ---
import logging

import carb
import gymnasium as gym
import omni.appwindow
import torch

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.utils.math import normalize, quat_from_angle_axis, quat_mul, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

# Action layout (16D):
#   indices  0..2   -> left  arm position    [x, y, z]  (base frame)
#   indices  3..6   -> left  arm quaternion  [w, x, y, z]  (base frame)
#   indices  7..9   -> right arm position    [x, y, z]  (base frame)
#   indices 10..13  -> right arm quaternion  [w, x, y, z]  (base frame)
#   index   14      -> left  gripper scalar  (+1 open / -1 close)
#   index   15      -> right gripper scalar  (+1 open / -1 close)
ACTION_DIM_ENV = 16

LEFT_ARM = "left"
RIGHT_ARM = "right"
LEFT_EE_BODY_NAME = "follower_left_link_6"
RIGHT_EE_BODY_NAME = "follower_right_link_6"

# Integration scales applied to the device's 6D delta output each step.
# The active arm's target is anchored to the current real EE pose each frame
# (not accumulated), so releasing a key/stick stops the arm immediately.
# Tune via --sensitivity if needed.
POS_SCALE = 0.05
ROT_SCALE = 0.05

# Gamepad buttons polled with rising-edge detection in the main loop.
# Using add_callback for these would double-fire (press + release), causing
# toggles to cancel themselves and making the controls appear unresponsive.
_GP_SWITCH  = carb.input.GamepadInput.Y   # arm switch
_GP_GRIPPER = carb.input.GamepadInput.A   # active-arm gripper toggle
_GP_RESET   = carb.input.GamepadInput.B   # environment reset


def _gamepad_button_value(input_iface, gamepad, btn: carb.input.GamepadInput) -> float:
    """Return the current analog value [0..1] of a gamepad button."""
    try:
        return float(input_iface.get_gamepad_value(gamepad, btn))
    except Exception:
        return 0.0


def _get_ee_base_pose(robot, body_idx: int, env_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the EE pose of a single body in the robot base frame.

    Returns two 1D tensors: position (3,) and quaternion wxyz (4,), both on
    the articulation's device.
    """
    pos_w = robot.data.body_pos_w[env_idx, body_idx]
    quat_w = robot.data.body_quat_w[env_idx, body_idx]
    root_pos_w = robot.data.root_pos_w[env_idx]
    root_quat_w = robot.data.root_quat_w[env_idx]

    pos_b, quat_b = subtract_frame_transforms(
        root_pos_w.unsqueeze(0),
        root_quat_w.unsqueeze(0),
        pos_w.unsqueeze(0),
        quat_w.unsqueeze(0),
    )
    return pos_b.squeeze(0), quat_b.squeeze(0)


def main() -> None:
    """Run dual-arm switchable IK-Abs teleoperation."""
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

    # Sanity-check action dimension. Warn loudly if an IK-Rel task was selected by mistake.
    try:
        action_dim = env.action_manager.total_action_dim
    except AttributeError:
        action_dim = None
    if action_dim is not None and action_dim != ACTION_DIM_ENV:
        logger.warning(
            f"Env action_dim={action_dim} but this script assembles {ACTION_DIM_ENV}D actions. "
            "You probably selected an IK-Rel task; expect a shape mismatch on env.step(). "
            "Use one of the Isaac-Reach-MobileAI-IK-Abs-* tasks."
        )

    # -- Resolve robot articulation and EE body indices once at startup --
    robot = env.scene["robot"]
    try:
        left_body_idx = robot.body_names.index(LEFT_EE_BODY_NAME)
        right_body_idx = robot.body_names.index(RIGHT_EE_BODY_NAME)
    except ValueError as exc:
        logger.error(
            f"Could not locate EE bodies on robot ({exc}). "
            f"Expected '{LEFT_EE_BODY_NAME}' and '{RIGHT_EE_BODY_NAME}' in robot.body_names. "
            f"Available: {list(robot.body_names)}"
        )
        env.close()
        simulation_app.close()
        return

    body_idx_map = {LEFT_ARM: left_body_idx, RIGHT_ARM: right_body_idx}

    # -- State --
    active_arm = LEFT_ARM
    should_reset = False
    teleoperation_active = True

    dev = args_cli.device

    # Absolute EE targets in the robot base frame.
    # The inactive arm's target is a fixed pose held from the last switch/reset.
    # The active arm's target is re-anchored to the current EE pose each frame
    # before the delta is applied, so releasing a key/stick stops the arm immediately.
    target_pos = {
        LEFT_ARM:  torch.zeros(3, dtype=torch.float32, device=dev),
        RIGHT_ARM: torch.zeros(3, dtype=torch.float32, device=dev),
    }
    target_quat = {
        LEFT_ARM:  torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev),
        RIGHT_ARM: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev),
    }
    # Per-arm gripper state: +1.0 open, -1.0 close.
    grip = {LEFT_ARM: 1.0, RIGHT_ARM: 1.0}

    def _seed_targets_from_ee() -> None:
        """Seed both arm targets from the current robot articulation state."""
        for arm, idx in body_idx_map.items():
            pos, quat = _get_ee_base_pose(robot, idx)
            target_pos[arm] = pos.to(dev)
            target_quat[arm] = quat.to(dev)
            grip[arm] = 1.0  # default open after reset

    # -- Callbacks (keyboard / SpaceMouse only) --
    # Gamepad toggles use rising-edge polling in the main loop instead of
    # add_callback, because the gamepad dispatches callbacks on press AND release,
    # which would double-fire toggles and make them appear unresponsive.
    def toggle_arm() -> None:
        nonlocal active_arm
        # Snapshot the inactive arm's current actual pose as its held target
        # before switching so it truly stays put after the switch.
        inactive = RIGHT_ARM if active_arm == LEFT_ARM else LEFT_ARM
        pos, quat = _get_ee_base_pose(robot, body_idx_map[inactive])
        target_pos[inactive] = pos.to(dev)
        target_quat[inactive] = quat.to(dev)
        active_arm = inactive
        g_state = "OPEN" if grip[active_arm] > 0 else "CLOSE"
        print(f"[ARM SWITCH] Now controlling: {active_arm.upper()} arm  gripper={g_state}")

    def toggle_gripper() -> None:
        grip[active_arm] = -1.0 if grip[active_arm] > 0 else 1.0
        state = "OPEN" if grip[active_arm] > 0 else "CLOSE"
        print(f"[GRIPPER] {active_arm.upper()} gripper -> {state}")

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

    # String-keyed callbacks work for keyboard and SpaceMouse.
    # They are deliberately NOT passed to the gamepad factory path because the
    # gamepad fires add_callback handlers on both press and release, making them
    # double-fire. Gamepad controls are handled via rising-edge polling below.
    keyboard_callbacks: dict[str, Callable[[], None]] = {
        "R":     reset_env,
        "START": start_teleop,
        "STOP":  stop_teleop,
        "RESET": reset_env,
    }

    # -- Build teleop device --
    teleop_interface = None
    sensitivity = args_cli.sensitivity
    is_gamepad = args_cli.teleop_device.lower() == "gamepad"

    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            # Pass string callbacks only for non-gamepad devices.
            cbs = {} if is_gamepad else keyboard_callbacks
            teleop_interface = create_teleop_device(
                args_cli.teleop_device,
                env_cfg.teleop_devices.devices,
                cbs,
            )
        else:
            logger.warning(f"Device '{args_cli.teleop_device}' not in env config — creating default.")
            if args_cli.teleop_device.lower() == "keyboard":
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
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not bind '{key}': {e}")

            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(
                        gripper_term=False,
                        pos_sensitivity=0.1 * sensitivity,
                        rot_sensitivity=0.1 * sensitivity,
                    )
                )
                # No add_callback calls here — all gamepad buttons use polling.

            elif args_cli.teleop_device.lower() == "spacemouse":
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

    # -- Apply gamepad dead zone --
    # The default carb dead_zone (0.01) is too small to catch hardware stick
    # drift, which can reach ~0.035 on a typical controller. Override it on the
    # device instance now so the event handler filters drift at the source,
    # per axis, before values accumulate in _delta_pose_raw.
    if isinstance(teleop_interface, Se3Gamepad):
        teleop_interface.dead_zone = args_cli.gamepad_dead_zone

    # -- Bind keyboard / SpaceMouse arm-switch and gripper toggle --
    arm_switch_key_desc = "(none)"
    gripper_key_desc = "(none)"
    reset_key_desc = "R key"
    try:
        if isinstance(teleop_interface, Se3Keyboard):
            teleop_interface.add_callback("TAB", toggle_arm)
            teleop_interface.add_callback("K", toggle_gripper)
            arm_switch_key_desc = "TAB"
            gripper_key_desc = "K"
        elif isinstance(teleop_interface, Se3Gamepad):
            # Gamepad: all toggles handled via rising-edge polling in the main loop.
            arm_switch_key_desc = "Y button (polled)"
            gripper_key_desc = "A button (polled)"
            reset_key_desc = "B button (polled)"
        elif isinstance(teleop_interface, Se3SpaceMouse):
            # SpaceMouse: left button → gripper, right button → reset (via "R" callback).
            teleop_interface.add_callback("L", toggle_gripper)
            gripper_key_desc = "Left button"
            arm_switch_key_desc = "(not available on SpaceMouse)"
            reset_key_desc = "Right button"
    except Exception as e:
        logger.warning(f"Could not bind arm-switch/gripper controls: {e}")

    # -- Gamepad carb polling setup --
    # Acquire raw carb input + gamepad handle once so the main loop can poll
    # button values directly. Only used when the device is a gamepad.
    gp_input_iface = None
    gp_handle = None
    if isinstance(teleop_interface, Se3Gamepad):
        try:
            gp_input_iface = carb.input.acquire_input_interface()
            gp_handle = omni.appwindow.get_default_app_window().get_gamepad(0)
        except Exception as e:
            logger.warning(f"Could not acquire gamepad handle for polling ({e}); button controls unavailable.")

    # Previous-frame button states for rising-edge detection (press, not hold).
    gp_prev = {_GP_SWITCH: False, _GP_GRIPPER: False, _GP_RESET: False}

    print(f"\nUsing teleop device: {teleop_interface}")
    print("=" * 60)
    print("DUAL ARM SWITCH TELEOPERATION (IK-Abs, 16D)")
    print(f"  Task:        {args_cli.task}")
    print(f"  Action dim:  {ACTION_DIM_ENV}  (left 7D + right 7D pose + 2 binary grippers)")
    print(f"  Active arm:  {active_arm.upper()}")
    print(f"  Arm switch:  {arm_switch_key_desc}")
    print(f"  Gripper:     {gripper_key_desc}  (toggles active arm open/close)")
    print(f"  Reset:       {reset_key_desc}")
    print(f"  Pos scale:   {POS_SCALE}  (tune with --sensitivity)")
    print("=" * 60)

    # -- Reset and seed initial targets --
    env.reset()
    teleop_interface.reset()
    _seed_targets_from_ee()

    step_count = 0

    # -- Main loop --
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # --- Gamepad rising-edge button polling ---
                # Must happen outside inference_mode because it touches Python state,
                # but we do it at the top of the try block for clean sequencing.
                if gp_input_iface is not None and gp_handle is not None:
                    for btn, action_fn in (
                        (_GP_SWITCH,  toggle_arm),
                        (_GP_GRIPPER, toggle_gripper),
                        (_GP_RESET,   reset_env),
                    ):
                        cur = _gamepad_button_value(gp_input_iface, gp_handle, btn) > 0.5
                        if cur and not gp_prev[btn]:
                            action_fn()
                        gp_prev[btn] = cur

                # 6D delta from device: [dx, dy, dz, rot_vec_x, rot_vec_y, rot_vec_z].
                # (gripper_term=False so advance() always returns exactly 6D)
                device_output = teleop_interface.advance()

                # Per-component dead zone: zero out any axis whose absolute value
                # is below the gamepad threshold (or a small floor for keyboard/
                # spacemouse). This catches any drift that slips through the
                # device-level filter and ensures has_motion is only True when
                # at least one axis carries a real intentional command.
                component_threshold = (
                    args_cli.gamepad_dead_zone
                    if isinstance(teleop_interface, Se3Gamepad)
                    else 1e-3
                )
                device_output = device_output.clone()
                device_output[device_output.abs() < component_threshold] = 0.0
                has_motion = device_output[:6].abs().max().item() > 0.0

                if teleoperation_active and has_motion:
                    delta_pos = device_output[:3].to(dev)
                    rot_vec = device_output[3:6].to(dev)  # axis * angle (radians)

                    # Anchor to the current real EE pose before adding the delta.
                    # This ensures that when delta → 0 (finger lifted), the target
                    # equals the actual arm position and motion stops immediately.
                    cur_pos, cur_quat = _get_ee_base_pose(robot, body_idx_map[active_arm])
                    target_pos[active_arm] = cur_pos.to(dev) + POS_SCALE * delta_pos

                    # Orientation: compose the small rotation delta onto the current
                    # real orientation (same anchoring principle).
                    angle = rot_vec.norm()
                    if angle > 1e-6:
                        axis = rot_vec / angle
                        dq = quat_from_angle_axis(ROT_SCALE * angle, axis)
                        target_quat[active_arm] = normalize(quat_mul(dq, cur_quat.to(dev)))
                    else:
                        target_quat[active_arm] = cur_quat.to(dev)

                # Assemble 16D absolute action regardless of teleoperation_active so the
                # arms hold their last targets when paused instead of going limp.
                actions = torch.cat([
                    target_pos[LEFT_ARM],
                    target_quat[LEFT_ARM],
                    target_pos[RIGHT_ARM],
                    target_quat[RIGHT_ARM],
                    torch.tensor(
                        [grip[LEFT_ARM], grip[RIGHT_ARM]],
                        dtype=torch.float32,
                        device=dev,
                    ),
                ])
                actions = actions.unsqueeze(0).repeat(env.num_envs, 1)

                if step_count % 60 == 0:
                    l_grip = "OPEN" if grip[LEFT_ARM] > 0 else "CLOSE"
                    r_grip = "OPEN" if grip[RIGHT_ARM] > 0 else "CLOSE"
                    l_pos = [f"{v:+.3f}" for v in target_pos[LEFT_ARM].tolist()]
                    r_pos = [f"{v:+.3f}" for v in target_pos[RIGHT_ARM].tolist()]
                    state = "ON" if teleoperation_active else "PAUSED"
                    print(
                        f"[step={step_count} {state} arm={active_arm.upper()} "
                        f"L_grip={l_grip} R_grip={r_grip}] "
                        f"L_pos={l_pos}  R_pos={r_pos}"
                    )

                env.step(actions)
                step_count += 1

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    _seed_targets_from_ee()
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
