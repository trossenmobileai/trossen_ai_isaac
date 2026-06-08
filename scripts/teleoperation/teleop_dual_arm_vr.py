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

Activation model:
    The script auto-starts as soon as hand tracking is stable. A warm-up guard
    (configurable via --warmup_frames / --warmup_min_pos) waits for both hand
    positions to be clearly non-zero for N consecutive frames before forwarding
    actions to the env. This avoids the arms snapping to the OpenXR origin
    while the operator is still putting the headset on.

    The Isaac Lab START/STOP/RESET callbacks remain wired up. They are no-ops
    in an ALVR+SteamVR setup (no CloudXR sample client to publish them) but
    will work automatically if CloudXR or another publisher is added later.

Anchor tuning:
    The OpenXR world frame (operator's room) and the robot base frame do not
    automatically align — what feels like "forward" to the operator may not be
    "forward" to the robot. The XR anchor in MobileAIReachEnvCfg_IK_ABS carries
    sensible defaults for the Quest 3 + ALVR + SteamVR + Mobile AI stack:

        anchor_pos = (-0.3, 0.0, -0.6)
        anchor_rot = (0.7071, 0.0, 0.0, -0.7071)   # wxyz, -90 deg about Z

    If the arms still reach toward the wrong place, override at the command
    line without recompiling the config:

        --anchor_pos  -0.4 0.0 -0.55          # x y z, in meters, robot frame
        --anchor_rot   0.7071 0 0 +0.7071     # wxyz; flip the last component
                                              # to swap yaw direction

    Practical tuning recipe (one variable at a time):
      1. Stand still in a comfortable telepresence pose.
      2. Adjust anchor_rot's last component (+/-0.7071) until "reach forward"
         with your right hand pushes the right arm out *in front* of the
         robot, not behind or to the side.
      3. Adjust anchor_pos.z (more negative = arms map lower) until your
         hand-at-chest pose puts the EE at a natural rest height.
      4. Adjust anchor_pos.x (more negative = operator stands further
         behind the robot) until the arm workspace covers your reach.

Prerequisites on the workstation:
    * Isaac Lab installed and `./isaaclab.sh -p ...` available
    * ALVR running, Meta Quest 3 connected, SteamVR providing the OpenXR runtime
    * Run via `./isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py ...`
    * In Isaac Sim's AR panel: set Output Plugin = OpenXR, then click "Start AR"

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
parser.add_argument(
    "--warmup_frames",
    type=int,
    default=30,
    help=(
        "Number of consecutive frames with non-zero hand positions required before the "
        "script starts forwarding actions to env.step(). Prevents the arms from snapping "
        "to the OpenXR origin while the operator puts the headset on. ~30 frames = 0.5s "
        "at 60 Hz."
    ),
)
parser.add_argument(
    "--warmup_min_pos",
    type=float,
    default=0.02,
    help=(
        "Per-hand position-vector norm (meters) above which a frame is considered "
        "'live tracking' for warm-up. The OpenXR device defaults all poses to (0,0,0) "
        "before tracking starts, so any threshold > 0 distinguishes live from default."
    ),
)
parser.add_argument(
    "--anchor_pos",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help=(
        "Override XrCfg.anchor_pos at launch time. Three floats in meters expressing "
        "the OpenXR origin (operator's feet) in the robot base frame. Useful for "
        "sweeping operator standing positions without re-editing ik_abs_env_cfg.py. "
        "If omitted, the value baked into the task config is used."
    ),
)
parser.add_argument(
    "--anchor_rot",
    type=float,
    nargs=4,
    metavar=("W", "X", "Y", "Z"),
    default=None,
    help=(
        "Override XrCfg.anchor_rot at launch time. Four floats (w, x, y, z) "
        "for a unit quaternion rotating OpenXR vectors into the robot base frame. "
        "Identity is (1,0,0,0); a -90 deg yaw about Z is (0.7071, 0, 0, -0.7071). "
        "If omitted, the value baked into the task config is used."
    ),
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

    # Optional CLI overrides for the OpenXR anchor (frame alignment between the
    # operator's headset world and the robot base). Applied here so they take
    # effect before the env (and its OpenXRDevice) is built.
    if args_cli.anchor_pos is not None:
        env_cfg.xr.anchor_pos = tuple(args_cli.anchor_pos)
    if args_cli.anchor_rot is not None:
        env_cfg.xr.anchor_rot = tuple(args_cli.anchor_rot)

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
    # Auto-start: there is no CloudXR sample client in an ALVR+SteamVR setup to
    # publish the "START" carb event, so we begin active and gate stepping on
    # the warm-up guard below. Stop is still respected if a publisher exists.
    teleoperation_active = True
    # Warm-up state: only forward actions to env.step() after both hands have
    # reported non-zero positions for `warmup_frames_required` consecutive
    # frames. This prevents the arms from snapping toward (0,0,0) while the
    # operator is still putting the headset on or before tracking goes live.
    warmup_frames_required = max(1, int(args_cli.warmup_frames))
    warmup_min_pos = float(args_cli.warmup_min_pos)
    warmup_complete = False
    warmup_valid_count = 0

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
    print(
        f"  Warmup:      {warmup_frames_required} frames @ |pos| > {warmup_min_pos:.3f} m "
        "(arms stay idle until hand tracking is stable)"
    )
    # Echo the effective XR anchor so the operator can iterate on it from the log
    # without grepping the config file. CLI overrides win over the config defaults.
    anchor_pos = tuple(env_cfg.xr.anchor_pos)
    anchor_rot = tuple(env_cfg.xr.anchor_rot)
    print(
        f"  Anchor pos:  ({anchor_pos[0]:+.3f}, {anchor_pos[1]:+.3f}, {anchor_pos[2]:+.3f}) m"
        f"{'  (CLI override)' if args_cli.anchor_pos is not None else ''}"
    )
    print(
        f"  Anchor rot:  ({anchor_rot[0]:+.4f}, {anchor_rot[1]:+.4f}, {anchor_rot[2]:+.4f}, {anchor_rot[3]:+.4f}) wxyz"
        f"{'  (CLI override)' if args_cli.anchor_rot is not None else ''}"
    )
    print("  STOP/RESET:  available if a teleop_command publisher is present (e.g. CloudXR)")
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

                # Warm-up gate: only forward actions once both hands have been
                # reporting non-zero positions for N consecutive frames. Until
                # then we render only, so the operator can put the headset on
                # without the arms jumping toward the OpenXR origin.
                l_pos_norm = float(action_14d[:3].norm().item())
                r_pos_norm = float(action_14d[7:10].norm().item())
                if not warmup_complete:
                    if l_pos_norm > warmup_min_pos and r_pos_norm > warmup_min_pos:
                        warmup_valid_count += 1
                        if warmup_valid_count >= warmup_frames_required:
                            warmup_complete = True
                            print(
                                f"[WARMUP] Hand tracking stable after "
                                f"{warmup_valid_count} frames -- arms now driven by VR."
                            )
                    else:
                        warmup_valid_count = 0

                step_actions = teleoperation_active and warmup_complete
                if step_actions:
                    actions = action_14d.unsqueeze(0).repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    # Render only — robot holds its last IK target. Without this,
                    # the viewer freezes while we're waiting on warm-up or paused.
                    env.sim.render()

                if step_count % 60 == 0:
                    l_pos = action_14d[:3].tolist()
                    r_pos = action_14d[7:10].tolist()
                    if not warmup_complete:
                        state = f"WARMUP {warmup_valid_count}/{warmup_frames_required}"
                    elif teleoperation_active:
                        state = "ON"
                    else:
                        state = "PAUSED"
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
                    # Re-warm-up after a reset: device pose cache was just cleared
                    # so we should re-verify tracking before re-engaging the arms.
                    warmup_complete = False
                    warmup_valid_count = 0
                    should_reset = False
                    print("[RESET] Done -- waiting for warm-up to re-engage teleop")

        except Exception as e:
            logger.error(f"Simulation step error: {e}")
            break

    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
