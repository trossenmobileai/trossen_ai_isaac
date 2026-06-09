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

The env action space is 14D absolute IK (left 7D + right 7D, [pos_xyz, quat_wxyz]
per arm). How the operator's hand poses become EE targets is controlled by
`--anchor_mode`:

  hand_anchored (default, recommended for room-scale VR)
    The first frame teleop is actively forwarding actions, the script snapshots
    the operator's hand pose AND the robot's current EE pose. Every subsequent
    frame, the EE target is composed via a "rigid bracket" coupling so the
    relative pose between the operator's hand and the EE stays constant:
        target_pos  = hand_curr_pos + (ee_init_pos - hand_init_pos)
        target_quat = hand_curr_quat * (hand_init_quat^-1 * ee_init_quat)
    Moving the hand 10 cm right moves the EE 10 cm right; rotating the hand by
    30 deg about an axis rotates the EE by the same 30 deg about the same axis
    in the world frame. Standing still leaves the EE still. The hand's absolute
    world position is irrelevant -- the operator can stand anywhere.

  absolute
    The hand pose (in the anchored OpenXR world frame) is fed directly as the
    IK target. The robot physically tries to BE where the hand is. Sensible
    only when the operator's body is meant to coincide with the robot (e.g.
    GR1T2-style humanoid avatars). Requires careful XrCfg.anchor_pos tuning.

Pipeline:
    OpenXRDevice.advance()
        -> torch.Tensor of shape [16] in the order declared in MobileAIReachEnvCfg_IK_ABS:
           [L_pose(7), R_pose(7), L_grip(1), R_grip(1)]
    -> slice [:14] = [L_pose, R_pose]              (raw hand action)
    -> compose with offsets (hand_anchored) OR pass through (absolute)
    -> broadcast to [num_envs, 14] and env.step()

The two gripper retargeter outputs are intentionally NOT fed to the env because the
env's ActionsCfg has no gripper_action term yet. They are logged for future use.

Activation model:
    The script auto-starts as soon as hand tracking is stable. A warm-up guard
    (configurable via --warmup_frames / --warmup_min_pos) waits for both hand
    positions to be clearly non-zero for N consecutive frames before forwarding
    actions to the env. This avoids the arms snapping to the OpenXR origin
    while the operator is still putting the headset on.

    In hand_anchored mode, the hand<->EE snapshot is taken at the first frame
    after warm-up. It is invalidated (and re-taken on the next active frame) on:
        * env.reset()
        * STOP -> START transitions (when a teleop_command publisher exists)
    so the operator can pause, reposition their body, and resume cleanly.

    The Isaac Lab START/STOP/RESET callbacks remain wired up. They are no-ops
    in an ALVR+SteamVR setup (no CloudXR sample client to publish them) but
    will work automatically if CloudXR or another publisher is added later.

XR anchor (viewpoint placement + frame alignment):
    XrCfg.anchor_prim_path / anchor_pos / anchor_rot together control where in
    the sim the operator's headset appears.

    By default the task config pins the XRAnchor under the robot's head-camera
    body (`/World/envs/env_0/Robot/cam_high_link`), so the operator gets a true
    first-person view from the robot's camera stand between the arms. The view
    follows the prim in world space, so if the robot's base moves, the operator
    moves with it.

    anchor_pos is then a USD child-transform offset relative to that prim.
    The default `(0, 0, -1.7)` cancels a 1.7 m tall operator's physical headset
    height so the eyes land *at* the camera height instead of well above it.
    Override for different operator heights:
        --anchor_pos 0 0 -1.6   # for 1.60 m
        --anchor_pos 0 0 -1.8   # for 1.80 m

    anchor_rot is the -90 deg yaw that aligns OpenXR's "forward" (operator's
    physical +Y) with the robot's "forward" (+X). It does NOT affect the
    hand_anchored IK math (that's frame-invariant once the snapshot is taken),
    but it does control how operator head rotation in the room maps to view
    rotation in the sim.

    To pick a different anchor prim (e.g. base_link for a natural standing
    view), use:
        --anchor_prim_path /World/envs/env_0/Robot/base_link

    Or run with --list_bodies once to print all robot bodies and their world
    positions; pick whichever body matches the desired viewpoint.

    To restore a static world anchor (no prim attachment), set --anchor_prim_path
    to the empty string is not currently supported; instead, edit the task
    config to pass `anchor_prim_path=None` to XrCfg.

Lock translation (--lock_translation):
    ALVR / SteamVR sometimes drop the stream and re-establish the OpenXR tracking
    origin on reconnect. That re-zero shifts the operator's headset world
    position by a few cm to a few m, so the FPV "jumps" each time.

    `--lock_translation` installs a per-frame hook that captures the operator's
    headset offset from the anchor prim on the first valid tracking frame and
    then continuously nudges `XrCfg.anchor_pos` so the headset world position
    stays at that same offset. Translational drift -- whether from reconnects
    or operator walking around -- is cancelled within ~1 frame. Head rotation
    is left untouched, so the operator can still look around normally.

    Recommended for unstable streaming sessions. When you ARE in a stable
    session and want to lean in / inspect, leave the flag off.

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
parser.add_argument(
    "--anchor_mode",
    type=str,
    default="hand_anchored",
    choices=["hand_anchored", "absolute"],
    help=(
        "How hand poses become EE targets. "
        "'hand_anchored' (default): snapshot the hand pose and EE pose the first frame "
        "teleop is active, then drive the EE only with the relative motion of the hand. "
        "Robot mirrors hand DELTA -- moving the hand 10 cm right moves the EE 10 cm right, "
        "regardless of the hand's absolute position. Recommended for room-scale VR with the "
        "operator standing alongside/behind the robot. "
        "'absolute': feed the hand world-pose directly as the IK target. The robot tries to "
        "physically be where your hand is. Only sensible if the operator's body is meant to "
        "coincide with the robot (e.g. humanoid avatars). XrCfg.anchor_pos must be tuned "
        "carefully in this mode."
    ),
)
parser.add_argument(
    "--anchor_prim_path",
    type=str,
    default=None,
    help=(
        "Override XrCfg.anchor_prim_path at launch time. USD prim path under which the "
        "XR anchor is parented; the operator's headset view then tracks that prim in "
        "world space (FPV). The task config defaults to the robot's head-camera body "
        "('/World/envs/env_0/Robot/cam_high_link'). Use --list_bodies to print the available "
        "robot body names if the default doesn't match this URDF."
    ),
)
parser.add_argument(
    "--list_bodies",
    action="store_true",
    help=(
        "Debug helper: print the list of robot body names (and their world poses for "
        "env 0) right after env construction. Useful for figuring out the correct "
        "--anchor_prim_path value. The script continues normally afterward."
    ),
)
parser.add_argument(
    "--lock_translation",
    action="store_true",
    help=(
        "Pin the headset's world position to a fixed offset from the anchor prim. "
        "Cancels translational drift from ALVR/SteamVR stream reconnects (where the "
        "OpenXR tracking origin gets re-zeroed and the FPV otherwise teleports). "
        "Head rotation is still tracked, so the operator can look around. "
        "Captures the operator's current head-to-anchor-prim offset the first valid "
        "tracking frame, then holds that offset. Leave off for stable sessions; "
        "turn on for laggy streams or when the POV keeps jumping on reconnect."
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
import contextlib
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.utils.math import quat_conjugate, quat_mul, subtract_frame_transforms
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

LEFT_EE_BODY_NAME = "follower_left_link_6"
RIGHT_EE_BODY_NAME = "follower_right_link_6"


def _get_ee_base_pose(robot, body_idx: int, env_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the EE pose of a single body in the robot base frame.

    Reads world-frame body pose + root pose from the articulation buffer and converts
    to the robot's base frame via :func:`subtract_frame_transforms`. Returns two 1D
    tensors: position (3,) and quaternion wxyz (4,), both on the articulation's device.
    """
    pos_w = robot.data.body_pos_w[env_idx, body_idx]  # (3,)
    quat_w = robot.data.body_quat_w[env_idx, body_idx]  # (4,) wxyz
    root_pos_w = robot.data.root_pos_w[env_idx]  # (3,)
    root_quat_w = robot.data.root_quat_w[env_idx]  # (4,) wxyz

    # subtract_frame_transforms expects batched (N, 3) / (N, 4) tensors; add a batch
    # axis for the call and strip it on the way out.
    pos_base, quat_base = subtract_frame_transforms(
        root_pos_w.unsqueeze(0),
        root_quat_w.unsqueeze(0),
        pos_w.unsqueeze(0),
        quat_w.unsqueeze(0),
    )
    return pos_base.squeeze(0), quat_base.squeeze(0)


def _install_translation_lock(
    xr_cfg: Any,
    xr_anchor_prim_path: str,
    *,
    invalid_pos_norm: float = 1e-3,
) -> Any:
    """Pin the headset's world translation; leave rotation tracked.

    Subscribes a callback to OpenXR's ``pre_sync_update`` event. The subscription
    is registered *after* the XR runtime is up (and after Isaac Lab's
    ``XrAnchorSynchronizer`` has been wired), so it runs second within the
    event and observes the freshly-set ``/XRAnchor`` pose.

    On the first valid tracking frame the callback snapshots
    ``eye_offset = headset_world - anchor_prim_world``. From then on, every
    frame it computes::

        target_h = anchor_prim_world + eye_offset
        delta    = target_h - headset_world
        xr_cfg.anchor_pos += delta     # synchronizer applies next frame
        /XRAnchor.world_pos += delta   # apply this frame too, no 1-frame lag

    The synchronizer adds ``anchor_pos`` directly in world frame (no rotation),
    so the ``anchor_prim_world = /XRAnchor_world - anchor_pos`` identity holds
    regardless of ``anchor_rot`` / ``anchor_rotation_mode``. Rotation is never
    touched here, so head tilting and looking around keep working normally.

    Args:
        xr_cfg: The ``XrCfg`` instance the OpenXRDevice was built with. Its
            ``anchor_pos`` field is mutated each frame.
        xr_anchor_prim_path: USD path of the ``/XRAnchor`` prim. Typically
            ``<anchor_prim_path>/XRAnchor`` when an anchor prim is configured,
            else ``/World/XRAnchor``.
        invalid_pos_norm: Headset positions with ``|pos| <= invalid_pos_norm``
            are treated as "not tracking yet" and ignored, so we don't latch
            the lock target to the (0, 0, 0) default the OpenXR device returns
            before tracking is live. Defaults to 1 mm.

    Returns:
        The carb subscription handle. The caller must keep a reference for the
        lifetime of the lock; dropping it unsubscribes.
    """
    # Imports are local so the file imports cleanly when omni.kit.xr.core
    # isn't present (tests, headless smoke checks, etc.).
    from isaacsim.core.prims import SingleXFormPrim
    from omni.kit.xr.core import XRCore, XRCoreEventType

    xr_core = XRCore.get_singleton()
    if xr_core is None:
        raise RuntimeError(
            "XRCore.get_singleton() returned None; --lock_translation requires the "
            "XR runtime to be initialised (i.e. AppLauncher launched with xr=True "
            "and the env has been constructed)."
        )

    anchor_prim = SingleXFormPrim(xr_anchor_prim_path)

    state: dict[str, Any] = {
        "captured": False,
        "eye_offset": None,
    }

    def _read_headset_world_pos() -> np.ndarray | None:
        head_device = xr_core.get_input_device("/user/head")
        if head_device is None:
            return None
        try:
            hmd = head_device.get_virtual_world_pose("")
        except Exception:
            return None
        if hmd is None:
            return None
        t = hmd.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float64)

    def _read_anchor_world_pose() -> tuple[np.ndarray, np.ndarray] | None:
        try:
            pos, quat = anchor_prim.get_world_pose()
        except Exception:
            return None
        return (
            np.asarray(pos, dtype=np.float64).reshape(3),
            np.asarray(quat, dtype=np.float64).reshape(4),
        )

    def _callback(_event: Any) -> None:
        h_curr = _read_headset_world_pos()
        if h_curr is None or float(np.linalg.norm(h_curr)) <= invalid_pos_norm:
            return
        anchor_pose = _read_anchor_world_pose()
        if anchor_pose is None:
            return
        a_anchor, a_quat = anchor_pose

        o_curr = np.asarray(xr_cfg.anchor_pos, dtype=np.float64)
        a_base = a_anchor - o_curr

        if not state["captured"]:
            state["eye_offset"] = h_curr - a_base
            state["captured"] = True
            print(
                "[LOCK] Translation pinned. eye_offset (world frame) = "
                f"[{state['eye_offset'][0]:+.3f}, {state['eye_offset'][1]:+.3f}, "
                f"{state['eye_offset'][2]:+.3f}] m"
            )
            return

        target_h = a_base + state["eye_offset"]
        delta = target_h - h_curr

        # No clamp on `delta`: a fresh reconnect can introduce a multi-metre jump,
        # and we want to cancel it in a single frame rather than drift back over
        # many frames.
        o_new = o_curr + delta
        xr_cfg.anchor_pos = (float(o_new[0]), float(o_new[1]), float(o_new[2]))

        # Apply directly to /XRAnchor so the current frame is corrected too.
        # Without this, there is a one-frame lag while we wait for the
        # synchronizer to read the new anchor_pos.
        with contextlib.suppress(Exception):
            anchor_prim.set_world_pose(position=a_anchor + delta, orientation=a_quat)

    return xr_core.get_message_bus().create_subscription_to_pop_by_type(
        XRCoreEventType.pre_sync_update,
        _callback,
        name="trossen_xr_lock_translation",
    )


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
    if args_cli.anchor_prim_path is not None:
        env_cfg.xr.anchor_prim_path = args_cli.anchor_prim_path

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

    # -- Resolve robot articulation and EE bodies for hand-anchored mode --
    # Looking these up once at startup avoids per-step name lookups and produces a
    # clear error if the env doesn't expose the expected bodies.
    robot = env.scene["robot"]

    # Optional debug dump: list every body name (and its current world pose for
    # env 0) so the operator can pick a sensible --anchor_prim_path. Print but
    # keep going so the rest of the session is unaffected.
    if args_cli.list_bodies:
        print("\n" + "=" * 60)
        print("Robot bodies (env 0): name -> world position [x, y, z] m")
        print("=" * 60)
        body_pos_w = robot.data.body_pos_w[0]  # (num_bodies, 3)
        for idx, name in enumerate(robot.body_names):
            pos = body_pos_w[idx]
            print(f"  [{idx:>2}] {name:<40} ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        print("=" * 60 + "\n")

    try:
        left_body_idx = robot.body_names.index(LEFT_EE_BODY_NAME)
        right_body_idx = robot.body_names.index(RIGHT_EE_BODY_NAME)
    except ValueError as exc:
        logger.error(
            f"Could not locate EE bodies on robot ({exc}). Expected '{LEFT_EE_BODY_NAME}' and "
            f"'{RIGHT_EE_BODY_NAME}' in robot.body_names. Available: {list(robot.body_names)}"
        )
        env.close()
        simulation_app.close()
        return

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

    # Hand-anchor state. In hand_anchored mode, the EE target is composed as
    #   target_pos  = hand_curr_pos + pos_offset
    #   target_quat = quat_offset * hand_curr_quat
    # where pos_offset and quat_offset are captured the first frame teleop is
    # actively forwarding actions, so the hand-at-snapshot maps exactly to the
    # EE-at-snapshot and the EE thereafter mirrors only the hand's *delta*.
    # In absolute mode, these stay None and the raw action is forwarded as-is.
    anchor_mode = args_cli.anchor_mode
    anchor_captured = False
    pos_offset_l: torch.Tensor | None = None
    quat_offset_l: torch.Tensor | None = None
    pos_offset_r: torch.Tensor | None = None
    quat_offset_r: torch.Tensor | None = None

    def _capture_anchor(action_14d: torch.Tensor) -> None:
        """Snapshot the current hand poses and EE poses to define delta anchors.

        Called the first frame teleop is actively forwarding actions. Captures both
        hand poses (from the retargeter output) and both EE poses (from the robot
        articulation, in the robot base frame), and stores the offsets that map
        hand-delta motion into EE-delta motion.
        """
        nonlocal anchor_captured, pos_offset_l, quat_offset_l, pos_offset_r, quat_offset_r

        hand_l_pos = action_14d[0:3]
        hand_l_quat = action_14d[3:7]
        hand_r_pos = action_14d[7:10]
        hand_r_quat = action_14d[10:14]

        ee_l_pos, ee_l_quat = _get_ee_base_pose(robot, left_body_idx)
        ee_r_pos, ee_r_quat = _get_ee_base_pose(robot, right_body_idx)

        # Articulation tensors might live on a slightly different device than the
        # action tensor (e.g. sim_device vs args_cli.device). Move EE tensors to
        # the action's device so all subsequent math stays on one device.
        ee_l_pos = ee_l_pos.to(hand_l_pos.device)
        ee_l_quat = ee_l_quat.to(hand_l_quat.device)
        ee_r_pos = ee_r_pos.to(hand_r_pos.device)
        ee_r_quat = ee_r_quat.to(hand_r_quat.device)

        pos_offset_l = ee_l_pos - hand_l_pos
        pos_offset_r = ee_r_pos - hand_r_pos
        # Rigid-bracket coupling: hand and EE move together as if linked by a
        # constant rigid offset. Equivalently, the hand-to-EE relative orientation
        # is fixed. Solving ee_curr = hand_curr * rot_offset gives:
        #   rot_offset = conj(hand_init) * ee_init
        # so the math composes as ee_curr = hand_curr * rot_offset (right-mul).
        quat_offset_l = quat_mul(quat_conjugate(hand_l_quat), ee_l_quat)
        quat_offset_r = quat_mul(quat_conjugate(hand_r_quat), ee_r_quat)

        anchor_captured = True
        print(
            "[ANCHOR] Captured. "
            f"L pos_offset=[{pos_offset_l[0]:+.3f}, {pos_offset_l[1]:+.3f}, {pos_offset_l[2]:+.3f}]  "
            f"R pos_offset=[{pos_offset_r[0]:+.3f}, {pos_offset_r[1]:+.3f}, {pos_offset_r[2]:+.3f}]  "
            "(EE will mirror hand delta from now on)"
        )

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("[RESET] Environment will reset on next step")

    def start_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("[TELEOP] Activated (left+right hands -> left+right arms)")

    def stop_teleop() -> None:
        nonlocal teleoperation_active, anchor_captured
        teleoperation_active = False
        # Invalidate the snapshot so the next START re-anchors the hand to wherever
        # the EE has come to rest. Without this, resuming would jolt the EE by the
        # delta accumulated while teleop was paused.
        anchor_captured = False
        print("[TELEOP] Deactivated (robot holds last pose; next START will re-anchor)")

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

    # -- Optional: lock headset translation (rotation still tracked) --
    # Held in `lock_sub` to keep the carb subscription alive for the session.
    lock_sub: Any = None
    if args_cli.lock_translation:
        xr_anchor_prim_path = getattr(teleop_interface, "_xr_anchor_headset_path", None)
        xr_cfg_obj = getattr(teleop_interface, "_xr_cfg", env_cfg.xr)
        if xr_anchor_prim_path is None:
            logger.warning(
                "--lock_translation requested but the OpenXRDevice does not expose "
                "'_xr_anchor_headset_path'. This Isaac Lab version may be older than "
                "expected; lock will be skipped."
            )
        else:
            try:
                lock_sub = _install_translation_lock(xr_cfg_obj, xr_anchor_prim_path)
            except Exception as exc:
                logger.warning(f"Failed to install --lock_translation hook: {exc}")

    print(f"\nUsing teleop device: {teleop_interface}")
    print("=" * 60)
    print("VR DUAL-ARM TELEOPERATION (hand tracking)")
    print(f"  Task:        {args_cli.task}")
    print(f"  Action dim:  {ACTION_DIM_ENV} (left 7D + right 7D, absolute IK)")
    if anchor_mode == "hand_anchored":
        print(
            "  Anchor mode: hand_anchored "
            "(EE mirrors hand DELTA from snapshot at first active frame)"
        )
    else:
        print(
            "  Anchor mode: absolute "
            "(EE target = hand world pose; XrCfg.anchor_pos governs alignment)"
        )
    print(
        f"  Warmup:      {warmup_frames_required} frames @ |pos| > {warmup_min_pos:.3f} m "
        "(arms stay idle until hand tracking is stable)"
    )
    # Echo the effective XR anchor so the operator can iterate on it from the log
    # without grepping the config file. CLI overrides win over the config defaults.
    anchor_pos = tuple(env_cfg.xr.anchor_pos)
    anchor_rot = tuple(env_cfg.xr.anchor_rot)
    anchor_prim_path = env_cfg.xr.anchor_prim_path
    if anchor_prim_path is None:
        anchor_prim_str = "(none -- static world anchor)"
    else:
        anchor_prim_str = anchor_prim_path
    print(
        f"  Anchor prim: {anchor_prim_str}"
        f"{'  (CLI override)' if args_cli.anchor_prim_path is not None else ''}"
    )
    print(
        f"  Anchor pos:  ({anchor_pos[0]:+.3f}, {anchor_pos[1]:+.3f}, {anchor_pos[2]:+.3f}) m"
        f"{'  (offset from anchor prim)' if anchor_prim_path is not None else ''}"
        f"{'  (CLI override)' if args_cli.anchor_pos is not None else ''}"
    )
    print(
        f"  Anchor rot:  ({anchor_rot[0]:+.4f}, {anchor_rot[1]:+.4f}, {anchor_rot[2]:+.4f}, {anchor_rot[3]:+.4f}) wxyz"
        f"{'  (CLI override)' if args_cli.anchor_rot is not None else ''}"
    )
    if args_cli.lock_translation:
        if lock_sub is not None:
            print(
                "  Lock trans:  ON  (headset world translation pinned; rotation still tracked)"
            )
        else:
            print(
                "  Lock trans:  REQUESTED but not installed (see warning above)"
            )
    else:
        print("  Lock trans:  off")
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
                    if anchor_mode == "hand_anchored":
                        # First active frame after warm-up / reset / STOP->START: snap
                        # the hand-EE relationship into place. Subsequent frames will
                        # use the captured offsets so the EE mirrors only hand deltas.
                        if not anchor_captured:
                            _capture_anchor(action_14d)

                        hand_l_pos = action_14d[0:3]
                        hand_l_quat = action_14d[3:7]
                        hand_r_pos = action_14d[7:10]
                        hand_r_quat = action_14d[10:14]

                        target_l_pos = hand_l_pos + pos_offset_l
                        # Right-multiply: see derivation in _capture_anchor. This gives
                        # the rigid-bracket feel -- rotating the hand in world frame
                        # rotates the EE by the same world-frame quaternion.
                        target_l_quat = quat_mul(hand_l_quat, quat_offset_l)
                        target_r_pos = hand_r_pos + pos_offset_r
                        target_r_quat = quat_mul(hand_r_quat, quat_offset_r)

                        target_action = torch.cat(
                            [target_l_pos, target_l_quat, target_r_pos, target_r_quat]
                        )
                    else:
                        # 'absolute' mode: pass the hand pose straight through as the
                        # IK target. Only sensible when the operator's body is meant
                        # to coincide with the robot.
                        target_action = action_14d

                    actions = target_action.unsqueeze(0).repeat(env.num_envs, 1)
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
                        if anchor_mode == "hand_anchored" and not anchor_captured:
                            state = "ANCHORING"
                        else:
                            state = f"ON/{anchor_mode}"
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
                    # Drop the snapshot too: the EE is back at the reset pose and the
                    # operator's hand may be anywhere. The next active frame will
                    # capture a fresh anchor.
                    anchor_captured = False
                    should_reset = False
                    print("[RESET] Done -- waiting for warm-up + re-anchor to re-engage teleop")

        except Exception as e:
            logger.error(f"Simulation step error: {e}")
            break

    # Drop the lock subscription before tearing down the env so its callback
    # stops firing while the stage is being torn down. Matches the pattern
    # OpenXRDevice.__del__ uses for its own carb subscriptions.
    if lock_sub is not None:
        lock_sub = None  # noqa: F841
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
