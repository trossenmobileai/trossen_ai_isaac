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

The env action space is 16D: 14D absolute IK (left 7D + right 7D, [pos_xyz,
quat_wxyz] per arm) plus 2 binary gripper scalars (left, right). How the
operator's hand poses become EE targets is controlled by `--anchor_mode`:

  hand_anchored (default, recommended for room-scale VR)
    The first frame teleop is actively forwarding actions, the script snapshots
    the operator's hand pose, the robot's current EE pose, AND the operator's
    head yaw. Every subsequent frame, the EE target is composed via a "rigid
    bracket" coupling expressed in a single control frame C = R_ctrl · R_yaw_inv:
        C           = R_ctrl * R_yaw_inv
        delta       = C * (hand_curr_pos - hand_init_pos)
        target_pos  = ee_init_pos + delta
        target_quat = C * hand_curr_quat * (hand_init_quat^-1 * C^-1 * ee_init_quat)
    where R_yaw is the head yaw (about Z) captured at snapshot time and R_ctrl is
    a fixed yaw about the robot's +Z set by --control_yaw_deg (default -90). The
    OpenXR hand frame sits ~90 deg from the robot base, which made "reach forward"
    drive the arm sideways (forward->left, right->forward) identically in every
    view; R_ctrl realigns it so "reach forward" drives the robot forward (+X).
    Crucially, BOTH the position delta and the orientation delta are conjugated by
    the same frame C, so a hand tilt/roll produces the matching EE tilt/roll
    rather than a scrambled-axis rotation. Moving the hand 10 cm in a
    head-relative direction moves the EE 10 cm in the matching robot-frame
    direction; standing still leaves the EE still. The hand's absolute world
    position is irrelevant. Set --control_yaw_deg 0 to disable the correction.

    The head yaw is snapshotted at anchor time, not tracked live, so turning the
    head WHILE teleoperating does not rotate the command frame (head wobble can't
    corrupt commands). If the operator physically re-orients their body, they
    should re-anchor (B key, reset, or stop->start) to re-capture the facing.

  absolute
    The hand pose (in the anchored OpenXR world frame) is fed directly as the
    IK target. The robot physically tries to BE where the hand is. Sensible
    only when the operator's body is meant to coincide with the robot (e.g.
    GR1T2-style humanoid avatars). Requires careful XrCfg.anchor_pos tuning.

Pipeline:
    OpenXRDevice.advance()
        -> torch.Tensor of shape [16] in the order declared in MobileAIReachEnvCfg_IK_ABS:
           [L_pose(7), R_pose(7), L_grip(1), R_grip(1)]
    -> split: pose_part = [:14], grip_part = [14:16]
    -> compose pose target with offsets (hand_anchored) OR pass through (absolute)
    -> concat [pose_target(14), grip_part(2)] = 16D
    -> broadcast to [num_envs, 16] and env.step()

The gripper scalars are the GripperRetargeter outputs (+1 open / -1 close, from the
thumb-index pinch). The env wires them to BinaryJointPositionAction terms on each
arm's carriage joint (see ik_abs_env_cfg.py).

Activation model:
    Staged start (default): the script begins INACTIVE. A warm-up guard
    (configurable via --warmup_frames / --warmup_min_pos) waits for both hand
    positions to be clearly non-zero for several consecutive frames, after which
    a second user at the workstation presses N to engage. The hand<->EE anchor
    is captured at that moment, so the arms never jump on connect. Pass
    --autostart to engage automatically once warm-up completes (old behavior).

    Workstation keyboard controls (a plain Se3Keyboard sidecar):
        N -> start/engage      M -> pause (hold pose; re-anchors on resume)
        B -> re-anchor only    J -> reset environment
    (SPACE/P/R/ENTER are deliberately avoided -- they collide with Kit shortcuts;
    SPACE in particular is the timeline play/pause and would freeze the sim.)

    In hand_anchored mode, the hand<->EE snapshot is taken on the first active
    frame and re-taken on env.reset(), M->N (pause/resume), or B (re-anchor),
    so the operator can pause, reposition or re-orient their body, and resume.

    The Isaac Lab START/STOP/RESET callbacks remain wired up too. They are no-ops
    in an ALVR+SteamVR setup (no CloudXR sample client to publish them) but
    will work automatically if CloudXR or another publisher is added later.

XR anchor (viewpoint placement + frame alignment):
    The --view preset chooses where in the sim the operator's headset appears by
    setting XrCfg.anchor_prim_path / anchor_pos / anchor_rot:
        first_person  -> inside the head camera (cam_high_link); robot's-eye view
        over_shoulder -> behind/above the arms looking forward (base_link)
        third_person  -> off the front-side looking back at the arms (base_link)
    Default is third_person (the head-camera FPV can feel cramped on this robot).
    The preset values are starting points; fine-tune at runtime with the
    --anchor_pos / --anchor_rot / --anchor_prim_path overrides (which take
    precedence over the preset). The view is decoupled from control: the
    hand_anchored mapping is frame-invariant once anchored.

    The anchor follows its prim in world space, so if the robot's base moves, the
    operator's viewpoint moves with it.

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

Prerequisites on the workstation:
    * Isaac Lab installed and `~/IsaacLab/isaaclab.sh -p ...` available
    * ALVR running, Meta Quest 3 connected, SteamVR providing the OpenXR runtime
    * Run via `~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py ...`
    * In Isaac Sim's AR panel: set Output Plugin = OpenXR, then click "Start AR"

Launch Isaac Sim Simulator first.
"""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="hand-tracking teleoperation for the Mobile AI bimanual robot."
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
    "--pinch_hold_dist",
    type=float,
    default=0.08,
    help=(
        "Thumb–index distance (meters) below which the arm ORIENTATION is frozen "
        "while position keeps tracking and only the gripper command changes. "
        "When fingers close, the Quest tracker re-estimates the wrist, causing a "
        "TCP orientation snap; holding orientation during the pinch suppresses that. "
        "On finger release the orientation coupling is re-snapshotted so the wrist "
        "jump on un-occlusion does not move the EE. Default 0.08 m. Set 0.0 to disable."
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
    "--view",
    type=str,
    default="third_person",
    choices=["first_person", "third_person", "over_shoulder"],
    help=(
        "Viewpoint preset. Sets the XR anchor prim + offset + rotation: "
        "'first_person' = inside the robot's head camera (cam_high_link); "
        "'over_shoulder' = behind/above the arms looking forward (base_link); "
        "'third_person' (default) = off to the front-side looking back at the arms "
        "(base_link). Explicit --anchor_prim_path/--anchor_pos/--anchor_rot override "
        "the corresponding field of the preset. The preset only affects the view; the "
        "hand_anchored control mapping is frame-invariant once anchored."
    ),
)
parser.add_argument(
    "--autostart",
    action="store_true",
    help=(
        "Begin teleoperation automatically once hand tracking warms up, instead of "
        "waiting for a workstation key press. Default (flag absent) is the staged "
        "workflow: the operator positions their hands, then a second user at the "
        "workstation presses N to engage, so the arms never jump on connect."
    ),
)
parser.add_argument(
    "--no_hand_markers",
    action="store_true",
    help=(
        "Disable the debug markers (RGB frames at the IK EE goals and small spheres "
        "at the tracked wrist/thumb/index keypoints). Markers are drawn by default to "
        "aid debugging; pass this flag to turn them off for performance or clutter."
    ),
)
parser.add_argument(
    "--control_yaw_deg",
    type=float,
    default=-90.0,
    help=(
        "Yaw (deg, about the robot's +Z) applied to the HAND-MOTION DELTA before it "
        "drives the EE, in hand_anchored mode. This rotates the control plane so "
        "'reach forward' maps to the robot's forward (+X) instead of sideways. The "
        "OpenXR hand frame is rotated ~90 deg from the robot base, which made forward "
        "hand motion move the arm left and rightward motion move it forward; the "
        "default -90 cancels that. Only the horizontal mapping is affected (up/down is "
        "untouched). Set 0 to disable the correction, or flip to +90/180 if your setup "
        "maps differently. Tune live without code edits."
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
import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

# Action layout produced by the four retargeters declared in
# MobileAIReachEnvCfg_IK_ABS.teleop_devices.devices["handtracking"]:
#   indices 0..6   -> left  arm pose [pos_xyz, quat_wxyz]
#   indices 7..13  -> right arm pose [pos_xyz, quat_wxyz]
#   index   14     -> left  gripper scalar
#   index   15     -> right gripper scalar
ACTION_DIM_PER_ARM = 7
POSE_DIM = 2 * ACTION_DIM_PER_ARM  # 14D of arm poses
ACTION_DIM_ENV = POSE_DIM + 2  # 16D goes to env.step (poses + 2 grippers)
LEFT_GRIP_IDX = 14
RIGHT_GRIP_IDX = 15

LEFT_EE_BODY_NAME = "follower_left_link_6"
RIGHT_EE_BODY_NAME = "follower_right_link_6"

# Viewpoint presets selected by --view. Each maps to an XR anchor prim, a child
# transform offset (meters, relative to that prim) and a rotation (wxyz). These
# are starting points and are expected to be fine-tuned at runtime via the
# --anchor_pos / --anchor_rot / --anchor_prim_path overrides. The -90 deg yaw
# (0.7071, 0, 0, -0.7071) aligns OpenXR "forward" (+Y) with the robot "forward"
# (+X); third_person uses the opposite yaw so the operator looks back at the arms.
_ENV0_ROBOT = "/World/envs/env_0/Robot"
VIEW_PRESETS: dict[str, dict] = {
    "first_person": {
        "anchor_prim_path": f"{_ENV0_ROBOT}/cam_high_link",
        "anchor_pos": (0.0, 0.0, -1.7),
        "anchor_rot": (0.7071068, 0.0, 0.0, -0.7071068),
    },
    "over_shoulder": {
        "anchor_prim_path": f"{_ENV0_ROBOT}/base_link",
        "anchor_pos": (-0.7, 0.0, 0.5),
        "anchor_rot": (0.7071068, 0.0, 0.0, -0.7071068),
    },
    "third_person": {
        "anchor_prim_path": f"{_ENV0_ROBOT}/base_link",
        "anchor_pos": (0.6, -1.0, 0.6),
        "anchor_rot": (0.7071068, 0.0, 0.0, 0.7071068),
    },
}

# Keypoint joints visualized per hand when markers are enabled.
HAND_KEYPOINT_JOINTS = ("wrist", "thumb_tip", "index_tip")


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


def _get_head_quat(device: torch.device | str) -> torch.Tensor | None:
    """Read the headset orientation in the XR anchored/world frame as wxyz.

    The hand poses produced by ``Se3AbsRetargeter`` are OpenXR *virtual world*
    poses (the wrist's pose in the XR anchored frame). The head pose read here via
    ``get_virtual_world_pose`` lives in that same frame, so a yaw extracted from it
    is directly comparable to the hand-position deltas — which is exactly what the
    head-relative remapping needs.

    Returns a (4,) tensor [w, x, y, z] on ``device``, or ``None`` if the XR runtime,
    head device, or its pose is unavailable this frame (caller falls back to the
    previous world-locked behavior).
    """
    try:
        from omni.kit.xr.core import XRCore
    except ModuleNotFoundError:
        return None

    xr_core = XRCore.get_singleton() if XRCore is not None else None
    if xr_core is None:
        return None
    head_device = xr_core.get_input_device("/user/head")
    if head_device is None:
        return None
    try:
        hmd = head_device.get_virtual_world_pose("")
    except Exception:
        return None
    if hmd is None:
        return None

    quat = hmd.ExtractRotationQuat()
    w = float(quat.GetReal())
    imag = quat.GetImaginary()
    return torch.tensor(
        [w, float(imag[0]), float(imag[1]), float(imag[2])],
        dtype=torch.float32,
        device=device,
    )


def _get_hand_keypoints() -> list[tuple[float, float, float]] | None:
    """Read selected hand-joint world positions for debug markers.

    Queries the XR runtime directly (same virtual-world frame the retargeters use)
    for the joints in :data:`HAND_KEYPOINT_JOINTS`, left hand then right hand.

    Returns a flat list of (x, y, z) world positions ordered
    [L_wrist, L_thumb_tip, L_index_tip, R_wrist, R_thumb_tip, R_index_tip], or
    ``None`` if the XR runtime/hands are unavailable. Missing individual joints are
    skipped, so the list may be shorter than 6.
    """
    try:
        from omni.kit.xr.core import XRCore
    except ModuleNotFoundError:
        return None

    xr_core = XRCore.get_singleton() if XRCore is not None else None
    if xr_core is None:
        return None

    points: list[tuple[float, float, float]] = []
    for hand_path in ("/user/hand/left", "/user/hand/right"):
        hand_device = xr_core.get_input_device(hand_path)
        if hand_device is None:
            continue
        try:
            joint_poses = hand_device.get_all_virtual_world_poses()
        except Exception:
            continue
        if not joint_poses:
            continue
        for joint_name in HAND_KEYPOINT_JOINTS:
            pose = joint_poses.get(joint_name)
            if pose is None:
                continue
            try:
                t = pose.pose_matrix.ExtractTranslation()
            except Exception:
                continue
            points.append((float(t[0]), float(t[1]), float(t[2])))

    return points or None


def _get_pinch_distances() -> tuple[float | None, float | None]:
    """Return the thumb–index-tip distance (m) for each hand this frame.

    Reads directly from the XR runtime (same virtual-world frame the retargeters
    use). Returns ``(left_dist, right_dist)``; a value is ``None`` when that hand's
    tracking data is unavailable.
    """
    try:
        from omni.kit.xr.core import XRCore
    except ModuleNotFoundError:
        return None, None

    xr_core = XRCore.get_singleton() if XRCore is not None else None
    if xr_core is None:
        return None, None

    dists: list[float | None] = []
    for hand_path in ("/user/hand/left", "/user/hand/right"):
        hand_device = xr_core.get_input_device(hand_path)
        if hand_device is None:
            dists.append(None)
            continue
        try:
            joint_poses = hand_device.get_all_virtual_world_poses()
        except Exception:
            dists.append(None)
            continue
        if not joint_poses:
            dists.append(None)
            continue
        thumb = joint_poses.get("thumb_tip")
        index = joint_poses.get("index_tip")
        if thumb is None or index is None:
            dists.append(None)
            continue
        try:
            tp = thumb.pose_matrix.ExtractTranslation()
            ip = index.pose_matrix.ExtractTranslation()
        except Exception:
            dists.append(None)
            continue

        # Guard against uninitialized joint data: before hand tracking goes
        # live, the XR runtime returns (0,0,0) for every joint. The operator
        # is never at the world origin (the anchor offsets them ~1.7 m), so a
        # near-zero position is a reliable indicator of invalid tracking.
        # Returning None lets the latch treat the frame as "no data" (safe,
        # non-latching) rather than "distance 0 → always pinching".
        tp_norm = math.sqrt(float(tp[0]) ** 2 + float(tp[1]) ** 2 + float(tp[2]) ** 2)
        ip_norm = math.sqrt(float(ip[0]) ** 2 + float(ip[1]) ** 2 + float(ip[2]) ** 2)
        if tp_norm < 0.01 or ip_norm < 0.01:
            dists.append(None)
            continue

        dists.append(
            math.sqrt(
                (float(tp[0]) - float(ip[0])) ** 2
                + (float(tp[1]) - float(ip[1])) ** 2
                + (float(tp[2]) - float(ip[2])) ** 2
            )
        )

    left_dist = dists[0] if len(dists) > 0 else None
    right_dist = dists[1] if len(dists) > 1 else None
    return left_dist, right_dist


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

    # Resolve the effective XR anchor: start from the --view preset, then let any
    # explicit --anchor_* flag override the corresponding field. Precedence:
    # explicit flag > --view preset > task-config default.
    preset = VIEW_PRESETS[args_cli.view]
    anchor_prim_path = preset["anchor_prim_path"]
    anchor_pos = preset["anchor_pos"]
    anchor_rot = preset["anchor_rot"]
    if args_cli.anchor_pos is not None:
        anchor_pos = tuple(args_cli.anchor_pos)
    if args_cli.anchor_rot is not None:
        anchor_rot = tuple(args_cli.anchor_rot)
    if args_cli.anchor_prim_path is not None:
        anchor_prim_path = args_cli.anchor_prim_path

    # Mirror the resolved anchor onto env_cfg.xr. NOTE: this alone does NOT move the
    # headset -- OpenXRDevice.__init__ reads the anchor from its OWN cfg.xr_cfg (the
    # OpenXRDeviceCfg under teleop_devices), builds the XRAnchor xform from it, and
    # sets the carb customAnchor. env_cfg.xr is kept in sync only so the startup
    # banner reports the truth; the device's xr_cfg (set below) is what actually
    # places the viewpoint.
    env_cfg.xr.anchor_prim_path = anchor_prim_path
    env_cfg.xr.anchor_pos = anchor_pos
    env_cfg.xr.anchor_rot = anchor_rot

    # Validate that the selected task actually exposes a handtracking device. Failing
    # loud here is much friendlier than a cryptic AttributeError 200 lines later.
    if not hasattr(env_cfg, "teleop_devices") or args_cli.device_name not in env_cfg.teleop_devices.devices:
        logger.error(
            f"Task '{args_cli.task}' does not declare a teleop device named '{args_cli.device_name}'. "
            "Use one of the Isaac-Reach-MobileAI-IK-Abs-* tasks, which register the 'handtracking' device."
        )
        simulation_app.close()
        return

    # Push the resolved anchor onto the OpenXR device cfg -- this is the copy the
    # device actually consumes when building the headset anchor. Without this, the
    # --view preset and --anchor_* overrides are silently ignored and the headset
    # falls back to whatever the task config baked in (the first-person default).
    device_cfg = env_cfg.teleop_devices.devices[args_cli.device_name]
    device_xr_cfg = getattr(device_cfg, "xr_cfg", None)
    if device_xr_cfg is not None:
        device_xr_cfg.anchor_prim_path = anchor_prim_path
        device_xr_cfg.anchor_pos = anchor_pos
        device_xr_cfg.anchor_rot = anchor_rot
    else:
        logger.warning(
            f"Teleop device '{args_cli.device_name}' has no xr_cfg; --view/--anchor_* "
            "overrides cannot be applied and the viewpoint will use the config default."
        )

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
    # Staged start: by default we begin INACTIVE so the operator can get their
    # hands into a comfortable neutral pose first; a workstation key (N) then
    # engages teleop and the anchor is captured at that pose, so the arms never
    # jump on connect. --autostart restores the old behavior (engage as soon as
    # hand tracking warms up). Either way the warm-up guard still gates stepping.
    teleoperation_active = bool(args_cli.autostart)
    # Warm-up state: only forward actions to env.step() after both hands have
    # reported non-zero positions for `warmup_frames_required` consecutive
    # frames. This prevents the arms from snapping toward (0,0,0) while the
    # operator is still putting the headset on or before tracking goes live.
    warmup_frames_required = max(1, int(args_cli.warmup_frames))
    warmup_min_pos = float(args_cli.warmup_min_pos)
    warmup_complete = False
    warmup_valid_count = 0

    # Thumb–index distance (m) below which arm ORIENTATION is frozen (position
    # still tracks) so pinch-induced wrist re-estimation does not snap the TCP.
    pinch_hold_dist = float(args_cli.pinch_hold_dist)

    # Control-plane yaw correction. The OpenXR hand frame is rotated ~90 deg about
    # Z relative to the robot base, so "reach forward" was driving the arm sideways
    # (forward->left, right->forward) identically across all viewpoints. We rotate
    # the hand-motion delta by this fixed yaw (about the robot's +Z) before applying
    # it to the EE, which realigns "forward" with the robot's +X. Default -90 deg;
    # tunable via --control_yaw_deg (0 disables). Built once here as a wxyz quat.
    control_yaw_rad = math.radians(float(args_cli.control_yaw_deg))
    control_corr = torch.tensor(
        [math.cos(control_yaw_rad / 2.0), 0.0, 0.0, math.sin(control_yaw_rad / 2.0)],
        dtype=torch.float32,
        device=args_cli.device,
    )

    # Hand-anchor state. In hand_anchored mode, the EE target is composed as
    #   C           = R_ctrl * R_yaw_inv          (captured once at anchor time)
    #   delta       = C * (hand_curr_pos - hand_init_pos)
    #   target_pos  = ee_init_pos + delta
    #   target_quat = C * hand_curr_quat * quat_offset
    # where quat_offset = conj(hand_init) * conj(C) * ee_init is pre-baked at
    # anchor time so both position AND orientation use the same control frame C.
    # This means a hand tilt/roll maps to the matching EE tilt/roll (no axis
    # scramble).  R_yaw is the operator's head yaw (about Z) captured at anchor
    # time; R_ctrl (control_corr) is the fixed --control_yaw_deg correction that
    # aligns the OpenXR hand plane with the robot base. With both = identity this
    # reduces to a world-locked rigid-bracket coupling.
    # In absolute mode, these stay None and the raw action is forwarded as-is.
    anchor_mode = args_cli.anchor_mode
    anchor_captured = False
    hand_init_pos_l: torch.Tensor | None = None
    hand_init_pos_r: torch.Tensor | None = None
    ee_init_pos_l: torch.Tensor | None = None
    ee_init_pos_r: torch.Tensor | None = None
    quat_offset_l: torch.Tensor | None = None
    quat_offset_r: torch.Tensor | None = None
    head_yaw_inv: torch.Tensor | None = None
    # Full control frame C = R_ctrl · R_yaw_inv, captured at anchor time.
    # Used (instead of head_yaw_inv alone) for the orientation coupling so that
    # rotation and translation are always conjugated by the same frame.
    control_frame: torch.Tensor | None = None

    def _capture_anchor(action_14d: torch.Tensor) -> None:
        """Snapshot hand poses, EE poses, and head yaw to define delta anchors.

        Called the first frame teleop is actively forwarding actions. Captures both
        hand poses (from the retargeter output), both EE poses (from the robot
        articulation, in the robot base frame), and the operator's head yaw in the
        anchored frame. The head yaw defines the control frame so that hand motion
        "forward relative to the headset" maps to the robot's forward regardless of
        where the operator stands or faces.
        """
        nonlocal anchor_captured, hand_init_pos_l, hand_init_pos_r
        nonlocal ee_init_pos_l, ee_init_pos_r, quat_offset_l, quat_offset_r, head_yaw_inv, control_frame

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

        # Capture head yaw (about Z) in the anchored frame. r_yaw rotates a vector
        # FROM the head-forward frame INTO the anchored frame; its inverse does the
        # reverse, which is what we use to fold the operator's facing out of the
        # hand delta. Fall back to identity (world-locked behavior) if head pose
        # isn't available this frame.
        head_quat = _get_head_quat(hand_l_pos.device)
        if head_quat is None:
            r_yaw = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=hand_l_pos.device)
            head_source = "identity (head pose unavailable)"
        else:
            r_yaw = yaw_quat(head_quat)
            head_source = "headset"
        head_yaw_inv = quat_conjugate(r_yaw)
        yaw_deg = math.degrees(2.0 * math.atan2(float(r_yaw[3]), float(r_yaw[0])))

        # Full control frame: the same composite rotation used to transform position
        # deltas.  C = R_ctrl · R_yaw_inv  (control_corr may live on a different
        # device if the script was invoked with --device cpu; move it here).
        control_frame = quat_mul(control_corr.to(hand_l_pos.device), head_yaw_inv)

        hand_init_pos_l = hand_l_pos.clone()
        hand_init_pos_r = hand_r_pos.clone()
        ee_init_pos_l = ee_l_pos.clone()
        ee_init_pos_r = ee_r_pos.clone()

        # Orientation coupling using the same control frame C as the position delta.
        # The world-frame hand delta is dR = hand_curr · conj(hand_init); expressing
        # it in frame C and applying it to the EE start orientation gives:
        #   target = C · dR · conj(C) · ee_init
        #          = C · hand_curr · conj(hand_init) · conj(C) · ee_init
        # Pre-baking everything except hand_curr into quat_offset:
        #   quat_offset = conj(hand_init) · conj(C) · ee_init
        #   target      = C · hand_curr · quat_offset
        # This keeps rotation and translation in the same frame so a hand pitch/roll
        # produces the matching EE pitch/roll instead of a scrambled-axis rotation.
        C_conj = quat_conjugate(control_frame)
        quat_offset_l = quat_mul(quat_conjugate(hand_l_quat), quat_mul(C_conj, ee_l_quat))
        quat_offset_r = quat_mul(quat_conjugate(hand_r_quat), quat_mul(C_conj, ee_r_quat))

        anchor_captured = True
        print(
            "[ANCHOR] Captured. "
            f"head_yaw={yaw_deg:+.1f} deg [{head_source}]  "
            f"ctrl_yaw={args_cli.control_yaw_deg:+.1f} deg  "
            f"L ee0=[{ee_init_pos_l[0]:+.3f}, {ee_init_pos_l[1]:+.3f}, {ee_init_pos_l[2]:+.3f}]  "
            f"R ee0=[{ee_init_pos_r[0]:+.3f}, {ee_init_pos_r[1]:+.3f}, {ee_init_pos_r[2]:+.3f}]  "
            "(EE mirrors head-relative hand delta from now on)"
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

    def reanchor() -> None:
        nonlocal anchor_captured
        # Re-snapshot the hand<->EE relationship (and head yaw) on the next active
        # frame WITHOUT pausing or resetting. Use after physically re-orienting the
        # body so "forward relative to the headset" maps to robot-forward again.
        anchor_captured = False
        print("[ANCHOR] Re-anchor requested -- will re-snapshot on next active frame")

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

    # -- Workstation keyboard sidecar --
    # A second user at the workstation drives the session controls (the headset
    # operator has no keyboard). The OpenXR device's START/STOP/RESET only fire
    # from CloudXR carb events, which ALVR+SteamVR doesn't publish, so we add a
    # plain Se3Keyboard purely for its key callbacks (its SE3 motion output is
    # ignored). Its advance() is pumped each loop so events are processed.
    #   N -> start/engage teleop      M -> pause (hold pose, re-anchor on resume)
    #   B -> re-anchor (no pause)     J -> reset environment
    #
    # Key choice matters: SPACE is Kit's timeline play/pause (pressing it both
    # engaged teleop AND paused the sim), and P/R/ENTER are bound to other Kit
    # shortcuts. N/M/B/J were verified clear of Kit and of Se3Keyboard's built-in
    # SE(3) keys (K, W, S, A, D, Q, E, Z, X, T, G, C, V, L).
    keyboard_interface = None
    try:
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg())
        keyboard_interface.add_callback("N", start_teleop)
        keyboard_interface.add_callback("M", stop_teleop)
        keyboard_interface.add_callback("B", reanchor)
        keyboard_interface.add_callback("J", reset_env)
    except Exception as exc:
        logger.warning(
            f"Failed to create keyboard sidecar ({exc}); workstation key controls "
            "(N/M/B/J) will be unavailable. Teleop can still run, but consider "
            "--autostart so it engages without a key press."
        )
        keyboard_interface = None

    # -- Debug markers (EE goal frames + hand keypoints) --
    show_markers = not args_cli.no_hand_markers
    ee_goal_markers = None
    hand_kp_markers = None
    if show_markers:
        try:
            ee_marker_cfg = FRAME_MARKER_CFG.copy()
            ee_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
            ee_goal_markers = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_goals"))
            hand_kp_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/hand_keypoints",
                    markers={
                        "keypoint": sim_utils.SphereCfg(
                            radius=0.012,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
                        ),
                    },
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to create debug markers ({exc}); continuing without them.")
            ee_goal_markers = None
            hand_kp_markers = None

    print(f"\nUsing teleop device: {teleop_interface}")
    print("=" * 60)
    print("VR DUAL-ARM TELEOPERATION (hand tracking)")
    print(f"  Task:        {args_cli.task}")
    print(f"  Action dim:  {ACTION_DIM_ENV} (left 7D + right 7D pose + 2 binary grippers)")
    print(f"  View:        {args_cli.view}")
    print("  Gripper:     pinch (thumb-index) -> binary open/close per hand")
    if anchor_mode == "hand_anchored":
        print(
            "  Anchor mode: hand_anchored "
            "(EE mirrors head-relative hand DELTA from snapshot at first active frame)"
        )
        print(
            f"  Control yaw: {args_cli.control_yaw_deg:+.1f} deg "
            "(rotates hand-motion plane so 'forward' -> robot +X; --control_yaw_deg, 0=off)"
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
    if args_cli.autostart:
        print("  Start:       autostart (engages as soon as hand tracking warms up)")
    else:
        print("  Start:       staged -- position hands, then press N at the workstation")
    if pinch_hold_dist > 0.0:
        print(
            f"  Pinch latch: orientation frozen when thumb–index < {pinch_hold_dist:.3f} m "
            "(position still tracks; --pinch_hold_dist 0 to disable)"
        )
    else:
        print("  Pinch latch: disabled (--pinch_hold_dist 0)")
    print(f"  Markers:     {'on (EE goals + hand keypoints)' if show_markers else 'off'}")
    if keyboard_interface is not None:
        print("  Keys:        N=start  M=pause  B=re-anchor  J=reset  (workstation keyboard)")
    else:
        print("  Keys:        keyboard sidecar unavailable -- use --autostart to engage")
    print("=" * 60)

    def _draw_debug_markers(target_pose: torch.Tensor | None) -> None:
        """Update EE-goal frame markers and hand-keypoint sphere markers.

        ``target_pose`` is the 14D IK pose target in the robot base frame (or None
        when teleop isn't actively stepping). EE goals are converted base->world for
        drawing; hand keypoints are read from the XR runtime in world frame. Any
        failure disables markers for the rest of the session rather than crashing.
        """
        nonlocal show_markers
        try:
            if ee_goal_markers is not None and target_pose is not None:
                root_pos_w = robot.data.root_pos_w[0].to(target_pose.device).unsqueeze(0)
                root_quat_w = robot.data.root_quat_w[0].to(target_pose.device).unsqueeze(0)
                gl_pos, gl_quat = combine_frame_transforms(
                    root_pos_w, root_quat_w, target_pose[0:3].unsqueeze(0), target_pose[3:7].unsqueeze(0)
                )
                gr_pos, gr_quat = combine_frame_transforms(
                    root_pos_w, root_quat_w, target_pose[7:10].unsqueeze(0), target_pose[10:14].unsqueeze(0)
                )
                ee_goal_markers.visualize(
                    translations=torch.cat([gl_pos, gr_pos], dim=0),
                    orientations=torch.cat([gl_quat, gr_quat], dim=0),
                )

            if hand_kp_markers is not None:
                pts = _get_hand_keypoints()
                if pts:
                    hand_kp_markers.visualize(
                        translations=torch.tensor(pts, dtype=torch.float32, device=args_cli.device)
                    )
        except Exception as exc:
            logger.warning(f"Marker visualization failed ({exc}); disabling markers.")
            show_markers = False

    # Pinch-latch state.  When thumb–index distance drops below `pinch_hold_dist`,
    # orientation is frozen to the last open-hand value while position keeps
    # tracking (so the operator can carry objects while gripping).  On release,
    # quat_offset is re-snapshotted so the wrist jump on finger un-occlusion
    # does not snap the TCP.
    latch_l_quat: torch.Tensor | None = None
    latch_r_quat: torch.Tensor | None = None
    l_was_pinching = False
    r_was_pinching = False

    # -- Reset --
    env.reset()
    teleop_interface.reset()

    step_count = 0

    # -- Main loop --
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # Pump the workstation keyboard so its key callbacks (N/M/B/J)
                # are processed. We ignore its SE3 motion output.
                if keyboard_interface is not None:
                    keyboard_interface.advance()

                # 1D tensor of shape [16]: [L_pose(7), R_pose(7), L_grip(1), R_grip(1)]
                raw = teleop_interface.advance()

                if raw is None or raw.numel() < ACTION_DIM_ENV:
                    # No XR data yet (operator hasn't put the headset on, or tracking
                    # is briefly lost). Render to keep the window responsive and try
                    # again next frame.
                    env.sim.render()
                    step_count += 1
                    continue

                # Full 16D action: [L_pose(7), R_pose(7), L_grip(1), R_grip(1)].
                action_full = raw[:ACTION_DIM_ENV].to(dtype=torch.float32, device=args_cli.device)
                pose_part = action_full[:POSE_DIM]  # 14D arm poses
                grip_part = action_full[POSE_DIM:ACTION_DIM_ENV]  # 2D binary grippers

                # Gripper scalars (+1 open / -1 close) for logging; they are also
                # forwarded to the env's binary gripper actions via grip_part.
                left_grip = float(action_full[LEFT_GRIP_IDX].item())
                right_grip = float(action_full[RIGHT_GRIP_IDX].item())

                # Warm-up gate: only forward actions once both hands have been
                # reporting non-zero positions for N consecutive frames. Until
                # then we render only, so the operator can put the headset on
                # without the arms jumping toward the OpenXR origin.
                l_pos_norm = float(pose_part[:3].norm().item())
                r_pos_norm = float(pose_part[7:10].norm().item())
                if not warmup_complete:
                    if l_pos_norm > warmup_min_pos and r_pos_norm > warmup_min_pos:
                        warmup_valid_count += 1
                        if warmup_valid_count >= warmup_frames_required:
                            warmup_complete = True
                            ready_msg = (
                                "arms now driven by VR."
                                if teleoperation_active
                                else "press N at the workstation to engage."
                            )
                            print(
                                f"[WARMUP] Hand tracking stable after "
                                f"{warmup_valid_count} frames -- {ready_msg}"
                            )
                    else:
                        warmup_valid_count = 0

                step_actions = teleoperation_active and warmup_complete
                target_pose = None  # 14D pose target, kept for marker drawing
                if step_actions:
                    if anchor_mode == "hand_anchored":
                        # First active frame after warm-up / reset / STOP->START: snap
                        # the hand-EE relationship into place. Subsequent frames will
                        # use the captured offsets so the EE mirrors only hand deltas.
                        if not anchor_captured:
                            _capture_anchor(pose_part)

                        hand_l_pos = pose_part[0:3]
                        hand_l_quat = pose_part[3:7]
                        hand_r_pos = pose_part[7:10]
                        hand_r_quat = pose_part[10:14]

                        # Position: rotate the hand delta out of the operator's
                        # head-yaw frame, then apply the fixed control-plane yaw
                        # correction (control_corr) that aligns the OpenXR hand frame
                        # with the robot base, then offset onto the EE start pose.
                        # "Reach forward" -> robot reaches forward (+X) instead of
                        # sideways (see _capture_anchor / --control_yaw_deg).
                        delta_l = quat_apply(control_corr, quat_apply(head_yaw_inv, hand_l_pos - hand_init_pos_l))
                        delta_r = quat_apply(control_corr, quat_apply(head_yaw_inv, hand_r_pos - hand_init_pos_r))
                        target_l_pos = ee_init_pos_l + delta_l
                        target_r_pos = ee_init_pos_r + delta_r
                        # Orientation: rigid-bracket coupling in the full control frame C.
                        #   target = C * hand_curr * quat_offset
                        # C = R_ctrl · R_yaw_inv is the same frame used for position
                        # deltas, so a hand pitch/roll maps to the matching EE axis.
                        target_l_quat = quat_mul(control_frame, quat_mul(hand_l_quat, quat_offset_l))
                        target_r_quat = quat_mul(control_frame, quat_mul(hand_r_quat, quat_offset_r))

                        target_pose = torch.cat(
                            [target_l_pos, target_l_quat, target_r_pos, target_r_quat]
                        )
                    else:
                        # 'absolute' mode: pass the hand pose straight through as the
                        # IK target. Only sensible when the operator's body is meant
                        # to coincide with the robot.
                        target_pose = pose_part

                    # Pinch latch: freeze ORIENTATION only while pinching so the
                    # Quest wrist re-estimation artefact does not snap the TCP.
                    # Position keeps tracking so the operator can move while gripping.
                    # On release, re-snapshot quat_offset so the wrist jump on finger
                    # un-occlusion does not move the EE either.
                    if pinch_hold_dist > 0.0:
                        l_pinch_dist, r_pinch_dist = _get_pinch_distances()
                        l_pinching = l_pinch_dist is not None and l_pinch_dist < pinch_hold_dist
                        r_pinching = r_pinch_dist is not None and r_pinch_dist < pinch_hold_dist

                        # Left arm
                        if l_pinching and latch_l_quat is not None:
                            target_pose = torch.cat(
                                [target_pose[0:3], latch_l_quat, target_pose[7:14]]
                            )
                        elif not l_pinching:
                            if (
                                l_was_pinching
                                and latch_l_quat is not None
                                and anchor_mode == "hand_anchored"
                            ):
                                # Release edge: re-snapshot orientation coupling.
                                hand_l_quat = pose_part[3:7]
                                C_conj = quat_conjugate(control_frame)
                                quat_offset_l = quat_mul(
                                    quat_conjugate(hand_l_quat),
                                    quat_mul(C_conj, latch_l_quat),
                                )
                                target_l_quat = quat_mul(
                                    control_frame, quat_mul(hand_l_quat, quat_offset_l)
                                )
                                target_pose = torch.cat(
                                    [target_pose[0:3], target_l_quat, target_pose[7:14]]
                                )
                            latch_l_quat = target_pose[3:7].clone()

                        # Right arm
                        if r_pinching and latch_r_quat is not None:
                            target_pose = torch.cat(
                                [target_pose[0:7], target_pose[7:10], latch_r_quat]
                            )
                        elif not r_pinching:
                            if (
                                r_was_pinching
                                and latch_r_quat is not None
                                and anchor_mode == "hand_anchored"
                            ):
                                hand_r_quat = pose_part[10:14]
                                C_conj = quat_conjugate(control_frame)
                                quat_offset_r = quat_mul(
                                    quat_conjugate(hand_r_quat),
                                    quat_mul(C_conj, latch_r_quat),
                                )
                                target_r_quat = quat_mul(
                                    control_frame, quat_mul(hand_r_quat, quat_offset_r)
                                )
                                target_pose = torch.cat(
                                    [target_pose[0:7], target_pose[7:10], target_r_quat]
                                )
                            latch_r_quat = target_pose[10:14].clone()

                        l_was_pinching = l_pinching
                        r_was_pinching = r_pinching

                    # Assemble the full 16D action: arm poses + the two binary
                    # gripper scalars straight from the GripperRetargeter.
                    target_action = torch.cat([target_pose, grip_part])
                    actions = target_action.unsqueeze(0).repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    # Render only — robot holds its last IK target. Without this,
                    # the viewer freezes while we're waiting on warm-up or paused.
                    env.sim.render()

                # Debug markers: EE goal frames (converted base->world) and the
                # tracked hand keypoints. Guarded so a viz error never kills teleop.
                if show_markers:
                    _draw_debug_markers(target_pose)

                if step_count % 60 == 0:
                    l_pos = pose_part[:3].tolist()
                    r_pos = pose_part[7:10].tolist()
                    if not warmup_complete:
                        state = f"WARMUP {warmup_valid_count}/{warmup_frames_required}"
                    elif not teleoperation_active:
                        state = "ARMED/WAITING" if not args_cli.autostart else "PAUSED"
                    elif anchor_mode == "hand_anchored" and not anchor_captured:
                        state = "ANCHORING"
                    else:
                        state = f"ON/{anchor_mode}"
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
                    # Clear pinch latches: the EE is at a new reset pose so any
                    # previously latched orientation would be stale.
                    latch_l_quat = None
                    latch_r_quat = None
                    l_was_pinching = False
                    r_was_pinching = False
                    should_reset = False
                    print("[RESET] Done -- waiting for warm-up + re-anchor to re-engage teleop")

        except Exception as e:
            logger.error(f"Simulation step error: {e}")
            break

    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
