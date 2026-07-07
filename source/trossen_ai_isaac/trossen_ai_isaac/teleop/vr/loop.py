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

"""VR hand-tracking teleoperation loop for Mobile AI."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
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
    yaw_quat,
)

from trossen_ai_isaac.recording.camera_compat import CameraCompatProbe
from trossen_ai_isaac.recording.schema import CAMERA_KEYS
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg
from trossen_ai_isaac.teleop.session import TeleopSession, shutdown_requested
from trossen_ai_isaac.teleop.vr.constants import (
    ACTION_DIM_ENV,
    LEFT_EE_BODY_NAME,
    LEFT_GRIP_IDX,
    POSE_DIM,
    RIGHT_EE_BODY_NAME,
    RIGHT_GRIP_IDX,
    VIEW_PRESETS,
)
from trossen_ai_isaac.teleop.vr.hand_tracking import (
    _get_ee_base_pose,
    _get_hand_keypoints,
    _get_head_quat,
    _get_pinch_distances,
)

if TYPE_CHECKING:
    from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder

logger = logging.getLogger(__name__)


def _should_keep_cameras(args_cli) -> bool:
    """Return whether the VR run should preserve task-declared camera sensors."""
    return bool(
        getattr(args_cli, "keep_cameras", False)
        or getattr(args_cli, "camera_probe_interval", 0) > 0
        or getattr(args_cli, "enable_cameras", False)
    )


def build_vr_env_cfg(args_cli):
    """Build the VR env config and optionally preserve dataset cameras."""
    env_cfg = make_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    keep_cameras = _should_keep_cameras(args_cli)
    if keep_cameras:
        scene_cfg = getattr(env_cfg, "scene", None)
        available = [name for name in CAMERA_KEYS if scene_cfg is not None and hasattr(scene_cfg, name)]
        if available:
            print(f"[VR CAMERA MODE] Keeping cameras enabled: {available}")
        else:
            print("[VR CAMERA MODE] Requested camera retention, but no record cameras were declared by the task")
    else:
        # XR commonly conflicts with USD sensor render products; keep the original
        # camera-stripping behavior unless the operator explicitly requests otherwise.
        env_cfg = remove_camera_configs(env_cfg)
        print("[VR CAMERA MODE] Stripping task-declared cameras for XR teleop")

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

    if not hasattr(env_cfg, "teleop_devices") or args_cli.device_name not in env_cfg.teleop_devices.devices:
        raise ValueError(
            f"Task '{args_cli.task}' does not declare a teleop device named '{args_cli.device_name}'. "
            "Use one of the Isaac-Reach-MobileAI-IK-Abs-* tasks, which register the 'handtracking' device."
        )

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

    env_cfg.sim.render.antialiasing_mode = "DLSS"
    return env_cfg


def run_vr_teleop_loop(simulation_app, args_cli) -> None:
    """Run dual-arm VR teleoperation until the simulation app stops."""
    try:
        env_cfg = build_vr_env_cfg(args_cli)
    except ValueError as exc:
        logger.error("%s", exc)
        simulation_app.close()
        return

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return
    try:
        run_vr_recording_loop(simulation_app, env, env_cfg, args_cli, recorder=None)
    finally:
        env.close()


def run_vr_recording_loop(
    simulation_app,
    env,
    env_cfg,
    args_cli,
    recorder: LeRobotRecorder | None = None,
) -> None:
    """Run the VR teleop core loop with optional camera probing and recording."""

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
    # Staged start: by default we begin INACTIVE so the operator can get their
    # hands into a comfortable neutral pose first; a workstation key (N) then
    # engages teleop and the anchor is captured at that pose, so the arms never
    # jump on connect. --autostart restores the old behavior (engage as soon as
    # hand tracking warms up). Either way the warm-up guard still gates stepping.
    session = TeleopSession(teleoperation_active=bool(args_cli.autostart))
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
    camera_probe = None
    camera_probe_interval = max(0, int(getattr(args_cli, "camera_probe_interval", 0)))
    if camera_probe_interval > 0:
        camera_probe = CameraCompatProbe(
            task=getattr(args_cli, "task_description", args_cli.task),
            output_path=getattr(args_cli, "camera_probe_output", None),
            capture_frame_during_probe=bool(getattr(args_cli, "camera_probe_capture_frame", False)),
        )

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
        if recorder is not None and session.episode_recording_active:
            session.episode_recording_active = False
            recorder.discard_episode()
            print("[RECORD] Recording stopped -- episode discarded before reset")
        session.request_reset()
        print("[RESET] Environment will reset on next step")

    def start_teleop() -> None:
        session.start()
        print("[TELEOP] Activated (left+right hands -> left+right arms)")

    def stop_teleop() -> None:
        nonlocal anchor_captured
        session.stop()
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

    def toggle_episode_recording() -> None:
        if recorder is None:
            return
        if not session.teleoperation_active:
            start_teleop()
        if not session.episode_recording_active:
            recorder.discard_episode()
            session.episode_recording_active = True
            print("[RECORD] Episode recording started -- press N again to save and reset")
            return
        session.episode_recording_active = False
        recorder.save_episode()
        session.request_reset()
        print("[RECORD] Episode saved -- resetting robot to initial pose")

    def discard_episode() -> None:
        if recorder is None:
            return
        session.episode_recording_active = False
        recorder.discard_episode()
        print("[RECORD] Episode buffer discarded -- recording stopped")

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
    #   teleop-only mode:   N -> start   M -> pause   B -> re-anchor   J -> reset
    #   recording mode:     U -> start   I -> pause   N -> start/save episode
    #                       M -> discard  B -> re-anchor   J -> reset
    #
    # Key choice matters: SPACE is Kit's timeline play/pause (pressing it both
    # engaged teleop AND paused the sim), and P/R/ENTER are bound to other Kit
    # shortcuts. N/M/B/J were verified clear of Kit and of Se3Keyboard's built-in
    # SE(3) keys (K, W, S, A, D, Q, E, Z, X, T, G, C, V, L).
    keyboard_interface = None
    try:
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg())
        if recorder is None:
            keyboard_interface.add_callback("N", start_teleop)
            keyboard_interface.add_callback("M", stop_teleop)
        else:
            keyboard_interface.add_callback("U", start_teleop)
            keyboard_interface.add_callback("I", stop_teleop)
            keyboard_interface.add_callback("N", toggle_episode_recording)
            keyboard_interface.add_callback("M", discard_episode)
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
    if keyboard_interface is not None and recorder is None:
        print("  Keys:        N=start  M=pause  B=re-anchor  J=reset  (workstation keyboard)")
    elif keyboard_interface is not None:
        print("  Keys:        U=start  I=pause  N=record/save  M=discard  B=re-anchor  J=reset")
    else:
        print("  Keys:        keyboard sidecar unavailable -- use --autostart to engage")
    if recorder is not None:
        print("  Recording:   N=toggle episode  M=discard  J=reset  (U/I control teleop)")
        print(f"  Dataset:     {recorder.dataset_root}")
    if camera_probe is not None:
        probe_mode = "frame_capture" if camera_probe.capture_frame_during_probe else "rgb_only"
        print(f"  Camera probe:{camera_probe_interval} step(s)  mode={probe_mode}")
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
    while simulation_app.is_running() and not shutdown_requested():
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
                                if session.teleoperation_active
                                else (
                                    "press N at the workstation to start recording."
                                    if recorder is not None
                                    else "press N at the workstation to engage."
                                )
                            )
                            print(
                                f"[WARMUP] Hand tracking stable after "
                                f"{warmup_valid_count} frames -- {ready_msg}"
                            )
                    else:
                        warmup_valid_count = 0

                step_actions = session.teleoperation_active and warmup_complete
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
                    if recorder is not None and session.episode_recording_active:
                        recorder.on_step(env)
                else:
                    # Render only — robot holds its last IK target. Without this,
                    # the viewer freezes while we're waiting on warm-up or paused.
                    env.sim.render()

                if camera_probe is not None and step_count % camera_probe_interval == 0:
                    camera_probe.probe(env, step_count)

                # Debug markers: EE goal frames (converted base->world) and the
                # tracked hand keypoints. Guarded so a viz error never kills teleop.
                if show_markers:
                    _draw_debug_markers(target_pose)

                if step_count % 60 == 0:
                    l_pos = pose_part[:3].tolist()
                    r_pos = pose_part[7:10].tolist()
                    if not warmup_complete:
                        state = f"WARMUP {warmup_valid_count}/{warmup_frames_required}"
                    elif not session.teleoperation_active:
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
                        f"{' REC' if session.episode_recording_active else ''}"
                    )
                step_count += 1

                if session.consume_reset():
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
                    print("[RESET] Done -- waiting for warm-up + re-anchor to re-engage teleop")

        except Exception as e:
            logger.error(f"Simulation step error: {e}")
            break

    if shutdown_requested():
        print("[EXIT] Shutdown requested -- leaving VR teleop loop.")
    if camera_probe is not None:
        camera_probe.finalize()
    print("Environment closed.")
