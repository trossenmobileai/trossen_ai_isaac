"""Argparse helpers shared by VR teleop and VR recording entrypoints."""

from __future__ import annotations

import argparse


def add_vr_teleop_args(parser: argparse.ArgumentParser) -> None:
    """Add common Mobile AI VR teleop flags."""
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
        "--dual_arm",
        action="store_true",
        help=(
            "Teleoperate BOTH arms simultaneously (left hand -> left arm, right hand -> "
            "right arm). Default (flag absent) is single-arm mode, where only one arm "
            "tracks its hand at a time and the other holds its last pose; press TAB at the "
            "workstation to switch the active arm. Single-arm mode avoids recording bad data "
            "when the idle hand's tracking drifts or drops."
        ),
    )
    parser.add_argument(
        "--start_arm",
        type=str,
        default="left",
        choices=["left", "right"],
        help=(
            "Which arm is active first in single-arm mode. Ignored when --dual_arm is set. "
            "Switch at runtime with TAB."
        ),
    )
    parser.add_argument(
        "--pinch_hold_dist",
        type=float,
        default=0.08,
        help=(
            "Thumb-index distance (meters) below which the arm ORIENTATION is frozen "
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


def add_vr_camera_args(parser: argparse.ArgumentParser) -> None:
    """Add XR camera compatibility experiment flags."""
    parser.add_argument(
        "--keep_cameras",
        action="store_true",
        help=(
            "Keep task-declared USD cameras enabled during XR instead of stripping them. "
            "Use this for VR recording and camera compatibility experiments."
        ),
    )
    parser.add_argument(
        "--camera_probe_interval",
        type=int,
        default=0,
        help=(
            "If > 0, probe the three record-camera RGB outputs every N simulation steps and "
            "print compatibility status. Useful for staged XR-plus-camera experiments."
        ),
    )
    parser.add_argument(
        "--camera_probe_capture_frame",
        action="store_true",
        help=(
            "During camera probes, also run the full recording frame-capture path "
            "(state/action/images) to validate dataset-read compatibility."
        ),
    )
    parser.add_argument(
        "--camera_probe_output",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON report summarizing camera probe successes/failures. "
            "Written on exit when camera probing is enabled."
        ),
    )
