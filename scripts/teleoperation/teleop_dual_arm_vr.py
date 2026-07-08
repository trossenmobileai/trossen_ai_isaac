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
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_scripts_dir = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
sys.path.insert(0, str(_scripts_dir / "lib"))
from vr_cli_loader import load_vr_cli

vr_cli = load_vr_cli()

parser = argparse.ArgumentParser(
    description="VR hand-tracking teleoperation for the Mobile AI bimanual robot."
)
vr_cli.add_vr_teleop_args(parser)
vr_cli.add_vr_camera_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.keep_cameras)

# Force XR runtime ON before constructing AppLauncher.
app_launcher_args = vars(args_cli)
app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from trossen_ai_isaac.teleop.vr import run_vr_teleop_loop


if __name__ == "__main__":
    run_vr_teleop_loop(simulation_app, args_cli)
    simulation_app.close()

