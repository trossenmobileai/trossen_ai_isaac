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

"""OpenXR hand tracking helpers for VR teleoperation."""

from __future__ import annotations

import logging
import math

import torch
from isaaclab.utils.math import subtract_frame_transforms, yaw_quat

from trossen_ai_isaac.teleop.vr.constants import HAND_KEYPOINT_JOINTS

logger = logging.getLogger(__name__)

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
