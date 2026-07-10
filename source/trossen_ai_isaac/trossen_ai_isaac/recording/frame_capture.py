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

"""Capture observations from the Mobile AI record environment."""

from __future__ import annotations

import numpy as np

from trossen_ai_isaac.recording.schema import RecordingLayout
from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.mdp.observations import (
    record_joint_pos_14,
    record_joint_target_14,
)


def _resolve_layout(layout: RecordingLayout | None) -> RecordingLayout:
    """Default to the full dual-arm layout when none is provided."""
    return layout if layout is not None else RecordingLayout.from_arm_mode("both")


def capture_state(env, layout: RecordingLayout | None = None) -> np.ndarray:
    """Return follower joint positions (sliced to ``layout``) as float32."""
    layout = _resolve_layout(layout)
    joints = record_joint_pos_14(env)[0].detach().cpu().numpy().astype(np.float32)
    return joints[layout.joint_indices]


def capture_action(env, layout: RecordingLayout | None = None) -> np.ndarray:
    """Return commanded joint targets (sliced to ``layout``) as float32."""
    layout = _resolve_layout(layout)
    joints = record_joint_target_14(env)[0].detach().cpu().numpy().astype(np.float32)
    return joints[layout.joint_indices]


def capture_images(env, layout: RecordingLayout | None = None) -> dict[str, np.ndarray]:
    """Return uint8 HWC RGB images keyed by the camera scene names in ``layout``."""
    layout = _resolve_layout(layout)
    images: dict[str, np.ndarray] = {}
    for cam_key in layout.camera_keys:
        rgb = env.scene[cam_key].data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        images[cam_key] = np.asarray(rgb, dtype=np.uint8).copy()
    return images


def capture_frame(env, task: str, layout: RecordingLayout | None = None) -> dict:
    """Bundle state, commanded action, images, and task string for LeRobot.

    ``layout`` selects which joints and cameras are recorded; when omitted the
    full dual-arm layout (14D + 3 cameras) is used for backward compatibility.
    """
    layout = _resolve_layout(layout)
    state = capture_state(env, layout)
    action = capture_action(env, layout)
    images = capture_images(env, layout)
    frame = {
        "observation.state": state,
        "action": action,
        "task": task,
    }
    for cam_key, image in images.items():
        frame[f"observation.images.{cam_key}"] = image
    return frame
