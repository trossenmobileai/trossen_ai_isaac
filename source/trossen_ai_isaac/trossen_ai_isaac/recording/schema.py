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

"""LeRobot feature schema aligned with LeRobot Dataset v3.0."""

from __future__ import annotations

from dataclasses import dataclass

# Logical joint names in LeRobot dataset order (14D).
JOINT_NAMES: list[str] = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
]

# Per-arm joint-name subsets (6 arm joints + 1 gripper each).
LEFT_JOINT_NAMES: list[str] = JOINT_NAMES[0:7]
RIGHT_JOINT_NAMES: list[str] = JOINT_NAMES[7:14]

CAMERA_KEYS: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_SHAPE: tuple[int, int, int] = (480, 640, 3)

# Wrist camera paired with each single-arm recording mode.
ARM_WRIST_CAMERA: dict[str, str] = {
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}


@dataclass(frozen=True)
class RecordingLayout:
    """Resolved joint slice + camera selection for a recording arm mode.

    ``joint_indices`` are positions into the full 14D capture vector produced by
    ``record_joint_pos_14`` / ``record_joint_target_14``; ``camera_keys`` is the
    subset of scene cameras written to the dataset for this mode.
    """

    arm_mode: str  # "left" | "right" | "both"
    joint_names: list[str]
    joint_indices: list[int]
    camera_keys: tuple[str, ...]

    @staticmethod
    def from_arm_mode(mode: str) -> "RecordingLayout":
        """Build the layout for ``mode`` ("left", "right", or "both")."""
        if mode == "both":
            return RecordingLayout("both", list(JOINT_NAMES), list(range(14)), CAMERA_KEYS)
        if mode == "left":
            return RecordingLayout(
                "left", list(LEFT_JOINT_NAMES), list(range(0, 7)), ("cam_high", "cam_left_wrist")
            )
        if mode == "right":
            return RecordingLayout(
                "right", list(RIGHT_JOINT_NAMES), list(range(7, 14)), ("cam_high", "cam_right_wrist")
            )
        raise ValueError(f"Unknown record arm mode: {mode!r} (expected 'left', 'right', or 'both')")


def lerobot_features(layout: RecordingLayout | None = None) -> dict:
    """Return the LeRobot ``features`` dict for ``LeRobotDataset.create()``.

    ``layout`` selects which joints and cameras are declared; when omitted the
    full dual-arm layout (14D + 3 cameras) is used for backward compatibility.
    """
    if layout is None:
        layout = RecordingLayout.from_arm_mode("both")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(layout.joint_names),),
            "names": {"motors": list(layout.joint_names)},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(layout.joint_names),),
            "names": {"motors": list(layout.joint_names)},
        },
    }
    for cam_key in layout.camera_keys:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        }
    return features
