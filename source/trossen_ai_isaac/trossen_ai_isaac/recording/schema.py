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

"""LeRobot feature schema aligned with ``trossen_ai_2026_project``."""

from __future__ import annotations

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

CAMERA_KEYS: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_SHAPE: tuple[int, int, int] = (480, 640, 3)


def lerobot_features() -> dict:
    """Return the LeRobot ``features`` dict for ``LeRobotDataset.create()``."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
    }
    for cam_key in CAMERA_KEYS:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
        }
    return features
