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

"""LeRobot dataset writer for sim teleoperation episodes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from trossen_ai_isaac.recording.frame_capture import capture_frame

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

TROSSEN_AI_2026_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "motors": [
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
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "motors": [
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
        },
    },
    "observation.images.cam_high": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.cam_left_wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.cam_right_wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
}


class LeRobotRecorder:
    """Thin wrapper around ``LeRobotDataset`` for per-step sim recording."""

    def __init__(
        self,
        repo_id: str,
        fps: int,
        task: str,
        root: str | Path | None = None,
        robot_type: str = "trossen_ai_mobile",
        overwrite: bool = False,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobot is required for recording. On Isaac Sim (Python 3.11) install with: "
                "pip install lerobot==0.4.4 && pip install 'numpy>=1.26,<2'"
            ) from exc

        root_path = Path(root).expanduser() if root is not None else None
        if root_path is not None and root_path.exists():
            if overwrite:
                shutil.rmtree(root_path)
            else:
                raise FileExistsError(
                    f"Dataset root already exists: {root_path}. "
                    "Pass --overwrite to replace it or choose a new --root."
                )

        self.task = task
        self._frame_count = 0
        self._finalized = False
        self._dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=TROSSEN_AI_2026_FEATURES,
            robot_type=robot_type,
            root=root,
            use_videos=True,
            image_writer_threads=4,
        )
        logger.info("LeRobot dataset created at %s (repo_id=%s, fps=%d)", self._dataset.root, repo_id, fps)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dataset_root(self) -> Path:
        return Path(self._dataset.root)

    def on_step(self, env) -> None:
        """Capture and append one frame after ``env.step()``."""
        frame = capture_frame(env, task=self.task)
        self._dataset.add_frame(frame)
        self._frame_count += 1

    def save_episode(self) -> None:
        """Flush the current episode buffer to disk."""
        if self._frame_count == 0:
            print("[RECORD] No frames to save — episode discarded.")
            return
        self._dataset.save_episode()
        print(f"[RECORD] Saved episode ({self._frame_count} frames) -> {self.dataset_root}")
        self._frame_count = 0

    def discard_episode(self) -> None:
        """Drop buffered frames without writing an episode."""
        if self._frame_count == 0:
            print("[RECORD] No frames in buffer to discard.")
            return
        if hasattr(self._dataset, "clear_episode_buffer"):
            self._dataset.clear_episode_buffer()
        self._frame_count = 0
        print("[RECORD] Discarded current episode buffer.")

    def on_reset(self, save_if_non_empty: bool = True) -> None:
        """Optionally save the current episode before an environment reset."""
        if save_if_non_empty and self._frame_count > 0:
            self.save_episode()
        elif self._frame_count > 0:
            self.discard_episode()

    def finalize(self) -> None:
        """Close writers and flush metadata (call before exit). Safe to call multiple times."""
        if self._finalized:
            return
        if self._frame_count > 0:
            self.save_episode()
        if hasattr(self._dataset, "finalize"):
            self._dataset.finalize()
        self._finalized = True
        print(f"[RECORD] Finalized dataset at {self.dataset_root}")
