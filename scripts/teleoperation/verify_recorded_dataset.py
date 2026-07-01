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

"""Offline validation for a recorded Mobile AI LeRobot dataset.

Checks schema metadata, parquet integrity, MP4 pixel content, and optional
LeRobotDataset round-trip loading. Does not require Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import av
except ImportError as exc:
    raise SystemExit(
        "PyAV is required for MP4 checks. Run with your LeRobot venv, e.g.\n"
        "  ~/lerobot_trossen/.venv/bin/python scripts/teleoperation/verify_recorded_dataset.py ...\n"
        "Or install: pip install av pyarrow"
    ) from exc

import pyarrow.parquet as pq

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EXPECTED_ROBOT_TYPE = "trossen_ai_mobile"
MIN_PIXEL_MAX = 10


def _image_max(arr) -> float:
    """Return the maximum pixel value on a 0–255 scale (handles float or uint8)."""
    import numpy as np

    data = np.asarray(arr)
    if data.dtype.kind == "f" and data.max() <= 1.0:
        return float(data.max() * 255.0)
    return float(data.max())


def _check_info(root: Path) -> dict:
    """Validate meta/info.json schema."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")

    info = json.loads(info_path.read_text())
    features = info.get("features", {})

    if info.get("robot_type") != EXPECTED_ROBOT_TYPE:
        raise ValueError(f"robot_type must be {EXPECTED_ROBOT_TYPE!r}, got {info.get('robot_type')!r}")

    for key in ("observation.state", "action"):
        shape = features.get(key, {}).get("shape", [])
        if shape != [14]:
            raise ValueError(f"{key} shape must be [14], got {shape}")

    for cam_key in CAMERA_KEYS:
        video_key = f"observation.images.{cam_key}"
        if video_key not in features:
            raise ValueError(f"Missing video feature {video_key!r}")
        shape = features[video_key].get("shape", [])
        if shape != [480, 640, 3]:
            raise ValueError(f"{video_key} shape must be [480, 640, 3], got {shape}")

    print(
        f"[OK] info.json: robot_type={info['robot_type']}, "
        f"episodes={info.get('total_episodes')}, frames={info.get('total_frames')}, fps={info.get('fps')}"
    )
    return info


def _check_parquet_dir(label: str, directory: Path) -> None:
    """Ensure all parquet files under a directory have valid footers."""
    parquet_files = sorted(directory.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {directory} ({label})")

    for path in parquet_files:
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows
        print(f"[OK] {label} parquet {path.relative_to(directory.parent)}: {rows} rows")


def _decode_mp4_frame(path: Path, frame_index: int) -> tuple[int, int]:
    """Decode a single frame and return (min, max) pixel values."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_index:
            img = frame.to_ndarray(format="rgb24")
            return int(img.min()), int(img.max())
    raise IndexError(f"Frame {frame_index} not found in {path}")


def _check_videos(root: Path, info: dict) -> None:
    """Decode first and middle frames from each camera MP4."""
    video_path_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    total_frames = info.get("total_frames", 0)
    mid_frame = max(0, total_frames // 2)

    for cam_key in CAMERA_KEYS:
        video_key = f"observation.images.{cam_key}"
        rel_path = video_path_tpl.format(video_key=video_key, chunk_index=0, file_index=0)
        path = root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing video file {path}")

        for frame_idx in (0, mid_frame):
            pix_min, pix_max = _decode_mp4_frame(path, frame_idx)
            if pix_max <= MIN_PIXEL_MAX:
                raise ValueError(f"{path} frame {frame_idx} appears black (max={pix_max})")
            print(f"[OK] {cam_key} frame {frame_idx}: pixel range {pix_min}–{pix_max}")


def _check_lerobot_load(root: Path, repo_id: str | None) -> None:
    """Round-trip load frame 0 via LeRobotDataset."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("[SKIP] LeRobot not installed — skipping LeRobotDataset round-trip")
        return

    ds_repo = repo_id or root.name
    ds = LeRobotDataset(ds_repo, root=str(root))
    sample = ds[0]
    img = sample["observation.images.cam_high"]
    if hasattr(img, "numpy"):
        arr = img.numpy()
    else:
        import numpy as np

        arr = np.asarray(img)
    pix_max = _image_max(arr)
    if pix_max <= MIN_PIXEL_MAX:
        raise ValueError(f"LeRobotDataset frame 0 cam_high appears black (max={pix_max})")
    print(f"[OK] LeRobotDataset load: {len(ds)} frames, cam_high max={pix_max:.1f}")


def main() -> int:
    """Run all dataset integrity checks."""
    parser = argparse.ArgumentParser(description="Verify a recorded Mobile AI LeRobot dataset.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Local root directory of the dataset.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot repo id (defaults to the root directory name).",
    )
    parser.add_argument(
        "--skip-lerobot",
        action="store_true",
        help="Skip LeRobotDataset round-trip load.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[FAIL] Dataset root does not exist: {root}", file=sys.stderr)
        return 1

    try:
        info = _check_info(root)
        _check_parquet_dir("data", root / "data")
        _check_parquet_dir("episodes", root / "meta" / "episodes")
        _check_videos(root, info)
        if not args.skip_lerobot:
            _check_lerobot_load(root, args.repo_id)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
