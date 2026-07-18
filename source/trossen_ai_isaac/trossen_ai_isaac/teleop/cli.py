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

"""Shared argparse helpers for Mobile AI teleop and recording CLIs."""

from __future__ import annotations

import argparse


def add_mobile_ai_teleop_args(parser: argparse.ArgumentParser) -> None:
    """Add common Mobile AI teleop flags (device, task, sensitivity, gamepad)."""
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        help="Teleop device: keyboard or gamepad.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Reach-MobileAI-IK-Abs-Play-v0",
        help="Gym task id.",
    )
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity scale factor.")
    parser.add_argument(
        "--gamepad_dead_zone",
        type=float,
        default=0.15,
        help="Per-axis dead zone for gamepad stick drift (default 0.15).",
    )
    parser.add_argument(
        "--step_log",
        action="store_true",
        help=(
            "Print a status line every 60 sim steps ([step=...] with arm / grip / pose). "
            "Off by default so key-event logs stay readable."
        ),
    )


def add_record_args(parser: argparse.ArgumentParser) -> None:
    """Add LeRobot dataset recording flags."""
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot dataset repo id, e.g. YourUser/trossen_ai_sim_reach.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional local root directory for the dataset (defaults to HF cache).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Dataset FPS; also sets env decimation (60 -> decimation 1, 30 -> decimation 2).",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="mobile_ai_reach",
        help="Natural-language task label stored in each frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset at --root instead of failing.",
    )
    parser.add_argument(
        "--record_arm",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help=(
            "Which arm(s) to record into the LeRobot dataset. "
            "'both' (default) => 14D observation.state/action + 3 cameras "
            "(cam_high, cam_left_wrist, cam_right_wrist). "
            "'left'/'right' => 7D that-arm joints (6 arm + gripper) + cam_high + "
            "that arm's wrist camera. All modes produce a standard LeRobot v3 dataset."
        ),
    )
