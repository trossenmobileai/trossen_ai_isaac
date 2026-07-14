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

"""Verify a Mobile AI LeRobot dataset matches the expected schema.

Accepts any arm mode recorded by this project (7D right, 7D left, 14D both)
and validates structure, parquet footers, and video frames.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_verify_dataset():
    try:
        from trossen_ai_isaac.validation.lerobot_dataset import verify_dataset

        return verify_dataset
    except (ImportError, ModuleNotFoundError):
        mod_path = _REPO_ROOT / "source/trossen_ai_isaac/trossen_ai_isaac/validation/lerobot_dataset.py"
        spec = importlib.util.spec_from_file_location("lerobot_dataset", mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load validation module from {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.verify_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a Mobile AI LeRobot dataset (7D or 14D arm mode)."
    )
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot repo id.")
    parser.add_argument("--skip-lerobot", action="store_true", help="Skip LeRobotDataset round-trip.")
    args = parser.parse_args()

    try:
        verify_dataset = _load_verify_dataset()
        verify_dataset(args.root, args.repo_id, skip_lerobot=args.skip_lerobot)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print("[OK] Dataset compatibility checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
