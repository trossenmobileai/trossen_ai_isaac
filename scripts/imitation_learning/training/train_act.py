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

"""Full ACT training for a Mobile AI sim LeRobot dataset."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_run_train_act():
    try:
        from trossen_ai_isaac.training.train_act import run_train_act

        return run_train_act
    except (ImportError, ModuleNotFoundError):
        mod_path = _REPO_ROOT / "source/trossen_ai_isaac/trossen_ai_isaac/training/train_act.py"
        spec = importlib.util.spec_from_file_location("train_act", mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load training module from {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_train_act


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an ACT policy on a Mobile AI sim dataset.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot repo id.")
    parser.add_argument("--output_dir", type=str, required=True, help="Training output directory.")
    parser.add_argument("--steps", type=int, default=100_000, help="Training steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Checkpoint save frequency.")
    parser.add_argument("--log_freq", type=int, default=200, help="Logging frequency.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", type=str, default=None, help="Training device (cuda/cpu).")
    parser.add_argument("--skip-verify", action="store_true", help="Skip dataset verification.")
    parser.add_argument("--keep-output", action="store_true", help="Keep existing output directory.")
    parser.add_argument("--wandb.enable", dest="wandb_enable", action="store_true", help="Enable W&B.")
    parser.add_argument("--policy.push_to_hub", dest="push_to_hub", action="store_true", help="Upload policy.")
    parser.add_argument("--job_name", type=str, default=None, help="Training job name.")
    parser.add_argument("--train-env", type=str, default=None, help="Training env root.")
    args = parser.parse_args()

    try:
        run_train_act = _load_run_train_act()
        return run_train_act(
            root=args.root,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            num_workers=args.num_workers,
            device=args.device,
            skip_verify=args.skip_verify,
            keep_output=args.keep_output,
            train_env=args.train_env,
            wandb_enable=args.wandb_enable,
            push_to_hub=args.push_to_hub,
            job_name=args.job_name,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
