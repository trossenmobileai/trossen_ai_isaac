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

"""Evaluate a trained LeRobot policy (ACT or Pi0) in the Mobile AI lift simulation.

Shared entry for ``run_play_act.sh`` and ``run_play_pi0.sh``. Default configuration
targets the 7D right-arm checkpoint from ``--record_arm right``. The policy
sidecar is launched automatically in the LeRobot Python environment and loads
the checkpoint type from config (``act``, ``pi0``, …).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Roll out an ACT or Pi0 policy in the Mobile AI lift sim (shared eval entry)."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0",
    help="Joint-position rollout task (env must have cameras enabled).",
)
parser.add_argument("--policy.path", dest="policy_path", type=str, required=True, help="Checkpoint directory.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--fps", type=int, default=60, help="Environment control rate.")
parser.add_argument(
    "--task_description",
    type=str,
    default="Pick up the cube, lift it, and place it back on the table",
    help="Language task prompt passed to the policy.",
)
parser.add_argument(
    "--sidecar-python",
    type=str,
    default="~/lerobot_trossen/.venv/bin/python",
    help="Python executable for policy inference (prefer lerobot_train conda env).",
)
parser.add_argument("--sidecar-host", type=str, default="127.0.0.1")
parser.add_argument("--sidecar-port", type=int, default=5555)
parser.add_argument("--output_dir", type=str, default=None, help="Optional directory for rollout_summary.json.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import trossen_ai_isaac.tasks  # noqa: F401, E402


def _load_run_act_rollout():
    try:
        from trossen_ai_isaac.evaluation.act_rollout import run_act_rollout

        return run_act_rollout
    except (ImportError, ModuleNotFoundError):
        repo_root = Path(__file__).resolve().parents[3]
        mod_path = repo_root / "source/trossen_ai_isaac/trossen_ai_isaac/evaluation/act_rollout.py"
        spec = importlib.util.spec_from_file_location("act_rollout", mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load act_rollout from {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_act_rollout


def main() -> int:
    try:
        run_act_rollout = _load_run_act_rollout()
        run_act_rollout(
            task=args_cli.task,
            policy_path=args_cli.policy_path,
            device=args_cli.device,
            num_episodes=args_cli.num_episodes,
            fps=args_cli.fps,
            task_description=args_cli.task_description,
            sidecar_python=args_cli.sidecar_python,
            sidecar_host=args_cli.sidecar_host,
            sidecar_port=args_cli.sidecar_port,
            output_dir=args_cli.output_dir,
            simulation_app=simulation_app,
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
