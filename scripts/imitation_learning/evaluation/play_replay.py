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

"""Open-loop replay of a recorded episode in the joint-position lift environment.

Supports 7D right-arm datasets (default from single-arm VR recording) as well
as 14D dual-arm datasets.  The dataset dimension is detected automatically from
``meta/info.json``; the left arm is kept at its current joint position when
replaying a 7D right-arm episode.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a LeRobot episode in the Mobile AI lift sim.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0")
parser.add_argument("--root", type=str, required=True, help="Dataset root directory.")
parser.add_argument("--repo_id", type=str, default=None, help="LeRobot repo id.")
parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
parser.add_argument("--fps", type=int, default=30, help="Environment control rate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import trossen_ai_isaac.tasks  # noqa: F401, E402
from trossen_ai_isaac.tasks.manager_based.manipulation.mobile_ai.reach.mdp.observations import (  # noqa: E402
    record_joint_pos_14,
)
from trossen_ai_isaac.teleop.mobile_ai_ik_abs import make_env_cfg  # noqa: E402


def _load_episode_from_parquet(root: str, episode: int) -> list[np.ndarray]:
    """Load ordered action vectors for one episode directly from parquet files."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not installed in the Isaac Sim Python env and PyArrow is "
            "unavailable for parquet fallback."
        ) from exc

    frames: list[tuple[int, np.ndarray]] = []
    data_dir = Path(root) / "data"
    for path in sorted(data_dir.rglob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["action", "episode_index", "frame_index"]):
            for epi, frame_idx, action in zip(
                batch.column("episode_index").to_pylist(),
                batch.column("frame_index").to_pylist(),
                batch.column("action").to_pylist(),
            ):
                if epi == episode:
                    frames.append((frame_idx, np.asarray(action, dtype=np.float32)))

    if not frames:
        raise ValueError(f"Episode {episode} not found in dataset")

    frames.sort(key=lambda item: item[0])
    return [action for _, action in frames]


def _load_episode_actions(root: str, repo_id: str | None, episode: int) -> list[np.ndarray]:
    """Load action sequence for one episode (LeRobot API with parquet fallback)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        return _load_episode_from_parquet(root, episode)

    ds_repo = repo_id or Path(root).name
    dataset = LeRobotDataset(ds_repo, root=root)
    episode_data = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    if len(episode_data) == 0:
        raise ValueError(f"Episode {episode} not found in dataset")
    return [np.asarray(row["action"], dtype=np.float32) for row in episode_data]


def _detect_action_dim(root: str) -> int:
    """Read action shape from meta/info.json."""
    import json

    info_path = Path(root) / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        shape = info.get("features", {}).get("action", {}).get("shape", [14])
        return shape[0] if shape else 14
    return 14


def _action_to_env(action: np.ndarray, env, action_dim: int) -> torch.Tensor:
    """Map a recorded action to a full 14D env action tensor.

    For 7D right-arm datasets: pad with current left-arm joint positions
    so the left arm holds still during replay.
    For 14D dual-arm datasets: pass through directly.
    """
    if action_dim == 14:
        full = action[:14].astype(np.float32)
    else:
        current_14d = record_joint_pos_14(env)[0].detach().cpu().numpy().astype(np.float32)
        full = current_14d.copy()
        full[7:14] = action[:7]
    return torch.as_tensor(full, device=env.device, dtype=torch.float32).unsqueeze(0)


def main() -> int:
    root = args_cli.root
    action_dim = _detect_action_dim(root)
    print(f"[INFO] Detected action_dim={action_dim} from dataset info.json")

    try:
        actions = _load_episode_actions(root, args_cli.repo_id, args_cli.episode)

        env_cfg = make_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, fps=args_cli.fps)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()

        for action in actions:
            env_action = _action_to_env(action, env, action_dim)
            env.step(env_action)
            simulation_app.update()

        env.close()
        print(f"[OK] Replayed episode {args_cli.episode} ({len(actions)} steps)")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
