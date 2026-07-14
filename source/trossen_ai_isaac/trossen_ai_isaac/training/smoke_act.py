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

"""ACT training smoke test helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path("~/lerobot_trossen/datasets/trossen_ai_integration_v1")
DEFAULT_REPO_ID = "trossen-admin/trossen_ai_integration_v1"
DEFAULT_VERIFY_VENV = Path("~/lerobot_trossen/.venv")
DEFAULT_TRAIN_CONDA = Path("~/miniconda3/envs/lerobot_train")


def _resolve_verify_python(explicit: Path | None = None) -> Path:
    """Return Python for dataset verification (lerobot_trossen venv with PyAV)."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Verify Python not found: {path}")

    default = DEFAULT_VERIFY_VENV.expanduser() / "bin" / "python"
    if default.is_file():
        return default

    raise FileNotFoundError(
        "Could not find verify Python. Install lerobot_trossen or pass --verify-python, e.g.\n"
        "  ~/lerobot_trossen/.venv/bin/python"
    )


def _resolve_train_toolkit(explicit_train_env: Path | None = None) -> tuple[Path, Path, str]:
    """Return (lerobot-train, python, env_label) for the training subprocess.

    Prefers the ``lerobot_train`` conda env (torch 2.11+, RTX 5090 compatible) over
    ``lerobot_trossen/.venv`` (recording stack; older PyTorch).
    """
    candidates: list[tuple[Path, str]] = []
    if explicit_train_env is not None:
        candidates.append((explicit_train_env.expanduser(), "custom train env"))
    candidates.append((DEFAULT_TRAIN_CONDA.expanduser(), "lerobot_train conda"))
    candidates.append((DEFAULT_VERIFY_VENV.expanduser(), "lerobot_trossen venv"))

    for env_root, label in candidates:
        bin_dir = env_root / "bin"
        lerobot_train = bin_dir / "lerobot-train"
        python_exe = bin_dir / "python"
        if lerobot_train.is_file() and python_exe.is_file():
            return lerobot_train, python_exe, label

    raise FileNotFoundError(
        "Could not find lerobot-train. Activate lerobot_train or pass --train-env, e.g.\n"
        "  conda activate lerobot_train\n"
        "  # or: --train-env ~/miniconda3/envs/lerobot_train"
    )


def _preflight_dataset(root: Path) -> dict:
    """Ensure required dataset metadata and data files exist."""
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {stats_path}")

    info = json.loads(info_path.read_text())
    parquet_files = list((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    video_path_tpl = info.get(
        "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    for feature_key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        rel_path = video_path_tpl.format(video_key=feature_key, chunk_index=0, file_index=0)
        video_path = root / rel_path
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video file {video_path}")

    print(
        f"[OK] Dataset preflight: episodes={info.get('total_episodes')}, "
        f"frames={info.get('total_frames')}, robot_type={info.get('robot_type')}"
    )
    return info


def _load_verify_dataset():
    """Load verify_dataset without importing the trossen_ai_isaac package root."""
    import importlib.util

    try:
        from trossen_ai_isaac.validation.lerobot_dataset import verify_dataset

        return verify_dataset
    except (ImportError, ModuleNotFoundError):
        mod_path = Path(__file__).resolve().parent.parent / "validation" / "lerobot_dataset.py"
        spec = importlib.util.spec_from_file_location("lerobot_dataset", mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load validation module from {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.verify_dataset


def _run_verify(root: Path, repo_id: str) -> None:
    """Run dataset validation without requiring the full extension in the train env."""
    verify_dataset = _load_verify_dataset()
    print(f"[RUN] verify_dataset(root={root}, repo_id={repo_id})")
    verify_dataset(root, repo_id)


def _cuda_supported(python_exe: Path) -> bool:
    """Return True when CUDA tensors can execute in the given Python environment."""
    probe = (
        "import torch\n"
        "assert torch.cuda.is_available()\n"
        "(torch.zeros(1, device='cuda') + 1)\n"
    )
    result = subprocess.run(
        [str(python_exe), "-c", probe],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _resolve_device(requested: str | None, train_python: Path) -> str:
    """Pick cuda when it works in the training env unless the user overrides."""
    if requested is not None:
        return requested
    if _cuda_supported(train_python):
        return "cuda"
    print(
        "[WARN] CUDA unavailable or incompatible in the training environment — "
        "using CPU for the smoke test"
    )
    return "cpu"


def _build_train_command(
    lerobot_train: Path,
    root: Path,
    repo_id: str,
    output_dir: Path,
    steps: int,
    batch_size: int,
    save_freq: int,
    log_freq: int,
    device: str,
    num_workers: int,
) -> list[str]:
    """Assemble a minimal ACT training command."""
    return [
        str(lerobot_train),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.dim_model=256",
        "--policy.n_encoder_layers=2",
        "--policy.n_decoder_layers=2",
        f"--policy.device={device}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        f"--num_workers={num_workers}",
        "--save_checkpoint=true",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name=act_smoke_{root.name}",
    ]


def _find_checkpoint(output_dir: Path) -> Path:
    """Locate a saved pretrained_model directory under output_dir/checkpoints."""
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints directory at {checkpoints_dir}")

    candidates: list[Path] = []
    for step_dir in sorted(checkpoints_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        pretrained = step_dir / "pretrained_model"
        if pretrained.is_dir():
            candidates.append(pretrained)

    if not candidates:
        raise FileNotFoundError(f"No pretrained_model directories under {checkpoints_dir}")

    return candidates[-1]


def _assert_checkpoint(pretrained_dir: Path) -> None:
    """Ensure the checkpoint contains model weights."""
    model_path = pretrained_dir / "model.safetensors"
    config_path = pretrained_dir / "config.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model weights at {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing policy config at {config_path}")
    print(f"[OK] Checkpoint: {pretrained_dir} ({model_path.stat().st_size} bytes model.safetensors)")


def _extract_last_loss(train_output: str) -> float | None:
    """Best-effort parse of the final logged training loss."""
    matches = re.findall(r"loss[:\s]+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", train_output, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None




def run_smoke_act(
    *,
    root: Path | str,
    repo_id: str,
    output_dir: Path | str | None = None,
    steps: int = 100,
    batch_size: int = 2,
    save_freq: int = 50,
    log_freq: int = 10,
    num_workers: int = 0,
    device: str | None = None,
    skip_verify: bool = False,
    keep_output: bool = False,
    verify_python: Path | str | None = None,
    train_env: Path | str | None = None,
) -> int:
    """Run dataset verification (optional) and a short ACT training smoke test."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(f"/tmp/act_sim_smoke_{root.name}")
    )

    train_env_root = Path(train_env).expanduser() if train_env else None
    lerobot_train, train_python, train_env_label = _resolve_train_toolkit(train_env_root)

    print(f"[OK] Training env: {train_env_label} ({lerobot_train})")
    _preflight_dataset(root)
    if not skip_verify:
        _run_verify(root, repo_id)

    device = _resolve_device(device, train_python)
    if device == "cuda":
        print(f"[OK] CUDA available in training env ({train_python})")
    if output_dir.exists() and not keep_output:
        shutil.rmtree(output_dir)

    cmd = _build_train_command(
        lerobot_train=lerobot_train,
        root=root,
        repo_id=repo_id,
        output_dir=output_dir,
        steps=steps,
        batch_size=batch_size,
        save_freq=save_freq,
        log_freq=log_freq,
        device=device,
        num_workers=num_workers,
    )
    print(f"[RUN] {' '.join(cmd)}")
    train_env_vars = os.environ.copy()
    if device == "cpu":
        train_env_vars["CUDA_VISIBLE_DEVICES"] = ""
    # Ensure HF datasets cache is always user-writable regardless of system state.
    # Falls back to a workspace-local path if HF_DATASETS_CACHE is not already set
    # (the system-default ~/.cache/huggingface/datasets/ may be root-owned).
    if "HF_DATASETS_CACHE" not in train_env_vars:
        user_hf_cache = Path(__file__).resolve().parents[4] / ".hf_datasets_cache"
        user_hf_cache.mkdir(parents=True, exist_ok=True)
        train_env_vars["HF_DATASETS_CACHE"] = str(user_hf_cache)
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, env=train_env_vars)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"lerobot-train failed with exit code {result.returncode}")

    combined_output = f"{result.stdout}\n{result.stderr}"
    last_loss = _extract_last_loss(combined_output)
    if last_loss is not None:
        print(f"[OK] Last logged loss: {last_loss}")

    pretrained_dir = _find_checkpoint(output_dir)
    _assert_checkpoint(pretrained_dir)
    print(f"Training smoke test passed. Output: {output_dir}")
    return 0
