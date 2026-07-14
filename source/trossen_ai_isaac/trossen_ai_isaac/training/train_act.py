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

"""Production ACT training helpers for Mobile AI sim datasets."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from trossen_ai_isaac.training.smoke_act import (
    _assert_checkpoint,
    _extract_last_loss,
    _find_checkpoint,
    _preflight_dataset,
    _resolve_device,
    _resolve_train_toolkit,
    _run_verify,
)


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
    wandb_enable: bool,
    push_to_hub: bool,
    job_name: str | None,
) -> list[str]:
    """Assemble a production ACT training command."""
    resolved_job = job_name or f"act_{root.name}"
    return [
        str(lerobot_train),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        f"--policy.device={device}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        f"--num_workers={num_workers}",
        "--save_checkpoint=true",
        f"--wandb.enable={'true' if wandb_enable else 'false'}",
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--output_dir={output_dir}",
        f"--job_name={resolved_job}",
    ]


def run_train_act(
    *,
    root: Path | str,
    repo_id: str,
    output_dir: Path | str,
    steps: int = 100_000,
    batch_size: int = 8,
    save_freq: int = 10_000,
    log_freq: int = 200,
    num_workers: int = 4,
    device: str | None = None,
    skip_verify: bool = False,
    keep_output: bool = False,
    verify_python: Path | str | None = None,
    train_env: Path | str | None = None,
    wandb_enable: bool = False,
    push_to_hub: bool = False,
    job_name: str | None = None,
) -> int:
    """Run dataset verification (optional) and a full ACT training job."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    train_env_root = Path(train_env).expanduser() if train_env else None
    lerobot_train, train_python, train_env_label = _resolve_train_toolkit(train_env_root)

    print(f"[OK] Training env: {train_env_label} ({lerobot_train})")
    _preflight_dataset(root)
    if not skip_verify:
        _run_verify(root, repo_id)

    device = _resolve_device(device, train_python)
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
        wandb_enable=wandb_enable,
        push_to_hub=push_to_hub,
        job_name=job_name,
    )
    print(f"[RUN] {' '.join(cmd)}")
    train_env_vars = os.environ.copy()
    if device == "cpu":
        train_env_vars["CUDA_VISIBLE_DEVICES"] = ""
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
    print(f"Training complete. Output: {output_dir}")
    print(f"Checkpoint: {pretrained_dir}")
    return 0
