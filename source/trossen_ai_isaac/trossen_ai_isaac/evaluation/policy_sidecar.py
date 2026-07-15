#!/usr/bin/env python3
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

"""Run ACT inference in a LeRobot-capable Python environment (sidecar server)."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from policy_transport import pack_message, unpack_message


def _send_msg(conn: socket.socket, payload: dict[str, Any]) -> None:
    try:
        data = pickle.dumps(pack_message(payload), protocol=pickle.HIGHEST_PROTOCOL)
        conn.sendall(struct.pack("!I", len(data)))
        conn.sendall(data)
    except (BrokenPipeError, ConnectionResetError) as exc:
        raise ConnectionError("Client disconnected") from exc


def _recv_msg(conn: socket.socket) -> dict[str, Any]:
    header = _recvall(conn, 4)
    (length,) = struct.unpack("!I", header)
    data = _recvall(conn, length)
    return unpack_message(pickle.loads(data))


def _recvall(conn: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < nbytes:
        chunk = conn.recv(nbytes - received)
        if not chunk:
            raise ConnectionError("Client disconnected")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
    """Convert sim capture (HWC uint8) to LeRobot training layout (CHW float32 in [0, 1])."""
    arr = np.asarray(image)
    if arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    else:
        tensor = torch.from_numpy(arr).contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    else:
        tensor = tensor.float()
    return tensor


def _observation_to_batch(observation: dict[str, np.ndarray], task: str) -> dict[str, Any]:
    """Build a LeRobot batch dict from raw sim observations."""
    batch: dict[str, Any] = {"task": [task]}
    for key, value in observation.items():
        if key.startswith("observation.images."):
            batch[key] = _image_to_chw_float(value)
        else:
            batch[key] = torch.as_tensor(value, dtype=torch.float32)
    return batch


def _load_act_policy(policy_path: Path):
    """Load an ACT checkpoint via the concrete ``ACTPolicy`` class."""
    errors: list[str] = []
    for module_name in (
        "lerobot.policies.act.modeling_act",
        "lerobot.common.policies.act.modeling_act",
    ):
        try:
            module = __import__(module_name, fromlist=["ACTPolicy"])
            return module.ACTPolicy.from_pretrained(str(policy_path))
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise RuntimeError(
        f"Failed to load ACT policy from {policy_path}. Tried:\n  " + "\n  ".join(errors)
    )


class ACTSidecar:
    def __init__(self, policy_path: Path, device: str) -> None:
        from lerobot.policies import make_pre_post_processors

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.policy_path = policy_path
        self.policy = _load_act_policy(policy_path)
        self.policy.to(self.device)
        self.policy.eval()

        device_override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(policy_path),
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def infer(self, observation: dict[str, np.ndarray], task: str) -> np.ndarray:
        batch = _observation_to_batch(observation, task)
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="ACT policy inference sidecar server.")
    parser.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        print(f"[FAIL] Policy path does not exist: {policy_path}", file=sys.stderr)
        return 1

    sidecar = ACTSidecar(policy_path, args.device)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[OK] ACT sidecar listening on {args.host}:{args.port}", flush=True)

    conn, _addr = server.accept()
    try:
        while True:
            request = _recv_msg(conn)
            cmd = request.get("cmd")
            if cmd == "shutdown":
                try:
                    _send_msg(conn, {"ok": True})
                except ConnectionError:
                    pass
                break
            if cmd == "reset":
                sidecar.reset()
                _send_msg(conn, {"ok": True})
                continue
            if cmd == "infer":
                action = sidecar.infer(request["observation"], request.get("task", ""))
                _send_msg(conn, {"ok": True, "action": action})
                continue
            _send_msg(conn, {"ok": False, "error": f"Unknown command: {cmd!r}"})
    except ConnectionError:
        pass
    finally:
        conn.close()
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
