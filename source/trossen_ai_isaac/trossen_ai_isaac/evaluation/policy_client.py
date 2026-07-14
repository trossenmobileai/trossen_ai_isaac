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

"""TCP client for the external ACT policy sidecar process."""

from __future__ import annotations

import pickle
import socket
import struct
from pathlib import Path
from typing import Any

import numpy as np


def _send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def _recv_msg(sock: socket.socket) -> dict[str, Any]:
    header = _recvall(sock, 4)
    if not header:
        raise ConnectionError("Policy sidecar closed the connection")
    (length,) = struct.unpack("!I", header)
    data = _recvall(sock, length)
    return pickle.loads(data)


def _recvall(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < nbytes:
        chunk = sock.recv(nbytes - received)
        if not chunk:
            raise ConnectionError("Policy sidecar closed the connection")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class PolicySidecarClient:
    """Exchange observations and actions with a ``policy_sidecar.py`` server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_s: float = 120.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                _send_msg(self._sock, {"cmd": "shutdown"})
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def reset(self) -> None:
        self._request({"cmd": "reset"})

    def infer(self, observation: dict[str, np.ndarray], task: str) -> np.ndarray:
        """Return a single 16D action vector for the current observation."""
        response = self._request(
            {
                "cmd": "infer",
                "task": task,
                "observation": observation,
            }
        )
        action = np.asarray(response["action"], dtype=np.float32)
        return action

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("PolicySidecarClient is not connected")
        _send_msg(self._sock, payload)
        response = _recv_msg(self._sock)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Policy sidecar request failed"))
        return response


def build_sidecar_command(
    policy_path: Path | str,
    *,
    python_exe: Path | str,
    host: str = "127.0.0.1",
    port: int = 5555,
    device: str = "cuda",
) -> list[str]:
    """Return argv for launching the policy sidecar subprocess."""
    sidecar = Path(__file__).resolve().parent / "policy_sidecar.py"
    return [
        str(python_exe),
        str(sidecar),
        f"--policy.path={Path(policy_path).expanduser()}",
        f"--host={host}",
        f"--port={port}",
        f"--device={device}",
    ]
