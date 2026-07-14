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

"""Portable IPC encoding for Isaac Sim ↔ policy sidecar messages.

Isaac Sim (Python 3.11) and the LeRobot sidecar (Python 3.12) often ship
different NumPy builds. Pickling ``ndarray`` objects across that boundary
fails with ``numpy._core`` import errors, so observations and actions are
sent as raw bytes + dtype/shape metadata instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_NDARRAY_TAG = "__ndarray__"


def _pack_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_TAG: True,
            "data": value.tobytes(),
            "dtype": str(value.dtype),
            "shape": value.shape,
        }
    return value


def _unpack_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get(_NDARRAY_TAG):
        arr = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return arr.reshape(value["shape"]).copy()
    return value


def pack_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a pickle-safe copy of ``payload`` with ndarrays encoded as bytes."""
    out = dict(payload)
    observation = out.get("observation")
    if isinstance(observation, dict):
        out["observation"] = {key: _pack_value(val) for key, val in observation.items()}
    action = out.get("action")
    if isinstance(action, np.ndarray):
        out["action"] = _pack_value(action)
    return out


def unpack_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore ndarrays from a message produced by :func:`pack_message`."""
    out = dict(payload)
    observation = out.get("observation")
    if isinstance(observation, dict):
        out["observation"] = {key: _unpack_value(val) for key, val in observation.items()}
    action = out.get("action")
    if action is not None:
        out["action"] = _unpack_value(action)
    return out
