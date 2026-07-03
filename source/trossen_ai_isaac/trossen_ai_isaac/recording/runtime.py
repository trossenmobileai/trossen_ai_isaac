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

"""Recording session lifecycle helpers (signal handlers, teleop wrapper)."""

from __future__ import annotations

import signal
from collections.abc import Callable

from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder
from trossen_ai_isaac.teleop.session import clear_shutdown, request_shutdown

_recorder_for_signal: LeRobotRecorder | None = None
_signal_finalized: bool = False


def _finalize_on_signal(signum, frame) -> None:
    """Finalize the dataset and request a graceful teleop loop exit."""
    global _signal_finalized
    del signum, frame
    request_shutdown()
    if _recorder_for_signal is not None and not _signal_finalized:
        print("\n[RECORD] Caught interrupt — finalizing dataset...")
        _recorder_for_signal.finalize()
        _signal_finalized = True


def install_recording_signal_handlers() -> None:
    """Register SIGINT/SIGTERM handlers that finalize an active recorder."""
    signal.signal(signal.SIGINT, _finalize_on_signal)
    signal.signal(signal.SIGTERM, _finalize_on_signal)


def run_recording_session(
    simulation_app,
    env,
    env_cfg,
    args_cli,
    recorder: LeRobotRecorder,
    teleop_loop_fn: Callable[..., None],
) -> None:
    """Run a teleop loop with LeRobot recording and graceful interrupt handling."""
    global _recorder_for_signal, _signal_finalized

    clear_shutdown()
    _signal_finalized = False
    _recorder_for_signal = recorder
    try:
        teleop_loop_fn(simulation_app, env, env_cfg, args_cli, recorder=recorder)
    finally:
        if not _signal_finalized:
            recorder.finalize()
        _recorder_for_signal = None


def signal_finalized() -> bool:
    """Return whether the recorder was already finalized by a signal handler."""
    return _signal_finalized
