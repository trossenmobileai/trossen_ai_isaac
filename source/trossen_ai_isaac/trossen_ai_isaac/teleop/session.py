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

"""Shared teleoperation session flags."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeleopSession:
    """High-level teleop session state shared across device backends."""

    teleoperation_active: bool = True
    should_reset: bool = False
    should_save_episode: bool = False
    should_discard_episode: bool = False

    def request_reset(self) -> None:
        self.should_reset = True

    def request_save_episode(self) -> None:
        self.should_save_episode = True

    def request_discard_episode(self) -> None:
        self.should_discard_episode = True

    def start(self) -> None:
        self.teleoperation_active = True

    def stop(self) -> None:
        self.teleoperation_active = False

    def consume_reset(self) -> bool:
        if not self.should_reset:
            return False
        self.should_reset = False
        return True

    def consume_save_episode(self) -> bool:
        if not self.should_save_episode:
            return False
        self.should_save_episode = False
        return True

    def consume_discard_episode(self) -> bool:
        if not self.should_discard_episode:
            return False
        self.should_discard_episode = False
        return True
