"""XR camera compatibility probes for VR teleoperation sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trossen_ai_isaac.recording.frame_capture import capture_frame
from trossen_ai_isaac.recording.schema import CAMERA_KEYS


@dataclass
class CameraCompatProbe:
    """Collect camera-read compatibility evidence during a VR session."""

    task: str
    output_path: str | Path | None = None
    capture_frame_during_probe: bool = False
    successful_probes: int = 0
    failed_probes: int = 0
    successful_frame_captures: int = 0
    failed_frame_captures: int = 0
    last_success_step: int | None = None
    errors: list[str] = field(default_factory=list)
    camera_shapes: dict[str, list[int]] = field(default_factory=dict)
    frame_keys: list[str] = field(default_factory=list)

    def probe(self, env, step_count: int) -> None:
        """Read camera tensors (and optionally full frames) to test compatibility."""
        try:
            for cam_key in CAMERA_KEYS:
                rgb = env.scene[cam_key].data.output["rgb"][0]
                self.camera_shapes[cam_key] = list(rgb.shape)
            self.successful_probes += 1
            self.last_success_step = step_count
            print(
                "[VR CAMERA PROBE] success "
                f"step={step_count} cameras={list(self.camera_shapes)} "
                f"shapes={self.camera_shapes}"
            )
        except Exception as exc:
            self.failed_probes += 1
            self._remember_error(f"camera probe failed at step {step_count}: {exc}")
            print(f"[VR CAMERA PROBE] failure step={step_count}: {exc}")
            return

        if not self.capture_frame_during_probe:
            return

        try:
            frame = capture_frame(env, task=self.task)
            self.frame_keys = sorted(frame.keys())
            self.successful_frame_captures += 1
            print(
                "[VR CAMERA PROBE] frame capture success "
                f"step={step_count} keys={self.frame_keys}"
            )
        except Exception as exc:
            self.failed_frame_captures += 1
            self._remember_error(f"frame capture failed at step {step_count}: {exc}")
            print(f"[VR CAMERA PROBE] frame capture failure step={step_count}: {exc}")

    def finalize(self) -> None:
        """Write the optional JSON report for offline review."""
        if self.output_path is None:
            return
        report_path = Path(self.output_path).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"[VR CAMERA PROBE] Wrote compatibility report -> {report_path}")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable summary."""
        return {
            "task": self.task,
            "successful_probes": self.successful_probes,
            "failed_probes": self.failed_probes,
            "successful_frame_captures": self.successful_frame_captures,
            "failed_frame_captures": self.failed_frame_captures,
            "last_success_step": self.last_success_step,
            "camera_shapes": self.camera_shapes,
            "frame_keys": self.frame_keys,
            "errors": self.errors,
        }

    def _remember_error(self, message: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(message)
