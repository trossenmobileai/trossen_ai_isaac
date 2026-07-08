"""Record Mobile AI VR teleoperation demonstrations to a LeRobot dataset."""

import argparse
import logging
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_scripts_dir = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
sys.path.insert(0, str(_scripts_dir / "lib"))
from teleop_cli_loader import load_teleop_cli
from vr_cli_loader import load_vr_cli

teleop_cli = load_teleop_cli()
vr_cli = load_vr_cli()

parser = argparse.ArgumentParser(
    description="Record Mobile AI VR hand-tracking demonstrations to a LeRobot dataset."
)
vr_cli.add_vr_teleop_args(parser)
vr_cli.add_vr_camera_args(parser)
parser.set_defaults(task="Isaac-Reach-MobileAI-Record-Play-v0", keep_cameras=True)
teleop_cli.add_record_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Force XR runtime ON before constructing AppLauncher.
app_launcher_args = vars(args_cli)
app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import trossen_ai_isaac.tasks  # noqa: F401
from trossen_ai_isaac.recording.lerobot_recorder import LeRobotRecorder
from trossen_ai_isaac.recording.runtime import install_recording_signal_handlers, run_recording_session
from trossen_ai_isaac.teleop.vr import build_vr_env_cfg, run_vr_recording_loop

logger = logging.getLogger(__name__)

install_recording_signal_handlers()


def main() -> None:
    """Run VR teleop and write LeRobot episodes to disk."""
    try:
        env_cfg = build_vr_env_cfg(args_cli)
    except ValueError as exc:
        logger.error("%s", exc)
        return
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error("Failed to create environment: %s", exc)
        return

    try:
        recorder = LeRobotRecorder(
            repo_id=args_cli.repo_id,
            fps=args_cli.fps,
            task=args_cli.task_description,
            root=args_cli.root,
            overwrite=args_cli.overwrite,
        )
    except (ImportError, FileExistsError) as exc:
        logger.error("%s", exc)
        env.close()
        return

    try:
        run_recording_session(
            simulation_app,
            env,
            env_cfg,
            args_cli,
            recorder,
            run_vr_recording_loop,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
