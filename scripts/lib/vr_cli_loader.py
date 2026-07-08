"""Load ``teleop/vr/cli.py`` before Isaac ``AppLauncher``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_PATH = _REPO_ROOT / "source/trossen_ai_isaac/trossen_ai_isaac/teleop/vr/cli.py"


def load_vr_cli() -> ModuleType:
    """Return the VR teleop CLI helper module (argparse only; safe pre-AppLauncher)."""
    spec = importlib.util.spec_from_file_location("trossen_ai_isaac_vr_cli", _CLI_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load VR CLI module from {_CLI_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
