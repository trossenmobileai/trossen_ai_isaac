#!/usr/bin/env bash
# ============================================================================
# Phase 4 — Collect right-arm pick/lift/place episodes via VR.
#
# Multi-session workflow
# ----------------------
# Each invocation records into its own shard subdirectory under
# $ROOT_BASE/shards/session_<TIMESTAMP> (or session_<LABEL> if you pass a
# label as the first argument).  After collecting all desired episodes across
# one or more sessions, run:
#
#   ./scripts/imitation_learning/run_merge_dataset.sh [--verify]
#
# to merge all shards into one valid LeRobot v3 dataset under
# $ROOT_BASE/merged.  Shards that share the same --record_arm mode and --fps
# can be merged; the merge step raises an error if they differ.
#
# Pre-conditions (must be done before running this script):
#   1. VR stack is running:
#      - Open ALVR Launcher → click "Launch"
#      - In ALVR dashboard → click "Launch SteamVR"
#      - On Quest 3 → open ALVR app → headset shows SteamVR home
#   2. Ensure CAP_SYS_NICE is set on vrcompositor-launcher (one-time):
#      sudo setcap CAP_SYS_NICE+eip \
#        ~/.steam/debian-installation/steamapps/common/SteamVR/bin/linux64/vrcompositor-launcher
#
# During the recording session:
#   - Isaac Sim opens and shows the table + cube scene.
#   - In Isaac Sim: set Output Plugin = OpenXR → click "Start AR".
#   - Right hand controls the right arm (left arm is frozen/locked).
#   - Workstation keyboard controls:
#       U  — engage teleop (right hand starts tracking the arm)
#       N  — start recording current episode; press N again to save + reset
#       M  — discard current episode buffer (bad attempt, does not count)
#       B  — re-anchor (re-snapshot hand pose → EE pose mapping)
#       J  — reset environment and discard in-progress episode
#   - Aim for one natural pick/lift/place demo per episode.
#   - Cube XY position and colour are randomised on every reset.
#   - Each saved episode prints: [RECORD] Saved episode (NNN frames) -> ...
#   - When you reach desired saved episodes, press Ctrl+C and wait for
#     "[RECORD] Finalized dataset at ..." before closing the terminal.
# ============================================================================

set -euo pipefail

# ---- Shared dataset identity ----
REPO_BASE="trossen-admin/mobile_ai_right_pick_place_20260714_v2"
ROOT_BASE="$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2"
TASK_DESC="Pick up the cube, lift it, and place it back on the table"
FPS=60

# ---- Per-session shard ----
# Pass an optional label as $1 to name the shard (e.g. "morning", "session3").
# Defaults to a YYYYMMDD_HHMMSS timestamp so every run is distinct.
SESSION="${1:-$(date +%Y%m%d_%H%M%S)}"
ROOT="$ROOT_BASE/shards/session_${SESSION}"
# Each shard gets its own repo_id so aggregate_datasets can distinguish sources.
REPO_ID="${REPO_BASE}_s${SESSION}"

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

cd "$REPO_DIR"

echo "============================================================"
echo "  Mobile AI right-arm pick/lift/place — Episode recording"
echo "  Session    : $SESSION"
echo "  Shard dir  : $ROOT"
echo "  FPS        : $FPS"
echo "  episode_length_s = 30 s per episode (auto-reset guard)"
echo "  Controls: U=engage  N=start/save  M=discard  B=reanchor  J=reset"
echo ""
echo "  After all sessions: run run_merge_dataset.sh to combine shards."
echo "============================================================"
echo ""

~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm_vr.py \
  --repo_id "$REPO_ID" \
  --root "$ROOT" \
  --record_arm right \
  --view first_person \
  --fps "$FPS" \
  --task_description "$TASK_DESC" \
  --overwrite
