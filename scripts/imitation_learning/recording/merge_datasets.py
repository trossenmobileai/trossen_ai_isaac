"""Merge multiple LeRobot dataset shards into one unified dataset.

Each shard is a complete, finalized LeRobot v3 dataset produced by a single
recording session.  This script uses LeRobot's built-in ``aggregate_datasets``
function to combine them, which validates that all shards share the same fps,
robot_type, and feature schema before merging.

Usage (no Isaac dependency -- run under the LeRobot venv):

    ~/lerobot_trossen/.venv/bin/python \\
        scripts/imitation_learning/recording/merge_datasets.py \\
        --shards_dir ~/lerobot_trossen/datasets/my_dataset/shards \\
        --repo_id trossen-admin/my_dataset \\
        --root ~/lerobot_trossen/datasets/my_dataset/merged

Or, point at specific shard roots:

    ~/lerobot_trossen/.venv/bin/python \\
        scripts/imitation_learning/recording/merge_datasets.py \\
        --inputs ~/lerobot_trossen/datasets/my_dataset/shards/session_1 \\
                 ~/lerobot_trossen/datasets/my_dataset/shards/session_2 \\
        --repo_id trossen-admin/my_dataset \\
        --root ~/lerobot_trossen/datasets/my_dataset/merged

All shards must have been collected with the same --record_arm mode and --fps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _read_repo_id(shard_root: Path, fallback: str) -> str:
    info = shard_root / "meta" / "info.json"
    if info.exists():
        try:
            data = json.loads(info.read_text())
            # LeRobot v3 info.json may not store repo_id; derive from folder name.
            return data.get("repo_id") or fallback
        except Exception:
            pass
    return fallback


def _episode_count(shard_root: Path) -> int:
    info = shard_root / "meta" / "info.json"
    if not info.exists():
        return 0
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0))
    except Exception:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot dataset shards into a single dataset."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        metavar="SHARD_ROOT",
        help="Explicit list of shard root directories to merge.",
    )
    source_group.add_argument(
        "--shards_dir",
        type=str,
        metavar="PARENT_DIR",
        help=(
            "Parent directory whose immediate subdirectories are treated as shards "
            "(sorted alphabetically).  Equivalent to passing all subdirs to --inputs."
        ),
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot repo id for the merged output dataset (e.g. 'trossen-admin/my_dataset').",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Output root directory for the merged dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output root directory if it already exists.",
    )

    args = parser.parse_args()

    # Resolve shard paths
    if args.shards_dir is not None:
        parent = Path(args.shards_dir).expanduser().resolve()
        if not parent.is_dir():
            print(f"[ERROR] --shards_dir {parent} does not exist or is not a directory.", file=sys.stderr)
            return 1
        shard_roots = sorted(p for p in parent.iterdir() if p.is_dir())
        if not shard_roots:
            print(f"[ERROR] No subdirectories found in {parent}.", file=sys.stderr)
            return 1
    else:
        shard_roots = [Path(p).expanduser().resolve() for p in args.inputs]

    for sr in shard_roots:
        if not sr.is_dir():
            print(f"[ERROR] Shard path does not exist or is not a directory: {sr}", file=sys.stderr)
            return 1

    # Drop empty shards (0 episodes) so aggregate_datasets never hits HF fallback.
    skipped = [sr for sr in shard_roots if _episode_count(sr) == 0]
    shard_roots = [sr for sr in shard_roots if _episode_count(sr) > 0]
    for sr in skipped:
        print(f"[MERGE] Skipping empty shard (0 episodes): {sr.name}")
    if not shard_roots:
        print("[ERROR] No non-empty shards to merge.", file=sys.stderr)
        return 1

    output_root = Path(args.root).expanduser().resolve()

    # Guard output
    if output_root.exists():
        if args.overwrite:
            print(f"[MERGE] Removing existing output at {output_root}")
            shutil.rmtree(output_root)
        else:
            print(
                f"[ERROR] Output root already exists: {output_root}\n"
                "       Pass --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1

    # Build repo_id / root lists for aggregate_datasets.
    # Use shard folder name as repo_id fallback (unique per session).
    repo_ids = [_read_repo_id(sr, sr.name) for sr in shard_roots]
    roots = shard_roots

    print("=" * 60)
    print("  LeRobot dataset merge")
    print(f"  Shards   : {len(shard_roots)}")
    for i, (rid, sr) in enumerate(zip(repo_ids, roots)):
        info_path = sr / "meta" / "info.json"
        ep_count = "?"
        if info_path.exists():
            try:
                ep_count = json.loads(info_path.read_text()).get("total_episodes", "?")
            except Exception:
                pass
        print(f"    [{i}] {sr.name}  repo_id={rid}  episodes={ep_count}")
    print(f"  Output   : {output_root}")
    print(f"  repo_id  : {args.repo_id}")
    print("=" * 60)
    print()

    try:
        from lerobot.datasets.aggregate import aggregate_datasets
    except ImportError as exc:
        print(
            f"[ERROR] Could not import lerobot.datasets.aggregate: {exc}\n"
            "        Run this script with the LeRobot venv:\n"
            "        ~/lerobot_trossen/.venv/bin/python <this_script>",
            file=sys.stderr,
        )
        return 1

    print("[MERGE] Starting aggregation...")
    try:
        aggregate_datasets(
            repo_ids=[str(r) for r in repo_ids],
            aggr_repo_id=args.repo_id,
            roots=[Path(r) for r in roots],
            aggr_root=output_root,
        )
    except Exception as exc:
        print(f"[ERROR] aggregate_datasets failed: {exc}", file=sys.stderr)
        return 1

    # Print a summary from the merged info.json
    merged_info = output_root / "meta" / "info.json"
    if merged_info.exists():
        try:
            info = json.loads(merged_info.read_text())
            print()
            print("[MERGE] Done.")
            print(f"  Total episodes : {info.get('total_episodes')}")
            print(f"  Total frames   : {info.get('total_frames')}")
            print(f"  FPS            : {info.get('fps')}")
            print(f"  robot_type     : {info.get('robot_type')}")
            print(f"  Output         : {output_root}")
        except Exception:
            print(f"[MERGE] Done. Output at {output_root}")
    else:
        print(f"[MERGE] Done. Output at {output_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
