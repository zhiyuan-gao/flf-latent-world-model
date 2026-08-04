#!/usr/bin/env python3
"""Create the fixed, shared episode split and window manifest for the bake-off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.manifest import DEFAULT_TASKS, build_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT / "data/robocasa365/v1.0/target/composite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/dynamics_bakeoff",
    )
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--camera", default="robot0_agentview_left")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        data_root=args.data_root,
        output_dir=args.output_dir,
        tasks=args.tasks,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        window_stride=args.window_stride,
        camera=args.camera,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "split_window_counts": manifest["split_window_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
