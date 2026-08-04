#!/usr/bin/env python3
"""Build episode-isolated offline windows for the rolling latent predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import build_rolling_manifest  # noqa: E402
from dynamics.manifest import DEFAULT_TASKS  # noqa: E402


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
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--camera", default="robot0_agentview_left")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help=(
            "Keep validation/test episodes fixed and extend the reference train "
            "split to --train-episodes."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_splits = None
    if args.reference_manifest is not None:
        reference = json.loads(args.reference_manifest.read_text())
        missing = set(args.tasks) - set(reference["tasks"])
        if missing:
            raise ValueError(f"Tasks absent from reference manifest: {sorted(missing)}")
        reference_splits = {
            task: reference["tasks"][task]["splits"] for task in args.tasks
        }
    manifest = build_rolling_manifest(
        data_root=args.data_root,
        output_dir=args.output_dir,
        tasks=args.tasks,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        horizon=args.horizon,
        window_stride=args.window_stride,
        camera=args.camera,
        seed=args.seed,
        reference_splits=reference_splits,
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
