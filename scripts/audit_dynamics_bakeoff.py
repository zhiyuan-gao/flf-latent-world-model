#!/usr/bin/env python3
"""Verify split isolation and every projected feature cache before training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.manifest import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("gr00t", "vjepa2"), required=True)
    parser.add_argument(
        "--manifest-dir", type=Path, default=PROJECT / "outputs/dynamics_bakeoff"
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_root = args.feature_root or args.manifest_dir / "features" / args.encoder
    manifest = json.loads((args.manifest_dir / "manifest.json").read_text())
    required: dict[str, dict[tuple[str, int], set[int]]] = {}
    for split in ("train", "val", "test"):
        grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
        for row in read_jsonl(args.manifest_dir / f"windows_{split}.jsonl"):
            key = (str(row["task"]), int(row["episode"]))
            grouped[key].update(int(value) for value in row["history_frames"])
            grouped[key].update(int(value) for value in row["future_frames"])
        required[split] = grouped

    split_episodes = {
        split: set(grouped) for split, grouped in required.items()
    }
    overlap = {
        "train_val": sorted(split_episodes["train"] & split_episodes["val"]),
        "train_test": sorted(split_episodes["train"] & split_episodes["test"]),
        "val_test": sorted(split_episodes["val"] & split_episodes["test"]),
    }
    errors: list[str] = []
    if any(overlap.values()):
        errors.append("episode split overlap detected")

    episode_count = 0
    frame_count = 0
    byte_count = 0
    expected_shape = (8, 8, 256)
    for split, grouped in required.items():
        for (task, episode), requested in sorted(grouped.items()):
            path = feature_root / task / split / f"episode_{episode:06d}.npz"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            byte_count += path.stat().st_size
            with np.load(path) as values:
                frames = values["frames"].astype(np.int64)
                features = values["features"]
            missing = sorted(set(requested) - set(frames.tolist()))
            if missing:
                errors.append(f"{path} missing {len(missing)} required frames")
            if features.shape[1:] != expected_shape:
                errors.append(f"{path} has feature shape {features.shape}")
            if len(frames) != len(features):
                errors.append(f"{path} frame/feature length mismatch")
            if not np.isfinite(features).all():
                errors.append(f"{path} contains non-finite values")
            episode_count += 1
            frame_count += len(frames)

    expected_episodes = sum(len(values) for values in required.values())
    report = {
        "encoder": args.encoder,
        "feature_root": str(feature_root.resolve()),
        "expected_episodes": expected_episodes,
        "valid_episode_files": episode_count,
        "cached_frames": frame_count,
        "cache_gib": byte_count / 2**30,
        "split_overlap": overlap,
        "errors": errors,
        "passed": not errors and episode_count == expected_episodes,
    }
    output = args.output or feature_root / "audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
