#!/usr/bin/env python3
"""Convert endpoint NPZ caches to memory-mapped NPY frame/feature pairs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".npy.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=(
            PROJECT
            / "outputs/checkvla_offline_predictor/features/vjepa2_native_256"
        ),
    )
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()

    allowed_tasks = set(args.tasks) if args.tasks else None
    sources = sorted(args.feature_root.glob("*/*/episode_*.npz"))
    if allowed_tasks is not None:
        sources = [path for path in sources if path.parts[-3] in allowed_tasks]
    if not sources:
        raise FileNotFoundError("No matching endpoint NPZ caches were found")

    started = time.monotonic()
    converted = 0
    reused = 0
    for index, source in enumerate(sources, 1):
        output_base = source.with_suffix("")
        frames_output = output_base.with_suffix(".frames.npy")
        features_output = output_base.with_suffix(".features.npy")
        if frames_output.is_file() and features_output.is_file():
            frames = np.load(frames_output)
            features = np.load(features_output, mmap_mode="r")
            reused += 1
        else:
            with np.load(source) as cached:
                frames = cached["frames"].astype(np.int64, copy=False)
                features = cached["features"].astype(np.float16, copy=False)
            atomic_save(features_output, features)
            atomic_save(frames_output, frames)
            converted += 1
        if features.shape != (len(frames), 16, 16, 1408):
            raise ValueError(f"Invalid endpoint feature shape in {source}: {features.shape}")
        if features.dtype != np.float16:
            raise ValueError(f"Invalid endpoint feature dtype in {source}: {features.dtype}")
        if index % 20 == 0 or index == len(sources):
            print(
                f"{index}/{len(sources)} converted={converted} reused={reused} "
                f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
