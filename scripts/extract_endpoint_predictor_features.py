#!/usr/bin/env python3
"""Cache native 16x16x1408 V-JEPA features for endpoint prediction."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import read_jsonl  # noqa: E402
from dynamics.encoders import build_encoder  # noqa: E402
from dynamics.manifest import episode_video  # noqa: E402
from dynamics.video_io import read_selected_rgb_frames  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT / "checkpoints/vjepa2-vitg-fpc64-256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Cache only the requested splits; use train val to keep test locked.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def grouped_endpoint_frames(
    manifest_dir: Path,
    split: str,
    allowed_tasks: set[str],
) -> dict[tuple[str, int], set[int]]:
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in read_jsonl(manifest_dir / f"windows_{split}.jsonl"):
        task = str(row["task"])
        if task not in allowed_tasks:
            continue
        current = int(row["current_frame"])
        horizon = int(row["horizon"])
        grouped[(task, int(row["episode"]))].update(
            (current, current + horizon)
        )
    return grouped


@torch.inference_mode()
def encode_native_grid(encoder, frames: np.ndarray, batch_size: int) -> torch.Tensor:
    outputs = []
    for start in range(0, len(frames), batch_size):
        tokens = encoder.encode(frames[start : start + batch_size])
        grid_size = math.isqrt(tokens.shape[1])
        if grid_size * grid_size != tokens.shape[1]:
            raise ValueError(
                f"Expected square V-JEPA patch tokens, received {tuple(tokens.shape)}"
            )
        outputs.append(tokens.reshape(len(tokens), grid_size, grid_size, -1).cpu())
    return torch.cat(outputs)


def main() -> int:
    args = parse_args()
    if args.crop_size != 256:
        raise ValueError("The endpoint baseline is fixed to the 256px V-JEPA 2-AC recipe")
    manifest = json.loads((args.manifest_dir / "manifest.json").read_text())
    all_tasks = set(manifest["tasks"])
    allowed_tasks = set(args.tasks) if args.tasks else all_tasks
    unknown = allowed_tasks - all_tasks
    if unknown:
        raise ValueError(f"Tasks are absent from manifest: {sorted(unknown)}")
    output_root = args.output_root or (
        args.manifest_dir / "features" / "vjepa2_native_256"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder(
        "vjepa2",
        args.checkpoint,
        device=args.device,
        vjepa_crop_size=args.crop_size,
    )

    requested_splits = tuple(dict.fromkeys(args.splits))
    grouped_by_split = {
        split: grouped_endpoint_frames(args.manifest_dir, split, allowed_tasks)
        for split in requested_splits
    }
    total = sum(len(values) for values in grouped_by_split.values())
    completed = 0
    started = time.monotonic()
    camera = str(manifest["camera"])
    observed_shape: tuple[int, ...] | None = None
    for split, grouped in grouped_by_split.items():
        for (task, episode), frame_indices in sorted(grouped.items()):
            output_base = output_root / task / split / f"episode_{episode:06d}"
            frames_output = output_base.with_suffix(".frames.npy")
            features_output = output_base.with_suffix(".features.npy")
            if frames_output.is_file() and features_output.is_file() and not args.overwrite:
                if observed_shape is None:
                    observed_shape = tuple(
                        np.load(features_output, mmap_mode="r").shape[1:]
                    )
                    if observed_shape != (16, 16, 1408):
                        raise ValueError(
                            f"Existing endpoint cache has invalid shape {observed_shape}: "
                            f"{features_output}"
                        )
                completed += 1
                continue
            dataset = Path(manifest["tasks"][task]["dataset"])
            selected = sorted(frame_indices)
            frames = read_selected_rgb_frames(
                episode_video(dataset, episode, camera), selected
            )
            features = encode_native_grid(encoder, frames, args.batch_size)
            observed_shape = tuple(features.shape[1:])
            if observed_shape != (16, 16, 1408):
                raise ValueError(
                    "The 256px V-JEPA endpoint cache must be 16x16x1408, "
                    f"received {observed_shape}"
                )
            output_base.parent.mkdir(parents=True, exist_ok=True)
            temporary_frames = frames_output.with_suffix(".npy.tmp")
            temporary_features = features_output.with_suffix(".npy.tmp")
            try:
                with temporary_frames.open("wb") as handle:
                    np.save(handle, np.asarray(selected, dtype=np.int64))
                with temporary_features.open("wb") as handle:
                    np.save(handle, features.numpy().astype(np.float16))
                temporary_features.replace(features_output)
                temporary_frames.replace(frames_output)
            finally:
                temporary_frames.unlink(missing_ok=True)
                temporary_features.unlink(missing_ok=True)
            completed += 1
            print(
                f"{completed}/{total} {task} {split} episode={episode:06d} "
                f"frames={len(selected)} elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )

    metadata = {
        "encoder": "vjepa2",
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str((args.manifest_dir / "manifest.json").resolve()),
        "crop_size": args.crop_size,
        "patch_size": 16,
        "grid_size": 16,
        "feature_dim": 1408,
        "projection": None,
        "storage": "npy_memmap_float16",
        "cached_offsets": [0, int(manifest["horizon"])],
        "tasks": sorted(allowed_tasks),
        "splits": list(requested_splits),
        "episodes": total,
        "observed_feature_shape": list(observed_shape) if observed_shape else None,
    }
    suffix = "_".join(sorted(allowed_tasks))
    (output_root / f"metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
