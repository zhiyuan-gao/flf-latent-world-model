#!/usr/bin/env python3
"""Cache dense frozen features for CheckVLA-style per-action prediction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import read_jsonl  # noqa: E402
from dynamics.encoders import (  # noqa: E402
    PCAWhiteningProjector,
    build_encoder,
    spatially_pool_tokens,
)
from dynamics.manifest import episode_video  # noqa: E402
from dynamics.video_io import read_selected_rgb_frames  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "gr00t": PROJECT
    / "checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000",
    "vjepa2": PROJECT / "checkpoints/vjepa2-vitg-fpc64-256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=sorted(DEFAULT_CHECKPOINTS), default="vjepa2")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--projector-frames", type=int, default=256)
    parser.add_argument("--projector-tokens-per-frame", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fit-projector-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def grouped_frames(
    manifest_dir: Path,
    split: str,
    allowed_tasks: set[str] | None = None,
) -> dict[tuple[str, int], set[int]]:
    result: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in read_jsonl(manifest_dir / f"windows_{split}.jsonl"):
        task = str(row["task"])
        if allowed_tasks is not None and task not in allowed_tasks:
            continue
        result[(task, int(row["episode"]))].update(int(value) for value in row["frames"])
    return result


def encode_pooled(encoder, frames: np.ndarray, batch_size: int, grid_size: int) -> torch.Tensor:
    values = []
    for start in range(0, len(frames), batch_size):
        tokens = encoder.encode(frames[start : start + batch_size])
        values.append(spatially_pool_tokens(tokens, grid_size).cpu())
    return torch.cat(values)


def fit_projector(
    args: argparse.Namespace,
    encoder,
    manifest: dict,
    grouped: dict[tuple[str, int], set[int]],
) -> PCAWhiteningProjector:
    candidates = [
        (task, episode, frame)
        for (task, episode), frames in grouped.items()
        for frame in sorted(frames)
    ]
    rng = np.random.default_rng(args.seed)
    if len(candidates) > args.projector_frames:
        chosen = rng.choice(len(candidates), args.projector_frames, replace=False)
        candidates = [candidates[int(index)] for index in chosen]
    selected: dict[tuple[str, int], list[int]] = defaultdict(list)
    for task, episode, frame in candidates:
        selected[(task, episode)].append(frame)
    samples = []
    camera = str(manifest["camera"])
    for (task, episode), frames in sorted(selected.items()):
        dataset = Path(manifest["tasks"][task]["dataset"])
        rgb = read_selected_rgb_frames(episode_video(dataset, episode, camera), frames)
        pooled = encode_pooled(encoder, rgb, args.batch_size, args.grid_size).flatten(0, 2)
        count = min(args.projector_tokens_per_frame * len(rgb), len(pooled))
        generator = torch.Generator().manual_seed(
            args.seed + episode + sum(ord(value) for value in task)
        )
        samples.append(pooled[torch.randperm(len(pooled), generator=generator)[:count]])
    values = torch.cat(samples)
    print(f"Fitting PCA/whitening on {tuple(values.shape)} V-JEPA tokens", flush=True)
    return PCAWhiteningProjector.fit(
        values,
        output_dim=args.feature_dim,
        device=args.device,
    )


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.manifest_dir / "manifest.json").read_text())
    all_tasks = set(manifest["tasks"])
    allowed_tasks = set(args.tasks) if args.tasks else all_tasks
    unknown = allowed_tasks - all_tasks
    if unknown:
        raise ValueError(f"Tasks are absent from manifest: {sorted(unknown)}")
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.encoder]
    output_root = args.output_root or args.manifest_dir / "features" / args.encoder
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Loading frozen {args.encoder} encoder from {checkpoint}", flush=True)
    encoder = build_encoder(args.encoder, checkpoint, device=args.device)

    all_train = grouped_frames(args.manifest_dir, "train")
    projector_path = output_root / "projector.pt"
    if projector_path.exists() and not args.overwrite:
        projector = PCAWhiteningProjector.load(projector_path)
    else:
        projector = fit_projector(args, encoder, manifest, all_train)
        projector.save(
            projector_path,
            {
                "encoder": args.encoder,
                "checkpoint": str(checkpoint.resolve()),
                "grid_size": args.grid_size,
                "feature_dim": args.feature_dim,
                "seed": args.seed,
                "dense_per_action": True,
            },
        )
    if args.fit_projector_only:
        print(projector_path.resolve())
        return 0

    grouped_by_split = {
        split: grouped_frames(args.manifest_dir, split, allowed_tasks)
        for split in ("train", "val", "test")
    }
    total = sum(len(grouped) for grouped in grouped_by_split.values())
    complete = 0
    camera = str(manifest["camera"])
    started = time.monotonic()
    for split, grouped in grouped_by_split.items():
        for (task, episode), indices in sorted(grouped.items()):
            output = output_root / task / split / f"episode_{episode:06d}.npz"
            if output.exists() and not args.overwrite:
                complete += 1
                continue
            dataset = Path(manifest["tasks"][task]["dataset"])
            selected = sorted(indices)
            frames = read_selected_rgb_frames(
                episode_video(dataset, episode, camera), selected
            )
            pooled = encode_pooled(encoder, frames, args.batch_size, args.grid_size)
            projected = projector.transform(pooled.to(args.device)).cpu().numpy().astype(np.float16)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                frames=np.asarray(selected, dtype=np.int64),
                features=projected,
            )
            complete += 1
            print(
                f"[{args.encoder}] {complete}/{total} {task} {split} "
                f"episode={episode:06d} frames={len(frames)} "
                f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
    suffix = "_".join(sorted(allowed_tasks))
    metadata = {
        "encoder": args.encoder,
        "checkpoint": str(checkpoint.resolve()),
        "manifest": str((args.manifest_dir / "manifest.json").resolve()),
        "grid_size": args.grid_size,
        "feature_dim": args.feature_dim,
        "dense_per_action": True,
        "tasks": sorted(allowed_tasks),
        "episodes": total,
    }
    (output_root / f"metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
