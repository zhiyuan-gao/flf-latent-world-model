#!/usr/bin/env python3
"""Cache matched 8x8x256 frozen features for one bake-off encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.encoders import (  # noqa: E402
    PCAWhiteningProjector,
    build_encoder,
    spatially_pool_tokens,
)
from dynamics.manifest import episode_video, read_jsonl  # noqa: E402
from dynamics.video_io import read_selected_rgb_frames  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "gr00t": PROJECT
    / "checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000",
    "vjepa2": PROJECT / "checkpoints/vjepa2-vitg-fpc64-256",
}


def checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        root = path
        files = sorted(
            value
            for pattern in ("*.json", "*.safetensors")
            for value in path.glob(pattern)
            if value.is_file()
        )
    for value in files:
        digest.update(str(value.relative_to(root)).encode())
        with value.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=sorted(DEFAULT_CHECKPOINTS), required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/dynamics_bakeoff",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--projector-frames", type=int, default=256)
    parser.add_argument("--projector-tokens-per-frame", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def grouped_required_frames(manifest_dir: Path, split: str) -> dict[tuple[str, int], set[int]]:
    result: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in read_jsonl(manifest_dir / f"windows_{split}.jsonl"):
        key = (str(row["task"]), int(row["episode"]))
        result[key].update(int(value) for value in row["history_frames"])
        result[key].update(int(value) for value in row["future_frames"])
    return result


def encode_pooled(
    encoder,
    frames: np.ndarray,
    batch_size: int,
    grid_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(frames), batch_size):
        tokens = encoder.encode(frames[start : start + batch_size])
        parts.append(spatially_pool_tokens(tokens, grid_size).cpu())
    return torch.cat(parts)


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
        selected_indices = rng.choice(len(candidates), args.projector_frames, replace=False)
        selected = [candidates[int(index)] for index in selected_indices]
    else:
        selected = candidates
    selected_grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for task, episode, frame in selected:
        selected_grouped[(task, episode)].append(frame)

    samples = []
    camera = str(manifest["camera"])
    for (task, episode), indices in sorted(selected_grouped.items()):
        dataset = Path(manifest["tasks"][task]["dataset"])
        frames = read_selected_rgb_frames(episode_video(dataset, episode, camera), indices)
        pooled = encode_pooled(encoder, frames, args.batch_size, args.grid_size)
        pooled = pooled.reshape(-1, pooled.shape[-1])
        count = min(args.projector_tokens_per_frame * len(frames), len(pooled))
        chosen = torch.randperm(len(pooled), generator=torch.Generator().manual_seed(args.seed))[
            :count
        ]
        samples.append(pooled[chosen])
    values = torch.cat(samples)
    print(f"Fitting PCA/whitening projector on {tuple(values.shape)} tokens", flush=True)
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
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.encoder]
    output_root = args.output_root or args.manifest_dir / "features" / args.encoder
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.manifest_dir / "manifest.json").read_text())
    if int(manifest["future_offsets"][-1]) != 16:
        raise ValueError("Manifest does not implement the 16-action/four-future contract")
    print(f"Loading frozen {args.encoder} encoder from {checkpoint}", flush=True)
    encoder = build_encoder(args.encoder, checkpoint, device=args.device)

    grouped_by_split = {
        split: grouped_required_frames(args.manifest_dir, split)
        for split in ("train", "val", "test")
    }
    projector_path = output_root / "projector.pt"
    if projector_path.exists() and not args.overwrite:
        projector = PCAWhiteningProjector.load(projector_path)
    else:
        projector = fit_projector(args, encoder, manifest, grouped_by_split["train"])
        projector.save(
            projector_path,
            {
                "encoder": args.encoder,
                "checkpoint": str(checkpoint.resolve()),
                "grid_size": args.grid_size,
                "feature_dim": args.feature_dim,
                "seed": args.seed,
            },
        )

    camera = str(manifest["camera"])
    total = sum(len(values) for values in grouped_by_split.values())
    complete = 0
    started = time.monotonic()
    for split, grouped in grouped_by_split.items():
        for (task, episode), indices in sorted(grouped.items()):
            output = output_root / task / split / f"episode_{episode:06d}.npz"
            if output.exists() and not args.overwrite:
                complete += 1
                continue
            dataset = Path(manifest["tasks"][task]["dataset"])
            sorted_indices = sorted(indices)
            frames = read_selected_rgb_frames(
                episode_video(dataset, episode, camera), sorted_indices
            )
            pooled = encode_pooled(encoder, frames, args.batch_size, args.grid_size)
            projected = projector.transform(pooled.to(args.device)).cpu().numpy().astype(np.float16)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                frames=np.asarray(sorted_indices, dtype=np.int64),
                features=projected,
            )
            complete += 1
            print(
                f"[{args.encoder}] {complete}/{total} {task} {split} episode={episode:06d} "
                f"frames={len(frames)} elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
    metadata = {
        "encoder": args.encoder,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "manifest": str((args.manifest_dir / "manifest.json").resolve()),
        "grid_size": args.grid_size,
        "feature_dim": args.feature_dim,
        "episodes": total,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
