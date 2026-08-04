#!/usr/bin/env python3
"""Cache frozen GR00T action-context features for progress tracking.

The split unit is an entire episode.  Frames are sampled at a fixed control
stride, with every official stage start/end added so boundary metrics retain
the exact annotation locations.  No expert actions or future observations are
given to the feature encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
GR00T_ROOT = PROJECT / "third_party/Isaac-GR00T"
sys.path.insert(0, str(GR00T_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402


DEFAULT_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)
LABEL_COLUMNS = (
    "subtask_idx",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT / "data/robocasa365/v1.0/target/composite",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT
        / "checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/progress_tracker/features_gr00t",
    )
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checkpoint_fingerprint(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        with (checkpoint / name).open("rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def find_dataset(task_root: Path) -> Path:
    matches = sorted(path.parent.parent for path in task_root.glob("*/lerobot/meta/info.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dataset under {task_root}, found {matches}")
    return matches[0]


def load_names(dataset: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        names[int(row["task_index"])] = str(row["task"])
    return names


def stable_task_seed(seed: int, task: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "little")
    return (int(seed) + suffix) % (2**32)


def split_episodes(
    episodes: list[int], train: int, val: int, test: int, seed: int
) -> dict[str, list[int]]:
    total = train + val + test
    if total > len(episodes):
        raise ValueError(f"Requested {total} episodes from only {len(episodes)}")
    shuffled = np.asarray(episodes, dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:train].tolist()),
        "val": sorted(shuffled[train : train + val].tolist()),
        "test": sorted(shuffled[train + val : total].tolist()),
    }


def label_table(dataset: Path, episode: int) -> pd.DataFrame:
    parquet = dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    return pd.read_parquet(parquet, columns=list(LABEL_COLUMNS))


def sampled_frames(labels: np.ndarray, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("frame stride must be positive")
    selected = set(range(0, len(labels), stride))
    starts = [0]
    starts.extend(index for index in range(1, len(labels)) if labels[index] != labels[index - 1])
    for position, start in enumerate(starts):
        end = starts[position + 1] - 1 if position + 1 < len(starts) else len(labels) - 1
        selected.add(start)
        selected.add(end)
    return np.asarray(sorted(selected), dtype=np.int64)


def stack_observations(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = [key for key in rows[0] if not key.startswith("action.")]
    return {key: np.stack([np.asarray(row[key]) for row in rows]) for key in keys}


@torch.inference_mode()
def encode_batch(policy: Gr00tPolicy, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    normalized = policy.apply_transforms(stack_observations(rows))
    backbone_inputs, action_inputs = policy.model.prepare_input(normalized)
    backbone = policy.model.backbone
    eagle_input_ids = backbone_inputs["eagle_input_ids"]
    image_token_id = int(backbone.eagle_model.config.image_token_index)
    image_mask = eagle_input_ids == image_token_id

    device_type = policy.model.device.type
    autocast_enabled = device_type == "cuda"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        encoded = backbone(backbone_inputs)
        encoded = policy.model.action_head.process_backbone_output(encoded)
    features = encoded["backbone_features"]
    batch = features.shape[0]
    token_counts = image_mask.sum(dim=1)
    if not torch.all(token_counts == token_counts[0]):
        raise RuntimeError(f"Variable image-token counts in batch: {token_counts.tolist()}")
    num_images = backbone_inputs["eagle_pixel_values"].shape[0] // batch
    total_image_tokens = int(token_counts[0])
    if total_image_tokens % num_images:
        raise RuntimeError(
            f"Cannot divide {total_image_tokens} image tokens across {num_images} cameras"
        )
    tokens_per_image = total_image_tokens // num_images
    selected = features[image_mask].reshape(
        batch, num_images, tokens_per_image, features.shape[-1]
    )
    # Keep camera identity but discard spatial token order in the first MVP.
    visual = selected.float().mean(dim=2).reshape(batch, -1)
    state = action_inputs["state"][:, -1].float()
    return visual.cpu().numpy().astype(np.float16), state.cpu().numpy().astype(np.float32)


def segment_metadata(frame_table: pd.DataFrame, names: dict[int, str]) -> list[dict[str, Any]]:
    labels = frame_table["subtask_idx"].to_numpy(dtype=np.int64)
    starts = [0]
    starts.extend(index for index in range(1, len(labels)) if labels[index] != labels[index - 1])
    segments = []
    for position, start in enumerate(starts):
        end = starts[position + 1] - 1 if position + 1 < len(starts) else len(labels) - 1
        segments.append(
            {
                "index": int(labels[start]),
                "start": int(start),
                "end": int(end),
                "instruction": names[int(frame_table["annotation.human.subtask"].iloc[start])],
                "atomic_skill": names[
                    int(frame_table["annotation.human.subtask_name"].iloc[start])
                ],
                "stage": names[int(frame_table["annotation.human.subtask_stage"].iloc[start])],
            }
        )
    return segments


def extract_episode(
    policy: Gr00tPolicy,
    dataset: LeRobotSingleDataset,
    dataset_path: Path,
    episode: int,
    names: dict[int, str],
    stride: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    table = label_table(dataset_path, episode)
    labels_all = table["subtask_idx"].to_numpy(dtype=np.int64)
    frames = sampled_frames(labels_all, stride)
    visual_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        rows = [dataset.get_step_data(episode, int(frame)) for frame in batch_frames]
        visual, state = encode_batch(policy, rows)
        visual_parts.append(visual)
        state_parts.append(state)
    return {
        "visual": np.concatenate(visual_parts),
        "state": np.concatenate(state_parts),
        "labels": labels_all[frames],
        "frames": frames,
        "episode_length": np.asarray([len(table)], dtype=np.int64),
        "segments_json": np.asarray(
            [json.dumps(segment_metadata(table, names), ensure_ascii=False)]
        ),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = DATA_CONFIG_MAP["panda_omron"]
    print(f"Loading frozen GR00T checkpoint: {args.checkpoint}", flush=True)
    policy = Gr00tPolicy(
        model_path=str(args.checkpoint),
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=4,
        device=args.device,
    )
    policy.model.eval()

    for task in args.tasks:
        dataset_path = find_dataset(args.data_root / task)
        parquet_paths = sorted(dataset_path.glob("data/*/episode_*.parquet"))
        episodes = [int(path.stem.split("_")[-1]) for path in parquet_paths]
        splits = split_episodes(
            episodes,
            args.train_episodes,
            args.val_episodes,
            args.test_episodes,
            stable_task_seed(args.seed, task),
        )
        names = load_names(dataset_path)
        first_table = label_table(dataset_path, episodes[0])
        canonical_segments = segment_metadata(first_table, names)
        task_dir = args.output_dir / task
        write_json(
            task_dir / "manifest.json",
            {
                "task": task,
                "dataset": str(dataset_path.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_fingerprint": checkpoint_fingerprint(args.checkpoint),
                "feature": "GR00T action-context image tokens, mean pooled per camera",
                "frame_stride": args.frame_stride,
                "fps": 20,
                "seed": args.seed,
                "splits": splits,
                "canonical_segments": canonical_segments,
            },
        )
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config.modality_config(),
            video_backend="opencv",
            transforms=None,
            embodiment_tag="new_embodiment",
        )
        total = sum(len(values) for values in splits.values())
        completed = 0
        started = time.monotonic()
        for split, split_episodes_list in splits.items():
            output_split = task_dir / split
            output_split.mkdir(parents=True, exist_ok=True)
            for episode in split_episodes_list:
                output_path = output_split / f"episode_{episode:06d}.npz"
                if output_path.exists() and not args.overwrite:
                    completed += 1
                    continue
                values = extract_episode(
                    policy,
                    dataset,
                    dataset_path,
                    episode,
                    names,
                    args.frame_stride,
                    args.batch_size,
                )
                np.savez_compressed(output_path, **values)
                completed += 1
                elapsed = time.monotonic() - started
                print(
                    f"[{task}] {completed}/{total} {split} episode={episode:06d} "
                    f"samples={len(values['labels'])} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
