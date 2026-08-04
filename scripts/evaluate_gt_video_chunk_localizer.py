#!/usr/bin/env python3
"""Evaluate pure-video localization of GT chunks inside their full stage videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from progress.video_localizer import GTVideoChunkLocalizer  # noqa: E402
from evaluate_reference_progress import StageSequence, load_sequences  # noqa: E402


DEFAULT_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT / "outputs/progress_tracker/features_gr00t_pilot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs/progress_tracker/gt_video_chunk_localization.json",
    )
    parser.add_argument("--chunk-lengths", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--queries-per-sequence", type=int, default=8)
    parser.add_argument(
        "--feature-source",
        choices=("rgb32", "cached_gr00t_visual"),
        default="rgb32",
    )
    parser.add_argument(
        "--camera-key",
        default="observation.images.robot0_agentview_left",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def query_starts(length: int, chunk_length: int, count: int) -> list[int]:
    available = length - chunk_length + 1
    if available < 1:
        return []
    if available <= count:
        return list(range(available))
    return sorted(
        set(np.rint(np.linspace(0, available - 1, count)).astype(np.int64).tolist())
    )


def stable_rng(seed: int, sequence: StageSequence, start: int) -> np.random.Generator:
    key = f"{seed}:{sequence.task}:{sequence.episode}:{sequence.stage_index}:{start}"
    suffix = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(suffix)


def replace_with_rgb_features(
    sequences: list[StageSequence],
    cache_dir: Path,
    split: str,
    camera_key: str,
    image_size: int = 32,
) -> list[StageSequence]:
    grouped: dict[tuple[str, int], list[StageSequence]] = defaultdict(list)
    for sequence in sequences:
        grouped[(sequence.task, sequence.episode)].append(sequence)
    result = []
    for (task, episode), episode_sequences in grouped.items():
        manifest = json.loads((cache_dir / task / "manifest.json").read_text())
        dataset = Path(manifest["dataset"])
        video = (
            dataset
            / f"videos/chunk-{episode // 1000:03d}"
            / camera_key
            / f"episode_{episode:06d}.mp4"
        )
        required = sorted(
            {int(frame) for sequence in episode_sequences for frame in sequence.frames}
        )
        required_set = set(required)
        decoded: dict[int, np.ndarray] = {}
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video}")
        frame_index = 0
        final_required = required[-1]
        while frame_index <= final_required:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in required_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                decoded[frame_index] = frame.astype(np.float32).reshape(-1) / 255.0
            frame_index += 1
        capture.release()
        missing = sorted(required_set - set(decoded))
        if missing:
            raise RuntimeError(f"Missing frames in {video}: {missing[:10]}")
        for sequence in episode_sequences:
            result.append(
                StageSequence(
                    task=sequence.task,
                    episode=sequence.episode,
                    stage_index=sequence.stage_index,
                    start=sequence.start,
                    end=sequence.end,
                    frames=sequence.frames,
                    visual=np.stack([decoded[int(frame)] for frame in sequence.frames]),
                )
            )
    return sorted(result, key=lambda row: (row.task, row.episode, row.stage_index))


def temporal_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = max(end_a, end_b) - min(start_a, start_b) + 1
    return intersection / union


def localize_queries(
    sequences: list[StageSequence],
    chunk_length: int,
    queries_per_sequence: int,
    localizer: GTVideoChunkLocalizer,
    query_variant: str = "normal",
    seed: int = 20260802,
) -> list[dict[str, Any]]:
    records = []
    for sequence in sequences:
        reference = torch.from_numpy(sequence.visual.astype(np.float32))
        for start in query_starts(len(reference), chunk_length, queries_per_sequence):
            query = reference[start : start + chunk_length].clone()
            if query_variant == "reversed":
                query = torch.flip(query, dims=(0,))
            elif query_variant == "shuffled":
                permutation = stable_rng(seed, sequence, start).permutation(chunk_length)
                query = query[torch.from_numpy(permutation)]
            elif query_variant != "normal":
                raise ValueError(query_variant)
            result = localizer.localize(reference, query)
            true_end = start + chunk_length - 1
            denominator = max(len(reference) - 1, 1)
            records.append(
                {
                    "task": sequence.task,
                    "episode": sequence.episode,
                    "stage": sequence.stage_index,
                    "reference_length": len(reference),
                    "true_start": start,
                    "true_end": true_end,
                    "predicted_start": result.start_index,
                    "predicted_end": result.end_index,
                    "start_error": abs(result.start_index - start),
                    "end_error": abs(result.end_index - true_end),
                    "progress_error": abs(result.end_index - true_end) / denominator,
                    "temporal_iou": temporal_iou(
                        result.start_index, result.end_index, start, true_end
                    ),
                    "exact": result.start_index == start,
                    "within_one": abs(result.start_index - start) <= 1,
                    "confidence_margin": result.confidence_margin,
                }
            )
    return records


def summarize(records: list[dict[str, Any]], include_per_task: bool = True) -> dict[str, Any]:
    if not records:
        return {"queries": 0}
    result = {
        "queries": len(records),
        "start_mae_samples": float(np.mean([row["start_error"] for row in records])),
        "end_mae_samples": float(np.mean([row["end_error"] for row in records])),
        "progress_mae_fraction": float(np.mean([row["progress_error"] for row in records])),
        "exact_window_accuracy": float(np.mean([row["exact"] for row in records])),
        "within_one_sample": float(np.mean([row["within_one"] for row in records])),
        "mean_temporal_iou": float(np.mean([row["temporal_iou"] for row in records])),
        "mean_confidence_margin": float(
            np.mean(
                [row["confidence_margin"] for row in records if np.isfinite(row["confidence_margin"])]
            )
        ),
    }
    if include_per_task:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            by_task[row["task"]].append(row)
        result["per_task"] = {
            task: summarize(rows, include_per_task=False) for task, rows in by_task.items()
        }
    return result


def tune_weights(
    validation: list[StageSequence],
    chunk_length: int,
    queries_per_sequence: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    best = None
    selected = None
    selected_metrics = None
    for motion_weight in (0.0, 0.25, 0.5, 1.0):
        for start_anchor_weight in (0.0, 0.25, 0.5, 1.0):
            config = {
                "appearance_weight": 1.0,
                "motion_weight": motion_weight,
                "start_anchor_weight": start_anchor_weight,
            }
            records = localize_queries(
                validation,
                chunk_length,
                queries_per_sequence,
                GTVideoChunkLocalizer(**config),
                seed=seed,
            )
            metrics = summarize(records)
            candidate = (
                metrics["exact_window_accuracy"],
                metrics["mean_temporal_iou"],
                -metrics["progress_mae_fraction"],
                -motion_weight - start_anchor_weight,
            )
            if best is None or candidate > best:
                best = candidate
                selected = config
                selected_metrics = metrics
    assert selected is not None and selected_metrics is not None
    return selected, selected_metrics


def main() -> int:
    args = parse_args()
    if min(args.chunk_lengths) < 1 or args.queries_per_sequence < 1:
        raise ValueError("chunk lengths and query count must be positive")
    tasks = list(args.tasks)
    validation = load_sequences(args.cache_dir, tasks, "val")
    test = load_sequences(args.cache_dir, tasks, "test")
    if args.feature_source == "rgb32":
        validation = replace_with_rgb_features(
            validation, args.cache_dir, "val", args.camera_key
        )
        test = replace_with_rgb_features(test, args.cache_dir, "test", args.camera_key)
    results: dict[str, Any] = {
        "protocol": {
            "task": "locate a GT query chunk inside the same full GT stage video",
            "tasks": tasks,
            "validation_videos": len(validation),
            "test_videos": len(test),
            "episode_split_isolated": True,
            "policy_or_action_input": False,
            "proprioception_or_state_input": False,
            "frame_or_time_input": False,
            "feature_source": args.feature_source,
            "camera_key": args.camera_key if args.feature_source == "rgb32" else None,
            "feature_frame_stride": 8,
            "queries_per_sequence": args.queries_per_sequence,
        },
        "chunk_lengths": {},
    }
    for chunk_length in args.chunk_lengths:
        config, validation_metrics = tune_weights(
            validation, chunk_length, args.queries_per_sequence, args.seed
        )
        localizer = GTVideoChunkLocalizer(**config)
        test_records = localize_queries(
            test,
            chunk_length,
            args.queries_per_sequence,
            localizer,
            seed=args.seed,
        )
        row: dict[str, Any] = {
            "weights": config,
            "validation": validation_metrics,
            "test": summarize(test_records),
        }
        if chunk_length > 1:
            row["reversed_query"] = summarize(
                localize_queries(
                    test,
                    chunk_length,
                    args.queries_per_sequence,
                    localizer,
                    query_variant="reversed",
                    seed=args.seed,
                )
            )
            row["shuffled_query"] = summarize(
                localize_queries(
                    test,
                    chunk_length,
                    args.queries_per_sequence,
                    localizer,
                    query_variant="shuffled",
                    seed=args.seed,
                )
            )
        results["chunk_lengths"][str(chunk_length)] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
