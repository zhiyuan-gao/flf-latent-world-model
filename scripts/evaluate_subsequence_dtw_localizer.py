#!/usr/bin/env python3
"""Evaluate fixed-span-free subsequence-DTW on nonuniform GT video warps."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from progress.video_localizer import SubsequenceDTWLocalizer  # noqa: E402
from evaluate_gt_video_chunk_localizer import (  # noqa: E402
    replace_with_rgb_features,
    temporal_iou,
)
from evaluate_reference_progress import StageSequence, load_sequences  # noqa: E402


DEFAULT_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)
WARP_DELTAS = {
    "normal": (1, 1, 1, 1, 1, 1, 1),
    "slow_with_stays": (0, 1, 0, 1, 0, 1, 0),
    "fast_with_skips": (2, 1, 2, 1, 2, 1, 2),
    "nonuniform": (0, 1, 3, 0, 2, 1, 3),
    "stall_then_fast": (0, 0, 0, 3, 3, 3, 2),
}


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
        default=PROJECT / "outputs/progress_tracker/subsequence_dtw_localization.json",
    )
    parser.add_argument("--queries-per-sequence", type=int, default=4)
    parser.add_argument("--motion-weight", type=float, default=0.0)
    parser.add_argument("--stay-penalty", type=float, default=0.0)
    parser.add_argument("--jump-penalty", type=float, default=0.0)
    parser.add_argument(
        "--camera-key",
        default="observation.images.robot0_agentview_left",
    )
    return parser.parse_args()


def starts_for_path(reference_length: int, path_span: int, count: int) -> list[int]:
    available = reference_length - path_span
    if available < 1:
        return []
    if available <= count:
        return list(range(available))
    return sorted(
        set(np.rint(np.linspace(0, available - 1, count)).astype(np.int64).tolist())
    )


def evaluate_mode(
    sequences: list[StageSequence],
    deltas: tuple[int, ...],
    mode: str,
    queries_per_sequence: int,
    localizer: SubsequenceDTWLocalizer,
) -> list[dict[str, Any]]:
    records = []
    relative_path = np.concatenate(([0], np.cumsum(np.asarray(deltas, dtype=np.int64))))
    true_span = int(relative_path[-1] - relative_path[0] + 1)
    for sequence in sequences:
        reference = torch.from_numpy(sequence.visual.astype(np.float32))
        for start in starts_for_path(len(reference), true_span, queries_per_sequence):
            true_path = relative_path + start
            query = reference[torch.from_numpy(true_path)].clone()
            prediction = localizer.localize(reference, query)
            predicted_path = np.asarray(prediction.path, dtype=np.int64)
            predicted_span = prediction.end_index - prediction.start_index + 1
            denominator = max(len(reference) - 1, 1)
            records.append(
                {
                    "mode": mode,
                    "task": sequence.task,
                    "episode": sequence.episode,
                    "stage": sequence.stage_index,
                    "true_path": true_path.tolist(),
                    "predicted_path": predicted_path.tolist(),
                    "start_error": abs(prediction.start_index - int(true_path[0])),
                    "end_error": abs(prediction.end_index - int(true_path[-1])),
                    "progress_error": abs(prediction.end_index - int(true_path[-1]))
                    / denominator,
                    "path_mae": float(np.mean(np.abs(predicted_path - true_path))),
                    "span_error": abs(predicted_span - true_span),
                    "speed_ratio_error": abs(predicted_span - true_span) / len(true_path),
                    "temporal_iou": temporal_iou(
                        prediction.start_index,
                        prediction.end_index,
                        int(true_path[0]),
                        int(true_path[-1]),
                    ),
                    "exact_path": bool(np.array_equal(predicted_path, true_path)),
                    "endpoint_within_one": abs(prediction.end_index - int(true_path[-1])) <= 1,
                    "mean_cost": prediction.mean_cost,
                    "confidence_margin": prediction.confidence_margin,
                }
            )
    return records


def summarize(records: list[dict[str, Any]], per_task: bool = True) -> dict[str, Any]:
    if not records:
        return {"queries": 0}
    result = {
        "queries": len(records),
        "start_mae_samples": float(np.mean([row["start_error"] for row in records])),
        "end_mae_samples": float(np.mean([row["end_error"] for row in records])),
        "progress_mae_fraction": float(np.mean([row["progress_error"] for row in records])),
        "path_mae_samples": float(np.mean([row["path_mae"] for row in records])),
        "span_mae_samples": float(np.mean([row["span_error"] for row in records])),
        "speed_ratio_mae": float(np.mean([row["speed_ratio_error"] for row in records])),
        "mean_temporal_iou": float(np.mean([row["temporal_iou"] for row in records])),
        "exact_path_accuracy": float(np.mean([row["exact_path"] for row in records])),
        "endpoint_within_one": float(np.mean([row["endpoint_within_one"] for row in records])),
        "mean_alignment_cost": float(np.mean([row["mean_cost"] for row in records])),
        "mean_confidence_margin": float(np.mean([row["confidence_margin"] for row in records])),
    }
    if per_task:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[row["task"]].append(row)
        result["per_task"] = {
            task: summarize(rows, per_task=False) for task, rows in grouped.items()
        }
    return result


def main() -> int:
    args = parse_args()
    if args.queries_per_sequence < 1:
        raise ValueError("queries-per-sequence must be positive")
    tasks = list(args.tasks)
    validation = replace_with_rgb_features(
        load_sequences(args.cache_dir, tasks, "val"),
        args.cache_dir,
        "val",
        args.camera_key,
    )
    test = replace_with_rgb_features(
        load_sequences(args.cache_dir, tasks, "test"),
        args.cache_dir,
        "test",
        args.camera_key,
    )
    config = {
        "motion_weight": args.motion_weight,
        "stay_penalty": args.stay_penalty,
        "jump_penalty": args.jump_penalty,
    }
    localizer = SubsequenceDTWLocalizer(**config)
    results: dict[str, Any] = {
        "protocol": {
            "task": "monotonic subsequence alignment with no fixed window or global speed",
            "tasks": tasks,
            "feature_source": "rgb32",
            "camera_key": args.camera_key,
            "feature_frame_stride": 8,
            "query_length": 8,
            "validation_videos": len(validation),
            "test_videos": len(test),
            "queries_per_sequence": args.queries_per_sequence,
            "hard_max_reference_span": None,
            "policy_action_state_or_time_input": False,
            "localizer_config": config,
        },
        "validation": {},
        "test": {},
    }
    for mode, deltas in WARP_DELTAS.items():
        validation_records = evaluate_mode(
            validation, deltas, mode, args.queries_per_sequence, localizer
        )
        test_records = evaluate_mode(test, deltas, mode, args.queries_per_sequence, localizer)
        results["validation"][mode] = summarize(validation_records)
        results["test"][mode] = summarize(test_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
