#!/usr/bin/env python3
"""Evaluate causal within-stage alignment to frozen-GR00T reference plans.

This is the Progress-P1 diagnostic.  The current official stage is supplied
as an oracle so that within-stage alignment is measured independently from
the stage classifier.  At test time the aligner sees only the current frozen
GR00T visual feature and plans built from training episodes.  Frame numbers
are used solely to construct evaluation targets and the fixed-clock baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)


@dataclass(frozen=True)
class StageSequence:
    task: str
    episode: int
    stage_index: int
    start: int
    end: int
    frames: np.ndarray
    visual: np.ndarray

    @property
    def key(self) -> tuple[str, int]:
        return self.task, self.stage_index

    @property
    def target(self) -> np.ndarray:
        denominator = max(self.end - self.start, 1)
        return np.clip((self.frames - self.start) / denominator, 0.0, 1.0)


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
        default=PROJECT / "outputs/progress_tracker/reference_alignment_pilot.json",
    )
    parser.add_argument(
        "--plan-bank-output",
        type=Path,
        default=PROJECT / "outputs/progress_tracker/reference_plan_prototypes.npz",
    )
    parser.add_argument("--num-nodes", type=int, default=12)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def resample(sequence: StageSequence, num_nodes: int) -> np.ndarray:
    position = sequence.target
    targets = np.linspace(0.0, 1.0, num_nodes, dtype=np.float32)
    right = np.searchsorted(position, targets, side="left")
    right = np.clip(right, 0, len(position) - 1)
    left = np.maximum(right - 1, 0)
    left_position = position[left]
    right_position = position[right]
    denominator = np.maximum(right_position - left_position, 1e-8)
    alpha = np.where(right == left, 0.0, (targets - left_position) / denominator)
    values = (1.0 - alpha[:, None]) * sequence.visual[left]
    values += alpha[:, None] * sequence.visual[right]
    return normalize(values)


def load_sequences(cache_dir: Path, tasks: list[str], split: str) -> list[StageSequence]:
    sequences: list[StageSequence] = []
    for task in tasks:
        for path in sorted((cache_dir / task / split).glob("episode_*.npz")):
            episode = int(path.stem.split("_")[-1])
            with np.load(path) as values:
                visual = values["visual"].astype(np.float32)
                labels = values["labels"].astype(np.int64)
                frames = values["frames"].astype(np.int64)
                segments = json.loads(str(values["segments_json"][0]))
            for segment in segments:
                if segment["stage"] == "done":
                    continue
                stage_index = int(segment["index"])
                selected = labels == stage_index
                if not np.any(selected):
                    raise RuntimeError(f"No samples for {task} episode {episode} stage {stage_index}")
                sequences.append(
                    StageSequence(
                        task=task,
                        episode=episode,
                        stage_index=stage_index,
                        start=int(segment["start"]),
                        end=int(segment["end"]),
                        frames=frames[selected],
                        visual=visual[selected],
                    )
                )
    return sequences


def build_plan_bank(
    train: list[StageSequence], num_nodes: int
) -> tuple[dict[tuple[str, int], list[tuple[int, np.ndarray]]], dict[tuple[str, int], np.ndarray]]:
    bank: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for sequence in train:
        bank[sequence.key].append((sequence.episode, resample(sequence, num_nodes)))
    prototype = {
        key: normalize(np.mean(np.stack([plan for _, plan in plans]), axis=0))
        for key, plans in bank.items()
    }
    return dict(bank), prototype


def select_plan(
    sequence: StageSequence,
    method: str,
    bank: dict[tuple[str, int], list[tuple[int, np.ndarray]]],
    prototype: dict[tuple[str, int], np.ndarray],
    plan_variant: str = "correct",
) -> np.ndarray:
    if method == "prototype":
        plan = prototype[sequence.key]
    elif method == "retrieved":
        query = normalize(sequence.visual[:1])[0]
        candidates = bank[sequence.key]
        scores = [float(query @ candidate[0]) for _, candidate in candidates]
        plan = candidates[int(np.argmax(scores))][1]
    else:
        raise ValueError(method)
    if plan_variant == "correct":
        return plan
    if plan_variant == "reversed":
        return plan[::-1]
    if plan_variant == "constant_first":
        return np.repeat(plan[:1], len(plan), axis=0)
    raise ValueError(plan_variant)


def causal_decode(similarity: np.ndarray, margin: float, patience: int) -> np.ndarray:
    """Advance at most one node when the next node has persistent evidence."""

    current = 0
    evidence = 0
    output = []
    for row in similarity:
        if current < similarity.shape[1] - 1:
            supports_advance = row[current + 1] >= row[current] + margin
            evidence = evidence + 1 if supports_advance else 0
            if evidence >= patience:
                current += 1
                evidence = 0
        output.append(current)
    return np.asarray(output, dtype=np.int64)


def predictions_for(
    sequences: list[StageSequence],
    method: str,
    bank: dict[tuple[str, int], list[tuple[int, np.ndarray]]],
    prototype: dict[tuple[str, int], np.ndarray],
    margin: float,
    patience: int,
    plan_variant: str = "correct",
    raw_nearest: bool = False,
) -> list[dict[str, Any]]:
    records = []
    for sequence in sequences:
        plan = select_plan(sequence, method, bank, prototype, plan_variant)
        similarity = normalize(sequence.visual) @ plan.T
        prediction = (
            np.argmax(similarity, axis=1).astype(np.int64)
            if raw_nearest
            else causal_decode(similarity, margin, patience)
        )
        records.append(
            {
                "task": sequence.task,
                "episode": sequence.episode,
                "stage": sequence.stage_index,
                "target": sequence.target,
                "prediction": prediction,
                "num_nodes": plan.shape[0],
            }
        )
    return records


def fixed_clock_predictions(
    train: list[StageSequence], sequences: list[StageSequence], num_nodes: int
) -> list[dict[str, Any]]:
    durations: dict[tuple[str, int], list[int]] = defaultdict(list)
    for sequence in train:
        durations[sequence.key].append(sequence.end - sequence.start)
    median_duration = {key: float(np.median(values)) for key, values in durations.items()}
    records = []
    for sequence in sequences:
        elapsed = sequence.frames - sequence.start
        progress = np.clip(elapsed / max(median_duration[sequence.key], 1.0), 0.0, 1.0)
        prediction = np.rint(progress * (num_nodes - 1)).astype(np.int64)
        records.append(
            {
                "task": sequence.task,
                "episode": sequence.episode,
                "stage": sequence.stage_index,
                "target": sequence.target,
                "prediction": prediction,
                "num_nodes": num_nodes,
            }
        )
    return records


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    within_one = []
    correlations = []
    final_errors = []
    early_endpoint = []
    regressions = 0
    forward_jumps = 0
    per_task_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        target_node = record["target"] * (record["num_nodes"] - 1)
        error = np.abs(record["prediction"] - target_node)
        errors.extend(error.tolist())
        within_one.extend((error <= 1.0).tolist())
        final_errors.append(float(error[-1]))
        early = (record["prediction"] == record["num_nodes"] - 1) & (record["target"] < 0.9)
        early_endpoint.extend(early.tolist())
        delta = np.diff(record["prediction"])
        regressions += int(np.sum(delta < 0))
        forward_jumps += int(np.sum(delta > 1))
        if len(record["target"]) > 1 and np.std(record["prediction"]) > 0:
            correlations.append(float(np.corrcoef(record["target"], record["prediction"])[0, 1]))
        per_task_records[record["task"]].append(record)
    result = {
        "progress_mae_nodes": float(np.mean(errors)),
        "progress_mae_fraction": float(np.mean(errors) / (records[0]["num_nodes"] - 1)),
        "within_one_node": float(np.mean(within_one)),
        "mean_sequence_correlation": float(np.mean(correlations)) if correlations else None,
        "final_frame_mae_nodes": float(np.mean(final_errors)),
        "early_endpoint_frame_fraction": float(np.mean(early_endpoint)),
        "monotonic_regressions": regressions,
        "forward_jumps": forward_jumps,
        "sequences": len(records),
        "frames": len(errors),
    }
    result["per_task"] = {
        task: {key: value for key, value in metric_summary(rows).items() if key != "per_task"}
        for task, rows in per_task_records.items()
    } if len(per_task_records) > 1 else {}
    return result


def tune(
    validation: list[StageSequence],
    method: str,
    bank: dict[tuple[str, int], list[tuple[int, np.ndarray]]],
    prototype: dict[tuple[str, int], np.ndarray],
) -> dict[str, float | int]:
    best: tuple[float, float, float, int] | None = None
    selected: dict[str, float | int] | None = None
    for margin in (-0.05, -0.02, 0.0, 0.01, 0.02, 0.05):
        for patience in (1, 2, 3):
            records = predictions_for(validation, method, bank, prototype, margin, patience)
            metrics = metric_summary(records)
            candidate = (
                -float(metrics["progress_mae_fraction"]),
                float(metrics["within_one_node"]),
                -abs(margin),
                -patience,
            )
            if best is None or candidate > best:
                best = candidate
                selected = {"margin": margin, "patience": patience}
    assert selected is not None
    return selected


def main() -> int:
    args = parse_args()
    if args.num_nodes < 2:
        raise ValueError("num-nodes must be at least two")
    tasks = list(args.tasks)
    train = load_sequences(args.cache_dir, tasks, "train")
    validation = load_sequences(args.cache_dir, tasks, "val")
    test = load_sequences(args.cache_dir, tasks, "test")
    bank, prototype = build_plan_bank(train, args.num_nodes)
    args.plan_bank_output.parent.mkdir(parents=True, exist_ok=True)
    prototype_arrays = {
        f"{task}__stage_{stage}": plan.astype(np.float16)
        for (task, stage), plan in prototype.items()
    }
    prototype_arrays["keys_json"] = np.asarray(
        [json.dumps(sorted(prototype_arrays), ensure_ascii=False)]
    )
    np.savez_compressed(args.plan_bank_output, **prototype_arrays)
    results: dict[str, Any] = {
        "protocol": {
            "tasks": tasks,
            "num_nodes": args.num_nodes,
            "oracle_current_stage": True,
            "test_time_inputs": "current frozen-GR00T visual feature + train-episode reference plan",
            "frame_or_time_input_to_aligner": False,
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "test_sequences": len(test),
            "prototype_plan_bank": str(args.plan_bank_output.resolve()),
        }
    }
    for method in ("prototype", "retrieved"):
        config = tune(validation, method, bank, prototype)
        validation_records = predictions_for(
            validation, method, bank, prototype, **config
        )
        test_records = predictions_for(test, method, bank, prototype, **config)
        results[method] = {
            "decoder_config": config,
            "validation": metric_summary(validation_records),
            "test": metric_summary(test_records),
        }
        if method == "prototype":
            results[method]["test_raw_nearest"] = metric_summary(
                predictions_for(
                    test,
                    method,
                    bank,
                    prototype,
                    **config,
                    raw_nearest=True,
                )
            )
            results[method]["test_reversed_plan"] = metric_summary(
                predictions_for(
                    test,
                    method,
                    bank,
                    prototype,
                    **config,
                    plan_variant="reversed",
                )
            )
            results[method]["test_constant_first_plan"] = metric_summary(
                predictions_for(
                    test,
                    method,
                    bank,
                    prototype,
                    **config,
                    plan_variant="constant_first",
                )
            )
    results["fixed_clock"] = {
        "uses_elapsed_time": True,
        "validation": metric_summary(
            fixed_clock_predictions(train, validation, args.num_nodes)
        ),
        "test": metric_summary(fixed_clock_predictions(train, test, args.num_nodes)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
