#!/usr/bin/env python3
"""Evaluate start-anchored, history-aware progress alignment.

The current official stage is supplied as an oracle so this experiment
isolates within-stage alignment.  The aligner sees the real stage-start
observation, the current and three previous frozen-GR00T visual features, and
a 12-node plan built only from training demonstrations.  It never sees frame
indices or elapsed time.
"""

from __future__ import annotations

import argparse
import json
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

from evaluate_reference_progress import (  # noqa: E402
    StageSequence,
    build_plan_bank,
    fixed_clock_predictions,
    load_sequences,
    metric_summary,
    normalize,
    resample,
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
        default=PROJECT / "outputs/progress_tracker/temporal_alignment_pilot.json",
    )
    parser.add_argument("--num-nodes", type=int, default=12)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--max-reference-span", type=int, default=3)
    return parser.parse_args()


def safe_row_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def reference_window(plan: np.ndarray, end: int, span: int, steps: int) -> np.ndarray:
    position = np.linspace(end - span, end, steps, dtype=np.float32)
    left = np.floor(position).astype(np.int64)
    right = np.ceil(position).astype(np.int64)
    alpha = (position - left)[:, None]
    return (1.0 - alpha) * plan[left] + alpha * plan[right]


def alignment_components(
    sequence: StageSequence,
    plan: np.ndarray,
    history_length: int,
    max_reference_span: int,
    reverse_motion: bool = False,
) -> dict[str, np.ndarray]:
    visual = normalize(sequence.visual)
    plan = normalize(plan)
    appearance = visual @ plan.T
    anchor = safe_row_normalize(visual - visual[:1]) @ safe_row_normalize(
        plan - plan[:1]
    ).T
    motion = np.zeros_like(appearance, dtype=np.float32)

    reference_deltas: dict[tuple[int, int, int], np.ndarray] = {}
    for steps in range(2, history_length + 1):
        for candidate in range(1, len(plan)):
            for span in range(1, min(candidate, max_reference_span) + 1):
                window = reference_window(plan, candidate, span, steps)
                reference_deltas[(steps, candidate, span)] = safe_row_normalize(
                    np.diff(window, axis=0)
                )

    for timestep in range(1, len(visual)):
        start = max(0, timestep - history_length + 1)
        history = visual[start : timestep + 1]
        if reverse_motion and len(history) > 2:
            history = np.concatenate((history[-2::-1], history[-1:]), axis=0)
        actual_delta = safe_row_normalize(np.diff(history, axis=0))
        steps = len(history)
        for candidate in range(1, len(plan)):
            best = -1.0
            for span in range(1, min(candidate, max_reference_span) + 1):
                reference_delta = reference_deltas[(steps, candidate, span)]
                score = float(np.sum(actual_delta * reference_delta, axis=-1).mean())
                best = max(best, score)
            motion[timestep, candidate] = best

    observed_motion = np.zeros(len(visual), dtype=bool)
    observed_motion[1:] = np.linalg.norm(np.diff(visual, axis=0), axis=-1) > 1e-6
    return {
        "appearance": appearance,
        "anchor": anchor,
        "motion": motion,
        "observed_motion": observed_motion,
    }


def decode(
    components: dict[str, np.ndarray],
    appearance_weight: float,
    anchor_weight: float,
    motion_weight: float,
    max_advance: int,
    margin: float,
    patience: int,
) -> np.ndarray:
    score = (
        appearance_weight * components["appearance"]
        + anchor_weight * components["anchor"]
        + motion_weight * components["motion"]
    )
    current = 0
    evidence = 0
    output = []
    for timestep, row in enumerate(score):
        if current < score.shape[1] - 1:
            candidate_end = min(current + max_advance, score.shape[1] - 1)
            relative_best = int(np.argmax(row[current : candidate_end + 1]))
            best = current + relative_best
            supports = bool(
                components["observed_motion"][timestep]
                and best > current
                and row[best] >= row[current] + margin
            )
            evidence = evidence + 1 if supports else 0
            if evidence >= patience:
                current = best
                evidence = 0
        output.append(current)
    return np.asarray(output, dtype=np.int64)


def prepare(
    sequences: list[StageSequence],
    prototype: dict[tuple[str, int], np.ndarray],
    history_length: int,
    max_reference_span: int,
    plan_variant: str = "correct",
    reverse_motion: bool = False,
    episode_plans: dict[tuple[str, int, int], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    prepared = []
    for sequence in sequences:
        episode_key = (sequence.task, sequence.episode, sequence.stage_index)
        plan = (
            episode_plans[episode_key]
            if episode_plans is not None
            else prototype[sequence.key]
        )
        if plan_variant == "reversed":
            plan = plan[::-1]
        elif plan_variant == "constant_first":
            plan = np.repeat(plan[:1], len(plan), axis=0)
        elif plan_variant != "correct":
            raise ValueError(plan_variant)
        prepared.append(
            {
                "sequence": sequence,
                "components": alignment_components(
                    sequence,
                    plan,
                    history_length,
                    max_reference_span,
                    reverse_motion,
                ),
            }
        )
    return prepared


def records_for(prepared: list[dict[str, Any]], config: dict[str, float | int]) -> list[dict[str, Any]]:
    records = []
    for row in prepared:
        sequence = row["sequence"]
        records.append(
            {
                "task": sequence.task,
                "episode": sequence.episode,
                "stage": sequence.stage_index,
                "target": sequence.target,
                "prediction": decode(row["components"], **config),
                "num_nodes": int(row["components"]["appearance"].shape[1]),
            }
        )
    return records


def tune(
    prepared: list[dict[str, Any]],
    variant: str,
    robustness_prepared: list[list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    if variant == "single_frame":
        weight_grid = [(1.0, 0.0, 0.0)]
    elif variant == "start_anchor":
        weight_grid = [(1.0, anchor, 0.0) for anchor in (0.25, 0.5, 1.0)]
    elif variant == "temporal":
        weight_grid = [
            (1.0, anchor, motion)
            for anchor in (0.25, 0.5, 1.0)
            for motion in (0.25, 0.5, 1.0)
        ]
    else:
        raise ValueError(variant)

    max_advance_grid = (1, 2) if variant == "temporal" else (1,)
    best_score = None
    best_config = None
    best_metrics = None
    for appearance, anchor, motion in weight_grid:
        for max_advance in max_advance_grid:
            for margin in (-0.05, -0.02, 0.0, 0.01, 0.02, 0.05):
                for patience in (1, 2, 3):
                    config = {
                        "appearance_weight": appearance,
                        "anchor_weight": anchor,
                        "motion_weight": motion,
                        "max_advance": max_advance,
                        "margin": margin,
                        "patience": patience,
                    }
                    metrics = metric_summary(records_for(prepared, config))
                    robustness_metrics = [
                        metric_summary(records_for(rows, config))
                        for rows in (robustness_prepared or [])
                    ]
                    all_metrics = [metrics, *robustness_metrics]
                    candidate = (
                        -float(np.mean([row["progress_mae_fraction"] for row in all_metrics])),
                        float(np.mean([row["within_one_node"] for row in all_metrics])),
                        -float(
                            np.mean([row["early_endpoint_frame_fraction"] for row in all_metrics])
                        ),
                    )
                    if best_score is None or candidate > best_score:
                        best_score = candidate
                        best_config = config
                        best_metrics = metrics
    assert best_config is not None and best_metrics is not None
    return best_config, best_metrics


def transform_speed(sequence: StageSequence, mode: str) -> StageSequence:
    if mode == "fast":
        indices = np.arange(0, len(sequence.frames), 2)
        if indices[-1] != len(sequence.frames) - 1:
            indices = np.append(indices, len(sequence.frames) - 1)
        frames = sequence.frames[indices]
        visual = sequence.visual[indices]
    elif mode == "slow":
        frames = np.repeat(sequence.frames, 2)
        visual = np.repeat(sequence.visual, 2, axis=0)
    else:
        raise ValueError(mode)
    return StageSequence(
        task=sequence.task,
        episode=sequence.episode,
        stage_index=sequence.stage_index,
        start=sequence.start,
        end=sequence.end,
        frames=frames,
        visual=visual,
    )


def frozen_frame_drift(
    sequences: list[StageSequence],
    prototype: dict[tuple[str, int], np.ndarray],
    history_length: int,
    max_reference_span: int,
    config: dict[str, float | int],
    repeats: int = 32,
    episode_plans: dict[tuple[str, int, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    false_nodes = []
    for sequence in sequences:
        midpoint = max(1, len(sequence.frames) // 2)
        prefix_frames = sequence.frames[: midpoint + 1]
        prefix_visual = sequence.visual[: midpoint + 1]
        frozen = StageSequence(
            task=sequence.task,
            episode=sequence.episode,
            stage_index=sequence.stage_index,
            start=sequence.start,
            end=sequence.end,
            frames=np.concatenate(
                (prefix_frames, np.repeat(prefix_frames[-1:], repeats))
            ),
            visual=np.concatenate(
                (prefix_visual, np.repeat(prefix_visual[-1:], repeats, axis=0)), axis=0
            ),
        )
        episode_key = (sequence.task, sequence.episode, sequence.stage_index)
        plan = (
            episode_plans[episode_key]
            if episode_plans is not None
            else prototype[sequence.key]
        )
        components = alignment_components(
            frozen,
            plan,
            history_length,
            max_reference_span,
        )
        prediction = decode(components, **config)
        false_nodes.append(int(prediction[-1] - prediction[midpoint]))
    return {
        "repeated_updates": repeats,
        "total_false_advance_nodes": int(np.sum(false_nodes)),
        "episodes_with_false_advance_fraction": float(np.mean(np.asarray(false_nodes) > 0)),
        "max_false_advance_nodes": int(np.max(false_nodes)),
    }


def evaluate_source(
    validation: list[StageSequence],
    test: list[StageSequence],
    prototype: dict[tuple[str, int], np.ndarray],
    history_length: int,
    max_reference_span: int,
    validation_episode_plans: dict[tuple[str, int, int], np.ndarray] | None = None,
    test_episode_plans: dict[tuple[str, int, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    validation_prepared = prepare(
        validation,
        prototype,
        history_length,
        max_reference_span,
        episode_plans=validation_episode_plans,
    )
    validation_speed_prepared = [
        prepare(
            [transform_speed(sequence, mode) for sequence in validation],
            prototype,
            history_length,
            max_reference_span,
            episode_plans=validation_episode_plans,
        )
        for mode in ("slow", "fast")
    ]
    test_prepared = prepare(
        test,
        prototype,
        history_length,
        max_reference_span,
        episode_plans=test_episode_plans,
    )
    result: dict[str, Any] = {}
    selected: dict[str, dict[str, float | int]] = {}
    for variant in ("single_frame", "start_anchor", "temporal"):
        config, validation_metrics = tune(
            validation_prepared,
            variant,
            validation_speed_prepared if variant == "temporal" else None,
        )
        selected[variant] = config
        result[variant] = {
            "decoder_config": config,
            "validation": validation_metrics,
            "test": metric_summary(records_for(test_prepared, config)),
        }

    temporal_config = selected["temporal"]
    reversed_prepared = prepare(
        test,
        prototype,
        history_length,
        max_reference_span,
        plan_variant="reversed",
        episode_plans=test_episode_plans,
    )
    constant_prepared = prepare(
        test,
        prototype,
        history_length,
        max_reference_span,
        plan_variant="constant_first",
        episode_plans=test_episode_plans,
    )
    reverse_motion_prepared = prepare(
        test,
        prototype,
        history_length,
        max_reference_span,
        reverse_motion=True,
        episode_plans=test_episode_plans,
    )
    result["temporal_ablations"] = {
        "reversed_plan": metric_summary(records_for(reversed_prepared, temporal_config)),
        "constant_first_plan": metric_summary(
            records_for(constant_prepared, temporal_config)
        ),
        "reversed_local_history": metric_summary(
            records_for(reverse_motion_prepared, temporal_config)
        ),
        "frozen_frame_drift": frozen_frame_drift(
            test,
            prototype,
            history_length,
            max_reference_span,
            temporal_config,
            episode_plans=test_episode_plans,
        ),
    }
    for mode in ("slow", "fast"):
        transformed = [transform_speed(sequence, mode) for sequence in test]
        prepared = prepare(
            transformed,
            prototype,
            history_length,
            max_reference_span,
            episode_plans=test_episode_plans,
        )
        result["temporal_ablations"][f"{mode}_speed"] = metric_summary(
            records_for(prepared, temporal_config)
        )
    return result


def main() -> int:
    args = parse_args()
    if args.history_length < 2:
        raise ValueError("history-length must be at least two")
    tasks = list(args.tasks)
    train = load_sequences(args.cache_dir, tasks, "train")
    validation = load_sequences(args.cache_dir, tasks, "val")
    test = load_sequences(args.cache_dir, tasks, "test")
    _, prototype = build_plan_bank(train, args.num_nodes)
    validation_oracle_plans = {
        (sequence.task, sequence.episode, sequence.stage_index): resample(
            sequence, args.num_nodes
        )
        for sequence in validation
    }
    test_oracle_plans = {
        (sequence.task, sequence.episode, sequence.stage_index): resample(
            sequence, args.num_nodes
        )
        for sequence in test
    }
    results: dict[str, Any] = {
        "protocol": {
            "tasks": tasks,
            "num_nodes": args.num_nodes,
            "history_length": args.history_length,
            "frame_stride": 8,
            "history_span_seconds": (args.history_length - 1) * 8 / 20,
            "oracle_current_stage": True,
            "real_stage_start_anchor": True,
            "frame_or_time_input_to_aligner": False,
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "test_sequences": len(test),
            "plan_sources": {
                "oracle_same_episode_gt": "held-out stage's own future video; privileged upper bound",
                "cross_episode_prototype": "mean plan from 20 train episodes per task",
            },
        }
    }
    results["oracle_same_episode_gt"] = evaluate_source(
        validation,
        test,
        prototype,
        args.history_length,
        args.max_reference_span,
        validation_oracle_plans,
        test_oracle_plans,
    )
    results["cross_episode_prototype"] = evaluate_source(
        validation,
        test,
        prototype,
        args.history_length,
        args.max_reference_span,
    )
    results["fixed_clock"] = {
        "uses_elapsed_time": True,
        "test": metric_summary(fixed_clock_predictions(train, test, args.num_nodes)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
