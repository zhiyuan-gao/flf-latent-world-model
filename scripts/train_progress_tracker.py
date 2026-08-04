#!/usr/bin/env python3
"""Train and evaluate supervised stage-level RoboCasa progress trackers."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from progress.data import CachedProgressEpisodes, collate_episodes  # noqa: E402
from progress.model import CausalMonotonicFilter, ProgressModel  # noqa: E402


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
        default=PROJECT / "outputs/progress_tracker/features_gr00t",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/progress_tracker/models",
    )
    parser.add_argument("--architectures", nargs="+", choices=("mlp", "gru"), default=["mlp", "gru"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def task_state_counts(cache_dir: Path, tasks: list[str]) -> list[int]:
    counts = []
    for task in tasks:
        manifest = json.loads((cache_dir / task / "manifest.json").read_text())
        segments = manifest["canonical_segments"]
        indices = [int(segment["index"]) for segment in segments]
        if indices != list(range(len(indices))):
            raise RuntimeError(f"{task} has a non-canonical state sequence: {indices}")
        counts.append(len(indices))
    return counts


def class_weight_matrix(
    dataset: CachedProgressEpisodes, state_counts: list[int]
) -> torch.Tensor:
    counts = torch.zeros(len(state_counts), max(state_counts), dtype=torch.float64)
    for row in dataset:
        task = int(row["task_id"])
        labels = row["labels"]
        counts[task] += torch.bincount(labels, minlength=max(state_counts)).double()
    weights = torch.zeros_like(counts, dtype=torch.float32)
    for task, num_states in enumerate(state_counts):
        valid = counts[task, :num_states].clamp_min(1)
        # Square-root inverse frequency keeps short DONE segments visible without
        # allowing them to dominate the much longer manipulation stages.
        current = torch.sqrt(valid.sum() / (num_states * valid))
        weights[task, :num_states] = current.clamp(0.25, 4.0).float()
    return weights


def classification_metrics(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    task_confusions: dict[str, np.ndarray] = {}
    total_correct = 0
    total_frames = 0
    violations = 0
    jumps = 0
    for record in records:
        task = record["task"]
        labels = record["labels"]
        predictions = record[key]
        num_states = int(record["num_states"])
        confusion = task_confusions.setdefault(
            task, np.zeros((num_states, num_states), dtype=np.int64)
        )
        for truth, predicted in zip(labels, predictions):
            confusion[int(truth), int(predicted)] += 1
        total_correct += int(np.sum(labels == predictions))
        total_frames += len(labels)
        delta = np.diff(predictions)
        violations += int(np.sum(delta < 0))
        jumps += int(np.sum(delta > 1))

    per_task = {}
    all_f1 = []
    for task, confusion in task_confusions.items():
        f1_values = []
        for state in range(len(confusion)):
            tp = confusion[state, state]
            fp = confusion[:, state].sum() - tp
            fn = confusion[state, :].sum() - tp
            denominator = 2 * tp + fp + fn
            f1_values.append(float(2 * tp / denominator) if denominator else 0.0)
        accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
        per_task[task] = {
            "frame_accuracy": accuracy,
            "macro_f1": float(np.mean(f1_values)),
            "per_state_f1": f1_values,
            "confusion": confusion.tolist(),
        }
        all_f1.extend(f1_values)
    return {
        "frame_accuracy": float(total_correct / max(total_frames, 1)),
        "macro_f1": float(np.mean(all_f1)),
        "monotonic_regressions": violations,
        "forward_jumps": jumps,
        "frames": total_frames,
        "per_task": per_task,
    }


def event_metrics(records: list[dict[str, Any]], key: str, fps: float = 20.0) -> dict[str, Any]:
    errors = []
    detected = 0
    expected = 0
    done_tp = done_fp = done_fn = 0
    done_episode_detected = 0
    for record in records:
        labels = record["labels"]
        predictions = record[key]
        frames = record["frames"]
        num_states = int(record["num_states"])
        done = num_states - 1
        done_tp += int(np.sum((predictions == done) & (labels == done)))
        done_fp += int(np.sum((predictions == done) & (labels != done)))
        done_fn += int(np.sum((predictions != done) & (labels == done)))
        if np.any(predictions == done):
            done_episode_detected += 1

        for state in range(1, num_states):
            expected += 1
            truth_positions = np.flatnonzero(labels >= state)
            predicted_positions = np.flatnonzero(predictions >= state)
            if not len(truth_positions) or not len(predicted_positions):
                continue
            detected += 1
            truth_frame = int(frames[truth_positions[0]])
            predicted_frame = int(frames[predicted_positions[0]])
            errors.append(predicted_frame - truth_frame)

    precision_denominator = done_tp + done_fp
    recall_denominator = done_tp + done_fn
    precision = done_tp / precision_denominator if precision_denominator else 0.0
    recall = done_tp / recall_denominator if recall_denominator else 0.0
    done_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors_array = np.asarray(errors, dtype=np.float64)
    return {
        "boundary_recall": detected / max(expected, 1),
        "boundary_mae_seconds": float(np.mean(np.abs(errors_array)) / fps)
        if len(errors_array)
        else None,
        "boundary_signed_delay_seconds": float(np.mean(errors_array) / fps)
        if len(errors_array)
        else None,
        "early_boundary_fraction": float(np.mean(errors_array < 0)) if len(errors_array) else None,
        "late_boundary_fraction": float(np.mean(errors_array > 0)) if len(errors_array) else None,
        "done_frame_precision": float(precision),
        "done_frame_recall": float(recall),
        "done_frame_f1": float(done_f1),
        "done_episode_recall": done_episode_detected / max(len(records), 1),
    }


@torch.inference_mode()
def collect_emissions(
    model: ProgressModel, loader: DataLoader, device: torch.device
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        visual = batch["visual"].to(device)
        state = batch["state"].to(device)
        task_id = batch["task_id"].to(device)
        logits, _ = model(visual, state, task_id)
        logits = logits.float().cpu().numpy()
        for index, length_tensor in enumerate(batch["lengths"]):
            length = int(length_tensor)
            task_index = int(batch["task_id"][index])
            num_states = model.task_state_counts[task_index]
            current_logits = logits[index, :length, :num_states]
            records.append(
                {
                    "task": batch["task"][index],
                    "task_id": task_index,
                    "episode": int(batch["episode"][index]),
                    "num_states": num_states,
                    "labels": batch["labels"][index, :length].numpy(),
                    "frames": batch["frames"][index, :length].numpy(),
                    "logits": current_logits,
                    "raw": np.argmax(current_logits, axis=-1),
                }
            )
    return records


def add_filtered_predictions(
    records: list[dict[str, Any]], configs: dict[str, dict[str, float | int]]
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        decoder = CausalMonotonicFilter(int(record["num_states"]), **configs[record["task"]])
        filtered = np.asarray(
            [decoder.update(torch.from_numpy(row)).index for row in record["logits"]],
            dtype=np.int64,
        )
        result.append({**record, "filtered": filtered})
    return result


def tune_decoder_configs(
    records: list[dict[str, Any]], tasks: list[str]
) -> dict[str, dict[str, float | int]]:
    grid = [
        {
            "threshold": threshold,
            "patience": patience,
            "margin": margin,
            "done_patience": done_patience,
        }
        for threshold in (0.25, 0.35, 0.45, 0.55, 0.65)
        for patience in (1, 2, 3)
        for margin in (0.75, 1.0, 1.5)
        for done_patience in (1, 2)
    ]
    selected = {}
    for task in tasks:
        task_records = [record for record in records if record["task"] == task]
        best = None
        for index, config in enumerate(grid):
            decoded = add_filtered_predictions(task_records, {task: config})
            metrics = classification_metrics(decoded, "filtered")
            events = event_metrics(decoded, "filtered")
            boundary_mae = events["boundary_mae_seconds"]
            candidate = (
                metrics["macro_f1"],
                metrics["frame_accuracy"],
                -(boundary_mae if boundary_mae is not None else math.inf),
                -index,
                config,
            )
            if best is None or candidate > best:
                best = candidate
        selected[task] = best[-1]
    return selected


def combined_metrics(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "classification": classification_metrics(records, key),
        "events": event_metrics(records, key),
    }


def train_one(
    architecture: str,
    args: argparse.Namespace,
    datasets: dict[str, CachedProgressEpisodes],
    state_counts: list[int],
    visual_dim: int,
    state_dim: int,
) -> dict[str, Any]:
    device = torch.device(args.device)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            collate_fn=collate_episodes,
        )
        for split, dataset in datasets.items()
    }
    model = ProgressModel(
        visual_dim=visual_dim,
        state_dim=state_dim,
        task_state_counts=state_counts,
        architecture=architecture,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    weights = class_weight_matrix(datasets["train"], state_counts).to(device)
    output_dir = args.output_dir / architecture
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_score = -math.inf
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_weight = 0.0
        for batch in loaders["train"]:
            visual = batch["visual"].to(device)
            state = batch["state"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            task_id = batch["task_id"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(visual, state, task_id)
            losses = F.cross_entropy(logits[mask], labels[mask], reduction="none")
            expanded_tasks = task_id[:, None].expand_as(labels)[mask]
            sample_weights = weights[expanded_tasks, labels[mask]]
            loss = (losses * sample_weights).sum() / sample_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float((losses * sample_weights).sum().item())
            running_weight += float(sample_weights.sum().item())

        val_records = collect_emissions(model, loaders["val"], device)
        val_raw = combined_metrics(val_records, "raw")
        score = val_raw["classification"]["macro_f1"]
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_weight, 1.0),
            "val_frame_accuracy": val_raw["classification"]["frame_accuracy"],
            "val_macro_f1": score,
        }
        history.append(row)
        print(f"[{architecture}] {row}", flush=True)
        if score > best_score + 1e-5:
            best_score = score
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    val_records = collect_emissions(model, loaders["val"], device)
    decoder_configs = tune_decoder_configs(val_records, list(args.tasks))
    results = {"architecture": architecture, "best_epoch": int(best["epoch"]), "history": history}
    for split in ("val", "test"):
        records = val_records if split == "val" else collect_emissions(model, loaders[split], device)
        decoded = add_filtered_predictions(records, decoder_configs)
        results[split] = {
            "raw": combined_metrics(decoded, "raw"),
            "causal_monotonic": combined_metrics(decoded, "filtered"),
        }

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config_dict(),
        "tasks": list(args.tasks),
        "task_state_counts": state_counts,
        "decoder_config_by_task": decoder_configs,
        "feature_cache": str(args.cache_dir.resolve()),
        "results": results,
    }
    torch.save(checkpoint, output_dir / "progress_tracker.pt")
    write_json(output_dir / "metrics.json", results)
    write_json(output_dir / "decoder_config.json", decoder_configs)
    return results


def main() -> int:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    datasets = {
        split: CachedProgressEpisodes(args.cache_dir, split, args.tasks)
        for split in ("train", "val", "test")
    }
    first = datasets["train"][0]
    visual_dim = int(first["visual"].shape[-1])
    state_dim = int(first["state"].shape[-1])
    state_counts = task_state_counts(args.cache_dir, args.tasks)
    all_results = {}
    for architecture in args.architectures:
        all_results[architecture] = train_one(
            architecture,
            args,
            datasets,
            state_counts,
            visual_dim,
            state_dim,
        )
    write_json(args.output_dir / "comparison.json", all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
