#!/usr/bin/env python3
"""Train a CheckVLA-style rolling predictor on offline RoboCasa demonstrations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import (  # noqa: E402
    CachedRollingWindows,
    collate_rolling_windows,
)
from dynamics.evaluation import evaluate_rolling_model  # noqa: E402
from dynamics.losses import rolling_huber_loss  # noqa: E402
from dynamics.model import CheckVLARollingPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Warm-start model weights from a predictor checkpoint. Optimizer and "
            "scheduler state are intentionally reset because legacy checkpoints do "
            "not contain them."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-ids",
        help="Comma-separated CUDA device ids for data parallelism, e.g. 0,1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--teacher-steps", type=int, default=60000)
    parser.add_argument("--self-rollout-steps", type=int, default=20000)
    parser.add_argument("--self-rollout-horizon", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=960)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--max-val-batches", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument(
        "--save-step-checkpoints",
        action="store_true",
        help="Retain an immutable step_XXXXXX.pt snapshot at every evaluation.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RankStridedSampler(Sampler[int]):
    """Deterministically shard evaluation without padding or duplicates."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        return max((self.size - self.rank + self.world_size - 1) // self.world_size, 0)


def merge_metric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Merge already-averaged RollingMetricAccumulator outputs exactly."""

    count = sum(int(row["count"]) for row in rows)
    if count < 1:
        raise RuntimeError("Cannot merge empty evaluation metrics")

    def weighted_vector(key: str) -> np.ndarray:
        total = sum(
            np.asarray(row[key], dtype=np.float64) * int(row["count"])
            for row in rows
        )
        return total / count

    mse = weighted_vector("mse_per_horizon")
    persistence = weighted_vector("persistence_mse_per_horizon")
    cosine = weighted_vector("delta_cosine_per_horizon")
    nearest = weighted_vector("endpoint_nearest_future_histogram")
    normalized = mse / np.maximum(persistence, 1e-8)
    result: dict[str, object] = {
        "count": count,
        "mse_per_horizon": mse.tolist(),
        "persistence_mse_per_horizon": persistence.tolist(),
        "normalized_error_per_horizon": normalized.tolist(),
        "mean_normalized_error": float(normalized.mean()),
        "endpoint_normalized_error": float(normalized[-1]),
        "delta_cosine_per_horizon": cosine.tolist(),
        "endpoint_delta_cosine": float(cosine[-1]),
        "endpoint_nearest_future_histogram": nearest.tolist(),
        "endpoint_retrieval_accuracy": float(nearest[-1]),
    }
    control_keys = (
        "endpoint_correct_better_zero_fraction",
        "endpoint_correct_better_shuffle_fraction",
        "endpoint_zero_action_prediction_delta_mse",
        "endpoint_shuffled_action_prediction_delta_mse",
    )
    if all(key in rows[0] for key in control_keys):
        for key in control_keys:
            result[key] = float(
                sum(float(row[key]) * int(row["count"]) for row in rows) / count
            )
    return result


def merge_evaluations(rows: list[dict[str, object]]) -> dict[str, object]:
    tasks = sorted(
        {
            task
            for row in rows
            for task in dict(row["by_task"]).keys()
        }
    )
    return {
        "overall": merge_metric_rows([dict(row["overall"]) for row in rows]),
        "by_task": {
            task: merge_metric_rows(
                [dict(dict(row["by_task"])[task]) for row in rows if task in dict(row["by_task"])]
            )
            for task in tasks
        },
    }


def gather_evaluation(
    local_metrics: dict[str, object],
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict[str, object] | None:
    if not distributed:
        return local_metrics
    gathered: list[dict[str, object] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    dist.gather_object(local_metrics, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None and all(row is not None for row in gathered)
    return merge_evaluations([row for row in gathered if row is not None])


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.teacher_steps < 0 or args.self_rollout_steps < 0:
        raise ValueError("training step counts cannot be negative")
    total_steps = args.teacher_steps + args.self_rollout_steps
    if total_steps < 1:
        raise ValueError("at least one training step is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if args.device_ids:
            raise ValueError("torchrun DDP and --device-ids cannot be used together")
        if not torch.cuda.is_available() or world_size > torch.cuda.device_count():
            raise RuntimeError("DDP requires one available CUDA device per process")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    is_primary = rank == 0
    set_seed(args.seed)
    device_ids: list[int] = []
    if args.device_ids:
        if device.type != "cuda":
            raise ValueError("--device-ids requires a CUDA primary device")
        device_ids = [int(value.strip()) for value in args.device_ids.split(",")]
        if len(device_ids) != len(set(device_ids)) or not device_ids:
            raise ValueError("--device-ids must contain unique CUDA device ids")
        primary_index = 0 if device.index is None else device.index
        if device_ids[0] != primary_index:
            raise ValueError("the first --device-ids entry must match --device")
        if max(device_ids) >= torch.cuda.device_count():
            raise ValueError("--device-ids contains an unavailable CUDA device")
    feature_root = args.feature_root or args.manifest_dir / "features" / "vjepa2"
    output_dir = args.output_dir or args.manifest_dir / "runs" / f"seed_{args.seed}"
    if is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    train_data = CachedRollingWindows(args.manifest_dir, feature_root, "train")
    val_data = CachedRollingWindows(args.manifest_dir, feature_root, "val")
    test_data = (
        None
        if args.skip_test
        else CachedRollingWindows(args.manifest_dir, feature_root, "test")
    )
    local_batch_size = args.batch_size
    if distributed:
        if args.batch_size % world_size:
            raise ValueError("global --batch-size must be divisible by DDP world size")
        local_batch_size = args.batch_size // world_size
    sampler = WeightedRandomSampler(
        train_data.sample_weights,
        num_samples=total_steps * local_batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    train_loader_kwargs = {
        "batch_size": local_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_rolling_windows,
    }
    train_loader = DataLoader(train_data, sampler=sampler, **train_loader_kwargs)
    eval_loader_kwargs = {
        **train_loader_kwargs,
        "batch_size": local_batch_size if distributed else args.batch_size,
    }
    val_sampler = (
        RankStridedSampler(len(val_data), rank, world_size) if distributed else None
    )
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        sampler=val_sampler,
        **eval_loader_kwargs,
    )
    test_sampler = (
        RankStridedSampler(len(test_data), rank, world_size)
        if distributed and test_data is not None
        else None
    )
    test_loader = (
        None
        if test_data is None
        else DataLoader(
            test_data,
            shuffle=False,
            sampler=test_sampler,
            **eval_loader_kwargs,
        )
    )
    example = train_data[0]
    horizon = int(example["actions"].shape[0])
    model = CheckVLARollingPredictor(
        feature_dim=int(example["latents"].shape[-1]),
        action_dim=int(example["actions"].shape[-1]),
        state_dim=int(example["anchor_state"].shape[-1]),
        grid_size=int(example["latents"].shape[-3]),
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        max_horizon=horizon,
        dropout=args.dropout,
    ).to(device)
    initial_checkpoint_step = None
    if args.init_checkpoint is not None:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_config = dict(initial["model_config"])
        if checkpoint_config != model.config_dict():
            raise ValueError(
                "Initial checkpoint model configuration does not match the requested "
                f"model: checkpoint={checkpoint_config}, requested={model.config_dict()}"
            )
        model.load_state_dict(initial["model_state"])
        initial_checkpoint_step = int(initial["global_step"])
        if is_primary:
            print(
                f"Warm-started from {args.init_checkpoint} "
                f"(source_step={initial_checkpoint_step}); optimizer state reset",
                flush=True,
            )
    train_model: torch.nn.Module = model
    if distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    elif len(device_ids) > 1:
        train_model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
        )
    parameters = sum(value.numel() for value in model.parameters())
    if is_primary:
        devices = f"DDP x{world_size}" if distributed else device_ids or [str(device)]
        print(
            f"CheckVLA rolling predictor: {parameters / 1e6:.2f}M params, "
            f"horizon={horizon}, devices={devices}, global_batch={args.batch_size}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda" and not args.no_fused_optimizer,
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    use_amp = not args.no_amp and device.type == "cuda"
    iterator = iter(train_loader)
    history_rows: list[dict[str, object]] = []
    best_score = float("inf")
    started = time.monotonic()
    running_loss = torch.zeros((), device=device)
    running_count = 0

    for global_step in range(total_steps):
        batch = next(iterator)
        latents = batch["latents"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        anchor_state = batch["anchor_state"].to(device, non_blocking=True)
        phase = "teacher_forcing" if global_step < args.teacher_steps else "self_rollout"
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            if phase == "teacher_forcing":
                prediction = train_model(
                    latents,
                    actions,
                    anchor_state,
                    mode="teacher_forcing",
                )
                target = latents[:, 1:]
            else:
                rollout_horizon = min(args.self_rollout_horizon, horizon)
                prediction = train_model(
                    latents,
                    actions,
                    anchor_state,
                    mode="self_rollout",
                    rollout_horizon=rollout_horizon,
                    detach_context=True,
                )
                target = latents[:, 1 : rollout_horizon + 1]
            losses = rolling_huber_loss(prediction, target)
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += losses["loss"].detach() * len(latents)
        running_count += len(latents)

        completed = global_step + 1
        should_evaluate = (
            completed == total_steps
            or completed == args.teacher_steps
            or completed % args.eval_every == 0
        )
        if not should_evaluate:
            continue
        loss_stats = torch.stack(
            (running_loss, running_loss.new_tensor(float(running_count)))
        )
        if distributed:
            dist.reduce(loss_stats, dst=0, op=dist.ReduceOp.SUM)
        eval_model = model if distributed else train_model
        local_val_metrics = evaluate_rolling_model(
            eval_model,
            val_loader,
            device,
            use_amp=use_amp,
            action_controls=False,
            max_batches=args.max_val_batches,
        )
        val_metrics = gather_evaluation(
            local_val_metrics, distributed, rank, world_size
        )
        if is_primary:
            assert val_metrics is not None
            score = float(val_metrics["overall"]["endpoint_normalized_error"])
            row = {
                "global_step": completed,
                "phase": phase,
                "train_loss_since_eval": float(
                    (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
                ),
                "val": val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_minutes": (time.monotonic() - started) / 60,
            }
            history_rows.append(row)
            (output_dir / "history.json").write_text(
                json.dumps(history_rows, indent=2) + "\n"
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model.config_dict(),
                "seed": args.seed,
                "global_step": completed,
                "phase": phase,
                "val_metrics": val_metrics,
                "parameters": parameters,
                "training": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "world_size": world_size,
                "initial_checkpoint": (
                    str(args.init_checkpoint.resolve())
                    if args.init_checkpoint is not None
                    else None
                ),
                "initial_checkpoint_step": initial_checkpoint_step,
                "method_notes": {
                    "encoder": "frozen V-JEPA2",
                    "loss": "per-step Huber",
                    "schedule": "teacher forcing followed by detached-context self-rollout",
                    "proprioception": "anchor state only; no privileged future state",
                },
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if args.save_step_checkpoints:
                torch.save(
                    checkpoint,
                    output_dir / f"step_{completed:06d}.pt",
                )
            if score < best_score:
                best_score = score
                torch.save(checkpoint, output_dir / "best.pt")
            print(
                f"step={completed:06d} phase={phase} "
                f"train={row['train_loss_since_eval']:.5f} "
                f"val_endpoint_norm={score:.4f} "
                f"val_endpoint_retrieval={val_metrics['overall']['endpoint_retrieval_accuracy']:.3f}",
                flush=True,
            )
        if distributed:
            dist.barrier(device_ids=[local_rank])
        train_model.train()
        running_loss.zero_()
        running_count = 0

    test_metrics = None
    if test_loader is not None:
        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        eval_model = model if distributed else train_model
        local_test_metrics = evaluate_rolling_model(
            eval_model,
            test_loader,
            device,
            use_amp=use_amp,
            action_controls=True,
            max_batches=args.max_test_batches,
        )
        test_metrics = gather_evaluation(
            local_test_metrics, distributed, rank, world_size
        )
    if is_primary and test_metrics is not None:
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2) + "\n"
        )
    if distributed:
        dist.barrier(device_ids=[local_rank])
    if is_primary:
        summary = {
            "seed": args.seed,
            "parameters": parameters,
            "teacher_steps": args.teacher_steps,
            "self_rollout_steps": args.self_rollout_steps,
            "self_rollout_horizon": args.self_rollout_horizon,
            "world_size": world_size,
            "initial_checkpoint": (
                str(args.init_checkpoint.resolve())
                if args.init_checkpoint is not None
                else None
            ),
            "initial_checkpoint_step": initial_checkpoint_step,
            "device_ids": (
                [f"cuda:{index}" for index in range(world_size)]
                if distributed
                else device_ids or [str(device)]
            ),
            "best_val_endpoint_normalized_error": best_score,
            "elapsed_minutes": (time.monotonic() - started) / 60,
            "peak_cuda_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else None
            ),
            "test": test_metrics["overall"] if test_metrics is not None else None,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
