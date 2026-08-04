#!/usr/bin/env python3
"""Train the native V-JEPA current-state-plus-actions endpoint predictor."""

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
from torch.utils.data import DataLoader, Sampler, Subset, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import (  # noqa: E402
    CachedBlockRollingWindows,
    CachedEndpointWindows,
    CachedMultiHorizonEndpointWindows,
    collate_block_rolling_windows,
    collate_endpoint_windows,
    collate_multi_horizon_endpoint_windows,
)
from dynamics.evaluation import (  # noqa: E402
    evaluate_endpoint_model,
    evaluate_multi_horizon_endpoint_model,
)
from dynamics.losses import (  # noqa: E402
    block_rolling_ac_loss,
    endpoint_loss,
    multi_horizon_endpoint_loss,
    multi_horizon_l1_loss,
)
from dynamics.model import (  # noqa: E402
    BlockRollingACPredictor,
    CausalMultiHorizonACPredictor,
    DirectEndpointACPredictor,
)


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
        "--device",
        default="cuda:0",
        help="Single-process fallback device; torchrun assigns one CUDA device per rank.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2_250,
        help="Global optimizer-step budget. Use 625 for the sample-matched pilot.",
    )
    parser.add_argument("--eval-every-steps", type=int, default=125)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop after this many validations without lower normalized error.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Global batch size, divided evenly across DDP processes.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Data-loader workers per DDP rank; memmapped features share the OS page cache.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=960)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-weight",
        type=float,
        default=0.0,
        help=(
            "Optional extra bounded weight on spatial tokens that change by t+16. "
            "The baseline default is ordinary unweighted Huber loss."
        ),
    )
    parser.add_argument(
        "--train-episodes-per-task",
        type=int,
        help="Use a deterministic nested subset of train episodes for learning curves.",
    )
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--multi-horizon",
        action="store_true",
        help="Train causal shared t+4/t+8/t+12/t+16 endpoint supervision.",
    )
    parser.add_argument(
        "--ac-normalized-l1",
        action="store_true",
        help=(
            "Apply V-JEPA2-AC affine-free token LayerNorm to cached targets "
            "and predictions, then optimize L1 instead of raw-latent Huber."
        ),
    )
    parser.add_argument(
        "--block-rolling",
        action="store_true",
        help=(
            "Train shared t->t+4 transitions with teacher forcing, differentiable "
            "self-rollout, and a learned candidate-conditioned proprio rollout."
        ),
    )
    parser.add_argument(
        "--proprio-weight",
        type=float,
        default=1.0,
        help="Weight of normalized future-state supervision for --block-rolling.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limited_loader(loader, maximum: int | None):
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


class RankStridedSampler(Sampler[int]):
    """Shard evaluation exactly, without duplicated padding samples."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        return max((self.size - self.rank + self.world_size - 1) // self.world_size, 0)


def merge_endpoint_metric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Merge already-averaged endpoint metrics using their sample counts."""

    count = sum(int(row["count"]) for row in rows)
    if count < 1:
        raise RuntimeError("Cannot merge empty endpoint metrics")

    def weighted_scalar(key: str) -> float:
        return float(
            sum(float(row[key]) * int(row["count"]) for row in rows) / count
        )

    mse = weighted_scalar("mse")
    persistence = weighted_scalar("persistence_mse")
    normalized = mse / max(persistence, 1e-8)
    result: dict[str, object] = {
        "count": count,
        "mse": mse,
        "persistence_mse": persistence,
        "normalized_error": normalized,
        "improvement_over_persistence": 1.0 - normalized,
        "delta_cosine": weighted_scalar("delta_cosine"),
    }
    control_keys = (
        "correct_better_than_zero_fraction",
        "correct_better_than_shuffle_fraction",
        "zero_action_prediction_delta_mse",
        "shuffled_action_prediction_delta_mse",
    )
    if all(all(key in row for key in control_keys) for row in rows):
        result.update({key: weighted_scalar(key) for key in control_keys})
    if all("nearest_future_histogram" in row for row in rows):
        histogram = sum(
            np.asarray(row["nearest_future_histogram"], dtype=np.float64)
            * int(row["count"])
            for row in rows
        ) / count
        result["nearest_future_histogram"] = histogram.tolist()
        result["endpoint_retrieval_accuracy"] = float(histogram[-1])
    if all("nearest_time_histogram" in row for row in rows):
        histogram = sum(
            np.asarray(row["nearest_time_histogram"], dtype=np.float64)
            * int(row["count"])
            for row in rows
        ) / count
        result["nearest_time_histogram"] = histogram.tolist()
        result["time_endpoint_retrieval_accuracy"] = float(histogram[-1])
    return result


def merge_endpoint_evaluations(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "overall": merge_endpoint_metric_rows(
            [dict(row["overall"]) for row in rows]
        )
    }
    group_names = [
        group_name
        for group_name in ("by_horizon", "by_task", "by_episode")
        if all(group_name in row for row in rows)
    ]
    for group_name in group_names:
        keys = sorted(
            {
                key
                for row in rows
                for key in dict(row[group_name]).keys()
            }
        )
        result[group_name] = {
            key: merge_endpoint_metric_rows(
                [
                    dict(dict(row[group_name])[key])
                    for row in rows
                    if key in dict(row[group_name])
                ]
            )
            for key in keys
        }
    return result


def gather_endpoint_evaluation(
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
    return merge_endpoint_evaluations([row for row in gathered if row is not None])


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.eval_every_steps < 1:
        raise ValueError("eval-every-steps must be positive")
    if args.early_stop_patience < 1:
        raise ValueError("early-stop-patience must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.ac_normalized_l1 and not args.multi_horizon:
        raise ValueError("--ac-normalized-l1 requires --multi-horizon")
    if args.block_rolling and (args.multi_horizon or args.ac_normalized_l1):
        raise ValueError("--block-rolling is a separate architecture mode")
    if args.proprio_weight < 0:
        raise ValueError("proprio-weight cannot be negative")
    if args.ac_normalized_l1 and args.dynamic_weight != 0.0:
        raise ValueError("--ac-normalized-l1 does not use dynamic token weighting")
    architecture = (
        "block_rolling_ac_nominal_proprio_v1"
        if args.block_rolling
        else (
            "causal_multihorizon_endpoint_ac_normalized_l1_v2"
            if args.ac_normalized_l1
            else (
                "causal_multihorizon_endpoint_masked_actions_v1"
                if args.multi_horizon
                else "direct_endpoint_anchor_state_v1"
            )
        )
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available() or world_size > torch.cuda.device_count():
            raise RuntimeError("DDP requires one available CUDA device per process")
        if args.batch_size % world_size:
            raise ValueError("global --batch-size must be divisible by DDP world size")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)
    set_seed(args.seed)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    is_primary = rank == 0
    local_batch_size = args.batch_size // world_size if distributed else args.batch_size
    feature_root = args.feature_root or (
        args.manifest_dir / "features" / "vjepa2_native_256"
    )
    output_dir = args.output_dir or (
        args.manifest_dir
        / "endpoint_runs"
        / "vjepa2_native_256"
        / (
            f"block_rolling_seed_{args.seed}_steps_{args.max_steps}"
            if args.block_rolling
            else (
                f"causal_multihorizon_acnorm_l1_seed_{args.seed}_steps_{args.max_steps}"
                if args.ac_normalized_l1
                else f"causal_multihorizon_seed_{args.seed}_steps_{args.max_steps}"
            )
            if args.multi_horizon
            else f"seed_{args.seed}_steps_{args.max_steps}"
        )
    )
    if is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    manifest = json.loads((args.manifest_dir / "manifest.json").read_text())
    window_stride = int(manifest["window_stride"])
    if window_stride != 4:
        raise ValueError(
            f"The first endpoint baseline is fixed to stride=4, received {window_stride}"
        )

    dataset_class = (
        CachedBlockRollingWindows
        if args.block_rolling
        else (
            CachedMultiHorizonEndpointWindows
            if args.multi_horizon
            else CachedEndpointWindows
        )
    )
    collate_fn = (
        collate_block_rolling_windows
        if args.block_rolling
        else (
            collate_multi_horizon_endpoint_windows
            if args.multi_horizon
            else collate_endpoint_windows
        )
    )
    evaluate_fn = (
        evaluate_multi_horizon_endpoint_model
        if args.multi_horizon or args.block_rolling
        else evaluate_endpoint_model
    )
    dataset_options = (
        {"normalize_representations": args.ac_normalized_l1}
        if args.multi_horizon
        else {}
    )
    train_data = dataset_class(
        args.manifest_dir,
        feature_root,
        "train",
        max_cached_episodes=512,
        **dataset_options,
    )
    val_data = dataset_class(
        args.manifest_dir,
        feature_root,
        "val",
        max_cached_episodes=512,
        **dataset_options,
    )
    test_data = (
        None
        if args.skip_test
        else dataset_class(
            args.manifest_dir,
            feature_root,
            "test",
            max_cached_episodes=512,
            **dataset_options,
        )
    )
    train_dataset = train_data
    train_weights = train_data.sample_weights
    if args.train_episodes_per_task is not None:
        if args.train_episodes_per_task < 1:
            raise ValueError("train-episodes-per-task must be positive")
        task_episodes: dict[str, list[int]] = {}
        for record in train_data.records:
            task_episodes.setdefault(str(record["task"]), []).append(
                int(record["episode"])
            )
        selected: set[tuple[str, int]] = set()
        for task, values in sorted(task_episodes.items()):
            unique = sorted(set(values))
            generator = random.Random(f"endpoint-learning-curve:{task}")
            generator.shuffle(unique)
            if args.train_episodes_per_task > len(unique):
                raise ValueError(
                    f"Requested {args.train_episodes_per_task} episodes for {task}, "
                    f"but only {len(unique)} are available"
                )
            selected.update(
                (task, episode)
                for episode in unique[: args.train_episodes_per_task]
            )
        indices = [
            index
            for index, record in enumerate(train_data.records)
            if (str(record["task"]), int(record["episode"])) in selected
        ]
        train_dataset = Subset(train_data, indices)
        train_weights = train_data.sample_weights[indices]
        if is_primary:
            print(
                f"Learning-curve subset: {args.train_episodes_per_task} episodes/task, "
                f"{len(train_dataset)} windows",
                flush=True,
            )
    sampler = WeightedRandomSampler(
        train_weights,
        num_samples=args.max_steps * local_batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    train_loader_kwargs = {
        "batch_size": local_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        **train_loader_kwargs,
    )
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

    example = train_dataset[0]
    endpoint_shape = tuple(example["current"].shape)
    if endpoint_shape != (16, 16, 1408):
        raise ValueError(
            "The native V-JEPA endpoint baseline requires [16, 16, 1408] "
            f"features, received {endpoint_shape}"
        )
    target_horizon = int(example["actions"].shape[0])
    if target_horizon != 16:
        raise ValueError(
            f"The endpoint baseline is fixed to t+16, received t+{target_horizon}"
        )
    model_class = (
        BlockRollingACPredictor
        if args.block_rolling
        else (
            CausalMultiHorizonACPredictor
            if args.multi_horizon
            else DirectEndpointACPredictor
        )
    )
    model_options = (
        {"normalize_representations": args.ac_normalized_l1}
        if args.multi_horizon
        else {}
    )
    model = model_class(
        feature_dim=int(example["current"].shape[-1]),
        action_dim=int(example["actions"].shape[-1]),
        state_dim=int(example["anchor_state"].shape[-1]),
        grid_size=int(example["current"].shape[-3]),
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        horizon=target_horizon,
        dropout=args.dropout,
        **model_options,
    ).to(device)
    train_model: torch.nn.Module = model
    if distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    parameters = sum(value.numel() for value in model.parameters())
    if is_primary:
        devices = f"DDP x{world_size}" if distributed else str(device)
        print(
            f"{'Block rolling' if args.block_rolling else ('Causal multi-horizon' if args.multi_horizon else 'Endpoint')} "
            f"predictor: {parameters / 1e6:.2f}M params, "
            f"devices={devices}, global_batch={args.batch_size}, "
            f"local_batch={local_batch_size}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda" and not args.no_fused_optimizer,
    )
    total_steps = args.max_steps
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    use_amp = not args.no_amp and device.type == "cuda"
    history_rows = []
    best_score = float("inf")
    global_step = 0
    validations_without_improvement = 0
    stopped_early = False
    started = time.monotonic()
    running: dict[str, float] = {}
    seen = 0
    train_interval_started = time.monotonic()
    train_iterator = iter(train_loader)
    train_model.train()
    for step_index in range(args.max_steps):
        batch = next(train_iterator)
        current = batch["current"].to(device, non_blocking=True)
        target_key = "targets" if args.multi_horizon or args.block_rolling else "target"
        target = batch[target_key].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        anchor_state = batch["anchor_state"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            if args.block_rolling:
                prediction = train_model(
                    current,
                    actions,
                    anchor_state,
                    teacher_context=target[:, :-1],
                    mode="joint",
                )
                future_state_targets = batch["future_state_targets"].to(
                    device, non_blocking=True
                )
                losses = block_rolling_ac_loss(
                    prediction,
                    target,
                    future_state_targets,
                    proprio_weight=args.proprio_weight,
                )
            else:
                prediction = train_model(current, actions, anchor_state)
            if args.ac_normalized_l1:
                losses = multi_horizon_l1_loss(prediction, target)
            elif args.multi_horizon:
                losses = multi_horizon_endpoint_loss(
                    prediction,
                    target,
                    current,
                    dynamic_weight=args.dynamic_weight,
                )
            elif not args.block_rolling:
                losses = endpoint_loss(
                    prediction,
                    target,
                    current,
                    dynamic_weight=args.dynamic_weight,
                )
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        batch_size = len(current)
        seen += batch_size
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach()) * batch_size
        global_step = step_index + 1
        should_validate = (
            global_step % args.eval_every_steps == 0
            or global_step == args.max_steps
        )
        if not should_validate:
            continue

        loss_keys = tuple(sorted(running))
        loss_stats = torch.tensor(
            [*(running[key] for key in loss_keys), float(seen)],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.reduce(loss_stats, dst=0, op=dist.ReduceOp.SUM)
        train_interval_seconds = time.monotonic() - train_interval_started
        local_val_metrics = evaluate_fn(
            model,
            limited_loader(val_loader, args.max_eval_batches),
            device,
            use_amp=use_amp,
            action_controls=False,
        )
        val_metrics = gather_endpoint_evaluation(
            local_val_metrics,
            distributed,
            rank,
            world_size,
        )
        if is_primary:
            assert val_metrics is not None
            score = float(val_metrics["overall"]["normalized_error"])
            train_since_validation = {
                key: float((loss_stats[index] / loss_stats[-1].clamp_min(1)).item())
                for index, key in enumerate(loss_keys)
            }
            row = {
                "global_step": global_step,
                "equivalent_epochs": global_step
                * args.batch_size
                / len(train_dataset),
                "train_since_validation": train_since_validation,
                "train_interval_seconds": train_interval_seconds,
                "train_examples_per_second": float(
                    loss_stats[-1].item() / max(train_interval_seconds, 1e-8)
                ),
                "val": val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_minutes": (time.monotonic() - started) / 60,
            }
            history_rows.append(row)
            (output_dir / "history.json").write_text(
                json.dumps(history_rows, indent=2) + "\n"
            )
            print(
                f"step={global_step:06d} "
                f"train={row['train_since_validation']['loss']:.5f} "
                f"val_norm={score:.4f} "
                f"val_cos={val_metrics['overall']['delta_cosine']:.3f}",
                flush=True,
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model.config_dict(),
                "encoder": "vjepa2_native_256",
                "architecture": architecture,
                "loss_name": (
                    "teacher_plus_rollout_l1_with_proprio_l1"
                    if args.block_rolling
                    else ("l1" if args.ac_normalized_l1 else "smooth_l1")
                ),
                "representation_normalization": (
                    "affine_free_token_layer_norm"
                    if args.ac_normalized_l1 or args.block_rolling
                    else "encoder_output"
                ),
                "seed": args.seed,
                "global_step": global_step,
                "val_metrics": val_metrics,
                "parameters": parameters,
                "world_size": world_size,
                "global_batch_size": args.batch_size,
                "local_batch_size": local_batch_size,
                "target_horizon": target_horizon,
                "supervised_horizons": [4, 8, 12, 16]
                if args.multi_horizon or args.block_rolling
                else [16],
                "window_stride": window_stride,
                "dynamic_weight": args.dynamic_weight,
                "proprio_weight": args.proprio_weight if args.block_rolling else None,
                "train_episodes_per_task": args.train_episodes_per_task,
                "training_schedule": {
                    "max_steps": args.max_steps,
                    "eval_every_steps": args.eval_every_steps,
                    "early_stop_patience": args.early_stop_patience,
                    "warmup_ratio": args.warmup_ratio,
                    "global_batch_size": args.batch_size,
                },
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if score < best_score:
                best_score = score
                validations_without_improvement = 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                validations_without_improvement += 1
                if validations_without_improvement >= args.early_stop_patience:
                    stopped_early = True
                    print(
                        "Early stopping: validation normalized error did not "
                        f"improve for {validations_without_improvement} evaluations.",
                        flush=True,
                    )
        stop_tensor = torch.tensor(
            [1 if stopped_early else 0],
            dtype=torch.int32,
            device=device,
        )
        if distributed:
            dist.broadcast(stop_tensor, src=0)
        stopped_early = bool(stop_tensor.item())
        running = {}
        seen = 0
        train_interval_started = time.monotonic()
        train_model.train()
        if stopped_early:
            break

    test_metrics = None
    if test_loader is not None:
        best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state"])
        local_test_metrics = evaluate_fn(
            model,
            test_loader,
            device,
            use_amp=use_amp,
            action_controls=True,
        )
        test_metrics = gather_endpoint_evaluation(
            local_test_metrics,
            distributed,
            rank,
            world_size,
        )
    if is_primary and test_metrics is not None:
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2) + "\n"
        )

    local_peak = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else None
    )
    peak_by_rank: list[float | None] | None = (
        [None] * world_size if distributed and is_primary else None
    )
    if distributed:
        dist.gather_object(local_peak, peak_by_rank, dst=0)
    elif is_primary:
        peak_by_rank = [local_peak]
    if is_primary:
        summary = {
            "encoder": "vjepa2_native_256",
            "architecture": architecture,
            "loss_name": (
                "teacher_plus_rollout_l1_with_proprio_l1"
                if args.block_rolling
                else ("l1" if args.ac_normalized_l1 else "smooth_l1")
            ),
            "representation_normalization": (
                "affine_free_token_layer_norm"
                if args.ac_normalized_l1 or args.block_rolling
                else "encoder_output"
            ),
            "seed": args.seed,
            "target_horizon": target_horizon,
            "supervised_horizons": [4, 8, 12, 16]
            if args.multi_horizon or args.block_rolling
            else [16],
            "window_stride": window_stride,
            "parameters": parameters,
            "world_size": world_size,
            "devices": (
                [f"cuda:{index}" for index in range(world_size)]
                if distributed
                else [str(device)]
            ),
            "global_batch_size": args.batch_size,
            "local_batch_size": local_batch_size,
            "dynamic_weight": args.dynamic_weight,
            "proprio_weight": args.proprio_weight if args.block_rolling else None,
            "train_episodes_per_task": args.train_episodes_per_task,
            "best_normalized_error": best_score,
            "completed_steps": global_step,
            "stopped_early": stopped_early,
            "training_schedule": {
                "max_steps": args.max_steps,
                "eval_every_steps": args.eval_every_steps,
                "early_stop_patience": args.early_stop_patience,
                "warmup_ratio": args.warmup_ratio,
                "global_batch_size": args.batch_size,
            },
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
            "elapsed_minutes": (time.monotonic() - started) / 60,
            "peak_cuda_memory_gib_by_rank": peak_by_rank,
            "test": test_metrics["overall"] if test_metrics is not None else None,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
