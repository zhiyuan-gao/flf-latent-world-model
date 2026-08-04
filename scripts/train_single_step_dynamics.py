#!/usr/bin/env python3
"""Train the t-to-t+4 V-JEPA2 visual dynamics gate."""

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
    CachedSingleStepDynamicsWindows,
    collate_single_step_dynamics_windows,
)
from dynamics.evaluation import (  # noqa: E402
    evaluate_single_step_proprio_model,
    merge_single_step_evaluations,
)
from dynamics.losses import single_step_proprio_loss, single_step_visual_loss  # noqa: E402
from dynamics.model import SingleStepProprioACPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/single_step_dynamics/seed_0",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Always run max-epochs, while still tracking the best validation score.",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "If supplied, save only epoch_NNN.pt at these epochs and do not write "
            "best.pt/last.pt. Passing the flag with no values disables checkpoints."
        ),
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        help="Cap optimizer steps per epoch for a short throughput/memory smoke run.",
    )
    parser.add_argument("--model-dim", type=int, default=960)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--proprio-weight", type=float, default=0.005)
    parser.add_argument(
        "--no-proprio-target",
        action="store_true",
        help=(
            "Keep current proprioception as an input condition, but remove the "
            "future-proprio prediction head and loss."
        ),
    )
    parser.add_argument(
        "--no-proprio-input",
        action="store_true",
        help=(
            "Remove the current proprioceptive input token. Requires "
            "--no-proprio-target for a completely proprio-free model."
        ),
    )
    parser.add_argument("--train-diagnostic-size", type=int, default=2048)
    parser.add_argument("--gradient-audit-batch-size", type=int, default=2)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the no-update gradient audit and exit before optimization.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Evaluate the locked test split after training. Off by default.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RankStridedSampler(Sampler[int]):
    """Shard evaluation without padding or duplicated samples."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        return max((self.size - self.rank + self.world_size - 1) // self.world_size, 0)


def setup_distributed(device_arg: str) -> tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device(f"cuda:{local_rank}"), True, rank, local_rank, world_size
    return torch.device(device_arg), False, 0, 0, 1


def fixed_task_balanced_indices(
    dataset: CachedSingleStepDynamicsWindows,
    maximum: int,
    seed: int,
) -> list[int]:
    if maximum < 1:
        raise ValueError("train diagnostic size must be positive")
    by_task: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.records):
        by_task.setdefault(str(row["task"]), []).append(index)
    rng = random.Random(f"single-step-diagnostic:{seed}")
    quota = maximum // len(by_task)
    remainder = maximum % len(by_task)
    selected: list[int] = []
    for task_index, (task, indices) in enumerate(sorted(by_task.items())):
        values = list(indices)
        rng.shuffle(values)
        task_quota = quota + int(task_index < remainder)
        selected.extend(values[: min(task_quota, len(values))])
    return sorted(selected)


def estimate_dynamic_threshold(loader: DataLoader) -> float:
    values = []
    for batch in loader:
        reduce_dims = tuple(range(1, batch["target"].ndim))
        change = torch.square(batch["target"] - batch["current"]).mean(dim=reduce_dims)
        values.append(change)
    if not values:
        raise RuntimeError("Cannot estimate a motion threshold from an empty loader")
    return float(torch.cat(values).median())


def shared_gradient_audit(
    model: SingleStepProprioACPredictor,
    batch: dict[str, object],
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    proprio_weight: float,
    audit_batch_size: int,
    predict_state: bool = True,
    use_proprio_condition: bool = True,
) -> dict[str, float]:
    model.train()
    size = min(int(audit_batch_size), len(batch["current"]))
    if size < 1:
        raise ValueError("gradient audit batch is empty")
    current = batch["current"][:size].to(device)
    target = batch["target"][:size].to(device)
    actions = batch["actions"][:size].to(device)
    anchor_state = (
        batch["anchor_state"][:size].to(device) if use_proprio_condition else None
    )
    outputs = model(current, actions, anchor_state)
    if predict_state:
        target_state = batch["target_state"][:size].to(device)
        losses = single_step_proprio_loss(
            outputs,
            target,
            target_state,
            state_mean,
            state_std,
            proprio_weight=proprio_weight,
        )
    else:
        losses = single_step_visual_loss(outputs, target)
    shared = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("visual_output")
        and not name.startswith("state_output")
        and parameter.requires_grad
    ]
    visual_gradients = torch.autograd.grad(
        losses["visual"], shared, retain_graph=True, allow_unused=True
    )
    visual_norm = torch.sqrt(
        sum(
            gradient.float().square().sum()
            for gradient in visual_gradients
            if gradient is not None
        )
    )
    del visual_gradients
    if predict_state:
        state_gradients = torch.autograd.grad(
            losses["state"], shared, allow_unused=True
        )
        state_norm = torch.sqrt(
            sum(
                gradient.float().square().sum()
                for gradient in state_gradients
                if gradient is not None
            )
        )
        del state_gradients
        state_loss = float(losses["state"].detach())
        weighted_ratio = float(
            proprio_weight * state_norm / visual_norm.clamp_min(1e-12)
        )
    else:
        state_norm = torch.zeros((), device=device)
        state_loss = 0.0
        weighted_ratio = 0.0
    del outputs
    model.zero_grad(set_to_none=True)
    return {
        "batch_size": size,
        "visual_loss": float(losses["visual"].detach()),
        "state_loss": state_loss,
        "visual_shared_grad_norm": float(visual_norm),
        "state_shared_grad_norm": float(state_norm),
        "weighted_state_to_visual_grad_ratio": weighted_ratio,
        "proprio_weight": float(proprio_weight if predict_state else 0.0),
        "future_proprio_supervision": float(predict_state),
        "current_proprio_condition": float(use_proprio_condition),
    }


def gather_evaluation(
    local: dict[str, object],
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict[str, object] | None:
    if not distributed:
        return local
    rows: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(rows, local)
    if rank != 0:
        return None
    return merge_single_step_evaluations([row for row in rows if row is not None])


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.max_epochs < 1 or args.min_epochs < 1:
        raise ValueError("batch and epoch settings must be positive")
    if args.min_epochs > args.max_epochs:
        raise ValueError("min-epochs cannot exceed max-epochs")
    if args.early_stop_patience < 1:
        raise ValueError("early-stop-patience must be positive")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch < 1:
        raise ValueError("max-steps-per-epoch must be positive")
    if args.checkpoint_epochs is not None:
        invalid = [
            epoch
            for epoch in args.checkpoint_epochs
            if epoch < 1 or epoch > args.max_epochs
        ]
        if invalid:
            raise ValueError(
                "checkpoint epochs must be within [1, max-epochs]: "
                f"{invalid}"
            )
    if args.proprio_weight < 0:
        raise ValueError("proprio-weight cannot be negative")
    if args.no_proprio_input and not args.no_proprio_target:
        raise ValueError("--no-proprio-input requires --no-proprio-target")
    predict_state = not args.no_proprio_target
    use_proprio_condition = not args.no_proprio_input
    effective_proprio_weight = args.proprio_weight if predict_state else 0.0
    if predict_state:
        architecture = "single_step_visual_proprio_t4_v1"
    elif use_proprio_condition:
        architecture = "single_step_visual_only_proprio_conditioned_t4_v1"
    else:
        architecture = "single_step_visual_only_no_proprio_t4_v1"

    device, distributed, rank, local_rank, world_size = setup_distributed(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    is_primary = rank == 0
    set_seed(args.seed + rank)
    local_batch = args.batch_size // world_size
    feature_root = args.feature_root or (
        args.manifest_dir / "features/vjepa2_native_256"
    )
    if is_primary:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    train_data = CachedSingleStepDynamicsWindows(
        args.manifest_dir, feature_root, "train", max_cached_episodes=512
    )
    val_data = CachedSingleStepDynamicsWindows(
        args.manifest_dir, feature_root, "val", max_cached_episodes=512
    )
    test_data = (
        CachedSingleStepDynamicsWindows(
            args.manifest_dir, feature_root, "test", max_cached_episodes=512
        )
        if args.run_test
        else None
    )
    diagnostic_indices = fixed_task_balanced_indices(
        train_data, args.train_diagnostic_size, args.seed
    )
    diagnostic_data = Subset(train_data, diagnostic_indices)
    if is_primary:
        (args.output_dir / "train_diagnostic_indices.json").write_text(
            json.dumps(diagnostic_indices, indent=2) + "\n"
        )

    train_loader_kwargs = {
        "batch_size": local_batch,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_single_step_dynamics_windows,
    }
    full_steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    steps_per_epoch = (
        min(full_steps_per_epoch, args.max_steps_per_epoch)
        if args.max_steps_per_epoch is not None
        else full_steps_per_epoch
    )
    max_steps = steps_per_epoch * args.max_epochs
    train_sampler = WeightedRandomSampler(
        train_data.sample_weights,
        num_samples=max_steps * local_batch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed + 1009 * rank),
    )
    train_loader = DataLoader(
        train_data,
        sampler=train_sampler,
        drop_last=True,
        **train_loader_kwargs,
    )
    eval_loader_kwargs = {
        **train_loader_kwargs,
        "batch_size": local_batch if distributed else args.batch_size,
    }

    def evaluation_loader(dataset) -> DataLoader:
        sampler = (
            RankStridedSampler(len(dataset), rank, world_size) if distributed else None
        )
        return DataLoader(
            dataset,
            shuffle=False,
            sampler=sampler,
            **eval_loader_kwargs,
        )

    diagnostic_loader = evaluation_loader(diagnostic_data)
    val_loader = evaluation_loader(val_data)
    test_loader = evaluation_loader(test_data) if test_data is not None else None

    if is_primary:
        threshold_loader = DataLoader(
            diagnostic_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_single_step_dynamics_windows,
        )
        dynamic_threshold = estimate_dynamic_threshold(threshold_loader)
    else:
        dynamic_threshold = 0.0
    threshold_tensor = torch.tensor(dynamic_threshold, device=device)
    if distributed:
        dist.broadcast(threshold_tensor, src=0)
    dynamic_threshold = float(threshold_tensor.item())

    example = train_data[0]
    if tuple(example["current"].shape) != (16, 16, 1408):
        raise ValueError(
            "Single-step dynamics requires native [16, 16, 1408] V-JEPA2 features"
        )
    if tuple(example["actions"].shape) != (4, 12):
        raise ValueError("Single-step dynamics requires four ordered 12-D actions")
    if tuple(example["anchor_state"].shape) != (16,):
        raise ValueError("Single-step dynamics requires 16-D PandaOmron state")

    model = SingleStepProprioACPredictor(
        feature_dim=1408,
        action_dim=12,
        state_dim=16,
        grid_size=16,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        action_steps=4,
        dropout=args.dropout,
        predict_state=predict_state,
        use_proprio_condition=use_proprio_condition,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    state_mean = torch.as_tensor(train_data.state_mean, device=device)
    state_std = torch.as_tensor(train_data.state_std, device=device)

    audit_batch = next(iter(train_loader))
    local_audit = shared_gradient_audit(
        model,
        audit_batch,
        device,
        state_mean,
        state_std,
        effective_proprio_weight,
        args.gradient_audit_batch_size,
        predict_state=predict_state,
        use_proprio_condition=use_proprio_condition,
    )
    audit_rows: list[dict[str, float] | None] = (
        [None] * world_size if distributed else [local_audit]
    )
    if distributed:
        dist.all_gather_object(audit_rows, local_audit)
    audit = {
        key: float(np.mean([row[key] for row in audit_rows if row is not None]))
        for key in local_audit
    }
    if is_primary:
        (args.output_dir / "gradient_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        print(
            f"Single-step predictor ({'joint target' if predict_state else 'visual-only target'}): "
            f"{parameters / 1e6:.2f}M params; "
            f"train={len(train_data)}, val={len(val_data)}, "
            f"steps/epoch={steps_per_epoch}, "
            f"gradient_ratio="
            f"{audit['weighted_state_to_visual_grad_ratio']:.3f}",
            flush=True,
        )
        if predict_state and not 0.10 <= audit[
            "weighted_state_to_visual_grad_ratio"
        ] <= 0.30:
            print(
                "Gradient audit is outside the target 0.10--0.30 interval; "
                "rescale proprio-weight before the formal run.",
                flush=True,
            )
    if args.audit_only:
        if distributed:
            dist.destroy_process_group()
        return 0

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda" and not args.no_fused_optimizer,
    )
    warmup_steps = max(1, int(args.warmup_ratio * max_steps))

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    use_amp = not args.no_amp and device.type == "cuda"
    train_iterator = iter(train_loader)
    history: list[dict[str, object]] = []
    running: dict[str, float] = {}
    running_count = 0
    best_score = float("inf")
    stale_epochs = 0
    stopped_early = False
    global_step = 0
    started = time.monotonic()

    for epoch in range(1, args.max_epochs + 1):
        train_model.train()
        for _ in range(steps_per_epoch):
            batch = next(train_iterator)
            current = batch["current"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            anchor_state = (
                batch["anchor_state"].to(device, non_blocking=True)
                if use_proprio_condition
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = train_model(current, actions, anchor_state)
                if predict_state:
                    target_state = batch["target_state"].to(
                        device, non_blocking=True
                    )
                    losses = single_step_proprio_loss(
                        outputs,
                        target,
                        target_state,
                        state_mean,
                        state_std,
                        proprio_weight=effective_proprio_weight,
                    )
                else:
                    losses = single_step_visual_loss(outputs, target)
            losses["loss"].backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_count = len(current)
            running_count += batch_count
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach()) * batch_count
            global_step += 1

        loss_keys = sorted(running)
        loss_stats = torch.tensor(
            [*(running[key] for key in loss_keys), float(running_count)],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.reduce(loss_stats, dst=0, op=dist.ReduceOp.SUM)
        local_train_metrics = evaluate_single_step_proprio_model(
            model,
            diagnostic_loader,
            device,
            state_mean,
            state_std,
            use_amp=use_amp,
            action_controls=True,
            dynamic_threshold=dynamic_threshold,
        )
        local_val_metrics = evaluate_single_step_proprio_model(
            model,
            val_loader,
            device,
            state_mean,
            state_std,
            use_amp=use_amp,
            action_controls=True,
            dynamic_threshold=dynamic_threshold,
        )
        train_metrics = gather_evaluation(
            local_train_metrics, distributed, rank, world_size
        )
        val_metrics = gather_evaluation(local_val_metrics, distributed, rank, world_size)

        if is_primary:
            assert train_metrics is not None and val_metrics is not None
            denominator = float(loss_stats[-1].clamp_min(1).item())
            train_loss = {
                key: float(loss_stats[index].item() / denominator)
                for index, key in enumerate(loss_keys)
            }
            score = float(val_metrics["overall"]["visual_normalized_error"])
            improved = score < best_score
            if improved:
                best_score = score
                stale_epochs = 0
            elif epoch > args.min_epochs:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_diagnostic": train_metrics,
                "validation": val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_minutes": (time.monotonic() - started) / 60,
            }
            history.append(row)
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model.config_dict(),
                "architecture": architecture,
                "encoder": "vjepa2_native_256",
                "parameters": parameters,
                "epoch": epoch,
                "global_step": global_step,
                "predicts_future_proprio": predict_state,
                "uses_current_proprio_condition": use_proprio_condition,
                "proprio_weight": effective_proprio_weight,
                "state_mean": train_data.state_mean.tolist(),
                "state_std": train_data.state_std.tolist(),
                "dynamic_threshold": dynamic_threshold,
                "gradient_audit": audit,
                "train_diagnostic": train_metrics,
                "validation": val_metrics,
            }
            if args.checkpoint_epochs is None:
                epoch_path = args.output_dir / f"epoch_{epoch:03d}.pt"
                torch.save(checkpoint, epoch_path)
                torch.save(checkpoint, args.output_dir / "last.pt")
                if improved:
                    torch.save(checkpoint, args.output_dir / "best.pt")
            elif epoch in set(args.checkpoint_epochs):
                epoch_path = args.output_dir / f"epoch_{epoch:03d}.pt"
                torch.save(checkpoint, epoch_path)
            print(
                f"epoch={epoch:02d} step={global_step:06d} "
                f"train_norm={train_metrics['overall']['visual_normalized_error']:.4f} "
                f"train_future={train_metrics['overall']['visual_future_closer_fraction']:.3f} "
                f"val_norm={score:.4f} "
                f"val_future={val_metrics['overall']['visual_future_closer_fraction']:.3f} "
                f"val_action_shuffle={val_metrics['overall']['correct_better_than_shuffle_fraction']:.3f}",
                flush=True,
            )
            if (
                not args.disable_early_stopping
                and epoch >= args.min_epochs
                and stale_epochs >= args.early_stop_patience
            ):
                stopped_early = True
        stop_tensor = torch.tensor(int(stopped_early), device=device)
        if distributed:
            dist.broadcast(stop_tensor, src=0)
        stopped_early = bool(stop_tensor.item())
        running = {}
        running_count = 0
        if stopped_early:
            break

    test_metrics = None
    if test_loader is not None:
        best_path = args.output_dir / "best.pt"
        if not best_path.exists():
            raise RuntimeError(
                "--run-test requires best.pt; omit --checkpoint-epochs for this run"
            )
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state"])
        local_test = evaluate_single_step_proprio_model(
            model,
            test_loader,
            device,
            state_mean,
            state_std,
            use_amp=use_amp,
            action_controls=True,
            dynamic_threshold=dynamic_threshold,
        )
        test_metrics = gather_evaluation(local_test, distributed, rank, world_size)

    if is_primary:
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        )
        summary = {
            "architecture": architecture,
            "encoder": "vjepa2_native_256",
            "seed": args.seed,
            "parameters": parameters,
            "train_windows": len(train_data),
            "validation_windows": len(val_data),
            "dropped_missing_train_features": train_data.dropped_missing_features,
            "dropped_missing_validation_features": val_data.dropped_missing_features,
            "global_batch_size": args.batch_size,
            "world_size": world_size,
            "full_steps_per_epoch": full_steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "completed_epochs": len(history),
            "completed_steps": global_step,
            "best_visual_normalized_error": best_score,
            "stopped_early": stopped_early,
            "predicts_future_proprio": predict_state,
            "uses_current_proprio_condition": use_proprio_condition,
            "proprio_weight": effective_proprio_weight,
            "gradient_audit": audit,
            "dynamic_threshold": dynamic_threshold,
            "checkpoint_epochs": args.checkpoint_epochs,
            "best_checkpoint": (
                str((args.output_dir / "best.pt").resolve())
                if (args.output_dir / "best.pt").exists()
                else None
            ),
            "elapsed_minutes": (time.monotonic() - started) / 60,
            "peak_cuda_memory_gib_rank0": peak_memory,
            "test": test_metrics,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
