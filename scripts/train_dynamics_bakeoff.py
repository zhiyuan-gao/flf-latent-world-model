#!/usr/bin/env python3
"""Train one shared four-task action-conditioned latent dynamics predictor."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.data import CachedDynamicsWindows, collate_dynamics_windows  # noqa: E402
from dynamics.evaluation import evaluate_dynamics_model  # noqa: E402
from dynamics.losses import dynamics_loss  # noqa: E402
from dynamics.model import FourHorizonACPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("gr00t", "vjepa2"), required=True)
    parser.add_argument(
        "--manifest-dir", type=Path, default=PROJECT / "outputs/dynamics_bakeoff"
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--action-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--motion-weight", type=float, default=0.5)
    parser.add_argument(
        "--train-rollout-steps",
        type=int,
        default=2,
        help="V-JEPA 2-AC-style autoregressive training depth; evaluation always uses four.",
    )
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    feature_root = args.feature_root or args.manifest_dir / "features" / args.encoder
    output_dir = args.output_dir or args.manifest_dir / "runs" / args.encoder / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = CachedDynamicsWindows(args.manifest_dir, feature_root, "train")
    val_data = CachedDynamicsWindows(args.manifest_dir, feature_root, "val")
    test_data = None if args.skip_test else CachedDynamicsWindows(
        args.manifest_dir, feature_root, "test"
    )
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_data.sample_weights,
        num_samples=len(train_data),
        replacement=True,
        generator=sampler_generator,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_dynamics_windows,
    }
    train_loader = DataLoader(train_data, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = (
        None if test_data is None else DataLoader(test_data, shuffle=False, **loader_kwargs)
    )

    example = train_data[0]
    grid_size = int(example["history"].shape[-3])
    feature_dim = int(example["history"].shape[-1])
    action_dim = int(example["actions"].shape[-1])
    model = FourHorizonACPredictor(
        feature_dim=feature_dim,
        action_dim=action_dim,
        grid_size=grid_size,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        action_depth=args.action_depth,
        dropout=args.dropout,
    ).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    print(f"Predictor parameters: {parameters / 1e6:.2f}M", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = min(
        len(train_loader),
        args.max_train_steps if args.max_train_steps is not None else len(train_loader),
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_scale,
    )
    use_amp = not args.no_amp and device.type == "cuda"
    history_rows = []
    best_score = float("inf")
    global_step = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        running: dict[str, float] = {}
        seen = 0
        for step, batch in enumerate(train_loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            history = batch["history"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                predictions = model(
                    history,
                    future,
                    actions,
                    rollout_steps=args.train_rollout_steps,
                )
                losses = dynamics_loss(
                    predictions,
                    future,
                    history[:, -1],
                    motion_weight=args.motion_weight,
                )
            losses["loss"].backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_size = len(history)
            seen += batch_size
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach()) * batch_size
            global_step += 1

        val_metrics = evaluate_dynamics_model(
            model,
            limited_loader(val_loader, args.max_eval_batches),
            device,
            use_amp=use_amp,
            action_controls=False,
        )
        score = float(val_metrics["overall"]["mean_normalized_error"])
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {key: value / max(seen, 1) for key, value in running.items()},
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.monotonic() - started) / 60,
        }
        history_rows.append(row)
        (output_dir / "history.json").write_text(json.dumps(history_rows, indent=2) + "\n")
        print(
            f"epoch={epoch:02d} train={row['train']['loss']:.5f} "
            f"val_norm={score:.4f} val_retrieval="
            f"{val_metrics['overall']['mean_retrieval_accuracy']:.3f}",
            flush=True,
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model.config_dict(),
            "encoder": args.encoder,
            "seed": args.seed,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "parameters": parameters,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")

    test_metrics = None
    if test_loader is not None:
        best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state"])
        test_metrics = evaluate_dynamics_model(
            model,
            test_loader,
            device,
            use_amp=use_amp,
            action_controls=True,
        )
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2) + "\n"
        )

    summary = {
        "encoder": args.encoder,
        "seed": args.seed,
        "parameters": parameters,
        "best_mean_normalized_error": best_score,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "elapsed_minutes": (time.monotonic() - started) / 60,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
        "test_mean_normalized_error": (
            test_metrics["overall"]["mean_normalized_error"]
            if test_metrics is not None
            else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
