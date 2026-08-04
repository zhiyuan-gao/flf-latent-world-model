#!/usr/bin/env python3
"""Evaluate saved CheckVLA-style predictor checkpoints on a complete split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import (  # noqa: E402
    CachedRollingWindows,
    collate_rolling_windows,
)
from dynamics.evaluation import evaluate_rolling_model  # noqa: E402
from dynamics.model import CheckVLARollingPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-ids",
        default="0,1",
        help="Comma-separated CUDA devices used by DataParallel; empty uses one device.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--action-controls", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_ids = [
        int(value.strip()) for value in args.device_ids.split(",") if value.strip()
    ]
    if device_ids:
        primary = 0 if device.index is None else device.index
        if device.type != "cuda" or device_ids[0] != primary:
            raise ValueError("--device-ids must begin with the selected CUDA device")
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("--device-ids must contain unique device ids")
        if max(device_ids) >= torch.cuda.device_count():
            raise ValueError("--device-ids contains an unavailable CUDA device")

    feature_root = args.feature_root or args.manifest_dir / "features" / "vjepa2"
    dataset = CachedRollingWindows(
        args.manifest_dir,
        feature_root,
        args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_rolling_windows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model_config = dict(checkpoint["model_config"])
        # Checkpoints created before condition-input normalization became
        # configurable always used LayerNorm. Preserve exact legacy evaluation.
        model_config.setdefault("normalize_condition_inputs", True)
        model = CheckVLARollingPredictor(**model_config)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        eval_model: torch.nn.Module = model
        if len(device_ids) > 1:
            eval_model = torch.nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=device_ids[0],
            )
        source_step = int(checkpoint["global_step"])
        print(
            f"Evaluating {checkpoint_path} at source_step={source_step} "
            f"on full {args.split} ({len(dataset)} windows)",
            flush=True,
        )
        metrics = evaluate_rolling_model(
            eval_model,
            loader,
            device,
            use_amp=not args.no_amp,
            action_controls=args.action_controls,
            max_batches=args.max_batches,
        )
        result = {
            "checkpoint": str(checkpoint_path.resolve()),
            "source_step": source_step,
            "source_phase": checkpoint["phase"],
            "split": args.split,
            "num_windows": len(dataset),
            "action_controls": args.action_controls,
            "metrics": metrics,
        }
        name = f"step_{source_step:06d}_{args.split}.json"
        (args.output_dir / name).write_text(json.dumps(result, indent=2) + "\n")
        overall = metrics["overall"]
        print(
            f"source_step={source_step} "
            f"endpoint_norm={overall['endpoint_normalized_error']:.4f} "
            f"endpoint_cos={overall['endpoint_delta_cosine']:.4f} "
            f"endpoint_retrieval={overall['endpoint_retrieval_accuracy']:.4f}",
            flush=True,
        )
        summaries.append(
            {
                "checkpoint": result["checkpoint"],
                "source_step": source_step,
                "source_phase": checkpoint["phase"],
                "overall": overall,
                "by_task": metrics["by_task"],
            }
        )
        del eval_model, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output_dir / f"summary_{args.split}.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
