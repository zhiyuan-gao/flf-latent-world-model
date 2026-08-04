#!/usr/bin/env python3
"""Evaluate a direct t+16 endpoint predictor on held-out episodes."""

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
    CachedEndpointWindows,
    CachedMultiHorizonEndpointWindows,
    collate_endpoint_windows,
    collate_multi_horizon_endpoint_windows,
)
from dynamics.evaluation import (  # noqa: E402
    evaluate_endpoint_model,
    evaluate_multi_horizon_endpoint_model,
)
from dynamics.model import (  # noqa: E402
    CausalMultiHorizonACPredictor,
    DirectEndpointACPredictor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT / "outputs/checkvla_offline_predictor",
    )
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = str(checkpoint.get("architecture", ""))
    supported = {
        "direct_endpoint_anchor_state_v1",
        "causal_multihorizon_endpoint_masked_actions_v1",
        "causal_multihorizon_endpoint_ac_normalized_l1_v2",
    }
    if architecture not in supported:
        raise ValueError(
            "Checkpoint is not the current anchor-state endpoint architecture: "
            f"{architecture or 'legacy/unknown'}"
        )
    multi_horizon = architecture.startswith("causal_multihorizon_endpoint_")
    normalize_representations = (
        architecture == "causal_multihorizon_endpoint_ac_normalized_l1_v2"
    )
    feature_root = args.feature_root or (
        args.manifest_dir / "features" / "vjepa2_native_256"
    )
    dataset_class = (
        CachedMultiHorizonEndpointWindows if multi_horizon else CachedEndpointWindows
    )
    collate_fn = (
        collate_multi_horizon_endpoint_windows
        if multi_horizon
        else collate_endpoint_windows
    )
    dataset_options = (
        {"normalize_representations": normalize_representations}
        if multi_horizon
        else {}
    )
    dataset = dataset_class(
        args.manifest_dir,
        feature_root,
        args.split,
        **dataset_options,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    model_class = (
        CausalMultiHorizonACPredictor if multi_horizon else DirectEndpointACPredictor
    )
    model = model_class(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    evaluate_fn = (
        evaluate_multi_horizon_endpoint_model if multi_horizon else evaluate_endpoint_model
    )
    metrics = evaluate_fn(
        model, loader, device, use_amp=not args.no_amp, action_controls=True
    )
    output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), "overall": metrics["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
