#!/usr/bin/env python3
"""Evaluate a trained bake-off checkpoint on held-out episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.data import CachedDynamicsWindows, collate_dynamics_windows  # noqa: E402
from dynamics.evaluation import evaluate_dynamics_model  # noqa: E402
from dynamics.model import FourHorizonACPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT / "outputs/dynamics_bakeoff")
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoder = str(checkpoint["encoder"])
    feature_root = args.feature_root or args.manifest_dir / "features" / encoder
    dataset = CachedDynamicsWindows(args.manifest_dir, feature_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_dynamics_windows,
    )
    model = FourHorizonACPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_dynamics_model(
        model, loader, device, use_amp=not args.no_amp
    )
    output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), "overall": metrics["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
