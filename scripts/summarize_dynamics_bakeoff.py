#!/usr/bin/env python3
"""Aggregate three-seed held-out metrics without comparing raw latent losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_KEYS = (
    "mean_normalized_error",
    "improvement_over_persistence",
    "mean_normalized_motion",
    "mean_retrieval_accuracy",
    "correct_better_than_zero_fraction",
    "correct_better_than_shuffle_fraction",
    "zero_action_prediction_delta_mse",
    "shuffled_action_prediction_delta_mse",
)
PAIRED_KEYS = (
    "mean_normalized_error",
    "mean_normalized_motion",
    "mean_retrieval_accuracy",
    "correct_better_than_shuffle_fraction",
    "shuffled_action_prediction_delta_mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT / "outputs/dynamics_bakeoff/runs",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, object] = {"seeds": args.seeds, "encoders": {}}
    missing = []
    loaded: dict[str, dict[int, dict[str, object]]] = {"gr00t": {}, "vjepa2": {}}
    for encoder in ("gr00t", "vjepa2"):
        rows = []
        for seed in args.seeds:
            path = args.runs_root / encoder / f"seed_{seed}" / "test_metrics.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            complete_metrics = json.loads(path.read_text())
            loaded[encoder][seed] = complete_metrics
            metrics = complete_metrics["overall"]
            rows.append({"seed": seed, **{key: metrics[key] for key in PRIMARY_KEYS}})
        summary = {}
        if rows:
            for key in PRIMARY_KEYS:
                values = np.asarray([row[key] for row in rows], dtype=np.float64)
                summary[key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                }
        result["encoders"][encoder] = {"per_seed": rows, "summary": summary}

    paired: dict[str, object] = {}
    rng = np.random.default_rng(args.seed)
    common_seeds = [
        seed for seed in args.seeds if seed in loaded["gr00t"] and seed in loaded["vjepa2"]
    ]
    if common_seeds and all(
        loaded[encoder][seed].get("by_episode")
        for encoder in ("gr00t", "vjepa2")
        for seed in common_seeds
    ):
        for key in PAIRED_KEYS:
            differences_by_seed: dict[int, np.ndarray] = {}
            for seed in common_seeds:
                left = loaded["gr00t"][seed]["by_episode"]
                right = loaded["vjepa2"][seed]["by_episode"]
                episodes = sorted(set(left) & set(right))
                differences_by_seed[seed] = np.asarray(
                    [right[episode][key] - left[episode][key] for episode in episodes],
                    dtype=np.float64,
                )
            seed_means = np.asarray(
                [differences_by_seed[seed].mean() for seed in common_seeds]
            )
            bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
            for index in range(args.bootstrap_samples):
                sampled_seeds = rng.choice(common_seeds, len(common_seeds), replace=True)
                values = []
                for seed in sampled_seeds:
                    episode_differences = differences_by_seed[int(seed)]
                    values.append(
                        rng.choice(
                            episode_differences,
                            len(episode_differences),
                            replace=True,
                        ).mean()
                    )
                bootstrap[index] = np.mean(values)
            paired[key] = {
                "definition": (
                    "V-JEPA minus GR00T; negative favors V-JEPA"
                    if key in {"mean_normalized_error", "mean_normalized_motion"}
                    else "V-JEPA minus GR00T; positive means V-JEPA is larger"
                ),
                "per_seed_mean_difference": {
                    str(seed): float(value) for seed, value in zip(common_seeds, seed_means)
                },
                "mean_difference": float(seed_means.mean()),
                "seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else None,
                "hierarchical_bootstrap_95_ci": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "bootstrap_probability_vjepa_less": float(np.mean(bootstrap < 0.0)),
            }
    result["paired_episode_comparison"] = paired
    result["missing"] = missing
    result["complete"] = not missing
    output = args.output or args.runs_root / "bakeoff_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
