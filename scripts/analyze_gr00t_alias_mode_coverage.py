#!/usr/bin/env python3
"""Check whether GR00T samples cover both expert chunks of alias-candidate pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
TASKS = ("PreSoakPan", "KettleBoiling", "LoadDishwasher", "RinseSinkBasin")
CONTINUOUS_DIMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias-dir", type=Path, default=PROJECT / "outputs/progress_aliasing")
    parser.add_argument(
        "--gr00t-dir", type=Path, default=PROJECT / "outputs/gr00t_chunk_multimodality"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "outputs/gr00t_chunk_multimodality/alias_mode_coverage.json"
    )
    return parser.parse_args()


def raw_expert_chunk(endpoint: dict, horizon: int = 16) -> np.ndarray:
    dataset = Path(endpoint["dataset_dir"])
    episode = int(endpoint["episode"])
    frame = int(endpoint["frame"])
    parquet = dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    df = pd.read_parquet(parquet, columns=["action"])
    raw = np.stack(df["action"].to_numpy()).astype(np.float32)
    indices = np.minimum(np.arange(frame, frame + horizon), len(raw) - 1)
    raw = raw[indices]
    # Dataset order: base4, control1, eef-pos3, eef-rot3, gripper1.
    # Policy order: eef-pos3, eef-rot3, gripper1, base4, control1.
    return np.concatenate(
        [raw[:, 5:8], raw[:, 8:11], raw[:, 11:12], raw[:, 0:4], raw[:, 4:5]],
        axis=-1,
    )


def rms(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    delta = samples[:, :, CONTINUOUS_DIMS] - reference[None, :, CONTINUOUS_DIMS]
    return np.sqrt(np.mean(np.square(delta), axis=(1, 2)))


def main() -> int:
    args = parse_args()
    rows = []
    for task in TASKS:
        pairs = [
            json.loads(line)
            for line in (args.alias_dir / task / "top_pairs.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for state_dir in sorted((args.gr00t_dir / task).glob("state_*")):
            summary = json.loads((state_dir / "summary.json").read_text())
            state = summary["state"]
            pair = pairs[int(state["alias_pair_rank"])]
            other_name = "neighbor" if state["alias_endpoint"] == "query" else "query"
            alternative = raw_expert_chunk(pair[other_name])
            saved = np.load(state_dir / "action_chunks.npz")
            samples = saved["samples"]
            own = saved["expert"]
            distance_own = rms(samples, own)
            distance_alternative = rms(samples, alternative)
            expert_gap = float(rms(own[None], alternative)[0])
            rows.append(
                {
                    "task": task,
                    "state_index": int(summary["state_index"]),
                    "episode": int(state["episode"]),
                    "frame": int(state["frame"]),
                    "subtask": state["subtask"],
                    "alias_pair_rank": int(state["alias_pair_rank"]),
                    "alias_same_subtask_label": bool(state["alias_same_subtask_label"]),
                    "expert_chunk_rms_gap": expert_gap,
                    "sample_to_own_expert_rms_min": float(distance_own.min()),
                    "sample_to_alternative_expert_rms_min": float(distance_alternative.min()),
                    "alternative_over_own_min_distance": float(
                        distance_alternative.min() / max(distance_own.min(), 1e-9)
                    ),
                    "samples_closer_to_alternative_fraction": float(
                        np.mean(distance_alternative < distance_own)
                    ),
                    "strong_multimodal_evidence": bool(summary["multimodal_evidence"]),
                    "weak_overlapping_cluster_evidence": bool(
                        summary["weak_overlapping_cluster_evidence"]
                    ),
                }
            )

    payload = {
        "states": len(rows),
        "samples_per_state": 64,
        "states_with_any_sample_closer_to_alternative": int(
            sum(row["samples_closer_to_alternative_fraction"] > 0 for row in rows)
        ),
        "overall_samples_closer_to_alternative_fraction": float(
            np.mean([row["samples_closer_to_alternative_fraction"] for row in rows])
        ),
        "median_alternative_over_own_min_distance": float(
            np.median([row["alternative_over_own_min_distance"] for row in rows])
        ),
        "median_expert_chunk_rms_gap": float(
            np.median([row["expert_chunk_rms_gap"] for row in rows])
        ),
        "rows": rows,
        "interpretation": (
            "Coverage of the paired expert chunk is necessary but not sufficient evidence of a "
            "second semantic mode; cluster separation and stability are evaluated separately."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
