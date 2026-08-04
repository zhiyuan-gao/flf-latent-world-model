#!/usr/bin/env python3
"""Directly test GR00T on same-episode, same-subtask, far-progress pairs.

The script compares the checkpoint's own vision tokens, fused image-token
context, state encoding, and repeated action distributions at both endpoints.
It is intentionally restricted to the two highest-priority MVP subtasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import BatchFeature


PROJECT = Path(__file__).resolve().parents[1]
GR00T_ROOT = PROJECT / "third_party/Isaac-GR00T"
sys.path.insert(0, str(GR00T_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values  # noqa: E402


PAIR_RANKS = {
    "KettleBoiling": [0, 8, 9, 12],
    "RinseSinkBasin": [7, 13, 19, 21],
}
ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)
CONTINUOUS_DIMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT
        / "checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000",
    )
    parser.add_argument("--alias-dir", type=Path, default=PROJECT / "outputs/progress_aliasing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/gr00t_same_subtask_pair_check",
    )
    parser.add_argument("--samples-per-endpoint", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def repeat_batch(value: Any, count: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != 1:
            raise ValueError(f"Expected batch 1, got {tuple(value.shape)}")
        return value.repeat((count,) + (1,) * (value.ndim - 1))
    if isinstance(value, BatchFeature):
        return BatchFeature(data={key: repeat_batch(item, count) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: repeat_batch(item, count) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(type(value).__name__)


def clone_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, BatchFeature):
        return BatchFeature(data={key: clone_batch(item) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: clone_batch(item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(type(value).__name__)


def concatenate_actions(action_dict: dict[str, Any]) -> np.ndarray:
    arrays = []
    for key in ACTION_KEYS:
        value = action_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(arrays, axis=-1)


def normalized_observation(policy: Gr00tPolicy, raw: dict[str, Any]) -> dict[str, Any]:
    obs = {key: value for key, value in raw.items() if not key.startswith("action.")}
    obs = unsqueeze_dict_values(obs)
    for key, value in obs.items():
        if not isinstance(value, np.ndarray):
            obs[key] = np.array(value)
    return policy.apply_transforms(obs)


@torch.inference_mode()
def encode_and_maybe_sample(
    policy: Gr00tPolicy,
    raw: dict[str, Any],
    sample_count: int,
    sample_batch_size: int,
) -> dict[str, Any]:
    normalized = normalized_observation(policy, raw)
    backbone_inputs, action_inputs = policy.model.prepare_input(normalized)
    backbone = policy.model.backbone

    eagle_input = {
        key.removeprefix("eagle_"): value
        for key, value in backbone_inputs.items()
        if key.startswith("eagle_")
    }
    image_token_id = int(backbone.eagle_model.config.image_token_index)
    image_mask = eagle_input["input_ids"][0] == image_token_id

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vision_tokens = backbone.eagle_model.extract_feature(eagle_input["pixel_values"])
        fused = backbone(backbone_inputs)
        processed = policy.model.action_head.process_backbone_output(clone_batch(fused))
        state_tokens = policy.model.action_head.state_encoder(
            action_inputs["state"], action_inputs["embodiment_id"]
        )

    result: dict[str, Any] = {
        "vision_tokens": vision_tokens.float().cpu().numpy(),
        "fused_image_tokens": fused["backbone_features"][0, image_mask].float().cpu().numpy(),
        "action_context_image_tokens": processed["backbone_features"][0, image_mask]
        .float()
        .cpu()
        .numpy(),
        "state_tokens": state_tokens[0].float().cpu().numpy(),
    }
    if sample_count <= 0:
        return result

    cached_actions = BatchFeature(
        data={
            "state": action_inputs["state"],
            "embodiment_id": action_inputs["embodiment_id"],
        }
    )
    samples = []
    for start in range(0, sample_count, sample_batch_size):
        current = min(sample_batch_size, sample_count - start)
        repeated_backbone = repeat_batch(fused, current)
        repeated_actions = repeat_batch(cached_actions, current)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = policy.model.action_head.get_action(
                repeated_backbone, repeated_actions
            )["action_pred"].float()
        unnormalized = policy._get_unnormalized_action(prediction)
        samples.append(concatenate_actions(unnormalized))
    result["samples"] = np.concatenate(samples)
    result["expert"] = concatenate_actions({key: raw[key] for key in ACTION_KEYS})
    return result


def token_cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Feature shape mismatch: {left.shape} vs {right.shape}")
    left = left.reshape(-1, left.shape[-1]).astype(np.float64)
    right = right.reshape(-1, right.shape[-1]).astype(np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    return float(np.mean(1.0 - cosine))


def choose_local_control(endpoint: dict[str, Any]) -> dict[str, Any]:
    dataset = Path(endpoint["dataset_dir"])
    episode = int(endpoint["episode"])
    frame = int(endpoint["frame"])
    parquet = dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    df = pd.read_parquet(parquet, columns=["subtask_idx"])
    subtask_idx = int(df["subtask_idx"].iloc[frame])
    for offset in (-8, 8, -4, 4, -1, 1):
        candidate = frame + offset
        if 0 <= candidate < len(df) and int(df["subtask_idx"].iloc[candidate]) == subtask_idx:
            control = dict(endpoint)
            control["frame"] = candidate
            control["local_control_offset"] = offset
            return control
    raise RuntimeError(f"No same-subtask local control around episode={episode}, frame={frame}")


def continuous_rms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[..., CONTINUOUS_DIMS] - right[..., CONTINUOUS_DIMS]
    axes = tuple(range(1, delta.ndim))
    return np.sqrt(np.mean(np.square(delta), axis=axes))


def action_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    samples_left = left["samples"]
    samples_right = right["samples"]
    expert_left = left["expert"]
    expert_right = right["expert"]
    mean_left = samples_left.mean(axis=0)
    mean_right = samples_right.mean(axis=0)
    predicted_separation = float(continuous_rms(mean_left[None], mean_right[None])[0])
    within_left = float(np.mean(continuous_rms(samples_left, mean_left[None])))
    within_right = float(np.mean(continuous_rms(samples_right, mean_right[None])))
    pooled_within = (within_left + within_right) / 2
    expert_separation = float(continuous_rms(expert_left[None], expert_right[None])[0])

    predicted_delta = (mean_right[:, CONTINUOUS_DIMS] - mean_left[:, CONTINUOUS_DIMS]).reshape(-1)
    expert_delta = (expert_right[:, CONTINUOUS_DIMS] - expert_left[:, CONTINUOUS_DIMS]).reshape(-1)
    delta_cosine = float(
        np.dot(predicted_delta, expert_delta)
        / max(np.linalg.norm(predicted_delta) * np.linalg.norm(expert_delta), 1e-12)
    )
    own_distance = np.concatenate(
        [
            continuous_rms(samples_left, expert_left[None]),
            continuous_rms(samples_right, expert_right[None]),
        ]
    )
    other_distance = np.concatenate(
        [
            continuous_rms(samples_left, expert_right[None]),
            continuous_rms(samples_right, expert_left[None]),
        ]
    )
    return {
        "expert_chunk_rms_separation": expert_separation,
        "predicted_mean_chunk_rms_separation": predicted_separation,
        "mean_within_endpoint_sampling_rms": pooled_within,
        "predicted_separation_over_sampling_variation": predicted_separation
        / max(pooled_within, 1e-12),
        "predicted_vs_expert_delta_cosine": delta_cosine,
        "samples_closer_to_own_expert_fraction": float(np.mean(own_distance < other_distance)),
        "mean_sample_to_own_expert_rms": float(own_distance.mean()),
        "mean_sample_to_other_expert_rms": float(other_distance.mean()),
    }


def representation_metrics(
    left: dict[str, Any],
    right: dict[str, Any],
    left_control: dict[str, Any],
    right_control: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    for key in (
        "vision_tokens",
        "fused_image_tokens",
        "action_context_image_tokens",
        "state_tokens",
    ):
        far = token_cosine_distance(left[key], right[key])
        local_left = token_cosine_distance(left[key], left_control[key])
        local_right = token_cosine_distance(right[key], right_control[key])
        local_mean = (local_left + local_right) / 2
        result[key] = {
            "far_progress_pair_cosine_distance": far,
            "mean_8_step_local_cosine_distance": local_mean,
            "far_over_local_distance": far / max(local_mean, 1e-12),
        }
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = DATA_CONFIG_MAP["panda_omron"]
    print(f"Loading checkpoint {args.checkpoint}", flush=True)
    policy = Gr00tPolicy(
        model_path=str(args.checkpoint),
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=4,
        device=args.device,
    )

    task_results = []
    for task, ranks in PAIR_RANKS.items():
        rows = [
            json.loads(line)
            for line in (args.alias_dir / task / "within_episode_top_pairs.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        rows_by_rank = {int(row["rank"]): row for row in rows}
        selected = [rows_by_rank[rank] for rank in ranks]
        if not all(row["same_subtask_label"] and row["same_subtask_rank"] for row in selected):
            raise RuntimeError(f"Selected {task} pair is not same-subtask: {selected}")

        dataset_dir = Path(selected[0]["query"]["dataset_dir"])
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_dir,
            modality_configs=config.modality_config(),
            video_backend="opencv",
            transforms=None,
            embodiment_tag="new_embodiment",
        )
        cache: dict[tuple[int, int, bool], dict[str, Any]] = {}

        def get_encoded(endpoint: dict[str, Any], sample: bool) -> dict[str, Any]:
            key = (int(endpoint["episode"]), int(endpoint["frame"]), sample)
            if key not in cache:
                raw = dataset.get_step_data(key[0], key[1])
                cache[key] = encode_and_maybe_sample(
                    policy,
                    raw,
                    args.samples_per_endpoint if sample else 0,
                    args.sample_batch_size,
                )
            return cache[key]

        pair_results = []
        for pair in selected:
            query = pair["query"]
            neighbor = pair["neighbor"]
            query_control = choose_local_control(query)
            neighbor_control = choose_local_control(neighbor)
            print(
                f"[{task}] rank={pair['rank']} episode={query['episode']} "
                f"frames={query['frame']}/{neighbor['frame']}",
                flush=True,
            )
            encoded_query = get_encoded(query, True)
            encoded_neighbor = get_encoded(neighbor, True)
            encoded_query_control = get_encoded(query_control, False)
            encoded_neighbor_control = get_encoded(neighbor_control, False)
            actions = action_metrics(encoded_query, encoded_neighbor)
            representations = representation_metrics(
                encoded_query,
                encoded_neighbor,
                encoded_query_control,
                encoded_neighbor_control,
            )
            distinguishes = bool(
                actions["predicted_separation_over_sampling_variation"] >= 2.0
                and actions["predicted_vs_expert_delta_cosine"] >= 0.5
                and actions["samples_closer_to_own_expert_fraction"] >= 0.75
            )
            pair_results.append(
                {
                    "rank": int(pair["rank"]),
                    "episode": int(query["episode"]),
                    "frames": [int(query["frame"]), int(neighbor["frame"])],
                    "global_progress": [
                        float(query["global_progress"]),
                        float(neighbor["global_progress"]),
                    ],
                    "subtask": query["subtask"],
                    "resnet_observation_distance_ratio": float(
                        pair["observation_distance_ratio"]
                    ),
                    "local_control_frames": [
                        int(query_control["frame"]),
                        int(neighbor_control["frame"]),
                    ],
                    "representations": representations,
                    "actions": actions,
                    "gr00t_distinguishes_pair": distinguishes,
                }
            )
        task_results.append(
            {
                "task": task,
                "pairs": len(pair_results),
                "pairs_gr00t_distinguishes": int(
                    sum(row["gr00t_distinguishes_pair"] for row in pair_results)
                ),
                "results": pair_results,
            }
        )

    all_pairs = [row for task in task_results for row in task["results"]]
    summary = {
        "checkpoint": str(args.checkpoint),
        "pair_source": "same episode, same subtask label and segment, progress gap >= 0.25",
        "samples_per_endpoint": args.samples_per_endpoint,
        "pairs": len(all_pairs),
        "pairs_gr00t_distinguishes": int(
            sum(row["gr00t_distinguishes_pair"] for row in all_pairs)
        ),
        "decision_rule": (
            "predicted mean chunk separation >=2x within-input sampling variation, "
            "predicted/expert delta cosine >=0.5, and >=75% samples closer to own expert"
        ),
        "tasks": task_results,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
