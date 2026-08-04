#!/usr/bin/env python3
"""Diagnose progress aliasing in the four RoboCasa365 MVP tasks.

The diagnostic searches for pairs of observations that are close under all
three camera views and robot proprioception, but far apart in annotated task
progress and different in their following 16-step expert action chunks.

This is an observational diagnostic, not a proof of exact state aliasing: the
dataset does not expose full object state to the policy.  It intentionally uses
the same current-only information available to the PandaOmron GR00T policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import av
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights, resnet18


MVP_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)
VIDEO_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)
CONTINUOUS_ACTION_DIMS = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10])


@dataclass
class Sample:
    task: str
    dataset_dir: str
    episode: int
    frame: int
    subtask_idx: int
    subtask: str
    subtask_stage: str
    segment_rank: int
    num_segments: int
    local_progress: float
    global_progress: float


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project / "data/robocasa365/v1.0/target/composite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "outputs/progress_aliasing",
    )
    parser.add_argument("--tasks", nargs="+", default=list(MVP_TASKS))
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--visual-neighbors", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-same", type=float, default=0.08)
    parser.add_argument("--progress-different", type=float, default=0.25)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def find_dataset_dir(task_root: Path) -> Path:
    matches = sorted(path.parent.parent for path in task_root.glob("*/lerobot/meta/info.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one lerobot dataset under {task_root}, found {matches}")
    return matches[0]


def load_task_names(dataset_dir: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with (dataset_dir / "meta/tasks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            names[int(row["task_index"])] = str(row["task"])
    return names


def choose_episodes(dataset_dir: Path, requested: int) -> list[int]:
    paths = sorted(dataset_dir.glob("data/*/episode_*.parquet"))
    episodes = np.array([int(path.stem.split("_")[-1]) for path in paths], dtype=np.int64)
    if requested >= len(episodes):
        return episodes.tolist()
    positions = np.linspace(0, len(episodes) - 1, requested).round().astype(np.int64)
    return episodes[positions].tolist()


def canonicalize_quaternions(states: np.ndarray) -> np.ndarray:
    states = states.copy()
    # Metadata stores base quaternion at 3:7 and relative EEF quaternion at 10:14.
    # q and -q denote the same rotation, so use a deterministic hemisphere.
    for start in (3, 10):
        q = states[:, start : start + 4]
        largest = np.argmax(np.abs(q), axis=1)
        signs = np.sign(q[np.arange(len(q)), largest])
        signs[signs == 0] = 1
        states[:, start : start + 4] = q * signs[:, None]
    return states


def padded_action_chunk(actions: np.ndarray, frame: int, horizon: int) -> np.ndarray:
    indices = np.minimum(np.arange(frame, frame + horizon), len(actions) - 1)
    return actions[indices]


def collect_episode_samples(
    task: str,
    dataset_dir: Path,
    episode: int,
    names: dict[int, str],
    stride: int,
    horizon: int,
) -> tuple[list[Sample], np.ndarray, np.ndarray]:
    parquet = dataset_dir / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    df = pd.read_parquet(parquet)
    subtask_indices = df["subtask_idx"].to_numpy(dtype=np.int64)
    raw_states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    raw_actions = np.stack(df["action"].to_numpy()).astype(np.float32)

    segments: list[tuple[int, int, int, str, str]] = []
    start = 0
    for end in range(1, len(df) + 1):
        if end == len(df) or subtask_indices[end] != subtask_indices[start]:
            subtask = names[int(df["annotation.human.subtask"].iloc[start])]
            stage = names[int(df["annotation.human.subtask_stage"].iloc[start])]
            terminal = subtask.strip().lower() in {"task complete", "done"} or stage.strip().lower() == "done"
            if not terminal:
                segments.append((start, end - 1, int(subtask_indices[start]), subtask, stage))
            start = end

    samples: list[Sample] = []
    states: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    for rank, (seg_start, seg_end, subtask_idx, subtask, stage) in enumerate(segments):
        frames = list(range(seg_start, seg_end + 1, stride))
        if frames[-1] != seg_end:
            frames.append(seg_end)
        denominator = max(seg_end - seg_start, 1)
        for frame in frames:
            local = (frame - seg_start) / denominator
            global_progress = (rank + local) / max(len(segments), 1)
            samples.append(
                Sample(
                    task=task,
                    dataset_dir=str(dataset_dir),
                    episode=episode,
                    frame=frame,
                    subtask_idx=subtask_idx,
                    subtask=subtask,
                    subtask_stage=stage,
                    segment_rank=rank,
                    num_segments=len(segments),
                    local_progress=float(local),
                    global_progress=float(global_progress),
                )
            )
            states.append(raw_states[frame])
            chunks.append(padded_action_chunk(raw_actions, frame, horizon))
    return samples, np.stack(states), np.stack(chunks)


def decode_selected_frames(video_path: Path, wanted: Iterable[int]) -> dict[int, np.ndarray]:
    wanted_set = set(int(value) for value in wanted)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted_set:
                decoded[index] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(wanted_set):
                    break
    missing = wanted_set - decoded.keys()
    if missing:
        raise RuntimeError(f"Missing frames {sorted(missing)[:10]} in {video_path}")
    return decoded


def video_path(dataset_dir: Path, episode: int, key: str) -> Path:
    return dataset_dir / f"videos/chunk-{episode // 1000:03d}/{key}/episode_{episode:06d}.mp4"


@torch.inference_mode()
def embed_episode_frames(
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    dataset_dir: Path,
    episode: int,
    frames: list[int],
    batch_size: int,
) -> np.ndarray:
    by_view = [decode_selected_frames(video_path(dataset_dir, episode, key), frames) for key in VIDEO_KEYS]
    tensors: list[torch.Tensor] = []
    for frame in frames:
        for view in range(len(VIDEO_KEYS)):
            tensors.append(preprocess(Image.fromarray(by_view[view][frame])))

    outputs: list[np.ndarray] = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(device, non_blocking=True)
        features = model(batch)
        features = F.normalize(features.float(), dim=-1)
        outputs.append(features.cpu().numpy())
    return np.concatenate(outputs).reshape(len(frames), len(VIDEO_KEYS), -1)


def top_visual_neighbors(
    features: np.ndarray,
    episodes: np.ndarray,
    count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    flattened = features.reshape(len(features), -1) / math.sqrt(features.shape[1])
    x = torch.from_numpy(flattened).to(device=device, dtype=torch.float16)
    ep = torch.from_numpy(episodes).to(device)
    count = min(count, len(features) - 1)
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    block = 512
    for start in range(0, len(features), block):
        query = x[start : start + block]
        scores = query @ x.T
        same_episode = ep[start : start + len(query), None] == ep[None, :]
        scores.masked_fill_(same_episode, -torch.inf)
        values, indices = torch.topk(scores, k=count, dim=1)
        all_indices.append(indices.cpu().numpy())
        all_scores.append(values.float().cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_scores)


def selected_action_rms(chunks: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = chunks[left][:, :, CONTINUOUS_ACTION_DIMS] - chunks[right][:, :, CONTINUOUS_ACTION_DIMS]
    return np.sqrt(np.mean(np.square(delta), axis=(1, 2)))


def selected_action_cosine(chunks: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = chunks[left][:, :, CONTINUOUS_ACTION_DIMS].reshape(len(left), -1)
    b = chunks[right][:, :, CONTINUOUS_ACTION_DIMS].reshape(len(left), -1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denominator, 1e-8)


def best_masked(score: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(mask, score, np.inf)
    positions = np.argmin(masked, axis=1)
    valid = np.isfinite(masked[np.arange(len(masked)), positions])
    return positions, valid


def analyze_task(
    task: str,
    samples: list[Sample],
    features: np.ndarray,
    states: np.ndarray,
    chunks: np.ndarray,
    visual_neighbors: int,
    same_progress: float,
    different_progress: float,
    device: torch.device,
) -> tuple[dict, list[dict], list[dict]]:
    episodes = np.array([sample.episode for sample in samples], dtype=np.int64)
    progress = np.array([sample.global_progress for sample in samples], dtype=np.float32)
    canonical_states = canonicalize_quaternions(states)
    state_center = np.median(canonical_states, axis=0)
    state_scale = np.quantile(canonical_states, 0.75, axis=0) - np.quantile(canonical_states, 0.25, axis=0)
    state_scale = np.maximum(state_scale, np.std(canonical_states, axis=0) * 0.25)
    state_scale = np.maximum(state_scale, 1e-4)
    state_z = np.clip((canonical_states - state_center) / state_scale, -8, 8)

    neighbor_idx, visual_similarity = top_visual_neighbors(features, episodes, visual_neighbors, device)
    visual_distance = np.maximum(1.0 - visual_similarity, 0.0)
    state_distance = np.sqrt(
        np.mean(np.square(state_z[:, None, :] - state_z[neighbor_idx]), axis=2)
    )
    visual_scale = max(float(np.median(visual_distance[:, 0])), 1e-3)
    state_scale_scalar = max(float(np.median(np.min(state_distance, axis=1))), 0.05)
    observation_distance = visual_distance / visual_scale + state_distance / state_scale_scalar

    progress_gap = np.abs(progress[:, None] - progress[neighbor_idx])
    same_mask = progress_gap <= same_progress
    different_mask = progress_gap >= different_progress
    same_pos, same_valid = best_masked(observation_distance, same_mask)
    diff_pos, diff_valid = best_masked(observation_distance, different_mask)
    comparable = same_valid & diff_valid
    rows = np.arange(len(samples))
    same_idx = neighbor_idx[rows, same_pos]
    diff_idx = neighbor_idx[rows, diff_pos]
    same_obs = observation_distance[rows, same_pos]
    diff_obs = observation_distance[rows, diff_pos]
    ratio = diff_obs / np.maximum(same_obs, 1e-6)
    same_action = selected_action_rms(chunks, rows, same_idx)
    diff_action = selected_action_rms(chunks, rows, diff_idx)
    diff_cosine = selected_action_cosine(chunks, rows, diff_idx)
    action_threshold = max(float(np.quantile(same_action[same_valid], 0.75)), 0.10)

    summary: dict[str, object] = {
        "task": task,
        "samples": len(samples),
        "episodes": len(set(episodes.tolist())),
        "frame_stride": None,
        "comparable_queries": int(comparable.sum()),
        "comparable_fraction": float(comparable.mean()),
        "calibration": {
            "visual_distance_scale": visual_scale,
            "state_distance_scale": state_scale_scalar,
            "same_progress_max": same_progress,
            "different_progress_min": different_progress,
            "action_rms_threshold": action_threshold,
        },
        "same_phase_neighbor_action_rms": {
            "median": float(np.median(same_action[same_valid])),
            "p75": float(np.quantile(same_action[same_valid], 0.75)),
            "p90": float(np.quantile(same_action[same_valid], 0.90)),
        },
        "different_phase_neighbor_action_rms": {
            "median": float(np.median(diff_action[diff_valid])),
            "p75": float(np.quantile(diff_action[diff_valid], 0.75)),
            "p90": float(np.quantile(diff_action[diff_valid], 0.90)),
        },
        "observation_distance_ratio_diff_over_same": {
            "median": float(np.median(ratio[comparable])),
            "p10": float(np.quantile(ratio[comparable], 0.10)),
            "p25": float(np.quantile(ratio[comparable], 0.25)),
        },
        "alias_rates": {},
    }
    for ratio_limit in (1.0, 1.1, 1.25, 1.5):
        alias = comparable & (ratio <= ratio_limit) & (diff_action >= action_threshold)
        summary["alias_rates"][str(ratio_limit)] = {
            "count": int(alias.sum()),
            "fraction_of_all_samples": float(alias.mean()),
            "fraction_of_comparable": float(alias.sum() / max(comparable.sum(), 1)),
        }

    alias_mask = comparable & (ratio <= 1.25) & (diff_action >= action_threshold)
    candidate_rows = np.where(alias_mask)[0]
    ranking = diff_action[candidate_rows] / np.maximum(ratio[candidate_rows], 0.1)
    candidate_rows = candidate_rows[np.argsort(-ranking)]
    pairs: list[dict] = []
    used: set[tuple[int, int]] = set()
    for query in candidate_rows:
        other = int(diff_idx[query])
        key = tuple(sorted((query, other)))
        if key in used:
            continue
        used.add(key)
        pair = {
            "rank": len(pairs),
            "query_index": int(query),
            "neighbor_index": other,
            "query": asdict(samples[query]),
            "neighbor": asdict(samples[other]),
            "progress_gap": float(abs(progress[query] - progress[other])),
            "visual_cosine": float(visual_similarity[query, diff_pos[query]]),
            "state_distance_normalized": float(state_distance[query, diff_pos[query]]),
            "observation_distance": float(diff_obs[query]),
            "same_phase_observation_distance": float(same_obs[query]),
            "observation_distance_ratio": float(ratio[query]),
            "expert_chunk_continuous_rms": float(diff_action[query]),
            "expert_chunk_continuous_cosine": float(diff_cosine[query]),
            "same_subtask_label": samples[query].subtask == samples[other].subtask,
            "same_subtask_rank": samples[query].segment_rank == samples[other].segment_rank,
        }
        pairs.append(pair)
        if len(pairs) >= 30:
            break

    summary["top_pair_composition"] = {
        "same_subtask_label": sum(bool(pair["same_subtask_label"]) for pair in pairs),
        "different_subtask_label": sum(not bool(pair["same_subtask_label"]) for pair in pairs),
    }

    # A stricter complementary check: search for long-range self-intersections
    # inside each episode. This holds kitchen layout and object instance fixed.
    feature_flat = features.reshape(len(features), -1) / math.sqrt(features.shape[1])
    within_idx = np.zeros(len(samples), dtype=np.int64)
    within_obs = np.full(len(samples), np.inf, dtype=np.float32)
    within_visual_similarity = np.full(len(samples), np.nan, dtype=np.float32)
    within_state_distance = np.full(len(samples), np.nan, dtype=np.float32)
    within_valid = np.zeros(len(samples), dtype=bool)
    for episode in np.unique(episodes):
        indices = np.where(episodes == episode)[0]
        local_visual_similarity = feature_flat[indices] @ feature_flat[indices].T
        local_visual_distance = np.maximum(1.0 - local_visual_similarity, 0.0)
        local_state_distance = np.sqrt(
            np.mean(np.square(state_z[indices, None, :] - state_z[indices][None, :, :]), axis=2)
        )
        local_observation_distance = (
            local_visual_distance / visual_scale + local_state_distance / state_scale_scalar
        )
        local_progress_gap = np.abs(progress[indices, None] - progress[indices][None, :])
        local_mask = local_progress_gap >= different_progress
        local_pos, local_valid = best_masked(local_observation_distance, local_mask)
        valid_rows = np.where(local_valid)[0]
        global_rows = indices[valid_rows]
        global_neighbors = indices[local_pos[valid_rows]]
        within_idx[global_rows] = global_neighbors
        within_obs[global_rows] = local_observation_distance[valid_rows, local_pos[valid_rows]]
        within_visual_similarity[global_rows] = local_visual_similarity[
            valid_rows, local_pos[valid_rows]
        ]
        within_state_distance[global_rows] = local_state_distance[valid_rows, local_pos[valid_rows]]
        within_valid[global_rows] = True

    within_comparable = within_valid & same_valid
    within_ratio = within_obs / np.maximum(same_obs, 1e-6)
    within_action = selected_action_rms(chunks, rows, within_idx)
    within_cosine = selected_action_cosine(chunks, rows, within_idx)
    within_same_subtask_label = np.array(
        [samples[index].subtask == samples[int(within_idx[index])].subtask for index in rows]
    )
    within_same_subtask_rank = np.array(
        [
            samples[index].segment_rank == samples[int(within_idx[index])].segment_rank
            for index in rows
        ]
    )
    summary["within_episode_long_range"] = {
        "definition": (
            "Closest observation in the same episode at least different_progress_min away, "
            "compared with the closest same-progress observation in another episode."
        ),
        "comparable_queries": int(within_comparable.sum()),
        "observation_distance_ratio_far_within_over_same_progress_cross_episode": {
            "median": float(np.median(within_ratio[within_comparable])),
            "p10": float(np.quantile(within_ratio[within_comparable], 0.10)),
            "p25": float(np.quantile(within_ratio[within_comparable], 0.25)),
        },
        "alias_rates": {},
        "nearest_far_pair_same_subtask_alias_rates": {},
        "nearest_far_pair_same_subtask_rank_alias_rates": {},
    }
    for ratio_limit in (1.0, 1.1, 1.25, 1.5):
        within_alias = (
            within_comparable
            & (within_ratio <= ratio_limit)
            & (within_action >= action_threshold)
        )
        summary["within_episode_long_range"]["alias_rates"][str(ratio_limit)] = {
            "count": int(within_alias.sum()),
            "fraction_of_all_samples": float(within_alias.mean()),
            "fraction_of_comparable": float(
                within_alias.sum() / max(within_comparable.sum(), 1)
            ),
        }
        for key, label_mask in (
            ("nearest_far_pair_same_subtask_alias_rates", within_same_subtask_label),
            ("nearest_far_pair_same_subtask_rank_alias_rates", within_same_subtask_rank),
        ):
            labeled_alias = within_alias & label_mask
            summary["within_episode_long_range"][key][str(ratio_limit)] = {
                "count": int(labeled_alias.sum()),
                "fraction_of_all_samples": float(labeled_alias.mean()),
                "fraction_of_comparable": float(
                    labeled_alias.sum() / max(within_comparable.sum(), 1)
                ),
            }

    within_alias_mask = (
        within_comparable & (within_ratio <= 1.25) & (within_action >= action_threshold)
    )
    within_candidate_rows = np.where(within_alias_mask)[0]
    within_ranking = within_action[within_candidate_rows] / np.maximum(
        within_ratio[within_candidate_rows], 0.1
    )
    within_candidate_rows = within_candidate_rows[np.argsort(-within_ranking)]
    within_pairs: list[dict] = []
    within_used: set[tuple[int, int]] = set()
    for query in within_candidate_rows:
        other = int(within_idx[query])
        key = tuple(sorted((int(query), other)))
        if key in within_used:
            continue
        within_used.add(key)
        within_pairs.append(
            {
                "rank": len(within_pairs),
                "query_index": int(query),
                "neighbor_index": other,
                "query": asdict(samples[query]),
                "neighbor": asdict(samples[other]),
                "progress_gap": float(abs(progress[query] - progress[other])),
                "visual_cosine": float(within_visual_similarity[query]),
                "state_distance_normalized": float(within_state_distance[query]),
                "observation_distance": float(within_obs[query]),
                "same_phase_cross_episode_observation_distance": float(same_obs[query]),
                "observation_distance_ratio": float(within_ratio[query]),
                "expert_chunk_continuous_rms": float(within_action[query]),
                "expert_chunk_continuous_cosine": float(within_cosine[query]),
                "same_subtask_label": samples[query].subtask == samples[other].subtask,
                "same_subtask_rank": samples[query].segment_rank == samples[other].segment_rank,
            }
        )
        if len(within_pairs) >= 30:
            break
    summary["within_episode_long_range"]["top_pair_composition"] = {
        "same_subtask_label": sum(bool(pair["same_subtask_label"]) for pair in within_pairs),
        "different_subtask_label": sum(
            not bool(pair["same_subtask_label"]) for pair in within_pairs
        ),
    }
    return summary, pairs, within_pairs


def read_frame(path: Path, frame_index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def save_pair_sheet(path: Path, pair: dict) -> None:
    rows: list[list[Image.Image]] = []
    for endpoint in (pair["query"], pair["neighbor"]):
        dataset_dir = Path(endpoint["dataset_dir"])
        rows.append(
            [
                read_frame(video_path(dataset_dir, int(endpoint["episode"]), key), int(endpoint["frame"]))
                for key in VIDEO_KEYS
            ]
        )
    width, height = rows[0][0].size
    label_height = 62
    canvas = Image.new("RGB", (width * 3, (height + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, endpoint in enumerate((pair["query"], pair["neighbor"])):
        y = row_index * (height + label_height)
        label = (
            f"ep={endpoint['episode']} frame={endpoint['frame']} "
            f"global={endpoint['global_progress']:.3f} local={endpoint['local_progress']:.3f}\n"
            f"subtask={endpoint['subtask']}"
        )
        draw.text((5, y + 3), label, fill="black")
        for view, image in enumerate(rows[row_index]):
            canvas.paste(image, (view * width, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.tasks) - set(MVP_TASKS))
    if unknown:
        raise ValueError(f"Unsupported tasks: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    preprocess = weights.transforms()

    manifest = {
        "tasks": args.tasks,
        "episodes_per_task": args.episodes_per_task,
        "frame_stride": args.frame_stride,
        "action_horizon": args.action_horizon,
        "visual_encoder": "torchvision ResNet-18 ImageNet1K v1, penultimate features",
        "views": list(VIDEO_KEYS),
        "device": str(device),
        "interpretation": (
            "Potential aliasing, not proof of identical simulator state. A pair must have a "
            "different-progress observation nearly as close as its matched-progress neighbor "
            "and a large 16-step expert-action difference."
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    all_summaries: list[dict] = []

    for task in args.tasks:
        print(f"[{task}] collecting metadata", flush=True)
        dataset_dir = find_dataset_dir(args.data_root / task)
        names = load_task_names(dataset_dir)
        episodes = choose_episodes(dataset_dir, args.episodes_per_task)
        all_samples: list[Sample] = []
        all_states: list[np.ndarray] = []
        all_chunks: list[np.ndarray] = []
        all_features: list[np.ndarray] = []

        for position, episode in enumerate(episodes):
            samples, states, chunks = collect_episode_samples(
                task,
                dataset_dir,
                episode,
                names,
                args.frame_stride,
                args.action_horizon,
            )
            frames = [sample.frame for sample in samples]
            features = embed_episode_frames(
                backbone,
                preprocess,
                device,
                dataset_dir,
                episode,
                frames,
                args.batch_size,
            )
            all_samples.extend(samples)
            all_states.append(states)
            all_chunks.append(chunks)
            all_features.append(features)
            if (position + 1) % 10 == 0 or position + 1 == len(episodes):
                print(
                    f"[{task}] embedded {position + 1}/{len(episodes)} episodes; "
                    f"samples={len(all_samples)}",
                    flush=True,
                )

        features = np.concatenate(all_features).astype(np.float32)
        states = np.concatenate(all_states).astype(np.float32)
        chunks = np.concatenate(all_chunks).astype(np.float32)
        print(f"[{task}] nearest-neighbor analysis over {len(all_samples)} samples", flush=True)
        summary, pairs, within_pairs = analyze_task(
            task,
            all_samples,
            features,
            states,
            chunks,
            args.visual_neighbors,
            args.progress_same,
            args.progress_different,
            device,
        )
        summary["frame_stride"] = args.frame_stride
        task_dir = args.output_dir / task
        write_json(task_dir / "summary.json", summary)
        with (task_dir / "top_pairs.jsonl").open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        with (task_dir / "within_episode_top_pairs.jsonl").open("w", encoding="utf-8") as handle:
            for pair in within_pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        for pair in pairs[:8]:
            save_pair_sheet(task_dir / "pair_sheets" / f"pair_{pair['rank']:02d}.jpg", pair)
        for pair in within_pairs[:8]:
            save_pair_sheet(
                task_dir / "within_episode_pair_sheets" / f"pair_{pair['rank']:02d}.jpg",
                pair,
            )
        all_summaries.append(summary)
        print(
            f"[{task}] cross-episode / within-episode potential alias fractions "
            f"@ratio<=1.25: {summary['alias_rates']['1.25']['fraction_of_comparable']:.3f} / "
            f"{summary['within_episode_long_range']['alias_rates']['1.25']['fraction_of_comparable']:.3f}",
            flush=True,
        )

    write_json(args.output_dir / "summary_all_tasks.json", all_summaries)
    print(f"Saved diagnostics to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
