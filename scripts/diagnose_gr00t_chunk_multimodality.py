#!/usr/bin/env python3
"""Test whether GR00T emits multiple stable 16-step action modes for one observation.

For every selected RoboCasa365 state, the script encodes the observation exactly
once and repeatedly samples only GR00T's stochastic flow-matching action head.
It then checks whether the continuous action chunks form separated, sufficiently
populated clusters.  This distinguishes genuine same-input sampling variation
from variation caused by changing the image, robot state, or language input.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import BatchFeature


PROJECT = Path(__file__).resolve().parents[1]
GR00T_ROOT = PROJECT / "third_party/Isaac-GR00T"
sys.path.insert(0, str(GR00T_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values  # noqa: E402


MVP_TASKS = ("PreSoakPan", "KettleBoiling", "LoadDishwasher", "RinseSinkBasin")
ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)
# Canonical concatenation is [eef pos 3, eef rot 3, grip 1, base 4, mode 1].
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
        "--output-dir", type=Path, default=PROJECT / "outputs/gr00t_chunk_multimodality"
    )
    parser.add_argument("--tasks", nargs="+", default=list(MVP_TASKS))
    parser.add_argument("--states-per-task", type=int, default=6)
    parser.add_argument("--samples-per-state", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=jsonable) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_states(alias_dir: Path, task: str, count: int) -> list[dict[str, Any]]:
    """Pick diverse high-ranked alias endpoints, preferring both label cases."""
    rows = read_jsonl(alias_dir / task / "top_pairs.jsonl")
    selected: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()

    def add_candidates(same_label: bool | None) -> None:
        for row in rows:
            if same_label is not None and bool(row["same_subtask_label"]) != same_label:
                continue
            for endpoint_name in ("query", "neighbor"):
                endpoint = copy.deepcopy(row[endpoint_name])
                key = (int(endpoint["episode"]), int(endpoint["frame"]))
                if key in used:
                    continue
                endpoint["alias_pair_rank"] = int(row["rank"])
                endpoint["alias_endpoint"] = endpoint_name
                endpoint["alias_same_subtask_label"] = bool(row["same_subtask_label"])
                endpoint["alias_observation_distance_ratio"] = float(
                    row["observation_distance_ratio"]
                )
                selected.append(endpoint)
                used.add(key)
                if len(selected) >= count:
                    return

    # Reserve roughly half the probes for the harder same-subtask ambiguity case.
    target_same = (count + 1) // 2
    for row in rows:
        if len(selected) >= target_same:
            break
        if not bool(row["same_subtask_label"]):
            continue
        for endpoint_name in ("query", "neighbor"):
            endpoint = copy.deepcopy(row[endpoint_name])
            key = (int(endpoint["episode"]), int(endpoint["frame"]))
            if key in used:
                continue
            endpoint.update(
                alias_pair_rank=int(row["rank"]),
                alias_endpoint=endpoint_name,
                alias_same_subtask_label=True,
                alias_observation_distance_ratio=float(row["observation_distance_ratio"]),
            )
            selected.append(endpoint)
            used.add(key)
            if len(selected) >= target_same:
                break
    add_candidates(False)
    add_candidates(None)
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} candidate states for {task}")
    return selected[:count]


def repeat_batch(value: Any, count: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != 1:
            raise ValueError(f"Expected cached batch size 1, received {tuple(value.shape)}")
        return value.repeat((count,) + (1,) * (value.ndim - 1))
    if isinstance(value, BatchFeature):
        return BatchFeature(data={key: repeat_batch(item, count) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: repeat_batch(item, count) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(f"Cannot repeat cached model value of type {type(value)}")


def concatenate_actions(action_dict: dict[str, Any]) -> np.ndarray:
    arrays = []
    for key in ACTION_KEYS:
        value = action_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(arrays, axis=-1)


@torch.inference_mode()
def sample_same_input(
    policy: Gr00tPolicy,
    raw: dict[str, Any],
    total: int,
    batch_size: int,
) -> np.ndarray:
    observations = {key: value for key, value in raw.items() if not key.startswith("action.")}
    observations = unsqueeze_dict_values(observations)
    for key, value in observations.items():
        if not isinstance(value, np.ndarray):
            observations[key] = np.array(value)
    normalized = policy.apply_transforms(observations)
    backbone_inputs, action_inputs = policy.model.prepare_input(normalized)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cached_backbone = policy.model.backbone(backbone_inputs)

    # FlowmatchingActionHead.prepare_input currently forwards the entire model
    # batch, including unbatched camera tensors. Inference only consumes these
    # two fields; keeping just them makes the repeated batch dimension explicit.
    cached_action_inputs = BatchFeature(
        data={
            "state": action_inputs["state"],
            "embodiment_id": action_inputs["embodiment_id"],
        }
    )

    samples: list[np.ndarray] = []
    for start in range(0, total, batch_size):
        current = min(batch_size, total - start)
        repeated_backbone = repeat_batch(cached_backbone, current)
        repeated_actions = repeat_batch(cached_action_inputs, current)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = policy.model.action_head.get_action(
                repeated_backbone, repeated_actions
            )["action_pred"].float()
        unnormalized = policy._get_unnormalized_action(prediction)
        samples.append(concatenate_actions(unnormalized))
    return np.concatenate(samples, axis=0)


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)


def kmeans(x: np.ndarray, clusters: int, seed: int, restarts: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_loss = np.inf
    for _ in range(restarts):
        centers = x[rng.choice(len(x), clusters, replace=False)].copy()
        labels = np.zeros(len(x), dtype=np.int64)
        for _ in range(100):
            new_labels = np.argmin(squared_distances(x, centers), axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for index in range(clusters):
                members = x[labels == index]
                centers[index] = members.mean(axis=0) if len(members) else x[rng.integers(len(x))]
        loss = np.min(squared_distances(x, centers), axis=1).sum()
        if loss < best_loss:
            best_loss = loss
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    distances = np.sqrt(np.maximum(squared_distances(x, x), 0.0))
    values = []
    for index in range(len(x)):
        own = labels == labels[index]
        own[index] = False
        a = distances[index, own].mean() if own.any() else 0.0
        other_values = [
            distances[index, labels == label].mean()
            for label in np.unique(labels)
            if label != labels[index]
        ]
        b = min(other_values)
        values.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(values))


def cluster_metrics(chunks: np.ndarray, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    continuous = chunks[:, :, CONTINUOUS_DIMS]
    x = continuous.reshape(len(continuous), -1)
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    explained = variance / max(variance.sum(), 1e-12)
    projection = centered @ vh[:2].T

    pairwise_rms = np.sqrt(np.mean((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    upper = pairwise_rms[np.triu_indices(len(x), k=1)]
    candidates = []
    label_sets: dict[int, np.ndarray] = {}
    for clusters in (2, 3, 4):
        labels = kmeans(x, clusters, seed + clusters)
        label_sets[clusters] = labels
        counts = np.bincount(labels, minlength=clusters)
        centers = np.stack([x[labels == index].mean(axis=0) for index in range(clusters)])
        within = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [(x[labels == index] - centers[index]) ** 2 for index in range(clusters)]
                    )
                )
            )
        )
        centroid_dist = np.sqrt(
            np.mean((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        )
        centroid_upper = centroid_dist[np.triu_indices(clusters, k=1)]
        score = silhouette(x, labels)
        halves = (x[::2], x[1::2])
        split_stable = False
        split_minimum_fraction = 0.0
        split_centroid_match_rms = float("inf")
        split_stability_ratio = float("inf")
        if min(map(len, halves)) >= 2 * clusters:
            split_labels = [
                kmeans(half, clusters, seed + 100 * clusters + index)
                for index, half in enumerate(halves)
            ]
            split_centers = [
                np.stack(
                    [half[half_labels == index].mean(axis=0) for index in range(clusters)]
                )
                for half, half_labels in zip(halves, split_labels)
            ]
            cross_rms = np.sqrt(
                np.mean(
                    (split_centers[0][:, None, :] - split_centers[1][None, :, :]) ** 2,
                    axis=-1,
                )
            )
            left_match, right_match = linear_sum_assignment(cross_rms)
            split_centroid_match_rms = float(cross_rms[left_match, right_match].max())
            split_minimum_fraction = float(
                min(
                    np.bincount(half_labels, minlength=clusters).min() / len(half)
                    for half, half_labels in zip(halves, split_labels)
                )
            )
            split_stability_ratio = split_centroid_match_rms / max(
                float(centroid_upper.min()), 1e-8
            )
            split_stable = bool(
                split_minimum_fraction >= 0.10 and split_stability_ratio <= 0.50
            )
        candidates.append(
            {
                "k": clusters,
                "silhouette": score,
                "cluster_counts": counts.tolist(),
                "minimum_cluster_fraction": float(counts.min() / len(x)),
                "within_cluster_rms": within,
                "minimum_centroid_rms": float(centroid_upper.min()),
                "maximum_centroid_rms": float(centroid_upper.max()),
                "split_half_minimum_cluster_fraction": split_minimum_fraction,
                "split_half_centroid_match_rms": split_centroid_match_rms,
                "split_half_stability_ratio": split_stability_ratio,
                "split_half_stable": split_stable,
            }
        )

    valid = [
        item
        for item in candidates
        if item["minimum_cluster_fraction"] >= 0.15
        and item["silhouette"] >= 0.25
        and item["minimum_centroid_rms"] >= max(0.08, 2.0 * item["within_cluster_rms"])
        and item["split_half_stable"]
    ]
    best = max(valid or candidates, key=lambda item: item["silhouette"])
    evidence = bool(best in valid)
    weak_evidence = bool(
        best["minimum_cluster_fraction"] >= 0.15
        and best["silhouette"] >= 0.25
        and best["minimum_centroid_rms"] >= 0.08
        and best["split_half_stable"]
    )
    labels = label_sets[int(best["k"])]
    metrics = {
        "multimodal_evidence": evidence,
        "weak_overlapping_cluster_evidence": weak_evidence and not evidence,
        "decision_rule": (
            "silhouette>=0.25, every cluster>=15%, and minimum centroid RMS "
            ">=max(0.08, 2*within-cluster RMS), plus independent even/odd halves "
            "recover all modes with >=10% support and matched-centroid/separation<=0.50"
        ),
        "selected_clustering": best,
        "all_clusterings": candidates,
        "pairwise_chunk_rms_median": float(np.median(upper)),
        "pairwise_chunk_rms_p90": float(np.quantile(upper, 0.9)),
        "pairwise_chunk_rms_max": float(upper.max()),
        "pca_explained_variance": explained[:4].tolist(),
        "pca_projection": projection,
    }
    return metrics, labels


def binary_pattern_metrics(chunks: np.ndarray, dimension: int) -> dict[str, Any]:
    binary = (chunks[:, :, dimension] > 0.5).astype(np.uint8)
    patterns, counts = np.unique(binary, axis=0, return_counts=True)
    order = np.argsort(-counts)
    patterns = patterns[order]
    counts = counts[order]
    probabilities = binary.mean(axis=0)
    entropy = -(
        probabilities * np.log2(np.maximum(probabilities, 1e-12))
        + (1 - probabilities) * np.log2(np.maximum(1 - probabilities, 1e-12))
    )
    changes = np.diff(binary.astype(np.int8), axis=1)
    monotonic_transition = np.logical_or(
        np.all(changes <= 0, axis=1), np.all(changes >= 0, axis=1)
    )
    top_patterns = [
        {"pattern": pattern.tolist(), "count": int(count), "fraction": float(count / len(binary))}
        for pattern, count in zip(patterns[:5], counts[:5])
    ]
    return {
        "unique_patterns": int(len(patterns)),
        "dominant_pattern_fraction": float(counts[0] / len(binary)),
        "top_patterns": top_patterns,
        "probability_one_per_step": probabilities.tolist(),
        "maximum_step_entropy_bits": float(entropy.max()),
        "ambiguous_steps_probability_0.2_to_0.8": int(
            np.logical_and(probabilities >= 0.2, probabilities <= 0.8).sum()
        ),
        "monotonic_single_transition_fraction": float(monotonic_transition.mean()),
        "interpretation": (
            "Many monotonic patterns indicate uncertainty in switch timing, not necessarily "
            "separate semantic action branches."
        ),
    }


def plot_state(
    path: Path,
    chunks: np.ndarray,
    expert: np.ndarray,
    metrics: dict[str, Any],
    labels: np.ndarray,
) -> None:
    projection = np.asarray(metrics["pca_projection"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label in np.unique(labels):
        mask = labels == label
        axes[0].scatter(projection[mask, 0], projection[mask, 1], s=24, label=f"cluster {label}")
    axes[0].set_title("Same-input action chunks (PCA)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=8)

    timesteps = np.arange(chunks.shape[1])
    for label in np.unique(labels):
        centroid = chunks[labels == label][:, :, CONTINUOUS_DIMS].mean(axis=0)
        magnitude = np.linalg.norm(centroid, axis=-1)
        axes[1].plot(timesteps, magnitude, marker="o", ms=2, label=f"cluster {label}")
    axes[1].plot(
        timesteps,
        np.linalg.norm(expert[:, CONTINUOUS_DIMS], axis=-1),
        color="black",
        linestyle="--",
        label="expert",
    )
    axes[1].set_title("Continuous-action magnitude")
    axes[1].set_xlabel("chunk step")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"multimodal evidence: {metrics['multimodal_evidence']}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU for GR00T N1.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_config = DATA_CONFIG_MAP["panda_omron"]
    print(f"Loading GR00T from {args.checkpoint}", flush=True)
    policy = Gr00tPolicy(
        model_path=str(args.checkpoint),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=args.denoising_steps,
        device=args.device,
    )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "data_config": "panda_omron",
        "observation_horizon": 1,
        "action_horizon": 16,
        "denoising_steps": args.denoising_steps,
        "states_per_task": args.states_per_task,
        "samples_per_identical_input": args.samples_per_state,
        "sample_batch_size": args.sample_batch_size,
        "tasks": args.tasks,
        "seed": args.seed,
        "important_control": (
            "The vision-language backbone is evaluated once per state and cached; repeated "
            "samples differ only in the action head's torch.randn initial action noise."
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    all_summaries = []
    for task in args.tasks:
        selected = select_states(args.alias_dir, task, args.states_per_task)
        dataset_dir = Path(selected[0]["dataset_dir"])
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_dir,
            modality_configs=data_config.modality_config(),
            video_backend="opencv",
            video_backend_kwargs=None,
            transforms=None,
            embodiment_tag="new_embodiment",
        )
        task_rows = []
        for state_index, metadata in enumerate(selected):
            episode = int(metadata["episode"])
            frame = int(metadata["frame"])
            print(
                f"[{task}] state {state_index + 1}/{len(selected)}: episode={episode} frame={frame}",
                flush=True,
            )
            raw = dataset.get_step_data(episode, frame)
            expert = concatenate_actions({key: raw[key] for key in ACTION_KEYS})
            chunks = sample_same_input(
                policy, raw, args.samples_per_state, args.sample_batch_size
            )
            metrics, labels = cluster_metrics(chunks, args.seed + 1000 * len(all_summaries) + state_index)
            expert_distance = np.sqrt(
                np.mean(
                    (
                        chunks[:, :, CONTINUOUS_DIMS]
                        - expert[None, :, CONTINUOUS_DIMS]
                    )
                    ** 2,
                    axis=(1, 2),
                )
            )
            row = {
                "task": task,
                "state_index": state_index,
                "state": metadata,
                "multimodal_evidence": metrics["multimodal_evidence"],
                "weak_overlapping_cluster_evidence": metrics[
                    "weak_overlapping_cluster_evidence"
                ],
                "selected_clustering": metrics["selected_clustering"],
                "pairwise_chunk_rms_median": metrics["pairwise_chunk_rms_median"],
                "pairwise_chunk_rms_p90": metrics["pairwise_chunk_rms_p90"],
                "pca_explained_variance": metrics["pca_explained_variance"],
                "sample_to_expert_rms_min": float(expert_distance.min()),
                "sample_to_expert_rms_median": float(np.median(expert_distance)),
                "unique_gripper_patterns": int(
                    len(np.unique((chunks[:, :, 6] > 0.5).astype(np.uint8), axis=0))
                ),
                "unique_control_mode_patterns": int(
                    len(np.unique((chunks[:, :, 11] > 0.5).astype(np.uint8), axis=0))
                ),
                "gripper_pattern_diagnostics": binary_pattern_metrics(chunks, 6),
                "control_mode_pattern_diagnostics": binary_pattern_metrics(chunks, 11),
            }
            state_dir = args.output_dir / task / f"state_{state_index:02d}_ep{episode:06d}_f{frame:04d}"
            state_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                state_dir / "action_chunks.npz", samples=chunks, expert=expert, labels=labels
            )
            plot_metrics = dict(metrics)
            plot_state(state_dir / "clusters.png", chunks, expert, plot_metrics, labels)
            plot_metrics.pop("pca_projection")
            write_json(state_dir / "summary.json", {**row, "cluster_diagnostics": plot_metrics})
            task_rows.append(row)

        summary = {
            "task": task,
            "states_tested": len(task_rows),
            "states_with_multimodal_evidence": int(
                sum(row["multimodal_evidence"] for row in task_rows)
            ),
            "states_with_weak_overlapping_cluster_evidence": int(
                sum(row["weak_overlapping_cluster_evidence"] for row in task_rows)
            ),
            "multimodal_fraction": float(
                np.mean([row["multimodal_evidence"] for row in task_rows])
            ),
            "states": task_rows,
        }
        write_json(args.output_dir / task / "summary.json", summary)
        all_summaries.append(summary)

    combined = {
        "states_tested": int(sum(row["states_tested"] for row in all_summaries)),
        "states_with_multimodal_evidence": int(
            sum(row["states_with_multimodal_evidence"] for row in all_summaries)
        ),
        "states_with_weak_overlapping_cluster_evidence": int(
            sum(
                row["states_with_weak_overlapping_cluster_evidence"]
                for row in all_summaries
            )
        ),
        "tasks": all_summaries,
    }
    combined["multimodal_fraction"] = (
        combined["states_with_multimodal_evidence"] / combined["states_tested"]
    )
    write_json(args.output_dir / "summary_all_tasks.json", combined)
    print(json.dumps(combined, indent=2, ensure_ascii=False, default=jsonable), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
