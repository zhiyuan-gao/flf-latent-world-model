"""Offline RoboCasa windows for a CheckVLA-style rolling latent predictor.

The data contract intentionally contains only information available before an
action chunk is executed: the current proprioceptive state, the candidate
actions, and the current visual latent.  Future proprioception is not exposed
to the model because it would be unavailable when scoring a GR00T candidate.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .manifest import (
    DEFAULT_TASKS,
    available_episodes,
    episode_parquet,
    find_dataset,
    load_task_names,
    split_episodes,
    stable_task_seed,
    write_json,
    write_jsonl,
)


DEFAULT_HORIZON = 16


@dataclass(frozen=True)
class RollingWindowRecord:
    task: str
    split: str
    episode: int
    current_frame: int
    subtask_index: int
    stage: str
    horizon: int = DEFAULT_HORIZON

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(range(self.current_frame, self.current_frame + self.horizon + 1))

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "split": self.split,
            "episode": self.episode,
            "current_frame": self.current_frame,
            "subtask_index": self.subtask_index,
            "stage": self.stage,
            "horizon": self.horizon,
            "frames": list(self.frames),
        }


def nested_episode_splits(
    episodes: Sequence[int],
    train_episodes: int,
    val_episodes: int,
    test_episodes: int,
    seed: int,
    reference_splits: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, list[int]]:
    """Create a deterministic split, optionally extending a fixed train set.

    A plain change from 50 to 100 train episodes would move the old validation
    and test episodes into training because ``split_episodes`` slices one
    shuffled list.  When a reference is supplied, its validation/test sets stay
    fixed and its training set is a strict subset of the returned training set.
    """

    if reference_splits is None:
        return split_episodes(
            episodes,
            train_episodes,
            val_episodes,
            test_episodes,
            seed,
        )
    required = {"train", "val", "test"}
    if set(reference_splits) != required:
        raise ValueError("reference_splits must contain exactly train, val, and test")
    reference = {
        split: [int(value) for value in reference_splits[split]]
        for split in ("train", "val", "test")
    }
    if len(reference["train"]) > train_episodes:
        raise ValueError("Requested train split is smaller than the reference train split")
    if len(reference["val"]) != val_episodes or len(reference["test"]) != test_episodes:
        raise ValueError("Requested validation/test counts differ from the reference split")
    flattened = [value for split in reference.values() for value in split]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Reference episode splits overlap")
    available = set(int(value) for value in episodes)
    missing = set(flattened) - available
    if missing:
        raise ValueError(f"Reference episodes are unavailable: {sorted(missing)}")

    shuffled = np.asarray(episodes, dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    reserved = set(flattened)
    candidates = [int(value) for value in shuffled if int(value) not in reserved]
    needed = train_episodes - len(reference["train"])
    if needed > len(candidates):
        raise ValueError(
            f"Requested {needed} additional train episodes from only {len(candidates)}"
        )
    return {
        "train": sorted(reference["train"] + candidates[:needed]),
        "val": sorted(reference["val"]),
        "test": sorted(reference["test"]),
    }


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_episode_table(dataset: Path, episode: int) -> pd.DataFrame:
    return pd.read_parquet(
        episode_parquet(dataset, episode),
        columns=[
            "observation.state",
            "action",
            "subtask_idx",
            "annotation.human.subtask_stage",
            "frame_index",
        ],
    )


def build_episode_windows(
    task: str,
    split: str,
    episode: int,
    table: pd.DataFrame,
    task_names: Mapping[int, str],
    horizon: int = DEFAULT_HORIZON,
    stride: int = 4,
) -> list[RollingWindowRecord]:
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    if len(table) <= horizon:
        return []
    indices = table["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(table), dtype=np.int64)):
        raise ValueError(f"{task} episode {episode} has non-contiguous frame indices")
    subtask = table["subtask_idx"].to_numpy(dtype=np.int64)
    stages = table["annotation.human.subtask_stage"].to_numpy(dtype=np.int64)
    rows: list[RollingWindowRecord] = []
    for current in range(0, len(table) - horizon, stride):
        target_stages = stages[current : current + horizon + 1]
        if any(task_names[int(value)].strip().lower() == "done" for value in target_stages):
            continue
        rows.append(
            RollingWindowRecord(
                task=task,
                split=split,
                episode=episode,
                current_frame=current,
                subtask_index=int(subtask[current]),
                stage=task_names[int(stages[current])],
                horizon=horizon,
            )
        )
    return rows


def _moments(sum_: np.ndarray, square_sum: np.ndarray, count: int) -> dict[str, object]:
    mean = sum_ / count
    variance = np.maximum(square_sum / count - np.square(mean), 1e-12)
    return {"count": count, "mean": mean.tolist(), "std": np.sqrt(variance).tolist()}


def build_rolling_manifest(
    data_root: Path,
    output_dir: Path,
    tasks: Sequence[str] = DEFAULT_TASKS,
    train_episodes: int = 50,
    val_episodes: int = 10,
    test_episodes: int = 20,
    horizon: int = DEFAULT_HORIZON,
    window_stride: int = 4,
    camera: str = "robot0_agentview_left",
    seed: int = 20260802,
    reference_splits: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows: dict[str, list[RollingWindowRecord]] = {"train": [], "val": [], "test": []}
    task_metadata: dict[str, object] = {}
    action_sum = action_square_sum = state_sum = state_square_sum = None
    action_count = state_count = 0

    for task in tasks:
        dataset = find_dataset(data_root / task)
        episodes = available_episodes(dataset, camera)
        splits = nested_episode_splits(
            episodes,
            train_episodes,
            val_episodes,
            test_episodes,
            stable_task_seed(seed, task),
            reference_splits.get(task) if reference_splits is not None else None,
        )
        names = load_task_names(dataset)
        task_counts: Counter[str] = Counter()
        for split, split_episodes_values in splits.items():
            for episode in split_episodes_values:
                table = load_episode_table(dataset, episode)
                rows = build_episode_windows(
                    task,
                    split,
                    episode,
                    table,
                    names,
                    horizon=horizon,
                    stride=window_stride,
                )
                windows[split].extend(rows)
                task_counts[split] += len(rows)
                actions = np.stack(table["action"].to_numpy()).astype(np.float32)
                states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
                episode_dir = output_dir / "episodes" / task / split
                episode_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    episode_dir / f"episode_{episode:06d}.npz",
                    actions=actions,
                    states=states,
                    subtask_index=table["subtask_idx"].to_numpy(dtype=np.int64),
                    stage_id=table["annotation.human.subtask_stage"].to_numpy(dtype=np.int64),
                )
                if split == "train":
                    for row in rows:
                        selected_actions = actions[
                            row.current_frame : row.current_frame + horizon
                        ].astype(np.float64)
                        selected_state = states[row.current_frame].astype(np.float64)
                        current_action_sum = selected_actions.sum(axis=0)
                        current_action_square = np.square(selected_actions).sum(axis=0)
                        action_sum = (
                            current_action_sum
                            if action_sum is None
                            else action_sum + current_action_sum
                        )
                        action_square_sum = (
                            current_action_square
                            if action_square_sum is None
                            else action_square_sum + current_action_square
                        )
                        state_sum = selected_state if state_sum is None else state_sum + selected_state
                        current_state_square = np.square(selected_state)
                        state_square_sum = (
                            current_state_square
                            if state_square_sum is None
                            else state_square_sum + current_state_square
                        )
                        action_count += len(selected_actions)
                        state_count += 1
        task_metadata[task] = {
            "dataset": str(dataset.resolve()),
            "available_episodes": len(episodes),
            "splits": splits,
            "window_counts": dict(task_counts),
        }

    if any(value is None for value in (action_sum, action_square_sum, state_sum, state_square_sum)):
        raise RuntimeError("No training windows were collected")
    split_counts: dict[str, int] = {}
    balance_counts: dict[str, dict[str, int]] = {}
    for split, rows in windows.items():
        split_counts[split] = write_jsonl(
            output_dir / f"windows_{split}.jsonl",
            (row.as_dict() for row in rows),
        )
        balance = Counter(
            f"{row.task}/subtask_{row.subtask_index}/{row.stage}" for row in rows
        )
        balance_counts[split] = dict(sorted(balance.items()))

    action_stats = _moments(action_sum, action_square_sum, action_count)
    state_stats = _moments(state_sum, state_square_sum, state_count)
    manifest = {
        "version": 1,
        "definition": "CheckVLA-style per-action rolling latent prediction",
        "data_root": str(data_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "tasks": task_metadata,
        "camera": camera,
        "fps": 20,
        "seed": seed,
        "window_stride": window_stride,
        "horizon": horizon,
        "split_window_counts": split_counts,
        "balance_counts": balance_counts,
        "action_stats": action_stats,
        "state_stats": state_stats,
    }
    write_json(output_dir / "action_stats.json", action_stats)
    write_json(output_dir / "state_stats.json", state_stats)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


class CachedRollingWindows(Dataset):
    """Join dense latent caches with offline actions and anchor proprioception."""

    def __init__(
        self,
        manifest_dir: Path,
        feature_root: Path,
        split: str,
        max_cached_episodes: int = 128,
    ) -> None:
        self.manifest_dir = Path(manifest_dir)
        self.feature_root = Path(feature_root)
        self.split = split
        self.max_cached_episodes = int(max_cached_episodes)
        self.records = list(read_jsonl(self.manifest_dir / f"windows_{split}.jsonl"))
        if not self.records:
            raise FileNotFoundError(f"No {split} windows under {self.manifest_dir}")
        action_stats = json.loads((self.manifest_dir / "action_stats.json").read_text())
        state_stats = json.loads((self.manifest_dir / "state_stats.json").read_text())
        self.action_mean = np.asarray(action_stats["mean"], dtype=np.float32)
        self.action_std = np.maximum(
            np.asarray(action_stats["std"], dtype=np.float32), 1e-6
        )
        self.state_mean = np.asarray(state_stats["mean"], dtype=np.float32)
        self.state_std = np.maximum(np.asarray(state_stats["std"], dtype=np.float32), 1e-6)
        self.normalized_zero_action = -self.action_mean / self.action_std
        self._cache: OrderedDict[tuple[str, int], tuple[dict, dict]] = OrderedDict()
        counts = Counter(
            (str(row["task"]), int(row["subtask_index"])) for row in self.records
        )
        self.sample_weights = torch.tensor(
            [
                1.0 / counts[(str(row["task"]), int(row["subtask_index"]))]
                for row in self.records
            ],
            dtype=torch.double,
        )
        self.sample_weights /= self.sample_weights.mean()

    def __len__(self) -> int:
        return len(self.records)

    def _load_episode(self, task: str, episode: int) -> tuple[dict, dict]:
        key = (task, episode)
        if key in self._cache:
            values = self._cache.pop(key)
            self._cache[key] = values
            return values
        episode_path = (
            self.manifest_dir / "episodes" / task / self.split / f"episode_{episode:06d}.npz"
        )
        feature_base = self.feature_root / task / self.split / f"episode_{episode:06d}"
        feature_frames_path = feature_base.with_suffix(".frames.npy")
        feature_values_path = feature_base.with_suffix(".features.npy")
        legacy_feature_path = feature_base.with_suffix(".npz")
        has_memmap_cache = feature_frames_path.is_file() and feature_values_path.is_file()
        if not has_memmap_cache and not legacy_feature_path.is_file():
            raise FileNotFoundError(
                "Missing feature cache: expected either "
                f"{feature_frames_path} + {feature_values_path}, or {legacy_feature_path}"
            )
        with np.load(episode_path) as source:
            episode_values = {key: source[key] for key in source.files}
        if has_memmap_cache:
            feature_values = {
                "frames": np.load(feature_frames_path),
                "features": np.load(feature_values_path, mmap_mode="r"),
            }
        else:
            with np.load(legacy_feature_path) as source:
                feature_values = {key: source[key] for key in source.files}
        feature_values["frames"] = feature_values["frames"].astype(np.int64)
        self._cache[key] = (episode_values, feature_values)
        while len(self._cache) > self.max_cached_episodes:
            self._cache.popitem(last=False)
        return episode_values, feature_values

    @staticmethod
    def _feature_positions(cached: np.ndarray, requested: Sequence[int]) -> np.ndarray:
        requested_values = np.asarray(requested, dtype=np.int64)
        positions = np.searchsorted(cached, requested_values)
        clipped = np.minimum(positions, max(len(cached) - 1, 0))
        if len(cached) == 0 or np.any(positions >= len(cached)) or not np.array_equal(
            cached[clipped], requested_values
        ):
            raise KeyError(f"Dense feature cache is missing requested frames {requested_values.tolist()}")
        return positions

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        task = str(row["task"])
        episode = int(row["episode"])
        current = int(row["current_frame"])
        horizon = int(row["horizon"])
        episode_values, feature_values = self._load_episode(task, episode)
        positions = self._feature_positions(feature_values["frames"], row["frames"])
        latents = feature_values["features"][positions].astype(np.float32)
        actions = episode_values["actions"][current : current + horizon].astype(np.float32)
        state = episode_values["states"][current].astype(np.float32)
        actions = (actions - self.action_mean) / self.action_std
        state = (state - self.state_mean) / self.state_std
        return {
            "latents": torch.from_numpy(latents),
            "actions": torch.from_numpy(actions),
            "anchor_state": torch.from_numpy(state),
            "zero_actions": torch.from_numpy(
                np.broadcast_to(self.normalized_zero_action, actions.shape).copy()
            ),
            "task": task,
            "episode": episode,
            "current_frame": current,
            "subtask_index": int(row["subtask_index"]),
            "stage": str(row["stage"]),
        }


class CachedEndpointWindows(CachedRollingWindows):
    """Load only the anchor and terminal visual latents for endpoint training.

    The rolling manifest remains the source of actions, the single anchor
    proprioceptive state, normalization statistics, and episode-isolated
    splits.  Feature caches need contain only ``current`` and
    ``current + horizon`` frames; no future proprioception is returned.
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        task = str(row["task"])
        episode = int(row["episode"])
        current_frame = int(row["current_frame"])
        horizon = int(row["horizon"])
        episode_values, feature_values = self._load_episode(task, episode)
        positions = self._feature_positions(
            feature_values["frames"],
            (current_frame, current_frame + horizon),
        )
        latents = feature_values["features"][positions].astype(np.float32)
        if latents.ndim != 4 or len(latents) != 2:
            raise ValueError(
                "Endpoint features must have [2, grid_h, grid_w, feature_dim]"
            )
        actions = episode_values["actions"][
            current_frame : current_frame + horizon
        ].astype(np.float32)
        if len(actions) != horizon:
            raise ValueError("Endpoint window does not contain its full action chunk")
        anchor_state = episode_values["states"][current_frame].astype(np.float32)
        actions = (actions - self.action_mean) / self.action_std
        anchor_state = (anchor_state - self.state_mean) / self.state_std
        return {
            "current": torch.from_numpy(latents[0]),
            "target": torch.from_numpy(latents[1]),
            "actions": torch.from_numpy(actions),
            "anchor_state": torch.from_numpy(anchor_state),
            "zero_actions": torch.from_numpy(
                np.broadcast_to(self.normalized_zero_action, actions.shape).copy()
            ),
            "task": task,
            "episode": episode,
            "current_frame": current_frame,
            "subtask_index": int(row["subtask_index"]),
            "stage": str(row["stage"]),
            "horizon": horizon,
        }


class CachedSingleStepDynamicsWindows(CachedRollingWindows):
    """Current visual/state plus four actions with visual/state targets at t+4.

    The class intentionally reuses the existing fixed split and t+16 window
    anchors.  Only the first four actions and the targets at ``current + 4``
    are exposed, which makes the new transition experiment directly
    comparable to the previous endpoint runs.  Recorded future state is
    returned only under ``target_state`` and is never a conditioning input.
    """

    prediction_horizon = 4

    def __init__(
        self,
        manifest_dir: Path,
        feature_root: Path,
        split: str,
        max_cached_episodes: int = 128,
    ) -> None:
        super().__init__(
            manifest_dir,
            feature_root,
            split,
            max_cached_episodes=max_cached_episodes,
        )
        records_by_episode: dict[tuple[str, int], list[dict[str, object]]] = {}
        for row in self.records:
            key = (str(row["task"]), int(row["episode"]))
            records_by_episode.setdefault(key, []).append(row)

        retained: list[dict[str, object]] = []
        dropped = 0
        for (task, episode), rows in sorted(records_by_episode.items()):
            feature_base = self.feature_root / task / self.split / f"episode_{episode:06d}"
            frames_path = feature_base.with_suffix(".frames.npy")
            legacy_path = feature_base.with_suffix(".npz")
            if frames_path.is_file():
                available = np.load(frames_path).astype(np.int64)
            elif legacy_path.is_file():
                with np.load(legacy_path) as source:
                    available = source["frames"].astype(np.int64)
            else:
                raise FileNotFoundError(f"Missing feature cache for {feature_base}")
            available_set = set(int(value) for value in available)
            for row in rows:
                current = int(row["current_frame"])
                if current in available_set and current + self.prediction_horizon in available_set:
                    retained.append(row)
                else:
                    dropped += 1
        if not retained:
            raise RuntimeError(f"No cached single-step windows remain for split {split}")
        self.records = retained
        self.dropped_missing_features = dropped

        group_counts = Counter(
            (str(row["task"]), int(row["subtask_index"])) for row in self.records
        )
        subtasks_per_task: dict[str, int] = Counter()
        for task, _ in group_counts:
            subtasks_per_task[task] += 1
        task_count = len(subtasks_per_task)
        weights = []
        for row in self.records:
            task = str(row["task"])
            group = (task, int(row["subtask_index"]))
            weights.append(
                1.0
                / (
                    task_count
                    * subtasks_per_task[task]
                    * group_counts[group]
                )
            )
        self.sample_weights = torch.tensor(weights, dtype=torch.double)
        self.sample_weights /= self.sample_weights.mean()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        task = str(row["task"])
        episode = int(row["episode"])
        current = int(row["current_frame"])
        episode_values, feature_values = self._load_episode(task, episode)
        target_frame = current + self.prediction_horizon
        positions = self._feature_positions(
            feature_values["frames"], (current, target_frame)
        )
        latents = feature_values["features"][positions].astype(np.float32)
        actions = episode_values["actions"][current:target_frame].astype(np.float32)
        if actions.shape != (self.prediction_horizon, len(self.action_mean)):
            raise ValueError("Single-step window lacks its complete four-action block")
        current_state = episode_values["states"][current].astype(np.float32)
        target_state = episode_values["states"][target_frame].astype(np.float32)
        actions = (actions - self.action_mean) / self.action_std
        current_state = (current_state - self.state_mean) / self.state_std
        target_state = (target_state - self.state_mean) / self.state_std
        return {
            "current": torch.from_numpy(latents[0]),
            "target": torch.from_numpy(latents[1]),
            "actions": torch.from_numpy(actions),
            "anchor_state": torch.from_numpy(current_state),
            "target_state": torch.from_numpy(target_state),
            "zero_actions": torch.from_numpy(
                np.broadcast_to(self.normalized_zero_action, actions.shape).copy()
            ),
            "task": task,
            "episode": episode,
            "current_frame": current,
            "subtask_index": int(row["subtask_index"]),
            "stage": str(row["stage"]),
            "horizon": self.prediction_horizon,
        }


class CachedMultiHorizonEndpointWindows(CachedRollingWindows):
    """Load current plus real t+4/t+8/t+12/t+16 visual targets."""

    horizons = (4, 8, 12, 16)

    def __init__(
        self,
        manifest_dir: Path,
        feature_root: Path,
        split: str,
        max_cached_episodes: int = 128,
        normalize_representations: bool = False,
    ) -> None:
        super().__init__(
            manifest_dir,
            feature_root,
            split,
            max_cached_episodes=max_cached_episodes,
        )
        self.normalize_representations = bool(normalize_representations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        task = str(row["task"])
        episode = int(row["episode"])
        current_frame = int(row["current_frame"])
        horizon = int(row["horizon"])
        if horizon != self.horizons[-1]:
            raise ValueError(
                f"Multi-horizon windows require t+{self.horizons[-1]}, received t+{horizon}"
            )
        episode_values, feature_values = self._load_episode(task, episode)
        requested_frames = (current_frame,) + tuple(
            current_frame + offset for offset in self.horizons
        )
        positions = self._feature_positions(
            feature_values["frames"], requested_frames
        )
        latents = feature_values["features"][positions].astype(np.float32)
        if latents.ndim != 4 or len(latents) != 1 + len(self.horizons):
            raise ValueError(
                "Multi-horizon features must have [5, grid_h, grid_w, feature_dim]"
            )
        actions = episode_values["actions"][
            current_frame : current_frame + horizon
        ].astype(np.float32)
        if len(actions) != horizon:
            raise ValueError("Multi-horizon window does not contain its full action chunk")
        anchor_state = episode_values["states"][current_frame].astype(np.float32)
        actions = (actions - self.action_mean) / self.action_std
        anchor_state = (anchor_state - self.state_mean) / self.state_std
        latent_tensor = torch.from_numpy(latents)
        if self.normalize_representations:
            # V-JEPA2-AC applies an additional affine-free LayerNorm to frozen
            # target-encoder representations before its prediction loss.
            latent_tensor = torch.nn.functional.layer_norm(
                latent_tensor, (latent_tensor.shape[-1],)
            )
        return {
            "current": latent_tensor[0],
            "targets": latent_tensor[1:],
            "actions": torch.from_numpy(actions),
            "anchor_state": torch.from_numpy(anchor_state),
            "zero_actions": torch.from_numpy(
                np.broadcast_to(self.normalized_zero_action, actions.shape).copy()
            ),
            "task": task,
            "episode": episode,
            "current_frame": current_frame,
            "subtask_index": int(row["subtask_index"]),
            "stage": str(row["stage"]),
            "horizons": self.horizons,
        }


class CachedBlockRollingWindows(CachedMultiHorizonEndpointWindows):
    """Four visual transition blocks plus proprio targets used only for supervision.

    The model input remains current visual state, current proprioception, and
    candidate actions.  Recorded future states are returned under an explicit
    target-only key and must never be passed into the visual predictor.
    """

    def __init__(
        self,
        manifest_dir: Path,
        feature_root: Path,
        split: str,
        max_cached_episodes: int = 128,
    ) -> None:
        super().__init__(
            manifest_dir,
            feature_root,
            split,
            max_cached_episodes=max_cached_episodes,
            normalize_representations=True,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        row = self.records[index]
        task = str(row["task"])
        episode = int(row["episode"])
        current_frame = int(row["current_frame"])
        horizon = int(row["horizon"])
        episode_values, _ = self._load_episode(task, episode)
        future_states = episode_values["states"][
            current_frame + 1 : current_frame + horizon + 1
        ].astype(np.float32)
        if future_states.shape != (horizon, len(self.state_mean)):
            raise ValueError("Block rolling window lacks complete future state targets")
        future_states = (future_states - self.state_mean) / self.state_std
        sample["future_state_targets"] = torch.from_numpy(future_states)
        return sample


def collate_endpoint_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    horizons = {int(row["horizon"]) for row in rows}
    if len(horizons) != 1:
        raise ValueError("An endpoint batch must use one common horizon")
    return {
        "current": torch.stack([row["current"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
        "anchor_state": torch.stack([row["anchor_state"] for row in rows]),
        "zero_actions": torch.stack([row["zero_actions"] for row in rows]),
        "task": [row["task"] for row in rows],
        "episode": torch.tensor([row["episode"] for row in rows], dtype=torch.long),
        "current_frame": torch.tensor(
            [row["current_frame"] for row in rows], dtype=torch.long
        ),
        "subtask_index": torch.tensor(
            [row["subtask_index"] for row in rows], dtype=torch.long
        ),
        "stage": [row["stage"] for row in rows],
        "horizon": next(iter(horizons)),
    }


def collate_single_step_dynamics_windows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    batch = collate_endpoint_windows(rows)
    batch["target_state"] = torch.stack([row["target_state"] for row in rows])
    return batch


def collate_multi_horizon_endpoint_windows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    horizons = {tuple(row["horizons"]) for row in rows}
    if len(horizons) != 1:
        raise ValueError("A multi-horizon batch must use one common horizon set")
    return {
        "current": torch.stack([row["current"] for row in rows]),
        "targets": torch.stack([row["targets"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
        "anchor_state": torch.stack([row["anchor_state"] for row in rows]),
        "zero_actions": torch.stack([row["zero_actions"] for row in rows]),
        "task": [row["task"] for row in rows],
        "episode": torch.tensor([row["episode"] for row in rows], dtype=torch.long),
        "current_frame": torch.tensor(
            [row["current_frame"] for row in rows], dtype=torch.long
        ),
        "subtask_index": torch.tensor(
            [row["subtask_index"] for row in rows], dtype=torch.long
        ),
        "stage": [row["stage"] for row in rows],
        "horizons": next(iter(horizons)),
    }


def collate_block_rolling_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate rolling inputs while keeping recorded future states target-only."""
    batch = collate_multi_horizon_endpoint_windows(rows)
    batch["future_state_targets"] = torch.stack(
        [row["future_state_targets"] for row in rows]
    )
    return batch


def collate_rolling_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "latents": torch.stack([row["latents"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
        "anchor_state": torch.stack([row["anchor_state"] for row in rows]),
        "zero_actions": torch.stack([row["zero_actions"] for row in rows]),
        "task": [row["task"] for row in rows],
        "episode": torch.tensor([row["episode"] for row in rows], dtype=torch.long),
        "current_frame": torch.tensor(
            [row["current_frame"] for row in rows], dtype=torch.long
        ),
        "subtask_index": torch.tensor(
            [row["subtask_index"] for row in rows], dtype=torch.long
        ),
        "stage": [row["stage"] for row in rows],
    }
