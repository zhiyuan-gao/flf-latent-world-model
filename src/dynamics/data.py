"""Cached-window dataset shared by both frozen visual encoders."""

from __future__ import annotations

import json
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ACTION_HORIZON, read_jsonl


class CachedDynamicsWindows(Dataset):
    """Join a common window manifest with one encoder's feature cache.

    Episode arrays are kept in a small per-worker LRU cache.  This avoids
    opening compressed files for every overlapping window without keeping the
    complete four-task dataset in RAM.
    """

    def __init__(
        self,
        manifest_dir: Path,
        feature_root: Path,
        split: str,
        max_cached_episodes: int = 256,
    ) -> None:
        self.manifest_dir = Path(manifest_dir)
        self.feature_root = Path(feature_root)
        self.split = split
        self.max_cached_episodes = int(max_cached_episodes)
        if self.max_cached_episodes < 1:
            raise ValueError("max_cached_episodes must be positive")
        self.records = list(read_jsonl(self.manifest_dir / f"windows_{split}.jsonl"))
        if not self.records:
            raise FileNotFoundError(f"No {split} windows in {self.manifest_dir}")
        stats = json.loads((self.manifest_dir / "action_stats.json").read_text())
        self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["std"], dtype=np.float32)
        self.action_std = np.maximum(self.action_std, 1e-6)
        self.normalized_zero_action = -self.action_mean / self.action_std
        self._episode_cache: OrderedDict[tuple[str, int], tuple[dict, dict]] = OrderedDict()

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
        if key in self._episode_cache:
            values = self._episode_cache.pop(key)
            self._episode_cache[key] = values
            return values

        episode_path = (
            self.manifest_dir
            / "episodes"
            / task
            / self.split
            / f"episode_{episode:06d}.npz"
        )
        feature_path = (
            self.feature_root
            / task
            / self.split
            / f"episode_{episode:06d}.npz"
        )
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing feature cache: {feature_path}")
        with np.load(episode_path) as values:
            episode_values = {name: values[name] for name in values.files}
        with np.load(feature_path) as values:
            feature_values = {name: values[name] for name in values.files}
        frames = feature_values["frames"].astype(np.int64)
        if len(frames) == 0 or np.any(np.diff(frames) <= 0):
            raise ValueError(f"Feature frames must be strictly increasing: {feature_path}")
        feature_values["frames"] = frames
        self._episode_cache[key] = (episode_values, feature_values)
        while len(self._episode_cache) > self.max_cached_episodes:
            self._episode_cache.popitem(last=False)
        return episode_values, feature_values

    @staticmethod
    def _feature_positions(cached_frames: np.ndarray, requested: Sequence[int]) -> np.ndarray:
        requested_array = np.asarray(requested, dtype=np.int64)
        positions = np.searchsorted(cached_frames, requested_array)
        valid = positions < len(cached_frames)
        if not np.all(valid) or not np.array_equal(cached_frames[positions], requested_array):
            missing = requested_array[~valid | (cached_frames[np.minimum(positions, len(cached_frames) - 1)] != requested_array)]
            raise KeyError(f"Feature cache is missing frames {missing.tolist()}")
        return positions

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        task = str(record["task"])
        episode = int(record["episode"])
        current = int(record["current_frame"])
        episode_values, feature_values = self._load_episode(task, episode)
        requested = [*record["history_frames"], *record["future_frames"]]
        positions = self._feature_positions(feature_values["frames"], requested)
        features = feature_values["features"][positions].astype(np.float32)
        if features.ndim != 4:
            raise ValueError("Cached features must have [frame, grid_h, grid_w, dim]")
        actions = episode_values["actions"][current : current + ACTION_HORIZON].astype(np.float32)
        if len(actions) != ACTION_HORIZON:
            raise ValueError("Window does not contain sixteen actions")
        actions = (actions - self.action_mean) / self.action_std
        return {
            "history": torch.from_numpy(features[:4]),
            "future": torch.from_numpy(features[4:]),
            "actions": torch.from_numpy(actions.reshape(4, 4, -1)),
            "zero_actions": torch.from_numpy(
                np.broadcast_to(self.normalized_zero_action, (4, 4, len(self.action_mean))).copy()
            ),
            "task": task,
            "episode": episode,
            "current_frame": current,
            "subtask_index": int(record["subtask_index"]),
            "stage": str(record["stage"]),
        }


def collate_dynamics_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "history": torch.stack([row["history"] for row in rows]),
        "future": torch.stack([row["future"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
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
