"""Build deterministic, episode-isolated RoboCasa dynamics windows.

The data contract is deliberately independent of either visual encoder.  A
window always contains four historical observations, sixteen expert actions,
and four fixed-time future observations.  The observation spacing is four
control steps; no temporal warping is used for future targets.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_TASKS = (
    "PreSoakPan",
    "KettleBoiling",
    "LoadDishwasher",
    "RinseSinkBasin",
)
HISTORY_OFFSETS = (-12, -8, -4, 0)
FUTURE_OFFSETS = (4, 8, 12, 16)
ACTION_HORIZON = 16


@dataclass(frozen=True)
class WindowRecord:
    task: str
    split: str
    episode: int
    current_frame: int
    subtask_index: int
    stage: str

    @property
    def history_frames(self) -> tuple[int, ...]:
        return tuple(self.current_frame + value for value in HISTORY_OFFSETS)

    @property
    def future_frames(self) -> tuple[int, ...]:
        return tuple(self.current_frame + value for value in FUTURE_OFFSETS)

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "split": self.split,
            "episode": self.episode,
            "current_frame": self.current_frame,
            "subtask_index": self.subtask_index,
            "stage": self.stage,
            "history_frames": list(self.history_frames),
            "future_frames": list(self.future_frames),
        }


def stable_task_seed(seed: int, task: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "little")
    return (int(seed) + suffix) % (2**32)


def find_dataset(task_root: Path) -> Path:
    matches = sorted(path.parent.parent for path in task_root.glob("*/lerobot/meta/info.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dataset under {task_root}, found {matches}")
    return matches[0]


def load_task_names(dataset: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[int(row["task_index"])] = str(row["task"])
    return result


def episode_parquet(dataset: Path, episode: int) -> Path:
    return dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def episode_video(dataset: Path, episode: int, camera: str) -> Path:
    return (
        dataset
        / f"videos/chunk-{episode // 1000:03d}"
        / f"observation.images.{camera}"
        / f"episode_{episode:06d}.mp4"
    )


def available_episodes(dataset: Path, camera: str) -> list[int]:
    episodes = []
    for parquet in sorted(dataset.glob("data/*/episode_*.parquet")):
        episode = int(parquet.stem.split("_")[-1])
        if episode_video(dataset, episode, camera).is_file():
            episodes.append(episode)
    return episodes


def split_episodes(
    episodes: Sequence[int], train: int, val: int, test: int, seed: int
) -> dict[str, list[int]]:
    total = train + val + test
    if total > len(episodes):
        raise ValueError(f"Requested {total} episodes from only {len(episodes)}")
    shuffled = np.asarray(episodes, dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:train].tolist()),
        "val": sorted(shuffled[train : train + val].tolist()),
        "test": sorted(shuffled[train + val : total].tolist()),
    }


def load_episode_table(dataset: Path, episode: int) -> pd.DataFrame:
    return pd.read_parquet(
        episode_parquet(dataset, episode),
        columns=[
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
    stride: int = 4,
) -> list[WindowRecord]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if len(table) <= abs(HISTORY_OFFSETS[0]) + FUTURE_OFFSETS[-1]:
        return []
    frame_index = table["frame_index"].to_numpy(dtype=np.int64)
    expected = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(frame_index, expected):
        raise ValueError(f"{task} episode {episode} has non-contiguous frame_index")

    subtask = table["subtask_idx"].to_numpy(dtype=np.int64)
    stage_ids = table["annotation.human.subtask_stage"].to_numpy(dtype=np.int64)
    records = []
    first = abs(HISTORY_OFFSETS[0])
    last = len(table) - FUTURE_OFFSETS[-1] - 1
    for current in range(first, last + 1, stride):
        stage = task_names[int(stage_ids[current])]
        # Terminal bookkeeping is not robot dynamics and must never become a target.
        target_slice = stage_ids[current : current + FUTURE_OFFSETS[-1] + 1]
        if any(task_names[int(value)].strip().lower() == "done" for value in target_slice):
            continue
        records.append(
            WindowRecord(
                task=task,
                split=split,
                episode=int(episode),
                current_frame=int(current),
                subtask_index=int(subtask[current]),
                stage=stage,
            )
        )
    return records


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_manifest(
    data_root: Path,
    output_dir: Path,
    tasks: Sequence[str] = DEFAULT_TASKS,
    train_episodes: int = 50,
    val_episodes: int = 10,
    test_episodes: int = 20,
    window_stride: int = 4,
    camera: str = "robot0_agentview_left",
    seed: int = 20260802,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows: dict[str, list[WindowRecord]] = {"train": [], "val": [], "test": []}
    task_metadata: dict[str, object] = {}
    action_sum: np.ndarray | None = None
    action_square_sum: np.ndarray | None = None
    action_count = 0

    for task in tasks:
        dataset = find_dataset(data_root / task)
        episodes = available_episodes(dataset, camera)
        splits = split_episodes(
            episodes,
            train_episodes,
            val_episodes,
            test_episodes,
            stable_task_seed(seed, task),
        )
        names = load_task_names(dataset)
        task_counts: Counter[str] = Counter()
        for split, split_values in splits.items():
            for episode in split_values:
                table = load_episode_table(dataset, episode)
                records = build_episode_windows(
                    task,
                    split,
                    episode,
                    table,
                    names,
                    window_stride,
                )
                windows[split].extend(records)
                task_counts[split] += len(records)
                actions = np.stack(table["action"].to_numpy()).astype(np.float32)
                episode_cache = output_dir / "episodes" / task / split
                episode_cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    episode_cache / f"episode_{episode:06d}.npz",
                    actions=actions,
                    subtask_index=table["subtask_idx"].to_numpy(dtype=np.int64),
                    stage_id=table["annotation.human.subtask_stage"].to_numpy(dtype=np.int64),
                )
                if split == "train":
                    # Match the training-window action distribution, including overlap.
                    for record in records:
                        selected = actions[
                            record.current_frame : record.current_frame + ACTION_HORIZON
                        ].astype(np.float64)
                        current_sum = selected.sum(axis=0)
                        current_square_sum = np.square(selected).sum(axis=0)
                        action_sum = current_sum if action_sum is None else action_sum + current_sum
                        action_square_sum = (
                            current_square_sum
                            if action_square_sum is None
                            else action_square_sum + current_square_sum
                        )
                        action_count += len(selected)
        task_metadata[task] = {
            "dataset": str(dataset.resolve()),
            "available_episodes": len(episodes),
            "splits": splits,
            "window_counts": dict(task_counts),
        }

    split_counts = {}
    balance_counts = {}
    for split, records in windows.items():
        path = output_dir / f"windows_{split}.jsonl"
        split_counts[split] = write_jsonl(path, (record.as_dict() for record in records))
        balance = Counter(
            f"{record.task}/subtask_{record.subtask_index}/{record.stage}" for record in records
        )
        balance_counts[split] = dict(sorted(balance.items()))

    manifest = {
        "version": 1,
        "definition": "four-history + sixteen-actions -> four fixed-time futures",
        "data_root": str(data_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "tasks": task_metadata,
        "camera": camera,
        "fps": 20,
        "seed": seed,
        "window_stride": window_stride,
        "history_offsets": list(HISTORY_OFFSETS),
        "future_offsets": list(FUTURE_OFFSETS),
        "action_horizon": ACTION_HORIZON,
        "split_window_counts": split_counts,
        "balance_counts": balance_counts,
    }
    if action_sum is None or action_square_sum is None or action_count == 0:
        raise RuntimeError("No training actions were collected")
    action_mean = action_sum / action_count
    action_variance = np.maximum(action_square_sum / action_count - np.square(action_mean), 1e-12)
    action_stats = {
        "count": action_count,
        "mean": action_mean.tolist(),
        "std": np.sqrt(action_variance).tolist(),
    }
    manifest["action_stats"] = action_stats
    write_json(output_dir / "action_stats.json", action_stats)
    write_json(output_dir / "manifest.json", manifest)
    return manifest
