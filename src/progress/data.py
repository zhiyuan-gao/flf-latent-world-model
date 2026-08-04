"""Cached-feature dataset helpers for progress tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    task: str
    task_id: int
    split: str
    episode: int


class CachedProgressEpisodes(Dataset):
    def __init__(self, cache_root: Path, split: str, tasks: Iterable[str]) -> None:
        self.tasks = tuple(tasks)
        self.task_to_id = {task: index for index, task in enumerate(self.tasks)}
        records: list[EpisodeRecord] = []
        for task in self.tasks:
            for path in sorted((cache_root / task / split).glob("episode_*.npz")):
                episode = int(path.stem.split("_")[-1])
                records.append(
                    EpisodeRecord(path, task, self.task_to_id[task], split, episode)
                )
        if not records:
            raise FileNotFoundError(f"No cached {split} episodes under {cache_root}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(record.path) as values:
            return {
                "visual": torch.from_numpy(values["visual"].astype(np.float32)),
                "state": torch.from_numpy(values["state"].astype(np.float32)),
                "labels": torch.from_numpy(values["labels"].astype(np.int64)),
                "frames": torch.from_numpy(values["frames"].astype(np.int64)),
                "task_id": record.task_id,
                "task": record.task,
                "episode": record.episode,
            }


def collate_episodes(rows: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(row["labels"]) for row in rows], dtype=torch.long)
    max_steps = int(lengths.max())
    batch = len(rows)
    visual_dim = int(rows[0]["visual"].shape[-1])
    state_dim = int(rows[0]["state"].shape[-1])

    visual = torch.zeros(batch, max_steps, visual_dim, dtype=torch.float32)
    state = torch.zeros(batch, max_steps, state_dim, dtype=torch.float32)
    labels = torch.full((batch, max_steps), -100, dtype=torch.long)
    frames = torch.full((batch, max_steps), -1, dtype=torch.long)
    mask = torch.zeros(batch, max_steps, dtype=torch.bool)
    for index, row in enumerate(rows):
        steps = int(lengths[index])
        visual[index, :steps] = row["visual"]
        state[index, :steps] = row["state"]
        labels[index, :steps] = row["labels"]
        frames[index, :steps] = row["frames"]
        mask[index, :steps] = True

    return {
        "visual": visual,
        "state": state,
        "labels": labels,
        "frames": frames,
        "mask": mask,
        "lengths": lengths,
        "task_id": torch.tensor([row["task_id"] for row in rows], dtype=torch.long),
        "task": [row["task"] for row in rows],
        "episode": [row["episode"] for row in rows],
    }
