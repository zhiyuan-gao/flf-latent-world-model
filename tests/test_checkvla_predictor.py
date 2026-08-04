from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import (  # noqa: E402
    CachedEndpointWindows,
    CachedMultiHorizonEndpointWindows,
    CachedRollingWindows,
    RollingWindowRecord,
    build_episode_windows,
    collate_endpoint_windows,
    collate_multi_horizon_endpoint_windows,
    nested_episode_splits,
)
from dynamics.losses import rolling_huber_loss  # noqa: E402
from dynamics.model import CheckVLARollingPredictor  # noqa: E402
from scripts.train_endpoint_predictor import (  # noqa: E402
    merge_endpoint_metric_rows,
)


def test_rolling_window_has_one_visual_target_per_action() -> None:
    row = RollingWindowRecord("Task", "train", 3, 7, 1, "pick", horizon=4)
    assert row.frames == (7, 8, 9, 10, 11)


def test_fixed_horizon_windows_drop_incomplete_episode_tail() -> None:
    length = 22
    table = pd.DataFrame(
        {
            "frame_index": np.arange(length),
            "subtask_idx": np.zeros(length, dtype=np.int64),
            "annotation.human.subtask_stage": np.zeros(length, dtype=np.int64),
        }
    )
    rows = build_episode_windows(
        "Task",
        "train",
        0,
        table,
        {0: "pick"},
        horizon=16,
        stride=4,
    )
    assert [row.current_frame for row in rows] == [0, 4]
    assert [row.current_frame + row.horizon for row in rows] == [16, 20]
    assert all(row.current_frame + row.horizon < length for row in rows)


def test_nested_episode_splits_preserve_reference_validation_and_test() -> None:
    reference = {
        "train": [0, 1],
        "val": [2],
        "test": [3, 4],
    }
    result = nested_episode_splits(
        list(range(10)),
        train_episodes=5,
        val_episodes=1,
        test_episodes=2,
        seed=7,
        reference_splits=reference,
    )
    assert set(reference["train"]).issubset(result["train"])
    assert result["val"] == reference["val"]
    assert result["test"] == reference["test"]
    assert len(result["train"]) == 5
    assert not (set(result["train"]) & set(result["val"] + result["test"]))


def test_endpoint_ddp_metrics_merge_sufficient_statistics() -> None:
    rows = [
        {
            "count": 2,
            "mse": 2.0,
            "persistence_mse": 4.0,
            "normalized_error": 0.5,
            "improvement_over_persistence": 0.5,
            "delta_cosine": 0.25,
        },
        {
            "count": 3,
            "mse": 5.0,
            "persistence_mse": 10.0,
            "normalized_error": 0.5,
            "improvement_over_persistence": 0.5,
            "delta_cosine": 0.75,
        },
    ]
    merged = merge_endpoint_metric_rows(rows)
    assert merged["count"] == 5
    assert np.isclose(merged["mse"], 3.8)
    assert np.isclose(merged["persistence_mse"], 7.6)
    assert np.isclose(merged["normalized_error"], 0.5)
    assert np.isclose(merged["delta_cosine"], 0.55)


def test_checkvla_predictor_teacher_force_and_self_rollout() -> None:
    model = CheckVLARollingPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=5,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        max_horizon=4,
    )
    latents = torch.randn(2, 5, 2, 2, 8)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 5)
    teacher = model.teacher_forced(latents, actions, state)
    rollout = model.rollout(latents[:, 0], actions, state, detach_context=True)
    dispatched_teacher = model(latents, actions, state, mode="teacher_forcing")
    dispatched_rollout = model(
        latents,
        actions,
        state,
        mode="self_rollout",
        rollout_horizon=4,
    )
    assert teacher.shape == latents[:, 1:].shape
    assert rollout.shape == latents[:, 1:].shape
    assert torch.equal(teacher, dispatched_teacher)
    assert torch.equal(rollout, dispatched_rollout)
    loss = rolling_huber_loss(teacher, latents[:, 1:])["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert isinstance(model.action_input, torch.nn.Linear)
    assert isinstance(model.state_input, torch.nn.Linear)
    assert model.action_input.weight.grad is not None


def test_checkvla_predictor_can_load_legacy_condition_encoders() -> None:
    model = CheckVLARollingPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=5,
        grid_size=2,
        model_dim=32,
        depth=1,
        heads=4,
        max_horizon=4,
        normalize_condition_inputs=True,
    )
    assert isinstance(model.action_input[0], torch.nn.LayerNorm)
    assert isinstance(model.state_input[0], torch.nn.LayerNorm)


def test_cached_rolling_dataset_does_not_return_future_states(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    (manifest / "episodes/Task/train").mkdir(parents=True)
    (features / "Task/train").mkdir(parents=True)
    row = RollingWindowRecord("Task", "train", 0, 1, 0, "pick", horizon=4)
    (manifest / "windows_train.jsonl").write_text(json.dumps(row.as_dict()) + "\n")
    (manifest / "action_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0], "count": 4})
    )
    (manifest / "state_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": 1})
    )
    np.savez_compressed(
        manifest / "episodes/Task/train/episode_000000.npz",
        actions=np.zeros((8, 2), dtype=np.float32),
        states=np.arange(24, dtype=np.float32).reshape(8, 3),
    )
    np.savez_compressed(
        features / "Task/train/episode_000000.npz",
        frames=np.asarray(row.frames),
        features=np.zeros((5, 2, 2, 6), dtype=np.float16),
    )
    sample = CachedRollingWindows(manifest, features, "train")[0]
    assert sample["latents"].shape == (5, 2, 2, 6)
    assert sample["actions"].shape == (4, 2)
    assert sample["anchor_state"].shape == (3,)
    assert "future_states" not in sample


def test_cached_endpoint_dataset_returns_only_endpoint_visual_target(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    (manifest / "episodes/Task/train").mkdir(parents=True)
    (features / "Task/train").mkdir(parents=True)
    row = RollingWindowRecord("Task", "train", 0, 1, 0, "pick", horizon=4)
    (manifest / "windows_train.jsonl").write_text(json.dumps(row.as_dict()) + "\n")
    (manifest / "action_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0], "count": 4})
    )
    (manifest / "state_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": 1})
    )
    np.savez_compressed(
        manifest / "episodes/Task/train/episode_000000.npz",
        actions=np.zeros((8, 2), dtype=np.float32),
        states=np.arange(24, dtype=np.float32).reshape(8, 3),
    )
    endpoint_features = np.stack(
        [
            np.zeros((2, 2, 6), dtype=np.float16),
            np.ones((2, 2, 6), dtype=np.float16),
        ]
    )
    np.save(
        features / "Task/train/episode_000000.frames.npy",
        np.asarray((row.current_frame, row.current_frame + row.horizon)),
    )
    np.save(
        features / "Task/train/episode_000000.features.npy",
        endpoint_features,
    )

    dataset = CachedEndpointWindows(manifest, features, "train")
    sample = dataset[0]
    batch = collate_endpoint_windows([sample])
    assert sample["current"].shape == (2, 2, 6)
    assert sample["target"].shape == (2, 2, 6)
    assert sample["actions"].shape == (4, 2)
    assert sample["anchor_state"].shape == (3,)
    assert "latents" not in sample
    assert "future_states" not in sample
    assert torch.equal(batch["current"][0], sample["current"])
    assert torch.equal(batch["target"][0], sample["target"])


def test_cached_multi_horizon_dataset_returns_four_real_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    (manifest / "episodes/Task/train").mkdir(parents=True)
    (features / "Task/train").mkdir(parents=True)
    row = RollingWindowRecord("Task", "train", 0, 0, 0, "pick", horizon=16)
    (manifest / "windows_train.jsonl").write_text(json.dumps(row.as_dict()) + "\n")
    (manifest / "action_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0], "count": 16})
    )
    (manifest / "state_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": 1})
    )
    np.savez_compressed(
        manifest / "episodes/Task/train/episode_000000.npz",
        actions=np.zeros((20, 2), dtype=np.float32),
        states=np.zeros((20, 3), dtype=np.float32),
    )
    frames = np.asarray((0, 4, 8, 12, 16))
    values = np.stack(
        [np.full((2, 2, 6), value, dtype=np.float16) for value in range(5)]
    )
    np.save(features / "Task/train/episode_000000.frames.npy", frames)
    np.save(features / "Task/train/episode_000000.features.npy", values)
    sample = CachedMultiHorizonEndpointWindows(manifest, features, "train")[0]
    batch = collate_multi_horizon_endpoint_windows([sample])
    assert sample["current"].shape == (2, 2, 6)
    assert sample["targets"].shape == (4, 2, 2, 6)
    assert sample["horizons"] == (4, 8, 12, 16)
    assert torch.equal(batch["targets"][0], sample["targets"])


def test_cached_multi_horizon_dataset_can_apply_ac_target_normalization(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    (manifest / "episodes/Task/train").mkdir(parents=True)
    (features / "Task/train").mkdir(parents=True)
    row = RollingWindowRecord("Task", "train", 0, 0, 0, "pick", horizon=16)
    (manifest / "windows_train.jsonl").write_text(json.dumps(row.as_dict()) + "\n")
    (manifest / "action_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0], "count": 16})
    )
    (manifest / "state_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": 1})
    )
    np.savez_compressed(
        manifest / "episodes/Task/train/episode_000000.npz",
        actions=np.zeros((20, 2), dtype=np.float32),
        states=np.zeros((20, 3), dtype=np.float32),
    )
    frames = np.asarray((0, 4, 8, 12, 16))
    channel_pattern = np.arange(6, dtype=np.float32)
    values = np.stack(
        [
            np.broadcast_to((index + 1) * channel_pattern + index, (2, 2, 6))
            for index in range(5)
        ]
    ).astype(np.float16)
    np.save(features / "Task/train/episode_000000.frames.npy", frames)
    np.save(features / "Task/train/episode_000000.features.npy", values)

    sample = CachedMultiHorizonEndpointWindows(
        manifest,
        features,
        "train",
        normalize_representations=True,
    )[0]
    all_latents = torch.cat((sample["current"][None], sample["targets"]), dim=0)
    assert torch.allclose(
        all_latents.mean(dim=-1), torch.zeros_like(all_latents[..., 0]), atol=1e-5
    )
    assert torch.allclose(
        all_latents.std(dim=-1, unbiased=False),
        torch.ones_like(all_latents[..., 0]),
        atol=1e-5,
    )
