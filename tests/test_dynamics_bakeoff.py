from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.data import CachedDynamicsWindows  # noqa: E402
from dynamics.evaluation import (  # noqa: E402
    EndpointMetricAccumulator,
    _far_shuffle_actions_within_task,
)
from dynamics.losses import (  # noqa: E402
    dynamics_loss,
    endpoint_loss,
    multi_horizon_endpoint_loss,
    multi_horizon_l1_loss,
    normalized_batch_metrics,
)
from dynamics.manifest import WindowRecord  # noqa: E402
from dynamics.model import (  # noqa: E402
    CausalMultiHorizonACPredictor,
    DirectEndpointACPredictor,
    FourHorizonACPredictor,
    block_causal_mask,
)


def test_window_contract_has_fixed_observation_times() -> None:
    row = WindowRecord("Example", "train", 2, 20, 1, "place")
    assert row.history_frames == (8, 12, 16, 20)
    assert row.future_frames == (24, 28, 32, 36)


def test_block_causal_mask_blocks_future_groups_only() -> None:
    mask = block_causal_mask(3, 2)
    assert mask.shape == (6, 6)
    assert not bool(mask[0, 1])  # same group is visible
    assert bool(mask[0, 2])  # future is blocked
    assert not bool(mask[4, 0])  # past is visible


def test_predictor_returns_teacher_and_open_loop_four_horizons() -> None:
    model = FourHorizonACPredictor(
        feature_dim=8,
        action_dim=3,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        action_depth=1,
    )
    history = torch.randn(2, 4, 2, 2, 8)
    future = torch.randn(2, 4, 2, 2, 8)
    actions = torch.randn(2, 4, 4, 3)
    output = model(history, future, actions)
    assert output["teacher_forced"].shape == future.shape
    assert output["rollout"].shape == future[:, :2].shape
    assert model.rollout(history, actions).shape == future.shape
    losses = dynamics_loss(output, future, history[:, -1])
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.action_encoder.input.weight.grad is not None


def test_direct_endpoint_predictor_uses_full_action_chunk() -> None:
    model = DirectEndpointACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=5,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        horizon=16,
    )
    current = torch.randn(2, 2, 2, 8)
    actions = torch.randn(2, 16, 3, requires_grad=True)
    anchor_state = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 2, 2, 8)
    prediction = model(current, actions, anchor_state)
    assert prediction.shape == target.shape
    losses = endpoint_loss(prediction, target, current)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert isinstance(model.action_input, torch.nn.Linear)
    assert isinstance(model.state_input, torch.nn.Linear)
    assert model.action_input.weight.grad is not None
    assert torch.isfinite(model.action_input.weight.grad).all()
    assert model.state_input.weight.grad is not None
    assert actions.grad is not None
    assert torch.all(actions.grad.square().sum(dim=(0, 2)) > 0)
    assert anchor_state.grad is not None
    assert torch.all(anchor_state.grad.square().sum(dim=0) > 0)
    assert model.config_dict()["horizon"] == 16


def test_causal_multi_horizon_predictor_blocks_later_action_gradients() -> None:
    model = CausalMultiHorizonACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=5,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        horizon=16,
        normalize_representations=True,
    )
    current = torch.randn(2, 2, 2, 8)
    actions = torch.randn(2, 16, 3, requires_grad=True)
    state = torch.randn(2, 5)
    prediction = model(current, actions, state)
    assert prediction.shape == (2, 4, 2, 2, 8)
    prediction[:, 0].square().mean().backward()
    assert actions.grad is not None
    assert torch.count_nonzero(actions.grad[:, :4]) > 0
    assert torch.count_nonzero(actions.grad[:, 4:]) == 0


def test_causal_multi_horizon_loss_supervises_all_four_times() -> None:
    current = torch.zeros(2, 2, 2, 3)
    target = torch.randn(2, 4, 2, 2, 3)
    prediction = torch.randn_like(target, requires_grad=True)
    losses = multi_horizon_endpoint_loss(prediction, target, current)
    assert all(f"h{offset}" in losses for offset in (4, 8, 12, 16))
    losses["loss"].backward()
    assert prediction.grad is not None
    assert torch.all(prediction.grad.square().sum(dim=(0, 2, 3, 4)) > 0)


def test_ac_normalized_predictor_outputs_affine_free_token_norms() -> None:
    model = CausalMultiHorizonACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=5,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        horizon=16,
        normalize_representations=True,
    )
    prediction = model(
        torch.randn(2, 2, 2, 8),
        torch.randn(2, 16, 3),
        torch.randn(2, 5),
    )
    assert torch.allclose(prediction.mean(dim=-1), torch.zeros_like(prediction[..., 0]), atol=1e-5)
    assert torch.allclose(
        prediction.std(dim=-1, unbiased=False),
        torch.ones_like(prediction[..., 0]),
        atol=1e-4,
    )
    assert model.config_dict()["normalize_representations"] is True


def test_multi_horizon_l1_matches_absolute_error_and_supervises_all_times() -> None:
    target = torch.randn(2, 4, 2, 2, 3)
    prediction = torch.randn_like(target, requires_grad=True)
    losses = multi_horizon_l1_loss(prediction, target)
    assert torch.allclose(losses["loss"], torch.abs(prediction - target).mean())
    assert all(f"h{offset}" in losses for offset in (4, 8, 12, 16))
    losses["loss"].backward()
    assert prediction.grad is not None
    assert torch.all(prediction.grad.square().sum(dim=(0, 2, 3, 4)) > 0)


def test_endpoint_loss_emphasizes_changed_tokens() -> None:
    current = torch.zeros(1, 2, 2, 2)
    target = current.clone()
    target[:, 0, 0] = 2.0
    prediction = target.clone()
    prediction[:, 0, 0] = 0.0
    weighted = endpoint_loss(prediction, target, current, dynamic_weight=1.0)
    unweighted = endpoint_loss(prediction, target, current, dynamic_weight=0.0)
    assert weighted["loss"] > unweighted["loss"]


def test_endpoint_metrics_retrieve_the_fourth_future() -> None:
    current = torch.zeros(2, 1, 1, 2)
    futures = torch.stack(
        [torch.full_like(current, float(value)) for value in (1, 2, 3, 4)],
        dim=1,
    )
    prediction = futures[:, -1].clone()
    metrics = EndpointMetricAccumulator()
    metrics.update(
        prediction,
        futures[:, -1],
        current,
        future_candidates=futures,
    )
    result = metrics.compute()
    assert result["endpoint_retrieval_accuracy"] == 1.0
    assert result["nearest_future_histogram"] == [0.0, 0.0, 0.0, 1.0]


def test_normalized_metrics_reward_beating_persistence() -> None:
    current = torch.zeros(2, 1, 1, 3)
    target = torch.ones(2, 4, 1, 1, 3)
    prediction = target.clone()
    metrics = normalized_batch_metrics(prediction, target, current)
    assert torch.allclose(metrics["normalized_error_per_horizon"], torch.zeros(4))
    assert torch.allclose(metrics["retrieval_accuracy_per_horizon"].mean(), torch.tensor(0.25))


def test_action_control_uses_distant_same_task_chunks_without_fixed_points() -> None:
    actions = torch.arange(6, dtype=torch.float32).view(6, 1, 1)
    tasks = ["a", "a", "a", "a", "b", "b"]
    shuffled = _far_shuffle_actions_within_task(actions, tasks)
    assert shuffled[:4, 0, 0].tolist() == [2.0, 3.0, 0.0, 1.0]
    assert shuffled[4:, 0, 0].tolist() == [5.0, 4.0]
    assert torch.all(shuffled != actions)


def test_cached_dataset_joins_actions_and_encoder_features(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    (manifest / "episodes/Example/train").mkdir(parents=True)
    (features / "Example/train").mkdir(parents=True)
    row = WindowRecord("Example", "train", 0, 12, 0, "pick")
    (manifest / "windows_train.jsonl").write_text(json.dumps(row.as_dict()) + "\n")
    (manifest / "action_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0], "count": 16})
    )
    np.savez_compressed(
        manifest / "episodes/Example/train/episode_000000.npz",
        actions=np.arange(64, dtype=np.float32).reshape(32, 2),
        subtask_index=np.zeros(32),
        stage_id=np.zeros(32),
    )
    frames = np.asarray([*row.history_frames, *row.future_frames])
    np.savez_compressed(
        features / "Example/train/episode_000000.npz",
        frames=frames,
        features=np.random.default_rng(0).normal(size=(8, 2, 2, 4)).astype(np.float16),
    )
    dataset = CachedDynamicsWindows(manifest, features, "train")
    sample = dataset[0]
    assert sample["history"].shape == (4, 2, 2, 4)
    assert sample["future"].shape == (4, 2, 2, 4)
    assert sample["actions"].shape == (4, 4, 2)
    assert sample["zero_actions"].shape == (4, 4, 2)
