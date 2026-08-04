from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamics.checkvla_data import (  # noqa: E402
    CachedSingleStepDynamicsWindows,
    RollingWindowRecord,
    collate_single_step_dynamics_windows,
)
from dynamics.evaluation import (  # noqa: E402
    SingleStepMetricAccumulator,
    evaluate_single_step_proprio_model,
    merge_single_step_evaluations,
)
from dynamics.losses import (  # noqa: E402
    semantic_proprio_loss,
    single_step_proprio_loss,
    single_step_visual_loss,
)
from dynamics.model import SingleStepProprioACPredictor  # noqa: E402


def make_single_step_cache(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    (manifest / "episodes/Task/train").mkdir(parents=True)
    (features / "Task/train").mkdir(parents=True)
    row = RollingWindowRecord("Task", "train", 0, 0, 2, "pick", horizon=16)
    (manifest / "windows_train.jsonl").write_text(json.dumps(row.as_dict()) + "\n")
    (manifest / "action_stats.json").write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": 16})
    )
    (manifest / "state_stats.json").write_text(
        json.dumps({"mean": [0.0] * 16, "std": [1.0] * 16, "count": 1})
    )
    actions = np.arange(60, dtype=np.float32).reshape(20, 3)
    states = np.arange(320, dtype=np.float32).reshape(20, 16)
    np.savez_compressed(
        manifest / "episodes/Task/train/episode_000000.npz",
        actions=actions,
        states=states,
    )
    np.save(features / "Task/train/episode_000000.frames.npy", np.asarray((0, 4, 16)))
    values = np.stack(
        [np.full((2, 2, 8), value, dtype=np.float16) for value in (0, 4, 16)]
    )
    np.save(features / "Task/train/episode_000000.features.npy", values)
    return manifest, features


def test_single_step_dataset_uses_only_t_to_t4(tmp_path: Path) -> None:
    manifest, features = make_single_step_cache(tmp_path)
    dataset = CachedSingleStepDynamicsWindows(manifest, features, "train")
    sample = dataset[0]
    batch = collate_single_step_dynamics_windows([sample])
    assert sample["horizon"] == 4
    assert sample["actions"].shape == (4, 3)
    assert torch.equal(sample["actions"], torch.arange(12).reshape(4, 3))
    assert torch.all(sample["current"] == 0)
    assert torch.all(sample["target"] == 4)
    assert torch.equal(sample["anchor_state"], torch.arange(16).float())
    assert torch.equal(sample["target_state"], torch.arange(64, 80).float())
    assert "target_state" in batch
    assert dataset.dropped_missing_features == 0


def test_single_step_predictor_returns_direct_visual_and_state() -> None:
    model = SingleStepProprioACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=16,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        action_steps=4,
    )
    current = torch.full((2, 2, 2, 8), 100.0)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 16)
    outputs = model(current, actions, state)
    assert outputs["visual"].shape == current.shape
    assert outputs["state"].shape == state.shape
    assert not torch.allclose(outputs["visual"], current)
    assert isinstance(model.action_input, torch.nn.Linear)
    assert isinstance(model.state_input, torch.nn.Linear)


def test_single_step_visual_only_predictor_keeps_state_condition_without_head() -> None:
    torch.manual_seed(7)
    joint = SingleStepProprioACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=16,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        action_steps=4,
        predict_state=True,
    )
    torch.manual_seed(7)
    visual_only = SingleStepProprioACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=16,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        action_steps=4,
        predict_state=False,
    )
    for name, value in visual_only.state_dict().items():
        assert torch.equal(value, joint.state_dict()[name]), name
    current = torch.randn(2, 2, 2, 8)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 16)
    outputs = visual_only(current, actions, state)
    assert set(outputs) == {"visual"}
    assert visual_only.state_output is None
    assert isinstance(visual_only.state_input, torch.nn.Linear)
    assert visual_only.config_dict()["predict_state"] is False


def test_single_step_no_proprio_predictor_removes_only_state_condition() -> None:
    torch.manual_seed(11)
    conditioned = SingleStepProprioACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=16,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        action_steps=4,
        predict_state=False,
        use_proprio_condition=True,
    )
    torch.manual_seed(11)
    no_proprio = SingleStepProprioACPredictor(
        feature_dim=8,
        action_dim=3,
        state_dim=16,
        grid_size=2,
        model_dim=32,
        depth=2,
        heads=4,
        action_steps=4,
        predict_state=False,
        use_proprio_condition=False,
    )
    for name, value in no_proprio.state_dict().items():
        assert torch.equal(value, conditioned.state_dict()[name]), name
    current = torch.randn(2, 2, 2, 8)
    actions = torch.randn(2, 4, 3)
    outputs = no_proprio(current, actions, None)
    assert set(outputs) == {"visual"}
    assert no_proprio.state_input is None
    assert no_proprio.state_type is None
    assert no_proprio.config_dict()["use_proprio_condition"] is False
    with torch.no_grad():
        try:
            no_proprio(current, actions, torch.randn(2, 16))
        except ValueError as error:
            assert "must be None" in str(error)
        else:
            raise AssertionError("no-proprio model accepted a proprio tensor")


def test_semantic_proprio_loss_is_quaternion_sign_invariant() -> None:
    target = torch.zeros(2, 16)
    prediction = target.clone()
    target[:, 3] = 1.0
    target[:, 10] = 1.0
    prediction[:, 3] = -1.0
    prediction[:, 10] = -1.0
    losses = semantic_proprio_loss(
        prediction,
        target,
        torch.zeros(16),
        torch.ones(16),
    )
    assert torch.allclose(losses["state"], torch.tensor(0.0))


def test_joint_loss_weights_state_by_point_one() -> None:
    visual_target = torch.zeros(2, 2, 2, 8)
    state_target = torch.zeros(2, 16)
    state_target[:, 3] = 1.0
    state_target[:, 10] = 1.0
    outputs = {
        "visual": torch.ones_like(visual_target),
        "state": state_target.clone(),
    }
    losses = single_step_proprio_loss(
        outputs,
        visual_target,
        state_target,
        torch.zeros(16),
        torch.ones(16),
        proprio_weight=0.1,
    )
    assert torch.allclose(losses["weighted_state"], 0.1 * losses["state"])
    assert torch.allclose(losses["loss"], losses["visual"] + losses["weighted_state"])


def test_visual_only_loss_has_no_state_term() -> None:
    target = torch.zeros(2, 2, 2, 8)
    losses = single_step_visual_loss({"visual": torch.ones_like(target)}, target)
    assert set(losses) == {"loss", "visual"}
    assert torch.equal(losses["loss"], losses["visual"])


def test_single_step_metrics_detect_future_prediction_and_merge() -> None:
    current = torch.zeros(2, 2, 2, 4)
    target = torch.ones_like(current)
    state_current = torch.zeros(2, 16)
    state_target = torch.ones(2, 16)
    state_target[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state_target[:, 10:14] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    outputs = {"visual": target.clone(), "state": state_target.clone()}
    accumulator = SingleStepMetricAccumulator(dynamic_threshold=0.5)
    accumulator.update(
        outputs,
        target,
        current,
        state_target,
        state_current,
        torch.zeros(16),
        torch.ones(16),
    )
    metrics = accumulator.compute()
    assert metrics["visual_normalized_error"] == 0.0
    assert metrics["visual_future_closer_fraction"] == 1.0
    assert metrics["state_future_closer_fraction"] == 1.0
    merged = merge_single_step_evaluations(
        [
            {"overall": metrics, "by_task": {"Task": metrics}},
            {"overall": metrics, "by_task": {"Task": metrics}},
        ]
    )
    assert merged["overall"]["count"] == 4
    assert merged["overall"]["visual_normalized_error"] == 0.0
    assert merged["by_task"]["Task"]["visual_future_closer_fraction"] == 1.0


def test_single_step_evaluation_keeps_global_state_stats_for_task_slices() -> None:
    class PerfectModel(torch.nn.Module):
        def forward(self, current, actions, anchor_state):
            return {"visual": current + 1.0, "state": anchor_state + 1.0}

    current = torch.zeros(2, 2, 2, 4)
    anchor_state = torch.zeros(2, 16)
    target_state = torch.ones(2, 16)
    target_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target_state[:, 10:14] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    batch = {
        "current": current,
        "target": current + 1.0,
        "actions": torch.zeros(2, 4, 3),
        "zero_actions": torch.zeros(2, 4, 3),
        "anchor_state": anchor_state,
        "target_state": target_state,
        "task": ["A", "B"],
    }
    metrics = evaluate_single_step_proprio_model(
        PerfectModel(),
        [batch],
        torch.device("cpu"),
        torch.zeros(16),
        torch.ones(16),
        use_amp=False,
        action_controls=False,
    )
    assert metrics["overall"]["count"] == 2
    assert metrics["by_task"]["A"]["count"] == 1
    assert metrics["by_task"]["B"]["count"] == 1


def test_single_step_evaluation_accepts_visual_only_outputs() -> None:
    class VisualOnlyModel(torch.nn.Module):
        def forward(self, current, actions, anchor_state):
            return {"visual": current + 1.0}

    current = torch.zeros(2, 2, 2, 4)
    batch = {
        "current": current,
        "target": current + 1.0,
        "actions": torch.zeros(2, 4, 3),
        "zero_actions": torch.zeros(2, 4, 3),
        "anchor_state": torch.zeros(2, 16),
        "target_state": torch.ones(2, 16),
        "task": ["A", "B"],
    }
    metrics = evaluate_single_step_proprio_model(
        VisualOnlyModel(),
        [batch],
        torch.device("cpu"),
        torch.zeros(16),
        torch.ones(16),
        use_amp=False,
        action_controls=False,
    )
    assert metrics["overall"]["visual_normalized_error"] == 0.0
    assert "state_normalized_error" not in metrics["overall"]
    merged = merge_single_step_evaluations([metrics, metrics])
    assert merged["overall"]["count"] == 4
    assert "state_normalized_error" not in merged["overall"]


def test_single_step_evaluation_does_not_pass_proprio_to_no_input_model() -> None:
    class NoProprioModel(torch.nn.Module):
        use_proprio_condition = False

        def forward(self, current, actions, anchor_state):
            assert anchor_state is None
            return {"visual": current + 1.0}

    current = torch.zeros(2, 2, 2, 4)
    batch = {
        "current": current,
        "target": current + 1.0,
        "actions": torch.zeros(2, 4, 3),
        "zero_actions": torch.zeros(2, 4, 3),
        "anchor_state": torch.full((2, 16), 123.0),
        "target_state": torch.full((2, 16), 456.0),
        "task": ["A", "B"],
    }
    metrics = evaluate_single_step_proprio_model(
        NoProprioModel(),
        [batch],
        torch.device("cpu"),
        torch.zeros(16),
        torch.ones(16),
        use_amp=False,
        action_controls=False,
    )
    assert metrics["overall"]["visual_normalized_error"] == 0.0
    assert "state_normalized_error" not in metrics["overall"]
