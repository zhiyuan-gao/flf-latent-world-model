from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from progress.model import (  # noqa: E402
    CausalPlanAligner,
    CausalTemporalPlanAligner,
    CausalMonotonicFilter,
    ProgressModel,
    load_progress_tracker,
)
from progress.video_localizer import (  # noqa: E402
    GTVideoChunkLocalizer,
    SubsequenceDTWLocalizer,
)


def test_monotonic_filter_never_regresses_or_jumps() -> None:
    decoder = CausalMonotonicFilter(
        num_states=4, threshold=0.25, patience=1, margin=1.0, done_patience=1
    )
    emissions = [
        [8.0, 0.0, 0.0, 0.0],
        [0.0, 8.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],  # misleading evidence must not regress
        [0.0, 0.0, 8.0, 0.0],
        [0.0, 0.0, 0.0, 8.0],
    ]
    states = [decoder.update(torch.tensor(row)).index for row in emissions]
    delta = torch.diff(torch.tensor(states))
    assert torch.all(delta >= 0)
    assert torch.all(delta <= 1)
    assert states[-1] == 3


def test_model_masks_states_that_do_not_exist_for_task() -> None:
    model = ProgressModel(
        visual_dim=12,
        state_dim=4,
        task_state_counts=[3, 6],
        architecture="mlp",
        hidden_dim=16,
    )
    visual = torch.randn(2, 5, 12)
    state = torch.randn(2, 5, 4)
    logits, hidden = model(visual, state, torch.tensor([0, 1]))
    assert hidden is None
    assert logits.shape == (2, 5, 6)
    assert torch.all(logits[0, :, 3:] < -1e20)
    assert torch.all(torch.isfinite(logits[1]))


def test_gru_streaming_shape() -> None:
    model = ProgressModel(
        visual_dim=8,
        state_dim=3,
        task_state_counts=[4],
        architecture="gru",
        hidden_dim=16,
    )
    logits, hidden = model(
        torch.randn(1, 1, 8), torch.randn(1, 1, 3), torch.tensor([0])
    )
    assert logits.shape == (1, 1, 4)
    assert hidden is not None and hidden.shape == (1, 1, 16)


def test_checkpoint_loader_builds_streaming_tracker(tmp_path: Path) -> None:
    model = ProgressModel(
        visual_dim=8,
        state_dim=3,
        task_state_counts=[4],
        architecture="gru",
        hidden_dim=16,
    )
    checkpoint = tmp_path / "tracker.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config_dict(),
            "tasks": ["ExampleTask"],
            "decoder_config_by_task": {
                "ExampleTask": {
                    "threshold": 0.25,
                    "patience": 1,
                    "margin": 1.0,
                    "done_patience": 1,
                }
            },
        },
        checkpoint,
    )
    tracker = load_progress_tracker(checkpoint, "ExampleTask")
    state = tracker.update(torch.randn(8), torch.randn(3))
    assert 0 <= state.index < 4


def test_reference_plan_aligner_is_causal_and_monotonic() -> None:
    plan = torch.eye(4)
    aligner = CausalPlanAligner(plan, margin=0.0, patience=1)
    observations = [plan[0], plan[1], plan[0], plan[2], plan[3]]
    states = [aligner.update(observation) for observation in observations]
    indices = torch.tensor([state.index for state in states])
    assert torch.all(torch.diff(indices) >= 0)
    assert torch.all(torch.diff(indices) <= 1)
    assert states[-1].at_endpoint


def test_temporal_aligner_does_not_advance_on_a_frozen_frame() -> None:
    plan = torch.eye(4)
    aligner = CausalTemporalPlanAligner(plan, margin=-0.1, patience=1)
    first = aligner.update(plan[0])
    frozen = [aligner.update(plan[0]) for _ in range(16)]
    assert first.index == 0
    assert all(state.index == 0 for state in frozen)


def test_temporal_aligner_uses_ordered_history() -> None:
    plan = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    aligner = CausalTemporalPlanAligner(
        plan,
        appearance_weight=0.0,
        anchor_weight=0.0,
        motion_weight=1.0,
        max_advance=2,
        margin=-0.1,
    )
    states = [aligner.update(row) for row in plan]
    assert states[-1].index > states[0].index


def test_gt_video_chunk_localizer_recovers_exact_window() -> None:
    reference = torch.randn(12, 16)
    query = reference[5:9].clone()
    result = GTVideoChunkLocalizer().localize(reference, query)
    assert result.start_index == 5
    assert result.end_index == 8
    assert result.progress == 8 / 11


def test_subsequence_dtw_recovers_stays_and_nonuniform_jumps() -> None:
    reference = torch.eye(10)
    true_path = torch.tensor([2, 2, 3, 6, 7])
    query = reference[true_path]
    result = SubsequenceDTWLocalizer(
        motion_weight=0.0,
        stay_penalty=0.0,
        jump_penalty=0.0,
    ).localize(reference, query)
    assert result.path == tuple(true_path.tolist())
    assert result.start_index == 2
    assert result.end_index == 7
