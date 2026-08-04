"""Small supervised progress models and an online monotonic decoder.

The official RoboCasa365 target-composite labels define an ordered state
sequence for every task.  The neural model produces per-frame emissions; the
decoder uses only past and current emissions and only allows staying in the
same state or advancing.  It therefore has the same information boundary as
online deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProgressState:
    """One online progress estimate."""

    index: int
    done: bool
    confidence: float


@dataclass(frozen=True)
class PlanProgressState:
    """One causal estimate within the current reference plan."""

    index: int
    fraction: float
    similarity: float
    at_endpoint: bool


class CausalPlanAligner:
    """Align current frozen visual features to an ordered reference plan.

    The aligner has no clock.  At each update it may stay at the current node
    or advance by exactly one when the next node has persistent visual
    evidence.  An upstream stage tracker remains responsible for declaring
    the stage complete; reaching the final plan node is only an endpoint cue.
    """

    def __init__(
        self,
        plan_tokens: torch.Tensor,
        margin: float = 0.0,
        patience: int = 1,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be positive")
        self.margin = float(margin)
        self.patience = int(patience)
        self.set_plan(plan_tokens)

    def set_plan(self, plan_tokens: torch.Tensor) -> None:
        if plan_tokens.ndim != 2 or plan_tokens.shape[0] < 2:
            raise ValueError("plan_tokens must have shape [N >= 2, D]")
        self.plan_tokens = F.normalize(plan_tokens.detach().float(), dim=-1)
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self._evidence_count = 0

    def update(self, observation_feature: torch.Tensor) -> PlanProgressState:
        feature = F.normalize(observation_feature.detach().float().flatten(), dim=0)
        if feature.numel() != self.plan_tokens.shape[1]:
            raise ValueError("Observation and plan feature dimensions differ")
        similarity = self.plan_tokens @ feature.to(self.plan_tokens.device)
        if self._index < len(self.plan_tokens) - 1:
            supports_advance = bool(
                similarity[self._index + 1] >= similarity[self._index] + self.margin
            )
            self._evidence_count = self._evidence_count + 1 if supports_advance else 0
            if self._evidence_count >= self.patience:
                self._index += 1
                self._evidence_count = 0
        denominator = len(self.plan_tokens) - 1
        return PlanProgressState(
            index=self._index,
            fraction=self._index / denominator,
            similarity=float(similarity[self._index].item()),
            at_endpoint=self._index == denominator,
        )


class CausalTemporalPlanAligner:
    """History-aware alignment using appearance, start anchor, and motion.

    A short observation history is compared with candidate reference windows
    ending at the current or next node.  Reference windows are interpolated
    over several temporal spans, so a different execution speed does not
    require frame-for-frame synchronization.
    """

    def __init__(
        self,
        plan_tokens: torch.Tensor,
        history_length: int = 4,
        appearance_weight: float = 1.0,
        anchor_weight: float = 0.5,
        motion_weight: float = 0.5,
        max_reference_span: int = 3,
        max_advance: int = 1,
        margin: float = 0.0,
        patience: int = 1,
        motion_epsilon: float = 1e-6,
    ) -> None:
        if history_length < 2:
            raise ValueError("history_length must be at least two")
        if max_reference_span < 1 or max_advance < 1 or patience < 1:
            raise ValueError("span and patience must be positive")
        self.history_length = int(history_length)
        self.appearance_weight = float(appearance_weight)
        self.anchor_weight = float(anchor_weight)
        self.motion_weight = float(motion_weight)
        self.max_reference_span = int(max_reference_span)
        self.max_advance = int(max_advance)
        self.margin = float(margin)
        self.patience = int(patience)
        self.motion_epsilon = float(motion_epsilon)
        self.set_plan(plan_tokens)

    def set_plan(self, plan_tokens: torch.Tensor) -> None:
        if plan_tokens.ndim != 2 or plan_tokens.shape[0] < 2:
            raise ValueError("plan_tokens must have shape [N >= 2, D]")
        self.plan_tokens = F.normalize(plan_tokens.detach().float(), dim=-1)
        self.reset()

    def reset(self, start_feature: torch.Tensor | None = None) -> None:
        self._index = 0
        self._evidence_count = 0
        self._history: list[torch.Tensor] = []
        self._start_feature = None
        if start_feature is not None:
            feature = self._prepare_feature(start_feature)
            self._start_feature = feature
            self._history.append(feature)

    def _prepare_feature(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.normalize(feature.detach().float().flatten(), dim=0)
        if feature.numel() != self.plan_tokens.shape[1]:
            raise ValueError("Observation and plan feature dimensions differ")
        return feature.to(self.plan_tokens.device)

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return left.new_zeros(())
        return torch.dot(left / left_norm, right / right_norm)

    def _reference_window(self, end: int, span: int, steps: int) -> torch.Tensor:
        position = torch.linspace(
            float(end - span),
            float(end),
            steps=steps,
            device=self.plan_tokens.device,
        )
        left = torch.floor(position).long()
        right = torch.ceil(position).long()
        alpha = (position - left).unsqueeze(-1)
        return (1.0 - alpha) * self.plan_tokens[left] + alpha * self.plan_tokens[right]

    def _motion_similarity(self, candidate: int) -> torch.Tensor:
        if len(self._history) < 2 or candidate < 1:
            return self.plan_tokens.new_zeros(())
        actual = torch.stack(self._history)
        actual_delta = actual[1:] - actual[:-1]
        actual_delta = F.normalize(actual_delta, dim=-1)
        best = self.plan_tokens.new_tensor(-1.0)
        for span in range(1, min(candidate, self.max_reference_span) + 1):
            reference = self._reference_window(candidate, span, len(actual))
            reference_delta = F.normalize(reference[1:] - reference[:-1], dim=-1)
            score = torch.sum(actual_delta * reference_delta, dim=-1).mean()
            best = torch.maximum(best, score)
        return best

    def _score(self, feature: torch.Tensor, candidate: int) -> torch.Tensor:
        appearance = torch.dot(feature, self.plan_tokens[candidate])
        anchor = self._cosine(
            feature - self._start_feature,
            self.plan_tokens[candidate] - self.plan_tokens[0],
        )
        motion = self._motion_similarity(candidate)
        return (
            self.appearance_weight * appearance
            + self.anchor_weight * anchor
            + self.motion_weight * motion
        )

    def update(self, observation_feature: torch.Tensor) -> PlanProgressState:
        feature = self._prepare_feature(observation_feature)
        previous_feature = self._history[-1] if self._history else None
        if self._start_feature is None:
            self._start_feature = feature
        self._history.append(feature)
        self._history = self._history[-self.history_length :]

        current_score = self._score(feature, self._index)
        if self._index < len(self.plan_tokens) - 1 and previous_feature is not None:
            candidate_end = min(
                self._index + self.max_advance,
                len(self.plan_tokens) - 1,
            )
            candidate_scores = torch.stack(
                [self._score(feature, index) for index in range(self._index, candidate_end + 1)]
            )
            relative_best = int(torch.argmax(candidate_scores).item())
            best_index = self._index + relative_best
            best_score = candidate_scores[relative_best]
            has_motion = bool(
                torch.linalg.vector_norm(feature - previous_feature) > self.motion_epsilon
            )
            supports_advance = bool(
                has_motion
                and best_index > self._index
                and best_score >= current_score + self.margin
            )
            self._evidence_count = self._evidence_count + 1 if supports_advance else 0
            if self._evidence_count >= self.patience:
                self._index = best_index
                self._evidence_count = 0
                current_score = best_score

        denominator = len(self.plan_tokens) - 1
        return PlanProgressState(
            index=self._index,
            fraction=self._index / denominator,
            similarity=float(current_score.item()),
            at_endpoint=self._index == denominator,
        )


class ProgressModel(nn.Module):
    """Task-conditioned MLP or causal-GRU emission model."""

    def __init__(
        self,
        visual_dim: int,
        state_dim: int,
        task_state_counts: Sequence[int],
        architecture: str = "gru",
        hidden_dim: int = 256,
        task_embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if architecture not in {"mlp", "gru"}:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if not task_state_counts or min(task_state_counts) < 2:
            raise ValueError("Every task needs at least one active state and DONE")

        self.architecture = architecture
        self.visual_dim = int(visual_dim)
        self.state_dim = int(state_dim)
        self.task_state_counts = tuple(int(value) for value in task_state_counts)
        self.max_states = max(self.task_state_counts)
        self.hidden_dim = int(hidden_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.dropout = float(dropout)

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, 64),
            nn.GELU(),
        )
        self.task_embedding = nn.Embedding(len(self.task_state_counts), self.task_embedding_dim)
        fused_dim = self.hidden_dim + 64 + self.task_embedding_dim

        if architecture == "gru":
            self.temporal = nn.GRU(
                input_size=fused_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            head_input_dim = self.hidden_dim
        else:
            self.temporal = nn.Sequential(
                nn.Linear(fused_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input_dim = self.hidden_dim

        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.max_states),
        )

    def forward(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        task_id: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return masked emission logits and the optional recurrent state.

        Args:
            visual: ``[B, T, visual_dim]`` frozen GR00T features.
            state: ``[B, T, state_dim]`` normalized GR00T state vector.
            task_id: ``[B]`` task indices.
            hidden: optional GRU state for streaming inference.
        """

        if visual.ndim != 3 or state.ndim != 3:
            raise ValueError("visual and state must have [B, T, D] shapes")
        if visual.shape[:2] != state.shape[:2]:
            raise ValueError("visual/state batch and time axes differ")

        batch, steps = visual.shape[:2]
        task_feature = self.task_embedding(task_id).unsqueeze(1).expand(batch, steps, -1)
        fused = torch.cat(
            (self.visual_projection(visual), self.state_projection(state), task_feature), dim=-1
        )
        if self.architecture == "gru":
            temporal, next_hidden = self.temporal(fused, hidden)
        else:
            temporal = self.temporal(fused)
            next_hidden = None
        logits = self.head(temporal)

        counts = torch.as_tensor(self.task_state_counts, device=logits.device)[task_id]
        classes = torch.arange(self.max_states, device=logits.device).view(1, 1, -1)
        invalid = classes >= counts.view(-1, 1, 1)
        logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        return logits, next_hidden

    def config_dict(self) -> dict[str, object]:
        return {
            "visual_dim": self.visual_dim,
            "state_dim": self.state_dim,
            "task_state_counts": list(self.task_state_counts),
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "task_embedding_dim": self.task_embedding_dim,
            "dropout": self.dropout,
        }


class CausalMonotonicFilter:
    """Evidence-gated online decoder that only stays or advances one state.

    Unlike a duration prior, this decoder never advances merely because time
    has elapsed.  It requires the next-state emission to beat an absolute
    confidence threshold and the current-state emission for a configurable
    number of consecutive observations.
    """

    def __init__(
        self,
        num_states: int,
        threshold: float = 0.35,
        patience: int = 2,
        margin: float = 1.0,
        done_patience: int = 1,
    ) -> None:
        if num_states < 2:
            raise ValueError("num_states must include an active state and DONE")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if patience < 1 or done_patience < 1:
            raise ValueError("patience values must be positive")
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.num_states = int(num_states)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.margin = float(margin)
        self.done_patience = int(done_patience)
        self.reset()

    def reset(self) -> None:
        self._output_index = 0
        self._evidence_count = 0

    def update(self, logits: torch.Tensor) -> ProgressState:
        logits = logits.detach().float().flatten()[: self.num_states]
        if logits.numel() != self.num_states:
            raise ValueError("Emission count does not match num_states")
        probability = F.softmax(logits, dim=-1)
        if self._output_index < self.num_states - 1:
            current = self._output_index
            next_index = current + 1
            supports_advance = bool(
                probability[next_index] >= self.threshold
                and probability[next_index] >= self.margin * probability[current]
            )
            self._evidence_count = self._evidence_count + 1 if supports_advance else 0
            required = self.done_patience if next_index == self.num_states - 1 else self.patience
            if self._evidence_count >= required:
                self._output_index = next_index
                self._evidence_count = 0
        confidence = float(probability[self._output_index].item())
        return ProgressState(
            index=self._output_index,
            done=self._output_index == self.num_states - 1,
            confidence=confidence,
        )


class ProgressTracker:
    """Streaming wrapper around ``ProgressModel`` and the monotonic filter."""

    def __init__(
        self,
        model: ProgressModel,
        task_id: int,
        decoder_config: Mapping[str, float | int],
        device: torch.device | str,
    ) -> None:
        self.model = model.eval().to(device)
        self.task_id = int(task_id)
        self.device = torch.device(device)
        self.decoder = CausalMonotonicFilter(
            model.task_state_counts[self.task_id], **dict(decoder_config)
        )
        self._hidden: torch.Tensor | None = None

    def reset(self) -> None:
        self.decoder.reset()
        self._hidden = None

    @torch.inference_mode()
    def update(self, visual: torch.Tensor, state: torch.Tensor) -> ProgressState:
        visual = visual.to(self.device).reshape(1, 1, -1)
        state = state.to(self.device).reshape(1, 1, -1)
        task = torch.tensor([self.task_id], device=self.device)
        logits, self._hidden = self.model(visual, state, task, self._hidden)
        return self.decoder.update(logits[0, 0])


def load_progress_tracker(
    checkpoint_path: Path | str,
    task: str,
    device: torch.device | str = "cpu",
) -> ProgressTracker:
    """Load one task-specific streaming tracker from a training checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tasks = list(checkpoint["tasks"])
    if task not in tasks:
        raise KeyError(f"Unknown task {task!r}; checkpoint tasks are {tasks}")
    model = ProgressModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    task_id = tasks.index(task)
    decoder_config = checkpoint["decoder_config_by_task"][task]
    return ProgressTracker(model, task_id, decoder_config, device)
