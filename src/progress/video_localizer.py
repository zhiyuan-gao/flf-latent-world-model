"""Pure-video temporal localization for a query chunk inside a full stage video."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoLocalizationResult:
    start_index: int
    end_index: int
    progress: float
    score: float
    confidence_margin: float


@dataclass(frozen=True)
class SubsequenceDTWResult:
    start_index: int
    end_index: int
    progress: float
    path: tuple[int, ...]
    mean_cost: float
    confidence_margin: float


class GTVideoChunkLocalizer:
    """Locate a video-feature chunk inside its full reference sequence.

    Both inputs contain visual features only.  No action, proprioception,
    robot state, policy output, frame index, or elapsed time is consumed.
    """

    def __init__(
        self,
        appearance_weight: float = 1.0,
        motion_weight: float = 0.5,
        start_anchor_weight: float = 0.5,
    ) -> None:
        if min(appearance_weight, motion_weight, start_anchor_weight) < 0:
            raise ValueError("localization weights must be non-negative")
        if appearance_weight + motion_weight + start_anchor_weight == 0:
            raise ValueError("at least one localization weight must be positive")
        self.appearance_weight = float(appearance_weight)
        self.motion_weight = float(motion_weight)
        self.start_anchor_weight = float(start_anchor_weight)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.detach().float(), dim=-1)

    @staticmethod
    def _safe_normalize(values: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        return values / torch.clamp(norm, min=1e-8)

    @torch.inference_mode()
    def score_windows(
        self,
        reference_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        if reference_features.ndim != 2 or query_features.ndim != 2:
            raise ValueError("reference and query must have [T, D] shapes")
        if reference_features.shape[1] != query_features.shape[1]:
            raise ValueError("reference and query feature dimensions differ")
        if not 1 <= len(query_features) <= len(reference_features):
            raise ValueError("query length must lie within the reference length")

        reference = self._normalize(reference_features)
        query = self._normalize(query_features).to(reference.device)
        candidates = len(reference) - len(query) + 1
        appearance = reference.new_zeros(candidates)
        for offset in range(len(query)):
            appearance += reference[offset : offset + candidates] @ query[offset]
        appearance /= len(query)

        motion = reference.new_zeros(candidates)
        if len(query) > 1:
            query_delta = self._safe_normalize(query[1:] - query[:-1])
            reference_delta = self._safe_normalize(reference[1:] - reference[:-1])
            for offset in range(len(query) - 1):
                motion += reference_delta[offset : offset + candidates] @ query_delta[offset]
            motion /= len(query) - 1

        query_from_start = self._safe_normalize(query[-1:] - reference[:1])[0]
        candidate_end = reference[len(query) - 1 :]
        candidate_from_start = self._safe_normalize(candidate_end - reference[:1])
        anchor = candidate_from_start @ query_from_start

        return (
            self.appearance_weight * appearance
            + self.motion_weight * motion
            + self.start_anchor_weight * anchor
        )

    @torch.inference_mode()
    def localize(
        self,
        reference_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> VideoLocalizationResult:
        scores = self.score_windows(reference_features, query_features)
        start = int(torch.argmax(scores).item())
        end = start + len(query_features) - 1
        if len(scores) > 1:
            top = torch.topk(scores, k=2).values
            margin = float((top[0] - top[1]).item())
        else:
            margin = float("inf")
        denominator = max(len(reference_features) - 1, 1)
        return VideoLocalizationResult(
            start_index=start,
            end_index=end,
            progress=end / denominator,
            score=float(scores[start].item()),
            confidence_margin=margin,
        )


class SubsequenceDTWLocalizer:
    """Monotonically align a query to any reference subsequence.

    There is no fixed candidate window or global speed.  Between consecutive
    query frames the reference index may stay fixed or advance by any amount;
    large skips receive a soft quadratic penalty instead of a hard limit.
    """

    def __init__(
        self,
        motion_weight: float = 0.0,
        stay_penalty: float = 0.0,
        jump_penalty: float = 0.0,
    ) -> None:
        if min(motion_weight, stay_penalty, jump_penalty) < 0:
            raise ValueError("alignment weights and penalties must be non-negative")
        self.motion_weight = float(motion_weight)
        self.stay_penalty = float(stay_penalty)
        self.jump_penalty = float(jump_penalty)

    @staticmethod
    def _safe_motion_cost(query_delta: torch.Tensor, reference_delta: torch.Tensor) -> torch.Tensor:
        query_norm = torch.linalg.vector_norm(query_delta)
        reference_norm = torch.linalg.vector_norm(reference_delta, dim=-1)
        both_static = (query_norm <= 1e-8) & (reference_norm <= 1e-8)
        one_static = (query_norm <= 1e-8) ^ (reference_norm <= 1e-8)
        query_unit = query_delta / torch.clamp(query_norm, min=1e-8)
        reference_unit = reference_delta / torch.clamp(reference_norm[:, None], min=1e-8)
        cost = 1.0 - reference_unit @ query_unit
        cost = torch.where(both_static, torch.zeros_like(cost), cost)
        cost = torch.where(one_static, torch.ones_like(cost), cost)
        return cost

    @torch.inference_mode()
    def localize(
        self,
        reference_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> SubsequenceDTWResult:
        if reference_features.ndim != 2 or query_features.ndim != 2:
            raise ValueError("reference and query must have [T, D] shapes")
        if reference_features.shape[1] != query_features.shape[1]:
            raise ValueError("reference and query feature dimensions differ")
        if len(reference_features) < 1 or len(query_features) < 1:
            raise ValueError("reference and query must be non-empty")

        reference = F.normalize(reference_features.detach().float(), dim=-1)
        query = F.normalize(query_features.detach().float(), dim=-1).to(reference.device)
        appearance_cost = 1.0 - query @ reference.T
        query_steps, reference_steps = appearance_cost.shape
        cumulative = torch.full_like(appearance_cost, torch.inf)
        backpointer = torch.full(
            (query_steps, reference_steps),
            -1,
            dtype=torch.long,
            device=reference.device,
        )
        # Subsequence initialization: the query may start at any reference node.
        cumulative[0] = appearance_cost[0]

        previous_index = torch.arange(reference_steps, device=reference.device)[:, None]
        current_index = torch.arange(reference_steps, device=reference.device)[None, :]
        advance = current_index - previous_index
        valid_transition = advance >= 0
        transition = torch.where(
            advance == 0,
            torch.full_like(advance, self.stay_penalty, dtype=torch.float32),
            self.jump_penalty * torch.clamp(advance.float() - 1.0, min=0.0).square(),
        )
        reference_gram = reference @ reference.T
        reference_delta_norm = torch.sqrt(
            torch.clamp(2.0 - 2.0 * reference_gram, min=0.0)
        )

        for query_index in range(1, query_steps):
            query_delta = query[query_index] - query[query_index - 1]
            query_norm = torch.linalg.vector_norm(query_delta)
            query_unit = query_delta / torch.clamp(query_norm, min=1e-8)
            projected = reference @ query_unit
            motion_similarity = (projected[None, :] - projected[:, None]) / torch.clamp(
                reference_delta_norm, min=1e-8
            )
            both_static = (query_norm <= 1e-8) & (reference_delta_norm <= 1e-8)
            one_static = (query_norm <= 1e-8) ^ (reference_delta_norm <= 1e-8)
            motion_cost = 1.0 - motion_similarity
            motion_cost = torch.where(both_static, torch.zeros_like(motion_cost), motion_cost)
            motion_cost = torch.where(one_static, torch.ones_like(motion_cost), motion_cost)
            candidate = (
                cumulative[query_index - 1, :, None]
                + transition
                + self.motion_weight * motion_cost
            )
            candidate = candidate.masked_fill(~valid_transition, torch.inf)
            best_value, best_previous = torch.min(candidate, dim=0)
            cumulative[query_index] = appearance_cost[query_index] + best_value
            backpointer[query_index] = best_previous

        endpoint_cost = cumulative[-1]
        end = int(torch.argmin(endpoint_cost).item())
        if reference_steps > 1:
            two = torch.topk(endpoint_cost, k=2, largest=False).values
            confidence = float(((two[1] - two[0]) / query_steps).item())
        else:
            confidence = float("inf")
        path = [end]
        current = end
        for query_index in range(query_steps - 1, 0, -1):
            current = int(backpointer[query_index, current].item())
            path.append(current)
        path.reverse()
        denominator = max(reference_steps - 1, 1)
        return SubsequenceDTWResult(
            start_index=path[0],
            end_index=path[-1],
            progress=path[-1] / denominator,
            path=tuple(path),
            mean_cost=float((endpoint_cost[end] / query_steps).item()),
            confidence_margin=confidence,
        )
