"""Training losses and normalized bake-off metrics."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _state_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target)


def _motion(values: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return torch.diff(torch.cat((current[:, None], values), dim=1), dim=1)


def dynamics_loss(
    predictions: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    current: torch.Tensor,
    teacher_weight: float = 1.0,
    rollout_weight: float = 1.0,
    motion_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    teacher = _state_loss(predictions["teacher_forced"], target)
    rollout_steps = predictions["rollout"].shape[1]
    if not 1 <= rollout_steps <= target.shape[1]:
        raise ValueError("rollout prediction has an invalid horizon")
    rollout_target = target[:, :rollout_steps]
    rollout = _state_loss(predictions["rollout"], rollout_target)
    target_motion = _motion(target, current)
    teacher_motion = _state_loss(
        _motion(predictions["teacher_forced"], current), target_motion
    )
    rollout_motion = _state_loss(
        _motion(predictions["rollout"], current), target_motion[:, :rollout_steps]
    )
    motion = 0.5 * (teacher_motion + rollout_motion)
    total = teacher_weight * teacher + rollout_weight * rollout + motion_weight * motion
    return {
        "loss": total,
        "teacher_state": teacher,
        "rollout_state": rollout,
        "teacher_motion": teacher_motion,
        "rollout_motion": rollout_motion,
    }


def endpoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    dynamic_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Direct endpoint loss with bounded emphasis on changing spatial tokens."""
    if prediction.shape != target.shape or current.shape != target.shape:
        raise ValueError("prediction, target, and current must have identical shapes")
    element_error = F.smooth_l1_loss(prediction, target, reduction="none")
    token_error = element_error.mean(dim=-1)
    token_change = torch.square(target - current).mean(dim=-1).detach()
    spatial_mean = token_change.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    relative_change = (token_change / spatial_mean).clamp(max=8.0)
    weights = 1.0 + float(dynamic_weight) * relative_change
    weighted_state = (token_error * weights).sum() / weights.sum().clamp_min(1e-8)
    unweighted_state = token_error.mean()
    return {
        "loss": weighted_state,
        "weighted_state": weighted_state,
        "unweighted_state": unweighted_state,
    }


def semantic_proprio_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """PandaOmron state loss with sign-invariant quaternion supervision."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("state prediction and target must match [B, state_dim]")
    if prediction.shape[-1] != 16:
        raise ValueError("PandaOmron semantic state loss requires 16 dimensions")
    if state_mean.shape != (16,) or state_std.shape != (16,):
        raise ValueError("state normalization tensors must have 16 dimensions")

    base_position = F.smooth_l1_loss(prediction[:, 0:3], target[:, 0:3])
    eef_position = F.smooth_l1_loss(prediction[:, 7:10], target[:, 7:10])
    gripper = F.smooth_l1_loss(prediction[:, 14:16], target[:, 14:16])

    prediction_raw = prediction * state_std + state_mean
    target_raw = target * state_std + state_mean

    def quaternion_distance(start: int, stop: int) -> torch.Tensor:
        predicted_q = F.normalize(prediction_raw[:, start:stop], dim=-1, eps=1e-8)
        target_q = F.normalize(target_raw[:, start:stop], dim=-1, eps=1e-8)
        return (1.0 - torch.abs((predicted_q * target_q).sum(dim=-1))).mean()

    base_rotation = quaternion_distance(3, 7)
    eef_rotation = quaternion_distance(10, 14)
    total = torch.stack(
        (base_position, base_rotation, eef_position, eef_rotation, gripper)
    ).mean()
    return {
        "state": total,
        "state_base_position": base_position,
        "state_base_rotation": base_rotation,
        "state_eef_position": eef_position,
        "state_eef_rotation": eef_rotation,
        "state_gripper": gripper,
    }


def single_step_proprio_loss(
    outputs: Mapping[str, torch.Tensor],
    visual_target: torch.Tensor,
    state_target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    proprio_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Primary visual Huber plus a weighted semantic next-state loss."""
    visual = outputs["visual"]
    state = outputs["state"]
    if visual.shape != visual_target.shape:
        raise ValueError("visual prediction and target must have identical shapes")
    visual_huber = F.smooth_l1_loss(visual, visual_target)
    state_losses = semantic_proprio_loss(
        state,
        state_target,
        state_mean,
        state_std,
    )
    total = visual_huber + float(proprio_weight) * state_losses["state"]
    return {
        "loss": total,
        "visual": visual_huber,
        "weighted_state": float(proprio_weight) * state_losses["state"],
        **state_losses,
    }


def single_step_visual_loss(
    outputs: Mapping[str, torch.Tensor],
    visual_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Visual-only Huber objective for the no-future-proprio ablation."""
    visual = outputs["visual"]
    if visual.shape != visual_target.shape:
        raise ValueError("visual prediction and target must have identical shapes")
    visual_huber = F.smooth_l1_loss(visual, visual_target)
    return {"loss": visual_huber, "visual": visual_huber}


def multi_horizon_endpoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    dynamic_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Equal-weight endpoint loss at t+4/t+8/t+12/t+16."""
    if prediction.ndim != 5 or prediction.shape != target.shape:
        raise ValueError("multi-horizon prediction and target must match [B, H, Y, X, C]")
    if current.shape != target.shape[:1] + target.shape[2:]:
        raise ValueError("current must have [B, Y, X, C]")
    expanded_current = current[:, None].expand_as(target)
    result = endpoint_loss(
        prediction,
        target,
        expanded_current,
        dynamic_weight=dynamic_weight,
    )
    per_horizon = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none"
    ).mean(dim=(0, 2, 3, 4))
    for index, offset in enumerate((4, 8, 12, 16)):
        result[f"h{offset}"] = per_horizon[index]
    return result


def multi_horizon_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """V-JEPA2-AC-style L1 loss on four normalized endpoint grids."""
    if prediction.ndim != 5 or prediction.shape != target.shape:
        raise ValueError("multi-horizon prediction and target must match [B, H, Y, X, C]")
    element_error = torch.abs(prediction - target)
    per_horizon = element_error.mean(dim=(0, 2, 3, 4))
    result = {"loss": per_horizon.mean(), "l1_state": per_horizon.mean()}
    for index, offset in enumerate((4, 8, 12, 16)):
        result[f"h{offset}"] = per_horizon[index]
    return result


def block_rolling_ac_loss(
    outputs: Mapping[str, torch.Tensor],
    visual_targets: torch.Tensor,
    future_state_targets: torch.Tensor,
    proprio_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Joint AC-normalized teacher, self-rollout, and nominal-state objective."""
    teacher = outputs["teacher_forced"]
    rollout = outputs["rollout"]
    nominal_states = outputs["nominal_states"]
    if teacher.shape != visual_targets.shape or rollout.shape != visual_targets.shape:
        raise ValueError("visual predictions and targets must match [B, 4, Y, X, C]")
    if nominal_states[:, 1:].shape != future_state_targets.shape:
        raise ValueError("nominal rollout and future state targets must match")
    teacher_l1 = torch.abs(teacher - visual_targets).mean()
    rollout_l1 = torch.abs(rollout - visual_targets).mean()
    proprio_l1 = torch.abs(nominal_states[:, 1:] - future_state_targets).mean()
    visual = teacher_l1 + rollout_l1
    total = visual + float(proprio_weight) * proprio_l1
    return {
        "loss": total,
        "visual": visual,
        "teacher_l1": teacher_l1,
        "rollout_l1": rollout_l1,
        "proprio_l1": proprio_l1,
    }


def rolling_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """CheckVLA's latent objective: Huber loss at every predicted step."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    per_element = F.smooth_l1_loss(prediction, target, reduction="none")
    per_step = per_element.flatten(2).mean(dim=2)
    return {
        "loss": per_step.mean(),
        "per_step": per_step.mean(dim=0),
    }


@torch.no_grad()
def normalized_batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reduce_dims = tuple(range(2, target.ndim))
    squared = torch.square(prediction - target).mean(dim=reduce_dims)
    persistence = torch.square(current[:, None] - target).mean(dim=reduce_dims)
    normalized = squared / persistence.clamp_min(1e-8)

    predicted_motion = _motion(prediction, current)
    target_motion = _motion(target, current)
    motion_error = torch.square(predicted_motion - target_motion).mean(dim=reduce_dims)
    zero_motion_error = torch.square(target_motion).mean(dim=reduce_dims)
    normalized_motion = motion_error / zero_motion_error.clamp_min(1e-8)

    # At each horizon, retrieve the matching GT time among this sample's four futures.
    batch, horizons = prediction.shape[:2]
    flat_prediction = prediction.reshape(batch, horizons, -1)
    flat_target = target.reshape(batch, horizons, -1)
    distances = torch.square(
        flat_prediction[:, :, None] - flat_target[:, None, :]
    ).mean(dim=-1)
    retrieval = (
        distances.argmin(dim=-1)
        == torch.arange(horizons, device=prediction.device).view(1, -1)
    ).float()
    return {
        "mse_per_horizon": squared.mean(dim=0),
        "persistence_mse_per_horizon": persistence.mean(dim=0),
        "normalized_error_per_horizon": normalized.mean(dim=0),
        "motion_mse_per_horizon": motion_error.mean(dim=0),
        "zero_motion_mse_per_horizon": zero_motion_error.mean(dim=0),
        "normalized_motion_per_horizon": normalized_motion.mean(dim=0),
        "retrieval_accuracy_per_horizon": retrieval.mean(dim=0),
    }
