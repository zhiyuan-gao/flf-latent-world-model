"""Comparable normalized evaluation for frozen-encoder dynamics models."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch

from .losses import normalized_batch_metrics, semantic_proprio_loss


def _far_shuffle_actions_within_task(
    actions: torch.Tensor,
    tasks: Sequence[str],
) -> torch.Tensor:
    """Pair each sample with a distant same-task action chunk.

    Validation windows are stored in temporal order.  A one-position roll would
    therefore compare adjacent stride-4 windows whose 16-step action chunks
    overlap by 12 steps.  A half-group cyclic shift has no fixed points for
    groups larger than one and provides a substantially less correlated control.
    """
    shuffled = actions.clone()
    for task in sorted(set(tasks)):
        indices = [index for index, value in enumerate(tasks) if value == task]
        source = torch.tensor(indices, device=actions.device)
        if len(indices) > 1:
            shift = max(1, len(indices) // 2)
            shuffled[source] = actions[source.roll(shifts=shift)]
        else:
            shuffled[source] = actions[source].flip(1)
    return shuffled


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.count = 0
        self.correct_better_zero = 0.0
        self.correct_better_shuffle = 0.0
        self.sensitivity_zero = 0.0
        self.sensitivity_shuffle = 0.0
        self.controls_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        current: torch.Tensor,
        zero_prediction: torch.Tensor | None = None,
        shuffled_prediction: torch.Tensor | None = None,
    ) -> None:
        metrics = normalized_batch_metrics(prediction, target, current)
        batch = len(prediction)
        for key, value in metrics.items():
            detached = value.detach().float().cpu() * batch
            self.sums[key] = detached if key not in self.sums else self.sums[key] + detached
        if zero_prediction is not None and shuffled_prediction is not None:
            reduce_dims = tuple(range(1, target.ndim))
            correct_error = torch.square(prediction - target).mean(dim=reduce_dims)
            zero_error = torch.square(zero_prediction - target).mean(dim=reduce_dims)
            shuffled_error = torch.square(shuffled_prediction - target).mean(dim=reduce_dims)
            self.correct_better_zero += float((correct_error < zero_error).sum())
            self.correct_better_shuffle += float((correct_error < shuffled_error).sum())
            self.sensitivity_zero += float(
                torch.square(prediction - zero_prediction).mean(dim=reduce_dims).sum()
            )
            self.sensitivity_shuffle += float(
                torch.square(prediction - shuffled_prediction).mean(dim=reduce_dims).sum()
            )
            self.controls_count += batch
        self.count += batch

    def compute(self) -> dict[str, object]:
        if self.count == 0:
            raise RuntimeError("No metrics were accumulated")
        averaged = {key: value / self.count for key, value in self.sums.items()}
        mse = averaged["mse_per_horizon"]
        persistence = averaged["persistence_mse_per_horizon"]
        motion = averaged["motion_mse_per_horizon"]
        zero_motion = averaged["zero_motion_mse_per_horizon"]
        normalized = mse / persistence.clamp_min(1e-8)
        normalized_motion = motion / zero_motion.clamp_min(1e-8)
        result = {
            "count": self.count,
            "mse_per_horizon": mse.tolist(),
            "persistence_mse_per_horizon": persistence.tolist(),
            "normalized_error_per_horizon": normalized.tolist(),
            "mean_normalized_error": float(normalized.mean()),
            "improvement_over_persistence": float(1.0 - normalized.mean()),
            "normalized_motion_per_horizon": normalized_motion.tolist(),
            "mean_normalized_motion": float(normalized_motion.mean()),
            "retrieval_accuracy_per_horizon": averaged[
                "retrieval_accuracy_per_horizon"
            ].tolist(),
            "mean_retrieval_accuracy": float(
                averaged["retrieval_accuracy_per_horizon"].mean()
            ),
        }
        if self.controls_count:
            result.update(
                {
                    "correct_better_than_zero_fraction": self.correct_better_zero
                    / self.controls_count,
                    "correct_better_than_shuffle_fraction": self.correct_better_shuffle
                    / self.controls_count,
                    "zero_action_prediction_delta_mse": self.sensitivity_zero
                    / self.controls_count,
                    "shuffled_action_prediction_delta_mse": self.sensitivity_shuffle
                    / self.controls_count,
                }
            )
        return result


@torch.inference_mode()
def evaluate_dynamics_model(
    model,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    use_amp: bool = True,
    action_controls: bool = True,
) -> dict[str, object]:
    model.eval()
    overall = MetricAccumulator()
    by_task: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_episode: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            prediction = model.rollout(history, actions)
            zero_prediction = None
            shuffled_prediction = None
            if action_controls:
                zero_actions = batch["zero_actions"].to(device, non_blocking=True)
                zero_prediction = model.rollout(history, zero_actions)
                tasks = batch["task"]
                shuffled_actions = _far_shuffle_actions_within_task(actions, tasks)
                shuffled_prediction = model.rollout(history, shuffled_actions)
        overall.update(
            prediction.float(),
            future.float(),
            history[:, -1].float(),
            zero_prediction.float() if zero_prediction is not None else None,
            shuffled_prediction.float() if shuffled_prediction is not None else None,
        )
        tasks = batch["task"]
        for task in sorted(set(tasks)):
            indices = torch.tensor(
                [index for index, value in enumerate(tasks) if value == task],
                device=device,
            )
            by_task[task].update(
                prediction[indices].float(),
                future[indices].float(),
                history[indices, -1].float(),
                zero_prediction[indices].float() if zero_prediction is not None else None,
                shuffled_prediction[indices].float()
                if shuffled_prediction is not None
                else None,
            )
        episode_values = batch["episode"].tolist()
        for task, episode in sorted(set(zip(tasks, episode_values))):
            indices = torch.tensor(
                [
                    index
                    for index, pair in enumerate(zip(tasks, episode_values))
                    if pair == (task, episode)
                ],
                device=device,
            )
            key = f"{task}/episode_{episode:06d}"
            by_episode[key].update(
                prediction[indices].float(),
                future[indices].float(),
                history[indices, -1].float(),
                zero_prediction[indices].float() if zero_prediction is not None else None,
                shuffled_prediction[indices].float()
                if shuffled_prediction is not None
                else None,
            )
    return {
        "overall": overall.compute(),
        "by_task": {task: values.compute() for task, values in sorted(by_task.items())},
        "by_episode": {
            episode: values.compute() for episode, values in sorted(by_episode.items())
        },
    }


class EndpointMetricAccumulator:
    """Aggregate direct ``t+16`` metrics without averaging batch ratios."""

    def __init__(self) -> None:
        self.count = 0
        self.squared_error = 0.0
        self.persistence_error = 0.0
        self.delta_cosine = 0.0
        self.correct_better_zero = 0.0
        self.correct_better_shuffle = 0.0
        self.sensitivity_zero = 0.0
        self.sensitivity_shuffle = 0.0
        self.controls_count = 0
        self.nearest_future_counts = torch.zeros(4, dtype=torch.long)
        self.retrieval_count = 0
        self.nearest_time_counts: torch.Tensor | None = None
        self.time_retrieval_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        current: torch.Tensor,
        zero_prediction: torch.Tensor | None = None,
        shuffled_prediction: torch.Tensor | None = None,
        future_candidates: torch.Tensor | None = None,
        time_candidates: torch.Tensor | None = None,
    ) -> None:
        reduce_dims = tuple(range(1, target.ndim))
        correct_error = torch.square(prediction - target).mean(dim=reduce_dims)
        persistence_error = torch.square(current - target).mean(dim=reduce_dims)
        predicted_delta = (prediction - current).flatten(1)
        target_delta = (target - current).flatten(1)
        cosine = torch.nn.functional.cosine_similarity(
            predicted_delta, target_delta, dim=1, eps=1e-8
        )
        self.squared_error += float(correct_error.sum())
        self.persistence_error += float(persistence_error.sum())
        self.delta_cosine += float(cosine.sum())
        batch = len(prediction)
        self.count += batch
        if zero_prediction is not None and shuffled_prediction is not None:
            zero_error = torch.square(zero_prediction - target).mean(dim=reduce_dims)
            shuffled_error = torch.square(shuffled_prediction - target).mean(
                dim=reduce_dims
            )
            self.correct_better_zero += float((correct_error < zero_error).sum())
            self.correct_better_shuffle += float(
                (correct_error < shuffled_error).sum()
            )
            self.sensitivity_zero += float(
                torch.square(prediction - zero_prediction)
                .mean(dim=reduce_dims)
                .sum()
            )
            self.sensitivity_shuffle += float(
                torch.square(prediction - shuffled_prediction)
                .mean(dim=reduce_dims)
                .sum()
            )
            self.controls_count += batch
        if future_candidates is not None:
            if future_candidates.shape[0] != batch or future_candidates.shape[1] != 4:
                raise ValueError("future_candidates must contain four times per sample")
            candidate_reduce = tuple(range(2, future_candidates.ndim))
            distances = torch.square(
                prediction[:, None] - future_candidates
            ).mean(dim=candidate_reduce)
            nearest = distances.argmin(dim=1).detach().cpu()
            self.nearest_future_counts += torch.bincount(nearest, minlength=4)
            self.retrieval_count += batch
        if time_candidates is not None:
            if time_candidates.shape[0] != batch:
                raise ValueError("time_candidates batch does not match prediction")
            candidate_reduce = tuple(range(2, time_candidates.ndim))
            distances = torch.square(
                prediction[:, None] - time_candidates
            ).mean(dim=candidate_reduce)
            nearest = distances.argmin(dim=1).detach().cpu()
            counts = torch.bincount(nearest, minlength=time_candidates.shape[1])
            self.nearest_time_counts = (
                counts
                if self.nearest_time_counts is None
                else self.nearest_time_counts + counts
            )
            self.time_retrieval_count += batch

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise RuntimeError("No endpoint metrics were accumulated")
        mse = self.squared_error / self.count
        persistence = self.persistence_error / self.count
        result: dict[str, float | int] = {
            "count": self.count,
            "mse": mse,
            "persistence_mse": persistence,
            "normalized_error": mse / max(persistence, 1e-8),
            "improvement_over_persistence": 1.0 - mse / max(persistence, 1e-8),
            "delta_cosine": self.delta_cosine / self.count,
        }
        if self.controls_count:
            result.update(
                {
                    "correct_better_than_zero_fraction": self.correct_better_zero
                    / self.controls_count,
                    "correct_better_than_shuffle_fraction": self.correct_better_shuffle
                    / self.controls_count,
                    "zero_action_prediction_delta_mse": self.sensitivity_zero
                    / self.controls_count,
                    "shuffled_action_prediction_delta_mse": self.sensitivity_shuffle
                    / self.controls_count,
                }
            )
        if self.retrieval_count:
            histogram = self.nearest_future_counts.float() / self.retrieval_count
            result.update(
                {
                    "nearest_future_histogram": histogram.tolist(),
                    "endpoint_retrieval_accuracy": float(histogram[-1]),
                }
            )
        if self.time_retrieval_count:
            assert self.nearest_time_counts is not None
            histogram = self.nearest_time_counts.float() / self.time_retrieval_count
            result.update(
                {
                    "nearest_time_histogram": histogram.tolist(),
                    "time_endpoint_retrieval_accuracy": float(histogram[-1]),
                }
            )
        return result


@torch.inference_mode()
def evaluate_endpoint_model(
    model,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    use_amp: bool = True,
    action_controls: bool = True,
) -> dict[str, object]:
    """Evaluate ``current + anchor state + action chunk -> endpoint``."""
    model.eval()
    overall = EndpointMetricAccumulator()
    by_task: dict[str, EndpointMetricAccumulator] = defaultdict(
        EndpointMetricAccumulator
    )
    by_episode: dict[str, EndpointMetricAccumulator] = defaultdict(
        EndpointMetricAccumulator
    )
    for batch in loader:
        current = batch["current"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        anchor_state = batch["anchor_state"].to(device, non_blocking=True)
        zero_prediction = None
        shuffled_prediction = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            prediction = model(current, actions, anchor_state)
            if action_controls:
                zero_actions = batch["zero_actions"].to(device, non_blocking=True)
                zero_prediction = model(current, zero_actions, anchor_state)
                tasks = batch["task"]
                shuffled_actions = _far_shuffle_actions_within_task(actions, tasks)
                shuffled_prediction = model(current, shuffled_actions, anchor_state)

        args = (
            prediction.float(),
            target.float(),
            current.float(),
            zero_prediction.float() if zero_prediction is not None else None,
            shuffled_prediction.float() if shuffled_prediction is not None else None,
        )
        overall.update(*args)
        tasks = batch["task"]
        for task in sorted(set(tasks)):
            indices = torch.tensor(
                [index for index, value in enumerate(tasks) if value == task],
                device=device,
            )
            by_task[task].update(
                *(value[indices] if value is not None else None for value in args),
            )
        episode_values = batch["episode"].tolist()
        for task, episode in sorted(set(zip(tasks, episode_values))):
            indices = torch.tensor(
                [
                    index
                    for index, pair in enumerate(zip(tasks, episode_values))
                    if pair == (task, episode)
                ],
                device=device,
            )
            key = f"{task}/episode_{episode:06d}"
            by_episode[key].update(
                *(value[indices] if value is not None else None for value in args),
            )
    return {
        "overall": overall.compute(),
        "by_task": {task: values.compute() for task, values in sorted(by_task.items())},
        "by_episode": {
            episode: values.compute()
            for episode, values in sorted(by_episode.items())
        },
    }


@torch.inference_mode()
def evaluate_multi_horizon_endpoint_model(
    model,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    use_amp: bool = True,
    action_controls: bool = True,
) -> dict[str, object]:
    """Evaluate causal t+4/t+8/t+12/t+16 direct predictions."""
    model.eval()
    overall = EndpointMetricAccumulator()
    by_horizon = {
        f"t+{offset}": EndpointMetricAccumulator() for offset in (4, 8, 12, 16)
    }
    by_task: dict[str, EndpointMetricAccumulator] = defaultdict(
        EndpointMetricAccumulator
    )
    by_episode: dict[str, EndpointMetricAccumulator] = defaultdict(
        EndpointMetricAccumulator
    )
    for batch in loader:
        current = batch["current"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        anchor_state = batch["anchor_state"].to(device, non_blocking=True)
        zero_prediction = shuffled_prediction = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            prediction_output = model(current, actions, anchor_state)
            prediction = (
                prediction_output["rollout"]
                if isinstance(prediction_output, dict)
                else prediction_output
            )
            if action_controls:
                zero_actions = batch["zero_actions"].to(device, non_blocking=True)
                zero_output = model(current, zero_actions, anchor_state)
                zero_prediction = (
                    zero_output["rollout"]
                    if isinstance(zero_output, dict)
                    else zero_output
                )
                shuffled_actions = _far_shuffle_actions_within_task(
                    actions, batch["task"]
                )
                shuffled_output = model(
                    current, shuffled_actions, anchor_state
                )
                shuffled_prediction = (
                    shuffled_output["rollout"]
                    if isinstance(shuffled_output, dict)
                    else shuffled_output
                )

        prediction = prediction.float()
        targets = targets.float()
        current = current.float()
        endpoint = prediction[:, -1]
        endpoint_target = targets[:, -1]
        zero_endpoint = (
            zero_prediction[:, -1].float() if zero_prediction is not None else None
        )
        shuffled_endpoint = (
            shuffled_prediction[:, -1].float()
            if shuffled_prediction is not None
            else None
        )
        time_candidates = torch.cat((current[:, None], targets), dim=1)
        overall.update(
            endpoint,
            endpoint_target,
            current,
            zero_endpoint,
            shuffled_endpoint,
            future_candidates=targets,
            time_candidates=time_candidates,
        )
        for index, offset in enumerate((4, 8, 12, 16)):
            by_horizon[f"t+{offset}"].update(
                prediction[:, index], targets[:, index], current
            )

        tasks = batch["task"]
        for task in sorted(set(tasks)):
            indices = torch.tensor(
                [index for index, value in enumerate(tasks) if value == task],
                device=device,
            )
            by_task[task].update(
                endpoint[indices],
                endpoint_target[indices],
                current[indices],
                zero_endpoint[indices] if zero_endpoint is not None else None,
                shuffled_endpoint[indices]
                if shuffled_endpoint is not None
                else None,
                future_candidates=targets[indices],
                time_candidates=time_candidates[indices],
            )
        episode_values = batch["episode"].tolist()
        for task, episode in sorted(set(zip(tasks, episode_values))):
            indices = torch.tensor(
                [
                    index
                    for index, pair in enumerate(zip(tasks, episode_values))
                    if pair == (task, episode)
                ],
                device=device,
            )
            key = f"{task}/episode_{episode:06d}"
            by_episode[key].update(
                endpoint[indices],
                endpoint_target[indices],
                current[indices],
                zero_endpoint[indices] if zero_endpoint is not None else None,
                shuffled_endpoint[indices]
                if shuffled_endpoint is not None
                else None,
                future_candidates=targets[indices],
                time_candidates=time_candidates[indices],
            )
    return {
        "overall": overall.compute(),
        "by_horizon": {
            key: value.compute() for key, value in by_horizon.items()
        },
        "by_task": {task: values.compute() for task, values in sorted(by_task.items())},
        "by_episode": {
            episode: values.compute()
            for episode, values in sorted(by_episode.items())
        },
    }


class RollingMetricAccumulator:
    """Aggregate per-action open-loop rollout metrics for the CheckVLA pilot."""

    def __init__(self) -> None:
        self.count = 0
        self.squared_error: torch.Tensor | None = None
        self.persistence_error: torch.Tensor | None = None
        self.delta_cosine: torch.Tensor | None = None
        self.endpoint_correct_better_zero = 0.0
        self.endpoint_correct_better_shuffle = 0.0
        self.endpoint_zero_sensitivity = 0.0
        self.endpoint_shuffle_sensitivity = 0.0
        self.controls_count = 0
        self.endpoint_nearest_counts: torch.Tensor | None = None

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor,
        zero_prediction: torch.Tensor | None = None,
        shuffled_prediction: torch.Tensor | None = None,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        reduce_dims = tuple(range(2, target.ndim))
        squared = torch.square(prediction - target).mean(dim=reduce_dims)
        persistence = torch.square(anchor[:, None] - target).mean(dim=reduce_dims)
        predicted_delta = (prediction - anchor[:, None]).flatten(2)
        target_delta = (target - anchor[:, None]).flatten(2)
        cosine = torch.nn.functional.cosine_similarity(
            predicted_delta, target_delta, dim=2, eps=1e-8
        )
        batch = len(prediction)
        for name, value in (
            ("squared_error", squared.sum(dim=0)),
            ("persistence_error", persistence.sum(dim=0)),
            ("delta_cosine", cosine.sum(dim=0)),
        ):
            current = getattr(self, name)
            setattr(self, name, value.detach().cpu() if current is None else current + value.detach().cpu())

        endpoint_distances = torch.square(
            prediction[:, -1, None] - target
        ).mean(dim=reduce_dims)
        nearest = endpoint_distances.argmin(dim=1).detach().cpu()
        histogram = torch.bincount(nearest, minlength=target.shape[1])
        self.endpoint_nearest_counts = (
            histogram
            if self.endpoint_nearest_counts is None
            else self.endpoint_nearest_counts + histogram
        )
        if zero_prediction is not None and shuffled_prediction is not None:
            endpoint_target = target[:, -1]
            endpoint_reduce = tuple(range(1, endpoint_target.ndim))
            correct_error = torch.square(prediction[:, -1] - endpoint_target).mean(
                dim=endpoint_reduce
            )
            zero_error = torch.square(zero_prediction[:, -1] - endpoint_target).mean(
                dim=endpoint_reduce
            )
            shuffled_error = torch.square(
                shuffled_prediction[:, -1] - endpoint_target
            ).mean(dim=endpoint_reduce)
            self.endpoint_correct_better_zero += float((correct_error < zero_error).sum())
            self.endpoint_correct_better_shuffle += float(
                (correct_error < shuffled_error).sum()
            )
            self.endpoint_zero_sensitivity += float(
                torch.square(prediction[:, -1] - zero_prediction[:, -1])
                .mean(dim=endpoint_reduce)
                .sum()
            )
            self.endpoint_shuffle_sensitivity += float(
                torch.square(prediction[:, -1] - shuffled_prediction[:, -1])
                .mean(dim=endpoint_reduce)
                .sum()
            )
            self.controls_count += batch
        self.count += batch

    def compute(self) -> dict[str, object]:
        if self.count == 0 or any(
            value is None
            for value in (
                self.squared_error,
                self.persistence_error,
                self.delta_cosine,
                self.endpoint_nearest_counts,
            )
        ):
            raise RuntimeError("No rolling metrics were accumulated")
        mse = self.squared_error / self.count
        persistence = self.persistence_error / self.count
        normalized = mse / persistence.clamp_min(1e-8)
        cosine = self.delta_cosine / self.count
        nearest = self.endpoint_nearest_counts.float() / self.count
        result: dict[str, object] = {
            "count": self.count,
            "mse_per_horizon": mse.tolist(),
            "persistence_mse_per_horizon": persistence.tolist(),
            "normalized_error_per_horizon": normalized.tolist(),
            "mean_normalized_error": float(normalized.mean()),
            "endpoint_normalized_error": float(normalized[-1]),
            "delta_cosine_per_horizon": cosine.tolist(),
            "endpoint_delta_cosine": float(cosine[-1]),
            "endpoint_nearest_future_histogram": nearest.tolist(),
            "endpoint_retrieval_accuracy": float(nearest[-1]),
        }
        if self.controls_count:
            result.update(
                {
                    "endpoint_correct_better_zero_fraction": self.endpoint_correct_better_zero
                    / self.controls_count,
                    "endpoint_correct_better_shuffle_fraction": self.endpoint_correct_better_shuffle
                    / self.controls_count,
                    "endpoint_zero_action_prediction_delta_mse": self.endpoint_zero_sensitivity
                    / self.controls_count,
                    "endpoint_shuffled_action_prediction_delta_mse": self.endpoint_shuffle_sensitivity
                    / self.controls_count,
                }
            )
        return result


@torch.inference_mode()
def evaluate_rolling_model(
    model,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    use_amp: bool = True,
    action_controls: bool = False,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    overall = RollingMetricAccumulator()
    by_task: dict[str, RollingMetricAccumulator] = defaultdict(RollingMetricAccumulator)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        latents = batch["latents"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        anchor_state = batch["anchor_state"].to(device, non_blocking=True)
        zero_prediction = shuffled_prediction = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            # Route evaluation through ``forward`` so nn.DataParallel can
            # scatter each batch across all configured GPUs.  Calling the
            # custom ``rollout`` method directly would silently use only the
            # primary device.
            prediction = model(
                latents,
                actions,
                anchor_state,
                mode="self_rollout",
                rollout_horizon=actions.shape[1],
                detach_context=True,
            )
            if action_controls:
                zero_actions = batch["zero_actions"].to(device, non_blocking=True)
                zero_prediction = model(
                    latents,
                    zero_actions,
                    anchor_state,
                    mode="self_rollout",
                    rollout_horizon=zero_actions.shape[1],
                    detach_context=True,
                )
                tasks = batch["task"]
                shuffled_actions = _far_shuffle_actions_within_task(actions, tasks)
                shuffled_prediction = model(
                    latents,
                    shuffled_actions,
                    anchor_state,
                    mode="self_rollout",
                    rollout_horizon=shuffled_actions.shape[1],
                    detach_context=True,
                )
        args = (
            prediction.float(),
            latents[:, 1:].float(),
            latents[:, 0].float(),
            zero_prediction.float() if zero_prediction is not None else None,
            shuffled_prediction.float() if shuffled_prediction is not None else None,
        )
        overall.update(*args)
        for task in sorted(set(batch["task"])):
            indices = torch.tensor(
                [index for index, value in enumerate(batch["task"]) if value == task],
                device=device,
            )
            by_task[task].update(
                *(value[indices] if value is not None else None for value in args)
            )
    return {
        "overall": overall.compute(),
        "by_task": {task: values.compute() for task, values in sorted(by_task.items())},
    }


class SingleStepMetricAccumulator:
    """Sufficient statistics for visual and optional proprio t-to-t+4 outputs."""

    def __init__(self, dynamic_threshold: float = 0.0) -> None:
        self.dynamic_threshold = float(dynamic_threshold)
        self.count = 0
        self.state_count = 0
        self.controls_count = 0
        self.dynamic_count = 0
        self.sums: dict[str, float] = defaultdict(float)

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        visual_target: torch.Tensor,
        visual_current: torch.Tensor,
        state_target: torch.Tensor,
        state_current: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        zero_outputs: dict[str, torch.Tensor] | None = None,
        shuffled_outputs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        visual = outputs["visual"].float()
        state = outputs.get("state")
        if state is not None:
            state = state.float()
        visual_target = visual_target.float()
        visual_current = visual_current.float()
        state_target = state_target.float()
        state_current = state_current.float()
        batch = len(visual)
        visual_reduce = tuple(range(1, visual.ndim))
        visual_error = torch.square(visual - visual_target).mean(dim=visual_reduce)
        visual_persistence = torch.square(
            visual_current - visual_target
        ).mean(dim=visual_reduce)
        visual_to_current = torch.square(visual - visual_current).mean(
            dim=visual_reduce
        )
        predicted_delta = (visual - visual_current).flatten(1)
        target_delta = (visual_target - visual_current).flatten(1)
        delta_cosine = torch.nn.functional.cosine_similarity(
            predicted_delta, target_delta, dim=1, eps=1e-8
        )
        delta_rms_ratio = predicted_delta.square().mean(dim=1).sqrt() / target_delta.square().mean(
            dim=1
        ).sqrt().clamp_min(1e-8)
        temporal_margin = (visual_to_current - visual_error) / visual_persistence.clamp_min(
            1e-8
        )

        values = {
            "visual_mse": visual_error,
            "visual_huber": torch.nn.functional.smooth_l1_loss(
                visual, visual_target, reduction="none"
            ).mean(dim=visual_reduce),
            "visual_persistence_mse": visual_persistence,
            "visual_future_closer": (visual_error < visual_to_current).float(),
            "visual_temporal_margin": temporal_margin,
            "visual_delta_cosine": delta_cosine,
            "visual_delta_rms_ratio": delta_rms_ratio,
        }
        if state is not None:
            state_reduce = tuple(range(1, state.ndim))
            state_error = torch.square(state - state_target).mean(dim=state_reduce)
            state_persistence = torch.square(state_current - state_target).mean(
                dim=state_reduce
            )
            state_to_current = torch.square(state - state_current).mean(
                dim=state_reduce
            )
            semantic = semantic_proprio_loss(
                state,
                state_target,
                state_mean.float(),
                state_std.float(),
            )["state"]
            values.update(
                {
                    "state_mse": state_error,
                    "state_persistence_mse": state_persistence,
                    "state_future_closer": (state_error < state_to_current).float(),
                }
            )
            self.sums["state_semantic_loss"] += float(semantic) * batch
            self.state_count += batch
        for key, value in values.items():
            self.sums[key] += float(value.sum())

        dynamic = visual_persistence > self.dynamic_threshold
        if dynamic.any():
            self.dynamic_count += int(dynamic.sum())
            self.sums["dynamic_visual_future_closer"] += float(
                (visual_error[dynamic] < visual_to_current[dynamic]).sum()
            )
            self.sums["dynamic_visual_normalized_numerator"] += float(
                visual_error[dynamic].sum()
            )
            self.sums["dynamic_visual_normalized_denominator"] += float(
                visual_persistence[dynamic].sum()
            )

        if zero_outputs is not None and shuffled_outputs is not None:
            zero_visual = zero_outputs["visual"].float()
            shuffled_visual = shuffled_outputs["visual"].float()
            zero_error = torch.square(zero_visual - visual_target).mean(
                dim=visual_reduce
            )
            shuffled_error = torch.square(shuffled_visual - visual_target).mean(
                dim=visual_reduce
            )
            self.sums["correct_better_zero"] += float(
                (visual_error < zero_error).sum()
            )
            self.sums["correct_better_shuffle"] += float(
                (visual_error < shuffled_error).sum()
            )
            self.sums["zero_sensitivity"] += float(
                torch.square(visual - zero_visual).mean(dim=visual_reduce).sum()
            )
            self.sums["shuffle_sensitivity"] += float(
                torch.square(visual - shuffled_visual).mean(dim=visual_reduce).sum()
            )
            self.controls_count += batch
        self.count += batch

    def compute(self) -> dict[str, float | int]:
        if self.count < 1:
            raise RuntimeError("No single-step metrics were accumulated")
        result: dict[str, float | int] = {
            "count": self.count,
            "visual_mse": self.sums["visual_mse"] / self.count,
            "visual_huber": self.sums["visual_huber"] / self.count,
            "visual_persistence_mse": self.sums["visual_persistence_mse"]
            / self.count,
            "visual_future_closer_fraction": self.sums["visual_future_closer"]
            / self.count,
            "visual_temporal_margin": self.sums["visual_temporal_margin"]
            / self.count,
            "visual_delta_cosine": self.sums["visual_delta_cosine"] / self.count,
            "visual_delta_rms_ratio": self.sums["visual_delta_rms_ratio"]
            / self.count,
            "dynamic_count": self.dynamic_count,
            "controls_count": self.controls_count,
        }
        result["visual_normalized_error"] = float(result["visual_mse"]) / max(
            float(result["visual_persistence_mse"]), 1e-8
        )
        if self.state_count:
            result.update(
                {
                    "state_mse": self.sums["state_mse"] / self.state_count,
                    "state_persistence_mse": self.sums["state_persistence_mse"]
                    / self.state_count,
                    "state_future_closer_fraction": self.sums[
                        "state_future_closer"
                    ]
                    / self.state_count,
                    "state_semantic_loss": self.sums["state_semantic_loss"]
                    / self.state_count,
                }
            )
            result["state_normalized_error"] = float(result["state_mse"]) / max(
                float(result["state_persistence_mse"]), 1e-8
            )
        if self.dynamic_count:
            result["dynamic_visual_future_closer_fraction"] = self.sums[
                "dynamic_visual_future_closer"
            ] / self.dynamic_count
            result["dynamic_visual_mse"] = self.sums[
                "dynamic_visual_normalized_numerator"
            ] / self.dynamic_count
            result["dynamic_visual_persistence_mse"] = self.sums[
                "dynamic_visual_normalized_denominator"
            ] / self.dynamic_count
            result["dynamic_visual_normalized_error"] = float(
                result["dynamic_visual_mse"]
            ) / max(float(result["dynamic_visual_persistence_mse"]), 1e-8)
        if self.controls_count:
            result.update(
                {
                    "correct_better_than_zero_fraction": self.sums[
                        "correct_better_zero"
                    ]
                    / self.controls_count,
                    "correct_better_than_shuffle_fraction": self.sums[
                        "correct_better_shuffle"
                    ]
                    / self.controls_count,
                    "zero_action_prediction_delta_mse": self.sums[
                        "zero_sensitivity"
                    ]
                    / self.controls_count,
                    "shuffled_action_prediction_delta_mse": self.sums[
                        "shuffle_sensitivity"
                    ]
                    / self.controls_count,
                }
            )
        return result


@torch.inference_mode()
def evaluate_single_step_proprio_model(
    model,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    use_amp: bool = True,
    action_controls: bool = True,
    dynamic_threshold: float = 0.0,
) -> dict[str, object]:
    model.eval()
    use_proprio_condition = bool(
        getattr(model, "use_proprio_condition", True)
    )
    overall = SingleStepMetricAccumulator(dynamic_threshold)
    by_task: dict[str, SingleStepMetricAccumulator] = defaultdict(
        lambda: SingleStepMetricAccumulator(dynamic_threshold)
    )
    state_mean = state_mean.to(device)
    state_std = state_std.to(device)
    for batch in loader:
        current = batch["current"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        anchor_state = (
            batch["anchor_state"].to(device, non_blocking=True)
            if use_proprio_condition
            else None
        )
        zero_outputs = shuffled_outputs = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            outputs = model(current, actions, anchor_state)
            if action_controls:
                zero_actions = batch["zero_actions"].to(device, non_blocking=True)
                zero_outputs = model(current, zero_actions, anchor_state)
                shuffled_actions = _far_shuffle_actions_within_task(
                    actions, batch["task"]
                )
                shuffled_outputs = model(current, shuffled_actions, anchor_state)
        # A visual-only model must not consume the recorded future state even
        # during diagnostics.  The current state is a same-device placeholder;
        # the accumulator ignores both state tensors when no state output exists.
        state_placeholder = (
            anchor_state
            if anchor_state is not None
            else torch.zeros(
                len(current), 16, dtype=torch.float32, device=device
            )
        )
        target_state = (
            batch["target_state"].to(device, non_blocking=True)
            if "state" in outputs
            else state_placeholder
        )
        args = (
            {key: value.float() for key, value in outputs.items()},
            target.float(),
            current.float(),
            target_state.float(),
            state_placeholder.float(),
            state_mean.float(),
            state_std.float(),
            (
                {key: value.float() for key, value in zero_outputs.items()}
                if zero_outputs is not None
                else None
            ),
            (
                {key: value.float() for key, value in shuffled_outputs.items()}
                if shuffled_outputs is not None
                else None
            ),
        )
        overall.update(*args)
        for task in sorted(set(batch["task"])):
            indices = torch.tensor(
                [index for index, value in enumerate(batch["task"]) if value == task],
                device=device,
            )
            selected_zero = (
                {key: value[indices] for key, value in args[7].items()}
                if args[7] is not None
                else None
            )
            selected_shuffle = (
                {key: value[indices] for key, value in args[8].items()}
                if args[8] is not None
                else None
            )
            by_task[task].update(
                {key: value[indices] for key, value in args[0].items()},
                args[1][indices],
                args[2][indices],
                args[3][indices],
                args[4][indices],
                args[5],
                args[6],
                selected_zero,
                selected_shuffle,
            )
    return {
        "overall": overall.compute(),
        "by_task": {task: values.compute() for task, values in sorted(by_task.items())},
    }


def merge_single_step_evaluations(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Merge exact DDP shards without averaging already-normalized ratios."""

    def merge_metrics(values: list[dict[str, float | int]]) -> dict[str, float | int]:
        count = sum(int(value["count"]) for value in values)
        if count < 1:
            raise RuntimeError("Cannot merge empty single-step metrics")
        dynamic_count = sum(int(value.get("dynamic_count", 0)) for value in values)
        controls_count = sum(int(value.get("controls_count", 0)) for value in values)

        def weighted(key: str, count_key: str = "count") -> float:
            denominator = sum(int(value.get(count_key, 0)) for value in values)
            return sum(
                float(value[key])
                * int(value.get(count_key, 0))
                for value in values
                if key in value
            ) / max(denominator, 1)

        count_keys = {
            "count",
            "dynamic_count",
            "controls_count",
            "visual_normalized_error",
            "state_normalized_error",
        }
        result: dict[str, float | int] = {
            "count": count,
            "dynamic_count": dynamic_count,
            "controls_count": controls_count,
        }
        for key in values[0]:
            if key in count_keys or key.startswith("dynamic_") or key.startswith(
                "correct_better"
            ) or key.endswith("prediction_delta_mse"):
                continue
            result[key] = weighted(key)
        result["visual_normalized_error"] = float(result["visual_mse"]) / max(
            float(result["visual_persistence_mse"]), 1e-8
        )
        if "state_mse" in result:
            result["state_normalized_error"] = float(result["state_mse"]) / max(
                float(result["state_persistence_mse"]), 1e-8
            )
        if dynamic_count:
            for key in (
                "dynamic_visual_future_closer_fraction",
                "dynamic_visual_mse",
                "dynamic_visual_persistence_mse",
            ):
                result[key] = weighted(key, "dynamic_count")
            result["dynamic_visual_normalized_error"] = float(
                result["dynamic_visual_mse"]
            ) / max(float(result["dynamic_visual_persistence_mse"]), 1e-8)
        if controls_count:
            for key in (
                "correct_better_than_zero_fraction",
                "correct_better_than_shuffle_fraction",
                "zero_action_prediction_delta_mse",
                "shuffled_action_prediction_delta_mse",
            ):
                result[key] = weighted(key, "controls_count")
        return result

    overall = merge_metrics([row["overall"] for row in rows])
    tasks = sorted(
        {task for row in rows for task in dict(row.get("by_task", {}))}
    )
    by_task = {
        task: merge_metrics(
            [row["by_task"][task] for row in rows if task in row.get("by_task", {})]
        )
        for task in tasks
    }
    return {"overall": overall, "by_task": by_task}
