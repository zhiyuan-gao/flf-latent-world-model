"""Action-conditioned latent dynamics predictors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block_causal_mask(num_groups: int, tokens_per_group: int, device=None) -> torch.Tensor:
    """Return a Transformer boolean mask (True means attention is blocked)."""
    groups = torch.arange(num_groups, device=device).repeat_interleave(tokens_per_group)
    return groups.view(1, -1) > groups.view(-1, 1)


class ActionBlockEncoder(nn.Module):
    """Encode all four ordered low-level actions without temporal pooling."""

    def __init__(self, action_dim: int, model_dim: int, depth: int = 2, heads: int = 8):
        super().__init__()
        self.input = nn.Linear(action_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, 1, 4, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 4 or actions.shape[-2] != 4:
            raise ValueError("actions must have shape [B, groups, 4, action_dim]")
        batch, groups, steps, _ = actions.shape
        values = self.input(actions) + self.position
        values = self.temporal(values.reshape(batch * groups, steps, -1))
        return values.reshape(batch, groups, steps, -1)


class FourHorizonACPredictor(nn.Module):
    """Predict four fixed-time future latent grids from history and 16 actions.

    Teacher forcing follows the V-JEPA 2-AC alignment: the representation at
    each time group, together with the action block starting at that time,
    predicts the next representation.  Rollout predictions are generated
    autoregressively and never receive the ground-truth future.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 12,
        grid_size: int = 8,
        model_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        action_depth: int = 2,
        max_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.grid_size = int(grid_size)
        self.model_dim = int(model_dim)
        self.max_groups = int(max_groups)
        self.visual_tokens = self.grid_size**2
        self.action_tokens = 4
        self.tokens_per_group = self.action_tokens + self.visual_tokens

        self.visual_input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
        )
        self.action_encoder = ActionBlockEncoder(
            action_dim=action_dim,
            model_dim=model_dim,
            depth=action_depth,
            heads=heads,
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, 1, self.visual_tokens, model_dim)
        )
        self.temporal_position = nn.Parameter(torch.zeros(1, max_groups, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
        )
        self.output = nn.Linear(model_dim, feature_dim)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        # Parameterize dynamics as a change from the current latent.  A tiny
        # head makes the initial model approximately the persistence baseline
        # instead of forcing it to relearn visual identity from scratch.
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def config_dict(self) -> dict[str, int | float | bool]:
        first_layer = self.predictor.layers[0]
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "grid_size": self.grid_size,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.layers),
            "heads": first_layer.self_attn.num_heads,
            "action_depth": len(self.action_encoder.temporal.layers),
            "max_groups": self.max_groups,
            "dropout": float(first_layer.dropout.p),
        }

    def _predict_sequence(self, visuals: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if visuals.ndim != 5:
            raise ValueError("visuals must have [B, groups, grid_h, grid_w, feature_dim]")
        batch, groups, grid_h, grid_w, feature_dim = visuals.shape
        if grid_h != self.grid_size or grid_w != self.grid_size:
            raise ValueError(f"Expected a {self.grid_size}x{self.grid_size} feature grid")
        if feature_dim != self.feature_dim or actions.shape[:2] != (batch, groups):
            raise ValueError("visual/action dimensions do not match model configuration")
        if groups > self.max_groups:
            raise ValueError("sequence is longer than max_groups")

        visual = visuals.reshape(batch, groups, self.visual_tokens, feature_dim)
        visual = self.visual_input(visual) + self.spatial_position
        action = self.action_encoder(actions)
        tokens = torch.cat((action, visual), dim=2)
        tokens = tokens + self.temporal_position[:, :groups]
        tokens = tokens.flatten(1, 2)
        mask = block_causal_mask(groups, self.tokens_per_group, tokens.device)
        predicted = self.predictor(tokens, mask=mask)
        predicted = predicted.reshape(batch, groups, self.tokens_per_group, self.model_dim)
        delta = self.output(predicted[:, :, self.action_tokens :])
        delta = delta.reshape(
            batch, groups, self.grid_size, self.grid_size, self.feature_dim
        )
        return visuals + delta

    def teacher_forced(
        self, history: torch.Tensor, future: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if history.shape[1] != 4 or future.shape[1] != 4 or actions.shape[1:3] != (4, 4):
            raise ValueError("Expected four history frames, four futures, and 4x4 actions")
        visual_inputs = torch.cat((history, future[:, :-1]), dim=1)
        empty_actions = torch.zeros(
            actions.shape[0],
            3,
            4,
            actions.shape[-1],
            dtype=actions.dtype,
            device=actions.device,
        )
        action_inputs = torch.cat((empty_actions, actions), dim=1)
        return self._predict_sequence(visual_inputs, action_inputs)[:, 3:]

    def rollout(
        self, history: torch.Tensor, actions: torch.Tensor, steps: int = 4
    ) -> torch.Tensor:
        if history.shape[1] != 4 or actions.shape[1:3] != (4, 4):
            raise ValueError("Expected four history frames and 4x4 actions")
        if not 1 <= steps <= 4:
            raise ValueError("rollout steps must be between one and four")
        generated: list[torch.Tensor] = []
        empty_actions = torch.zeros(
            actions.shape[0],
            3,
            4,
            actions.shape[-1],
            dtype=actions.dtype,
            device=actions.device,
        )
        for horizon in range(steps):
            visual_inputs = torch.cat((history, *generated), dim=1) if generated else history
            action_inputs = torch.cat((empty_actions, actions[:, : horizon + 1]), dim=1)
            next_frame = self._predict_sequence(visual_inputs, action_inputs)[:, -1:]
            generated.append(next_frame)
        return torch.cat(generated, dim=1)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        actions: torch.Tensor,
        rollout_steps: int = 2,
    ) -> dict[str, torch.Tensor]:
        return {
            "teacher_forced": self.teacher_forced(history, future, actions),
            "rollout": self.rollout(history, actions, steps=rollout_steps),
        }


class DirectEndpointACPredictor(nn.Module):
    """Predict one visual endpoint from an anchor state and an action chunk.

    The conditioning contract deliberately mirrors the simple continuous
    tokenization used by V-JEPA 2-AC: every standardized action vector and the
    single standardized anchor proprioceptive vector are mapped by independent
    affine projections.  There is no input-vector LayerNorm and no future
    proprioceptive rollout.  A joint Transformer uses the current visual patch
    tokens, one state token, and all ordered action tokens to predict the
    visual representation at the end of the chunk in one non-autoregressive
    pass.
    """

    def __init__(
        self,
        feature_dim: int = 1408,
        action_dim: int = 12,
        state_dim: int = 16,
        grid_size: int = 16,
        model_dim: int = 960,
        depth: int = 7,
        heads: int = 12,
        horizon: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.grid_size = int(grid_size)
        self.model_dim = int(model_dim)
        self.visual_tokens = self.grid_size**2
        self.action_steps = int(horizon)
        if self.action_steps < 1:
            raise ValueError("horizon must be positive")

        self.visual_input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
        )
        # Actions and state are standardized channel by channel in the dataset.
        # Do not apply another LayerNorm across a complete control vector: its
        # absolute magnitude is part of the residual-policy signal.
        self.action_input = nn.Linear(action_dim, model_dim)
        self.state_input = nn.Linear(state_dim, model_dim)
        self.action_position = nn.Parameter(
            torch.zeros(1, self.action_steps, model_dim)
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.visual_tokens, model_dim)
        )
        self.action_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
        )
        self.output = nn.Linear(model_dim, feature_dim)
        nn.init.trunc_normal_(self.action_position, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.action_type, std=0.02)
        nn.init.trunc_normal_(self.state_type, std=0.02)
        nn.init.trunc_normal_(self.visual_type, std=0.02)
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def config_dict(self) -> dict[str, int | float]:
        first_layer = self.predictor.layers[0]
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "grid_size": self.grid_size,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.layers),
            "heads": first_layer.self_attn.num_heads,
            "horizon": self.action_steps,
            "dropout": float(first_layer.dropout.p),
        }

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
    ) -> torch.Tensor:
        if current.ndim != 4:
            raise ValueError("current must have [B, grid_h, grid_w, feature_dim]")
        if actions.ndim != 3 or actions.shape[1] != self.action_steps:
            raise ValueError(
                f"actions must have [B, {self.action_steps}, action_dim]"
            )
        if anchor_state.ndim != 2:
            raise ValueError("anchor_state must have [B, state_dim]")
        batch, grid_h, grid_w, feature_dim = current.shape
        if (grid_h, grid_w, feature_dim) != (
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        ):
            raise ValueError("current dimensions do not match model configuration")
        if actions.shape[0] != batch or actions.shape[-1] != self.action_dim:
            raise ValueError("action dimensions do not match model configuration")
        if anchor_state.shape != (batch, self.state_dim):
            raise ValueError("anchor state dimensions do not match model configuration")

        visual = current.reshape(batch, self.visual_tokens, self.feature_dim)
        visual = self.visual_input(visual)
        visual = visual + self.spatial_position + self.visual_type
        state = self.state_input(anchor_state).unsqueeze(1) + self.state_type
        action = self.action_input(actions) + self.action_position + self.action_type
        # The endpoint legitimately conditions on the complete action chunk, so
        # this pass is bidirectional rather than block-causal.
        tokens = self.predictor(torch.cat((state, action, visual), dim=1))
        visual_output = tokens[:, 1 + self.action_steps :]
        delta = self.output(visual_output).reshape(
            batch, self.grid_size, self.grid_size, self.feature_dim
        )
        return current + delta


class SingleStepProprioACPredictor(nn.Module):
    """Predict visual state, and optionally proprioception, after four actions.

    The conditioning path contains only the current visual latent, one current
    standardized state token, and four independently embedded standardized
    action tokens.  Neither action nor state receives a second vector-wise
    LayerNorm.  Outputs are direct next-state predictions; the visual head is
    not initialized as a persistence residual.  ``predict_state=False`` keeps
    current proprioception as an input condition while removing the future
    proprioception head entirely for a visual-target-only ablation.
    ``use_proprio_condition=False`` additionally removes the current-state
    token; it is valid only together with ``predict_state=False``.
    """

    def __init__(
        self,
        feature_dim: int = 1408,
        action_dim: int = 12,
        state_dim: int = 16,
        grid_size: int = 16,
        model_dim: int = 960,
        depth: int = 7,
        heads: int = 12,
        action_steps: int = 4,
        dropout: float = 0.0,
        predict_state: bool = True,
        use_proprio_condition: bool = True,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if action_steps < 1:
            raise ValueError("action_steps must be positive")
        if predict_state and not use_proprio_condition:
            raise ValueError(
                "future proprio prediction requires current proprio conditioning"
            )
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.grid_size = int(grid_size)
        self.model_dim = int(model_dim)
        self.action_steps = int(action_steps)
        self.predict_state = bool(predict_state)
        self.use_proprio_condition = bool(use_proprio_condition)
        self.visual_tokens = self.grid_size**2

        self.visual_input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
        )
        self.action_input = nn.Linear(action_dim, model_dim)
        self.state_input = nn.Linear(state_dim, model_dim)
        self.action_position = nn.Parameter(
            torch.zeros(1, self.action_steps, model_dim)
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.visual_tokens, model_dim)
        )
        self.action_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.visual_output = nn.Linear(model_dim, feature_dim)
        self.state_output = nn.Linear(model_dim, state_dim)
        for parameter in (
            self.action_position,
            self.spatial_position,
            self.action_type,
            self.state_type,
            self.visual_type,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.trunc_normal_(self.visual_output.weight, std=0.02)
        nn.init.zeros_(self.visual_output.bias)
        nn.init.trunc_normal_(self.state_output.weight, std=0.02)
        nn.init.zeros_(self.state_output.bias)
        # Construct and initialize the optional head before dropping it.  This
        # preserves the RNG stream, so all shared parameters have identical
        # seed initialization in the joint-target and visual-only ablations.
        if not self.predict_state:
            self.state_output = None
        if not self.use_proprio_condition:
            self.state_input = None
            self.state_type = None

    def config_dict(self) -> dict[str, int | float | bool]:
        first_layer = self.predictor.layers[0]
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "grid_size": self.grid_size,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.layers),
            "heads": first_layer.self_attn.num_heads,
            "action_steps": self.action_steps,
            "dropout": float(first_layer.dropout.p),
            "direct_visual_output": True,
            "predict_state": self.predict_state,
            "use_proprio_condition": self.use_proprio_condition,
        }

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if current.ndim != 4:
            raise ValueError("current must have [B, grid_h, grid_w, feature_dim]")
        batch, grid_h, grid_w, feature_dim = current.shape
        if (grid_h, grid_w, feature_dim) != (
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        ):
            raise ValueError("current dimensions do not match model configuration")
        if actions.shape != (batch, self.action_steps, self.action_dim):
            raise ValueError("actions do not match model configuration")
        if self.use_proprio_condition:
            if anchor_state is None or anchor_state.shape != (batch, self.state_dim):
                raise ValueError("anchor state does not match model configuration")
        elif anchor_state is not None:
            raise ValueError("anchor state must be None when proprio conditioning is disabled")

        visual = current.reshape(batch, self.visual_tokens, self.feature_dim)
        visual = self.visual_input(visual)
        visual = visual + self.spatial_position + self.visual_type
        action = self.action_input(actions) + self.action_position + self.action_type
        input_tokens = [action, visual]
        visual_start = self.action_steps
        if self.state_input is not None and self.state_type is not None:
            assert anchor_state is not None
            state = self.state_input(anchor_state).unsqueeze(1) + self.state_type
            input_tokens.insert(0, state)
            visual_start += 1
        predicted = self.predictor(torch.cat(input_tokens, dim=1))
        predicted_visual = self.visual_output(
            predicted[:, visual_start:]
        ).reshape(batch, self.grid_size, self.grid_size, self.feature_dim)
        outputs = {"visual": predicted_visual}
        if self.state_output is not None:
            outputs["state"] = self.state_output(predicted[:, 0])
        return outputs


class CausalMultiHorizonACPredictor(nn.Module):
    """Predict four endpoints with fixed action slots and causal prefix masks.

    Each source window is expanded to four examples.  Every expanded example
    contains 16 action slots, but action slots beyond its requested horizon are
    zeroed and excluded as attention keys at every layer.  A learned horizon
    token distinguishes t+4/t+8/t+12/t+16 while all predictor weights are
    shared.
    """

    def __init__(
        self,
        feature_dim: int = 1408,
        action_dim: int = 12,
        state_dim: int = 16,
        grid_size: int = 16,
        model_dim: int = 960,
        depth: int = 7,
        heads: int = 12,
        horizon: int = 16,
        horizons: tuple[int, ...] = (4, 8, 12, 16),
        dropout: float = 0.0,
        normalize_representations: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if not horizons or tuple(sorted(horizons)) != tuple(horizons):
            raise ValueError("horizons must be non-empty and increasing")
        if horizons[-1] != horizon:
            raise ValueError("the final supervised horizon must equal horizon")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.grid_size = int(grid_size)
        self.model_dim = int(model_dim)
        self.visual_tokens = self.grid_size**2
        self.action_steps = int(horizon)
        self.horizons = tuple(int(value) for value in horizons)
        self.normalize_representations = bool(normalize_representations)

        self.visual_input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
        )
        self.action_input = nn.Linear(action_dim, model_dim)
        self.state_input = nn.Linear(state_dim, model_dim)
        self.action_position = nn.Parameter(
            torch.zeros(1, self.action_steps, model_dim)
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.visual_tokens, model_dim)
        )
        self.horizon_token = nn.Parameter(
            torch.zeros(1, len(self.horizons), model_dim)
        )
        self.action_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(model_dim, feature_dim)
        for parameter in (
            self.action_position,
            self.spatial_position,
            self.horizon_token,
            self.action_type,
            self.state_type,
            self.visual_type,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def config_dict(self) -> dict[str, object]:
        first_layer = self.predictor.layers[0]
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "grid_size": self.grid_size,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.layers),
            "heads": first_layer.self_attn.num_heads,
            "horizon": self.action_steps,
            "horizons": self.horizons,
            "dropout": float(first_layer.dropout.p),
            "normalize_representations": self.normalize_representations,
        }

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
    ) -> torch.Tensor:
        if current.ndim != 4:
            raise ValueError("current must have [B, grid_h, grid_w, feature_dim]")
        if actions.shape[1:] != (self.action_steps, self.action_dim):
            raise ValueError(
                f"actions must have [B, {self.action_steps}, {self.action_dim}]"
            )
        batch, grid_h, grid_w, feature_dim = current.shape
        if (grid_h, grid_w, feature_dim) != (
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        ):
            raise ValueError("current dimensions do not match model configuration")
        if actions.shape[0] != batch:
            raise ValueError("action batch does not match current batch")
        if anchor_state.shape != (batch, self.state_dim):
            raise ValueError("anchor state dimensions do not match model configuration")

        horizon_count = len(self.horizons)
        expanded_current = current[:, None].expand(
            batch, horizon_count, grid_h, grid_w, feature_dim
        ).reshape(batch * horizon_count, grid_h, grid_w, feature_dim)
        expanded_actions = actions[:, None].expand(
            batch, horizon_count, self.action_steps, self.action_dim
        ).reshape(batch * horizon_count, self.action_steps, self.action_dim)
        expanded_state = anchor_state[:, None].expand(
            batch, horizon_count, self.state_dim
        ).reshape(batch * horizon_count, self.state_dim)
        horizon_indices = torch.arange(horizon_count, device=current.device).repeat(batch)
        horizon_steps = torch.tensor(self.horizons, device=current.device)
        action_indices = torch.arange(self.action_steps, device=current.device)
        valid_actions = action_indices[None] < horizon_steps[horizon_indices, None]
        expanded_actions = expanded_actions * valid_actions.unsqueeze(-1)

        visual = expanded_current.reshape(
            batch * horizon_count, self.visual_tokens, self.feature_dim
        )
        visual = self.visual_input(visual)
        visual = visual + self.spatial_position + self.visual_type
        state = self.state_input(expanded_state).unsqueeze(1) + self.state_type
        action = self.action_input(expanded_actions)
        action = action + self.action_position + self.action_type
        horizon = self.horizon_token[:, horizon_indices].transpose(0, 1)

        padding_mask = torch.zeros(
            batch * horizon_count,
            2 + self.action_steps + self.visual_tokens,
            dtype=torch.bool,
            device=current.device,
        )
        padding_mask[:, 2 : 2 + self.action_steps] = ~valid_actions
        tokens = self.predictor(
            torch.cat((horizon, state, action, visual), dim=1),
            src_key_padding_mask=padding_mask,
        )
        visual_output = tokens[:, 2 + self.action_steps :]
        delta = self.output(visual_output).reshape(
            batch, horizon_count, grid_h, grid_w, feature_dim
        )
        prediction = current[:, None] + delta
        if self.normalize_representations:
            # Match V-JEPA2-AC: compare affine-free, per-token normalized
            # predictor outputs with equivalently normalized encoder targets.
            prediction = F.layer_norm(prediction, (self.feature_dim,))
        return prediction


class LearnedNominalProprioRollout(nn.Module):
    """Roll standardized robot state forward using only candidate actions."""

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 12,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_input = nn.Linear(state_dim, hidden_dim)
        self.action_input = nn.Linear(action_dim, hidden_dim)
        self.transition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.trunc_normal_(self.transition[-1].weight, std=1e-3)
        nn.init.zeros_(self.transition[-1].bias)

    def forward(
        self,
        anchor_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_state.ndim != 2 or actions.ndim != 3:
            raise ValueError("state/actions must be [B, S] and [B, T, A]")
        batch, steps, action_dim = actions.shape
        if anchor_state.shape != (batch, self.state_dim):
            raise ValueError("anchor state dimensions do not match configuration")
        if action_dim != self.action_dim:
            raise ValueError("action dimensions do not match configuration")
        states = [anchor_state]
        state = anchor_state
        for step in range(steps):
            encoded = torch.cat(
                (self.state_input(state), self.action_input(actions[:, step])),
                dim=-1,
            )
            state = state + self.transition(encoded)
            states.append(state)
        return torch.stack(states, dim=1)


class BlockRollingACPredictor(nn.Module):
    """Shared four-action visual transition with learned nominal proprio rollout.

    Recorded future proprioception is never an input.  A small recurrent model
    first rolls the current standardized state through all candidate actions.
    Four source-state/action tokens condition each shared visual transition
    from t to t+4, t+4 to t+8, t+8 to t+12, and t+12 to t+16.
    """

    def __init__(
        self,
        feature_dim: int = 1408,
        action_dim: int = 12,
        state_dim: int = 16,
        grid_size: int = 16,
        model_dim: int = 960,
        depth: int = 7,
        heads: int = 12,
        horizon: int = 16,
        block_size: int = 4,
        proprio_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if horizon % block_size:
            raise ValueError("horizon must be divisible by block_size")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.grid_size = int(grid_size)
        self.model_dim = int(model_dim)
        self.horizon = int(horizon)
        self.block_size = int(block_size)
        self.num_blocks = self.horizon // self.block_size
        self.proprio_hidden_dim = int(proprio_hidden_dim)
        self.visual_tokens = self.grid_size**2

        self.proprio_dynamics = LearnedNominalProprioRollout(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=proprio_hidden_dim,
        )
        self.visual_input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
        )
        self.action_input = nn.Linear(action_dim, model_dim)
        self.state_input = nn.Linear(state_dim, model_dim)
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.visual_tokens, model_dim)
        )
        self.within_block_position = nn.Parameter(
            torch.zeros(1, self.block_size, model_dim)
        )
        self.block_position = nn.Parameter(
            torch.zeros(1, self.num_blocks, model_dim)
        )
        self.visual_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(model_dim, feature_dim)
        for parameter in (
            self.spatial_position,
            self.within_block_position,
            self.block_position,
            self.visual_type,
            self.action_type,
            self.state_type,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def config_dict(self) -> dict[str, object]:
        first_layer = self.predictor.layers[0]
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "grid_size": self.grid_size,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.layers),
            "heads": first_layer.self_attn.num_heads,
            "horizon": self.horizon,
            "block_size": self.block_size,
            "proprio_hidden_dim": self.proprio_hidden_dim,
            "dropout": float(first_layer.dropout.p),
        }

    def _condition_blocks(
        self,
        actions: torch.Tensor,
        nominal_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        action_blocks = actions.reshape(
            batch, self.num_blocks, self.block_size, self.action_dim
        )
        source_states = nominal_states[:, :-1].reshape(
            batch, self.num_blocks, self.block_size, self.state_dim
        )
        return action_blocks, source_states.detach()

    def _transition(
        self,
        visuals: torch.Tensor,
        action_blocks: torch.Tensor,
        state_blocks: torch.Tensor,
        block_indices: torch.Tensor,
    ) -> torch.Tensor:
        if visuals.ndim != 5:
            raise ValueError("visuals must be [B, N, Y, X, C]")
        batch, blocks, grid_h, grid_w, feature_dim = visuals.shape
        if (grid_h, grid_w, feature_dim) != (
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        ):
            raise ValueError("visual dimensions do not match configuration")
        expected_actions = (
            batch,
            blocks,
            self.block_size,
            self.action_dim,
        )
        expected_states = (
            batch,
            blocks,
            self.block_size,
            self.state_dim,
        )
        if action_blocks.shape != expected_actions or state_blocks.shape != expected_states:
            raise ValueError("block condition dimensions do not match configuration")
        if block_indices.shape != (blocks,):
            raise ValueError("one block index is required per visual transition")

        visual = visuals.reshape(
            batch * blocks, self.visual_tokens, self.feature_dim
        )
        visual = self.visual_input(visual)
        visual = visual + self.spatial_position + self.visual_type
        actions = self.action_input(
            action_blocks.reshape(batch * blocks, self.block_size, self.action_dim)
        )
        actions = actions + self.within_block_position + self.action_type
        states = self.state_input(
            state_blocks.reshape(batch * blocks, self.block_size, self.state_dim)
        )
        states = states + self.within_block_position + self.state_type
        block = self.block_position[:, block_indices].expand(batch, -1, -1)
        block = block.reshape(batch * blocks, 1, self.model_dim)
        tokens = self.predictor(torch.cat((block, states, actions, visual), dim=1))
        visual_output = tokens[:, 1 + 2 * self.block_size :]
        delta = self.output(visual_output).reshape(
            batch, blocks, grid_h, grid_w, feature_dim
        )
        return F.layer_norm(visuals + delta, (self.feature_dim,))

    def rollout(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
        nominal_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("actions do not match configured horizon")
        if nominal_states is None:
            nominal_states = self.proprio_dynamics(anchor_state, actions)
        action_blocks, state_blocks = self._condition_blocks(actions, nominal_states)
        visual = current
        outputs: list[torch.Tensor] = []
        for block_index in range(self.num_blocks):
            prediction = self._transition(
                visual[:, None],
                action_blocks[:, block_index : block_index + 1],
                state_blocks[:, block_index : block_index + 1],
                torch.tensor([block_index], device=current.device),
            )
            visual = prediction[:, 0]
            outputs.append(visual)
        return torch.stack(outputs, dim=1), nominal_states

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
        teacher_context: torch.Tensor | None = None,
        mode: str = "rollout",
    ) -> dict[str, torch.Tensor]:
        nominal_states = self.proprio_dynamics(anchor_state, actions)
        rollout, _ = self.rollout(
            current,
            actions,
            anchor_state,
            nominal_states=nominal_states,
        )
        result = {"rollout": rollout, "nominal_states": nominal_states}
        if mode == "rollout":
            return result
        if mode != "joint":
            raise ValueError(f"Unsupported block rolling mode: {mode}")
        if teacher_context is None or teacher_context.shape != (
            current.shape[0],
            self.num_blocks - 1,
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        ):
            raise ValueError("joint training requires three teacher visual contexts")
        visual_sources = torch.cat((current[:, None], teacher_context), dim=1)
        action_blocks, state_blocks = self._condition_blocks(actions, nominal_states)
        result["teacher_forced"] = self._transition(
            visual_sources,
            action_blocks,
            state_blocks,
            torch.arange(self.num_blocks, device=current.device),
        )
        return result


class CheckVLARollingPredictor(nn.Module):
    """Per-action block-causal latent dynamics in the style of CheckVLA.

    Each temporal block contains the current visual tokens, the action that
    follows them, and the anchor proprioceptive state.  Block ``i`` predicts
    the visual tokens at ``i + 1``.  Teacher forcing processes all blocks in a
    single causal pass; self-rollout replaces later visual inputs with the
    model's own predictions.

    CheckVLA conditions on a nominal proprioceptive rollout.  RoboCasa's mixed
    base/EEF controller does not expose a trustworthy analytic rollout, so the
    first offline implementation uses only the normalized anchor state.  This
    avoids leaking actual future proprioception into candidate-action scoring.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 12,
        state_dim: int = 16,
        grid_size: int = 4,
        model_dim: int = 960,
        depth: int = 7,
        heads: int = 12,
        max_horizon: int = 16,
        dropout: float = 0.0,
        normalize_condition_inputs: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.grid_size = int(grid_size)
        self.model_dim = int(model_dim)
        self.max_horizon = int(max_horizon)
        self.normalize_condition_inputs = bool(normalize_condition_inputs)
        self.visual_tokens = self.grid_size**2
        self.tokens_per_group = self.visual_tokens + 2

        self.visual_input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
        )
        # Actions and states are already standardized channel by channel using
        # training-set statistics.  A second LayerNorm across the complete
        # vector would suppress candidate-specific magnitude information.
        # Keep the old path available only for loading legacy checkpoints.
        if self.normalize_condition_inputs:
            self.action_input = nn.Sequential(
                nn.LayerNorm(action_dim),
                nn.Linear(action_dim, model_dim),
            )
            self.state_input = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, model_dim),
            )
        else:
            self.action_input = nn.Linear(action_dim, model_dim)
            self.state_input = nn.Linear(state_dim, model_dim)
        self.spatial_position = nn.Parameter(
            torch.zeros(1, 1, self.visual_tokens, model_dim)
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.max_horizon, 1, model_dim)
        )
        self.visual_type = nn.Parameter(torch.zeros(1, 1, 1, model_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, 1, model_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
        )
        self.output = nn.Linear(model_dim, feature_dim)
        for value in (
            self.spatial_position,
            self.temporal_position,
            self.visual_type,
            self.action_type,
            self.state_type,
        ):
            nn.init.trunc_normal_(value, std=0.02)
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def config_dict(self) -> dict[str, int | float]:
        first_layer = self.predictor.layers[0]
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "grid_size": self.grid_size,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.layers),
            "heads": first_layer.self_attn.num_heads,
            "max_horizon": self.max_horizon,
            "dropout": float(first_layer.dropout.p),
            "normalize_condition_inputs": self.normalize_condition_inputs,
        }

    def _predict_sequence(
        self,
        visuals: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
    ) -> torch.Tensor:
        if visuals.ndim != 5:
            raise ValueError("visuals must be [B, steps, grid_h, grid_w, feature_dim]")
        if actions.ndim != 3 or anchor_state.ndim != 2:
            raise ValueError("actions/state must be [B, steps, A] and [B, S]")
        batch, steps, grid_h, grid_w, feature_dim = visuals.shape
        if not 1 <= steps <= self.max_horizon:
            raise ValueError("sequence length is outside configured horizon")
        if (grid_h, grid_w, feature_dim) != (
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        ):
            raise ValueError("visual dimensions do not match model configuration")
        if actions.shape != (batch, steps, self.action_dim):
            raise ValueError("action dimensions do not match model configuration")
        if anchor_state.shape != (batch, self.state_dim):
            raise ValueError("state dimensions do not match model configuration")

        visual = visuals.reshape(batch, steps, self.visual_tokens, feature_dim)
        visual = self.visual_input(visual)
        visual = visual + self.spatial_position + self.visual_type
        action = self.action_input(actions)[:, :, None] + self.action_type
        state = self.state_input(anchor_state)[:, None, None].expand(-1, steps, -1, -1)
        state = state + self.state_type
        # Keep the action and state next to the visual block they condition.
        tokens = torch.cat((action, state, visual), dim=2)
        tokens = tokens + self.temporal_position[:, :steps]
        tokens = tokens.flatten(1, 2)
        mask = block_causal_mask(steps, self.tokens_per_group, tokens.device)
        predicted = self.predictor(tokens, mask=mask)
        predicted = predicted.reshape(batch, steps, self.tokens_per_group, self.model_dim)
        delta = self.output(predicted[:, :, 2:]).reshape(
            batch,
            steps,
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        )
        return visuals + delta

    def teacher_forced(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
    ) -> torch.Tensor:
        if latents.shape[1] != actions.shape[1] + 1:
            raise ValueError("latents must contain the anchor plus one target per action")
        return self._predict_sequence(latents[:, :-1], actions, anchor_state)

    def rollout(
        self,
        anchor_latent: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
        steps: int | None = None,
        detach_context: bool = True,
    ) -> torch.Tensor:
        if anchor_latent.ndim != 4:
            raise ValueError("anchor_latent must be [B, grid_h, grid_w, feature_dim]")
        rollout_steps = actions.shape[1] if steps is None else int(steps)
        if not 1 <= rollout_steps <= actions.shape[1] or rollout_steps > self.max_horizon:
            raise ValueError("invalid rollout length")
        context = [anchor_latent[:, None]]
        outputs: list[torch.Tensor] = []
        for index in range(rollout_steps):
            visual_inputs = torch.cat(context, dim=1)
            next_latent = self._predict_sequence(
                visual_inputs,
                actions[:, : index + 1],
                anchor_state,
            )[:, -1:]
            outputs.append(next_latent)
            context.append(next_latent.detach() if detach_context else next_latent)
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        anchor_state: torch.Tensor,
        mode: str = "teacher_forcing",
        rollout_horizon: int | None = None,
        detach_context: bool = True,
    ) -> torch.Tensor:
        """Dispatch through ``nn.Module.__call__`` for data-parallel training."""
        if mode == "teacher_forcing":
            return self.teacher_forced(latents, actions, anchor_state)
        if mode == "self_rollout":
            steps = actions.shape[1] if rollout_horizon is None else rollout_horizon
            return self.rollout(
                latents[:, 0],
                actions,
                anchor_state,
                steps=steps,
                detach_context=detach_context,
            )
        raise ValueError(f"Unsupported predictor mode: {mode}")
