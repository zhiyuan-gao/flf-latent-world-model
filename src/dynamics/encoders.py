"""Frozen visual encoder adapters and a shared PCA/whitening projection."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FrozenFrameEncoder(ABC):
    """Return one spatial token grid per independently encoded RGB frame."""

    @abstractmethod
    def encode(self, rgb_frames: np.ndarray) -> torch.Tensor:
        """Encode uint8 ``[B,H,W,3]`` frames as float ``[B,N,D]`` tokens."""


def spatially_pool_tokens(tokens: torch.Tensor, output_grid: int = 8) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, N, D]")
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise ValueError(f"Token count {tokens.shape[1]} is not a square grid")
    values = tokens.reshape(tokens.shape[0], side, side, tokens.shape[-1])
    values = values.permute(0, 3, 1, 2).float()
    values = F.adaptive_avg_pool2d(values, (output_grid, output_grid))
    return values.permute(0, 2, 3, 1).contiguous()


class PCAWhiteningProjector:
    def __init__(self, mean: torch.Tensor, components: torch.Tensor, scale: torch.Tensor):
        if mean.ndim != 1 or components.ndim != 2 or scale.ndim != 1:
            raise ValueError("Invalid PCA projector shapes")
        if components.shape != (mean.numel(), scale.numel()):
            raise ValueError("PCA dimensions do not agree")
        self.mean = mean.float().cpu()
        self.components = components.float().cpu()
        self.scale = scale.float().clamp_min(1e-6).cpu()

    @property
    def input_dim(self) -> int:
        return self.mean.numel()

    @property
    def output_dim(self) -> int:
        return self.scale.numel()

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        device = values.device
        mean = self.mean.to(device)
        components = self.components.to(device)
        scale = self.scale.to(device)
        return ((values.float() - mean) @ components) / scale

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean": self.mean,
                "components": self.components,
                "scale": self.scale,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PCAWhiteningProjector":
        values = torch.load(path, map_location="cpu", weights_only=True)
        return cls(values["mean"], values["components"], values["scale"])

    @classmethod
    def fit(
        cls,
        samples: torch.Tensor,
        output_dim: int = 256,
        niter: int = 4,
        device: str = "cuda",
    ) -> "PCAWhiteningProjector":
        if samples.ndim != 2:
            raise ValueError("PCA samples must have shape [samples, channels]")
        if min(samples.shape) <= output_dim:
            raise ValueError(
                f"Need more than {output_dim} samples and channels, got {tuple(samples.shape)}"
            )
        values = samples.float().to(device)
        mean = values.mean(dim=0)
        centered = values - mean
        _, singular, components = torch.pca_lowrank(
            centered,
            q=output_dim,
            center=False,
            niter=niter,
        )
        scale = singular / max((len(values) - 1) ** 0.5, 1.0)
        return cls(mean.cpu(), components.cpu(), scale.cpu())


class GR00TVisualEncoder(FrozenFrameEncoder):
    """Visual-only Eagle2 tower from the post-trained GR00T checkpoint."""

    def __init__(self, checkpoint: Path, device: str = "cuda:0", dtype=torch.bfloat16):
        project = Path(__file__).resolve().parents[2]
        gr00t_root = project / "third_party/Isaac-GR00T"
        sys.path.insert(0, str(gr00t_root))
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
        from gr00t.model.policy import Gr00tPolicy
        from gr00t.model.transforms import build_eagle_processor

        config = DATA_CONFIG_MAP["panda_omron"]
        self.device = torch.device(device)
        self.dtype = dtype
        self.policy = Gr00tPolicy(
            model_path=str(checkpoint),
            modality_config=config.modality_config(),
            modality_transform=config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=4,
            device=device,
        )
        self.policy.model.eval()
        self.policy.model.requires_grad_(False)
        self.backbone = self.policy.model.backbone
        self.processor = build_eagle_processor(DEFAULT_EAGLE_PATH)

    @torch.inference_mode()
    def encode(self, rgb_frames: np.ndarray) -> torch.Tensor:
        images = [Image.fromarray(frame) for frame in rgb_frames]
        processed = self.processor.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda",
        ):
            tokens = self.backbone.eagle_model.extract_feature(pixel_values)
            tokens = self.backbone.eagle_linear(tokens)
        return tokens.float()


class VJEPA2VisualEncoder(FrozenFrameEncoder):
    """Image-as-two-frame adapter matching the V-JEPA 2-AC DROID recipe."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        crop_size: int = 256,
    ):
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("A Transformers build with V-JEPA 2 support is required") from error
        self.device = torch.device(device)
        self.dtype = dtype
        try:
            self.model = AutoModel.from_pretrained(
                checkpoint,
                local_files_only=True,
                dtype=dtype,
            ).to(self.device)
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                "This Transformers version cannot load V-JEPA 2; use .venv-vjepa2"
            ) from error
        self.model.eval()
        self.model.requires_grad_(False)
        self.crop_size = int(crop_size)
        if self.crop_size < 16 or self.crop_size % 16:
            raise ValueError("V-JEPA crop_size must be a positive multiple of 16")
        checkpoint_crop = int(getattr(self.model.config, "crop_size", self.crop_size))
        checkpoint_image = int(getattr(self.model.config, "image_size", checkpoint_crop))
        if checkpoint_crop != self.crop_size or checkpoint_image != self.crop_size:
            raise ValueError(
                "V-JEPA checkpoint/input resolution mismatch: "
                f"checkpoint crop_size={checkpoint_crop}, image_size={checkpoint_image}, "
                f"requested crop_size={self.crop_size}. Use the native checkpoint for "
                "the requested resolution."
            )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _preprocess(self, rgb_frames: np.ndarray) -> torch.Tensor:
        values = torch.from_numpy(rgb_frames).to(self.device).permute(0, 3, 1, 2).float()
        values = values / 255.0
        height, width = values.shape[-2:]
        # Match the official processor: resize the short edge to 292, then
        # center-crop to 256 (and preserve that ratio for explicit variants).
        resize_short_side = round(self.crop_size * 292 / 256)
        scale = resize_short_side / min(height, width)
        resized = (int(round(height * scale)), int(round(width * scale)))
        values = F.interpolate(values, resized, mode="bilinear", align_corners=False, antialias=True)
        top = max((values.shape[-2] - self.crop_size) // 2, 0)
        left = max((values.shape[-1] - self.crop_size) // 2, 0)
        values = values[
            :, :, top : top + self.crop_size, left : left + self.crop_size
        ]
        values = (values - self.mean) / self.std
        # V-JEPA uses tubelets of two; repeating each image creates one latent time step.
        return values[:, None].repeat(1, 2, 1, 1, 1).to(self.dtype)

    @torch.inference_mode()
    def encode(self, rgb_frames: np.ndarray) -> torch.Tensor:
        pixel_values = self._preprocess(rgb_frames)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda",
        ):
            tokens = self.model.get_vision_features(pixel_values)
        if hasattr(tokens, "last_hidden_state"):
            tokens = tokens.last_hidden_state
        return tokens.float()


def build_encoder(
    name: str,
    checkpoint: Path,
    device: str = "cuda:0",
    vjepa_crop_size: int = 256,
) -> FrozenFrameEncoder:
    if name == "gr00t":
        return GR00TVisualEncoder(checkpoint, device=device)
    if name == "vjepa2":
        return VJEPA2VisualEncoder(
            checkpoint,
            device=device,
            crop_size=vjepa_crop_size,
        )
    raise ValueError(f"Unknown encoder: {name}")
