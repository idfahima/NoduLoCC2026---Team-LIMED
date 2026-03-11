from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    from rad_dino import RadDino
except ImportError:
    RadDino = None


def _load_rad_dino_backbone() -> nn.Module:
    repo_id = "microsoft/rad-dino"
    if AutoModel is not None:
        try:
            return AutoModel.from_pretrained(repo_id, local_files_only=True).eval()
        except Exception:
            return AutoModel.from_pretrained(repo_id).eval()
    if RadDino is None:
        raise ImportError("rad-dino is required for medical_rad_dino")
    return RadDino().model


@dataclass(slots=True)
class BackboneFeatures:
    pooled: torch.Tensor
    spatial: torch.Tensor


class BackboneWrapper(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone_type = "timm"
        self.required_input_size: int | None = None
        self.register_buffer("input_mean", torch.zeros(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.ones(1, 3, 1, 1), persistent=False)

        if backbone_name == "medical_rad_dino":
            self.backbone = _load_rad_dino_backbone()
            self.backbone_type = "rad_dino"
            self.feature_dim = int(self.backbone.config.hidden_size)
            self.required_input_size = 518
            self.input_mean = torch.tensor([0.5307, 0.5307, 0.5307], dtype=torch.float32).view(1, 3, 1, 1)
            self.input_std = torch.tensor([0.2583, 0.2583, 0.2583], dtype=torch.float32).view(1, 3, 1, 1)
        elif backbone_name == "medical_biomedclip":
            if open_clip is None:
                raise ImportError("open-clip-torch is required for medical_biomedclip")
            self.backbone, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.backbone_type = "biomedclip"
            self.feature_dim = 512
            self.required_input_size = 224
            self.input_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
            self.input_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            create_kwargs = {
                "pretrained": pretrained,
                "num_classes": 0,
                "in_chans": 3,
            }
            if not any(token in backbone_name for token in ["dinov2", "dinov3", ".mae"]):
                create_kwargs["global_pool"] = "avg"
            self.backbone = timm.create_model(backbone_name, **create_kwargs)
            self.feature_dim = getattr(self.backbone, "num_features")
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=True)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.input_mean.to(device=x.device, dtype=x.dtype)
        std = self.input_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if self.required_input_size is None:
            return x
        if x.shape[-1] == self.required_input_size and x.shape[-2] == self.required_input_size:
            return x
        return F.interpolate(x, size=(self.required_input_size, self.required_input_size), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).pooled

    def _sequence_to_spatial(self, sequence: torch.Tensor) -> torch.Tensor:
        token_count = sequence.shape[1]
        grid_size = int(round(math.sqrt(token_count)))
        if grid_size * grid_size != token_count:
            if token_count > 1:
                sequence = sequence[:, 1:, :]
                token_count = sequence.shape[1]
                grid_size = int(round(math.sqrt(token_count)))
            if grid_size * grid_size != token_count:
                raise ValueError(f"Cannot reshape token sequence of length {token_count} into a square grid")
        return sequence.transpose(1, 2).reshape(sequence.shape[0], sequence.shape[2], grid_size, grid_size)

    def forward_features(self, x: torch.Tensor) -> BackboneFeatures:
        if self.backbone_type == "rad_dino":
            x = self._resize_if_needed(x)
            outputs = self.backbone(pixel_values=self._normalize(x))
            sequence = outputs.last_hidden_state
            return BackboneFeatures(pooled=sequence[:, 0], spatial=self._sequence_to_spatial(sequence[:, 1:]))
        if self.backbone_type == "biomedclip":
            x = self._resize_if_needed(x)
            pooled = self.backbone.encode_image(self._normalize(x))
            spatial = pooled[:, :, None, None]
            return BackboneFeatures(pooled=pooled, spatial=spatial)

        features = self.backbone.forward_features(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 4:
            pooled = features.mean(dim=(2, 3))
            if hasattr(self.backbone, "forward_head"):
                pooled = self.backbone.forward_head(features, pre_logits=True)
            return BackboneFeatures(pooled=pooled, spatial=features)
        if features.ndim == 3:
            pooled = features.mean(dim=1)
            if hasattr(self.backbone, "forward_head"):
                pooled = self.backbone.forward_head(features, pre_logits=True)
            spatial = self._sequence_to_spatial(features)
            return BackboneFeatures(pooled=pooled, spatial=spatial)
        return BackboneFeatures(pooled=features, spatial=features[:, :, None, None])

    def target_layer(self) -> nn.Module:
        if self.backbone_type != "timm":
            raise AttributeError(f"CAM target layer is not implemented for {self.backbone_name}")
        if hasattr(self.backbone, "stages"):
            return self.backbone.stages[-1].blocks[-1]
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        raise AttributeError("Could not infer target CAM layer for this backbone")

