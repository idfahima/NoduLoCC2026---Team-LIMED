from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nodulo.models.backbones import BackboneWrapper


@dataclass(slots=True)
class ClassificationOutputs:
    logits: torch.Tensor


@dataclass(slots=True)
class LocalizationOutputs:
    presence_logits: torch.Tensor
    heatmap_logits: torch.Tensor
    cam_heatmap: torch.Tensor


class PatchHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            # Layer 1: 37x37 -> 74x74
            nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            # Layer 2: 74x74 -> 148x148
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            # Layer 3: 148x148 -> 296x296
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.GELU(),
        )
        self.final_conv = nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.final_conv.bias, -4.6)

    def forward(self, spatial_features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.decoder(spatial_features)
        heatmap_logits = self.final_conv(x)
        if heatmap_logits.shape[-2:] != output_size:
            heatmap_logits = F.interpolate(heatmap_logits, size=output_size, mode="bilinear", align_corners=False)
        return heatmap_logits


class RadDinoClassifier(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = BackboneWrapper(backbone_name=backbone_name, pretrained=pretrained)
        self.classifier_head = nn.Linear(self.encoder.feature_dim, 1)

    def forward(self, image: torch.Tensor) -> ClassificationOutputs:
        features = self.encoder.forward_features(image)
        logits = self.classifier_head(features.pooled).squeeze(-1)
        return ClassificationOutputs(logits=logits)


class HeatmapLocalizer(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, heatmap_hidden_channels: int = 256) -> None:
        super().__init__()
        self.encoder = BackboneWrapper(backbone_name=backbone_name, pretrained=pretrained)
        self.classifier_head = nn.Linear(self.encoder.feature_dim, 1)
        self.heatmap_head = PatchHeatmapHead(self.encoder.feature_dim, heatmap_hidden_channels)

    def compute_cam(self, spatial_features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        weight = self.classifier_head.weight.view(1, -1, 1, 1)
        bias = self.classifier_head.bias.view(1, 1, 1, 1)
        cam = torch.sum(spatial_features * weight, dim=1, keepdim=True) + bias
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        flat = cam.flatten(start_dim=1)
        min_vals = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_vals = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return (cam - min_vals) / (max_vals - min_vals + 1e-6)

    def load_from_classifier(self, state_dict: dict[str, torch.Tensor]) -> None:
        encoder_state = {key: value for key, value in state_dict.items() if key.startswith("encoder.")}
        classifier_state = {key.replace("classifier_head.", "classifier_head."): value for key, value in state_dict.items() if key.startswith("classifier_head.")}
        self.load_state_dict({**encoder_state, **classifier_state}, strict=False)

    def forward(
        self,
        image: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> LocalizationOutputs:
        features = self.encoder.forward_features(image)
        if output_size is None:
            output_size = (image.shape[-2], image.shape[-1])
        presence_logits = self.classifier_head(features.pooled).squeeze(-1)
        return LocalizationOutputs(
            presence_logits=presence_logits,
            heatmap_logits=self.heatmap_head(features.spatial, output_size=output_size),
            cam_heatmap=self.compute_cam(features.spatial, output_size=output_size),
        )
