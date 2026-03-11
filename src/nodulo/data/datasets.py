from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from nodulo.data.io import SampleRecord, apply_clahe, load_grayscale_image


MAX_NODULES = 4


def to_three_channel_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
    return tensor.repeat(3, 1, 1)


def draw_gaussian(heatmap: np.ndarray, x: float, y: float, sigma: float) -> None:
    if sigma <= 0:
        return
    radius = max(1, int(round(3.0 * sigma)))
    height, width = heatmap.shape
    x0 = max(0, int(np.floor(x)) - radius)
    x1 = min(width, int(np.floor(x)) + radius + 1)
    y0 = max(0, int(np.floor(y)) - radius)
    y1 = min(height, int(np.floor(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    peak = float(gaussian.max())
    if peak > 0:
        gaussian = gaussian / peak
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)


def preprocess_grayscale_image(image_path: str | Path, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    image = load_grayscale_image(image_path)
    return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


class NoduleTransformFactory:
    @staticmethod
    def build(config: dict[str, Any], train: bool) -> A.Compose:
        aug_cfg = config["augmentation"]
        image_size = int(config["data"]["image_size"])
        transforms: list[Any] = [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
        ]
        if train:
            transforms.extend(
                [
                    A.HorizontalFlip(p=float(aug_cfg.get("horizontal_flip_prob", 0.5))),
                    A.Rotate(limit=int(aug_cfg.get("rotation_limit", 10)), border_mode=cv2.BORDER_CONSTANT, p=0.8),
                    A.RandomBrightnessContrast(
                        brightness_limit=float(aug_cfg.get("brightness_limit", 0.12)),
                        contrast_limit=float(aug_cfg.get("contrast_limit", 0.12)),
                        p=0.5,
                    ),
                    A.GaussianBlur(blur_limit=(3, 5), p=float(aug_cfg.get("blur_prob", 0.05))),
                ]
            )
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )


class MultitaskNoduleDataset(Dataset):
    def __init__(self, records: list[SampleRecord], config: dict[str, Any], train: bool) -> None:
        self.records = records
        self.config = config
        self.train = train
        self.transform = NoduleTransformFactory.build(config, train=train)
        self.image_size = int(config["data"]["image_size"])
        self.sigma = float(config["model"].get("heatmap_sigma", 6.0))
        self.clip_limit = float(config["augmentation"].get("clahe_clip_limit", 2.0))
        self.tile_grid_size = int(config["augmentation"].get("clahe_tile_grid_size", 8))

    def __len__(self) -> int:
        return len(self.records)

    def _build_heatmap(self, keypoints: list[tuple[float, float]]) -> torch.Tensor:
        heatmap = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for x, y in keypoints[:MAX_NODULES]:
            draw_gaussian(heatmap, x=x, y=y, sigma=self.sigma)
        return torch.from_numpy(heatmap).unsqueeze(0)

    @staticmethod
    def _build_point_tensors(keypoints: list[tuple[float, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        points = torch.zeros((MAX_NODULES, 2), dtype=torch.float32)
        point_mask = torch.zeros(MAX_NODULES, dtype=torch.float32)
        for idx, (x, y) in enumerate(keypoints[:MAX_NODULES]):
            points[idx, 0] = float(x)
            points[idx, 1] = float(y)
            point_mask[idx] = 1.0
        return points, point_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = preprocess_grayscale_image(record.image_path, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        raw_keypoints = [(float(point["x"]), float(point["y"])) for point in record.points[:MAX_NODULES]]
        transformed = self.transform(image=image, keypoints=raw_keypoints)
        transformed_image = transformed["image"]
        transformed_keypoints = [(float(x), float(y)) for x, y in transformed["keypoints"]]
        points, point_mask = self._build_point_tensors(transformed_keypoints)
        annotated_positive = float(record.label == 1 and record.nodule_count_known and record.nodule_count > 0)
        unannotated_positive = float(record.label == 1 and not (record.nodule_count_known and record.nodule_count > 0))
        negative = float(record.label == 0)
        return {
            "image": to_three_channel_tensor(transformed_image),
            "presence_label": torch.tensor(float(record.label), dtype=torch.float32),
            "heatmap_target": self._build_heatmap(transformed_keypoints if annotated_positive > 0 else []),
            "negative_mask": torch.tensor(negative, dtype=torch.float32),
            "annotated_mask": torch.tensor(annotated_positive, dtype=torch.float32),
            "unannotated_mask": torch.tensor(unannotated_positive, dtype=torch.float32),
            "points": points,
            "point_mask": point_mask,
            "file_name": record.file_name,
            "original_size": torch.tensor([record.height, record.width], dtype=torch.long),
            "nodule_count_known": torch.tensor(1.0 if record.nodule_count_known else 0.0, dtype=torch.float32),
            "nodule_count": torch.tensor(record.nodule_count, dtype=torch.long),
        }
