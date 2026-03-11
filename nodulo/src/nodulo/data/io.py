from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from nodulo.utils import load_json, save_json


@dataclass(slots=True)
class SampleRecord:
    file_name: str
    image_path: str
    label: int
    label_name: str
    source: str
    width: int
    height: int
    lidc_id: str | None
    points: list[dict[str, float]]
    nodule_count: int
    nodule_count_known: bool

    def has_points(self) -> bool:
        return len(self.points) > 0

    def structured_group(self) -> str:
        if self.label == 0:
            return "neg"
        if self.nodule_count_known and 1 <= self.nodule_count <= 4:
            return f"pos_{self.nodule_count}"
        return "pos_unknown"


LABEL_MAP = {"No Finding": 0, "Nodule": 1}


def resolve_image_path(root_dir: Path, nih_dir: str, lidc_dir: str, file_name: str) -> Path:
    lidc_path = root_dir / lidc_dir / file_name
    if lidc_path.exists():
        return lidc_path
    nih_path = root_dir / nih_dir / file_name
    if nih_path.exists():
        return nih_path
    raise FileNotFoundError(f"Could not resolve image path for {file_name}")


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image -= image.min()
    max_value = image.max()
    if max_value > 0:
        image /= max_value
    return image


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(image_uint8)
    return enhanced.astype(np.float32) / 255.0


def build_metadata(config: dict[str, Any]) -> list[SampleRecord]:
    data_cfg = config["data"]
    root_dir = Path(data_cfg["root_dir"]).resolve()
    cache_path = root_dir / "metadata_cache.json"
    if data_cfg.get("cache_metadata", True):
        cached = load_json(cache_path)
        if cached:
            if all("nodule_count" in item and "nodule_count_known" in item for item in cached):
                return [SampleRecord(**item) for item in cached]

    cls_df = pd.read_csv(root_dir / data_cfg["classification_csv"])
    loc_df = pd.read_csv(root_dir / data_cfg["localization_csv"])
    point_map: dict[str, list[dict[str, float]]] = {}
    for row in loc_df.fillna(0.0).to_dict(orient="records"):
        point_map.setdefault(row["file_name"], []).append(
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "confidence": float(row.get("confidence", 1.0) or 1.0),
            }
        )

    records: list[SampleRecord] = []
    for row in cls_df.fillna("").to_dict(orient="records"):
        label_name = row["label"]
        image_path = resolve_image_path(
            root_dir,
            data_cfg["nih_dir"],
            data_cfg["lidc_dir"],
            row["file_name"],
        )
        width, height = read_image_size(image_path)
        lidc_id = row.get("LIDC_ID") or None
        points = point_map.get(row["file_name"], [])
        if LABEL_MAP[label_name] == 0:
            nodule_count = 0
            nodule_count_known = True
        elif points:
            nodule_count = min(len(points), 4)
            nodule_count_known = True
        else:
            nodule_count = 0
            nodule_count_known = False
        records.append(
            SampleRecord(
                file_name=row["file_name"],
                image_path=str(image_path),
                label=LABEL_MAP[label_name],
                label_name=label_name,
                source="lidc" if "lidc_png" in str(image_path) else "nih",
                width=width,
                height=height,
                lidc_id=lidc_id,
                points=points,
                nodule_count=nodule_count,
                nodule_count_known=nodule_count_known,
            )
        )

    if data_cfg.get("cache_metadata", True):
        save_json([asdict(record) for record in records], cache_path)
    return records


def create_stratified_folds(records: list[SampleRecord], num_folds: int, seed: int) -> dict[int, dict[str, list[int]]]:
    structured_labels = [record.structured_group() for record in records]
    label_counts = pd.Series(structured_labels).value_counts().to_dict()
    if all(count >= num_folds for count in label_counts.values()):
        labels = structured_labels
    else:
        labels = [record.label for record in records]
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds: dict[int, dict[str, list[int]]] = {}
    for fold_idx, (train_ids, val_ids) in enumerate(splitter.split(np.zeros(len(records)), labels)):
        folds[fold_idx] = {"train": train_ids.tolist(), "val": val_ids.tolist()}
    return folds


def sample_records(records: list[SampleRecord], max_samples: int, seed: int) -> list[SampleRecord]:
    if max_samples <= 0 or max_samples >= len(records):
        return records

    rng = random.Random(seed)
    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]
    annotated_positives = [record for record in positives if record.nodule_count_known and record.nodule_count > 0]
    unannotated_positives = [record for record in positives if not (record.nodule_count_known and record.nodule_count > 0)]
    rng.shuffle(annotated_positives)
    rng.shuffle(unannotated_positives)
    rng.shuffle(negatives)

    positive_ratio = len(positives) / max(len(records), 1)
    target_positives = max(len(annotated_positives), max(1, round(max_samples * positive_ratio)))
    target_positives = max(target_positives, min(len(positives), len(annotated_positives) * 2))
    num_positives = min(len(positives), target_positives)
    num_negatives = min(len(negatives), max_samples - num_positives)

    if num_positives + num_negatives < max_samples:
        remaining = max_samples - (num_positives + num_negatives)
        extra_negatives = negatives[num_negatives : num_negatives + remaining]
        num_negatives += len(extra_negatives)
    remaining_positive_budget = max(0, num_positives - len(annotated_positives))
    sampled_positives = annotated_positives + unannotated_positives[:remaining_positive_budget]
    sampled = sampled_positives + negatives[:num_negatives]
    rng.shuffle(sampled)
    return sampled


def sample_with_all_positives(records: list[SampleRecord], max_negative_samples: int, seed: int) -> list[SampleRecord]:
    if max_negative_samples <= 0:
        return records
    rng = random.Random(seed)
    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]
    rng.shuffle(negatives)
    sampled = positives + negatives[: min(len(negatives), max_negative_samples)]
    rng.shuffle(sampled)
    return sampled


def point_to_normalized_box(x: float, y: float, width: int, height: int, box_size_px: float) -> list[float]:
    half_w = box_size_px / 2.0
    half_h = box_size_px / 2.0
    x0 = max(0.0, x - half_w)
    y0 = max(0.0, y - half_h)
    x1 = min(float(width), x + half_w)
    y1 = min(float(height), y + half_h)
    cx = ((x0 + x1) / 2.0) / float(width)
    cy = ((y0 + y1) / 2.0) / float(height)
    w = max(1.0, x1 - x0) / float(width)
    h = max(1.0, y1 - y0) / float(height)
    return [cx, cy, w, h]


def scale_normalized_point(x_norm: float, y_norm: float, width: int, height: int) -> tuple[float, float]:
    return x_norm * width, y_norm * height
