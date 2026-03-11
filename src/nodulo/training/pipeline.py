from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from nodulo.data.datasets import MAX_NODULES, MultitaskNoduleDataset, to_three_channel_tensor
from nodulo.data.io import SampleRecord, apply_clahe, build_metadata, create_stratified_folds, sample_records, sample_with_all_positives
from nodulo.models.heads import HeatmapLocalizer, RadDinoClassifier
from nodulo.training.losses import AsymmetricBinaryLoss, FocalHeatmapLoss
from nodulo.utils import AverageMeter, ensure_dir, save_json, set_seed


def build_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(config: dict[str, Any], device: torch.device) -> bool:
    return device.type == "cuda" and bool(config["training"].get("mixed_precision", True))


def build_grad_scaler(config: dict[str, Any], device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=amp_enabled(config, device))


def autocast_context(config: dict[str, Any], device: torch.device) -> torch.amp.autocast_mode.autocast:
    return torch.amp.autocast(device_type=device.type, enabled=amp_enabled(config, device))


def confidence_to_float(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return float(array)
    if array.size == 0:
        return 0.0
    return float(array.max())


def smooth_binary_targets(targets: torch.Tensor, positive_target: float, negative_target: float) -> torch.Tensor:
    targets = targets.float()
    return torch.where(targets > 0.5, torch.full_like(targets, positive_target), torch.full_like(targets, negative_target))


def log_metrics(stage: str, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
    formatted = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    tqdm.write(f"[{stage}] epoch {epoch}/{total_epochs} | {formatted}")


class ClassificationStructuredSampler(Sampler[list[int]]):
    def __init__(self, records: list[SampleRecord], batch_size: int, batches_per_epoch: int | None, seed: int, neg_subsample_fraction: float = 1.0) -> None:
        if batch_size < 2:
            raise ValueError("Phase 1 structured batching requires batch_size >= 2")
        self.seed = seed
        self.neg_subsample_fraction = max(0.0, min(1.0, neg_subsample_fraction))
        self._epoch = 0
        self.groups: dict[str, list[int]] = {"neg": [], "pos": []}
        for idx, record in enumerate(records):
            if record.label == 0:
                self.groups["neg"].append(idx)
            else:
                self.groups["pos"].append(idx)
        if not self.groups["neg"] or not self.groups["pos"]:
            raise ValueError("Phase 1 sampler requires both negatives and positives")
        self.negative_quota = batch_size // 2
        self.positive_quota = batch_size - self.negative_quota
        if batches_per_epoch:
            self.batches_per_epoch = batches_per_epoch
        else:
            effective_neg = max(self.negative_quota, round(len(self.groups["neg"]) * self.neg_subsample_fraction))
            effective_total = effective_neg + len(self.groups["pos"])
            self.batches_per_epoch = max(1, math.ceil(effective_total / batch_size))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    @staticmethod
    def _draw(rng: random.Random, pool: list[int], count: int, fallback: list[int] | None = None) -> list[int]:
        source = pool if pool else (fallback or [])
        if not source:
            raise ValueError("Unable to draw a required subgroup for the structured sampler")
        return [rng.choice(source) for _ in range(count)]

    def __iter__(self):
        epoch_rng = random.Random(self.seed + self._epoch)
        all_neg = self.groups["neg"]
        if self.neg_subsample_fraction < 1.0:
            n_keep = max(self.negative_quota, round(len(all_neg) * self.neg_subsample_fraction))
            neg_pool = epoch_rng.sample(all_neg, min(n_keep, len(all_neg)))
        else:
            neg_pool = all_neg
        rng = random.Random(self.seed + self._epoch + 1)
        for _ in range(self.batches_per_epoch):
            batch = self._draw(rng, neg_pool, self.negative_quota)
            batch.extend(self._draw(rng, self.groups["pos"], self.positive_quota))
            rng.shuffle(batch)
            yield batch


class LocalizationStructuredSampler(Sampler[list[int]]):
    def __init__(self, records: list[SampleRecord], batch_size: int, batches_per_epoch: int | None, seed: int, neg_subsample_fraction: float = 1.0) -> None:
        if batch_size < 6:
            raise ValueError("Phase 2 structured batching requires batch_size >= 6")
        self.seed = seed
        self.neg_subsample_fraction = max(0.0, min(1.0, neg_subsample_fraction))
        self._epoch = 0
        self.groups: dict[str, list[int]] = {"neg": [], "annotated": [], "unannotated": []}
        for idx, record in enumerate(records):
            if record.label == 0:
                self.groups["neg"].append(idx)
            elif record.nodule_count_known and record.nodule_count > 0:
                self.groups["annotated"].append(idx)
            else:
                self.groups["unannotated"].append(idx)
        if not self.groups["neg"] or not self.groups["annotated"] or not self.groups["unannotated"]:
            raise ValueError("Phase 2 sampler requires negatives, annotated positives, and unannotated positives")
        self.negative_quota = batch_size // 3
        self.annotated_quota = batch_size // 3
        self.unannotated_quota = batch_size - self.negative_quota - self.annotated_quota
        if batches_per_epoch:
            self.batches_per_epoch = batches_per_epoch
        else:
            effective_neg = max(self.negative_quota, round(len(self.groups["neg"]) * self.neg_subsample_fraction))
            effective_total = effective_neg + len(self.groups["annotated"]) + len(self.groups["unannotated"])
            self.batches_per_epoch = max(1, math.ceil(effective_total / batch_size))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    @staticmethod
    def _draw(rng: random.Random, pool: list[int], count: int) -> list[int]:
        return [rng.choice(pool) for _ in range(count)]

    def __iter__(self):
        epoch_rng = random.Random(self.seed + self._epoch)
        all_neg = self.groups["neg"]
        if self.neg_subsample_fraction < 1.0:
            n_keep = max(self.negative_quota, round(len(all_neg) * self.neg_subsample_fraction))
            neg_pool = epoch_rng.sample(all_neg, min(n_keep, len(all_neg)))
        else:
            neg_pool = all_neg
        rng = random.Random(self.seed + self._epoch + 1)
        for _ in range(self.batches_per_epoch):
            batch = self._draw(rng, neg_pool, self.negative_quota)
            batch.extend(self._draw(rng, self.groups["annotated"], self.annotated_quota))
            batch.extend(self._draw(rng, self.groups["unannotated"], self.unannotated_quota))
            rng.shuffle(batch)
            yield batch


def filter_structured_records(records: list[SampleRecord]) -> list[SampleRecord]:
    return [record for record in records if record.label == 0 or record.label == 1]


def sample_eval_records(records: list[SampleRecord], max_samples: int, seed: int) -> list[SampleRecord]:
    """Sample records for evaluation, always keeping all positives (label==1) first."""
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    all_positives = [record for record in records if record.label == 1]
    if len(all_positives) >= max_samples:
        rng = random.Random(seed)
        sampled = all_positives.copy()
        rng.shuffle(sampled)
        return sampled[:max_samples]
    remaining = [record for record in records if record.label == 0]
    sampled_remaining = sample_records(remaining, max_samples=max_samples - len(all_positives), seed=seed)
    return all_positives + sampled_remaining


def split_held_out(metadata: list[SampleRecord], held_out_fraction: float, seed: int) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Stratified split: held_out ~= held_out_fraction of each group, never seen during any training fold."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(metadata):
        groups.setdefault(record.structured_group(), []).append(idx)
    held_out_indices: list[int] = []
    training_indices: list[int] = []
    for group_name, idxs in groups.items():
        # pos_unknown are useless in held-out (filtered out by sample_phase2_eval_records) — keep all in training
        if group_name == "pos_unknown":
            training_indices.extend(idxs)
            continue
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n_held = max(1, round(len(shuffled) * held_out_fraction))
        held_out_indices.extend(shuffled[:n_held])
        training_indices.extend(shuffled[n_held:])
    held_out = [metadata[i] for i in sorted(held_out_indices)]
    training_pool = [metadata[i] for i in sorted(training_indices)]
    return held_out, training_pool


def sample_phase2_eval_records(records: list[SampleRecord], max_negatives: int, seed: int) -> list[SampleRecord]:
    """Keep ALL annotated positives (coords known) + at most max_negatives negatives. Discard pos_unknown."""
    annotated = [r for r in records if r.nodule_count_known and r.nodule_count > 0]
    negatives = [r for r in records if r.label == 0]
    if max_negatives > 0 and len(negatives) > max_negatives:
        rng = random.Random(seed)
        rng.shuffle(negatives)
        negatives = negatives[:max_negatives]
    return annotated + negatives


def create_eval_loader(records: list[SampleRecord], config: dict[str, Any]) -> DataLoader:
    dataset = MultitaskNoduleDataset(records, config, train=False)
    eval_batch_size = int(
        config.get("evaluation", {}).get(
            "batch_size",
            max(int(config["phase1"].get("batch_size", 8)), int(config["phase2"].get("batch_size", 8))),
        )
    )
    return DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
    )


def create_phase1_train_loader(records: list[SampleRecord], config: dict[str, Any]) -> DataLoader:
    dataset = MultitaskNoduleDataset(records, config, train=True)
    sampler = ClassificationStructuredSampler(
        records=records,
        batch_size=int(config["phase1"]["batch_size"]),
        batches_per_epoch=config["phase1"].get("batches_per_epoch"),
        seed=int(config.get("seed", 42)),
        neg_subsample_fraction=float(config["phase1"].get("neg_subsample_fraction", 1.0)),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
    )


def create_phase2_train_loader(records: list[SampleRecord], config: dict[str, Any]) -> DataLoader:
    dataset = MultitaskNoduleDataset(records, config, train=True)
    sampler = LocalizationStructuredSampler(
        records=records,
        batch_size=int(config["phase2"]["batch_size"]),
        batches_per_epoch=config["phase2"].get("batches_per_epoch"),
        seed=int(config.get("seed", 42)),
        neg_subsample_fraction=float(config["phase2"].get("neg_subsample_fraction", 1.0)),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().view(-1)
    values = values.view(-1)
    if torch.count_nonzero(mask) == 0:
        return values.new_tensor(0.0)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def build_asl(config_section: dict[str, Any]) -> AsymmetricBinaryLoss:
    return AsymmetricBinaryLoss(
        gamma_neg=float(config_section.get("asl_gamma_neg", 4.0)),
        gamma_pos=float(config_section.get("asl_gamma_pos", 1.0)),
        clip=float(config_section.get("asl_clip", 0.05)),
        eps=float(config_section.get("asl_eps", 1e-8)),
        reduction="none",
    )


def build_focal_heatmap_loss(config_section: dict[str, Any]) -> FocalHeatmapLoss:
    return FocalHeatmapLoss(
        alpha=float(config_section.get("focal_alpha", 2.0)),
        beta=float(config_section.get("focal_beta", 4.0)),
        eps=float(config_section.get("focal_eps", 1e-6)),
        positive_threshold=float(config_section.get("focal_positive_threshold", 0.95)),
        reduction="none",
    )


def pixelwise_asl_loss(criterion: AsymmetricBinaryLoss, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return criterion(logits, targets).flatten(start_dim=1).mean(dim=1)


def pixelwise_focal_heatmap_loss(criterion: FocalHeatmapLoss, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return criterion(logits, targets)


def compute_annotated_heatmap_loss(
    loss_name: str,
    asl_criterion: AsymmetricBinaryLoss,
    focal_criterion: FocalHeatmapLoss,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if loss_name == "focal_heatmap":
        return pixelwise_focal_heatmap_loss(focal_criterion, logits, targets)
    return pixelwise_asl_loss(asl_criterion, logits, targets)


def build_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, search_threshold: bool) -> dict[str, float]:
    thresholds = [threshold]
    if search_threshold:
        thresholds = np.linspace(0.05, 0.95, 19).tolist()
    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = 0.0
        auprc = 0.0
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float] | None = None
    for current_threshold in thresholds:
        y_pred = (y_prob >= current_threshold).astype(np.int64)
        metrics = {
            "threshold": float(current_threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auroc": auroc,
            "auprc": auprc,
        }
        key = (metrics["f1"], metrics["precision"], metrics["recall"])
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
    return best_metrics or {
        "threshold": float(threshold),
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
    }


def localization_score(
    gt_points: list[tuple[float, float]],
    gt_confidences: list[float],
    predicted_points: list[tuple[float, float, float]],
    width: int,
    height: int,
) -> float:
    if not gt_points and not predicted_points:
        return 1.0
    if gt_points and not predicted_points:
        return 0.0
    diag = math.sqrt(width * width + height * height)
    bandwidth = 0.05 * diag
    total = 0.0
    total_weight = 0.0
    for idx, (gt_x, gt_y) in enumerate(gt_points):
        weight = gt_confidences[idx] if idx < len(gt_confidences) else 1.0
        distances = [math.dist((gt_x, gt_y), (pred_x, pred_y)) for pred_x, pred_y, _ in predicted_points]
        best = min(distances) if distances else diag
        total += weight * math.exp(-best / max(bandwidth, 1e-6))
        total_weight += weight
    return total / max(total_weight, 1e-6)


def extract_peaks_from_heatmap(heatmap: torch.Tensor, top_k: int, peak_threshold: float, nms_kernel: int) -> list[tuple[float, float, float]]:
    if top_k <= 0:
        return []
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(0)
    heatmap = heatmap.float().clamp(0.0, 1.0)
    padded = torch.nn.functional.max_pool2d(
        heatmap.unsqueeze(0).unsqueeze(0),
        kernel_size=nms_kernel,
        stride=1,
        padding=nms_kernel // 2,
    ).squeeze(0).squeeze(0)
    keep = (heatmap >= padded) & (heatmap >= peak_threshold)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    if ys.numel() == 0:
        return []
    scores = heatmap[ys, xs]
    order = torch.argsort(scores, descending=True)[:top_k]
    peaks: list[tuple[float, float, float]] = []
    for idx in order.tolist():
        peaks.append((float(xs[idx].item()), float(ys[idx].item()), float(scores[idx].item())))
    return peaks


def extract_cam_targets(cam_heatmap: torch.Tensor, peak_threshold: float, nms_kernel: int) -> tuple[torch.Tensor, torch.Tensor]:
    if cam_heatmap.ndim != 4:
        raise ValueError("Expected [batch, 1, H, W] CAM heatmaps")
    pooled = torch.nn.functional.max_pool2d(cam_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
    keep = (cam_heatmap >= pooled) & (cam_heatmap >= peak_threshold)
    target = (cam_heatmap * keep.float()).detach()
    valid = (target.flatten(start_dim=1).max(dim=1).values > 0).float()
    return target, valid


def weak_consistency_weight(epoch_index: int, config: dict[str, Any]) -> float:
    warmup_epochs = int(config["phase2"].get("weak_warmup_epochs", 5))
    ramp_epochs = int(config["phase2"].get("weak_ramp_epochs", 5))
    max_weight = float(config["phase2"].get("weak_loss_weight", 1.0))
    if epoch_index <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_weight
    progress = min(max((epoch_index - warmup_epochs) / ramp_epochs, 0.0), 1.0)
    return max_weight * progress


def evaluate_classifier(model: RadDinoClassifier, loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            outputs = model(image)
            presence_prob = torch.sigmoid(outputs.logits).cpu().numpy()
            presence_labels = batch["presence_label"].cpu().numpy().astype(np.int64)
            y_true.extend(presence_labels.tolist())
            y_prob.extend(presence_prob.tolist())
    metrics = build_threshold_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_prob, dtype=np.float32),
        threshold=float(config["inference"].get("presence_threshold", 0.6)),
        search_threshold=bool(config["evaluation"].get("search_threshold", True)),
    )
    metrics["score"] = metrics["f1"]
    return metrics


def evaluate_localizer(model: HeatmapLocalizer, loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    model.eval()
    
    # We always use a low threshold for extraction during eval to allow later threshold sweeps
    # Default search threshold is 0.05
    eval_peak_threshold = 0.05
    nms_kernel = int(config["inference"].get("nms_kernel", 7))
    max_detections = int(config["inference"].get("max_detections_per_image", 8))
    
    all_final_peaks: list[list[tuple[float, float, float]]] = []
    all_gt_points: list[list[tuple[float, float]]] = []
    negative_mask_list: list[bool] = []
    annotated_mask_list: list[bool] = []
    
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            outputs = model(image, output_size=(image.shape[-2], image.shape[-1]))
            heatmaps = torch.sigmoid(outputs.heatmap_logits).cpu()
            point_mask = batch["point_mask"].cpu().numpy()
            points = batch["points"].cpu().numpy()
            negatives = batch["negative_mask"].cpu().numpy()
            annotated = batch["annotated_mask"].cpu().numpy()
            
            for idx in range(image.size(0)):
                peaks = extract_peaks_from_heatmap(heatmaps[idx], top_k=max_detections, peak_threshold=eval_peak_threshold, nms_kernel=nms_kernel)
                all_final_peaks.append(peaks)
                negative_mask_list.append(bool(negatives[idx] > 0.5))
                annotated_bool = bool(annotated[idx] > 0.5)
                annotated_mask_list.append(annotated_bool)
                if annotated_bool:
                    gt = [tuple(map(float, points[idx][p_idx])) for p_idx in range(MAX_NODULES) if point_mask[idx][p_idx] > 0.5]
                    all_gt_points.append(gt)
                else:
                    all_gt_points.append([])

    # Threshold Sweep
    search_threshold = bool(config["evaluation"].get("search_threshold", True))
    if search_threshold:
        thresholds = np.linspace(0.05, 0.8, 16).tolist()
    else:
        thresholds = [float(config["inference"].get("peak_threshold", 0.25))]

    best_metrics: dict[str, float] = {}
    best_score = -float("inf")
    neg_penalty_weight = float(config["evaluation"].get("negative_peak_penalty", 0.1))

    for threshold in thresholds:
        loc_scores: list[float] = []
        neg_peaks: list[float] = []
        
        for idx, peaks in enumerate(all_final_peaks):
            # Filter peaks by current threshold
            filtered_peaks = [p for p in peaks if p[2] >= threshold]
            
            if negative_mask_list[idx]:
                neg_peaks.append(max((p[2] for p in filtered_peaks), default=0.0))
            
            if annotated_mask_list[idx]:
                loc_scores.append(
                    localization_score(
                        gt_points=all_gt_points[idx],
                        gt_confidences=[1.0] * len(all_gt_points[idx]),
                        predicted_points=filtered_peaks,
                        width=int(config["data"]["image_size"]),
                        height=int(config["data"]["image_size"]),
                    )
                )
        
        avg_loc = float(np.mean(loc_scores) if loc_scores else 0.0)
        avg_neg_peak = float(np.mean(neg_peaks) if neg_peaks else 0.0)
        current_score = avg_loc - neg_penalty_weight * avg_neg_peak
        
        if current_score >= best_score:
            best_score = current_score
            best_metrics = {
                "localization_score": avg_loc,
                "negative_peak_mean": avg_neg_peak,
                "score": current_score,
                "peak_threshold": threshold,
                "n_annotated": float(len(loc_scores)),
            }
            
    return best_metrics


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    merged = copy.deepcopy(state_dicts[0])
    for key in merged:
        accumulator = state_dicts[0][key].clone()
        for state in state_dicts[1:]:
            accumulator += state[key]
        merged[key] = accumulator / len(state_dicts)
    return merged


def train_phase1_fold(config: dict[str, Any], train_records: list[SampleRecord], val_records: list[SampleRecord], output_dir: Path) -> tuple[Path, dict[str, float]]:
    device = build_device()
    structured_train_records = filter_structured_records(train_records)
    train_loader = create_phase1_train_loader(structured_train_records, config)
    eval_records = sample_eval_records(
        val_records,
        max_samples=int(config["evaluation"].get("max_eval_samples", 2048) or 0),
        seed=int(config.get("seed", 42)),
    )
    val_loader = create_eval_loader(eval_records, config)
    model = RadDinoClassifier(
        backbone_name=config["model"]["backbone_name"],
        pretrained=bool(config["model"].get("pretrained", True)),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["phase1"]["lr"]), weight_decay=float(config["phase1"].get("weight_decay", 1e-4)))
    scaler = build_grad_scaler(config, device)
    criterion = build_asl(config["phase1"])
    best_metrics: dict[str, float] = {}
    best_path = output_dir / "classifier_best.pt"
    best_score = -1.0
    total_epochs = int(config["phase1"]["epochs"])
    positive_target = float(config["phase1"].get("positive_target", 1.0))
    negative_target = float(config["phase1"].get("negative_target", 0.0))
    for epoch in range(total_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()
        pos_seen = 0
        sample_seen = 0
        for batch in tqdm(train_loader, desc=f"Phase1 epoch {epoch + 1}", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            presence_label = batch["presence_label"].to(device, non_blocking=True)
            smoothed_presence_label = smooth_binary_targets(presence_label, positive_target=positive_target, negative_target=negative_target)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                outputs = model(image)
                loss = criterion(outputs.logits.unsqueeze(1), smoothed_presence_label.unsqueeze(1)).mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["phase1"].get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            batch_size = image.size(0)
            pos_seen += int((presence_label > 0.5).sum().item())
            sample_seen += batch_size
            loss_meter.update(float(loss.item()), batch_size)
        metrics = evaluate_classifier(model, val_loader, config, device)
        log_metrics(
            "phase1",
            epoch + 1,
            total_epochs,
            {
                "train_loss": loss_meter.avg,
                "batch_pos_ratio": pos_seen / max(sample_seen, 1),
                "val_threshold": metrics["threshold"],
                "val_f1": metrics["f1"],
                "val_score": metrics["score"],
            },
        )
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_metrics = metrics
            torch.save({"state_dict": model.state_dict(), "metrics": metrics, "config": config}, best_path)
    return best_path, best_metrics


def evaluate_phase1_soup(state_dicts: list[dict[str, torch.Tensor]], val_records_by_fold: list[list[SampleRecord]], config: dict[str, Any], device: torch.device, model: RadDinoClassifier | None = None) -> dict[str, float]:
    own_model = model is None
    if own_model:
        model = RadDinoClassifier(
            backbone_name=config["model"]["backbone_name"],
            pretrained=False,
        ).to(device)
    model.load_state_dict(average_state_dicts(state_dicts))
    records = [item for fold_records in val_records_by_fold for item in fold_records]
    soup_eval_max_samples = int(config["evaluation"].get("soup_max_eval_samples", config["evaluation"].get("max_eval_samples", 2048)) or 0)
    records = sample_eval_records(records, max_samples=soup_eval_max_samples, seed=int(config.get("seed", 42)))
    return evaluate_classifier(model, create_eval_loader(records, config), config, device)


def _load_ranked_checkpoints(checkpoint_paths: list[Path], soup_max_checkpoints: int) -> list[dict[str, Any]]:
    ranked_entries: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ranked_entries.append(
            {
                "path": path,
                "state_dict": payload["state_dict"],
                "metrics": payload.get("metrics", {}),
                "score": float(payload.get("metrics", {}).get("score", 0.0)),
            }
        )
    ranked_entries.sort(key=lambda item: item["score"], reverse=True)
    return ranked_entries[:soup_max_checkpoints]


def _greedy_select_phase1_states(ranked_entries: list[dict[str, Any]], val_records_by_fold: list[list[SampleRecord]], config: dict[str, Any], device: torch.device, model: RadDinoClassifier | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
    selected_entries = [ranked_entries[0]]
    best_metrics = evaluate_phase1_soup([item["state_dict"] for item in selected_entries], val_records_by_fold, config, device, model=model)
    best_score = best_metrics["score"]
    for candidate in ranked_entries[1:]:
        trial_entries = selected_entries + [candidate]
        trial_metrics = evaluate_phase1_soup([item["state_dict"] for item in trial_entries], val_records_by_fold, config, device, model=model)
        if trial_metrics["score"] >= best_score:
            selected_entries = trial_entries
            best_metrics = trial_metrics
            best_score = trial_metrics["score"]
    return selected_entries, best_metrics


def greedy_phase1_soup(checkpoint_paths: list[Path], val_records_by_fold: list[list[SampleRecord]], config: dict[str, Any], output_path: Path) -> Path:
    device = build_device()
    ranked_entries = _load_ranked_checkpoints(checkpoint_paths, int(config["phase1"].get("soup_max_checkpoints", 5)))
    if not ranked_entries:
        raise ValueError("No phase 1 checkpoints available for greedy soup")
    candidate_soups: list[dict[str, Any]] = []

    # Build model once — reused for all candidate evaluations to avoid repeated backbone init
    soup_model = RadDinoClassifier(
        backbone_name=config["model"]["backbone_name"],
        pretrained=False,
    ).to(device)

    best_single = ranked_entries[0]
    candidate_soups.append(
        {
            "name": "best_single",
            "state_dict": best_single["state_dict"],
            "metrics": best_single["metrics"],
            "members": [str(best_single["path"])],
        }
    )

    greedy_entries, greedy_metrics = _greedy_select_phase1_states(ranked_entries, val_records_by_fold, config, device, model=soup_model)
    candidate_soups.append(
        {
            "name": "greedy",
            "state_dict": average_state_dicts([item["state_dict"] for item in greedy_entries]),
            "metrics": greedy_metrics,
            "members": [str(item["path"]) for item in greedy_entries],
        }
    )

    if len(ranked_entries) >= 2:
        uniform_all_states = [item["state_dict"] for item in ranked_entries]
        uniform_all_metrics = evaluate_phase1_soup(uniform_all_states, val_records_by_fold, config, device, model=soup_model)
        candidate_soups.append(
            {
                "name": "uniform_all",
                "state_dict": average_state_dicts(uniform_all_states),
                "metrics": uniform_all_metrics,
                "members": [str(item["path"]) for item in ranked_entries],
            }
        )

    if len(ranked_entries) >= 3:
        kept_entries = ranked_entries[:-1]
        drop_worst_states = [item["state_dict"] for item in kept_entries]
        drop_worst_metrics = evaluate_phase1_soup(drop_worst_states, val_records_by_fold, config, device, model=soup_model)
        candidate_soups.append(
            {
                "name": "uniform_drop_worst",
                "state_dict": average_state_dicts(drop_worst_states),
                "metrics": drop_worst_metrics,
                "members": [str(item["path"]) for item in kept_entries],
            }
        )

    selected_candidate = max(candidate_soups, key=lambda item: float(item["metrics"].get("score", 0.0)))
    payload = {
        "state_dict": selected_candidate["state_dict"],
        "metrics": selected_candidate["metrics"],
        "config": config,
        "soup": {
            "strategy": selected_candidate["name"],
            "members": selected_candidate["members"],
            "candidates": {item["name"]: item["metrics"] for item in candidate_soups},
        },
    }
    torch.save(payload, output_path)
    return output_path


def train_phase2_fold(
    config: dict[str, Any],
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    classifier_checkpoint: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, float]]:
    device = build_device()
    train_loader = create_phase2_train_loader(train_records, config)
    eval_records = sample_phase2_eval_records(
        val_records,
        max_negatives=int(config["evaluation"].get("phase2_eval_max_negatives", 512) or 0),
        seed=int(config.get("seed", 42)),
    )
    val_loader = create_eval_loader(eval_records, config)
    model = HeatmapLocalizer(
        backbone_name=config["model"]["backbone_name"],
        pretrained=bool(config["model"].get("pretrained", True)),
        heatmap_hidden_channels=int(config["model"].get("heatmap_head_channels", 256)),
    ).to(device)
    classifier_state = torch.load(classifier_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_from_classifier(classifier_state)
    if bool(config["phase2"].get("freeze_classifier_head", True)):
        for parameter in model.classifier_head.parameters():
            parameter.requires_grad = False
    if bool(config["phase2"].get("freeze_encoder", False)):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["phase2"]["lr"]),
        weight_decay=float(config["phase2"].get("weight_decay", 1e-4)),
    )
    scaler = build_grad_scaler(config, device)
    negative_criterion = build_asl(config["phase2"])
    annotated_asl_criterion = build_asl(config["phase2"])
    annotated_focal_criterion = build_focal_heatmap_loss(config["phase2"])
    annotated_loss_name = str(config["phase2"].get("annotated_loss", "asl")).lower()
    best_metrics: dict[str, float] = {}
    best_path = output_dir / "localizer_best.pt"
    best_score = -1.0
    total_epochs = int(config["phase2"]["epochs"])
    lambda_annotated = float(config["phase2"].get("annotated_loss_weight", 1.0))
    lambda_negative = float(config["phase2"].get("negative_loss_weight", 1.0))
    peak_threshold = float(config["phase2"].get("cam_peak_threshold", 0.4))
    nms_kernel = int(config["phase2"].get("cam_nms_kernel", 7))

    if total_epochs == 0:
        if best_path.exists():
            checkpoint_data = torch.load(best_path, map_location="cpu", weights_only=False)
            loaded_metrics = checkpoint_data.get("metrics", {})
            tqdm.write(f"[fold] epochs=0 → found existing checkpoint: {best_path} (score={loaded_metrics.get('score', 'N/A')})")
            return best_path, loaded_metrics
        else:
            tqdm.write(f"[fold] epochs=0 but NO checkpoint found at {best_path}. Evaluating initial model...")
            initial_metrics = evaluate_localizer(model, val_loader, config, device)
            return best_path, initial_metrics

    for epoch in range(total_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        if bool(config["phase2"].get("freeze_encoder", False)):
            model.encoder.eval()
        total_meter = AverageMeter()
        ann_meter = AverageMeter()
        neg_meter = AverageMeter()
        weak_meter = AverageMeter()
        weak_weight = weak_consistency_weight(epoch + 1, config)
        for batch in tqdm(train_loader, desc=f"Phase2 epoch {epoch + 1}", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            heatmap_target = batch["heatmap_target"].to(device, non_blocking=True)
            annotated_mask = batch["annotated_mask"].to(device, non_blocking=True)
            negative_mask = batch["negative_mask"].to(device, non_blocking=True)
            unannotated_mask = batch["unannotated_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                outputs = model(image, output_size=(image.shape[-2], image.shape[-1]))
                annotated_loss = masked_mean(
                    compute_annotated_heatmap_loss(
                        annotated_loss_name,
                        annotated_asl_criterion,
                        annotated_focal_criterion,
                        outputs.heatmap_logits,
                        heatmap_target,
                    ),
                    annotated_mask,
                )
                negative_target = torch.zeros_like(outputs.heatmap_logits)
                negative_loss = masked_mean(pixelwise_asl_loss(negative_criterion, outputs.heatmap_logits, negative_target), negative_mask)
                
                # Peak suppression on negatives: penalise activations that approach the
                # detection threshold (0.4). A margin of 0.15 gives the model room to
                # maintain overall confidence without triggering false detections.
                if negative_mask.any():
                    neg_preds = torch.sigmoid(outputs.heatmap_logits)
                    neg_max_vals = (neg_preds * negative_mask.view(-1, 1, 1, 1)).flatten(start_dim=1).max(dim=1)[0]
                    peak_loss = torch.mean(torch.clamp(neg_max_vals - 0.15, min=0.0))
                else:
                    peak_loss = 0.0

                cam_target, weak_valid = extract_cam_targets(outputs.cam_heatmap, peak_threshold=peak_threshold, nms_kernel=nms_kernel)
                weak_mask = unannotated_mask * weak_valid.to(unannotated_mask.device)
                weak_loss = masked_mean(pixelwise_asl_loss(negative_criterion, outputs.heatmap_logits, cam_target), weak_mask)
                
                loss = (
                    lambda_annotated * annotated_loss +
                    lambda_negative * negative_loss +
                    2.0 * peak_loss +
                    weak_weight * weak_loss
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["phase2"].get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            batch_size = image.size(0)
            total_meter.update(float(loss.item()), batch_size)
            ann_meter.update(float(annotated_loss.item()), batch_size)
            neg_meter.update(float(negative_loss.item()), batch_size)
            weak_meter.update(float(weak_loss.item()), batch_size)
        metrics = evaluate_localizer(model, val_loader, config, device)
        log_metrics(
            "phase2",
            epoch + 1,
            total_epochs,
            {
                "train_loss": total_meter.avg,
                "annotated_loss": ann_meter.avg,
                "negative_loss": neg_meter.avg,
                "weak_loss": weak_meter.avg,
                "weak_weight": weak_weight,
                "val_loc": metrics["localization_score"],
                "val_neg_peak": metrics["negative_peak_mean"],
                "val_score": metrics["score"],
            },
        )
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_metrics = metrics
            torch.save({"state_dict": model.state_dict(), "metrics": metrics, "config": config}, best_path)
    return best_path, best_metrics


def evaluate_phase2_soup(state_dicts: list[dict[str, torch.Tensor]], val_records_by_fold: list[list[SampleRecord]], config: dict[str, Any], device: torch.device, model: HeatmapLocalizer | None = None, tuning_records: list[SampleRecord] | None = None) -> dict[str, float]:
    own_model = model is None
    if own_model:
        model = HeatmapLocalizer(
            backbone_name=config["model"]["backbone_name"],
            pretrained=False,
            heatmap_hidden_channels=int(config["model"].get("heatmap_head_channels", 256)),
        ).to(device)
    model.load_state_dict(average_state_dicts(state_dicts))
    
    if tuning_records:
        records = tuning_records
    else:
        records = [item for fold_records in val_records_by_fold for item in fold_records]
        # Use soup_phase2_eval_max_negatives for the pooled soup eval; fall back to phase2_eval_max_negatives
        max_negatives = int(config["evaluation"].get(
            "soup_phase2_eval_max_negatives",
            config["evaluation"].get("phase2_eval_max_negatives", 512),
        ) or 0)
        records = sample_phase2_eval_records(records, max_negatives=max_negatives, seed=int(config.get("seed", 42)))
        
    return evaluate_localizer(model, create_eval_loader(records, config), config, device)


def _greedy_select_phase2_states(ranked_entries: list[dict[str, Any]], val_records_by_fold: list[list[SampleRecord]], config: dict[str, Any], device: torch.device, model: HeatmapLocalizer | None = None, tuning_records: list[SampleRecord] | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
    selected_entries = [ranked_entries[0]]
    best_metrics = evaluate_phase2_soup([item["state_dict"] for item in selected_entries], val_records_by_fold, config, device, model=model, tuning_records=tuning_records)
    best_score = best_metrics["score"]
    for candidate in ranked_entries[1:]:
        trial_entries = selected_entries + [candidate]
        trial_metrics = evaluate_phase2_soup([item["state_dict"] for item in trial_entries], val_records_by_fold, config, device, model=model, tuning_records=tuning_records)
        trial_score = trial_metrics["score"]
        if trial_score >= best_score:
            selected_entries = trial_entries
            best_score = trial_score
            best_metrics = trial_metrics
    return selected_entries, best_metrics


def greedy_phase2_soup(checkpoint_paths: list[Path], val_records_by_fold: list[list[SampleRecord]], config: dict[str, Any], output_path: Path, tuning_records: list[SampleRecord] | None = None) -> Path:
    device = build_device()
    ranked_entries = _load_ranked_checkpoints(checkpoint_paths, int(config["phase2"].get("soup_max_checkpoints", 5)))
    if not ranked_entries:
        raise ValueError("No phase 2 checkpoints available for greedy soup")
    candidate_soups: list[dict[str, Any]] = []

    # Build model once — reused for all candidate evaluations to avoid repeated backbone init
    soup_model = HeatmapLocalizer(
        backbone_name=config["model"]["backbone_name"],
        pretrained=False,
        heatmap_hidden_channels=int(config["model"].get("heatmap_head_channels", 256)),
    ).to(device)

    best_single = ranked_entries[0]
    best_single_metrics = evaluate_phase2_soup([best_single["state_dict"]], val_records_by_fold, config, device, model=soup_model, tuning_records=tuning_records)
    candidate_soups.append(
        {
            "name": "best_single",
            "state_dict": best_single["state_dict"],
            "metrics": best_single_metrics,
            "members": [str(best_single["path"])],
        }
    )

    greedy_entries, greedy_metrics = _greedy_select_phase2_states(ranked_entries, val_records_by_fold, config, device, model=soup_model, tuning_records=tuning_records)
    candidate_soups.append(
        {
            "name": "greedy",
            "state_dict": average_state_dicts([item["state_dict"] for item in greedy_entries]),
            "metrics": greedy_metrics,
            "members": [str(item["path"]) for item in greedy_entries],
        }
    )

    if len(ranked_entries) >= 2:
        uniform_all_states = [item["state_dict"] for item in ranked_entries]
        uniform_all_metrics = evaluate_phase2_soup(uniform_all_states, val_records_by_fold, config, device, model=soup_model, tuning_records=tuning_records)
        candidate_soups.append(
            {
                "name": "uniform_all",
                "state_dict": average_state_dicts(uniform_all_states),
                "metrics": uniform_all_metrics,
                "members": [str(item["path"]) for item in ranked_entries],
            }
        )

    if len(ranked_entries) >= 3:
        kept_entries = ranked_entries[:-1]
        drop_worst_states = [item["state_dict"] for item in kept_entries]
        drop_worst_metrics = evaluate_phase2_soup(drop_worst_states, val_records_by_fold, config, device, model=soup_model, tuning_records=tuning_records)
        candidate_soups.append(
            {
                "name": "uniform_drop_worst",
                "state_dict": average_state_dicts(drop_worst_states),
                "metrics": drop_worst_metrics,
                "members": [str(item["path"]) for item in kept_entries],
            }
        )

    selected_candidate = max(candidate_soups, key=lambda item: float(item["metrics"].get("score", 0.0)))
    tqdm.write(f"[phase2][soup] winner: {selected_candidate['name']} | score={selected_candidate['metrics']['score']:.4f} | threshold={selected_candidate['metrics'].get('peak_threshold', 'N/A')}")
    payload = {
        "state_dict": selected_candidate["state_dict"],
        "metrics": selected_candidate["metrics"],
        "config": config,
        "soup": {
            "strategy": selected_candidate["name"],
            "members": selected_candidate["members"],
            "candidates": {item["name"]: item["metrics"] for item in candidate_soups},
        },
    }
    torch.save(payload, output_path)
    return output_path


def evaluate_classifier_checkpoint(checkpoint_path: str | Path, records: list[SampleRecord], config: dict[str, Any]) -> dict[str, float]:
    device = build_device()
    model = load_classifier_for_inference(checkpoint_path, config)
    return evaluate_classifier(model, create_eval_loader(records, config), config, device)


def evaluate_localizer_checkpoint(checkpoint_path: str | Path, records: list[SampleRecord], config: dict[str, Any]) -> dict[str, float]:
    device = build_device()
    model = load_localizer_for_inference(checkpoint_path, config)
    return evaluate_localizer(model, create_eval_loader(records, config), config, device)


def attach_checkpoint_metrics(checkpoint_path: str | Path, metrics: dict[str, float], replace_primary_metrics: bool = False) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state["full_dataset_metrics"] = metrics
    if replace_primary_metrics:
        merged_metrics = dict(state.get("metrics", {}))
        merged_metrics.update(metrics)
        state["metrics"] = merged_metrics
    torch.save(state, checkpoint_path)


def run_full_dataset_evaluation(
    output_root: Path,
    metadata: list[SampleRecord],
    phase1_checkpoints: list[Path],
    phase2_checkpoints: list[Path],
    classifier_soup: Path,
    localizer_soup: Path,
    config: dict[str, Any],
) -> Path:
    eval_config = copy.deepcopy(config)
    post_training_cfg = eval_config.get("post_training", {})
    if "search_threshold_on_full_dataset" in post_training_cfg:
        eval_config.setdefault("evaluation", {})["search_threshold"] = bool(post_training_cfg["search_threshold_on_full_dataset"])

    report: dict[str, Any] = {"classification": {}, "localization": {}, "num_records": len(metadata)}

    for fold_idx, checkpoint_path in enumerate(phase1_checkpoints):
        metrics = evaluate_classifier_checkpoint(checkpoint_path, metadata, eval_config)
        report["classification"][f"fold_{fold_idx}"] = metrics
        tqdm.write(
            f"[full-eval][phase1][fold {fold_idx}] threshold={metrics['threshold']:.4f} | f1={metrics['f1']:.4f} | "
            f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f}"
        )

    soup_classifier_metrics = evaluate_classifier_checkpoint(classifier_soup, metadata, eval_config)
    report["classification"]["soup"] = soup_classifier_metrics
    tqdm.write(
        f"[full-eval][phase1][soup] threshold={soup_classifier_metrics['threshold']:.4f} | f1={soup_classifier_metrics['f1']:.4f} | "
        f"precision={soup_classifier_metrics['precision']:.4f} | recall={soup_classifier_metrics['recall']:.4f}"
    )
    attach_checkpoint_metrics(
        classifier_soup,
        soup_classifier_metrics,
        replace_primary_metrics=bool(post_training_cfg.get("calibrate_classifier_soup_threshold", False)),
    )

    for fold_idx, checkpoint_path in enumerate(phase2_checkpoints):
        metrics = evaluate_localizer_checkpoint(checkpoint_path, metadata, eval_config)
        report["localization"][f"fold_{fold_idx}"] = metrics
        tqdm.write(
            f"[full-eval][phase2][fold {fold_idx}] score={metrics['score']:.4f} | loc={metrics['localization_score']:.4f} | "
            f"neg_peak={metrics['negative_peak_mean']:.4f}"
        )

    soup_localizer_metrics = evaluate_localizer_checkpoint(localizer_soup, metadata, eval_config)
    report["localization"]["soup"] = soup_localizer_metrics
    tqdm.write(
        f"[full-eval][phase2][soup] score={soup_localizer_metrics['score']:.4f} | loc={soup_localizer_metrics['localization_score']:.4f} | "
        f"neg_peak={soup_localizer_metrics['negative_peak_mean']:.4f}"
    )
    attach_checkpoint_metrics(localizer_soup, soup_localizer_metrics, replace_primary_metrics=False)

    report_path = output_root / "full_dataset_evaluation.json"
    save_json(report, report_path)
    return report_path


def train_two_phase_pipeline(config: dict[str, Any]) -> dict[str, str]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    output_root = ensure_dir(config.get("output_root", "outputs"))
    metadata = build_metadata(config)
    max_negative_samples = int(config["data"].get("max_negative_samples", 0) or 0)
    max_samples = int(config["data"].get("max_samples", 0) or 0)
    if max_negative_samples > 0:
        metadata = sample_with_all_positives(metadata, max_negative_samples=max_negative_samples, seed=seed)
    elif max_samples > 0:
        metadata = sample_records(metadata, max_samples=max_samples, seed=seed)

    # Optionally split a held-out set that is excluded from ALL training folds.
    # Used for localizer soup evaluation (avoids any data leakage for detection).
    held_out_fraction = float(config["data"].get("held_out_fraction", 0.0))
    held_out_records: list[SampleRecord] = []
    if held_out_fraction > 0.0:
        held_out_records, metadata = split_held_out(metadata, held_out_fraction, seed)
        held_out_groups: dict[str, int] = {}
        for record in held_out_records:
            held_out_groups[record.structured_group()] = held_out_groups.get(record.structured_group(), 0) + 1
        tqdm.write(f"[setup] held_out={len(held_out_records)} groups={held_out_groups} | training_pool={len(metadata)}")

    skip_phase1 = (int(config["phase1"].get("epochs", 100)) == 0)
    num_folds = max(2, int(config["training"].get("folds", 5)))
    folds = create_stratified_folds(metadata, num_folds=num_folds, seed=seed)
    phase1_checkpoints: list[Path] = []
    phase2_checkpoints: list[Path] = []
    val_records_by_fold: list[list[SampleRecord]] = []
    phase1_metrics: dict[str, Any] = {}
    phase2_metrics: dict[str, Any] = {}
    soup_dir = ensure_dir(output_root / "soups")
    # Pre-resolve classifier soup if skipping phase 1
    classifier_soup = None
    if skip_phase1:
        classifier_soup = soup_dir / "classifier_soup.pt"
        if not classifier_soup.exists():
            raise FileNotFoundError(f"phase1.epochs=0 but no classifier soup found at {classifier_soup}")
        tqdm.write(f"[phase1] epochs=0 → skipping training, using existing soup: {classifier_soup}")

    for fold_idx, split in folds.items():
        fold_dir = ensure_dir(output_root / f"fold_{fold_idx}")
        train_records = [metadata[idx] for idx in split["train"]]
        val_records = [metadata[idx] for idx in split["val"]]

        # Phase 1: Classifier
        if not skip_phase1:
            phase1_dir = ensure_dir(fold_dir / "phase1_classifier")
            phase1_checkpoint, fold_phase1_metrics = train_phase1_fold(config, train_records, val_records, phase1_dir)
            save_json(fold_phase1_metrics, phase1_dir / "metrics.json")
            phase1_checkpoints.append(phase1_checkpoint)
            phase1_metrics[f"fold_{fold_idx}"] = fold_phase1_metrics
        else:
            phase1_checkpoint = classifier_soup

        # Phase 2: Localizer (Always symmetrical with Phase 1)
        phase2_dir = ensure_dir(fold_dir / "phase2_localizer")
        phase2_checkpoint, fold_phase2_metrics = train_phase2_fold(config, train_records, val_records, phase1_checkpoint, phase2_dir)
        save_json(fold_phase2_metrics, phase2_dir / "metrics.json")
        phase2_checkpoints.append(phase2_checkpoint)
        phase2_metrics[f"fold_{fold_idx}"] = fold_phase2_metrics

        val_records_by_fold.append(val_records)

    # Soul Selection
    if not skip_phase1:
        classifier_soup = greedy_phase1_soup(phase1_checkpoints, val_records_by_fold, config, soup_dir / "classifier_soup.pt")

    # Phase 2 Soup Selection (requested by user: same method as phase 1)
    # Use held_out_records for soup tuning if available
    tuning_recs = None
    if held_out_records:
        tqdm.write(f"[phase2][soup] tuning on held_out_records ({len(held_out_records)} samples)")
        tuning_recs = sample_phase2_eval_records(held_out_records, max_negatives=int(config["evaluation"].get("soup_phase2_eval_max_negatives", 512)), seed=seed)
    
    localizer_soup = greedy_phase2_soup(phase2_checkpoints, val_records_by_fold, config, soup_dir / "localizer_soup.pt", tuning_records=tuning_recs)
    phase1_metrics_path = output_root / "phase1_fold_metrics.json"
    phase2_metrics_path = output_root / "phase2_fold_metrics.json"
    save_json(phase1_metrics, phase1_metrics_path)
    save_json(phase2_metrics, phase2_metrics_path)
    outputs = {
        "classifier_checkpoint": str(classifier_soup),
        "localizer_checkpoint": str(localizer_soup),
        "phase1_metrics_path": str(phase1_metrics_path),
        "phase2_metrics_path": str(phase2_metrics_path),
    }
    if bool(config.get("post_training", {}).get("report_full_dataset_metrics", False)):
        full_dataset_report_path = run_full_dataset_evaluation(
            output_root=output_root,
            metadata=metadata,
            phase1_checkpoints=phase1_checkpoints,
            phase2_checkpoints=phase2_checkpoints,
            classifier_soup=classifier_soup,
            localizer_soup=localizer_soup,
            config=config,
        )
        outputs["full_dataset_evaluation_path"] = str(full_dataset_report_path)
    return outputs


def load_classifier_for_inference(checkpoint_path: str | Path, config: dict[str, Any]) -> RadDinoClassifier:
    device = build_device()
    model = RadDinoClassifier(
        backbone_name=config["model"]["backbone_name"],
        pretrained=False,
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def resolve_classifier_threshold(checkpoint_path: str | Path, config: dict[str, Any]) -> float:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = state.get("metrics", {})
    tuned_threshold = metrics.get("threshold")
    if tuned_threshold is not None:
        return float(tuned_threshold)
    return float(config["inference"].get("presence_threshold", 0.6))


def load_localizer_for_inference(checkpoint_path: str | Path, config: dict[str, Any]) -> HeatmapLocalizer:
    device = build_device()
    model = HeatmapLocalizer(
        backbone_name=config["model"]["backbone_name"],
        pretrained=False,
        heatmap_hidden_channels=int(config["model"].get("heatmap_head_channels", 256)),
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def infer_from_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    classifier_checkpoint: str | Path,
    localizer_checkpoint: str | Path,
    config: dict[str, Any],
) -> dict[str, str]:
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    device = build_device()
    classifier = load_classifier_for_inference(classifier_checkpoint, config)
    localizer = load_localizer_for_inference(localizer_checkpoint, config)
    threshold = resolve_classifier_threshold(classifier_checkpoint, config)
    peak_threshold = float(config["inference"].get("peak_threshold", 0.25))
    nms_kernel = int(config["inference"].get("nms_kernel", 7))
    max_detections = int(config["inference"].get("max_detections_per_image", 8))
    image_paths = sorted([path for path in input_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    classification_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    image_size = int(config["data"]["image_size"])

    for image_path in tqdm(image_paths, desc="Inference", leave=False):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        normalized = image.astype(np.float32)
        normalized -= normalized.min()
        if normalized.max() > 0:
            normalized /= normalized.max()
        processed = apply_clahe(
            normalized,
            clip_limit=float(config["augmentation"].get("clahe_clip_limit", 2.0)),
            tile_grid_size=int(config["augmentation"].get("clahe_tile_grid_size", 8)),
        )
        image_resized = cv2.resize(processed, (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensor = to_three_channel_tensor(image_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            classifier_outputs = classifier(tensor)
            prob = float(torch.sigmoid(classifier_outputs.logits)[0].item())
            label_name = "Nodule" if prob >= threshold else "No Finding"
            classification_rows.append({"file_name": image_path.name, "label": label_name, "confidence": prob})
            peaks: list[tuple[float, float, float]] = []
            if prob >= threshold:
                localizer_outputs = localizer(tensor, output_size=(image_size, image_size))
                peaks = extract_peaks_from_heatmap(torch.sigmoid(localizer_outputs.heatmap_logits[0]).cpu(), top_k=max_detections, peak_threshold=peak_threshold, nms_kernel=nms_kernel)
        for x_resized, y_resized, confidence_value in peaks:
            x_px = (x_resized / max(image_size - 1, 1)) * max(width - 1, 1)
            y_px = (y_resized / max(image_size - 1, 1)) * max(height - 1, 1)
            localization_rows.append(
                {
                    "file_name": image_path.name,
                    "x": x_px,
                    "y": y_px,
                    "confidence": confidence_value,
                }
            )

    classification_path = output_dir / "classification_test_results.csv"
    localization_path = output_dir / "localization_test_results.csv"
    pd.DataFrame(classification_rows).to_csv(classification_path, index=False)
    pd.DataFrame(localization_rows, columns=["file_name", "x", "y", "confidence"]).to_csv(localization_path, index=False)
    return {
        "classification_csv": str(classification_path),
        "localization_csv": str(localization_path),
    }
