from __future__ import annotations

import torch
import torch.nn as nn
 

class AsymmetricBinaryLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        pos_term = targets * torch.log(xs_pos.clamp(min=self.eps, max=1.0))
        neg_term = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps, max=1.0))
        loss = pos_term + neg_term
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = -loss * torch.pow(1.0 - pt, gamma)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class FocalHeatmapLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6, positive_threshold: float = 0.95, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.positive_threshold = positive_threshold
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)
        pos_mask = (targets >= self.positive_threshold).float()
        neg_mask = (targets < self.positive_threshold).float()
        neg_weights = torch.pow(1.0 - targets, self.beta)

        pos_loss = -torch.log(probs) * torch.pow(1.0 - probs, self.alpha) * pos_mask
        neg_loss = -torch.log(1.0 - probs) * torch.pow(probs, self.alpha) * neg_weights * neg_mask

        # Normalize positive loss by number of positive pixels to ensure a strong signal
        pos_loss_sum = pos_loss.flatten(start_dim=1).sum(dim=1)
        pos_pixel_count = pos_mask.flatten(start_dim=1).sum(dim=1).clamp(min=1.0)
        pos_loss_norm = pos_loss_sum / pos_pixel_count

        # Normalize negative loss by total pixels (default stable behavior)
        neg_loss_norm = neg_loss.flatten(start_dim=1).mean(dim=1)

        loss = pos_loss_norm + neg_loss_norm

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
