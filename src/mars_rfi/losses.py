"""Loss and metric helpers for binary RFI segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if weight is None:
        weight = torch.ones_like(target)
    intersection = (prob * target * weight).sum(dim=(1, 2, 3))
    denom = ((prob + target) * weight).sum(dim=(1, 2, 3)).clamp_min(eps)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def astro_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    astro_mask: torch.Tensor | None = None,
    fallback_to_negative: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Penalize RFI probability on clean astronomical pixels.

    If ``astro_mask`` has no positive pixels, optionally fall back to all
    non-RFI target pixels so the loss remains defined for non-FRB batches.
    """

    prob = torch.sigmoid(logits)
    if astro_mask is not None:
        support = (astro_mask * (1.0 - target)).clamp(0.0, 1.0)
        denom = support.sum()
        astro_loss = ((prob * prob) * support).sum() / denom.clamp_min(eps)
        has_astro = (denom > eps).to(dtype=logits.dtype)
        if not fallback_to_negative:
            return astro_loss * has_astro

        negative = (1.0 - target).clamp(0.0, 1.0)
        negative_loss = ((prob * prob) * negative).sum() / negative.sum().clamp_min(eps)
        return has_astro * astro_loss + (1.0 - has_astro) * negative_loss

    negative = (1.0 - target).clamp(0.0, 1.0)
    denom = negative.sum().clamp_min(eps)
    return ((prob * prob) * negative).sum() / denom


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    pos_weight: float = 1.0,
    lambda_dice: float = 1.0,
    focal_gamma: float = 0.0,
    lambda_astro: float = 0.0,
    astro_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pos = torch.as_tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos,
    )
    if focal_gamma and focal_gamma > 0.0:
        pt = torch.exp(-bce.detach())
        bce = ((1.0 - pt) ** float(focal_gamma)) * bce
    if weight is not None:
        bce = bce * weight
    loss = bce.mean()
    if lambda_dice and lambda_dice > 0.0:
        loss = loss + float(lambda_dice) * dice_loss_from_logits(logits, target, weight)
    if lambda_astro and lambda_astro > 0.0:
        loss = loss + float(lambda_astro) * astro_loss_from_logits(logits, target, astro_mask)
    return loss


@torch.no_grad()
def binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1.0e-7,
) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= float(threshold))
    truth = target >= 0.5
    tp = (pred & truth).sum(dtype=torch.float64)
    fp = (pred & ~truth).sum(dtype=torch.float64)
    fn = (~pred & truth).sum(dtype=torch.float64)
    tn = (~pred & ~truth).sum(dtype=torch.float64)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "iou": float(iou.item()),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "fn": float(fn.item()),
        "tn": float(tn.item()),
    }
