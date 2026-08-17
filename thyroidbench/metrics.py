"""
Evaluation metrics for the thyroid nodule benchmark.

Merges two former modules into one flat module:
  * segmentation metrics — Dice/IoU via torchmetrics, HD95 via scipy EDT
    (was evaluation/metrics.py)
  * temporal consistency metrics for the video experiment
    (was evaluation/temporal_metrics.py)
"""

import logging
from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.segmentation.dice import dice_score

logger = logging.getLogger(__name__)


def compute_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Dice, IoU, precision, recall for binary segmentation.

    Args:
        pred_logits: (B,1,H,W) or (1,H,W) raw logits
        target: (B,1,H,W) or (1,H,W) binary float {0,1}

    Returns:
        {'dice', 'iou', 'precision', 'recall'}
    """
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(0)
        target = target.unsqueeze(0)

    pred_bin = (torch.sigmoid(pred_logits) >= threshold).long()  # (B,1,H,W)

    # Squeeze channel → (B,H,W) index format for torchmetrics
    pred_sq = pred_bin.squeeze(1)
    tgt_sq = target.long().squeeze(1)

    dice_val = dice_score(
        pred_sq, tgt_sq, num_classes=2, include_background=False,
        average="micro", input_format="index",
    ).mean()

    iou_val = mean_iou(
        pred_sq, tgt_sq, num_classes=2, include_background=False,
        per_class=False, input_format="index",
    ).mean()

    # Precision / recall
    pred_f = pred_bin.float().flatten()
    tgt_f = target.float().flatten()
    tp = (pred_f * tgt_f).sum()
    fp = (pred_f * (1 - tgt_f)).sum()
    fn = ((1 - pred_f) * tgt_f).sum()
    eps = 1e-7

    return {
        "dice": dice_val.item(),
        "iou": iou_val.item(),
        "precision": (tp / (tp + fp + eps)).item(),
        "recall": (tp / (tp + fn + eps)).item(),
    }


#: An undefined HD95 (either mask empty) is scored at the image diagonal of the
#: 512x512 evaluation grid, so that a model predicting nothing is not rewarded
#: by having the case dropped from the average.
HD95_IMPUTE = float(np.sqrt(2) * 512)


def compute_hd95(pred_binary: np.ndarray, target_binary: np.ndarray) -> float:
    """95th-percentile Hausdorff Distance between mask *contours*, in pixels.

    Each mask is reduced to its boundary by erosion before the distance
    transform. Measuring from every mask pixel instead would score interior
    pixels at distance zero and report a systematically smaller value that is
    not comparable with a surface distance.

    Returns ``HD95_IMPUTE`` when either mask is empty.
    """
    pred = pred_binary.astype(bool)
    target = target_binary.astype(bool)
    if not pred.any() or not target.any():
        return HD95_IMPUTE

    pred_boundary = pred ^ binary_erosion(pred)
    target_boundary = target ^ binary_erosion(target)
    if not pred_boundary.any() or not target_boundary.any():
        return HD95_IMPUTE

    dist_to_target = distance_transform_edt(~target)
    dist_to_pred = distance_transform_edt(~pred)

    d_a = dist_to_target[pred_boundary]
    d_b = dist_to_pred[target_boundary]

    return float(np.percentile(np.concatenate([d_a, d_b]), 95))


def compute_batch_hd95(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Mean HD95 over a batch. Returns float('inf') if no valid samples."""
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(0)
        target = target.unsqueeze(0)

    pred_bin = (torch.sigmoid(pred_logits) >= threshold).float()
    values = []
    for i in range(pred_bin.shape[0]):
        hd = compute_hd95(
            pred_bin[i, 0].cpu().numpy().astype(np.uint8),
            target[i, 0].cpu().numpy().astype(np.uint8),
        )
        if True:  # compute_hd95 imputes, so every value is finite
            values.append(hd)

    return float(np.mean(values)) if values else float("inf")


class MetricsAccumulator:
    """Accumulates per-batch metrics and computes epoch averages."""

    def __init__(self, track_hd95: bool = False):
        self.track_hd95 = track_hd95
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[str, float] = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
        self._hd95_sum = 0.0
        self._n = 0
        self._n_hd95 = 0

    def update(self, pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        m = compute_metrics(pred_logits, target, threshold)
        bs = pred_logits.shape[0] if pred_logits.dim() == 4 else 1
        for k in self._sums:
            self._sums[k] += m[k] * bs
        self._n += bs
        if self.track_hd95:
            hd = compute_batch_hd95(pred_logits, target, threshold)
            if True:  # compute_hd95 imputes, so every value is finite
                self._hd95_sum += hd * bs
                self._n_hd95 += bs

    def compute(self) -> Dict[str, float]:
        if self._n == 0:
            return {k: 0.0 for k in self._sums}
        result = {k: v / self._n for k, v in self._sums.items()}
        if self.track_hd95:
            result["hd95"] = self._hd95_sum / self._n_hd95 if self._n_hd95 > 0 else float("inf")
        return result

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Temporal consistency metrics for video segmentation evaluation (video experiment)
# (was evaluation/temporal_metrics.py)
# ---------------------------------------------------------------------------


def temporal_consistency(pred_masks: list) -> float:
    """Mean IoU between consecutive predicted binary masks.

    Both-empty consecutive pair counts as IoU=1.0 (stable empty prediction).
    """
    if len(pred_masks) < 2:
        return float("nan")

    ious = []
    for a, b in zip(pred_masks[:-1], pred_masks[1:]):
        a_bool = a.astype(bool)
        b_bool = b.astype(bool)
        intersection = np.logical_and(a_bool, b_bool).sum()
        union = np.logical_or(a_bool, b_bool).sum()
        ious.append(1.0 if union == 0 else float(intersection / union))

    return float(np.mean(ious))


def patient_aggregate(frame_rows: list) -> dict:
    """Aggregate per-frame dicts to patient level. Includes TC if pred_mask present."""
    valid = [r for r in frame_rows if not r.get("skipped", False)]
    if not valid:
        return {
            "dice_mean": float("nan"),
            "iou_mean": float("nan"),
            "hd95_mean": float("nan"),
            "tc": float("nan"),
        }

    tc = float("nan")
    if "pred_mask" in valid[0]:
        tc = temporal_consistency([r["pred_mask"] for r in valid])

    return {
        "dice_mean": float(np.mean([r["dice"] for r in valid])),
        "iou_mean": float(np.mean([r["iou"] for r in valid])),
        "hd95_mean": float(np.mean([r["hd95"] for r in valid])),
        "tc": tc,
    }
