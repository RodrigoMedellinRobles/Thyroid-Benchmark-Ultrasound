"""Bounding-box helpers for prompt-conditioned segmentation.

For Sec 5.1 / 5.3 / 5.4 we want a fair comparison between foundation models
(SAM2, MedSAM, MedSAM-2 — which receive a box prompt) and traditional
baselines (UNet, TransUNet — which historically take only the image). To
match the foundation-model regime, every traditional baseline now also
receives the GT-derived bounding box as a 4th input channel (binary
rasterised box). Jitter (±10 % per side) is applied during training only,
matching the box jitter used by the LoRA-adapted foundation models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned bbox (x1, y1, x2, y2) per mask.

    Args:
        masks: (B, 1, H, W) float in [0, 1] or uint8 binary.

    Returns:
        boxes: (B, 4) float32, on the same device. Empty masks default to the
        full image extent so downstream code never sees a degenerate prompt.
    """
    B, _, H, W = masks.shape
    boxes = torch.zeros(B, 4, dtype=torch.float32, device=masks.device)
    bin_m = masks > 0.5
    for i in range(B):
        ys, xs = torch.where(bin_m[i, 0])
        if ys.numel() == 0:
            boxes[i] = torch.tensor([0, 0, W - 1, H - 1], device=masks.device)
            continue
        boxes[i, 0] = xs.min().float()
        boxes[i, 1] = ys.min().float()
        boxes[i, 2] = xs.max().float()
        boxes[i, 3] = ys.max().float()
    return boxes


def jitter_boxes(boxes: torch.Tensor, H: int, W: int,
                 jitter_frac: float = 0.1,
                 generator: Optional[torch.Generator] = None,
                 min_side_frac: Optional[float] = None) -> torch.Tensor:
    """Apply ±jitter_frac * box_dim noise per coord, clamped to image bounds.

    Same fractional convention as the LoRA foundation-model box jitter,
    but the noise is drawn from an explicit ``generator`` so the jitter is fully
    reproducible (seed=42). The noise is
    sampled on CPU through ``generator`` and moved to the box device, so the
    result is identical regardless of the CUDA RNG state. When ``generator`` is
    None the draw falls back to the global RNG.

    ``min_side_frac`` floors each side at that fraction of the original side,
    expanding about the perturbed centre. At large jitter fractions an unlucky
    draw can otherwise collapse a box toward zero area, which would measure
    degenerate-prompt behaviour rather than prompt imprecision. Used by the 50%
    jitter level.
    """
    b = boxes.clone()
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    # (B, 4) uniform noise in [-jitter_frac, +jitter_frac], seeded on CPU.
    noise = torch.rand(b.shape[0], 4, generator=generator) * 2 * jitter_frac - jitter_frac
    noise = noise.to(device=b.device, dtype=b.dtype)
    b[:, 0] += w * noise[:, 0]
    b[:, 1] += h * noise[:, 1]
    b[:, 2] += w * noise[:, 2]
    b[:, 3] += h * noise[:, 3]
    b[:, 0::2].clamp_(0, W - 1)
    b[:, 1::2].clamp_(0, H - 1)
    # Ensure x2>=x1 and y2>=y1 after jitter
    x1 = torch.minimum(b[:, 0], b[:, 2])
    x2 = torch.maximum(b[:, 0], b[:, 2])
    y1 = torch.minimum(b[:, 1], b[:, 3])
    y2 = torch.maximum(b[:, 1], b[:, 3])
    if min_side_frac is not None:
        min_w = w * min_side_frac
        min_h = h * min_side_frac
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        nw = torch.maximum(x2 - x1, min_w)
        nh = torch.maximum(y2 - y1, min_h)
        x1 = (cx - nw / 2).clamp(0, W - 1)
        x2 = (cx + nw / 2).clamp(0, W - 1)
        y1 = (cy - nh / 2).clamp(0, H - 1)
        y2 = (cy + nh / 2).clamp(0, H - 1)
    b[:, 0], b[:, 1], b[:, 2], b[:, 3] = x1, y1, x2, y2
    return b


def shift_boxes(boxes: torch.Tensor, H: int, W: int,
                shift_frac: float = 0.5) -> torch.Tensor:
    """Translate each box centre by ``shift_frac`` * box_dim (still axis-aligned).

    Direction is deterministic (toward the image centre, so the shifted box does
    not simply fall off-frame for peripheral nodules). The box keeps its size and
    typically retains partial overlap with the nodule — the "shifted" stress
    level between tight and adversarial.
    """
    b = boxes.clone()
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    # Sign: push toward image centre to stay in-frame.
    sx = torch.where(cx < W / 2, 1.0, -1.0)
    sy = torch.where(cy < H / 2, 1.0, -1.0)
    dx = sx * shift_frac * w
    dy = sy * shift_frac * h
    b[:, 0] += dx; b[:, 2] += dx
    b[:, 1] += dy; b[:, 3] += dy
    b[:, 0::2].clamp_(0, W - 1)
    b[:, 1::2].clamp_(0, H - 1)
    return b


def adversarial_boxes(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Place a same-size box flush in the image corner farthest from the nodule.

    Used to probe box-collapse: a healthy model that reads the image tracks the
    true nodule and ignores this wrong box; a collapsed model that only paints
    the box follows it and craters toward the adversarial floor (~0).
    Corner placement guarantees minimal
    overlap regardless of nodule position — reflecting across the centre would
    fail for centred nodules. Deterministic.
    """
    b = boxes.clone()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    # Farthest horizontal/vertical edge from the centroid.
    x1 = torch.where(cx <= (W - 1) / 2, (W - 1) - w, torch.zeros_like(w))
    y1 = torch.where(cy <= (H - 1) / 2, (H - 1) - h, torch.zeros_like(h))
    b[:, 0] = x1; b[:, 2] = x1 + w
    b[:, 1] = y1; b[:, 3] = y1 + h
    b[:, 0::2].clamp_(0, W - 1)
    b[:, 1::2].clamp_(0, H - 1)
    return b


# Canonical eval-box modes for the box-sensitivity panel (B7) and the box-fill
# floor (A5). "tight" is the headline regime used everywhere else.
EVAL_BOX_MODES = ("tight", "jitter10", "jitter25", "jitter50", "shifted", "adversarial")


def build_eval_boxes(masks: torch.Tensor, mode: str, H: int, W: int,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Derive (B, 4) boxes from masks under a named eval-box mode.

    All modes start from the tight oracle box (``masks_to_boxes``). The jitter
    modes are stochastic and take a seeded ``generator`` (reproducibility);
    shifted/adversarial are deterministic.
    """
    boxes = masks_to_boxes(masks)
    if mode == "tight":
        return boxes
    if mode == "jitter10":
        return jitter_boxes(boxes, H=H, W=W, jitter_frac=0.10, generator=generator)
    if mode == "jitter25":
        return jitter_boxes(boxes, H=H, W=W, jitter_frac=0.25, generator=generator)
    if mode == "jitter50":
        return jitter_boxes(boxes, H=H, W=W, jitter_frac=0.50, generator=generator,
                            min_side_frac=0.10)
    if mode == "shifted":
        return shift_boxes(boxes, H=H, W=W, shift_frac=0.5)
    if mode == "adversarial":
        return adversarial_boxes(boxes, H=H, W=W)
    raise ValueError(f"Unknown eval box mode '{mode}'; expected one of {EVAL_BOX_MODES}")


def count_fullimage_fallbacks(masks: torch.Tensor) -> int:
    """Number of masks in the batch with no foreground (→ full-image box).

    ``masks_to_boxes`` returns the full image extent for empty masks; in the
    box-conditioned setting that silently turns the 4th channel all-ones. We
    count these so the smoke test / training can report the fallback rate
    instead of letting it pass unnoticed.
    """
    bin_m = masks > 0.5
    per_image = bin_m.flatten(1).any(dim=1)   # (B,) True if any foreground
    return int((~per_image).sum().item())


def boxes_to_channel(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Rasterise bboxes into a binary channel (B, 1, H, W) float32 in {0, 1}."""
    B = boxes.shape[0]
    chan = torch.zeros(B, 1, H, W, dtype=torch.float32, device=boxes.device)
    b = boxes.long()
    for i in range(B):
        x1, y1, x2, y2 = b[i].tolist()
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 < x1 or y2 < y1:
            continue
        chan[i, 0, y1:y2 + 1, x1:x2 + 1] = 1.0
    return chan


def build_box_input(images: torch.Tensor, masks: torch.Tensor,
                    training: bool, jitter_frac: float = 0.1,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Assemble a 4-channel input: RGB image + binary box channel.

    Args:
        images: (B, 3, H, W) z-score normalised tensors from the dataloader.
        masks:  (B, 1, H, W) float.
        training: if True, jitter the boxes before rasterising.
        generator: seeded RNG for the train-time jitter (reproducibility, D5).
            Ignored when ``training`` is False (eval uses the tight box).

    Returns:
        (B, 4, H, W) float32 — the 4th channel is in {0, 1} (NOT z-score
        normalised; the model learns to interpret it).
    """
    H, W = images.shape[-2:]
    boxes = masks_to_boxes(masks)
    if training:
        boxes = jitter_boxes(boxes, H=H, W=W, jitter_frac=jitter_frac,
                             generator=generator)
    chan = boxes_to_channel(boxes, H=H, W=W)
    return torch.cat([images, chan], dim=1)
