"""ThyroidBench — a thin, importable API for the promptable thyroid-nodule
segmentation benchmark.

This top-level package re-exports the small, notebook-facing public surface::

    from thyroidbench import (
        ThyroidDataset,   # dataset + augmentation
        get_dataloader,   # dataloader factory
        build_model,      # model factory (unet/transunet/sam*/medsam*/sam3*)
        dice_score,       # torchmetrics Dice used by compute_metrics
        compute_metrics,  # Dice/IoU/precision/recall
        DiceBCELoss,      # segmentation loss (+ get_loss factory)
        get_loss,
        box_from_mask,    # boxes.masks_to_boxes  (derive boxes from masks)
        jitter_box,       # boxes.jitter_boxes    (seeded box jitter)
        build_box_input,  # 4-channel image+box tensor
        Trainer,          # training loop for U-Net / TransUNet
        apply_lora,       # lora.build_lora_wrapper (LoRA few-shot wrapper)
    )

Importing this package pulls in the torch-backed submodules eagerly, so it
requires the scientific stack (torch, albumentations, torchmetrics, ...) to be
installed. The heavy foundation-model backends (sam2, sam3, peft, ...) are only
imported lazily when the corresponding model is actually built.
"""

from .data import ThyroidDataset, get_dataloader, get_all_dataloaders
from .metrics import compute_metrics, dice_score
from .losses import DiceLoss, DiceBCELoss, DiceFocalLoss, get_loss
from .boxes import (
    build_box_input,
    masks_to_boxes as box_from_mask,
    jitter_boxes as jitter_box,
)
from .trainer import Trainer
from .lora import build_lora_wrapper as apply_lora
from .models import build_model

__all__ = [
    # data
    "ThyroidDataset",
    "get_dataloader",
    "get_all_dataloaders",
    # metrics
    "compute_metrics",
    "dice_score",
    # losses
    "DiceLoss",
    "DiceBCELoss",
    "DiceFocalLoss",
    "get_loss",
    # boxes
    "build_box_input",
    "box_from_mask",
    "jitter_box",
    # training
    "Trainer",
    # lora / models
    "apply_lora",
    "build_model",
]
