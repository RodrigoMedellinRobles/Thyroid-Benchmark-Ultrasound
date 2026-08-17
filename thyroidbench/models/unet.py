"""
U-Net with ResNet50 encoder (ImageNet pretrained) for thyroid nodule segmentation.

Default: 3-channel RGB input (image-only / fully-automatic regime).
Optional: 4-channel input (RGB + binary bbox prompt) is still supported via
in_channels=4 for ablations, but the main paper protocol uses image-only
traditional baselines vs prompted SAM-family foundation models.
"""

import logging
from typing import Dict, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

UNET_CONFIG = {
    "architecture": "unet_resnet50",
    "input_size": 512,
    "input_channels": 3,            # image-only (Approach A)
    "box_conditioned": False,
    "output_channels": 1,
    "optimizer": "Adam",
    "lr_search": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "batch_size": 16,
    "loss": "dice_bce_50_50",
    "max_epochs": 100,
    "early_stop_patience": 15,
    "lr_scheduler": "CosineAnnealingLR",
    "weight_decay": 1e-4,
    "seed": 42,
}


def _expand_conv1_to_4ch(model: smp.Unet) -> None:
    """Expand encoder.conv1 from 3 → 4 in_channels, preserving pretraining.

    The first ResNet conv layer has shape (64, 3, 7, 7). We replace it with
    (64, 4, 7, 7) where channels 0..2 are copied verbatim and channel 3 (the
    box-prompt channel) is ZERO-initialised. Zero-init means the box channel
    contributes nothing at step 0, so the box-conditioned model is numerically
    identical to its image-only twin at initialisation; any later divergence is
    a learned use of the box prior. This identity is what makes the within-image
    paired box-effect test valid.
    """
    encoder = model.encoder
    old_conv = encoder.conv1
    if old_conv.in_channels == 4:
        return
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = 0.0  # zero-init box channel (twin-at-init)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    encoder.conv1 = new_conv


class UNet(nn.Module):
    """U-Net with ResNet50 encoder. Returns raw logits (no sigmoid).

    Input: (B, 4, H, W)  — RGB + binary bbox prompt channel.
    Output: (B, 1, H, W) — raw logits.
    """

    def __init__(self, encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        # Build the SMP UNet with 3 in_channels first so the ImageNet weights
        # load cleanly, then expand conv1 to 4 channels.
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        if in_channels == 4:
            _expand_conv1_to_4ch(self.model)
        elif in_channels != 3:
            raise ValueError(f"UNet only supports in_channels in {{3, 4}}, got {in_channels}")
        logger.info("UNet ResNet50 — pretrained=%s — in_channels=%d — params=%s",
                    encoder_weights, in_channels, f"{self.n_parameters:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict:
        return UNET_CONFIG.copy()


def build_unet(pretrained: bool = True, device: str = "cuda",
               in_channels: int = 3) -> UNet:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    model = UNet(encoder_weights="imagenet" if pretrained else None,
                 in_channels=in_channels)
    return model.to(device)
