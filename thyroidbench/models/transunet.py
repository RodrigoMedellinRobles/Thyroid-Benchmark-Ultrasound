#!/usr/bin/env python3
"""
TransUNet R50-ViT-B_16 for binary segmentation of thyroid nodules.
Encoder: ResNet50 (ImageNet pretrained) + ViT-B bottleneck.
Decoder: 5-stage upsampling with skip connections from ResNet layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torchvision.models import ResNet50_Weights, resnet50

TRANSUNET_CONFIG = {
    "architecture": "transunet_r50_vitb16",
    "input_size": 512,
    "input_channels": 3,                # image-only (Approach A)
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


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBnRelu(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    """
    TransUNet R50-ViT-B_16. Input: (B,3,512,512). Output: (B,1,512,512) raw logits.

    Encoder stages (with 512x512 input, stride=2 at stem+maxpool → /4 after layer0):
      stem+pool:  /4  → 128x128, C=64  (not a skip)
      layer1:     /4  → 128x128, C=256  (skip1)
      layer2:     /8  →  64x64,  C=512  (skip2)
      layer3:    /16  →  32x32,  C=1024 (skip3)
      layer4:    /32  →  16x16,  C=2048 → ViT bottleneck
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        if in_channels not in (3, 4):
            raise ValueError(f"TransUNet only supports in_channels in {{3, 4}}, got {in_channels}")
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # NOTE: the backbone (and every sub-module below) is built at 3 input
        # channels so the RNG stream — and therefore the random init of
        # pos_embed / ViT blocks / decoder / head — is IDENTICAL to the
        # image-only twin. The 4th (box) channel is grafted onto the stem conv
        # at the very end of __init__ (see _expand_stem_to_4ch), mirroring the
        # U-Net pattern. Building the 4-ch conv inline here would shift the RNG
        # stream and break the twin-at-init identity.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 128×128, C=256
        self.layer2 = backbone.layer2   # 64×64,   C=512
        self.layer3 = backbone.layer3   # 32×32,   C=1024
        self.layer4 = backbone.layer4   # 16×16,   C=2048

        # Project 2048 → 768 (ViT-B hidden dim)
        self.proj = nn.Conv2d(2048, 768, kernel_size=1, bias=False)

        # Learnable positional embedding for 16×16=256 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 768))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ViT-B: 12 layers, 12 heads, mlp_ratio=4, dim=768
        self.blocks = nn.ModuleList([
            Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                  proj_drop=0.1, attn_drop=0.1)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(768)

        # Decoder: each stage doubles spatial resolution
        self.dec1 = DecoderStage(768, 1024, 512)    # 16→32,  +skip3
        self.dec2 = DecoderStage(512, 512,  256)    # 32→64,  +skip2
        self.dec3 = DecoderStage(256, 256,  128)    # 64→128, +skip1
        self.dec4 = DecoderStage(128, 0,    64)     # 128→256, no skip
        self.dec5 = DecoderStage(64,  0,    16)     # 256→512, no skip

        self.head = nn.Conv2d(16, 1, kernel_size=1)

        # Graft the box channel onto the stem conv LAST, after all random init
        # is done, so the box-conditioned model is numerically identical to its
        # image-only twin at step 0 (C3/S4). The 4th filter is zero-initialised
        # → contributes nothing at init; any later divergence is a learned use
        # of the box prior, which is what makes the paired box-effect test valid.
        if in_channels == 4:
            self._expand_stem_to_4ch()

    def _expand_stem_to_4ch(self) -> None:
        old = self.stem[0]   # conv1, weight: (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            in_channels=4, out_channels=old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old.weight
            new_conv.weight[:, 3:4, :, :] = 0.0  # zero-init box channel (twin-at-init)
            if old.bias is not None:
                new_conv.bias.copy_(old.bias)
        self.stem[0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        s1 = self.layer1(x)   # (B,256,128,128)
        s2 = self.layer2(s1)  # (B,512,64,64)
        s3 = self.layer3(s2)  # (B,1024,32,32)
        x  = self.layer4(s3)  # (B,2048,16,16)

        x = self.proj(x)      # (B,768,16,16)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) + self.pos_embed  # (B,256,768)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B,768,16,16)

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        x = self.dec4(x)
        x = self.dec5(x)
        return self.head(x)


def build_transunet(pretrained: bool = True, device: str = "cuda",
                    in_channels: int = 3) -> TransUNet:
    return TransUNet(pretrained=pretrained, in_channels=in_channels).to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_transunet(pretrained=False, device=device)
    x = torch.randn(2, 3, 512, 512, device=device)
    out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}  ← expected (2, 1, 512, 512)")
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")
