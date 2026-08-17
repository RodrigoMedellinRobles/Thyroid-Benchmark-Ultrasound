# Segmentation losses: Dice, Dice+BCE and Dice+Focal.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: sigmoid-activated probability map
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target.float())
        pred_prob = torch.sigmoid(pred_logits)
        dice = self.dice(pred_prob, target)
        return self.bce_weight * bce + self.dice_weight * dice


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        pred_prob = torch.sigmoid(pred_logits)
        dice = self.dice(pred_prob, target)
        return self.focal_weight * focal.mean() + self.dice_weight * dice


def get_loss(name: str) -> nn.Module:
    """Factory: 'dice' | 'dice_bce_50_50' | 'dice_focal_50_50'."""
    if name == "dice":
        return DiceLoss()
    elif name == "dice_bce_50_50":
        return DiceBCELoss(0.5, 0.5)
    elif name == "dice_focal_50_50":
        return DiceFocalLoss(0.5, 0.5)
    else:
        raise ValueError(f"Unknown loss '{name}'. Choices: dice, dice_bce_50_50, dice_focal_50_50")
