"""
MedSAM2 zero-shot predictor. Checkpoint: MedSAM2_latest.pt (149 MB).
Hiera-Tiny at 512px. Uses fork sam2 (bowang-lab/MedSAM2) via sys.path priority —
the fork fixes the 64×32 feature-size mismatch in the FB sam2 package at 512px.
Ref: https://github.com/bowang-lab/MedSAM2
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .foundation_base import FoundationPredictor

_ROOT         = Path(__file__).resolve().parents[2]
_CKPT         = _ROOT / "pretrained_models" / "medsam2" / "MedSAM2_latest.pt"
_FORK_ROOT    = _ROOT / "pretrained_models" / "medsam2_fork"
# Config name as Hydra sees it after fork registers its config module
_CONFIG       = "configs/sam2.1_hiera_t512"

# Prepend fork so its sam2/ shadows the FB package — must happen before any sam2 import
if str(_FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FORK_ROOT))


class MedSAM2Predictor(FoundationPredictor):

    def __init__(self, device: str = "cuda"):
        self.device = device
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(_CONFIG, str(_CKPT), device=device, mode="eval")
        self.predictor = SAM2ImagePredictor(model, apply_postprocessing=False)

    def predict(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        self.predictor.set_image(image)
        _, _, low_res_masks = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None],         # (1, 4)
            multimask_output=False,
        )
        logit_t  = torch.from_numpy(low_res_masks[[0]]).unsqueeze(0)  # (1,1,128,128) at 512px
        logit_up = F.interpolate(logit_t, size=(H, W), mode="bilinear", align_corners=False)
        return logit_up[0, 0].numpy()

    def close(self) -> None:
        del self.predictor
        torch.cuda.empty_cache()
