"""SAM2 zero-shot predictor. Checkpoint: sam2.1_hiera_large.pt (857 MB)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .foundation_base import FoundationPredictor

_ROOT = Path(__file__).resolve().parents[2]
_CKPT   = _ROOT / "pretrained_models" / "sam2" / "sam2.1_hiera_large.pt"
_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class SAM2Predictor(FoundationPredictor):

    def __init__(self, device: str = "cuda"):
        self.device = device
        model = build_sam2(_CONFIG, str(_CKPT), device=device, mode="eval")
        self.predictor = SAM2ImagePredictor(model)

    def predict(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        self.predictor.set_image(image)
        # low_res_masks: (N, 256, 256) float32 logits
        _, _, low_res_masks = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None],         # (1, 4)
            multimask_output=False,
        )
        logit_t = torch.from_numpy(low_res_masks[[0]]).unsqueeze(0)  # (1,1,256,256)
        logit_up = F.interpolate(logit_t, size=(H, W), mode="bilinear", align_corners=False)
        return logit_up[0, 0].numpy()

    def close(self) -> None:
        del self.predictor
        torch.cuda.empty_cache()
