"""SAM ViT-H zero-shot predictor. Checkpoint: sam_vit_h_4b8939.pth (~2.6 GB)."""

from pathlib import Path

import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor

from .foundation_base import FoundationPredictor

_ROOT = Path(__file__).resolve().parents[2]
_CKPT = _ROOT / "pretrained_models" / "sam" / "sam_vit_h_4b8939.pth"


class SAMPredictor(FoundationPredictor):
    """Zero-shot wrapper around segment_anything.SamPredictor for SAM ViT-H."""

    def __init__(self, device: str = "cuda"):
        if not _CKPT.exists():
            raise FileNotFoundError(f"SAM ViT-H checkpoint not found at {_CKPT}")
        self.device = device
        sam = sam_model_registry["vit_h"](checkpoint=str(_CKPT))
        sam.to(device).eval()
        self.predictor = SamPredictor(sam)

    def predict(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Run zero-shot box-prompted segmentation; return float logits at (H, W)."""
        H, W = image.shape[:2]
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None],  # (1, 4) xyxy
            multimask_output=False,
            return_logits=True,
        )
        # masks: (1, H, W) float32 logit values at original image resolution
        out = masks[0].astype(np.float32)
        if out.shape != (H, W):
            t = torch.from_numpy(out).unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            out = t[0, 0].numpy()
        return out

    def close(self) -> None:
        del self.predictor
        torch.cuda.empty_cache()
