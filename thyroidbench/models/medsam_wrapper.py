"""
MedSAM zero-shot predictor. Checkpoint: medsam_vit_b.pth (358 MB).
Uses SAM v1 with custom per-image min-max normalization and 1024×1024 resize.
Ref: https://github.com/bowang-lab/medsam
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry

from .foundation_base import FoundationPredictor

_ROOT = Path(__file__).resolve().parents[2]
_CKPT  = _ROOT / "pretrained_models" / "medsam" / "medsam_vit_b.pth"
_SIZE  = 1024  # MedSAM encoder input size


class MedSAMPredictor(FoundationPredictor):

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = sam_model_registry["vit_b"](checkpoint=str(_CKPT))
        self.model.eval().to(device)

    def _preprocess_image(self, image: np.ndarray):
        """Resize to 1024×1024, per-image min-max normalize to [0,1]. Returns (1,3,1024,1024)."""
        img = cv2.resize(image, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        lo, hi = img.min(), img.max()
        img = (img - lo) / max(hi - lo, 1e-8)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,H,W)
        return t

    @torch.no_grad()
    def predict(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        img_t = self._preprocess_image(image)

        # Scale box to 1024 space
        box_1024 = box / np.array([W, H, W, H], dtype=np.float32) * _SIZE
        box_torch = torch.as_tensor(box_1024[None, None], dtype=torch.float32, device=self.device)
        # shape: (1, 1, 4) → prompt_encoder expects (B, 1, 4) for boxes

        img_embed = self.model.image_encoder(img_t)  # (1, 256, 64, 64)

        sparse_emb, dense_emb = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        # low_res_logits: (1, 1, 256, 256) — upscale to original HxW
        logit_up = F.interpolate(low_res_logits, size=(H, W), mode="bilinear", align_corners=False)
        return logit_up[0, 0].cpu().numpy()

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
