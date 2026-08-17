"""SAM 3 vanilla zero-shot predictor (sam3.pt).

Replicates the SAM-1-style box-prompt path from
``Sam3Image.inst_interactive_predictor``.
Input expected at 1008x1008 with mean=0.5, std=0.5 normalization
per ``sam3_image_processor`` defaults.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .foundation_base import FoundationPredictor

_ROOT = Path(__file__).resolve().parents[2]
_CKPT = _ROOT / "pretrained_models" / "sam3_vanilla" / "sam3.pt"
_INPUT_SIZE = 1008
_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class SAM3Predictor(FoundationPredictor):
    """Zero-shot wrapper for SAM 3 vanilla using the SAM-1 box-prompt branch."""

    def __init__(self, device: str = "cuda"):
        if not _CKPT.exists():
            raise FileNotFoundError(f"SAM3 vanilla checkpoint not found at {_CKPT}")
        from sam3.model_builder import build_sam3_image_model  # type: ignore

        self.device = device
        self.model = build_sam3_image_model(
            device=str(device),
            eval_mode=True,
            checkpoint_path=str(_CKPT),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )
        self.model.eval()
        self._mean = _MEAN.to(device)
        self._std = _STD.to(device)

    @torch.inference_mode()
    def predict(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Run zero-shot box-prompted segmentation; return float logits at (H, W)."""
        H, W = image.shape[:2]

        # uint8 RGB (H, W, 3) → float tensor (1, 3, 1008, 1008) normalized to [-1, 1]
        img_t = torch.from_numpy(image).to(self.device, dtype=torch.float32)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, H, W) in [0, 1]
        img_t = F.interpolate(img_t, size=(_INPUT_SIZE, _INPUT_SIZE),
                              mode="bilinear", align_corners=False)
        img_t = (img_t - self._mean) / self._std

        # Scale box xyxy from original (H, W) coordinates to 1008x1008
        sx = _INPUT_SIZE / float(W)
        sy = _INPUT_SIZE / float(H)
        box_scaled = torch.tensor(
            [[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]],
            dtype=torch.float32, device=self.device,
        )  # (1, 4)
        boxes_b1 = box_scaled.unsqueeze(1)  # (B=1, 1, 4)

        # SAM3 box-prompt forward path (matches the LoRA sam3 branch).
        backbone_out = self.model.backbone.forward_image(img_t)
        sam2_backbone_out = backbone_out["sam2_backbone_out"]
        ipred = self.model.inst_interactive_predictor

        sam2_backbone_out["backbone_fpn"][0] = (
            ipred.model.sam_mask_decoder.conv_s0(sam2_backbone_out["backbone_fpn"][0])
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            ipred.model.sam_mask_decoder.conv_s1(sam2_backbone_out["backbone_fpn"][1])
        )
        _, vision_feats, _, _ = ipred.model._prepare_backbone_features(sam2_backbone_out)
        vision_feats[-1] = vision_feats[-1] + ipred.model.no_mem_embed

        B = 1
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], ipred._bb_feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        # Convert (B, 1, 4) xyxy → (B, 2, 2) corner-points + labels [2, 3]
        box_coords = boxes_b1.reshape(B, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device).repeat(B, 1)
        sparse_emb, dense_emb = ipred.model.sam_prompt_encoder(
            points=(box_coords, box_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, _, _, _ = ipred.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=ipred.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        # Upsample logits to original (H, W)
        logit_t = F.interpolate(low_res_masks, size=(H, W),
                                mode="bilinear", align_corners=False)
        return logit_t[0, 0].float().cpu().numpy()

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
