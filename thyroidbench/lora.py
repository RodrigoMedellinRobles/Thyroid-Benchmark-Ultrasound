"""LoRA wrapper for foundation models used for few-shot and full-supervision LoRA training."""

import functools
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _cp
from peft import LoraConfig, get_peft_model

# __file__ = <repo>/thyroidbench/lora.py → parents[1] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SAM2_CKPT         = _PROJECT_ROOT / "pretrained_models/sam2/sam2.1_hiera_large.pt"
_SAM2_CFG          = "configs/sam2.1/sam2.1_hiera_l.yaml"
_MEDSAM_CKPT       = _PROJECT_ROOT / "pretrained_models/medsam/medsam_vit_b.pth"
_SAM1_CKPT         = _PROJECT_ROOT / "pretrained_models/sam/sam_vit_h_4b8939.pth"
_MEDSAM2_CKPT      = _PROJECT_ROOT / "pretrained_models/medsam2/MedSAM2_latest.pt"
_MEDSAM2_CFG       = "configs/sam2.1_hiera_t512"
_MEDSAM2_FORK_ROOT = _PROJECT_ROOT / "pretrained_models/medsam2_fork"
_SAM3_CKPT         = _PROJECT_ROOT / "pretrained_models/sam3_vanilla/sam3.pt"

# SAM2 family uses ImageNet stats in [0,1] space
_SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_SAM2_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# SAM3 uses (0.5, 0.5, 0.5) per Sam3Processor (sam3_image_processor.py:26)
_SAM3_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SAM3_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

# One fixed LoRA config for every model and dataset: no per-model search, so no
# cell of the benchmark gets a tuning advantage over another.
_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["attn.qkv", "attn.proj"],
    bias="none",
)
# SAM3: PEFT is applied only to the ViT trunk submodule, so `sam_mask_decoder`
# (which lives outside the trunk) cannot be reached. Suffix-matching `attn.qkv` /
# `attn.proj` is therefore safe — it can only hit the 32 ViT blocks (= 64 modules).
_LORA_CONFIG_SAM3 = _LORA_CONFIG

_MEDSAM2_FORK_INITIALIZED = False


def _apply_gc_to_blocks_reentrant(peft_model: nn.Module) -> bool:
    """Block-level activation checkpointing with use_reentrant=True for sam3.

    use_reentrant=False is incompatible with bfloat16 autocast in the SAM3 ViT-L
    trunk (saved-vs-recomputed metadata mismatch during backward). use_reentrant=True
    preserves autocast context across the forward/backward boundary, but requires at
    least one input to the checkpoint to have requires_grad=True; the wrapper's forward
    forces this on `imgs` before calling backbone.forward_image.
    """
    try:
        base_encoder = peft_model.base_model.model
    except AttributeError:
        return False
    blocks = getattr(base_encoder, "blocks", None)
    if not blocks:
        return False

    for blk in blocks:
        orig = blk.forward

        def _make_ckpt(fwd):
            @functools.wraps(fwd)
            def _ckpt(*args, **kwargs):
                if kwargs:
                    return _cp.checkpoint(
                        functools.partial(fwd, **kwargs), *args, use_reentrant=True
                    )
                return _cp.checkpoint(fwd, *args, use_reentrant=True)
            return _ckpt

        blk.forward = _make_ckpt(orig)

    return True


def _apply_gc_to_blocks(peft_model: nn.Module) -> bool:
    """Gradient checkpointing via block-level patching for PEFT 0.19.1.

    Tries PEFT-native first (may work for SAM2/Hiera via newer PEFT versions),
    then falls back to patching each block.forward with torch.utils.checkpoint.
    The closure-capture bug is avoided by binding original_forward as a default arg.
    """
    try:
        peft_model.gradient_checkpointing_enable()
        return True
    except Exception:
        pass

    try:
        base_encoder = peft_model.base_model.model
    except AttributeError:
        return False

    blocks = getattr(base_encoder, "blocks", None)
    if not blocks:
        return False

    for blk in blocks:
        orig = blk.forward

        def _make_ckpt(fwd):
            @functools.wraps(fwd)
            def _ckpt(*args, **kwargs):
                if kwargs:
                    return _cp.checkpoint(
                        functools.partial(fwd, **kwargs), *args, use_reentrant=False
                    )
                return _cp.checkpoint(fwd, *args, use_reentrant=False)
            return _ckpt

        blk.forward = _make_ckpt(orig)

    return True


def _jitter_boxes(boxes: torch.Tensor, input_size: int, jitter_frac: float = 0.1) -> torch.Tensor:
    """±jitter_frac * box_dim random noise per coord, clipped to [0, input_size-1]."""
    b = boxes.clone()
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    b[:, 0] += w * (torch.rand_like(w) * 2 * jitter_frac - jitter_frac)
    b[:, 1] += h * (torch.rand_like(h) * 2 * jitter_frac - jitter_frac)
    b[:, 2] += w * (torch.rand_like(w) * 2 * jitter_frac - jitter_frac)
    b[:, 3] += h * (torch.rand_like(h) * 2 * jitter_frac - jitter_frac)
    b[:, 0::2].clamp_(0, input_size - 1)
    b[:, 1::2].clamp_(0, input_size - 1)
    return b


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FoundationLoRAWrapper(nn.Module):
    """Wraps a foundation model image encoder with LoRA for few-shot adaptation."""

    def __init__(self, model_name: str, device: str, ds_mean: List[float], ds_std: List[float]):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.register_buffer("ds_mean", torch.tensor(ds_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("ds_std",  torch.tensor(ds_std).view(1, 3, 1, 1), persistent=False)

        builders = {
            "sam2":    (1024, self._build_sam2),
            "medsam":  (1024, self._build_medsam),
            "sam":     (1024, self._build_sam),
            "medsam2": (512,  self._build_medsam2),
            "sam3":    (1008, self._build_sam3),
        }
        if model_name not in builders:
            raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(builders)}")

        self.input_size, build_fn = builders[model_name]
        self.model = build_fn()
        self._apply_lora()
        self._freeze_non_encoder()

        n_train = _count_trainable(self.model)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[LoRA/{model_name}] trainable={n_train:,} / total={n_total:,} ({100*n_train/n_total:.2f}%)")

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_sam2(self) -> nn.Module:
        from sam2.build_sam import build_sam2  # type: ignore
        return build_sam2(_SAM2_CFG, str(_SAM2_CKPT), device=self.device, mode="eval")

    def _build_medsam(self) -> nn.Module:
        from segment_anything import sam_model_registry  # type: ignore
        model = sam_model_registry["vit_b"](checkpoint=str(_MEDSAM_CKPT))
        return model.eval().to(self.device)

    def _build_sam(self) -> nn.Module:
        """SAM ViT-H (original, 636M params) — baseline for the SAM family progression."""
        from segment_anything import sam_model_registry  # type: ignore
        model = sam_model_registry["vit_h"](checkpoint=str(_SAM1_CKPT))
        return model.eval().to(self.device)

    def _build_medsam2(self) -> nn.Module:
        global _MEDSAM2_FORK_INITIALIZED
        if not _MEDSAM2_FORK_INITIALIZED:
            from hydra.core.global_hydra import GlobalHydra  # type: ignore
            GlobalHydra.instance().clear()
            sys.path.insert(0, str(_MEDSAM2_FORK_ROOT))
            _MEDSAM2_FORK_INITIALIZED = True
        from sam2.build_sam import build_sam2  # type: ignore  # noqa: F811
        return build_sam2(_MEDSAM2_CFG, str(_MEDSAM2_CKPT), device=self.device, mode="eval")

    def _build_sam3(self) -> nn.Module:
        """SAM 3 vanilla (sam3.pt) with SAM-1-style box-prompt path enabled.

        Loads `Sam3Image` with `enable_inst_interactivity=True` so that the
        sam_prompt_encoder + sam_mask_decoder branch (under
        `model.inst_interactive_predictor.model`) is built and populated.
        ViT trunk lives at `model.backbone.vision_backbone.trunk` (32 blocks,
        embed_dim=1024). Activation checkpointing is built-in (use_act_checkpoint=True).
        """
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        return build_sam3_image_model(
            device=str(self.device),
            eval_mode=True,
            checkpoint_path=str(_SAM3_CKPT),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------

    def _apply_lora(self) -> None:
        if self.model_name == "sam3":
            # SAM3: LoRA targets the ViT trunk inside model.backbone.vision_backbone.trunk.
            # The trunk's internal use_act_checkpoint=True (vitdet.py:838) uses
            # use_reentrant=False which is incompatible with bf16 autocast (saved vs
            # recomputed tensor metadata mismatch during backward). Disable it and apply
            # block-level checkpointing manually with use_reentrant=True, which preserves
            # autocast context. The forward forces imgs.requires_grad=True so reentrant
            # checkpoint has the grad-tracked input it needs.
            trunk = self.model.backbone.vision_backbone.trunk
            trunk.use_act_checkpoint = False
            trunk_peft = get_peft_model(trunk, _LORA_CONFIG_SAM3)
            self.model.backbone.vision_backbone.trunk = trunk_peft
            self.gc_applied = _apply_gc_to_blocks_reentrant(trunk_peft)
            status = "block-level use_reentrant=True" if self.gc_applied else "disabled"
            print(f"[LoRA/{self.model_name}] gradient_checkpointing={status}")
            return
        encoder_peft = get_peft_model(self.model.image_encoder, _LORA_CONFIG)
        self.gc_applied = _apply_gc_to_blocks(encoder_peft)
        status = "block-level" if self.gc_applied else "disabled"
        print(f"[LoRA/{self.model_name}] gradient_checkpointing={status}")
        self.model.image_encoder = encoder_peft

    def _freeze_non_encoder(self) -> None:
        if self.model_name == "sam3":
            for name, param in self.model.named_parameters():
                if "backbone.vision_backbone.trunk" not in name:
                    param.requires_grad_(False)
            return
        for name, param in self.model.named_parameters():
            if "image_encoder" not in name:
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_train_mode(self) -> None:
        self.model.train()
        if self.model_name == "sam3":
            # Only the ViT trunk (LoRA target) trains; prompt encoder + mask decoder
            # + DETR transformer + text encoder all stay in eval mode.
            self.model.backbone.vision_backbone.trunk.train()
            ipred = self.model.inst_interactive_predictor.model
            ipred.sam_prompt_encoder.eval()
            ipred.sam_mask_decoder.eval()
            self.model.transformer.eval()
            if getattr(self.model, "segmentation_head", None) is not None:
                self.model.segmentation_head.eval()
            self.model.geometry_encoder.eval()
            if getattr(self.model, "dot_prod_scoring", None) is not None:
                self.model.dot_prod_scoring.eval()
            return
        self.model.image_encoder.train()
        if self.model_name in ("sam2", "medsam2"):
            self.model.sam_prompt_encoder.eval()
            self.model.sam_mask_decoder.eval()
        else:  # medsam, sam
            self.model.prompt_encoder.eval()
            self.model.mask_decoder.eval()

    def forward(self, images_norm: torch.Tensor, boxes_512: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_norm: (B, 3, 512, 512) dataset-normalized tensors
            boxes_512:   (B, 4) GT boxes in 512px coords
        Returns:
            (B, 1, 512, 512) logits (pre-sigmoid), float32
        """
        B   = images_norm.shape[0]
        dev = images_norm.device

        # Denormalize to [0, 1]
        images_01 = (images_norm * self.ds_std.to(dev) + self.ds_mean.to(dev)).clamp(0.0, 1.0)

        # Scale boxes 512px → model input_size
        scale    = self.input_size / 512.0
        boxes_is = boxes_512 * scale
        # Gate on the wrapped model, not on this wrapper: evaluation paths call
        # wrapper.model.eval(), and nothing calls eval() on the wrapper itself,
        # so self.training would still be True at test time.
        if self.model.training:
            boxes_is = _jitter_boxes(boxes_is, self.input_size, jitter_frac=0.1)

        # Resize + model-specific normalization
        if self.model_name in ("sam2", "medsam2"):
            imgs = F.interpolate(images_01, size=(self.input_size, self.input_size),
                                 mode="bilinear", align_corners=False)
            imgs = (imgs - _SAM2_MEAN.to(dev)) / _SAM2_STD.to(dev)
        elif self.model_name == "sam3":
            imgs = F.interpolate(images_01, size=(self.input_size, self.input_size),
                                 mode="bilinear", align_corners=False)
            imgs = (imgs - _SAM3_MEAN.to(dev)) / _SAM3_STD.to(dev)
        else:  # medsam, sam — both expect MedSAM-style per-image min-max [0,1]
            imgs = F.interpolate(images_01, size=(1024, 1024),
                                 mode="bilinear", align_corners=False)
            lo = imgs.flatten(1).min(1).values.view(B, 1, 1, 1)
            hi = imgs.flatten(1).max(1).values.view(B, 1, 1, 1)
            imgs = (imgs - lo) / (hi - lo + 1e-8)

        boxes_b1 = boxes_is.unsqueeze(1)  # (B, 1, 4)

        if self.model_name == "sam3":
            # SAM-1 box-prompt path through Sam3Image.inst_interactive_predictor.
            # Mirrors Sam3Image.predict_inst (sam3_image.py:599) but differentiable
            # (no @torch.no_grad / @torch.inference_mode on the methods we call).
            # The ViT trunk uses torch.utils.checkpoint(use_reentrant=False) per block
            # (vitdet.py:838); that variant requires AT LEAST ONE input tensor to have
            # requires_grad=True or the output loses grad_fn. Force it here during training.
            if self.training:
                imgs = imgs.detach().requires_grad_(True)
            backbone_out = self.model.backbone.forward_image(imgs)
            sam2_backbone_out = backbone_out["sam2_backbone_out"]
            ipred = self.model.inst_interactive_predictor
            # Apply sam2_neck convs to the two highest-res FPN levels
            # (replicates sam3_image_processor.py:63-72)
            sam2_backbone_out["backbone_fpn"][0] = (
                ipred.model.sam_mask_decoder.conv_s0(sam2_backbone_out["backbone_fpn"][0])
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                ipred.model.sam_mask_decoder.conv_s1(sam2_backbone_out["backbone_fpn"][1])
            )
            # Prepare features (replicates sam1_task_predictor.py:102-115)
            _, vision_feats, _, _ = ipred.model._prepare_backbone_features(sam2_backbone_out)
            vision_feats[-1] = vision_feats[-1] + ipred.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(B, -1, *fs)
                for feat, fs in zip(vision_feats[::-1], ipred._bb_feat_sizes[::-1])
            ][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            # Convert (B, 1, 4) xyxy boxes → (B, 2, 2) corner-points format with labels [2,3]
            box_coords = boxes_b1.reshape(B, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=dev).repeat(B, 1)
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
        elif self.model_name in ("sam2", "medsam2"):
            backbone_out = self.model.forward_image(imgs)
            _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)
            pix_feat = vision_feats[-1].permute(1, 2, 0).view(
                B, self.model.hidden_dim, *feat_sizes[-1])
            high_res_feats = [
                f.permute(1, 2, 0).view(B, -1, *s)
                for f, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
            sparse_emb, dense_emb = self.model.sam_prompt_encoder(
                points=None, boxes=boxes_b1, masks=None)
            low_res_masks, _, _, _ = self.model.sam_mask_decoder(
                image_embeddings=pix_feat,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
        elif self.model_name in ("medsam", "sam"):
            # SAM v1 decoder designed for (1 img, N prompts) — encode batch, decode per-image
            img_embed = self.model.image_encoder(imgs)
            masks_list = []
            for i in range(B):
                sp, de = self.model.prompt_encoder(
                    points=None, boxes=boxes_b1[i:i+1], masks=None)
                m, _ = self.model.mask_decoder(
                    image_embeddings=img_embed[i:i+1],
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sp,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )
                masks_list.append(m)
            low_res_masks = torch.cat(masks_list, dim=0)
        else:  # samus
            img_embed = self.model.image_encoder(imgs)
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None, boxes=boxes_b1, masks=None)
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

        return F.interpolate(low_res_masks.float(), size=(512, 512),
                             mode="bilinear", align_corners=False)


def build_lora_wrapper(
    model_name: str,
    device: str,
    ds_mean: List[float],
    ds_std: List[float],
) -> FoundationLoRAWrapper:
    return FoundationLoRAWrapper(model_name, device, ds_mean, ds_std)
