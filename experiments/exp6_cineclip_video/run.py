#!/usr/bin/env python3
"""
Experiment 6: framewise vs video-propagation segmentation on Stanford AIMI cine-clips.

Usage:
    python experiments/exp6_cineclip_video/run.py --model sam2    --mode framewise
    python experiments/exp6_cineclip_video/run.py --model medsam2 --mode video
    python experiments/exp6_cineclip_video/run.py --model sam2    --mode framewise --smoke_test
    python experiments/exp6_cineclip_video/run.py --model sam3    --mode video --fraction 0.05

Models:   sam2 | medsam2 | sam3  (all expose a memory-driven video predictor)
Modes:    framewise — independent box prompt per frame (matched to the still-image
                      experiments, which prompt every image)
          video     — frame-0 box prompt -> memory bank -> propagate_in_video
Fraction: omit for the zero-shot native checkpoint, or pass 0.05 / 1.0 to fold in
          the matching LoRA adapter, giving the three supervision levels of the
          reported label-efficiency ladder.

Outputs (results/{model}_{mode}[_f{fraction}]/):
    per_frame_metrics.csv
    per_sequence_metrics.csv
    summary.json
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

# ── Project root must be first so `from thyroidbench.*` works ───────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
# MedSAM2 fork is inserted AFTER parse_args, only when model == medsam2.

from thyroidbench.models.foundation_base import ZeroShotDataset, get_gt_box, jitter_box
from thyroidbench.metrics import compute_metrics, compute_hd95
from thyroidbench.metrics import temporal_consistency

_BASE_SEED = 1000  # per-frame RNG: seed = _BASE_SEED + global_frame_idx

# Checkpoints / configs
_CKPT_SAM2     = _ROOT / "pretrained_models" / "sam2" / "sam2.1_hiera_large.pt"
_CFG_SAM2      = "configs/sam2.1/sam2.1_hiera_l.yaml"
_CKPT_MEDSAM2  = _ROOT / "pretrained_models" / "medsam2" / "MedSAM2_latest.pt"
_CFG_MEDSAM2   = "configs/sam2.1_hiera_t512"
# SAM3 paths are resolved inside thyroidbench.models.sam3_video_wrapper.


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 6: framewise versus video propagation")
    p.add_argument("--model",      required=True, choices=["sam2", "medsam2", "sam3"])
    p.add_argument("--mode",       required=True, choices=["framewise", "video"])
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--img_size",   type=int, default=512)
    p.add_argument("--data_root",  default="data/processed")
    p.add_argument("--split_dir",  default="data/splits")
    p.add_argument("--video_dir",  default="data/cineclip_videos")
    p.add_argument("--output_dir", default="experiments/exp6_cineclip_video/results")
    p.add_argument("--smoke_test", action="store_true",
                   help="Process only the first patient (~84 frames) then exit")
    p.add_argument("--fraction", type=float, default=None,
                   help="LoRA adapter fraction to inject. None=zero-shot native "
                        "weights (default). 0.05=Experiment 3 f0.05 adapter, 1.0=Experiment 1 "
                        "fullsup adapter. Only the weights change; protocol is "
                        "identical to zero-shot.")
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────

def load_image_predictor(model_name: str, device: str):
    """Framewise predictor with a SAM2ImagePredictor-shaped surface.

    SAM2 / MedSAM2 use the upstream ``SAM2ImagePredictor``; SAM3 uses an
    adapter (see ``thyroidbench.models.sam3_video_wrapper``) that wraps the existing
    1008×1008 :class:`SAM3Predictor` in the same set_image / predict API so
    :func:`predict_framewise` stays model-agnostic. MedSAM2 fork path is
    already on sys.path when this is called.
    """
    if model_name == "sam3":
        from thyroidbench.models.sam3_video_wrapper import build_sam3_framewise_predictor
        return build_sam3_framewise_predictor(device=device)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if model_name == "sam2":
        model = build_sam2(_CFG_SAM2, str(_CKPT_SAM2), device=device, mode="eval")
        return SAM2ImagePredictor(model)
    # medsam2
    model = build_sam2(_CFG_MEDSAM2, str(_CKPT_MEDSAM2), device=device, mode="eval")
    return SAM2ImagePredictor(model, apply_postprocessing=False)


def load_video_predictor(model_name: str, device: str):
    """Video predictor with a SAM2VideoPredictor-shaped surface.

    SAM2 / MedSAM2 go through :func:`build_sam2_video_predictor`; SAM3 uses an
    adapter around :class:`Sam3TrackerPredictor` (attached to the joint
    detector + tracker model so the ``sam3.pt`` checkpoint's tracker weights
    are loaded). The adapter mirrors init_state / reset_state /
    add_new_points_or_box / propagate_in_video so :func:`run_video` stays
    model-agnostic.
    """
    if model_name == "sam3":
        from thyroidbench.models.sam3_video_wrapper import build_sam3_video_predictor
        return build_sam3_video_predictor(device=device)

    from sam2.build_sam import build_sam2_video_predictor

    if model_name == "sam2":
        return build_sam2_video_predictor(_CFG_SAM2, str(_CKPT_SAM2), device=device)
    # medsam2
    return build_sam2_video_predictor(_CFG_MEDSAM2, str(_CKPT_MEDSAM2), device=device)


# ── LoRA adapter injection (few-shot / fullsup weights) ────────────────────
#
# Adapter checkpoints are the SAME artifacts trained by Experiment 3 (few-shot) and
# Experiment 1 (full-supervised foundation) via ``FoundationLoRAWrapper`` (thyroidbench/lora.py).
# In that wrapper the whole model sits under ``self.model``, so the saved
# ``model_state`` keys are prefixed ``model.``. LoRA lives ONLY in the image
# encoder / ViT trunk:
#   SAM2 / MedSAM-2 : model.image_encoder.base_model.model.trunk.<...>.lora_{A,B}
#   SAM3            : model.backbone.vision_backbone.trunk.base_model.model.<...>.lora_{A,B}
#
# We re-wrap the SAME submodule the wrapper wrapped (get_peft_model with the
# identical _LORA_CONFIG), load the matching sub-state-dict (strict=False so the
# frozen non-LoRA base weights that are ALSO present just overwrite themselves),
# assert every LoRA key was consumed, then merge_and_unload() to fold the LoRA
# deltas into the base weights for clean, adapter-free inference. A silently
# wrong load (missing LoRA keys) raises here rather than producing plausible
# garbage downstream.

_ADAPTER_FEWSHOT = _ROOT / "experiments" / "exp3_fewshot_lora" / "results"
_ADAPTER_FULLSUP = _ROOT / "experiments" / "exp1_fullsupervised" / "foundation_lora" / "results"


def _adapter_ckpt_path(model_name: str, fraction: float) -> Path:
    """Resolve the adapter checkpoint for (model, fraction) on Stanford AIMI.

    fraction == 1.0  -> Experiment 1 full-supervised checkpoint.
    else             -> Experiment 3 few-shot f{fraction:g} checkpoint.
    """
    if fraction == 1.0:
        run = f"fullsup_{model_name}_stanford_aimi"
        return _ADAPTER_FULLSUP / run / f"{run}_best.pt"
    run = f"fewshot_{model_name}_stanford_aimi_f{fraction:g}"
    return _ADAPTER_FEWSHOT / run / f"{run}_best.pt"


def _inject_lora_submodule(submodule, state: dict, prefix: str, device: str):
    """Re-wrap ``submodule`` with LoRA, load matching keys, merge, return module.

    Args:
        submodule: the exact nn.Module the adapter was fit on (image_encoder or
                   the SAM3 ViT trunk).
        state:     full ``model_state`` dict from the adapter checkpoint.
        prefix:    key prefix to strip so the remainder aligns with a fresh
                   ``get_peft_model(submodule, _LORA_CONFIG)`` state dict.
    Returns:
        The merged (adapter-free) nn.Module ready for inference.
    """
    from peft import get_peft_model
    from thyroidbench.lora import _LORA_CONFIG

    sub_sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    n_lora_ckpt = sum(1 for k in sub_sd if "lora_" in k)
    if n_lora_ckpt == 0:
        raise RuntimeError(
            f"No LoRA keys found under prefix '{prefix}' in adapter checkpoint "
            f"(found {len(sub_sd)} keys total). Injection would be a no-op."
        )
    peft_mod = get_peft_model(submodule, _LORA_CONFIG)
    missing, unexpected = peft_mod.load_state_dict(sub_sd, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        raise RuntimeError(
            f"LoRA injection broken: {len(lora_missing)} LoRA keys missing after "
            f"load (prefix '{prefix}'). First few: {lora_missing[:3]}"
        )
    lora_unexpected = [k for k in unexpected if "lora_" in k]
    if lora_unexpected:
        raise RuntimeError(
            f"LoRA injection broken: {len(lora_unexpected)} LoRA keys in "
            f"checkpoint had no target module. First few: {lora_unexpected[:3]}"
        )
    print(f"[LoRA] loaded {n_lora_ckpt} LoRA tensors under '{prefix}' "
          f"(missing={len(missing)}, unexpected={len(unexpected)}, non-LoRA base "
          f"overwrites OK) -> merging")
    merged = peft_mod.merge_and_unload()
    return merged.to(device)


def inject_adapter(predictor, model_name: str, mode: str,
                   fraction: float, device: str):
    """Fold the (model, fraction) adapter into ``predictor`` in place.

    Handles both framewise and video predictor surfaces for all three models.
    Returns the (possibly same) predictor object.
    """
    ckpt_path = _adapter_ckpt_path(model_name, fraction)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {ckpt_path}")
    print(f"[LoRA] injecting adapter model={model_name} mode={mode} "
          f"fraction={fraction:g}\n       from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)["model_state"]

    if model_name in ("sam2", "medsam2"):
        prefix = "model.image_encoder."
        if mode == "framewise":
            # SAM2ImagePredictor: encoder at predictor.model.image_encoder
            predictor.model.image_encoder = _inject_lora_submodule(
                predictor.model.image_encoder, state, prefix, device)
        else:
            # SAM2VideoPredictor is itself the SAM2Base model: .image_encoder
            predictor.image_encoder = _inject_lora_submodule(
                predictor.image_encoder, state, prefix, device)
        return predictor

    if model_name == "sam3":
        prefix = "model.backbone.vision_backbone.trunk."
        if mode == "framewise":
            # SAM3FramewiseAdapter -> SAM3Predictor at ._inner ; trunk under
            # ._inner.model.backbone.vision_backbone.trunk (build_sam3_image_model).
            inner = predictor._inner.model
            inner.backbone.vision_backbone.trunk = _inject_lora_submodule(
                inner.backbone.vision_backbone.trunk, state, prefix, device)
        else:
            # SAM3VideoAdapter drives ._tracker whose feature backbone is the
            # detector's backbone (hot-swapped in __init__). LoRA was fit on the
            # image model's backbone.vision_backbone.trunk; the video detector
            # exposes the SAME module path. Inject into the live tracker backbone.
            trunk = predictor._tracker.backbone.vision_backbone.trunk
            predictor._tracker.backbone.vision_backbone.trunk = _inject_lora_submodule(
                trunk, state, prefix, device)
        return predictor

    raise ValueError(f"Unknown model for adapter injection: {model_name}")


# ── GT mask helpers ────────────────────────────────────────────────────────

def load_mask_for_frame(mask_dir: Path, patient_id: str, frame_num: int,
                        img_size: int) -> np.ndarray | None:
    """
    Load binary GT mask for one frame.
    Filename pattern: {patient_id}_frame_{frame_num:04d}.png
    patient_id ends with '_' (e.g. '202_'), so the full filename becomes
    '202__frame_0001.png' — matching the Stanford AIMI convention.
    """
    fn  = f"{patient_id}_frame_{frame_num:04d}.png"
    pth = mask_dir / fn
    if not pth.exists():
        return None
    with Image.open(pth) as img:
        m = (np.array(img.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    if m.shape[0] != img_size or m.shape[1] != img_size:
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return m


# ── CSV helpers ────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── Prediction from image predictor → logit + binary mask ─────────────────

def predict_framewise(image_predictor, image: np.ndarray, box_j: np.ndarray,
                      H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logit_HxW_float32, pred_bin_HxW_uint8). low_res_masks is numpy."""
    image_predictor.set_image(image)
    _, _, low_res_masks = image_predictor.predict(
        point_coords=None, point_labels=None, box=box_j[None], multimask_output=False,
    )
    # low_res_masks is numpy from SAM2ImagePredictor — no .cpu().numpy()
    logit_t  = torch.from_numpy(low_res_masks[[0]]).unsqueeze(0)  # (1,1,128/256,128/256)
    logit_up = F.interpolate(logit_t, size=(H, W), mode="bilinear", align_corners=False)
    logit_np = logit_up[0, 0].numpy()
    pred_bin = (logit_np >= 0.0).astype(np.uint8)
    return logit_np, pred_bin


# ── Metrics helpers ────────────────────────────────────────────────────────

# HD95 imputation policy (matches paper §sec:methods:metrics).
# Non-finite HD95 (empty predicted mask or empty GT) is imputed to the image
# diagonal sqrt(2)*512 ≈ 724.08 px before any aggregation. Single source of
# truth across point estimates, bootstrap CIs, and paired Wilcoxon tests.
HD95_IMG_DIAG_PX = float(np.sqrt(2) * 512)


def frame_metrics(logit_np: np.ndarray, pred_bin: np.ndarray,
                  gt_mask: np.ndarray) -> dict:
    pred_t   = torch.from_numpy(logit_np).unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m    = compute_metrics(pred_t, target_t)
    hd95 = compute_hd95(pred_bin, gt_mask)
    return {
        "dice": m["dice"],
        "iou":  m["iou"],
        "hd95": float(hd95) if np.isfinite(hd95) else HD95_IMG_DIAG_PX,
        "is_empty_pred": bool(pred_bin.sum() == 0),
    }


def sequence_summary(frame_rows: list[dict], pred_masks: list[np.ndarray]) -> dict:
    dice_v  = [r["dice"] for r in frame_rows if r["dice"] is not None]
    iou_v   = [r["iou"]  for r in frame_rows if r["iou"]  is not None]
    hd95_v  = [r["hd95"] for r in frame_rows if r["hd95"] is not None]
    n_empty = sum(1 for r in frame_rows if r.get("is_empty_pred", False))
    tc = temporal_consistency(pred_masks) if len(pred_masks) >= 2 else float("nan")
    return {
        "dice_mean":     float(np.mean(dice_v))  if dice_v  else float("nan"),
        "iou_mean":      float(np.mean(iou_v))   if iou_v   else float("nan"),
        "hd95_mean":     float(np.mean(hd95_v))  if hd95_v  else float("nan"),
        "tc":            tc,
        "n_frames":      len(frame_rows),
        "n_empty_pred":  n_empty,
        "empty_pred_pct": round(100.0 * n_empty / max(len(frame_rows), 1), 2),
    }


# ── FRAMEWISE mode ─────────────────────────────────────────────────────────

def run_framewise(args, image_predictor, dataset: ZeroShotDataset,
                  output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frame_rows: list[dict] = []
    patient_data: dict[str, dict] = defaultdict(lambda: {"rows": [], "masks": []})

    # Smoke test: collect patient list to limit to 1
    smoke_patient: str | None = None
    if args.smoke_test:
        smoke_patient = dataset.patient_ids[0]
        print(f"[smoke_test] Running 1 patient: {smoke_patient}")

    total = len(dataset)
    # NOTE: peak VRAM is reset once in main() BEFORE model load so the reported
    # value reflects the full deployment footprint (model + activations), not
    # only inference activations.
    t0 = time.time()
    n_ok = 0

    for idx in range(total):
        image, gt_mask, filename, patient_id = dataset[idx]
        if args.smoke_test and patient_id != smoke_patient:
            continue

        H, W = image.shape[:2]
        frame_num = int(filename.split("__frame_")[1].split(".")[0])

        box = get_gt_box(gt_mask)
        if box is None:
            all_frame_rows.append(dict(filename=filename, patient_id=patient_id,
                                       frame_num=frame_num, dice=None, iou=None,
                                       hd95=None, skipped=True))
            continue

        rng   = np.random.default_rng(_BASE_SEED + idx)
        box_j = jitter_box(box, rng, 0.1, H, W)

        try:
            logit_np, pred_bin = predict_framewise(image_predictor, image, box_j, H, W)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[OOM] {filename}")
            all_frame_rows.append(dict(filename=filename, patient_id=patient_id,
                                       frame_num=frame_num, dice=None, iou=None,
                                       hd95=None, skipped=True))
            continue
        except Exception:
            traceback.print_exc()
            all_frame_rows.append(dict(filename=filename, patient_id=patient_id,
                                       frame_num=frame_num, dice=None, iou=None,
                                       hd95=None, skipped=True))
            continue

        m = frame_metrics(logit_np, pred_bin, gt_mask)
        row = dict(filename=filename, patient_id=patient_id, frame_num=frame_num,
                   skipped=False, **m)
        all_frame_rows.append(row)
        patient_data[patient_id]["rows"].append(m)
        patient_data[patient_id]["masks"].append(pred_bin)
        n_ok += 1

        wandb.log({"frame_dice": m["dice"], "frame_iou": m["iou"],
                   "frame_hd95": m["hd95"] or 0.0})

        if n_ok % 50 == 0:
            print(f"[{n_ok}] patient={patient_id} frame={frame_num} "
                  f"dice={m['dice']:.4f}")

    elapsed   = time.time() - t0
    fps       = n_ok / elapsed if elapsed > 0 else 0.0
    vram_gb   = torch.cuda.max_memory_allocated() / 1e9

    # Per-sequence
    seq_rows: list[dict] = []
    all_tc: list[float] = []
    for pid, data in patient_data.items():
        ss = sequence_summary(data["rows"], data["masks"])
        seq_rows.append({"patient_id": pid, **ss})
        if not np.isnan(ss["tc"]):
            all_tc.append(ss["tc"])

    # Per-frame CSV (extended with is_empty_pred flag)
    write_csv(output_dir / "per_frame_metrics.csv", all_frame_rows,
              ["filename", "patient_id", "frame_num", "dice", "iou", "hd95",
               "is_empty_pred", "skipped"])
    write_csv(output_dir / "per_sequence_metrics.csv", seq_rows,
              ["patient_id", "dice_mean", "iou_mean", "hd95_mean", "tc",
               "n_frames", "n_empty_pred", "empty_pred_pct"])

    # Global summary
    valid_frames = [r for r in all_frame_rows if not r["skipped"]]
    dice_all  = [r["dice"] for r in valid_frames]
    iou_all   = [r["iou"]  for r in valid_frames]
    hd95_all  = [r["hd95"] for r in valid_frames if r["hd95"] is not None]
    n_empty_global = sum(1 for r in valid_frames if r.get("is_empty_pred", False))
    summary = dict(
        model=args.model, mode=args.mode,
        n_patients=len(patient_data), n_frames=n_ok,
        n_skipped=sum(1 for r in all_frame_rows if r["skipped"]),
        n_empty_pred=n_empty_global,
        empty_pred_pct=round(100.0 * n_empty_global / max(n_ok, 1), 2),
        dice_mean=float(np.mean(dice_all)), dice_std=float(np.std(dice_all)),
        iou_mean=float(np.mean(iou_all)),   iou_std=float(np.std(iou_all)),
        hd95_mean=float(np.mean(hd95_all)) if hd95_all else float("nan"),
        hd95_std=float(np.std(hd95_all))   if hd95_all else float("nan"),
        tc_mean=float(np.mean(all_tc)) if all_tc else float("nan"),
        tc_std=float(np.std(all_tc))  if all_tc else float("nan"),
        fps=fps, vram_gb=vram_gb,
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    wandb.run.summary.update(summary)
    _print_summary(summary)
    return summary


# ── VIDEO mode ─────────────────────────────────────────────────────────────

def run_video(args, video_predictor, patient_ids: list[str],
              mask_dir: Path, output_dir: Path,
              dataset_idx_lookup: dict[tuple[str, int], int]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_base = Path(args.video_dir)

    if args.smoke_test:
        patient_ids = patient_ids[:1]
        print(f"[smoke_test] Running 1 patient: {patient_ids[0]}")

    all_frame_rows: list[dict] = []
    seq_rows: list[dict] = []
    all_tc: list[float] = []
    n_ok   = 0
    n_skip = 0

    # peak VRAM reset is performed in main() before model load.
    t0 = time.time()

    for pid in patient_ids:
        patient_jpg_dir = video_base / pid
        if not patient_jpg_dir.exists():
            print(f"[WARN] video dir missing: {patient_jpg_dir} — skip patient")
            n_skip += 1
            continue

        # SAM2's init_state requires purely numeric filenames (00000.jpg, ...).
        # prepare_videos.py writes them this way and a sidecar JSON maps the
        # numeric video index back to the original Stanford AIMI frame number.
        jpgs = sorted(patient_jpg_dir.glob("[0-9]*.jpg"),
                      key=lambda p: int(p.stem))
        if len(jpgs) < 2:
            print(f"[WARN] {pid}: only {len(jpgs)} frames — skip")
            n_skip += 1
            continue

        index_map_path = patient_jpg_dir / "frame_index_map.json"
        if not index_map_path.is_file():
            print(f"[WARN] {pid}: missing frame_index_map.json — skip")
            n_skip += 1
            continue
        with open(index_map_path) as f:
            index_map: dict[str, int] = json.load(f)

        # Frame-0 GT mask for box prompt — map numeric video idx → original frame
        frame0_num = int(index_map[jpgs[0].name])
        mask0 = load_mask_for_frame(mask_dir, pid, frame0_num, args.img_size)
        if mask0 is None:
            print(f"[WARN] {pid}: no GT mask for frame {frame0_num} — skip")
            n_skip += 1
            continue
        box0 = get_gt_box(mask0)
        if box0 is None:
            print(f"[WARN] {pid}: empty mask at frame {frame0_num} — skip")
            n_skip += 1
            continue

        H, W = mask0.shape[:2]
        # Align the frame-0 jitter seed with what framewise mode uses for the
        # same (patient_id, frame0_num) — fair "same prompt, different inference"
        # comparison. Framewise uses _BASE_SEED + idx where idx is the dataset
        # position; here we reproduce that idx for (pid, frame0_num).
        try:
            frame0_idx = dataset_idx_lookup[(pid, frame0_num)]
        except KeyError as e:
            raise RuntimeError(
                f"frame-0 jitter alignment broken: ({pid}, {frame0_num}) not in "
                f"dataset_idx_lookup. Cannot produce a comparable box to framewise "
                f"mode. Investigate prepare_videos.py vs stanford_aimi_test.csv "
                f"consistency."
            ) from e
        rng0  = np.random.default_rng(_BASE_SEED + frame0_idx)
        box0_j = jitter_box(box0, rng0, 0.1, H, W)

        inference_state = video_predictor.init_state(video_path=str(patient_jpg_dir))
        video_predictor.reset_state(inference_state)

        try:
            video_predictor.add_new_points_or_box(
                inference_state, frame_idx=0, obj_id=1, box=box0_j[None]
            )
        except Exception as e:
            print(f"[WARN] {pid}: add_new_points_or_box failed: {e}")
            video_predictor.reset_state(inference_state)
            n_skip += 1
            continue

        patient_frame_rows: list[dict] = []
        patient_masks: list[np.ndarray] = []

        for out_frame_idx, _obj_ids, out_mask_logits in \
                video_predictor.propagate_in_video(inference_state):
            # Video API: out_mask_logits shape (num_objects, 1, H, W)
            logit_t  = out_mask_logits[0]               # (1, H, W) float tensor
            # `>= 0.0` matches framewise convention (run.py:142) and
            # compute_metrics' sigmoid >= 0.5 threshold (mode-symmetric).
            pred_bin = (logit_t >= 0.0).cpu().numpy().squeeze().astype(np.uint8)

            frame_num = int(index_map[jpgs[out_frame_idx].name])
            gt_mask   = load_mask_for_frame(mask_dir, pid, frame_num, args.img_size)
            if gt_mask is None:
                patient_frame_rows.append(dict(filename=jpgs[out_frame_idx].name,
                                               patient_id=pid, frame_num=frame_num,
                                               dice=None, iou=None, hd95=None,
                                               skipped=True))
                continue

            # compute_metrics needs logits (1,1,H,W): one unsqueeze on (1,H,W)
            pred_logit_t = logit_t.unsqueeze(0).cpu().float()
            target_t     = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            m    = compute_metrics(pred_logit_t, target_t)
            hd95 = compute_hd95(pred_bin, gt_mask)
            # HD95 impute-aware (matches §methods:metrics policy)
            m_dict = dict(dice=m["dice"], iou=m["iou"],
                          hd95=float(hd95) if np.isfinite(hd95) else HD95_IMG_DIAG_PX,
                          is_empty_pred=bool(pred_bin.sum() == 0))

            patient_frame_rows.append(dict(filename=jpgs[out_frame_idx].name,
                                           patient_id=pid, frame_num=frame_num,
                                           skipped=False, **m_dict))
            patient_masks.append(pred_bin)
            n_ok += 1

            wandb.log({"frame_dice": m["dice"], "frame_iou": m["iou"],
                       "frame_hd95": m_dict["hd95"] or 0.0})

            if n_ok % 50 == 0:
                print(f"[{n_ok}] patient={pid} frame={frame_num} dice={m['dice']:.4f}")

        video_predictor.reset_state(inference_state)
        all_frame_rows.extend(patient_frame_rows)

        valid_rows = [r for r in patient_frame_rows if not r["skipped"]]
        if valid_rows and len(patient_masks) >= 2:
            ss = sequence_summary(
                [{"dice": r["dice"], "iou": r["iou"], "hd95": r["hd95"],
                  "is_empty_pred": r.get("is_empty_pred", False)} for r in valid_rows],
                patient_masks,
            )
            seq_rows.append({"patient_id": pid, **ss})
            if not np.isnan(ss["tc"]):
                all_tc.append(ss["tc"])

    elapsed = time.time() - t0
    fps     = n_ok / elapsed if elapsed > 0 else 0.0
    vram_gb = torch.cuda.max_memory_allocated() / 1e9

    write_csv(output_dir / "per_frame_metrics.csv", all_frame_rows,
              ["filename", "patient_id", "frame_num", "dice", "iou", "hd95",
               "is_empty_pred", "skipped"])
    write_csv(output_dir / "per_sequence_metrics.csv", seq_rows,
              ["patient_id", "dice_mean", "iou_mean", "hd95_mean", "tc",
               "n_frames", "n_empty_pred", "empty_pred_pct"])

    valid_frames = [r for r in all_frame_rows if not r["skipped"]]
    dice_all  = [r["dice"] for r in valid_frames]
    iou_all   = [r["iou"]  for r in valid_frames]
    hd95_all  = [r["hd95"] for r in valid_frames if r["hd95"] is not None]
    n_empty_global = sum(1 for r in valid_frames if r.get("is_empty_pred", False))
    summary = dict(
        model=args.model, mode=args.mode,
        n_patients=len(patient_ids) - n_skip, n_frames=n_ok,
        n_skipped=n_skip,
        n_empty_pred=n_empty_global,
        empty_pred_pct=round(100.0 * n_empty_global / max(n_ok, 1), 2),
        dice_mean=float(np.mean(dice_all)) if dice_all else float("nan"),
        dice_std=float(np.std(dice_all))   if dice_all else float("nan"),
        iou_mean=float(np.mean(iou_all))   if iou_all  else float("nan"),
        iou_std=float(np.std(iou_all))     if iou_all  else float("nan"),
        hd95_mean=float(np.mean(hd95_all)) if hd95_all else float("nan"),
        hd95_std=float(np.std(hd95_all))   if hd95_all else float("nan"),
        tc_mean=float(np.mean(all_tc)) if all_tc else float("nan"),
        tc_std=float(np.std(all_tc))   if all_tc else float("nan"),
        fps=fps, vram_gb=vram_gb,
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    wandb.run.summary.update(summary)
    _print_summary(summary)
    return summary


# ── Print ──────────────────────────────────────────────────────────────────

def _print_summary(s: dict) -> None:
    print(f"\n=== SUMMARY {s['model']} {s['mode']} ===")
    print(f"  DSC:        {s['dice_mean']:.4f} ± {s['dice_std']:.4f}")
    print(f"  IoU:        {s['iou_mean']:.4f} ± {s['iou_std']:.4f}")
    print(f"  HD95 (px):  {s['hd95_mean']:.2f} ± {s['hd95_std']:.2f}  (impute-aware)")
    print(f"  Jt (TC):    {s['tc_mean']:.4f} ± {s['tc_std']:.4f}")
    print(f"  FPS:        {s['fps']:.2f}   VRAM: {s['vram_gb']:.2f} GB")
    print(f"  n_frames={s['n_frames']}  patients={s['n_patients']}  skipped={s['n_skipped']}")
    print(f"  empty_pred: {s.get('n_empty_pred', 0)} ({s.get('empty_pred_pct', 0.0)}%)")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Reset peak VRAM BEFORE loading any model so summary.json reports the
    # full deployment footprint (encoder + decoder + activations).
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # MedSAM2 fork AFTER parse_args, BEFORE any sam2 import. SAM3 does not need
    # the MedSAM2 fork path: it pulls its weights and code from the installed
    # ``sam3`` package via thyroidbench.models.sam3_video_wrapper.
    if args.model == "medsam2":
        sys.path.insert(0, str(_ROOT / "pretrained_models" / "medsam2_fork"))

    # Adapted runs (fraction set) get a distinct tag/dir so they never overwrite
    # the zero-shot native-weight results.
    frac_tag = f"_f{args.fraction:g}" if args.fraction is not None else ""
    run_tag = f"{args.model}_{args.mode}{frac_tag}"
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    wb_tags = ["cineclip_video", args.model, args.mode, "stanford_aimi"]
    if args.fraction is not None:
        wb_tags.append(f"f{args.fraction:g}")
    wandb.init(
        project="thyroidbench",
        name=f"cineclip_{run_tag}",
        tags=wb_tags,
    )
    wandb.config.update(vars(args))

    split_csv = Path(args.split_dir) / "stanford_aimi_test.csv"
    data_root = Path(args.data_root)
    mask_dir  = data_root / "stanford_aimi" / "masks"

    dataset = ZeroShotDataset(
        dataset_name="stanford_aimi",
        split_csv=split_csv,
        data_root=data_root,
        img_size=args.img_size,
    )

    # Build (patient_id, frame_num) → dataset index lookup; used by video mode
    # to seed frame-0 jitter the SAME way framewise mode does at that position.
    # Read directly from the CSV-derived attributes (no per-frame I/O).
    dataset_idx_lookup: dict[tuple[str, int], int] = {}
    for i, (fn, pid) in enumerate(zip(dataset.filenames, dataset.patient_ids)):
        try:
            fnum = int(fn.split("__frame_")[1].split(".")[0])
            dataset_idx_lookup[(pid, fnum)] = i
        except (IndexError, ValueError):
            continue

    if args.mode == "framewise":
        predictor = load_image_predictor(args.model, args.device)
        if args.fraction is not None:
            predictor = inject_adapter(predictor, args.model, args.mode,
                                       args.fraction, args.device)
        summary   = run_framewise(args, predictor, dataset, output_dir)
        del predictor
    else:
        predictor  = load_video_predictor(args.model, args.device)
        if args.fraction is not None:
            predictor = inject_adapter(predictor, args.model, args.mode,
                                       args.fraction, args.device)
        patient_ids = sorted(set(dataset.patient_ids))
        summary    = run_video(args, predictor, patient_ids, mask_dir, output_dir,
                               dataset_idx_lookup)
        del predictor

    torch.cuda.empty_cache()
    wandb.finish()

    if args.smoke_test:
        print("\n[smoke_test] PASSED")


if __name__ == "__main__":
    main()
