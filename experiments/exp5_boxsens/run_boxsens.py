#!/usr/bin/env python3
"""Box-Prompt Sensitivity sweep for foundation models (Experiment 5, paper Section 5.5).

For one (regime, model, dataset) it evaluates the model under five box-prompt
perturbation modes and, for each mode, also reports the box-fill NULL floor
(filling the perturbed box itself as the prediction). This quantifies prompt
robustness: how DSC degrades as the oracle box is jittered / shifted / made
adversarial.

Regimes:
  * zeroshot : the published foundation checkpoint, no fine-tuning (mirrors Exp 2).
  * lora     : the few-shot LoRA f=0.5 adapted checkpoint (mirrors Exp 3 eval).

The CNN side (box-conditioned U-Net / TransUNet) is already covered by
box_eval.py (*_box_sensitivity.csv); nnU-Net box-cond sensitivity is handled
separately once its training finishes.

Perturbation modes (identical to the CNN box_eval, via build_eval_boxes):
  tight, jitter10 (+/-10%), jitter25 (+/-25%), shifted (50% shift), adversarial.
The perturbed box for a given (dataset, mode, image) is made deterministic and
seeded per image index, so every model sees the SAME boxes (fair comparison).

Usage:
  python experiments/exp5_boxsens/run_boxsens.py --regime zeroshot --model sam2 --dataset ddti
  python experiments/exp5_boxsens/run_boxsens.py --regime lora --model sam2 --dataset ddti --fraction 0.05
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from thyroidbench.models.foundation_base import ZeroShotDataset             # noqa: E402
from thyroidbench.metrics import compute_metrics, compute_hd95             # noqa: E402
from thyroidbench.boxes import build_eval_boxes, EVAL_BOX_MODES            # noqa: E402

DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")
SEED_BASE = 1000   # per-image jitter seed offset (mirrors Exp 2 convention)
IMG_SIZE = 512
EXP_DIR = PROJECT_ROOT / "experiments" / "exp5_boxsens"
OUT_DIR = EXP_DIR / "results"


# ----------------------------------------------------------------------------
# Model loading (replicated from the Exp 2 / Exp 3 entry points)
# ----------------------------------------------------------------------------
def load_zeroshot_predictor(model_name: str, device: str):
    if model_name == "sam2":
        from thyroidbench.models.sam2_wrapper import SAM2Predictor
        return SAM2Predictor(device=device)
    if model_name == "medsam":
        from thyroidbench.models.medsam_wrapper import MedSAMPredictor
        return MedSAMPredictor(device=device)
    if model_name == "medsam2":
        from thyroidbench.models.medsam2_wrapper import MedSAM2Predictor
        return MedSAM2Predictor(device=device)
    if model_name == "sam":
        from thyroidbench.models.sam_wrapper import SAMPredictor
        return SAMPredictor(device=device)
    if model_name == "sam3":
        from thyroidbench.models.sam3_wrapper import SAM3Predictor
        return SAM3Predictor(device=device)
    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------------------------------------------------------
# Box-fill NULL floor (per perturbed box) — model-agnostic reference
# ----------------------------------------------------------------------------
def floor_metrics(box: np.ndarray, mask: np.ndarray) -> dict:
    """Metrics of the filled (perturbed) box taken as the prediction."""
    H, W = mask.shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    pred = np.zeros_like(mask, dtype=np.float32)
    if x2 >= x1 and y2 >= y1:
        pred[y1:y2 + 1, x1:x2 + 1] = 1.0
    m = (mask > 0).astype(np.float32)
    inter = float((pred * m).sum())
    ps, ms = float(pred.sum()), float(m.sum())
    dice = 2 * inter / (ps + ms + 1e-10)
    iou = inter / (ps + ms - inter + 1e-10)
    prec = inter / (ps + 1e-10)
    rec = inter / (ms + 1e-10)
    hd95 = compute_hd95(pred.astype(np.uint8), m.astype(np.uint8))
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "hd95": hd95}


def perturbed_box(mask_np: np.ndarray, mode: str, idx: int) -> np.ndarray | None:
    """Deterministic per-image perturbed box (same across models)."""
    m = (mask_np > 0).astype(np.float32)
    if m.sum() == 0:
        return None
    mask_t = torch.from_numpy(m)[None, None]            # (1,1,H,W)
    H, W = m.shape
    gen = torch.Generator()
    gen.manual_seed(SEED_BASE + idx)                    # per-image -> identical across models
    box_t = build_eval_boxes(mask_t, mode, H=H, W=W, generator=gen)
    return box_t[0].cpu().numpy()


def metric_row(logit_hw: np.ndarray, mask_np: np.ndarray) -> dict:
    """DSC/IoU/Prec/Rec/HD95 from a (H,W) float logit and binary mask."""
    pred_t = torch.from_numpy(logit_hw).unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m = compute_metrics(pred_t, target_t)
    pred_bin = (torch.sigmoid(pred_t) >= 0.5).numpy()[0, 0].astype(np.uint8)
    hd95 = compute_hd95(pred_bin, (mask_np > 0).astype(np.uint8))
    return {"dice": m["dice"], "iou": m["iou"], "precision": m["precision"],
            "recall": m["recall"], "hd95": hd95}


# ----------------------------------------------------------------------------
# Regime: zero-shot
# ----------------------------------------------------------------------------
def run_zeroshot(model_name: str, dataset: str, device: str, modes) -> list[dict]:
    predictor = load_zeroshot_predictor(model_name, device)
    data = ZeroShotDataset(dataset_name=dataset,
                           split_csv=PROJECT_ROOT / "data/splits" / f"{dataset}_test.csv",
                           data_root=PROJECT_ROOT / "data/processed", img_size=IMG_SIZE)
    acc = {mode: {"d": [], "i": [], "p": [], "r": [], "h": [],
                  "fd": [], "fi": [], "fh": []} for mode in modes}
    for idx in range(len(data)):
        image, mask, _fn, _pid = data[idx]
        if (mask > 0).sum() == 0:
            continue
        for mode in modes:
            box = perturbed_box(mask, mode, idx)
            if box is None:
                continue
            try:
                logit = predictor.predict(image, box)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            mm = metric_row(logit, mask)
            fm = floor_metrics(box, mask)
            a = acc[mode]
            a["d"].append(mm["dice"]); a["i"].append(mm["iou"]); a["p"].append(mm["precision"])
            a["r"].append(mm["recall"]); a["h"].append(mm["hd95"])
            a["fd"].append(fm["dice"]); a["fi"].append(fm["iou"]); a["fh"].append(fm["hd95"])
    return _aggregate(acc, "zeroshot", model_name, dataset, modes)


# ----------------------------------------------------------------------------
# Regime: LoRA f=0.5
# ----------------------------------------------------------------------------
@torch.no_grad()  # eval only; without this the LoRA wrapper's requires_grad=True
                  # params build an autograd graph and retain activations, OOM-ing
                  # the large SAM3 multiplex backbone on big targets (thyroidxl).
def run_lora(model_name: str, dataset: str, device: str, modes, fraction: float) -> list[dict]:
    from thyroidbench.data import get_all_dataloaders
    from thyroidbench.data import DATASET_STATS
    from thyroidbench.lora import build_lora_wrapper

    ds_stats = DATASET_STATS[dataset]
    wrapper = build_lora_wrapper(model_name=model_name, device=device,
                                 ds_mean=ds_stats["mean"], ds_std=ds_stats["std"])
    frac_str = f"f{fraction:g}"
    if fraction == 1.0:
        # The full-supervision (f=100%) adapters are trained by Experiment 1,
        # not Experiment 3; same LoRA wrapper and state-dict format, different
        # run directory.
        ckpt = (PROJECT_ROOT / "experiments/exp1_fullsupervised/foundation_lora/results"
                / f"fullsup_{model_name}_{dataset}"
                / f"fullsup_{model_name}_{dataset}_best.pt")
    else:
        ckpt = (PROJECT_ROOT / "experiments/exp3_fewshot_lora/results"
                / f"fewshot_{model_name}_{dataset}_{frac_str}"
                / f"fewshot_{model_name}_{dataset}_{frac_str}_best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    wrapper.load_state_dict(state["model_state"])
    del state
    torch.cuda.empty_cache()
    wrapper.model.eval()

    loaders = get_all_dataloaders(dataset_name=dataset,
                                  data_root=PROJECT_ROOT / "data",
                                  regime="few_shot", few_shot_fraction=fraction)
    test_loader = loaders["test"] if isinstance(loaders, dict) else loaders[-1]

    acc = {mode: {"d": [], "i": [], "p": [], "r": [], "h": [],
                  "fd": [], "fi": [], "fh": []} for mode in modes}
    idx = 0
    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        B = images.shape[0]
        for i in range(B):
            img_i = images[i:i + 1]
            mask_np = (masks[i, 0].cpu().numpy() > 0).astype(np.uint8)
            if mask_np.sum() == 0:
                idx += 1
                continue
            for mode in modes:
                box = perturbed_box(mask_np, mode, idx)
                if box is None:
                    continue
                box_t = torch.tensor(box, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = wrapper(img_i, box_t)
                logit = logits.float()[0, 0].detach().cpu().numpy()
                mm = metric_row(logit, mask_np)
                fm = floor_metrics(box, mask_np)
                a = acc[mode]
                a["d"].append(mm["dice"]); a["i"].append(mm["iou"]); a["p"].append(mm["precision"])
                a["r"].append(mm["recall"]); a["h"].append(mm["hd95"])
                a["fd"].append(fm["dice"]); a["fi"].append(fm["iou"]); a["fh"].append(fm["hd95"])
            idx += 1
    return _aggregate(acc, "lora", model_name, dataset, modes)


# ----------------------------------------------------------------------------
def _mean(xs):
    xs = [v for v in xs if np.isfinite(v)]
    return float(np.mean(xs)) if xs else float("nan")


def _aggregate(acc, regime, model_name, dataset, modes) -> list[dict]:
    rows = []
    for mode in modes:
        a = acc[mode]
        rows.append(dict(
            regime=regime, model=model_name, dataset=dataset, mode=mode,
            n=len(a["d"]),
            dice=round(_mean(a["d"]), 6), iou=round(_mean(a["i"]), 6),
            precision=round(_mean(a["p"]), 6), recall=round(_mean(a["r"]), 6),
            hd95=round(_mean(a["h"]), 4),
            floor_dice=round(_mean(a["fd"]), 6), floor_iou=round(_mean(a["fi"]), 6),
            floor_hd95=round(_mean(a["fh"]), 4),
            delta_dice_over_floor=round(_mean(a["d"]) - _mean(a["fd"]), 6),
        ))
    return rows


# Canonical display/merge order for the perturbation modes.
_MODE_ORDER = ["tight", "jitter10", "jitter25", "jitter50", "shifted", "adversarial"]


def _merge_with_existing(out_csv: Path, new_rows: list[dict], key: str) -> list[dict]:
    """Merge freshly-computed rows into an existing CSV instead of clobbering it.

    Rows whose ``key`` (mode) matches a freshly-computed mode are replaced; all
    other previously-computed modes are preserved. This lets us add a single mode
    (e.g. jitter50) with ``--modes jitter50`` without re-running the rest. The
    result is sorted by the canonical mode order.
    """
    new_modes = {r[key] for r in new_rows}
    cols = list(new_rows[0].keys())
    merged: list[dict] = []
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            for old in csv.DictReader(f):
                if old.get(key) not in new_modes:
                    merged.append({c: old.get(c, "") for c in cols})
    merged.extend({c: r[c] for c in cols} for r in new_rows)
    merged.sort(key=lambda r: _MODE_ORDER.index(r[key])
                if r.get(key) in _MODE_ORDER else 99)
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description="Box-prompt sensitivity sweep (foundation).")
    p.add_argument("--regime", required=True, choices=["zeroshot", "lora"])
    p.add_argument("--model", required=True,
                   choices=["sam", "sam2", "sam3", "medsam", "medsam2"])
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--fraction", type=float, default=0.05,
                   help="LoRA fraction (lora regime): 0.05 or 1.0 as reported.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--modes", nargs="+", default=list(EVAL_BOX_MODES))
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.regime == "zeroshot":
        rows = run_zeroshot(args.model, args.dataset, args.device, args.modes)
        tag = f"{args.regime}_{args.model}_{args.dataset}"
    else:
        rows = run_lora(args.model, args.dataset, args.device, args.modes, args.fraction)
        tag = f"{args.regime}_{args.model}_{args.dataset}_f{args.fraction:g}"

    out_csv = OUT_DIR / f"{tag}_boxsens.csv"
    rows = _merge_with_existing(out_csv, rows, key="mode")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {out_csv}")
    for r in rows:
        print(f"  {r['mode']:11s} DSC={float(r['dice']):.3f} "
              f"(floor {float(r['floor_dice']):.3f}, "
              f"Δ={float(r['delta_dice_over_floor']):+.3f}) "
              f"HD95={float(r['hd95']):.1f} n={r['n']}")


if __name__ == "__main__":
    main()
