#!/usr/bin/env python3
"""
Experiment 2: zero-shot foundation-model inference across datasets.
Usage: python experiments/exp2_zeroshot/run.py --model sam2 --dataset tn3k
"""

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import wandb

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thyroidbench.models.foundation_base import ZeroShotDataset, get_gt_box, jitter_box
from thyroidbench.metrics import compute_metrics, compute_hd95

_BASE_SEED = 1000  # per-image RNG: seed = _BASE_SEED + idx


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: zero-shot inference")
    p.add_argument("--model",   required=True,
                   choices=["sam", "sam2", "medsam", "medsam2", "sam3"])
    p.add_argument("--dataset", required=True,
                   choices=["thyroidxl", "tn3k", "ddti", "stanford_aimi"])
    p.add_argument("--device",    default="cuda")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--jitter_frac", type=float, default=0.0,
                   help="Box-prompt jitter fraction at eval. 0.0 = tight oracle box "
                        "The reported protocol evaluates with a tight box everywhere. "
                        "Set 0.1 to reproduce jittered zero-shot numbers instead.")
    p.add_argument("--img_size",  type=int, default=512)
    p.add_argument("--data_root", default="data/processed")
    p.add_argument("--split_dir", default="data/splits")
    p.add_argument("--output_dir", default="experiments/exp2_zeroshot/results")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate only the first N test images (0 = all). Use a small "
                        "value as a setup check: it exercises weights, preprocessing, "
                        "splits and metrics end to end in a couple of minutes. Results "
                        "from a limited run are written with a _limitN suffix so they "
                        "can never be mistaken for a full evaluation.")
    return p.parse_args()


def load_predictor(model_name: str, device: str):
    if model_name == "sam2":
        from thyroidbench.models.sam2_wrapper import SAM2Predictor
        return SAM2Predictor(device=device)
    elif model_name == "medsam":
        from thyroidbench.models.medsam_wrapper import MedSAMPredictor
        return MedSAMPredictor(device=device)
    elif model_name == "medsam2":
        from thyroidbench.models.medsam2_wrapper import MedSAM2Predictor
        return MedSAM2Predictor(device=device)
    elif model_name == "sam":
        from thyroidbench.models.sam_wrapper import SAMPredictor
        return SAMPredictor(device=device)
    elif model_name == "sam3":
        from thyroidbench.models.sam3_wrapper import SAM3Predictor
        return SAM3Predictor(device=device)
    raise ValueError(f"Unknown model: {model_name}")


def _empty_row(filename, patient_id, box=None):
    return dict(
        filename=filename, patient_id=patient_id,
        dice="", iou="", precision="", recall="", hd95="",
        box_x1="" if box is None else float(box[0]),
        box_y1="" if box is None else float(box[1]),
        box_x2="" if box is None else float(box[2]),
        box_y2="" if box is None else float(box[3]),
        skipped=True,
    )


def compute_summary(model, dataset, results):
    valid = [r for r in results if not r["skipped"] and r["dice"] != ""]
    n_total, n_skip = len(results), sum(1 for r in results if r["skipped"])
    if not valid:
        return dict(model=model, dataset=dataset, n_images=n_total, n_skipped=n_skip,
                    dice_mean=0, dice_std=0, iou_mean=0, iou_std=0,
                    precision_mean=0, precision_std=0, recall_mean=0, recall_std=0,
                    hd95_mean=0, hd95_std=0)
    dice  = np.array([r["dice"]      for r in valid], dtype=float)
    iou   = np.array([r["iou"]       for r in valid], dtype=float)
    prec  = np.array([r["precision"] for r in valid], dtype=float)
    rec   = np.array([r["recall"]    for r in valid], dtype=float)
    hd95s = np.array([r["hd95"] for r in valid if r["hd95"] != "" and np.isfinite(float(r["hd95"]))],
                     dtype=float)
    return dict(
        model=model, dataset=dataset, n_images=n_total, n_skipped=n_skip,
        dice_mean=float(np.mean(dice)),   dice_std=float(np.std(dice)),
        iou_mean=float(np.mean(iou)),     iou_std=float(np.std(iou)),
        precision_mean=float(np.mean(prec)), precision_std=float(np.std(prec)),
        recall_mean=float(np.mean(rec)),  recall_std=float(np.std(rec)),
        hd95_mean=float(np.mean(hd95s))  if len(hd95s) else 0.0,
        hd95_std=float(np.std(hd95s))    if len(hd95s) else 0.0,
    )


def save_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # A limited run is a setup check, not an evaluation: give it its own directory
    # so partial numbers can never be aggregated as if they were a full test split.
    _suffix = f"_limit{args.limit}" if args.limit else ""
    run_dir = Path(args.output_dir) / f"{args.model}_{args.dataset}{_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="thyroidbench",
        name=f"exp2_{args.model}_{args.dataset}",
        config=vars(args),
        tags=["exp2_zeroshot", args.model, args.dataset],
    )

    print(f"Loading {args.model} on {args.device}...")
    predictor = load_predictor(args.model, args.device)

    split_csv = Path(args.split_dir) / f"{args.dataset}_test.csv"
    dataset = ZeroShotDataset(
        dataset_name=args.dataset,
        split_csv=split_csv,
        data_root=Path(args.data_root),
        img_size=args.img_size,
    )
    n_images = len(dataset)
    if args.limit and args.limit < n_images:
        n_images = args.limit
        print(f"Dataset: {args.dataset} test = {len(dataset)} images "
              f"-> LIMITED to the first {n_images} (setup check, not a full run)")
    else:
        print(f"Dataset: {args.dataset} test = {n_images} images")

    results = []

    for idx in range(n_images):
        image, mask, filename, patient_id = dataset[idx]
        H, W = image.shape[:2]

        box = get_gt_box(mask)
        if box is None:
            results.append(_empty_row(filename, patient_id))
            print(f"[{idx+1}/{n_images}] SKIP {filename} — empty mask")
            continue

        if args.jitter_frac > 0.0:
            rng = np.random.default_rng(seed=_BASE_SEED + idx)
            box_prompt = jitter_box(box, rng, jitter_frac=args.jitter_frac, img_h=H, img_w=W)
        else:
            box_prompt = box  # tight oracle box (no jitter) — reported eval protocol

        try:
            logit = predictor.predict(image, box_prompt)  # (H, W) float32
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{idx+1}/{n_images}] OOM {filename}")
            results.append(_empty_row(filename, patient_id, box))
            continue
        except Exception as exc:
            print(f"[{idx+1}/{n_images}] ERROR {filename}: {exc}")
            traceback.print_exc()
            results.append(_empty_row(filename, patient_id, box))
            continue

        pred_t   = torch.from_numpy(logit).unsqueeze(0).unsqueeze(0)                   # (1,1,H,W)
        target_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0) # (1,1,H,W)
        m = compute_metrics(pred_t, target_t)

        pred_bin = (torch.sigmoid(pred_t) >= 0.5).numpy()[0, 0].astype(np.uint8)
        hd95 = compute_hd95(pred_bin, mask)

        row = dict(
            filename=filename, patient_id=patient_id,
            dice=round(m["dice"], 6),   iou=round(m["iou"], 6),
            precision=round(m["precision"], 6), recall=round(m["recall"], 6),
            hd95=round(hd95, 4) if np.isfinite(hd95) else "",
            box_x1=float(box[0]), box_y1=float(box[1]),
            box_x2=float(box[2]), box_y2=float(box[3]),
            skipped=False,
        )
        results.append(row)
        wandb.log({"dice": m["dice"], "iou": m["iou"],
                   "hd95": hd95 if np.isfinite(hd95) else None,
                   "precision": m["precision"], "recall": m["recall"]})

        if (idx + 1) % 50 == 0:
            valid_so_far = [r for r in results if not r["skipped"] and r["dice"] != ""]
            mean_d = np.mean([r["dice"] for r in valid_so_far]) if valid_so_far else 0.0
            print(f"[{idx+1}/{n_images}] running dice={mean_d:.4f}")

    # Save per-image
    per_img_path = str(run_dir / "per_image_results.csv")
    save_csv(results, per_img_path, fieldnames=[
        "filename", "patient_id", "dice", "iou", "precision", "recall",
        "hd95", "box_x1", "box_y1", "box_x2", "box_y2", "skipped",
    ])
    print(f"Saved: {per_img_path}")

    # Save summary
    summary = compute_summary(args.model, args.dataset, results)
    sum_path = str(run_dir / "summary_metrics.csv")
    save_csv([summary], sum_path, fieldnames=list(summary.keys()))
    print(f"Saved: {sum_path}")
    print(f"\n=== SUMMARY ===")
    print(f"  DSC:  {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"  IoU:  {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
    print(f"  HD95: {summary['hd95_mean']:.2f} ± {summary['hd95_std']:.2f}")
    print(f"  n={summary['n_images']}, skipped={summary['n_skipped']}")

    wandb.run.summary.update(summary)
    wandb.finish()

    predictor.close()


if __name__ == "__main__":
    main()
