#!/usr/bin/env python3
"""Evaluate nnU-Net predictions. Output CSV matches U-Net results format."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

DATASET_IDS = {
    "ddti": "001",
    "tn3k": "002",
    "thyroidxl": "003",
    "stanford_aimi": "004",
}


HD95_INF_CAP = float(np.sqrt(2) * 512)  # 724.08 px image diagonal; matches compute_stats.py policy.


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return HD95_INF_CAP
    pb = pred ^ binary_erosion(pred)
    gb = gt ^ binary_erosion(gt)
    dt_p = distance_transform_edt(~pb.astype(bool))
    dt_g = distance_transform_edt(~gb.astype(bool))
    dists = np.concatenate([dt_g[pb > 0], dt_p[gb > 0]])
    return float(np.percentile(dists, 95))


def metrics_for(pred_path: Path, gt_path: Path) -> dict:
    pred = (np.array(Image.open(pred_path).convert("L")) > 0).astype(np.float32)
    gt   = (np.array(Image.open(gt_path).convert("L"))   > 0).astype(np.float32)
    pred_b = pred.astype(np.uint8)
    gt_b   = gt.astype(np.uint8)
    inter = (pred * gt).sum()
    dice  = 2 * inter / (pred.sum() + gt.sum() + 1e-10)
    iou   = inter / (pred.sum() + gt.sum() - inter + 1e-10)
    tp = inter
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    return {"dice": float(dice), "iou": float(iou),
            "precision": float(prec), "recall": float(rec),
            "hd95": hd95(pred_b, gt_b)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=list(DATASET_IDS.keys()))
    args = p.parse_args()

    did = DATASET_IDS[args.dataset]
    data_root = Path("data")
    exp_dir   = Path("experiments/exp1_fullsupervised/traditional_imageonly/results/nnunet")

    # find dataset dir
    candidates = list((data_root / "nnunet_raw").glob(f"Dataset{did}_*"))
    if not candidates:
        raise FileNotFoundError(f"Dataset{did}_* not found in data/nnunet_raw/")
    ds_dir = candidates[0]
    ds_name = ds_dir.name

    pred_dir = exp_dir / "predictions" / ds_name
    gt_dir   = ds_dir / "labelsTs"
    mapping  = json.loads((ds_dir / "case_mapping.json").read_text())

    all_m = []
    for pred_file in sorted(pred_dir.glob("*.png")):
        cid = pred_file.stem
        gt_file = gt_dir / f"{cid}.png"
        if not gt_file.exists():
            print(f"  Warning: no GT for {cid}, skipping")
            continue
        m = metrics_for(pred_file, gt_file)
        m["case_id"] = cid
        m["original_filename"] = mapping.get(cid, cid)
        all_m.append(m)

    if not all_m:
        print("No predictions found.")
        return

    avg = {k: float(np.mean([m[k] for m in all_m]))
           for k in ("dice", "iou", "precision", "recall", "hd95")}

    results_dir = Path("experiments/exp1_fullsupervised/traditional_imageonly/results/nnunet")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate CSV (same format as U-Net)
    csv_path = results_dir / "nnunet_results.csv"
    row = {
        "dataset": args.dataset,
        "model": "nnunet_v2",
        "lr": "self_configured",
        "val_dice": "N/A",
        "test_dice": round(avg["dice"], 6),
        "test_iou": round(avg["iou"], 6),
        "test_precision": round(avg["precision"], 6),
        "test_recall": round(avg["recall"], 6),
        "test_hd95": round(avg["hd95"], 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": 42,
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

    # Per-case detail CSV
    detail_path = results_dir / f"nnunet_results_{args.dataset}_details.csv"
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "original_filename",
                                           "dice", "iou", "precision", "recall", "hd95"])
        w.writeheader()
        for m in all_m:
            w.writerow({k: round(m[k], 4) if isinstance(m[k], float) else m[k]
                        for k in w.fieldnames})

    print(f"[{args.dataset}] test_dice={avg['dice']:.4f} | "
          f"iou={avg['iou']:.4f} | hd95={avg['hd95']:.2f}")
    print(f"  Aggregate → {csv_path}")
    print(f"  Per-case  → {detail_path}")


if __name__ == "__main__":
    main()
