#!/usr/bin/env python3
"""
Score one box-conditioned nnU-Net cross-dataset cell (Stream B).

Given a directory of nnU-Net predictions for (source-trained model applied to
target test images) and the target's ground-truth labels, compute per-case +
summary metrics in the SAME schema Stream A (run.py) emits, so the
CNN and nnU-Net box-conditioned cross rows merge into one table.

Metric definitions are copied verbatim from
experiments/exp1_fullsupervised/traditional_boxcond/eval_nnunet_boxcond.py (Dice/IoU/precision/
recall + boundary HD95) EXCEPT the empty-prediction HD95 policy: for the
cross-dataset table we report HD95 finite-only (empty -> inf, dropped from the
mean), matching the foundation cross harness and Stream A. The per-case CSV
keeps an empty string for inf so the count of finite cases is recoverable.

Usage:
    python experiments/exp4_crossdataset/traditional/eval_nnunet_cross.py \
        --src ddti --tgt tn3k \
        --pred_dir experiments/exp4_crossdataset/traditional/preds/src_ddti_tgt_tn3k
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

DATASET_IDS = {"ddti": "011", "tn3k": "012", "thyroidxl": "013", "stanford_aimi": "014"}
_OUT_DIR = Path("experiments/exp4_crossdataset/traditional/results")


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Boundary HD95. Empty pred/gt -> inf (finite-only policy for cross table)."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    pb = pred ^ binary_erosion(pred)
    gb = gt ^ binary_erosion(gt)
    dt_p = distance_transform_edt(~pb.astype(bool))
    dt_g = distance_transform_edt(~gb.astype(bool))
    dists = np.concatenate([dt_g[pb > 0], dt_p[gb > 0]])
    return float(np.percentile(dists, 95))


def metrics_for(pred_path: Path, gt_path: Path) -> dict:
    pred = (np.array(Image.open(pred_path).convert("L")) > 0).astype(np.float32)
    gt = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.float32)
    inter = (pred * gt).sum()
    dice = 2 * inter / (pred.sum() + gt.sum() + 1e-10)
    iou = inter / (pred.sum() + gt.sum() - inter + 1e-10)
    tp, fp, fn = inter, pred.sum() - inter, gt.sum() - inter
    return {
        "dice": float(dice), "iou": float(iou),
        "precision": float(tp / (tp + fp + 1e-10)),
        "recall": float(tp / (tp + fn + 1e-10)),
        "hd95": hd95(pred.astype(np.uint8), gt.astype(np.uint8)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Score one nnU-Net box-cond cross cell")
    p.add_argument("--src", required=True, choices=list(DATASET_IDS))
    p.add_argument("--tgt", required=True, choices=list(DATASET_IDS))
    p.add_argument("--pred_dir", required=True)
    args = p.parse_args()

    tgt_did = DATASET_IDS[args.tgt]
    cand = list(Path("data/nnunet_raw").glob(f"Dataset{tgt_did}_*"))
    if not cand:
        raise FileNotFoundError(f"target Dataset{tgt_did}_* not in data/nnunet_raw/")
    gt_dir = cand[0] / "labelsTs"

    pred_dir = Path(args.pred_dir)
    rows = []
    for pred_file in sorted(pred_dir.glob("*.png")):
        cid = pred_file.stem
        gt_file = gt_dir / f"{cid}.png"
        if not gt_file.exists():
            print(f"  warn: no GT for {cid}, skip")
            continue
        m = metrics_for(pred_file, gt_file)
        rows.append({
            "filename": cid, "patient_id": "NA",
            "dice": round(m["dice"], 6), "iou": round(m["iou"], 6),
            "precision": round(m["precision"], 6), "recall": round(m["recall"], 6),
            "hd95": round(m["hd95"], 4) if np.isfinite(m["hd95"]) else "",
            "skipped": False,
        })
    if not rows:
        raise RuntimeError(f"no predictions scored in {pred_dir}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"nnunet_v2_boxcond_src{args.src}_tgt{args.tgt}"
    pd.DataFrame(rows).to_csv(_OUT_DIR / f"{tag}_per_image.csv", index=False)

    dice = np.array([r["dice"] for r in rows], dtype=float)
    iou = np.array([r["iou"] for r in rows], dtype=float)
    prec = np.array([r["precision"] for r in rows], dtype=float)
    rec = np.array([r["recall"] for r in rows], dtype=float)
    # Impute-aware HD95 (paper-wide policy). Empty / non-finite predictions are
    # penalised at the image diagonal (sqrt(2)*512 = 724.077 px) BEFORE the
    # per-image mean, rather than being silently dropped (which understates HD95;
    # that drop was the cross-dataset figure/table defect). Ground truth:
    # the impute-aware HD95 recomputation described below
    HD95_IMPUTE_PX = float(np.sqrt(2) * 512)  # 724.077
    hd = np.array([float(r["hd95"]) for r in rows if r["hd95"] != ""], dtype=float)
    hd_impute = np.array(
        [float(r["hd95"]) if r["hd95"] != "" else HD95_IMPUTE_PX for r in rows],
        dtype=float)
    summary = dict(
        model="nnunet_v2_boxcond", src_dataset=args.src, tgt_dataset=args.tgt,
        fraction=1.0, status="ok", box_conditioned=True, diagonal=(args.src == args.tgt),
        n_images=len(rows), n_hd95_finite=int(len(hd)),
        n_hd95_empty=int(len(hd_impute) - len(hd)),
        dice_mean=float(np.mean(dice)), dice_std=float(np.std(dice)),
        iou_mean=float(np.mean(iou)), precision_mean=float(np.mean(prec)),
        recall_mean=float(np.mean(rec)),
        hd95_mean=float(np.mean(hd_impute)) if len(hd_impute) else float("nan"),
        hd95_std=float(np.std(hd_impute)) if len(hd_impute) else float("nan"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), seed=42,
    )
    master = _OUT_DIR / "boxcond_nnunet_crossdataset_master.csv"
    df = pd.DataFrame([summary])
    if master.exists():
        df = pd.concat([pd.read_csv(master), df], ignore_index=True)
        df = df.drop_duplicates(subset=["model", "src_dataset", "tgt_dataset"], keep="last")
    df.to_csv(master, index=False)
    print(f"[{args.src}->{args.tgt}] DSC={summary['dice_mean']:.4f} "
          f"HD95={summary['hd95_mean']:.2f} (n={len(rows)}, finite_hd95={len(hd)})")


if __name__ == "__main__":
    main()
