#!/usr/bin/env python3
"""Box-fill NULL floor per dataset for the box-prompted regimes (Exp 1 box-cond + Exp 2).

The floor is the DSC/IoU obtained by taking the tight ground-truth bounding box,
filled, as the prediction (no network) — the trivial lower bound that any
box-prompted model must exceed. It depends only on (mask, tight box), so it is a
single value per dataset shared by every box-prompted model. Computed here with
the exact ZeroShotDataset + get_gt_box used by Exp 2, which reproduces the
box-conditioned CNN floors (box_eval.py) to four decimals — confirming the tight
boxes are pixel-identical across pipelines.

Usage: python analysis/compute_boxfill_floor.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thyroidbench.models.foundation_base import ZeroShotDataset, get_gt_box
from thyroidbench.metrics import compute_hd95

DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
DATA_ROOT = Path("data/processed")
SPLIT_DIR = Path("data/splits")


def main() -> None:
    for ds in DATASETS:
        data = ZeroShotDataset(dataset_name=ds,
                               split_csv=SPLIT_DIR / f"{ds}_test.csv",
                               data_root=DATA_ROOT, img_size=512)
        dd, ii, pp, rr, hh = [], [], [], [], []
        for i in range(len(data)):
            _image, mask, _fn, _pid = data[i]
            box = get_gt_box(mask)
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            pred = np.zeros_like(mask, dtype=np.float32)
            pred[y1:y2 + 1, x1:x2 + 1] = 1.0
            m = (mask > 0).astype(np.float32)
            inter = float((pred * m).sum())
            ps, ms = pred.sum(), m.sum()
            dd.append(2 * inter / (ps + ms + 1e-10))
            ii.append(inter / (ps + ms - inter + 1e-10))
            pp.append(inter / (ps + 1e-10))           # precision = |mask|/|box|
            rr.append(inter / (ms + 1e-10))           # recall = 1.0 (box encloses mask)
            hh.append(compute_hd95(pred.astype(np.uint8), m.astype(np.uint8)))
        print(f"{ds:14s} DSC={np.mean(dd):.4f} IoU={np.mean(ii):.4f} "
              f"Prec={np.mean(pp):.4f} Rec={np.mean(rr):.4f} HD95={np.mean(hh):.2f} n={len(dd)}")


if __name__ == "__main__":
    main()
