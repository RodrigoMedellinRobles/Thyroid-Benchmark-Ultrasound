"""
Compute per-dataset mean/std from train splits (post-CLAHE).

Outputs:
  - data/preprocessing_stats.json (canonical stats consumed by thyroidbench/data.py)
  - data/processed/preprocessing_summary.csv (mean/std/n_train rows updated)
  - figures/data_prep/{dataset}_compute_stats_overview.png
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preprocess_utils import update_stats_in_csv

DATASETS = ["tn3k", "ddti", "thyroidxl", "stanford_aimi"]
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "figures" / "data_prep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on luminance channel, return BGR uint8."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_stats(dataset: str) -> dict:
    train_csv = SPLITS_DIR / f"{dataset}_train.csv"
    df = pd.read_csv(train_csv)
    img_dir = PROCESSED_DIR / dataset / "images"

    sum_px = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    print(f"[{dataset}] computing stats from {len(df)} train images...")
    for i, row in df.iterrows():
        img_path = img_dir / row["filename"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        img = apply_clahe(img)
        img_f = img.astype(np.float64) / 255.0  # [0,1]
        # BGR → RGB channel order
        img_f = img_f[:, :, ::-1]
        h, w, c = img_f.shape
        sum_px += img_f.reshape(-1, 3).sum(axis=0)
        sum_sq += (img_f ** 2).reshape(-1, 3).sum(axis=0)
        n_pixels += h * w

    mean = sum_px / n_pixels
    std = np.sqrt(sum_sq / n_pixels - mean ** 2)
    return {"mean": mean.tolist(), "std": std.tolist(), "n_train": int(len(df))}


def save_sample_grid(dataset: str, stats: dict):
    train_csv = SPLITS_DIR / f"{dataset}_train.csv"
    df = pd.read_csv(train_csv)
    img_dir = PROCESSED_DIR / dataset / "images"
    mask_dir = PROCESSED_DIR / dataset / "masks"

    # pick 3 samples
    sample_rows = df.sample(n=min(3, len(df)), random_state=42)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"{dataset} — original / post-CLAHE / normalized (z-score)", fontsize=13)

    col_labels = ["Original", "CLAHE", "Z-score normalized"]
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=11)

    for row_i, (_, sr) in enumerate(sample_rows.iterrows()):
        img_path = img_dir / sr["filename"]
        mask_path = mask_dir / sr["filename"]

        img_orig = cv2.imread(str(img_path))
        img_clahe = apply_clahe(img_orig)

        # normalized
        img_f = img_clahe.astype(np.float32) / 255.0
        img_f = img_f[:, :, ::-1]  # BGR→RGB
        img_norm = (img_f - mean) / std

        # for display: scale back to [0,1]
        img_disp = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

        # original RGB
        orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

        # overlay mask on original
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            orig_overlay = orig_rgb.copy()
            cv2.drawContours(orig_overlay, contours, -1, (255, 0, 0), 2)
        else:
            orig_overlay = orig_rgb

        axes[row_i, 0].imshow(orig_overlay)
        axes[row_i, 0].axis("off")
        axes[row_i, 0].set_ylabel(sr["filename"][:20], fontsize=8)

        axes[row_i, 1].imshow(clahe_rgb)
        axes[row_i, 1].axis("off")

        axes[row_i, 2].imshow(img_disp)
        axes[row_i, 2].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / f"{dataset}_compute_stats_overview.png"
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    all_stats = {}
    for ds in DATASETS:
        stats = compute_stats(ds)
        all_stats[ds] = stats
        print(f"  mean={[f'{v:.4f}' for v in stats['mean']]}  std={[f'{v:.4f}' for v in stats['std']]}")
        save_sample_grid(ds, stats)
        # Update community summary CSV with mean/std/n_train (channel 0 of mean/std,
        # all 3 channels are nearly identical for grayscale-derived inputs).
        update_stats_in_csv(
            dataset=ds,
            mean=float(stats["mean"][0]),
            std=float(stats["std"][0]),
            n_train=int(stats["n_train"]),
        )

    out_json = DATA_DIR / "preprocessing_stats.json"
    with open(out_json, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved → {out_json}")

    print("\n=== SUMMARY ===")
    for ds, s in all_stats.items():
        m = [f"{v:.4f}" for v in s["mean"]]
        st = [f"{v:.4f}" for v in s["std"]]
        print(f"{ds:20s}  mean={m}  std={st}  n_train={s['n_train']}")
