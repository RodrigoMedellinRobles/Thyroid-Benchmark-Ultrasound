"""Shared helpers used by 01-04_preprocess_*.py.

Two responsibilities:
  1. update_summary_csv() — append/overwrite a per-dataset row in a project-wide
     CSV at data/processed/preprocessing_summary.csv. Each preprocess script writes
     the rows it owns; 06_compute_dataset_stats.py later fills in mean/std/n_train.
  2. save_overview_figure() — produce a per-dataset PNG with raw vs processed
     samples + binary mask + overlay, plus a header summarising the operations
     applied. Saved under figures/data_prep/.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 00_setup/_lib/ -> repo root
SUMMARY_CSV  = PROJECT_ROOT / "data" / "processed" / "preprocessing_summary.csv"
FIGURES_DIR  = PROJECT_ROOT / "figures" / "data_prep"

CSV_COLUMNS = [
    "dataset",
    "n_images",
    "n_masks",
    "target_size",
    "mask_threshold",
    "mean",          # filled by 06_compute_dataset_stats.py
    "std",           # filled by 06_compute_dataset_stats.py
    "n_train",       # filled by 06_compute_dataset_stats.py
    "observation",
    "preprocessed_at",
    "stats_at",      # filled by 06_compute_dataset_stats.py
]


def _read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def update_summary_csv(
    dataset: str,
    n_images: int,
    n_masks: int,
    target_size: int,
    mask_threshold: str,
    observation: str,
    csv_path: Path = SUMMARY_CSV,
) -> Path:
    """Insert or overwrite the row for `dataset` with preprocess-time fields.

    mean/std/n_train/stats_at are left untouched (filled later by stats script).
    """
    rows = _read_rows(csv_path)
    keep = [r for r in rows if r.get("dataset") != dataset]
    existing = next((r for r in rows if r.get("dataset") == dataset), {})
    new_row = {
        "dataset":        dataset,
        "n_images":       n_images,
        "n_masks":        n_masks,
        "target_size":    target_size,
        "mask_threshold": mask_threshold,
        "mean":           existing.get("mean", ""),
        "std":            existing.get("std", ""),
        "n_train":        existing.get("n_train", ""),
        "observation":    observation,
        "preprocessed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats_at":       existing.get("stats_at", ""),
    }
    keep.append(new_row)
    keep.sort(key=lambda r: r["dataset"])
    _write_rows(csv_path, keep)
    return csv_path


def update_stats_in_csv(
    dataset: str,
    mean: float,
    std: float,
    n_train: int,
    csv_path: Path = SUMMARY_CSV,
) -> Path:
    """Called by 06_compute_dataset_stats.py to fill the post-split fields."""
    rows = _read_rows(csv_path)
    found = False
    for r in rows:
        if r.get("dataset") == dataset:
            r["mean"]    = f"{mean:.6f}"
            r["std"]     = f"{std:.6f}"
            r["n_train"] = str(n_train)
            r["stats_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if not found:
        rows.append({
            "dataset":  dataset,
            "n_images": "",
            "n_masks":  "",
            "target_size":    "",
            "mask_threshold": "",
            "mean":     f"{mean:.6f}",
            "std":      f"{std:.6f}",
            "n_train":  str(n_train),
            "observation":    "",
            "preprocessed_at": "",
            "stats_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    rows.sort(key=lambda r: r["dataset"])
    _write_rows(csv_path, rows)
    return csv_path


def save_overview_figure(
    dataset: str,
    samples: Iterable[tuple[np.ndarray, np.ndarray]],
    title_lines: list[str],
    out_dir: Path = FIGURES_DIR,
    n_show: int = 3,
) -> Path:
    """Render a multi-row figure: raw image | binary mask | mask-on-image overlay.

    Args:
        dataset: short tag used in filename, e.g. "ddti".
        samples: iterable yielding (image_uint8_HxWx3, mask_uint8_HxW) pairs.
        title_lines: pre-formatted text shown as the figure subtitle.
    """
    samples = list(samples)[:n_show]
    if not samples:
        return out_dir / f"{dataset}_preprocess_overview.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_preprocess_overview.png"

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.6 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i, (img, mask) in enumerate(samples):
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image (sample {i+1})", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Mask (binary)", fontsize=10)
        axes[i, 1].axis("off")

        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=-1)
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        m3 = (mask > 0)[..., None]
        overlay = np.where(m3, (0.55 * overlay + 0.45 * red).astype(np.uint8), overlay)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay (mask in red)", fontsize=10)
        axes[i, 2].axis("off")

    fig.suptitle("\n".join(title_lines), fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
