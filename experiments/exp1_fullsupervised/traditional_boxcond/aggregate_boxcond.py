"""Aggregate Exp 1 box-conditioned traditional baselines into one summary CSV.

Collects the per-model box-conditioned test metrics for U-Net, TransUNet and
nnU-Net v2 across the four datasets and writes a single tidy table with one row
per (model, dataset):

    results/exp1_boxcond_summary.csv
      model, dataset, dsc, iou, precision, recall, hd95

Input discovery is tolerant: any missing per-model fragment is skipped (with a
warning) rather than raising, so a partial reproduction still yields a partial
summary. Re-run the Exp 1 traditional pipeline to regenerate the per-model
fragments, then run this script.

Usage:
    python experiments/exp1_fullsupervised/traditional_boxcond/aggregate_boxcond.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _collect_cnn(subdir: str, suffix: str, model_label: str) -> list[dict]:
    """One `<dataset>_<suffix>.csv` per dataset, each holding a single row."""
    out: list[dict] = []
    for ds in DATASETS:
        path = RESULTS_DIR / subdir / f"{ds}_{suffix}.csv"
        rows = _read_rows(path)
        if not rows:
            print(f"[boxcond] skip missing {path.relative_to(RESULTS_DIR.parent)}")
            continue
        r = rows[0]
        out.append({
            "model": model_label,
            "dataset": r["dataset"],
            "dsc": float(r["test_dice"]),
            "iou": float(r["test_iou"]),
            "precision": float(r["test_precision"]),
            "recall": float(r["test_recall"]),
            "hd95": float(r["test_hd95"]),
        })
    return out


def _collect_nnunet() -> list[dict]:
    """Single summary CSV with one row per dataset."""
    path = RESULTS_DIR / "nnunet" / "nnunet_boxcond_results.csv"
    rows = _read_rows(path)
    if not rows:
        print(f"[boxcond] skip missing {path.relative_to(RESULTS_DIR.parent)}")
        return []
    out: list[dict] = []
    for r in rows:
        out.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "dsc": float(r["test_dice"]),
            "iou": float(r["test_iou"]),
            "precision": float(r["test_precision"]),
            "recall": float(r["test_recall"]),
            "hd95": float(r["test_hd95"]),
        })
    return out


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"[boxcond] ERROR: results dir missing: {RESULTS_DIR}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    rows += _collect_cnn("unet", "unet_boxcond_results", "unet_resnet50")
    rows += _collect_cnn("transunet", "transunet_boxcond_results", "transunet_r50_vitb16")
    rows += _collect_nnunet()

    if not rows:
        print("[boxcond] no fragments found; nothing written")
        return 0

    out_path = RESULTS_DIR / "exp1_boxcond_summary.csv"
    fields = ["model", "dataset", "dsc", "iou", "precision", "recall", "hd95"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[boxcond] wrote {out_path.relative_to(RESULTS_DIR.parent)}  rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
