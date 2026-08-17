"""Aggregate Exp 1 fully-supervised (f=1.0) foundation-model LoRA runs.

Walks the per-run oracle-box directories
`results/fullsup_<model>_<dataset>/summary_metrics.csv` and writes one tidy
table with a single row per (model, dataset):

    results/exp1_foundation_fullsup_summary.csv
      model, dataset, fraction, n_images, dsc, iou, precision, recall, hd95

The `_yolo` (automatic-prompt) variants are ignored; this summary reports the
oracle-box regime only. Input discovery is tolerant: missing per-run
directories are skipped with a warning so a partial reproduction still yields a
partial summary.

Usage:
    python experiments/exp1_fullsupervised/foundation_lora/aggregate_fullsup.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS = ("sam2", "sam3", "medsam", "medsam2")
DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")


def _read_first_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"[fullsup] ERROR: results dir missing: {RESULTS_DIR}", file=sys.stderr)
        return 1

    out_rows: list[dict] = []
    for model in MODELS:
        for ds in DATASETS:
            path = RESULTS_DIR / f"fullsup_{model}_{ds}" / "summary_metrics.csv"
            r = _read_first_row(path)
            if r is None:
                print(f"[fullsup] skip missing fullsup_{model}_{ds}/summary_metrics.csv")
                continue
            out_rows.append({
                "model": r["model"],
                "dataset": r["dataset"],
                "fraction": float(r.get("fraction", 1.0)),
                "n_images": int(float(r["n_images"])),
                "dsc": float(r["dice_mean"]),
                "iou": float(r["iou_mean"]),
                "precision": float(r["precision_mean"]),
                "recall": float(r["recall_mean"]),
                "hd95": float(r["hd95_mean"]),
            })

    if not out_rows:
        print("[fullsup] no fragments found; nothing written")
        return 0

    out_path = RESULTS_DIR / "exp1_foundation_fullsup_summary.csv"
    fields = ["model", "dataset", "fraction", "n_images",
              "dsc", "iou", "precision", "recall", "hd95"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[fullsup] wrote {out_path.name}  rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
