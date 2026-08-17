"""Aggregate Exp 4.5 cross-dataset LoRA evaluation results.

Walks `experiments/exp4_crossdataset/foundation/results/exp4_5_*/summary_metrics.csv`,
concatenates into a master CSV and produces per-(model, fraction) 4x4 matrices
for DSC and HD95. Diagonal cells are filled from the matched Exp 3 in-domain
runs (`experiments/exp3_fewshot_lora/results/fewshot_<model>_<dataset>_f<frac>/
summary_metrics.csv`) so the matrices read as drop-in replacements for
the cross-dataset LoRA table (tab:exp45) in the paper.

Usage:
    python experiments/exp4_crossdataset/foundation/aggregate_exp4_5.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "exp4_crossdataset" / "foundation" / "results"
EXP3_RESULTS = PROJECT_ROOT / "experiments" / "exp3_fewshot_lora" / "results"

MODELS = ("sam2", "sam3", "medsam", "medsam2")
DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")
FRACTIONS = (0.05, 0.5)


def _load_summary(csv_path: Path) -> dict | None:
    if not csv_path.is_file():
        return None
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return rows[0]


def collect_offdiag() -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        for src in DATASETS:
            for tgt in DATASETS:
                if src == tgt:
                    continue
                for frac in FRACTIONS:
                    run_name = f"exp4_5_{model}_src{src}_tgt{tgt}_f{frac}"
                    summary = _load_summary(RESULTS_DIR / run_name / "summary_metrics.csv")
                    if summary is None:
                        rows.append({
                            "model": model, "src_dataset": src, "tgt_dataset": tgt,
                            "fraction": frac, "status": "missing", "dice_mean": np.nan,
                            "dice_std": np.nan, "iou_mean": np.nan, "iou_std": np.nan,
                            "hd95_mean": np.nan, "hd95_std": np.nan, "n_images": np.nan,
                        })
                        continue
                    rows.append({
                        "model": model, "src_dataset": src, "tgt_dataset": tgt,
                        "fraction": frac, "status": "ok",
                        "dice_mean": float(summary["dice_mean"]),
                        "dice_std": float(summary["dice_std"]),
                        "iou_mean": float(summary["iou_mean"]),
                        "iou_std": float(summary["iou_std"]),
                        "hd95_mean": float(summary["hd95_mean"]),
                        "hd95_std": float(summary["hd95_std"]),
                        "n_images": int(summary["n_images"]),
                    })
    return pd.DataFrame(rows)


def collect_diag_from_exp3() -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        for ds in DATASETS:
            for frac in FRACTIONS:
                run_name = f"fewshot_{model}_{ds}_f{frac}"
                summary = _load_summary(EXP3_RESULTS / run_name / "summary_metrics.csv")
                if summary is None:
                    rows.append({
                        "model": model, "src_dataset": ds, "tgt_dataset": ds,
                        "fraction": frac, "status": "missing_exp3",
                        "dice_mean": np.nan, "dice_std": np.nan,
                        "iou_mean": np.nan, "iou_std": np.nan,
                        "hd95_mean": np.nan, "hd95_std": np.nan, "n_images": np.nan,
                    })
                    continue
                rows.append({
                    "model": model, "src_dataset": ds, "tgt_dataset": ds,
                    "fraction": frac, "status": "ok_exp3",
                    "dice_mean": float(summary["dice_mean"]),
                    "dice_std": float(summary["dice_std"]),
                    "iou_mean": float(summary.get("iou_mean", "nan") or "nan"),
                    "iou_std": float(summary.get("iou_std", "nan") or "nan"),
                    "hd95_mean": float(summary.get("hd95_mean", "nan") or "nan"),
                    "hd95_std": float(summary.get("hd95_std", "nan") or "nan"),
                    "n_images": int(summary.get("n_images", 0) or 0),
                })
    return pd.DataFrame(rows)


def build_matrix(df: pd.DataFrame, model: str, frac: float, metric: str) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["fraction"] == frac)].copy()
    mat = sub.pivot(index="src_dataset", columns="tgt_dataset", values=metric)
    mat = mat.reindex(index=DATASETS, columns=DATASETS)
    return mat


def main() -> int:
    print(f"[aggregate] results dir: {RESULTS_DIR}")
    if not RESULTS_DIR.is_dir():
        print(f"[aggregate] ERROR: results dir missing", file=sys.stderr)
        return 1

    off = collect_offdiag()
    diag = collect_diag_from_exp3()
    full = pd.concat([off, diag], ignore_index=True)

    # Guard: if no source per-run results are present (e.g. a fresh clone before
    # the experiments are re-run), do NOT overwrite the committed master/matrices
    # with all-"missing" NaN rows. Regenerate the per-run inputs first.
    if (off["status"] == "ok").sum() == 0 and (diag["status"] == "ok_exp3").sum() == 0:
        print("[aggregate] no source cells found; leaving committed masters/matrices untouched")
        return 0

    out_master = RESULTS_DIR / "exp4_5_master.csv"
    full.to_csv(out_master, index=False, float_format="%.6f")
    print(f"[aggregate] master CSV -> {out_master}  rows={len(full)}")

    n_off = (off["status"] == "ok").sum()
    n_off_missing = (off["status"] == "missing").sum()
    n_diag = (diag["status"] == "ok_exp3").sum()
    n_diag_missing = (diag["status"] == "missing_exp3").sum()
    print(f"[aggregate] off-diag cells filled: {n_off}/{len(off)}  missing={n_off_missing}")
    print(f"[aggregate] diag    cells filled: {n_diag}/{len(diag)}  missing={n_diag_missing}")

    matrices_dir = RESULTS_DIR / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for frac in FRACTIONS:
            for metric in ("dice_mean", "hd95_mean"):
                mat = build_matrix(full, model, frac, metric)
                csv_path = matrices_dir / f"{model}_f{frac}_{metric}.csv"
                mat.to_csv(csv_path, float_format="%.4f")
                print(f"[aggregate] matrix -> {csv_path}")

    summary_lines = []
    for model in MODELS:
        for frac in FRACTIONS:
            dsc = build_matrix(full, model, frac, "dice_mean")
            diag_vals = np.diag(dsc.values)
            off_mask = ~np.eye(len(DATASETS), dtype=bool)
            off_vals = dsc.values[off_mask]
            diag_mean = float(np.nanmean(diag_vals))
            off_mean = float(np.nanmean(off_vals))
            drop = diag_mean - off_mean
            summary_lines.append(
                f"{model:8s}  f={frac}  diag_DSC={diag_mean:.4f}  "
                f"off_DSC={off_mean:.4f}  drop={drop:+.4f}"
            )
    summary_txt = "\n".join(summary_lines)
    print("\n[aggregate] cross-dataset DSC drop summary:")
    print(summary_txt)
    (RESULTS_DIR / "exp4_5_drop_summary.txt").write_text(summary_txt + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
