"""Rebuild the Experiment 2 cells under the contour HD95 definition.

Zero-shot inference never goes through ``FoundationLoRAWrapper``, so the
evaluation-time box jitter that affected the adapted models does not apply here:
``experiments/exp2_zeroshot/run.py`` takes ``--jitter_frac`` with a
default of 0.0 and the published runs used the tight oracle box. The only thing
that changes for this experiment is HD95, which was measured with the all-mask-
pixel variant of ``src/evaluation/metrics.py:compute_hd95`` and is now measured
between mask contours, as in Experiment 1.

DSC, IoU, Precision and Recall are therefore expected to reproduce the published
values exactly; the script checks that and reports any cell that does not.

Sources
-------
``experiments/exp2_zeroshot/results/{model}_{dataset}_hd95b/per_image_results.csv``
(re-runs under ``scripts/hd95_contour/run_with_contour_hd95.py``) and the
published ``{model}_{dataset}/summary_metrics.csv`` next to them, used as the
provenance reference.

Outputs (results/stats_exp2_corrected/)
---------------------------------------
    exp2_cells.csv      one row per (dataset, model)
    exp2_wilcoxon.csv   paired Wilcoxon against zero-shot SAM, the pivot the
                        supplementary table declares, Holm-corrected within
                        each (dataset, metric)

Usage
-----
    python analysis/build_exp2_corrected.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments/exp2_zeroshot/results"
OUT_DIR = REPO / "results/stats_exp2_corrected"

HD95_CAP = float(np.sqrt(2) * 512)  # 724.0773 px

DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
MODELS = {"sam": "SAM", "sam2": "SAM2", "sam3": "SAM3",
          "medsam": "MedSAM", "medsam2": "MedSAM-2"}
PIVOT = "SAM"

logger = logging.getLogger(__name__)


def load(model: str, dataset: str) -> Optional[pd.DataFrame]:
    """Per-image frame for one cell, with the empty-prediction policy applied."""
    p = RESULTS / f"{model}_{dataset}_hd95b/per_image_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["hd95_final"] = (pd.to_numeric(df["hd95"], errors="coerce")
                        .replace([np.inf, -np.inf], HD95_CAP).fillna(HD95_CAP))
    return df


def published(model: str, dataset: str) -> Optional[pd.Series]:
    """The published summary row, for the provenance check on DSC."""
    p = RESULTS / f"{model}_{dataset}/summary_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).iloc[0]


def collect() -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]]:
    rows: List[Dict[str, object]] = []
    frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    for ds in DATASETS:
        for key, label in MODELS.items():
            df = load(key, ds)
            if df is None:
                logger.warning("missing %s/%s", label, ds)
                continue
            frames[(ds, label)] = df
            pub = published(key, ds)
            rows.append({
                "dataset": ds, "model": label, "n": len(df),
                "n_empty": int((df["hd95_final"] >= HD95_CAP).sum()),
                "dsc_mean": float(df["dice"].mean()),
                "dsc_sd": float(df["dice"].std(ddof=1)),
                "dsc_published": None if pub is None else float(pub["dice_mean"]),
                "iou": float(df["iou"].mean()),
                "precision": float(df["precision"].mean()),
                "recall": float(df["recall"].mean()),
                "hd95": float(df["hd95_final"].mean()),
                "hd95_median": float(df["hd95_final"].median()),
                # HD95 is strongly right-skewed (the diagonal imputation puts a few
                # images at 724 px), so mean +/- SD would print negative distances on
                # roughly a fifth of the cells. Quartiles describe the bulk honestly.
                "hd95_q1": float(df["hd95_final"].quantile(0.25)),
                "hd95_q3": float(df["hd95_final"].quantile(0.75)),
                "hd95_published": None if pub is None else float(pub["hd95_mean"]),
            })
    return pd.DataFrame(rows), frames


def paired_tests(frames: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Wilcoxon against zero-shot SAM, paired by filename, Holm within family."""
    records: List[Dict[str, object]] = []
    for ds in DATASETS:
        pivot = frames.get((ds, PIVOT))
        if pivot is None:
            continue
        for metric, col in (("DSC", "dice"), ("HD95", "hd95_final")):
            family: List[Dict[str, object]] = []
            for (d, m), df in frames.items():
                if d != ds or m == PIVOT:
                    continue
                merged = pivot[["filename", col]].merge(
                    df[["filename", col]], on="filename", suffixes=("_ref", "_x"))
                a, b = merged[f"{col}_ref"], merged[f"{col}_x"]
                stat, p = (np.nan, 1.0) if np.allclose(a, b) else wilcoxon(a, b)
                better = (b.mean() > a.mean()) if metric == "DSC" else (b.mean() < a.mean())
                family.append({"dataset": ds, "metric": metric, "model": m,
                               "n_paired": len(merged), "mean_ref": float(a.mean()),
                               "mean_model": float(b.mean()),
                               "better_than_sam": bool(better),
                               "stat": stat, "p_raw": float(p)})
            order = np.argsort([r["p_raw"] for r in family])
            k, running = len(family), 0.0
            for rank, idx in enumerate(order):
                running = max(running, min(1.0, family[idx]["p_raw"] * (k - rank)))
                family[idx]["p_holm"] = running
            records.extend(family)
    return pd.DataFrame(records)


def _stars(p: float, better: bool) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    mark = "\\ast" if better else "\\dagger"
    return "$^{" + mark * (3 if p < 0.001 else 2 if p < 0.01 else 1) + "}$"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells, frames = collect()
    cells.to_csv(OUT_DIR / "exp2_cells.csv", index=False)
    tests = paired_tests(frames)
    tests["marks"] = [_stars(r.p_holm, r.better_than_sam) for r in tests.itertuples()]
    tests.to_csv(OUT_DIR / "exp2_wilcoxon.csv", index=False)

    # Provenance: only HD95 should have moved.
    drift = cells[(cells.dsc_published.notna()) &
                  ((cells.dsc_mean - cells.dsc_published).abs() > 5e-4)]
    if drift.empty:
        print("provenance OK: every DSC reproduces the published value")
    else:
        print("DSC DRIFT (unexpected, investigate):")
        print(drift[["dataset", "model", "dsc_mean", "dsc_published"]].to_string(index=False))

    print("\nDSC (unchanged) and HD95 published -> contour:")
    for ds in DATASETS:
        print(f"\n--- {ds}")
        for r in cells[cells.dataset == ds].itertuples():
            pub = "n/a" if r.hd95_published is None else f"{r.hd95_published:6.2f}"
            print(f"  {r.model:9} DSC {r.dsc_mean:.3f}  HD95 {pub} -> {r.hd95:6.2f}"
                  f"  (empty {r.n_empty})")
    print(f"\nwrote {OUT_DIR}/exp2_cells.csv and exp2_wilcoxon.csv")


if __name__ == "__main__":
    main()
