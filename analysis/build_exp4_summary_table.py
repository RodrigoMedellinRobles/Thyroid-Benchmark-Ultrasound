"""Build the Experiment 4 summary table (main-text Table 5) from the per-cell data.

The seven-row summary was transcribed by hand, so its In-dom., Cross and Delta
columns could drift from the tables they summarise, and they did: the In-dom.
HD95 column mixed two measurement definitions, with only the nnU-Net row on the
contour one. This script derives every cell from the same sources the
supplementary matrices use, so the summary cannot disagree with them.

Column definitions, unchanged from the published table:
    In-dom.  mean over the four diagonal (same-dataset) cells
    Cross    mean over the twelve off-diagonal (transfer) cells
    Delta    degradation under shift; DSC: In-dom. - Cross, HD95: Cross - In-dom.

Sources
-------
Cross cells      experiments/exp4_crossdataset/traditional/results/exp4_hd95_impute_aware_recompute.csv
                 for HD95 (rebuild it first with build_exp4_hd95_table.py) and the
                 per-image CSVs for DSC.
Diagonals        results/stats_exp1_corrected/exp1_cells.csv for the CNNs and for
                 foundation f=100 %; results/stats_exp3_corrected/exp3_cells.csv
                 for foundation f=5 %.

HD95 uses the impute-aware policy throughout: an empty prediction is counted at
the image diagonal rather than dropped.

Usage
-----
    python analysis/build_exp4_summary_table.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HD95_CSV = REPO / "experiments/exp4_crossdataset/traditional/results/exp4_hd95_impute_aware_recompute.csv"
E1 = REPO / "results/stats_exp1_corrected/exp1_cells.csv"
E3 = REPO / "results/stats_exp3_corrected/exp3_cells.csv"
CNN_DIR = REPO / "experiments/exp4_crossdataset/traditional/results_boxcond_cnn"
NNUNET_DIR = REPO / "experiments/exp4_crossdataset/traditional/results_boxcond_nnunet"
LORA_NEW = REPO / "experiments/exp4_crossdataset/foundation/results_hd95b"
LORA_OLD = REPO / "experiments/exp4_crossdataset/foundation/results"
OUT = REPO / "results/stats_exp4_corrected"

HD95_CAP = float(np.sqrt(2) * 512)
DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]

# (display name, key in the HD95 csv, fraction, block/label used by the diagonals)
ROWS = [
    ("U-Net",     "unet_resnet50",        1.0,  ("exp1", "box-cond", "U-Net")),
    ("TransUNet", "transunet_r50_vitb16", 1.0,  ("exp1", "box-cond", "TransUNet")),
    ("nnU-Net v2", "nnunet_v2_boxcond",   1.0,  ("exp1", "box-cond", "nnU-Net")),
    ("SAM2",      "sam2",                 0.05, ("exp3", None, "SAM2")),
    ("SAM3",      "sam3",                 0.05, ("exp3", None, "SAM3")),
    ("MedSAM",    "medsam",               0.05, ("exp3", None, "MedSAM")),
    ("MedSAM-2",  "medsam2",              0.05, ("exp3", None, "MedSAM-2")),
    ("SAM2",      "sam2",                 1.0,  ("exp1", "foundation-f100", "SAM2")),
    ("SAM3",      "sam3",                 1.0,  ("exp1", "foundation-f100", "SAM3")),
    ("MedSAM",    "medsam",               1.0,  ("exp1", "foundation-f100", "MedSAM")),
    ("MedSAM-2",  "medsam2",              1.0,  ("exp1", "foundation-f100", "MedSAM-2")),
]

logger = logging.getLogger(__name__)


def _cross_dsc(key: str, frac: float) -> Optional[float]:
    """Mean DSC over the twelve transfer cells, from the per-image files."""
    vals: List[float] = []
    for src in DATASETS:
        for tgt in DATASETS:
            if src == tgt:
                continue
            if key in ("unet_resnet50", "transunet_r50_vitb16"):
                p = CNN_DIR / f"{key}_src{src}_tgt{tgt}_per_image.csv"
            elif key == "nnunet_v2_boxcond":
                p = NNUNET_DIR / f"{key}_src{src}_tgt{tgt}_per_image.csv"
            else:
                name = f"exp4_5_{key}_src{src}_tgt{tgt}_f{frac}/per_image_results.csv"
                p = LORA_NEW / name
                if not p.exists():
                    p = LORA_OLD / name
            if p.exists():
                vals.append(float(pd.read_csv(p)["dice"].mean()))
    return float(np.mean(vals)) if vals else None


def _diagonal(spec, frac: float) -> Dict[str, Optional[float]]:
    """In-domain reference: the four same-dataset cells, from the corrected tables."""
    source, block, model = spec
    path = E1 if source == "exp1" else E3
    if not path.exists():
        return {"dsc": None, "hd95": None}
    df = pd.read_csv(path)
    g = df[(df.model == model)] if source == "exp3" else df[(df.block == block) & (df.model == model)]
    if source == "exp3":
        g = g[np.isclose(g.fraction, frac)]
    if g.empty:
        return {"dsc": None, "hd95": None}
    return {"dsc": float(g.dsc_mean.mean()), "hd95": float(g.hd95.mean())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if not HD95_CSV.exists():
        sys.exit(f"missing {HD95_CSV}; run build_exp4_hd95_table.py first")
    hd = pd.read_csv(HD95_CSV)
    OUT.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for name, key, frac, spec in ROWS:
        sub = hd[(hd.model == key) & (np.isclose(hd.fraction, frac))]
        off = sub[sub.source != sub.target]
        cross_hd = float(off.hd95_impute_aware.mean()) if not off.empty else None
        diag = _diagonal(spec, frac)
        cross_dsc = _cross_dsc(key, frac)
        rows.append({
            "model": name, "fraction": frac,
            "indom_dsc": diag["dsc"], "cross_dsc": cross_dsc,
            "delta_dsc": None if None in (diag["dsc"], cross_dsc) else diag["dsc"] - cross_dsc,
            "indom_hd95": diag["hd95"], "cross_hd95": cross_hd,
            "delta_hd95": None if None in (diag["hd95"], cross_hd) else cross_hd - diag["hd95"],
            "n_cross_cells": int(len(off)),
            "n_empty_cross": int(off.n_empty.sum()) if not off.empty else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp4_summary.csv", index=False)

    print(f"{'model':<12}{'f':>6}{'In-dom DSC':>12}{'Cross':>8}{'Δ':>8}"
          f"{'In-dom HD95':>13}{'Cross':>8}{'Δ':>8}{'cells':>7}{'empty':>7}")
    for r in df.itertuples():
        f = lambda v, d=3: "  n/a" if pd.isna(v) else f"{v:.{d}f}"
        print(f"{r.model:<12}{r.fraction:>6}{f(r.indom_dsc):>12}{f(r.cross_dsc):>8}"
              f"{f(r.delta_dsc):>8}{f(r.indom_hd95,2):>13}{f(r.cross_hd95,2):>8}"
              f"{f(r.delta_hd95,2):>8}{r.n_cross_cells:>7}{r.n_empty_cross:>7}")
    print(f"\nwrote {OUT}/exp4_summary.csv")


if __name__ == "__main__":
    main()
