"""Rebuild every Experiment 1 cell under one HD95 definition and a jitter-free prompt.

Two measurement bugs were found after the first draft of Table 1 was written:

1. **Two HD95 implementations.** ``src/evaluation/metrics.py:compute_hd95`` takes
   distances from every mask pixel, so interior pixels contribute 0 and the 95th
   percentile lands inside that zero mass. Only the nnU-Net paths eroded to the
   contour first. The published HD95 column therefore scored nnU-Net under a
   stricter metric than the other seven models.
2. **Evaluation-time box jitter.** ``FoundationLoRAWrapper`` gated its training
   jitter on ``self.training``, a flag no call site ever cleared, so every
   foundation-LoRA test score was measured with a +/-10 % perturbed box.

Both are fixed at the source. This script does no inference: it reads the
per-image CSVs produced by the re-evaluations and emits the corrected table
cells plus the paired tests, so the numbers in the manuscript can be traced to
one command.

Sources per block
-----------------
image-only    U-Net/TransUNet : experiments/exp1_fullsupervised/traditional_imageonly/results/hd95_contour/
              nnU-Net         : experiments/exp1_fullsupervised/traditional_imageonly/results/nnunet/{ds}/results/
box-cond      U-Net/TransUNet : experiments/exp1_fullsupervised/traditional_boxcond/results/hd95_contour/
              nnU-Net         : experiments/exp1_fullsupervised/traditional_boxcond/results/nnunet/
foundation    f=100 % LoRA    : experiments/exp1_fullsupervised/foundation_lora/results/fullsup_{m}_{ds}_hd95b/

HD95 policy: an empty prediction has no defined surface distance, so it is
imputed to the image diagonal (sqrt(2)*512 = 724.08 px) before averaging. The
image-only nnU-Net evaluator writes 512.0 for empty predictions instead of inf,
so that sentinel is remapped as well (verified: no row with dice > 0 carries it).

Outputs (results/stats_exp1_corrected/)
---------------------------------------
    exp1_cells.csv      one row per (block, dataset, model): DSC mean/SD, IoU,
                        Precision, Recall, HD95 impute-aware, n, n_empty
    exp1_wilcoxon.csv   paired Wilcoxon vs the box-conditioned U-Net, per
                        dataset x metric x block, Holm-corrected within family

Usage
-----
    python scripts/stats/build_exp1_corrected.py
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
OUT_DIR = REPO / "results/stats_exp1_corrected"

HD95_CAP = float(np.sqrt(2) * 512)  # 724.0773 px
NNUNET_EMPTY_SENTINEL = 512.0

DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
CNN = {"unet_resnet50": "U-Net", "transunet_r50_vitb16": "TransUNet"}
FOUNDATION = {"sam2": "SAM2", "sam3": "SAM3", "medsam": "MedSAM", "medsam2": "MedSAM-2"}

logger = logging.getLogger(__name__)


def _impute(hd: pd.Series, dice: pd.Series, nnunet: bool = False) -> pd.Series:
    """Empty prediction -> image diagonal, so a collapse cannot vanish from the mean."""
    out = pd.to_numeric(hd, errors="coerce").replace([np.inf, -np.inf], HD95_CAP)
    if nnunet:
        # The plain evaluator caps empty predictions at the image side, not the
        # diagonal; those rows are exactly the ones that scored dice == 0.
        out = out.mask((dice <= 0) & np.isclose(out, NNUNET_EMPTY_SENTINEL), HD95_CAP)
    return out.fillna(HD95_CAP)


def _cell(df: pd.DataFrame, block: str, dataset: str, model: str) -> Dict[str, object]:
    return {
        "block": block, "dataset": dataset, "model": model, "n": len(df),
        # A zero-overlap prediction is not necessarily an empty one; use the
        # recorded flag where the source carries it.
        "n_empty": int(df["pred_empty"].sum()) if "pred_empty" in df
                   else int((df["hd95_final"] >= HD95_CAP).sum()),
        "dsc_mean": float(df["dice"].mean()), "dsc_sd": float(df["dice"].std(ddof=1)),
        "iou": float(df["iou"].mean()) if "iou" in df else np.nan,
        "precision": float(df["precision"].mean()) if "precision" in df else np.nan,
        "recall": float(df["recall"].mean()) if "recall" in df else np.nan,
        "hd95": float(df["hd95_final"].mean()),
        "hd95_median": float(df["hd95_final"].median()),
        # HD95 is strongly right-skewed (the diagonal imputation puts a few
        # images at 724 px), so mean +/- SD would print negative distances on
        # roughly a fifth of the cells. Quartiles describe the bulk honestly.
        "hd95_q1": float(df["hd95_final"].quantile(0.25)),
        "hd95_q3": float(df["hd95_final"].quantile(0.75)),
    }


def _load_contour_cnn(root: Path, model: str, dataset: str) -> Optional[pd.DataFrame]:
    """Re-scored U-Net/TransUNet cell: contour HD95 alongside the original DSC."""
    p = root / f"{model}_{dataset}_per_image.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["hd95_final"] = _impute(df["hd95_contour"], df["dice"])
    return df


def _load_nnunet(p: Path) -> Optional[pd.DataFrame]:
    """nnU-Net cell — already contour-based; only the empty sentinel is remapped."""
    if not p.exists():
        return None
    df = pd.read_csv(p).rename(columns={"original_filename": "filename"})
    df["hd95_final"] = _impute(df["hd95"], df["dice"], nnunet=True)
    return df


def _load_foundation(model: str, dataset: str) -> Optional[pd.DataFrame]:
    """Foundation f=100 % cell, re-evaluated with a tight box and contour HD95."""
    p = (REPO / "experiments/exp1_fullsupervised/foundation_lora/results"
         / f"fullsup_{model}_{dataset}_hd95b/per_image_results.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["hd95_final"] = _impute(df["hd95"], df["dice"])
    return df


def collect() -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], pd.DataFrame]]:
    """All Exp-1 cells as one table, plus the per-image frames for the paired tests."""
    rows: List[Dict[str, object]] = []
    frames: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    io_root = REPO / "experiments/exp1_fullsupervised/traditional_imageonly/results/hd95_contour"
    bc_root = REPO / "experiments/exp1_fullsupervised/traditional_boxcond/results/hd95_contour"

    for ds in DATASETS:
        for key, label in CNN.items():
            for block, root in (("image-only", io_root), ("box-cond", bc_root)):
                df = _load_contour_cnn(root, key, ds)
                if df is None:
                    logger.warning("missing %s %s/%s", block, label, ds)
                    continue
                rows.append(_cell(df, block, ds, label))
                frames[(block, ds, label)] = df

        df = _load_nnunet(REPO / "experiments/exp1_fullsupervised/traditional_imageonly/results/nnunet"
                          / ds / "results" / f"nnunet_results_{ds}_details.csv")
        if df is not None:
            rows.append(_cell(df, "image-only", ds, "nnU-Net"))
            frames[("image-only", ds, "nnU-Net")] = df

        df = _load_nnunet(REPO / "experiments/exp1_fullsupervised/traditional_boxcond/results/nnunet"
                          / f"nnunet_boxcond_results_{ds}_details.csv")
        if df is not None:
            rows.append(_cell(df, "box-cond", ds, "nnU-Net"))
            frames[("box-cond", ds, "nnU-Net")] = df

        for key, label in FOUNDATION.items():
            df = _load_foundation(key, ds)
            if df is None:
                logger.warning("missing foundation %s/%s", label, ds)
                continue
            rows.append(_cell(df, "foundation-f100", ds, label))
            frames[("foundation-f100", ds, label)] = df

    return pd.DataFrame(rows), frames


def paired_tests(frames: Dict[Tuple[str, str, str], pd.DataFrame]) -> pd.DataFrame:
    """Wilcoxon vs the box-conditioned U-Net, paired by filename, Holm within family.

    A family is one (dataset, metric, block): the contrasts a reader compares
    side by side in one block of the supplementary table.
    """
    records: List[Dict[str, object]] = []
    for ds in DATASETS:
        pivot = frames.get(("box-cond", ds, "U-Net"))
        if pivot is None:
            continue
        for block in ("box-cond", "foundation-f100"):
            for metric, col in (("DSC", "dice"), ("HD95", "hd95_final")):
                family: List[Dict[str, object]] = []
                for (b, d, m), df in frames.items():
                    if b != block or d != ds or m == "U-Net":
                        continue
                    merged = pivot[["filename", col]].merge(
                        df[["filename", col]], on="filename", suffixes=("_ref", "_x"))
                    if len(merged) < 10:
                        logger.warning("only %d paired rows for %s/%s/%s",
                                       len(merged), ds, block, m)
                        continue
                    a, b_ = merged[f"{col}_ref"], merged[f"{col}_x"]
                    if np.allclose(a, b_):
                        stat, p = np.nan, 1.0
                    else:
                        stat, p = wilcoxon(a, b_)
                    better = (b_.mean() > a.mean()) if metric == "DSC" else (b_.mean() < a.mean())
                    family.append({
                        "dataset": ds, "block": block, "metric": metric, "model": m,
                        "n_paired": len(merged), "mean_ref": float(a.mean()),
                        "mean_model": float(b_.mean()), "better_than_unet": bool(better),
                        "stat": stat, "p_raw": float(p),
                    })
                # Holm-Bonferroni inside the family.
                order = np.argsort([r["p_raw"] for r in family])
                k = len(family)
                running = 0.0
                for rank, idx in enumerate(order):
                    adj = min(1.0, family[idx]["p_raw"] * (k - rank))
                    running = max(running, adj)  # Holm is monotone in rank
                    family[idx]["p_holm"] = running
                records.extend(family)
    return pd.DataFrame(records)


def _stars(p: float, better: bool) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    mark = "\\ast" if better else "\\dagger"
    n = 3 if p < 0.001 else 2 if p < 0.01 else 1
    return "$^{" + mark * n + "}$"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells, frames = collect()
    cells.to_csv(OUT_DIR / "exp1_cells.csv", index=False)
    tests = paired_tests(frames)
    tests["marks"] = [_stars(r.p_holm, r.better_than_unet) for r in tests.itertuples()]
    tests.to_csv(OUT_DIR / "exp1_wilcoxon.csv", index=False)

    for block in ("image-only", "box-cond", "foundation-f100"):
        sub = cells[cells.block == block]
        if sub.empty:
            continue
        print(f"\n=== {block} ===")
        print(sub.pivot(index="model", columns="dataset",
                        values=["dsc_mean", "hd95"]).round(4).to_string())
    print(f"\nwrote {OUT_DIR}/exp1_cells.csv and exp1_wilcoxon.csv")


if __name__ == "__main__":
    main()
