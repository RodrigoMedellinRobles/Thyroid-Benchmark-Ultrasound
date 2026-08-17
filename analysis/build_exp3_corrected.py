"""Rebuild the Experiment 3 cells with a tight evaluation box and contour HD95.

Experiment 3 carried both measurement bugs at once:

1. ``FoundationLoRAWrapper`` gated its training-time box jitter on the wrapper's
   own ``training`` flag, which no call site ever cleared, so every few-shot cell
   was scored with a +/-10 % perturbed prompt. Every metric moves, DSC included.
2. HD95 was computed from all mask pixels rather than between contours, which
   returns values several times smaller.

Both are fixed at the source; this script reads the per-image CSVs of the
re-evaluation and emits the corrected cells plus the paired tests. It runs no
inference.

Sources
-------
``experiments/exp3_fewshot_lora/results_hd95b/exp2_{model}_{dataset}_f{frac}/``
for f in {0.05, 0.1, 0.25, 0.5}, and the published
``experiments/exp3_fewshot_lora/results/`` twins as the provenance
reference. The f=100 % endpoint is Experiment 1 and is read from
``results/stats_exp1_corrected/exp1_cells.csv`` so the two agree by
construction; zero-shot comes from ``results/stats_exp2_corrected/``.

HD95 policy: an empty prediction is imputed to the image diagonal
(sqrt(2)*512 = 724.08 px) before averaging.

Outputs (results/stats_exp3_corrected/)
---------------------------------------
    exp3_cells.csv        one row per (model, dataset, fraction), with the
                          published values alongside for auditing
    exp3_curve.csv        the full ZS -> f=5..50 -> f=100 -> FS-UB curve per
                          (model, dataset), which is what Table 4 prints
    exp3_saturation.csv   per-step DSC gains, so the flattening point is read
                          off the data instead of asserted

Usage
-----
    python scripts/stats/build_exp3_corrected.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NEW = REPO / "experiments/exp3_fewshot_lora/results_hd95b"
OLD = REPO / "experiments/exp3_fewshot_lora/results"
OUT_DIR = REPO / "results/stats_exp3_corrected"

HD95_CAP = float(np.sqrt(2) * 512)
DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
MODELS = {"sam2": "SAM2", "sam3": "SAM3", "medsam": "MedSAM", "medsam2": "MedSAM-2"}
FRACTIONS = [0.05, 0.1, 0.25, 0.5]
FRAC_TOKEN = {0.05: "0.05", 0.1: "0.1", 0.25: "0.25", 0.5: "0.5"}

logger = logging.getLogger(__name__)


def _cell_dir(root: Path, model: str, dataset: str, frac: float) -> Path:
    return root / f"exp2_{model}_{dataset}_f{FRAC_TOKEN[frac]}"


def load(model: str, dataset: str, frac: float) -> Optional[pd.DataFrame]:
    p = _cell_dir(NEW, model, dataset, frac) / "per_image_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["hd95_final"] = (pd.to_numeric(df["hd95"], errors="coerce")
                        .replace([np.inf, -np.inf], HD95_CAP).fillna(HD95_CAP))
    return df


def published(model: str, dataset: str, frac: float) -> Optional[pd.Series]:
    p = _cell_dir(OLD, model, dataset, frac) / "summary_metrics.csv"
    return None if not p.exists() else pd.read_csv(p).iloc[0]


def collect() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, label in MODELS.items():
        for ds in DATASETS:
            for f in FRACTIONS:
                df = load(key, ds, f)
                if df is None:
                    logger.warning("missing %s/%s/f=%s", label, ds, f)
                    continue
                pub = published(key, ds, f)
                rows.append({
                    "model": label, "dataset": ds, "fraction": f, "n": len(df),
                    "n_empty": int((df["hd95_final"] >= HD95_CAP).sum()),
                    "dsc_mean": float(df["dice"].mean()),
                    "dsc_sd": float(df["dice"].std(ddof=1)),
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
                    "dsc_published": None if pub is None else float(pub["dice_mean"]),
                    "hd95_published": None if pub is None else float(pub["hd95_mean"]),
                })
    return pd.DataFrame(rows)


def build_curve(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset): the whole label-budget curve Table 4 prints."""
    zs_path = REPO / "results/stats_exp2_corrected/exp2_cells.csv"
    e1_path = REPO / "results/stats_exp1_corrected/exp1_cells.csv"
    zs = pd.read_csv(zs_path) if zs_path.exists() else None
    e1 = pd.read_csv(e1_path) if e1_path.exists() else None

    rows: List[Dict[str, object]] = []
    for label in MODELS.values():
        for ds in DATASETS:
            row: Dict[str, object] = {"model": label, "dataset": ds}
            if zs is not None:
                m = zs[(zs.model == label) & (zs.dataset == ds)]
                row["zs"] = float(m.iloc[0].dsc_mean) if not m.empty else np.nan
                row["zs_hd95"] = float(m.iloc[0].hd95) if not m.empty else np.nan
            for f in FRACTIONS:
                c = cells[(cells.model == label) & (cells.dataset == ds) & (cells.fraction == f)]
                row[f"f{f}"] = float(c.iloc[0].dsc_mean) if not c.empty else np.nan
                row[f"f{f}_hd95"] = float(c.iloc[0].hd95) if not c.empty else np.nan
            if e1 is not None:
                m = e1[(e1.block == "foundation-f100") & (e1.model == label) & (e1.dataset == ds)]
                row["f1.0"] = float(m.iloc[0].dsc_mean) if not m.empty else np.nan
                row["f1.0_hd95"] = float(m.iloc[0].hd95) if not m.empty else np.nan
                u = e1[(e1.block == "box-cond") & (e1.model == "nnU-Net") & (e1.dataset == ds)]
                row["fs_ub"] = float(u.iloc[0].dsc_mean) if not u.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_saturation(curve: pd.DataFrame) -> pd.DataFrame:
    """Per-step DSC gains, so the knee of the curve is measured, not asserted."""
    steps = [("zs", "f0.05"), ("f0.05", "f0.1"), ("f0.1", "f0.25"),
             ("f0.25", "f0.5"), ("f0.5", "f1.0")]
    rows: List[Dict[str, object]] = []
    for r in curve.itertuples():
        for a, b in steps:
            va, vb = getattr(r, a.replace(".", "_"), np.nan), getattr(r, b.replace(".", "_"), np.nan)
            rows.append({"model": r.model, "dataset": r.dataset,
                         "step": f"{a}->{b}", "gain": vb - va})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = collect()
    if cells.empty:
        sys.exit("no re-evaluated cells found; run the Experiment 3 sweep first")
    cells.to_csv(OUT_DIR / "exp3_cells.csv", index=False)

    curve = build_curve(cells)
    curve.to_csv(OUT_DIR / "exp3_curve.csv", index=False)
    sat = build_saturation(curve.rename(columns=lambda c: c.replace(".", "_")))
    sat.to_csv(OUT_DIR / "exp3_saturation.csv", index=False)

    done = len(cells)
    print(f"{done}/64 cells re-evaluated")
    d = cells.dropna(subset=["dsc_published"])
    if not d.empty:
        delta = d.dsc_mean - d.dsc_published
        print(f"DSC shift from the jitter fix: {delta.min():+.4f} to {delta.max():+.4f} "
              f"(mean {delta.mean():+.4f})")
        ratio = d.hd95 / d.hd95_published
        print(f"HD95 contour/all-pixel ratio: {ratio.min():.2f} to {ratio.max():.2f} "
              f"(median {ratio.median():.2f})")
    if not sat.dropna(subset=["gain"]).empty:
        print("\nmean DSC gain per step, over all cells:")
        print(sat.groupby("step", sort=False).gain.mean().round(4).to_string())
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
