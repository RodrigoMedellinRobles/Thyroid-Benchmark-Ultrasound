"""Rebuild the Experiment 4 cross-dataset HD95 table from the per-image CSVs.

Why this exists
---------------
``experiments/exp4_crossdataset/traditional/results/exp4_hd95_impute_aware_recompute.csv`` is
the single source for the cross-dataset forest figure and for both HD95
longtables of the supplementary, but nothing in the repository wrote it: it was
committed once by hand. This script reproduces it from the per-image results, so
the figure and the tables can be regenerated after a re-evaluation.

Two policies matter here and both were previously applied inconsistently.

*Empty predictions.* A prediction with no foreground has no surface distance.
The per-image CSVs record it as a blank or a non-finite value. Averaging only
the finite entries silently removes a model's worst failures, so each blank is
imputed to the image diagonal (sqrt(2)*512 = 724.0773 px) before the per-cell
mean. Both columns are emitted, ``hd95_finite_only`` and ``hd95_impute_aware``,
so the difference stays auditable; the paper uses the impute-aware one.

*One HD95 definition.* Only the nnU-Net path ever eroded masks to their contour;
every other path measured from all mask pixels and reported values several times
smaller. The re-evaluation under ``scripts/hd95_contour/run_with_contour_hd95.py``
puts the rest on the same definition. This script does not compute HD95, it only
aggregates whatever the per-image files hold, so run it after that sweep.

Sources
-------
CNN cross          experiments/exp4_crossdataset/traditional/results_boxcond_cnn/
nnU-Net cross      experiments/exp4_crossdataset/traditional/results_boxcond_nnunet/
foundation cross   experiments/exp4_crossdataset/foundation/results_hd95b/ (falls back
                   to results/ for any cell not yet re-evaluated, and says so)

Usage
-----
    python analysis/build_exp4_hd95_table.py
    python analysis/build_exp4_hd95_table.py --out /tmp/check.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CNN_DIR = REPO / "experiments/exp4_crossdataset/traditional/results_boxcond_cnn"
NNUNET_DIR = REPO / "experiments/exp4_crossdataset/traditional/results_boxcond_nnunet"
LORA_NEW = REPO / "experiments/exp4_crossdataset/foundation/results_hd95b"
LORA_OLD = REPO / "experiments/exp4_crossdataset/foundation/results"
DEFAULT_OUT = REPO / "experiments/exp4_crossdataset/traditional/results/exp4_hd95_impute_aware_recompute.csv"

HD95_CAP = float(np.sqrt(2) * 512)  # 724.0773 px
DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
CNN_MODELS = ["unet_resnet50", "transunet_r50_vitb16"]
NNUNET_MODEL = "nnunet_v2_boxcond"
LORA_MODELS = ["sam2", "sam3", "medsam", "medsam2"]
LORA_FRACTIONS = [0.05, 1.0]

logger = logging.getLogger(__name__)


def _summarise(df: pd.DataFrame, col: str = "hd95") -> Dict[str, float]:
    """Per-cell HD95 under both policies, plus the counts behind them."""
    v = pd.to_numeric(df[col], errors="coerce")
    finite = v[np.isfinite(v)]
    imputed = v.copy()
    imputed[~np.isfinite(imputed)] = HD95_CAP
    return {
        "n_images": int(len(v)),
        "n_empty": int((~np.isfinite(v)).sum()),
        "hd95_finite_only": float(finite.mean()) if len(finite) else float("nan"),
        "hd95_impute_aware": float(imputed.mean()),
    }


def _row(model: str, fraction: float, src: str, tgt: str,
         path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hd95" not in df.columns:
        logger.warning("no hd95 column in %s", path)
        return None
    return {"model": model, "fraction": fraction, "source": src, "target": tgt,
            **_summarise(df)}


def collect() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    fallback: List[str] = []

    for model in CNN_MODELS:
        for src in DATASETS:
            for tgt in DATASETS:
                r = _row(model, 1.0, src, tgt,
                         CNN_DIR / f"{model}_src{src}_tgt{tgt}_per_image.csv")
                rows.append(r) if r else missing.append(f"{model} {src}->{tgt}")

    for src in DATASETS:
        for tgt in DATASETS:
            r = _row(NNUNET_MODEL, 1.0, src, tgt,
                     NNUNET_DIR / f"{NNUNET_MODEL}_src{src}_tgt{tgt}_per_image.csv")
            if r is None and src == tgt:
                # The in-domain reference for nnU-Net is Experiment 1, not a transfer run.
                r = _row(NNUNET_MODEL, 1.0, src, tgt, REPO
                         / "experiments/exp1_fullsupervised/traditional_boxcond/results/nnunet"
                         / f"nnunet_boxcond_results_{tgt}_details.csv")
            rows.append(r) if r else missing.append(f"nnU-Net {src}->{tgt}")

    for model in LORA_MODELS:
        for frac in LORA_FRACTIONS:
            for src in DATASETS:
                for tgt in DATASETS:
                    name = f"exp4_5_{model}_src{src}_tgt{tgt}_f{frac}/per_image_results.csv"
                    r = _row(model, frac, src, tgt, LORA_NEW / name)
                    if r is None and src == tgt:
                        # In-domain reference: Experiment 3 at f=5 %, Experiment 1 at f=100 %.
                        diag = (REPO / "experiments/exp1_fullsupervised/foundation_lora/results"
                                / f"fullsup_{model}_{tgt}_hd95b/per_image_results.csv"
                                if frac == 1.0 else
                                REPO / "experiments/exp3_fewshot_lora/results_hd95b"
                                / f"exp2_{model}_{tgt}_f{frac}/per_image_results.csv")
                        r = _row(model, frac, src, tgt, diag)
                    if r is None:
                        r = _row(model, frac, src, tgt, LORA_OLD / name)
                        if r is not None:
                            fallback.append(f"{model} f={frac} {src}->{tgt}")
                    rows.append(r) if r else missing.append(f"{model} f={frac} {src}->{tgt}")

    if missing:
        logger.warning("%d cells missing: %s%s", len(missing), ", ".join(missing[:6]),
                       " ..." if len(missing) > 6 else "")
    if fallback:
        logger.warning("%d foundation cells fell back to the pre-fix directory: %s%s",
                       len(fallback), ", ".join(fallback[:6]),
                       " ..." if len(fallback) > 6 else "")
    return pd.DataFrame([r for r in rows if r])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = collect()
    if df.empty:
        sys.exit("no per-image files found")
    df = df.sort_values(["model", "fraction", "source", "target"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}  ({len(df)} cells)")
    off = df[df.source != df.target]
    print("\nmean cross-site HD95 (impute-aware), off-diagonal only:")
    for (m, f), g in off.groupby(["model", "fraction"]):
        drop = g.hd95_impute_aware.mean() - g.hd95_finite_only.mean()
        print(f"  {m:22} f={f:<5} {g.hd95_impute_aware.mean():7.2f} px "
              f"(finite-only {g.hd95_finite_only.mean():7.2f}, empties add {drop:+.2f}) "
              f"n_empty={int(g.n_empty.sum())}")


if __name__ == "__main__":
    main()
