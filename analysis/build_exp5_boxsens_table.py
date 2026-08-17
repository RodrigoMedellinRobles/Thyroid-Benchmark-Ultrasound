"""Rebuild the Experiment 5 box-prompt sensitivity table from the per-cell CSVs.

Why this exists
---------------
The LoRA rows of ``tab:boxsens`` were evaluated while
``FoundationLoRAWrapper`` still applied its training-time +/-10 % box jitter at
inference, so the column labelled "Tight" was really jitter 10 % and "Jitter
50 %" was jitter 50 % stacked on jitter 10 %. The CNN rows and the zero-shot
regime reach the model through other code paths and were never affected, which
means the published table compared the two families at unequal prompt quality.

``run_boxsens.py`` writes one CSV per (regime, model, dataset, fraction) with one
row per perturbation mode. This script only aggregates those files into the
table's two blocks; it runs no inference. Re-run the sweep first.

Aggregation, matching the published table
-----------------------------------------
Static block  unweighted mean of the DDTI, TN3K and ThyroidXL per-cell means
Stanford      reported on its own, cine-clip frames not pooled with the rest
DeltaFloor    tight DSC minus the box-fill floor at the tight box

The chosen convention is verified against the pre-fix backup: recomputing the
published rows from ``results.doublejitter_backup/`` must reproduce the numbers
currently in the paper, otherwise the aggregation here is not the one that was
used and the comparison would be meaningless.

Usage
-----
    python analysis/build_exp5_boxsens_table.py
    python analysis/build_exp5_boxsens_table.py --check-backup
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
RESULTS = REPO / "experiments/exp5_boxsens/results"
BACKUP = REPO / "experiments/exp5_boxsens/results.doublejitter_backup"
OUT_DIR = REPO / "results/stats_exp5_corrected"

STATIC = ["ddti", "tn3k", "thyroidxl"]
STANFORD = "stanford_aimi"
MODES = ["tight", "jitter10", "jitter25", "jitter50"]
MODELS = {"sam2": "SAM2", "sam3": "SAM3", "medsam": "MedSAM", "medsam2": "MedSAM-2"}
FRACTIONS = {"0.05": "LoRA $f{=}5\\%$", "1": "LoRA $f{=}100\\%$"}

logger = logging.getLogger(__name__)


def _cell(root: Path, model: str, dataset: str, frac: str) -> Optional[pd.DataFrame]:
    p = root / f"lora_{model}_{dataset}_f{frac}_boxsens.csv"
    return pd.read_csv(p) if p.exists() else None


def _mode_value(df: pd.DataFrame, mode: str, col: str) -> float:
    row = df[df["mode"] == mode]
    return float(row.iloc[0][col]) if not row.empty else float("nan")


def block(root: Path, datasets: List[str]) -> pd.DataFrame:
    """One row per (fraction, model): the mean DSC per mode plus DeltaFloor."""
    rows: List[Dict[str, object]] = []
    for frac in FRACTIONS:
        for key, label in MODELS.items():
            per_mode: Dict[str, List[float]] = {m: [] for m in MODES}
            floors: List[float] = []
            missing: List[str] = []
            for ds in datasets:
                df = _cell(root, key, ds, frac)
                if df is None:
                    missing.append(ds)
                    continue
                for m in MODES:
                    per_mode[m].append(_mode_value(df, m, "dice"))
                floors.append(_mode_value(df, "tight", "floor_dice"))
            if missing:
                logger.warning("missing %s f=%s: %s", label, frac, ", ".join(missing))
            r: Dict[str, object] = {"fraction": frac, "model": label,
                                    "n_datasets": len(datasets) - len(missing)}
            for m in MODES:
                r[m] = float(np.mean(per_mode[m])) if per_mode[m] else float("nan")
            r["floor"] = float(np.mean(floors)) if floors else float("nan")
            r["delta_floor"] = r["tight"] - r["floor"]
            rows.append(r)
    return pd.DataFrame(rows)


def latex_rows(df: pd.DataFrame) -> str:
    out: List[str] = []
    for frac, label in FRACTIONS.items():
        g = df[df.fraction == frac]
        out.append(f"\\multirow{{4}}{{*}}{{{label}}}")
        for r in g.itertuples():
            out.append(f"  & {r.model} & {r.tight:.3f} & {r.jitter10:.3f} & "
                       f"{r.jitter25:.3f} & {r.jitter50:.3f} & "
                       f"{r.delta_floor:+.3f} \\\\")
        out.append("\\midrule")
    return "\n".join(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-backup", action="store_true",
                    help="also print the pre-fix numbers, to confirm the aggregation")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, datasets in (("static", STATIC), ("stanford", [STANFORD])):
        new = block(RESULTS, datasets)
        new.insert(0, "block", name)
        new.to_csv(OUT_DIR / f"exp5_boxsens_{name}.csv", index=False)
        print(f"\n===== {name} =====")
        print(latex_rows(new))
        if args.check_backup and BACKUP.exists():
            old = block(BACKUP, datasets)
            print(f"--- pre-fix ({name}), for comparison ---")
            print(latex_rows(old))
            d = new.set_index(["fraction", "model"]).tight - old.set_index(["fraction", "model"]).tight
            print(f"tight-DSC shift: {d.min():+.4f} to {d.max():+.4f} (mean {d.mean():+.4f})")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
