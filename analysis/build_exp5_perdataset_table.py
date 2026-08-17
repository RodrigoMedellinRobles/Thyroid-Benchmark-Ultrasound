"""Emit the per-dataset box-sensitivity table for the supplementary.

The main-text table collapses DDTI, TN3K and ThyroidXL into one static mean, so
nothing in the paper lets a reader see whether a model's box robustness holds
across datasets or is carried by one of them. This writes that breakdown.

Sources are the same CSVs the main table aggregates:
    experiments/exp5_boxsens/results/zeroshot_{model}_{dataset}_boxsens.csv
    experiments/exp5_boxsens/results/lora_{model}_{dataset}_f{0.05,1}_boxsens.csv
    experiments/exp5_boxsens/results/nnunet_{dataset}_boxcond_boxsens.csv
    experiments/exp1_fullsupervised/traditional_boxcond/results/{unet,transunet}/{dataset}_{arch}_box_sensitivity.csv

Usage
-----
    python scripts/stats/build_exp5_perdataset_table.py > /tmp/table.tex
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BOX = REPO / "experiments/exp5_boxsens/results"
CNN = REPO / "experiments/exp1_fullsupervised/traditional_boxcond/results"

DATASETS = [("ddti", "DDTI"), ("tn3k", "TN3K"), ("thyroidxl", "ThyroidXL"),
            ("stanford_aimi", "Stanford AIMI")]
MODES = ["tight", "jitter10", "jitter25", "jitter50"]
ZS = [("sam", "SAM"), ("sam2", "SAM2"), ("sam3", "SAM3"),
      ("medsam", "MedSAM"), ("medsam2", "MedSAM-2")]
LORA = [("sam2", "SAM2"), ("sam3", "SAM3"), ("medsam", "MedSAM"),
        ("medsam2", "MedSAM-2")]


def _read(path: Path, mode_col: str, dice_col: str) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out: Dict[str, float] = {}
    for m in MODES:
        r = df[df[mode_col] == m]
        if r.empty:
            return None
        out[m] = float(r.iloc[0][dice_col])
    return out


def row(label: str, vals: Optional[Dict[str, float]]) -> str:
    if vals is None:
        return f"  & {label} & \\multicolumn{{4}}{{c}}{{--}} \\\\"
    cells = " & ".join(f"{vals[m]:.3f}" for m in MODES)
    return f"  & {label} & {cells} \\\\"


def main() -> None:
    lines: List[str] = []
    for ds, ds_label in DATASETS:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{ds_label}}}}}\\\\")
        lines.append("\\midrule")

        lines.append("\\multirow{5}{*}{Zero-shot}")
        for key, label in ZS:
            lines.append(row(label, _read(BOX / f"zeroshot_{key}_{ds}_boxsens.csv",
                                          "mode", "dice")))
        for frac, tag in (("0.05", "LoRA $f{=}5\\%$"), ("1", "LoRA $f{=}100\\%$")):
            lines.append("\\cmidrule(l){2-6}")
            lines.append(f"\\multirow{{4}}{{*}}{{{tag}}}")
            for key, label in LORA:
                lines.append(row(label, _read(
                    BOX / f"lora_{key}_{ds}_f{frac}_boxsens.csv", "mode", "dice")))

        lines.append("\\cmidrule(l){2-6}")
        lines.append("\\multirow{3}{*}{Box-cond CNN}")
        for arch, label in (("unet", "U-Net"), ("transunet", "TransUNet")):
            lines.append(row(label, _read(
                CNN / arch / f"{ds}_{arch}_box_sensitivity.csv",
                "eval_box_mode", "model_dice")))
        lines.append(row("nnU-Net v2", _read(
            BOX / f"nnunet_{ds}_boxcond_boxsens.csv", "eval_box_mode", "model_dice")))

    print("\n".join(lines))
    missing = sum(1 for l in lines if "multicolumn{4}" in l)
    print(f"% {missing} cells unavailable", file=sys.stderr)


if __name__ == "__main__":
    main()
