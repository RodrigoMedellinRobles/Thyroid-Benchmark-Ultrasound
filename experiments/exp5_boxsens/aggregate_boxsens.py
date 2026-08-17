"""Aggregate Exp 5 box-prompt sensitivity fragments into one tidy summary.

Four fragment families feed the summary, one CSV per model x dataset (x LoRA
fraction):

  * zeroshot_<model>_<dataset>_boxsens.csv          (regime=zeroshot)
  * lora_<model>_<dataset>_f<fraction>_boxsens.csv  (regime=lora)
  * nnunet_<dataset>_boxcond_boxsens.csv            (regime=nnunet_boxcond)
  * <dataset>_<model>_box_sensitivity.csv           (regime=cnn_boxcond, written
    by the Exp 1 box_eval.py and read from the Exp 1 results tree)

Each holds one row per perturbation mode (tight, jitter10, jitter25, jitter50,
shifted, adversarial). This script folds them into a single long table:

    results/boxsens_summary.csv
      regime, model, dataset, fraction, mode, n, dsc, iou, hd95,
      delta_dsc_over_floor

`fraction` is the share of the training split used to fit the LoRA adapter. It
is the axis that separates the LoRA rows of the reported table, so it must be
carried through: zero-shot rows get 0.0, the fully-supervised box-conditioned
CNNs get 1.0, and LoRA rows take the fraction parsed from the file name.

`delta_dsc_over_floor` is the tight-box DSC minus the trivial box-fill floor,
the check that a model segments from the prompt rather than reproducing the box.

nnU-Net and CNN box-sensitivity fragments carry no HD95 column; that field is
left blank for those rows. Input discovery uses a glob, so a missing fragment is
simply absent from the summary rather than raising. If no fragment at all is
found the existing summary is left untouched, so running this before the sweep
cannot blank out a summary you already built.

Usage:
    python experiments/exp5_boxsens/aggregate_boxsens.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_FRAC_RE = re.compile(r"_f([0-9.]+)_boxsens\.csv$")


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fraction_from_name(path: str) -> float | None:
    """LoRA fragments tag their training fraction in the file name."""
    m = _FRAC_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else None


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"[boxsens] ERROR: results dir missing: {RESULTS_DIR}", file=sys.stderr)
        return 1

    out_rows: list[dict] = []

    # zeroshot / lora fragments share the same schema.
    for path in sorted(glob.glob(str(RESULTS_DIR / "zeroshot_*_boxsens.csv"))) + \
                sorted(glob.glob(str(RESULTS_DIR / "lora_*_boxsens.csv"))):
        fraction = _fraction_from_name(path)
        for r in _read_rows(path):
            if r["regime"] == "zeroshot":
                frac = 0.0
            elif fraction is None:
                # A LoRA fragment with no fraction tag is ambiguous: it cannot be
                # placed in the table, so skip it rather than guess.
                continue
            else:
                frac = fraction
            out_rows.append({
                "regime": r["regime"],
                "model": r["model"],
                "dataset": r["dataset"],
                "fraction": frac,
                "mode": r["mode"],
                "n": int(float(r["n"])),
                "dsc": float(r["dice"]),
                "iou": float(r["iou"]),
                "hd95": float(r["hd95"]),
                "delta_dsc_over_floor": float(r["delta_dice_over_floor"]),
            })

    # nnU-Net box-conditioned fragments use a different column layout.
    for path in sorted(glob.glob(str(RESULTS_DIR / "nnunet_*_boxcond_boxsens.csv"))):
        for r in _read_rows(path):
            out_rows.append({
                "regime": "nnunet_boxcond",
                "model": r["model"],
                "dataset": r["dataset"],
                "fraction": 1.0,
                "mode": r["eval_box_mode"],
                "n": int(float(r["n_images"])),
                "dsc": float(r["model_dice"]),
                "iou": float(r["model_iou"]),
                "hd95": "",
                "delta_dsc_over_floor": float(r["delta_dice"]),
            })

    # Box-conditioned U-Net / TransUNet box-sensitivity are produced by the Exp 1
    # box_eval.py and live under the Exp 1 results tree; fold them in as regime
    # "cnn_boxcond" so this summary carries every box-conditioned CNN row of the
    # box-sensitivity table.
    _cnn_dir = Path(__file__).resolve().parents[1] / \
        "exp1_fullsupervised" / "traditional_boxcond" / "results"
    for sub in ("unet", "transunet"):
        for path in sorted(glob.glob(str(_cnn_dir / sub / "*_box_sensitivity.csv"))):
            for r in _read_rows(path):
                out_rows.append({
                    "regime": "cnn_boxcond",
                    "model": sub,
                    "dataset": r["dataset"],
                    "fraction": 1.0,
                    "mode": r["eval_box_mode"],
                    "n": int(float(r["n_images"])),
                    "dsc": float(r["model_dice"]),
                    "iou": float(r["model_iou"]),
                    "hd95": "",
                    "delta_dsc_over_floor": float(r["delta_dice"]),
                })

    if not out_rows:
        print("[boxsens] no fragments found; existing summary left untouched")
        return 0

    out_path = RESULTS_DIR / "boxsens_summary.csv"
    fields = ["regime", "model", "dataset", "fraction", "mode", "n",
              "dsc", "iou", "hd95", "delta_dsc_over_floor"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[boxsens] wrote {out_path.name}  rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
