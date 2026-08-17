"""Collate the few-shot LoRA runs into one table.

Reads results/fewshot_{model}_{dataset}_f{fraction}/summary_metrics.csv and
writes results/exp3_summary.csv, one row per (model, dataset, fraction) cell.
The f=100% end of the same curve is trained in exp1_fullsupervised/foundation_lora
and is not collected here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
OUT = RESULTS / "exp3_summary.csv"

MODELS = ["medsam", "medsam2", "sam2", "sam3"]
DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
FRACTIONS = ["0.05", "0.1", "0.25", "0.5"]


def main() -> None:
    rows: list[dict] = []
    missing: list[str] = []
    for model in MODELS:
        for dataset in DATASETS:
            for frac in FRACTIONS:
                p = RESULTS / f"fewshot_{model}_{dataset}_f{frac}" / "summary_metrics.csv"
                if not p.exists():
                    missing.append(f"{model}/{dataset}/f{frac}")
                    continue
                row = pd.read_csv(p).iloc[0].to_dict()
                row.update(model=model, dataset=dataset, fraction=float(frac))
                rows.append(row)

    if not rows:
        raise SystemExit(f"no summaries under {RESULTS}; run run.py first")

    df = pd.DataFrame(rows).sort_values(["model", "dataset", "fraction"])
    lead = ["model", "dataset", "fraction"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    total = len(MODELS) * len(DATASETS) * len(FRACTIONS)
    print(f"wrote {OUT} ({len(df)} of {total} cells)")
    if missing:
        print(f"missing {len(missing)}: " + ", ".join(missing[:8])
              + (" ..." if len(missing) > 8 else ""))


if __name__ == "__main__":
    main()
