"""Collate the per-run zero-shot summaries into one table.

Reads results/{model}_{dataset}/summary_metrics.csv for every completed run and
writes results/zeroshot_summary.csv. Runs produced with --limit land in a
_limitN directory and are skipped, so a partial sweep can never be aggregated as
if it were a full test split.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
OUT = RESULTS / "zeroshot_summary.csv"

MODELS = ["sam", "sam2", "sam3", "medsam", "medsam2"]
DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]


def main() -> None:
    rows: list[dict] = []
    missing: list[str] = []
    for model in MODELS:
        for dataset in DATASETS:
            p = RESULTS / f"{model}_{dataset}" / "summary_metrics.csv"
            if not p.exists():
                missing.append(f"{model}/{dataset}")
                continue
            row = pd.read_csv(p).iloc[0].to_dict()
            row.update(model=model, dataset=dataset)
            rows.append(row)

    if not rows:
        raise SystemExit(f"no summaries under {RESULTS}; run run.py first")

    df = pd.DataFrame(rows)
    lead = [c for c in ("model", "dataset") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"wrote {OUT} ({len(df)} of {len(MODELS) * len(DATASETS)} cells)")
    if missing:
        print("missing: " + ", ".join(missing))
    skipped = sorted(d.name for d in RESULTS.glob("*_limit*") if d.is_dir())
    if skipped:
        print("skipped partial runs: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
