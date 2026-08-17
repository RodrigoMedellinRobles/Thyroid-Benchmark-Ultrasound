#!/usr/bin/env python3
"""
Exp 1 statistical analysis — bootstrap 95% CI + pairwise Wilcoxon + Holm-Bonferroni.

Reads per-image DSC and HD95 from each (model, dataset) run produced by run.py
and writes:
    experiments/exp2_zeroshot/results/stats/bootstrap_ci95.csv
    experiments/exp2_zeroshot/results/stats/wilcoxon.csv
    experiments/exp2_zeroshot/results/stats/summary.txt

Usage:
    python experiments/exp2_zeroshot/compute_stats.py
    python experiments/exp2_zeroshot/compute_stats.py --n_boot 5000 --seed 42

The script is self-contained: no project imports beyond stdlib + numpy/pandas/scipy/statsmodels.
Run after all 5*4=20 zero-shot runs in results/<model>_<dataset>/per_image_results.csv exist.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUT_DIR = RESULTS_DIR / "stats"

MODELS = ("sam", "sam2", "sam3", "medsam", "medsam2")
DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_per_image(model: str, dataset: str) -> pd.DataFrame | None:
    """Load per_image_results.csv for one (model, dataset) cell.

    Returns DataFrame filtered to non-skipped rows with numeric DSC and HD95,
    or None if the run directory does not exist.
    """
    path = RESULTS_DIR / f"{model}_{dataset}" / "per_image_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["skipped"] == False].copy()
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df["hd95"] = pd.to_numeric(df["hd95"], errors="coerce")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float,
                 seed: int) -> tuple[float, float, float]:
    """Percentile-method bootstrap CI on the mean of `values`.

    Drops non-finite entries before resampling (handles HD95 inf for empty
    pred-vs-empty-GT pairs that the metrics layer leaves as NaN).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


# ----------------------------------------------------------------------------
# Wilcoxon helpers
# ----------------------------------------------------------------------------
def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def pairwise_wilcoxon_dsc(cells: dict[tuple[str, str], pd.DataFrame],
                          dataset: str, alpha: float) -> list[dict]:
    """All-vs-all Wilcoxon signed-rank on per-image DSC for one dataset.

    Pairs images by filename so two models compared on the same images.
    Holm-Bonferroni correction applied within the dataset over all 10 pairs.
    """
    rows = []
    pvals = []
    pair_keys = []

    for m_a, m_b in combinations(MODELS, 2):
        df_a = cells.get((m_a, dataset))
        df_b = cells.get((m_b, dataset))
        if df_a is None or df_b is None:
            continue

        merged = df_a[["filename", "dice"]].merge(
            df_b[["filename", "dice"]], on="filename", suffixes=("_a", "_b")
        )
        merged = merged.dropna(subset=["dice_a", "dice_b"])
        if len(merged) < 10:
            continue

        diffs = merged["dice_a"].to_numpy() - merged["dice_b"].to_numpy()
        if np.all(diffs == 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(diffs, alternative="two-sided",
                               zero_method="wilcox", correction=False)
        rows.append(dict(
            dataset=dataset, model_a=m_a, model_b=m_b,
            n_paired=len(merged),
            wilcoxon_W=float(stat) if np.isfinite(stat) else float("nan"),
            median_diff_a_minus_b=round(float(np.median(diffs)), 5),
            p_value=round(float(p), 6),
        ))
        pvals.append(p)
        pair_keys.append((m_a, m_b))

    if pvals:
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        for r, pa, rj in zip(rows, p_adj, reject):
            r["p_holm"] = round(float(pa), 6)
            r["reject_at_alpha_holm"] = bool(rj)
            r["stars"] = stars(pa)

    return rows


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap CIs + pairwise Wilcoxon for Exp 1 zero-shot."
    )
    ap.add_argument("--n_boot", type=int, default=1000,
                    help="Bootstrap resamples per cell (default: 1000).")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level for CIs and Holm correction.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for bootstrap.")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory (default: results/stats/ inside exp).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load all cells once ----------------------------------------
    cells: dict[tuple[str, str], pd.DataFrame] = {}
    missing: list[str] = []
    for ds in DATASETS:
        for m in MODELS:
            df = load_per_image(m, ds)
            if df is None or len(df) == 0:
                missing.append(f"{m}_{ds}")
                continue
            cells[(m, ds)] = df

    if missing:
        print(f"[WARN] missing or empty per_image_results.csv for: {missing}")

    # -------- Bootstrap CIs ----------------------------------------------
    boot_rows = []
    for ds in DATASETS:
        for m in MODELS:
            df = cells.get((m, ds))
            if df is None:
                continue
            dsc = df["dice"].dropna().to_numpy()
            hd95 = df["hd95"].dropna().to_numpy()
            dsc_mean, dsc_lo, dsc_hi = bootstrap_ci(
                dsc, args.n_boot, args.alpha, args.seed
            )
            hd95_mean, hd95_lo, hd95_hi = bootstrap_ci(
                hd95, args.n_boot, args.alpha, args.seed
            )
            boot_rows.append(dict(
                dataset=ds, model=m,
                n_dsc=len(dsc),
                dsc_mean=round(dsc_mean, 4),
                dsc_ci_low=round(dsc_lo, 4),
                dsc_ci_high=round(dsc_hi, 4),
                n_hd95=len(hd95),
                hd95_mean=round(hd95_mean, 3),
                hd95_ci_low=round(hd95_lo, 3),
                hd95_ci_high=round(hd95_hi, 3),
            ))

    boot_path = args.output_dir / "bootstrap_ci95.csv"
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    print(f"[OK] {boot_path} ({len(boot_rows)} cells)")

    # -------- Pairwise Wilcoxon ------------------------------------------
    wilcoxon_rows = []
    for ds in DATASETS:
        wilcoxon_rows.extend(pairwise_wilcoxon_dsc(cells, ds, args.alpha))

    wilcoxon_path = args.output_dir / "wilcoxon.csv"
    if wilcoxon_rows:
        cols = [
            "dataset", "model_a", "model_b", "n_paired",
            "wilcoxon_W", "median_diff_a_minus_b",
            "p_value", "p_holm", "reject_at_alpha_holm", "stars",
        ]
        pd.DataFrame(wilcoxon_rows)[cols].to_csv(wilcoxon_path, index=False)
        print(f"[OK] {wilcoxon_path} ({len(wilcoxon_rows)} pairwise tests)")

    # -------- Human-readable summary -------------------------------------
    summary_path = args.output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Exp 1 zero-shot — bootstrap CIs + pairwise Wilcoxon (Holm-Bonferroni)\n")
        f.write(f"alpha={args.alpha} n_boot={args.n_boot} seed={args.seed}\n")
        f.write(f"Cells loaded: {len(cells)} / {len(MODELS)*len(DATASETS)}\n")
        if missing:
            f.write(f"Missing: {', '.join(missing)}\n")
        f.write("\n=== Bootstrap mean DSC [95% CI] ===\n")
        for r in boot_rows:
            f.write(
                f"  {r['dataset']:14s} {r['model']:9s} "
                f"n={r['n_dsc']:>4} "
                f"DSC={r['dsc_mean']:.4f} [{r['dsc_ci_low']:.4f}, {r['dsc_ci_high']:.4f}] "
                f"HD95={r['hd95_mean']:.2f} [{r['hd95_ci_low']:.2f}, {r['hd95_ci_high']:.2f}]\n"
            )
        f.write("\n=== Pairwise Wilcoxon DSC (Holm-Bonferroni, two-sided) ===\n")
        for r in wilcoxon_rows:
            f.write(
                f"  {r['dataset']:14s} {r['model_a']:8s} vs {r['model_b']:8s} "
                f"n={r['n_paired']:>4} "
                f"med_diff={r['median_diff_a_minus_b']:+.4f} "
                f"p_holm={r['p_holm']:.6f} {r['stars']}\n"
            )
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
