#!/usr/bin/env python3
"""Exp 3 statistical analysis — bootstrap 95% CI + pairwise Wilcoxon (Holm).

Reads per_image_results.csv for all (model, dataset, fraction) cells of Exp 3
and emits:

    results/stats/exp3_bootstrap_ci95.csv     — per-cell bootstrap CIs on DSC, IoU, HD95.
    results/stats/exp3_wilcoxon_fraction.csv  — within (model, dataset), each lower
                                                 fraction vs f=0.5 on paired DSC.
    results/stats/exp3_wilcoxon_model_f50.csv — within dataset at f=0.5, all
                                                 model-vs-model pairs on paired DSC.
    results/stats/exp3_stats_summary.txt      — human-readable digest.

Holm-Bonferroni correction is applied within each (model, dataset) for the
fraction-saturation tests and within each dataset for the model-competition
tests.

Usage:
    python experiments/exp3_fewshot_lora/compute_stats.py
    python experiments/exp3_fewshot_lora/compute_stats.py --n_boot 5000 --seed 42

The script is self-contained: stdlib + numpy/pandas/scipy/statsmodels.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUT_DIR = RESULTS_DIR / "stats"

MODELS = ("sam2", "sam3", "medsam", "medsam2")
DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")
FRACTIONS = (0.05, 0.10, 0.25, 0.50)


def _run_dir(model: str, dataset: str, fraction: float) -> Path:
    """Parse a run directory name of the form {prefix}_{model}_{dataset}_f{fraction}."""
    return RESULTS_DIR / f"fewshot_{model}_{dataset}_f{fraction:g}"


def load_per_image(model: str, dataset: str, fraction: float) -> pd.DataFrame | None:
    path = _run_dir(model, dataset, fraction) / "per_image_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["skipped"] == False].copy()
    for col in ("dice", "iou", "hd95"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float,
                 seed: int) -> tuple[float, float, float]:
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


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _wilcoxon_dsc(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[int, float, float, float]:
    """Paired Wilcoxon on DSC; returns (n_paired, W, median_diff_a_minus_b, p)."""
    merged = df_a[["filename", "dice"]].merge(
        df_b[["filename", "dice"]], on="filename", suffixes=("_a", "_b")
    ).dropna(subset=["dice_a", "dice_b"])
    if len(merged) < 10:
        return len(merged), float("nan"), float("nan"), float("nan")
    diffs = merged["dice_a"].to_numpy() - merged["dice_b"].to_numpy()
    if np.all(diffs == 0):
        return len(merged), float("nan"), 0.0, 1.0
    stat, p = wilcoxon(diffs, alternative="two-sided",
                       zero_method="wilcox", correction=False)
    return len(merged), float(stat), float(np.median(diffs)), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load every cell once.
    cells: dict[tuple[str, str, float], pd.DataFrame] = {}
    missing: list[str] = []
    for ds in DATASETS:
        for m in MODELS:
            for f in FRACTIONS:
                df = load_per_image(m, ds, f)
                if df is None or len(df) == 0:
                    missing.append(f"{m}_{ds}_f{f:g}")
                    continue
                cells[(m, ds, f)] = df

    if missing:
        print(f"[WARN] missing or empty per_image_results.csv for: {missing}")

    # ----- Bootstrap CIs per cell --------------------------------------------
    boot_rows = []
    for ds in DATASETS:
        for m in MODELS:
            for f in FRACTIONS:
                df = cells.get((m, ds, f))
                if df is None:
                    continue
                dsc = df["dice"].dropna().to_numpy()
                iou = df["iou"].dropna().to_numpy()
                hd95 = df["hd95"].dropna().to_numpy()

                dsc_mean, dsc_lo, dsc_hi = bootstrap_ci(dsc, args.n_boot, args.alpha, args.seed)
                iou_mean, iou_lo, iou_hi = bootstrap_ci(iou, args.n_boot, args.alpha, args.seed)
                hd95_mean, hd95_lo, hd95_hi = bootstrap_ci(hd95, args.n_boot, args.alpha, args.seed)

                # Store at high precision so consumers can format to any
                # display precision without double-rounding artefacts.
                boot_rows.append(dict(
                    dataset=ds, model=m, fraction=f,
                    n_dsc=len(dsc),
                    dsc_mean=dsc_mean,
                    dsc_ci_low=dsc_lo,
                    dsc_ci_high=dsc_hi,
                    iou_mean=iou_mean,
                    iou_ci_low=iou_lo,
                    iou_ci_high=iou_hi,
                    n_hd95=len(hd95),
                    hd95_mean=hd95_mean,
                    hd95_ci_low=hd95_lo,
                    hd95_ci_high=hd95_hi,
                ))

    boot_path = args.output_dir / "exp3_bootstrap_ci95.csv"
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    print(f"[OK] {boot_path} ({len(boot_rows)} cells)")

    # ----- Wilcoxon fraction-vs-f50 (saturation test) -------------------------
    frac_rows = []
    for ds in DATASETS:
        for m in MODELS:
            df_ref = cells.get((m, ds, 0.50))
            if df_ref is None:
                continue
            pvals, recs = [], []
            for f in (0.05, 0.10, 0.25):
                df_q = cells.get((m, ds, f))
                if df_q is None:
                    continue
                n, W, med, p = _wilcoxon_dsc(df_q, df_ref)
                recs.append(dict(
                    dataset=ds, model=m, fraction_a=f, fraction_b=0.50,
                    n_paired=n,
                    wilcoxon_W=round(W, 4) if np.isfinite(W) else float("nan"),
                    median_diff_a_minus_b=round(med, 5),
                    p_value=round(p, 6) if np.isfinite(p) else float("nan"),
                ))
                pvals.append(p)
            if pvals:
                reject, p_adj, _, _ = multipletests(pvals, alpha=args.alpha, method="holm")
                for r, pa, rj in zip(recs, p_adj, reject):
                    r["p_holm"] = round(float(pa), 6)
                    r["reject_at_alpha_holm"] = bool(rj)
                    r["stars"] = stars(pa)
            frac_rows.extend(recs)

    frac_path = args.output_dir / "exp3_wilcoxon_fraction.csv"
    if frac_rows:
        cols = [
            "dataset", "model", "fraction_a", "fraction_b", "n_paired",
            "wilcoxon_W", "median_diff_a_minus_b",
            "p_value", "p_holm", "reject_at_alpha_holm", "stars",
        ]
        pd.DataFrame(frac_rows)[cols].to_csv(frac_path, index=False)
        print(f"[OK] {frac_path} ({len(frac_rows)} tests)")

    # ----- Wilcoxon model-vs-model at f=50% (architecture competition) --------
    arch_rows = []
    for ds in DATASETS:
        pvals, recs = [], []
        for m_a, m_b in combinations(MODELS, 2):
            df_a = cells.get((m_a, ds, 0.50))
            df_b = cells.get((m_b, ds, 0.50))
            if df_a is None or df_b is None:
                continue
            n, W, med, p = _wilcoxon_dsc(df_a, df_b)
            recs.append(dict(
                dataset=ds, model_a=m_a, model_b=m_b, fraction=0.50,
                n_paired=n,
                wilcoxon_W=round(W, 4) if np.isfinite(W) else float("nan"),
                median_diff_a_minus_b=round(med, 5),
                p_value=round(p, 6) if np.isfinite(p) else float("nan"),
            ))
            pvals.append(p)
        if pvals:
            reject, p_adj, _, _ = multipletests(pvals, alpha=args.alpha, method="holm")
            for r, pa, rj in zip(recs, p_adj, reject):
                r["p_holm"] = round(float(pa), 6)
                r["reject_at_alpha_holm"] = bool(rj)
                r["stars"] = stars(pa)
        arch_rows.extend(recs)

    arch_path = args.output_dir / "exp3_wilcoxon_model_f50.csv"
    if arch_rows:
        cols = [
            "dataset", "model_a", "model_b", "fraction", "n_paired",
            "wilcoxon_W", "median_diff_a_minus_b",
            "p_value", "p_holm", "reject_at_alpha_holm", "stars",
        ]
        pd.DataFrame(arch_rows)[cols].to_csv(arch_path, index=False)
        print(f"[OK] {arch_path} ({len(arch_rows)} tests)")

    # ----- Human-readable summary --------------------------------------------
    summary_path = args.output_dir / "exp3_stats_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("Exp 3 few-shot LoRA — bootstrap CIs + Wilcoxon (Holm-Bonferroni)\n")
        fh.write(f"alpha={args.alpha} n_boot={args.n_boot} seed={args.seed}\n")
        fh.write(f"Cells loaded: {len(cells)} / {len(MODELS)*len(DATASETS)*len(FRACTIONS)}\n")
        if missing:
            fh.write(f"Missing: {', '.join(missing)}\n")
        fh.write("\n=== Bootstrap mean DSC / IoU / HD95 [95% CI] ===\n")
        for r in boot_rows:
            fh.write(
                f"  {r['dataset']:14s} {r['model']:8s} f={r['fraction']:.2f} "
                f"n={r['n_dsc']:>4} "
                f"DSC={r['dsc_mean']:.4f}[{r['dsc_ci_low']:.4f},{r['dsc_ci_high']:.4f}] "
                f"IoU={r['iou_mean']:.4f}[{r['iou_ci_low']:.4f},{r['iou_ci_high']:.4f}] "
                f"HD95={r['hd95_mean']:.2f}[{r['hd95_ci_low']:.2f},{r['hd95_ci_high']:.2f}]\n"
            )
        fh.write("\n=== Saturation: fraction f vs f=0.5 on paired DSC (Holm per model,dataset) ===\n")
        for r in frac_rows:
            fh.write(
                f"  {r['dataset']:14s} {r['model']:8s} "
                f"f={r['fraction_a']:.2f} vs f=0.50 "
                f"n={r['n_paired']:>4} "
                f"med_diff={r['median_diff_a_minus_b']:+.4f} "
                f"p_holm={r.get('p_holm', float('nan')):.6f} {r.get('stars', 'ns')}\n"
            )
        fh.write("\n=== Architecture competition at f=0.5 (Holm per dataset, 6 pairs) ===\n")
        for r in arch_rows:
            fh.write(
                f"  {r['dataset']:14s} {r['model_a']:8s} vs {r['model_b']:8s} "
                f"n={r['n_paired']:>4} "
                f"med_diff={r['median_diff_a_minus_b']:+.4f} "
                f"p_holm={r.get('p_holm', float('nan')):.6f} {r.get('stars', 'ns')}\n"
            )
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
