#!/usr/bin/env python3
"""Exp 4.5 statistical analysis — bootstrap CIs + Wilcoxon (Holm).

Reads per_image_results.csv from
    experiments/exp4_crossdataset/foundation/results/exp4_5_<model>_src<src>_tgt<tgt>_f<frac>/
and (for paired reference) from
    experiments/exp2_zeroshot/results/<model>_<tgt>/per_image_results.csv      (zero-shot baseline)
    experiments/exp3_fewshot_lora/results/fewshot_<model>_<tgt>_f<frac>/per_image_results.csv  (in-domain LoRA reference)

Emits:
    results/stats/exp4_5_bootstrap_ci95.csv   — per-cell DSC / HD95 mean + 95% CI on cross-eval cells.
    results/stats/exp4_5_wilcoxon_vs_zeroshot.csv  — per (model, src, tgt, frac), paired test:
                                                     cross-eval DSC vs zero-shot DSC on the tgt test set.
    results/stats/exp4_5_wilcoxon_vs_indomain.csv  — per (model, src, tgt, frac), paired test:
                                                     cross-eval DSC vs in-domain LoRA DSC on the tgt test set.
    results/stats/exp4_5_stats_summary.txt    — human-readable digest.

Holm-Bonferroni correction is applied within each (model, fraction) family
(over 12 src→tgt off-diagonal contrasts) for each pairwise test.

Usage:
    python experiments/exp4_crossdataset/foundation/compute_stats.py
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUT_DIR = RESULTS_DIR / "stats"

ZS_RESULTS = PROJECT_ROOT / "experiments" / "exp2_zeroshot" / "results"
EXP3_RESULTS = PROJECT_ROOT / "experiments" / "exp3_fewshot_lora" / "results"

MODELS = ("sam2", "sam3", "medsam", "medsam2")
DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")
FRACTIONS = (0.05, 0.5)


def _load_per_image(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "skipped" in df.columns:
        df = df[df["skipped"] == False].copy()
    for col in ("dice", "iou", "hd95"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_cross(model: str, src: str, tgt: str, frac: float) -> pd.DataFrame | None:
    run = f"exp4_5_{model}_src{src}_tgt{tgt}_f{frac:g}"
    return _load_per_image(RESULTS_DIR / run / "per_image_results.csv")


def load_zeroshot(model: str, tgt: str) -> pd.DataFrame | None:
    return _load_per_image(ZS_RESULTS / f"{model}_{tgt}" / "per_image_results.csv")


def load_indomain(model: str, tgt: str, frac: float) -> pd.DataFrame | None:
    run = f"fewshot_{model}_{tgt}_f{frac:g}"
    return _load_per_image(EXP3_RESULTS / run / "per_image_results.csv")


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


def _paired_wilcoxon(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[int, float, float, float]:
    """Paired Wilcoxon on per-image DSC (A vs B)."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cross: dict[tuple[str, str, str, float], pd.DataFrame] = {}
    missing_cross: list[str] = []
    for m in MODELS:
        for src in DATASETS:
            for tgt in DATASETS:
                if src == tgt:
                    continue
                for f in FRACTIONS:
                    df = load_cross(m, src, tgt, f)
                    if df is None or len(df) == 0:
                        missing_cross.append(f"{m}_src{src}_tgt{tgt}_f{f:g}")
                        continue
                    cross[(m, src, tgt, f)] = df

    if missing_cross:
        print(f"[WARN] missing cross-eval cells: {len(missing_cross)}")
        for k in missing_cross[:5]:
            print(f"  - {k}")
        if len(missing_cross) > 5:
            print(f"  ... and {len(missing_cross) - 5} more")

    # ----- Bootstrap CIs -----------------------------------------------------
    boot_rows = []
    for (m, src, tgt, f), df in cross.items():
        dsc = df["dice"].dropna().to_numpy()
        hd95 = df["hd95"].dropna().to_numpy() if "hd95" in df.columns else np.array([])
        dsc_m, dsc_lo, dsc_hi = bootstrap_ci(dsc, args.n_boot, args.alpha, args.seed)
        hd_m, hd_lo, hd_hi = bootstrap_ci(hd95, args.n_boot, args.alpha, args.seed)
        boot_rows.append(dict(
            model=m, src_dataset=src, tgt_dataset=tgt, fraction=f,
            n_dsc=len(dsc),
            dsc_mean=dsc_m, dsc_ci_low=dsc_lo, dsc_ci_high=dsc_hi,
            n_hd95=len(hd95) if hd95.size else 0,
            hd95_mean=hd_m, hd95_ci_low=hd_lo, hd95_ci_high=hd_hi,
        ))
    boot_path = args.output_dir / "exp4_5_bootstrap_ci95.csv"
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    print(f"[OK] {boot_path} ({len(boot_rows)} cells)")

    # ----- Wilcoxon: cross vs zero-shot (Exp 2) on tgt test set ---------------
    zs_rows = []
    for m in MODELS:
        for f in FRACTIONS:
            pvals, recs = [], []
            for src in DATASETS:
                for tgt in DATASETS:
                    if src == tgt:
                        continue
                    df_cross = cross.get((m, src, tgt, f))
                    df_zs = load_zeroshot(m, tgt)
                    if df_cross is None or df_zs is None:
                        continue
                    n, W, med, p = _paired_wilcoxon(df_cross, df_zs)
                    recs.append(dict(
                        model=m, src_dataset=src, tgt_dataset=tgt, fraction=f,
                        comparison="cross_vs_zeroshot",
                        n_paired=n, wilcoxon_W=W,
                        median_diff_a_minus_b=med, p_value=p,
                    ))
                    pvals.append(p)
            if pvals:
                pvals_arr = np.array([p if np.isfinite(p) else 1.0 for p in pvals])
                reject, p_adj, _, _ = multipletests(pvals_arr, alpha=args.alpha, method="holm")
                for r, pa, rj in zip(recs, p_adj, reject):
                    r["p_holm"] = float(pa)
                    r["reject_at_alpha_holm"] = bool(rj)
                    r["stars"] = stars(pa)
            zs_rows.extend(recs)
    zs_path = args.output_dir / "exp4_5_wilcoxon_vs_zeroshot.csv"
    if zs_rows:
        pd.DataFrame(zs_rows).to_csv(zs_path, index=False)
        print(f"[OK] {zs_path} ({len(zs_rows)} tests)")

    # ----- Wilcoxon: cross vs in-domain LoRA (Exp 3 same model, tgt, frac) ----
    id_rows = []
    for m in MODELS:
        for f in FRACTIONS:
            pvals, recs = [], []
            for src in DATASETS:
                for tgt in DATASETS:
                    if src == tgt:
                        continue
                    df_cross = cross.get((m, src, tgt, f))
                    df_id = load_indomain(m, tgt, f)
                    if df_cross is None or df_id is None:
                        continue
                    n, W, med, p = _paired_wilcoxon(df_cross, df_id)
                    recs.append(dict(
                        model=m, src_dataset=src, tgt_dataset=tgt, fraction=f,
                        comparison="cross_vs_indomain",
                        n_paired=n, wilcoxon_W=W,
                        median_diff_a_minus_b=med, p_value=p,
                    ))
                    pvals.append(p)
            if pvals:
                pvals_arr = np.array([p if np.isfinite(p) else 1.0 for p in pvals])
                reject, p_adj, _, _ = multipletests(pvals_arr, alpha=args.alpha, method="holm")
                for r, pa, rj in zip(recs, p_adj, reject):
                    r["p_holm"] = float(pa)
                    r["reject_at_alpha_holm"] = bool(rj)
                    r["stars"] = stars(pa)
            id_rows.extend(recs)
    id_path = args.output_dir / "exp4_5_wilcoxon_vs_indomain.csv"
    if id_rows:
        pd.DataFrame(id_rows).to_csv(id_path, index=False)
        print(f"[OK] {id_path} ({len(id_rows)} tests)")

    # ----- Human summary -----------------------------------------------------
    summary_path = args.output_dir / "exp4_5_stats_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("Exp 4.5 cross-dataset LoRA — bootstrap CIs + Wilcoxon (Holm)\n")
        fh.write(f"alpha={args.alpha} n_boot={args.n_boot} seed={args.seed}\n")
        fh.write(f"Cross-eval cells loaded: {len(cross)} / {len(MODELS)*12*len(FRACTIONS)}\n")
        if missing_cross:
            fh.write(f"Missing: {len(missing_cross)}  (e.g. {missing_cross[:3]})\n")

        fh.write("\n=== Bootstrap mean DSC / HD95 [95% CI] (cross-eval cells) ===\n")
        for r in boot_rows:
            fh.write(
                f"  {r['model']:8s}  {r['src_dataset']:13s} -> {r['tgt_dataset']:13s}  "
                f"f={r['fraction']:g}  n={r['n_dsc']:>4}  "
                f"DSC={r['dsc_mean']:.4f}[{r['dsc_ci_low']:.4f},{r['dsc_ci_high']:.4f}]  "
                f"HD95={r['hd95_mean']:.2f}[{r['hd95_ci_low']:.2f},{r['hd95_ci_high']:.2f}]\n"
            )
        fh.write("\n=== Cross-eval DSC vs zero-shot DSC (paired Wilcoxon, Holm per model+frac) ===\n")
        for r in zs_rows:
            fh.write(
                f"  {r['model']:8s}  src={r['src_dataset']:13s} tgt={r['tgt_dataset']:13s}  "
                f"f={r['fraction']:g}  n={r['n_paired']:>4}  "
                f"med_diff={r['median_diff_a_minus_b']:+.4f}  "
                f"p_holm={r.get('p_holm', float('nan')):.6f} {r.get('stars', 'ns')}\n"
            )
        fh.write("\n=== Cross-eval DSC vs in-domain LoRA DSC (paired Wilcoxon, Holm per model+frac) ===\n")
        for r in id_rows:
            fh.write(
                f"  {r['model']:8s}  src={r['src_dataset']:13s} tgt={r['tgt_dataset']:13s}  "
                f"f={r['fraction']:g}  n={r['n_paired']:>4}  "
                f"med_diff={r['median_diff_a_minus_b']:+.4f}  "
                f"p_holm={r.get('p_holm', float('nan')):.6f} {r.get('stars', 'ns')}\n"
            )
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
