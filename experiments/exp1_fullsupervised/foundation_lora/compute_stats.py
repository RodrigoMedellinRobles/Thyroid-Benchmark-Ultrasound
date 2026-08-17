#!/usr/bin/env python3
"""Bootstrap 95% CIs + empty counts for the full-supervised FOUNDATION block (Exp 9).

Replicates the exact methodology of experiments/exp1_fullsupervised/traditional_boxcond/
aggregate_boxcond.py so the foundation rows are directly comparable to the
traditional / box-conditioned appendix tables:
  * HD95 +inf (empty pred/GT) imputed to the image diagonal sqrt(2)*512 px.
  * Bootstrap: 1,000 percentile resamples, seed 42, scipy.stats.bootstrap.

Per-image sources (already on disk -- NO re-run needed):
  experiments/exp1_fullsupervised/foundation_lora/results/fullsup_{model}_{dataset}/per_image_results.csv

Outputs:
  results/stats/exp9_foundation_bootstrap_ci95.csv
  results/stats/exp9_foundation_appendix_rows.tex  (LaTeX body rows for tab:exp1_bootstrap)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

HERE = Path(__file__).resolve().parent
HD95_INF_CAP = float(np.sqrt(2) * 512)  # 724.0773 px, matches compute_stats.py
DATASETS = (("ddti", "DDTI"), ("tn3k", "TN3K"),
            ("thyroidxl", "ThyroidXL"), ("stanford_aimi", "Stanford AIMI"))
MODELS = (("sam2", "SAM2 (Hiera-L)"), ("medsam", "MedSAM (ViT-B)"),
          ("medsam2", "MedSAM-2"), ("sam3", "SAM3 (vanilla)"))
OUT_DIR = HERE / "results" / "stats"


def load_per_image(model: str, dataset: str) -> pd.DataFrame:
    f = HERE / "results" / f"fullsup_{model}_{dataset}" / "per_image_results.csv"
    df = pd.read_csv(f)[["dice", "hd95", "skipped"]].copy()
    # blank / non-numeric hd95 (empty prediction) -> +inf, matching impute-aware policy.
    df["hd95"] = pd.to_numeric(df["hd95"], errors="coerce").fillna(np.inf)
    return df


def boot_ci(vals: np.ndarray) -> tuple:
    rng = np.random.default_rng(42)
    res = bootstrap((vals,), statistic=np.mean, n_resamples=1000,
                    confidence_level=0.95, method="percentile", random_state=rng)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boot_rows, tex_lines = [], []
    for dkey, dlabel in DATASETS:
        tex_lines.append(f"\\multirow{{4}}{{*}}{{{dlabel}}}")
        for mkey, mlabel in MODELS:
            df = load_per_image(mkey, dkey)
            dsc = df["dice"].to_numpy(float)
            hd_raw = df["hd95"].to_numpy(float)
            n_inf = int(np.isinf(hd_raw).sum())
            hd = np.where(np.isinf(hd_raw), HD95_INF_CAP, hd_raw)
            d_lo, d_hi = boot_ci(dsc)
            h_lo, h_hi = boot_ci(hd)
            fail_pct = 100 * n_inf / len(df)
            boot_rows.append(dict(
                dataset=dkey, model=mkey, n_total=len(df), n_empty=n_inf,
                empty_pct=round(fail_pct, 2),
                dsc_mean=round(float(dsc.mean()), 4),
                dsc_ci95_lo=round(d_lo, 4), dsc_ci95_hi=round(d_hi, 4),
                hd95_mean=round(float(hd.mean()), 2),
                hd95_ci95_lo=round(h_lo, 2), hd95_ci95_hi=round(h_hi, 2)))
            tex_lines.append(
                f"  & {mlabel:<15}& {n_inf:>3d} & {fail_pct:.2f} "
                f"& {dsc.mean():.3f} & [{d_lo:.3f}, {d_hi:.3f}] "
                f"& {hd.mean():.2f} & [{h_lo:.2f}, {h_hi:.2f}] \\\\")
        tex_lines.append("\\midrule" if dkey != "stanford_aimi" else "\\bottomrule")

    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "exp9_foundation_bootstrap_ci95.csv", index=False)
    (OUT_DIR / "exp9_foundation_appendix_rows.tex").write_text("\n".join(tex_lines) + "\n")
    print("\n".join(tex_lines))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
