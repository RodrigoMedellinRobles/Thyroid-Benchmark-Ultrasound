"""
Stratified analysis for Exp 6.

Inputs:
    experiments/exp7_stratified/results/stratifier_table_long.csv
        (produced by build_stratifier_table.py)

Outputs (under experiments/exp7_stratified/results/):
    figure3_data.csv          per-(dataset,stratifier,bin,model) median + CI95
    stats_per_bin.csv         Wilcoxon paired (per (dataset,stratifier,bin)
                              for each foundation-vs-traditional model pair)
                              with raw + FDR-BH adjusted p-values.
    descriptive_per_bin.csv   bin-level n and bin assignments per dataset
    confounders_spearman.csv  Spearman(TIRADS,size_mm,age) per dataset

Statistics:
    - Bootstrap CI95: cluster bootstrap on patient_id (n_boot=1000, seed=42).
    - Wilcoxon signed-rank: paired (per patient/nodule) DSC between two models.
    - FDR-BH q=0.10 applied across the *exploratory* family of tests.
    - Bins with n<20 patients are flagged 'underpowered' and skipped for
      Wilcoxon (descriptive only).
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_LONG = REPO_ROOT / "experiments/exp7_stratified/results/stratifier_table_long.csv"
PATH_OUT_DIR = REPO_ROOT / "experiments/exp7_stratified/results"

RNG_SEED = 42
N_BOOT = 1000
MIN_N_FOR_WILCOXON = 20  # below this, descriptive-only

FOUNDATION = ["sam2", "medsam", "medsam2", "sam3"]  # SAM (ViT-H) dropped
TRADITIONAL = ["unet", "transunet", "nnunet"]

# Foundation regimes carried in the 2026-07 reconfig long table.
FOUNDATION_REGIMES = ["fewshot_f0.05", "fullsup"]

# HD95 imputation floor (image diagonal, px). Already applied upstream in
# build_stratifier_table.py; kept here for documentation.
HD95_IMPUTE = float(np.sqrt(2.0) * 512.0)


# ---------------------------------------------------------------------------
# Binning per stratifier (native granularity preserved)
# ---------------------------------------------------------------------------
def bin_tirads(value, dataset: str):
    """Native TIRADS granularity per dataset (no collapse)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if dataset == "ddti":
        # Pedraza native: '2','3','4a','4b','4c','5'
        return f"TR{s}"
    # ThyroidXL/Stanford: ACR TR1-5 integer scale
    try:
        return f"TR{int(float(s))}"
    except ValueError:
        return np.nan


def bin_size_mm(value):
    """4 bins: <10, 10-20, 20-30, ≥30 mm."""
    if pd.isna(value):
        return np.nan
    v = float(value)
    if v < 10:
        return "<10mm"
    if v < 20:
        return "10-20mm"
    if v < 30:
        return "20-30mm"
    return "≥30mm"


def bin_age(value):
    """3 bins: <40, 40-60, ≥60."""
    if pd.isna(value):
        return np.nan
    v = float(value)
    if v < 40:
        return "<40"
    if v < 60:
        return "40-60"
    return "≥60"


def bin_sex(value):
    if pd.isna(value):
        return np.nan
    return str(value)


STRATIFIERS = {
    "tirads": lambda df: df.apply(lambda r: bin_tirads(r["tirads"], r["dataset"]),
                                  axis=1),
    "size":   lambda df: df["size_mm"].apply(bin_size_mm),
    "age":    lambda df: df["age"].apply(bin_age),
    "sex":    lambda df: df["sex"].apply(bin_sex),
}


# ---------------------------------------------------------------------------
# Cluster bootstrap CI95
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(values: np.ndarray, cluster_ids: np.ndarray,
                         statistic=np.median, n_boot: int = N_BOOT,
                         alpha: float = 0.05, seed: int = RNG_SEED):
    """Return (point estimate, low CI, high CI) using a patient-clustered
    bootstrap: resample cluster_ids with replacement, include all observations
    from each sampled cluster, recompute statistic."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) < 2:
        # Degenerate case: not enough independent units.
        point = float(statistic(values)) if len(values) else np.nan
        return point, np.nan, np.nan

    # Pre-build an index of rows per cluster for speed.
    idx_per_cluster = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters),
                                      replace=True)
        rows = np.concatenate([idx_per_cluster[c] for c in sampled_clusters])
        boots[b] = statistic(values[rows])

    point = float(statistic(values))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Paired Wilcoxon between two models within a bin
# ---------------------------------------------------------------------------
def paired_wilcoxon_within_bin(df_bin: pd.DataFrame,
                               model_a: str, regime_a: str,
                               model_b: str, regime_b: str,
                               metric: str = "dice"):
    """Pair patients that appear in both (model_a,regime_a) and
    (model_b,regime_b) within this bin. Returns
    (n_pairs, statistic, p_value, median_diff_a_minus_b)."""
    sa = df_bin[(df_bin["model"] == model_a) & (df_bin["regime"] == regime_a)]
    sb = df_bin[(df_bin["model"] == model_b) & (df_bin["regime"] == regime_b)]
    # In case of duplicate patient_id rows (shouldn't happen for a single
    # (model,regime) but be defensive), aggregate to median per patient.
    sa = sa.groupby("patient_id")[metric].median()
    sb = sb.groupby("patient_id")[metric].median()
    common = sa.index.intersection(sb.index)
    if len(common) < MIN_N_FOR_WILCOXON:
        return len(common), np.nan, np.nan, np.nan
    a_v, b_v = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()
    mask = np.isfinite(a_v) & np.isfinite(b_v)
    if mask.sum() < MIN_N_FOR_WILCOXON:
        return int(mask.sum()), np.nan, np.nan, np.nan
    res = stats.wilcoxon(a_v[mask], b_v[mask], zero_method="wilcox",
                         alternative="two-sided")
    return (int(mask.sum()), float(res.statistic), float(res.pvalue),
            float(np.median(a_v[mask] - b_v[mask])))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="all",
                        help="Which regime to analyze. Default 'all' keeps "
                             "every (model,regime) present in the long table "
                             "(traditional fullsup + foundation fewshot_f0.05 "
                             "+ foundation fullsup). Pass a single regime name "
                             "to restrict.")
    parser.add_argument("--include-all-regimes", action="store_true",
                        help="Deprecated no-op kept for backward compat; "
                             "the default now keeps all regimes.")
    args = parser.parse_args()

    print("[1/4] Loading long-format table...")
    long = pd.read_csv(PATH_LONG)
    print(f"   Rows: {len(long):,}")

    # Filter to regime(s) of interest. Default keeps everything.
    if args.regime not in ("all", None) and not args.include_all_regimes:
        long = long[long["regime"] == args.regime].copy()
    print(f"   After regime filter: {len(long):,} rows "
          f"({long['model'].nunique()} models, "
          f"{long['dataset'].nunique()} datasets, "
          f"regimes={sorted(long['regime'].unique())})")

    print("\n[2/4] Assigning stratifier bins (native granularity)...")
    for sname, fn in STRATIFIERS.items():
        long[f"bin_{sname}"] = fn(long)

    # ---- Descriptive: bin counts per (dataset, stratifier, bin) ----
    desc_rows = []
    for sname in STRATIFIERS:
        col = f"bin_{sname}"
        for (ds, b), sub in long.dropna(subset=[col]).groupby(["dataset", col]):
            desc_rows.append(dict(dataset=ds, stratifier=sname, bin=b,
                                   n_observations=len(sub),
                                   n_patients=sub["patient_id"].nunique()))
    desc_df = pd.DataFrame(desc_rows).sort_values(
        ["dataset", "stratifier", "bin"])
    desc_df.to_csv(PATH_OUT_DIR / "descriptive_per_bin.csv", index=False)
    print(f"   Wrote descriptive_per_bin.csv ({len(desc_df)} bin rows)")

    # ---- Figure 3 data: cluster bootstrap CI95 per (ds, strat, bin, model) ----
    print("\n[3/4] Cluster bootstrap CI95 per (dataset,stratifier,bin,model)...")
    fig_rows = []
    for sname in STRATIFIERS:
        col = f"bin_{sname}"
        groups = (long.dropna(subset=[col, "dice"])
                       .groupby(["dataset", col, "model", "regime"]))
        for (ds, b, m, r), sub in groups:
            point, lo, hi = cluster_bootstrap_ci(
                sub["dice"].to_numpy(),
                sub["patient_id"].to_numpy())
            # HD95: same two-stage aggregation as DSC (Stanford already
            # collapsed to per-nodule median upstream). Non-finite HD95 was
            # imputed to the image diagonal (sqrt(2)*512 px) before this step.
            hd_sub = sub.dropna(subset=["hd95"])
            if len(hd_sub):
                hd_point, hd_lo, hd_hi = cluster_bootstrap_ci(
                    hd_sub["hd95"].to_numpy(),
                    hd_sub["patient_id"].to_numpy())
            else:
                hd_point = hd_lo = hd_hi = np.nan
            n_pat = sub["patient_id"].nunique()
            fig_rows.append(dict(dataset=ds, stratifier=sname, bin=b,
                                  model=m, regime=r,
                                  n_observations=len(sub),
                                  n_patients=n_pat,
                                  median_dsc=point,
                                  ci_low=lo, ci_high=hi,
                                  median_hd95=hd_point,
                                  hd95_ci_low=hd_lo, hd95_ci_high=hd_hi,
                                  underpowered=n_pat < MIN_N_FOR_WILCOXON))
    fig_df = pd.DataFrame(fig_rows)
    fig_df.to_csv(PATH_OUT_DIR / "figure3_data.csv", index=False)
    print(f"   Wrote figure3_data.csv ({len(fig_df)} rows)")

    # ---- Wilcoxon: foundation vs traditional within each bin ----
    # Comparison family (exploratory): each foundation model under
    # {zeroshot, fewshot_f0.25} paired against each traditional model under
    # fullsup. All p-values feed into FDR-BH q=0.10.
    print("\n[4/4] Wilcoxon paired per bin (FDR-BH q=0.10)...")
    stat_rows = []
    available = long[["model", "regime"]].drop_duplicates()
    foundation_present = [(m, r) for m, r in available.itertuples(index=False)
                          if m in FOUNDATION
                          and r in FOUNDATION_REGIMES]
    traditional_present = [(m, r) for m, r in available.itertuples(index=False)
                           if m in TRADITIONAL and r == "fullsup"]
    pairs = [(fm, fr, tm, tr) for fm, fr in foundation_present
             for tm, tr in traditional_present]

    for sname in STRATIFIERS:
        col = f"bin_{sname}"
        for (ds, b), sub in long.dropna(subset=[col]).groupby(["dataset", col]):
            for fm, fr, tm, tr in pairs:
                n, w, p, med_diff = paired_wilcoxon_within_bin(
                    sub, fm, fr, tm, tr)
                stat_rows.append(dict(dataset=ds, stratifier=sname, bin=b,
                                       model_a=fm, regime_a=fr,
                                       model_b=tm, regime_b=tr,
                                       n_pairs=n, wilcoxon_stat=w,
                                       p_raw=p,
                                       median_diff_a_minus_b=med_diff))
    stat_df = pd.DataFrame(stat_rows)

    # FDR-BH across all valid (non-NaN p) rows.
    mask = stat_df["p_raw"].notna()
    if mask.any():
        rej, padj, _, _ = multipletests(stat_df.loc[mask, "p_raw"].to_numpy(),
                                         alpha=0.10, method="fdr_bh")
        stat_df.loc[mask, "p_fdr_bh"] = padj
        stat_df.loc[mask, "sig_fdr_bh"] = rej
    stat_df.to_csv(PATH_OUT_DIR / "stats_per_bin.csv", index=False)
    print(f"   Wrote stats_per_bin.csv ({len(stat_df)} test rows, "
          f"{mask.sum()} with valid p)")

    # ---- Confounders: Spearman per dataset ----
    conf_rows = []
    for ds in long["dataset"].unique():
        # One row per patient: prefer rows with non-NaN stratifiers, then
        # drop duplicates (defensive — most datasets already have 1 row/pid
        # in metadata; ThyroidXL has multiple imgs/pid so dedup is needed).
        sub = (long[(long["dataset"] == ds)]
               .dropna(subset=["tirads", "size_mm", "age"], how="all")
               .drop_duplicates(subset=["patient_id"]))
        # TIRADS as ordinal numeric: strip 'a/b/c' suffix (Pedraza) and cast.
        def tr_num(v):
            if pd.isna(v):
                return np.nan
            s = str(v).strip().rstrip("abc")
            try:
                return float(s)
            except ValueError:
                return np.nan
        tr_n = sub["tirads"].apply(tr_num)
        for var_a, val_a in [("tirads", tr_n), ("size_mm", sub["size_mm"]),
                              ("age", sub["age"])]:
            for var_b, val_b in [("tirads", tr_n), ("size_mm", sub["size_mm"]),
                                  ("age", sub["age"])]:
                if var_a >= var_b:
                    continue
                m = val_a.notna() & val_b.notna()
                if m.sum() < 10:
                    continue
                rho, p = stats.spearmanr(val_a[m], val_b[m])
                conf_rows.append(dict(dataset=ds, var_a=var_a, var_b=var_b,
                                       n=int(m.sum()),
                                       spearman_rho=float(rho),
                                       p_value=float(p)))
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(PATH_OUT_DIR / "confounders_spearman.csv", index=False)
    print(f"   Wrote confounders_spearman.csv ({len(conf_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
