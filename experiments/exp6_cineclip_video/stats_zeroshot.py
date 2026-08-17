"""Bootstrap CIs + paired Wilcoxon + Holm-Bonferroni for Experiment 6 (video propagation).

For each (model, mode) cell: per-patient bootstrap 95 % CI on DSC,
HD95-impute-aware, HD95-nonempty, and inter-frame Jaccard (Jt).
For each model: paired Wilcoxon signed-rank test comparing framewise
vs video on DSC, HD95-impute-aware, HD95-nonempty, and Jt across the
shared patient set, with family-wise Holm-Bonferroni adjustment over
the full 4-metric * N-model grid.

Inputs:
    experiments/exp6_cineclip_video/results/{model}_{mode}/per_sequence_metrics.csv
    experiments/exp6_cineclip_video/results/{model}_{mode}/per_frame_metrics.csv

Outputs:
    experiments/exp6_cineclip_video/results/zeroshot_bootstrap_ci95.csv
    experiments/exp6_cineclip_video/results/zeroshot_wilcoxon.csv

Usage:
    python experiments/exp6_cineclip_video/stats_zeroshot.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"

MODELS = ("sam2", "medsam2", "sam3")
MODES = ("framewise", "video")
METRICS = ("dice_mean", "hd95_mean", "hd95_nonempty_mean", "tc")
METRIC_DIRECTION = {
    # +1 → higher is better; -1 → lower is better. Used only to annotate
    # the Wilcoxon output for readability.
    "dice_mean": +1,
    "hd95_mean": -1,
    "hd95_nonempty_mean": -1,
    "tc": +1,
}
METRIC_LABEL = {
    "dice_mean": "DSC",
    "hd95_mean": "HD95 (impute-aware, px)",
    "hd95_nonempty_mean": "HD95 (non-empty frames, px)",
    "tc": "Jt (inter-frame Jaccard)",
}

N_BOOT = 10000
SEED = 42
ALPHA = 0.05


# ── I/O ────────────────────────────────────────────────────────────────────


def load_per_sequence(model: str, mode: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{model}_{mode}" / "per_sequence_metrics.csv"
    if not path.is_file():
        print(f"[warn] missing {path}")
        return None
    return pd.read_csv(path)


def load_per_frame(model: str, mode: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{model}_{mode}" / "per_frame_metrics.csv"
    if not path.is_file():
        print(f"[warn] missing {path}")
        return None
    return pd.read_csv(path)


def patient_hd95_nonempty(model: str, mode: str) -> pd.DataFrame | None:
    """Per-patient HD95 mean over frames with is_empty_pred == False."""
    df = load_per_frame(model, mode)
    if df is None or len(df) == 0:
        return None
    valid = df[
        (~df["skipped"].astype(bool)) & (~df["is_empty_pred"].astype(bool))
    ].copy()
    if len(valid) == 0:
        return None
    return (
        valid.groupby("patient_id")["hd95"]
        .mean()
        .reset_index()
        .rename(columns={"hd95": "hd95_nonempty_mean"})
    )


def patient_table(model: str, mode: str) -> pd.DataFrame | None:
    """Merge per_sequence with per_frame-derived hd95_nonempty into one df."""
    seq = load_per_sequence(model, mode)
    if seq is None or len(seq) == 0:
        return None
    ne = patient_hd95_nonempty(model, mode)
    if ne is not None:
        seq = seq.merge(ne, on="patient_id", how="left")
    else:
        seq["hd95_nonempty_mean"] = np.nan
    return seq


# ── Stats ──────────────────────────────────────────────────────────────────


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap on per-patient values; (mean, ci_low, ci_high)."""
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


def paired_wilcoxon(
    fw: np.ndarray, vd: np.ndarray
) -> tuple[float, float, int, float]:
    """Two-sided paired Wilcoxon. Returns (W, p, n_pairs, median_delta).

    Pairs with NaN on either side are dropped before the test. Returns NaN
    p / W when fewer than 3 finite pairs remain (scipy's minimum sample size
    for a stable two-sided test).
    """
    fw = np.asarray(fw, dtype=float)
    vd = np.asarray(vd, dtype=float)
    mask = np.isfinite(fw) & np.isfinite(vd)
    fw = fw[mask]
    vd = vd[mask]
    n = len(fw)
    if n < 3:
        return float("nan"), float("nan"), n, float("nan")
    res = wilcoxon(vd, fw, alternative="two-sided", zero_method="wilcox")
    median_delta = float(np.median(vd - fw))
    return float(res.statistic), float(res.pvalue), n, median_delta


# ── Builders ───────────────────────────────────────────────────────────────


def build_bootstrap_table() -> pd.DataFrame:
    rows: list[dict] = []
    for m in MODELS:
        for mo in MODES:
            df = patient_table(m, mo)
            if df is None or len(df) == 0:
                for met in METRICS:
                    rows.append(
                        {
                            "model": m,
                            "mode": mo,
                            "metric": met,
                            "metric_label": METRIC_LABEL[met],
                            "n_patients": 0,
                            "mean": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "status": "missing",
                        }
                    )
                continue
            for met in METRICS:
                if met not in df.columns:
                    rows.append(
                        {
                            "model": m,
                            "mode": mo,
                            "metric": met,
                            "metric_label": METRIC_LABEL[met],
                            "n_patients": len(df),
                            "mean": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "status": "column_missing",
                        }
                    )
                    continue
                values = df[met].to_numpy(dtype=float)
                mean, lo, hi = bootstrap_ci(values)
                rows.append(
                    {
                        "model": m,
                        "mode": mo,
                        "metric": met,
                        "metric_label": METRIC_LABEL[met],
                        "n_patients": int(np.isfinite(values).sum()),
                        "mean": round(mean, 6),
                        "ci_low": round(lo, 6),
                        "ci_high": round(hi, 6),
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def build_wilcoxon_table() -> pd.DataFrame:
    rows: list[dict] = []
    p_vals: list[float] = []
    p_idx: list[int] = []
    for m in MODELS:
        fw = patient_table(m, "framewise")
        vd = patient_table(m, "video")
        if fw is None or vd is None:
            for met in METRICS:
                rows.append(
                    {
                        "model": m,
                        "metric": met,
                        "metric_label": METRIC_LABEL[met],
                        "n_pairs": 0,
                        "median_delta": np.nan,
                        "wilcoxon_W": np.nan,
                        "p_raw": np.nan,
                        "p_holm": np.nan,
                        "reject_holm": False,
                        "status": "missing",
                    }
                )
            continue
        merged = fw.merge(vd, on="patient_id", suffixes=("_fw", "_vd"))
        for met in METRICS:
            col_fw = f"{met}_fw"
            col_vd = f"{met}_vd"
            if col_fw not in merged.columns or col_vd not in merged.columns:
                rows.append(
                    {
                        "model": m,
                        "metric": met,
                        "metric_label": METRIC_LABEL[met],
                        "n_pairs": 0,
                        "median_delta": np.nan,
                        "wilcoxon_W": np.nan,
                        "p_raw": np.nan,
                        "p_holm": np.nan,
                        "reject_holm": False,
                        "status": "column_missing",
                    }
                )
                continue
            W, p, n, md = paired_wilcoxon(
                merged[col_fw].to_numpy(), merged[col_vd].to_numpy()
            )
            row_idx = len(rows)
            rows.append(
                {
                    "model": m,
                    "metric": met,
                    "metric_label": METRIC_LABEL[met],
                    "n_pairs": n,
                    "median_delta": round(md, 6) if np.isfinite(md) else np.nan,
                    "wilcoxon_W": round(W, 6) if np.isfinite(W) else np.nan,
                    "p_raw": p if np.isfinite(p) else np.nan,
                    "p_holm": np.nan,
                    "reject_holm": False,
                    "status": "ok",
                }
            )
            if np.isfinite(p):
                p_vals.append(p)
                p_idx.append(row_idx)

    if p_vals:
        reject, p_holm, _, _ = multipletests(p_vals, alpha=ALPHA, method="holm")
        for i, idx in enumerate(p_idx):
            rows[idx]["p_holm"] = float(p_holm[i])
            rows[idx]["reject_holm"] = bool(reject[i])

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Exp 5 statistics — bootstrap CIs + paired Wilcoxon (Holm-corrected)")
    print("=" * 72)

    boot = build_bootstrap_table()
    boot_path = RESULTS_DIR / "zeroshot_bootstrap_ci95.csv"
    boot.to_csv(boot_path, index=False)
    print(f"wrote {boot_path} ({len(boot)} rows)")

    wilcox = build_wilcoxon_table()
    wilcox_path = RESULTS_DIR / "zeroshot_wilcoxon.csv"
    wilcox.to_csv(wilcox_path, index=False)
    print(f"wrote {wilcox_path} ({len(wilcox)} rows)")

    print("\n--- Wilcoxon framewise vs video (median Δ = video − framewise) ---")
    print(
        wilcox[
            [
                "model",
                "metric",
                "n_pairs",
                "median_delta",
                "wilcoxon_W",
                "p_raw",
                "p_holm",
                "reject_holm",
            ]
        ].to_string(index=False)
    )

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
