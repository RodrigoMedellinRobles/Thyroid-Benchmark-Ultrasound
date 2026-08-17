#!/usr/bin/env python3
"""
Aggregate Experiment 6 (cine-clip video-mode evaluation) outputs.

Walks `experiments/exp6_cineclip_video/results/{model}_{mode}/summary.json` plus the
per-frame and per-sequence CSVs for every (model, mode) cell of the 2x2
matrix and produces:

  experiments/exp6_cineclip_video/results/zeroshot_summary.csv
      Headline table: one row per (model, mode), columns:
      DSC mean/std (impute-aware), IoU mean/std, HD95 mean/std,
      inter-frame Jaccard Jt mean/std (temporal coherence),
      FPS, peak VRAM (GB), n_empty_pred, empty_pred_pct.

  experiments/exp6_cineclip_video/results/zeroshot_paired_per_patient.csv
      Per-patient framewise vs video comparison for each model:
      patient_id, model, n_frames,
      dsc_framewise, dsc_video, delta_dsc,
      tc_framewise,  tc_video,  delta_tc,
      collapsed_bool (delta_dsc <= -0.10).

  experiments/exp6_cineclip_video/results/zeroshot_collapse_summary.csv
      Per-model collapse rate aggregate:
      model, n_patients, n_collapsed, collapse_rate_pct.

Usage:
    python experiments/exp6_cineclip_video/aggregate_zeroshot.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"

MODELS  = ("sam2", "medsam2", "sam3")
MODES   = ("framewise", "video")
COLLAPSE_THRESHOLD = -0.10  # delta_dsc (video - framewise); below = collapse


def load_summary(model: str, mode: str) -> Optional[dict]:
    p = RESULTS_DIR / f"{model}_{mode}" / "summary.json"
    if not p.is_file():
        print(f"[warn] missing {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_per_sequence(model: str, mode: str) -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / f"{model}_{mode}" / "per_sequence_metrics.csv"
    if not p.is_file():
        print(f"[warn] missing {p}")
        return None
    return pd.read_csv(p)


def load_per_frame(model: str, mode: str) -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / f"{model}_{mode}" / "per_frame_metrics.csv"
    if not p.is_file():
        print(f"[warn] missing {p}")
        return None
    return pd.read_csv(p)


def patient_hd95_nonempty(model: str, mode: str) -> Optional[pd.DataFrame]:
    """Patient-level HD95 mean computed over **non-empty predicted frames only**.

    The per-sequence ``hd95_mean`` column is impute-aware: empty predictions
    are replaced with the image diagonal (≈724 px). For modes with high
    empty-prediction rates (SAM2 video on Stanford AIMI hits 21 %) that
    inflates the per-sequence HD95 to a value that conflates spatial
    accuracy with the failure rate.

    This function reads ``per_frame_metrics.csv``, drops skipped /
    is_empty_pred=True rows, and emits a per-patient HD95 mean reflecting
    spatial accuracy alone. The empty-prediction count is reported as a
    separate column in :func:`build_headline_table`.
    """
    df = load_per_frame(model, mode)
    if df is None or len(df) == 0:
        return None
    valid = df[(~df["skipped"].astype(bool)) & (~df["is_empty_pred"].astype(bool))].copy()
    if len(valid) == 0:
        return None
    return (
        valid.groupby("patient_id")["hd95"]
        .agg(hd95_nonempty_mean="mean", hd95_nonempty_n="count")
        .reset_index()
    )


def build_headline_table() -> pd.DataFrame:
    """Headline table. mean ± std are computed across PER-PATIENT means
    (not per-frame), matching the patient-level aggregation policy of
    §sec:methods:metrics. HD95 is reported twice: ``hd95_mean`` /
    ``hd95_std`` are the impute-aware per-sequence aggregates (empty
    predictions replaced with the image diagonal) used by the bootstrap
    and Wilcoxon stats; ``hd95_nonempty_mean`` / ``hd95_nonempty_std``
    are computed over non-empty predicted frames only and report spatial
    accuracy without conflating it with the empty-prediction rate. FPS /
    VRAM / empty-pred counts come from summary.json."""
    rows = []
    for m in MODELS:
        for mo in MODES:
            s   = load_summary(m, mo)
            seq = load_per_sequence(m, mo)
            if s is None or seq is None or len(seq) == 0:
                rows.append({
                    "model": m, "mode": mo, "status": "missing",
                    "n_patients": np.nan, "n_frames": np.nan,
                    "dice_mean": np.nan, "dice_std": np.nan,
                    "iou_mean":  np.nan, "iou_std":  np.nan,
                    "hd95_mean": np.nan, "hd95_std": np.nan,
                    "hd95_nonempty_mean": np.nan, "hd95_nonempty_std": np.nan,
                    "tc_mean":   np.nan, "tc_std":   np.nan,
                    "fps":       np.nan, "vram_gb":  np.nan,
                    "n_empty_pred":   np.nan, "empty_pred_pct": np.nan,
                })
                continue
            ne = patient_hd95_nonempty(m, mo)
            if ne is not None and len(ne) > 0:
                hd_ne_mean = float(ne["hd95_nonempty_mean"].mean())
                hd_ne_std  = float(ne["hd95_nonempty_mean"].std(ddof=1)) if len(ne) > 1 else 0.0
            else:
                hd_ne_mean = np.nan
                hd_ne_std  = np.nan
            rows.append({
                "model": m, "mode": mo, "status": "ok",
                "n_patients": int(len(seq)),
                "n_frames":   s.get("n_frames"),
                # Patient-level mean ± std (NaN-safe)
                "dice_mean":  float(seq["dice_mean"].mean()),
                "dice_std":   float(seq["dice_mean"].std(ddof=1)) if len(seq) > 1 else 0.0,
                "iou_mean":   float(seq["iou_mean"].mean()),
                "iou_std":    float(seq["iou_mean"].std(ddof=1))  if len(seq) > 1 else 0.0,
                "hd95_mean":  float(seq["hd95_mean"].mean()),
                "hd95_std":   float(seq["hd95_mean"].std(ddof=1)) if len(seq) > 1 else 0.0,
                "hd95_nonempty_mean": hd_ne_mean,
                "hd95_nonempty_std":  hd_ne_std,
                "tc_mean":    float(seq["tc"].mean()),
                "tc_std":     float(seq["tc"].std(ddof=1))        if len(seq) > 1 else 0.0,
                "fps":        s.get("fps"),
                "vram_gb":    s.get("vram_gb"),
                "n_empty_pred":   s.get("n_empty_pred", 0),
                "empty_pred_pct": s.get("empty_pred_pct", 0.0),
            })
    return pd.DataFrame(rows)


def build_paired_table() -> pd.DataFrame:
    """Per-patient framewise vs video DSC and TC, with collapse flag."""
    rows = []
    for m in MODELS:
        fw = load_per_sequence(m, "framewise")
        vd = load_per_sequence(m, "video")
        if fw is None or vd is None:
            continue
        # Outer join with indicator so patients present in only ONE mode are
        # not silently dropped. video-only-failure patients = collapsed by
        # default (the deployment scenario where video crashed but framewise
        # would have produced a number).
        merged = fw.merge(vd, on="patient_id", suffixes=("_fw", "_vd"),
                          how="outer", indicator=True)
        for _, r in merged.iterrows():
            present_fw = (r["_merge"] in ("both", "left_only"))
            present_vd = (r["_merge"] in ("both", "right_only"))
            dsc_fw = float(r["dice_mean_fw"]) if present_fw and not pd.isna(r["dice_mean_fw"]) else np.nan
            dsc_vd = float(r["dice_mean_vd"]) if present_vd and not pd.isna(r["dice_mean_vd"]) else np.nan
            if np.isnan(dsc_fw) or np.isnan(dsc_vd):
                d_dsc = np.nan
                collapsed = bool(present_fw and not present_vd)  # video missing = treated as collapse
            else:
                d_dsc = dsc_vd - dsc_fw
                collapsed = bool(d_dsc <= COLLAPSE_THRESHOLD)
            tc_fw = float(r["tc_fw"]) if present_fw and not pd.isna(r["tc_fw"]) else np.nan
            tc_vd = float(r["tc_vd"]) if present_vd and not pd.isna(r["tc_vd"]) else np.nan
            d_tc  = (tc_vd - tc_fw) if (not np.isnan(tc_fw) and not np.isnan(tc_vd)) else np.nan
            rows.append({
                "patient_id":    r["patient_id"],
                "model":         m,
                "present_in":    str(r["_merge"]),
                "n_frames":      int(min(
                    r["n_frames_fw"] if present_fw and not pd.isna(r["n_frames_fw"]) else 0,
                    r["n_frames_vd"] if present_vd and not pd.isna(r["n_frames_vd"]) else 0,
                )),
                "dsc_framewise": round(dsc_fw, 4) if not np.isnan(dsc_fw) else np.nan,
                "dsc_video":     round(dsc_vd, 4) if not np.isnan(dsc_vd) else np.nan,
                "delta_dsc":     round(d_dsc, 4) if not np.isnan(d_dsc) else np.nan,
                "tc_framewise":  round(tc_fw, 4) if not np.isnan(tc_fw) else np.nan,
                "tc_video":      round(tc_vd, 4) if not np.isnan(tc_vd) else np.nan,
                "delta_tc":      round(d_tc, 4) if not np.isnan(d_tc) else np.nan,
                "collapsed":     collapsed,
            })
    return pd.DataFrame(rows)


def build_collapse_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        sub = paired[paired["model"] == m]
        if len(sub) == 0:
            rows.append({"model": m, "n_patients": 0,
                         "n_collapsed": 0, "collapse_rate_pct": np.nan})
            continue
        n_total = len(sub)
        n_collapsed = int(sub["collapsed"].sum())
        rows.append({
            "model": m,
            "n_patients": n_total,
            "n_collapsed": n_collapsed,
            "collapse_rate_pct": round(100.0 * n_collapsed / n_total, 2),
        })
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Exp 5 aggregation — 2x2 (model x mode) headline + paired analysis")
    print("=" * 72)

    headline = build_headline_table()
    headline_path = out_dir / "zeroshot_summary.csv"
    headline.to_csv(headline_path, index=False)
    print(f"wrote {headline_path} ({len(headline)} rows)")
    print(headline.to_string(index=False))

    paired = build_paired_table()
    paired_path = out_dir / "zeroshot_paired_per_patient.csv"
    paired.to_csv(paired_path, index=False)
    print(f"\nwrote {paired_path} ({len(paired)} rows)")

    collapse = build_collapse_summary(paired)
    collapse_path = out_dir / "zeroshot_collapse_summary.csv"
    collapse.to_csv(collapse_path, index=False)
    print(f"wrote {collapse_path}")
    print(collapse.to_string(index=False))

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
