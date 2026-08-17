#!/usr/bin/env python3
"""Aggregate the LoRA-adapted Experiment 6 cine-clip runs.

Reads the fraction-suffixed output dirs produced by ``run.py --fraction``:
    experiments/exp6_cineclip_video/results/{model}_{mode}_f{tag}/  (tag: 0.05 or 1)
for the 3x2x2 grid (model x fraction x mode) and writes, WITHOUT touching the
zero-shot tables written by aggregate_zeroshot.py:

    adapted_summary.csv              headline per (model, fraction, mode)
    adapted_paired_per_patient.csv   per-patient framewise vs video + Delta
    adapted_collapse_summary.csv     per (model, fraction) collapse rate

Logic mirrors aggregate_zeroshot.py: patient-level mean +/- std, impute-aware
plus non-empty HD95, and a collapse threshold of -0.10 on the framewise-to-video
DSC delta.

adapted_paired_per_patient.csv keys rows by Stanford AIMI patient id, so it is a
local intermediate only and is not redistributed with this repository.

Usage:
    python experiments/exp6_cineclip_video/aggregate_adapted.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"

MODELS = ("sam2", "medsam2", "sam3")
MODES = ("framewise", "video")
FRACTIONS = (0.05, 1.0)
COLLAPSE_THRESHOLD = -0.10


def _tag(fraction: float) -> str:
    return f"{fraction:g}"  # 0.05 -> "0.05", 1.0 -> "1"


def _dir(model: str, mode: str, fraction: float) -> Path:
    return RESULTS_DIR / f"{model}_{mode}_f{_tag(fraction)}"


def load_summary(model: str, mode: str, fraction: float) -> Optional[dict]:
    import json
    p = _dir(model, mode, fraction) / "summary.json"
    if not p.is_file():
        print(f"[warn] missing {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_per_sequence(model: str, mode: str, fraction: float) -> Optional[pd.DataFrame]:
    p = _dir(model, mode, fraction) / "per_sequence_metrics.csv"
    if not p.is_file():
        print(f"[warn] missing {p}")
        return None
    return pd.read_csv(p)


def load_per_frame(model: str, mode: str, fraction: float) -> Optional[pd.DataFrame]:
    p = _dir(model, mode, fraction) / "per_frame_metrics.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def patient_hd95_nonempty(model: str, mode: str, fraction: float) -> Optional[pd.DataFrame]:
    df = load_per_frame(model, mode, fraction)
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
    rows = []
    for m in MODELS:
        for fr in FRACTIONS:
            for mo in MODES:
                s = load_summary(m, mo, fr)
                seq = load_per_sequence(m, mo, fr)
                if s is None or seq is None or len(seq) == 0:
                    rows.append({
                        "model": m, "fraction": fr, "mode": mo, "status": "missing",
                        "n_patients": np.nan, "n_frames": np.nan,
                        "dice_mean": np.nan, "dice_std": np.nan,
                        "iou_mean": np.nan, "iou_std": np.nan,
                        "hd95_mean": np.nan, "hd95_std": np.nan,
                        "hd95_nonempty_mean": np.nan, "hd95_nonempty_std": np.nan,
                        "tc_mean": np.nan, "tc_std": np.nan,
                        "fps": np.nan, "vram_gb": np.nan,
                        "n_empty_pred": np.nan, "empty_pred_pct": np.nan,
                    })
                    continue
                ne = patient_hd95_nonempty(m, mo, fr)
                if ne is not None and len(ne) > 0:
                    hd_ne_mean = float(ne["hd95_nonempty_mean"].mean())
                    hd_ne_std = float(ne["hd95_nonempty_mean"].std(ddof=1)) if len(ne) > 1 else 0.0
                else:
                    hd_ne_mean = np.nan
                    hd_ne_std = np.nan
                rows.append({
                    "model": m, "fraction": fr, "mode": mo, "status": "ok",
                    "n_patients": int(len(seq)),
                    "n_frames": s.get("n_frames"),
                    "dice_mean": float(seq["dice_mean"].mean()),
                    "dice_std": float(seq["dice_mean"].std(ddof=1)) if len(seq) > 1 else 0.0,
                    "iou_mean": float(seq["iou_mean"].mean()),
                    "iou_std": float(seq["iou_mean"].std(ddof=1)) if len(seq) > 1 else 0.0,
                    "hd95_mean": float(seq["hd95_mean"].mean()),
                    "hd95_std": float(seq["hd95_mean"].std(ddof=1)) if len(seq) > 1 else 0.0,
                    "hd95_nonempty_mean": hd_ne_mean,
                    "hd95_nonempty_std": hd_ne_std,
                    "tc_mean": float(seq["tc"].mean()),
                    "tc_std": float(seq["tc"].std(ddof=1)) if len(seq) > 1 else 0.0,
                    "fps": s.get("fps"),
                    "vram_gb": s.get("vram_gb"),
                    "n_empty_pred": s.get("n_empty_pred", 0),
                    "empty_pred_pct": s.get("empty_pred_pct", 0.0),
                })
    return pd.DataFrame(rows)


def build_paired_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for fr in FRACTIONS:
            fw = load_per_sequence(m, "framewise", fr)
            vd = load_per_sequence(m, "video", fr)
            if fw is None or vd is None:
                continue
            merged = fw.merge(vd, on="patient_id", suffixes=("_fw", "_vd"),
                              how="outer", indicator=True)
            for _, r in merged.iterrows():
                present_fw = (r["_merge"] in ("both", "left_only"))
                present_vd = (r["_merge"] in ("both", "right_only"))
                dsc_fw = float(r["dice_mean_fw"]) if present_fw and not pd.isna(r["dice_mean_fw"]) else np.nan
                dsc_vd = float(r["dice_mean_vd"]) if present_vd and not pd.isna(r["dice_mean_vd"]) else np.nan
                if np.isnan(dsc_fw) or np.isnan(dsc_vd):
                    d_dsc = np.nan
                    collapsed = bool(present_fw and not present_vd)
                else:
                    d_dsc = dsc_vd - dsc_fw
                    collapsed = bool(d_dsc <= COLLAPSE_THRESHOLD)
                tc_fw = float(r["tc_fw"]) if present_fw and not pd.isna(r["tc_fw"]) else np.nan
                tc_vd = float(r["tc_vd"]) if present_vd and not pd.isna(r["tc_vd"]) else np.nan
                d_tc = (tc_vd - tc_fw) if (not np.isnan(tc_fw) and not np.isnan(tc_vd)) else np.nan
                rows.append({
                    "patient_id": r["patient_id"], "model": m, "fraction": fr,
                    "present_in": str(r["_merge"]),
                    "dsc_framewise": round(dsc_fw, 4) if not np.isnan(dsc_fw) else np.nan,
                    "dsc_video": round(dsc_vd, 4) if not np.isnan(dsc_vd) else np.nan,
                    "delta_dsc": round(d_dsc, 4) if not np.isnan(d_dsc) else np.nan,
                    "tc_framewise": round(tc_fw, 4) if not np.isnan(tc_fw) else np.nan,
                    "tc_video": round(tc_vd, 4) if not np.isnan(tc_vd) else np.nan,
                    "delta_tc": round(d_tc, 4) if not np.isnan(d_tc) else np.nan,
                    "collapsed": collapsed,
                })
    return pd.DataFrame(rows)


def build_collapse_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for fr in FRACTIONS:
            sub = paired[(paired["model"] == m) & (paired["fraction"] == fr)]
            if len(sub) == 0:
                rows.append({"model": m, "fraction": fr, "n_patients": 0,
                             "n_collapsed": 0, "collapse_rate_pct": np.nan})
                continue
            n_total = len(sub)
            n_collapsed = int(sub["collapsed"].sum())
            rows.append({
                "model": m, "fraction": fr, "n_patients": n_total,
                "n_collapsed": n_collapsed,
                "collapse_rate_pct": round(100.0 * n_collapsed / n_total, 2),
            })
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Experiment 6 adapted aggregation - 3 models x 2 fractions x 2 modes")
    print("=" * 72)

    headline = build_headline_table()
    hp = RESULTS_DIR / "adapted_summary.csv"
    headline.to_csv(hp, index=False)
    print(f"wrote {hp} ({len(headline)} rows)")
    print(headline.to_string(index=False))

    paired = build_paired_table()
    pp = RESULTS_DIR / "adapted_paired_per_patient.csv"
    paired.to_csv(pp, index=False)
    print(f"\nwrote {pp} ({len(paired)} rows)")

    collapse = build_collapse_summary(paired)
    cp = RESULTS_DIR / "adapted_collapse_summary.csv"
    collapse.to_csv(cp, index=False)
    print(f"wrote {cp}")
    print(collapse.to_string(index=False))
    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
