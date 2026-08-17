#!/usr/bin/env python3
"""Build the label-efficiency ladder for the cine-clip experiment (Experiment 6).

Three supervision tiers per model in {sam2, medsam2, sam3}:
  tier 0   (zero-shot)   -> results/{model}_{framewise,video}/per_sequence_metrics.csv
  tier 5   (LoRA f=5%)   -> results/adapted_paired_per_patient.csv, fraction == 0.05
  tier 100 (LoRA f=100%) -> results/adapted_paired_per_patient.csv, fraction == 1.0

One collapse definition is applied identically to every tier: a patient collapses
iff delta_dsc = dsc_video - dsc_framewise <= -0.10. Any pre-existing 'collapsed'
column in a source CSV is ignored and recomputed here, so the three tiers are
never compared under different thresholds.

If a tier-0 per_sequence file is incomplete (fewer than the 29 test patients),
the framewise DSC for that model is taken from the canonical paired file
results/zeroshot_paired_per_patient.csv instead, which carries the same values.
The run asserts a full set of 29 patients per model and tier, so a partial input
fails loudly rather than silently shrinking the cohort.

Outputs:
    ladder_per_patient.csv        per-patient, per-tier deltas (local only: keyed
                                  by Stanford AIMI patient id, not redistributed)
    ladder_summary.csv            mean framewise / video DSC + collapse rate per
                                  (model, tier) — the reported cine-clip table
    ladder_5v100_wilcoxon.csv     paired Wilcoxon on video DSC, f=5% vs f=100%

Usage:
    python experiments/exp6_cineclip_video/build_ladder.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE / "results"

COLLAPSE_THR = -0.10
MODELS = ["sam2", "medsam2", "sam3"]
FRACTION_TO_TIER = {0.05: 5, 1.0: 100}


def _flag(delta: pd.Series) -> pd.Series:
    """Canonical collapse flag: delta_dsc <= -0.10."""
    return delta <= COLLAPSE_THR


def build_tier0() -> pd.DataFrame:
    """Zero-shot tier: pair framewise vs video per_sequence DSC by patient_id."""
    # Canonical paired file, used when a per_sequence file is incomplete.
    recover = pd.read_csv(R / "zeroshot_paired_per_patient.csv")

    rows = []
    for model in MODELS:
        video = pd.read_csv(R / f"{model}_video" / "per_sequence_metrics.csv")
        video = video[["patient_id", "dice_mean"]].rename(columns={"dice_mean": "dsc_video"})

        fw_path = R / f"{model}_framewise" / "per_sequence_metrics.csv"
        fw = pd.read_csv(fw_path)[["patient_id", "dice_mean"]].rename(
            columns={"dice_mean": "dsc_framewise"}
        )
        if len(fw) < 29:
            # Incomplete: take framewise DSC from the canonical paired file.
            rec = recover[recover.model == model][["patient_id", "dsc_framewise"]]
            assert len(rec) == 29, f"recovery source incomplete for {model}"
            fw = rec.copy()
            print(
                f"[tier0] {fw_path} is incomplete -> taking framewise DSC for "
                f"29 patients from zeroshot_paired_per_patient.csv instead"
            )

        merged = fw.merge(video, on="patient_id", how="inner")
        assert len(merged) == 29, f"tier0 {model}: expected 29 patients, got {len(merged)}"
        merged["model"] = model
        merged["tier"] = 0
        merged["delta_dsc"] = merged["dsc_video"] - merged["dsc_framewise"]
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


def build_adapted() -> pd.DataFrame:
    """Tiers 5% and 100% from the already-paired adapted per-patient file."""
    a = pd.read_csv(R / "adapted_paired_per_patient.csv")
    out = []
    for frac, tier in FRACTION_TO_TIER.items():
        sub = a[np.isclose(a.fraction, frac)].copy()
        for model in MODELS:
            g = sub[sub.model == model][["patient_id", "dsc_framewise", "dsc_video"]].copy()
            assert len(g) == 29, f"tier{tier} {model}: expected 29, got {len(g)}"
            # Recompute delta from DSC columns (do not trust source delta/collapsed).
            g["delta_dsc"] = g["dsc_video"] - g["dsc_framewise"]
            g["model"] = model
            g["tier"] = tier
            out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    tier0 = build_tier0()
    adapted = build_adapted()

    ladder = pd.concat([tier0, adapted], ignore_index=True)
    ladder["collapsed"] = _flag(ladder["delta_dsc"])
    ladder = ladder[
        ["patient_id", "model", "tier", "dsc_framewise", "dsc_video", "delta_dsc", "collapsed"]
    ]
    # Order rows model -> tier -> patient for readability.
    ladder["model"] = pd.Categorical(ladder["model"], categories=MODELS, ordered=True)
    ladder = ladder.sort_values(["model", "tier", "patient_id"]).reset_index(drop=True)
    assert len(ladder) == 261, f"expected 261 rows, got {len(ladder)}"

    per_patient_path = R / "ladder_per_patient.csv"
    ladder.to_csv(per_patient_path, index=False)

    # ---- Summary per (model, tier) ----
    summ_rows = []
    for model in MODELS:
        for tier in [0, 5, 100]:
            g = ladder[(ladder.model == model) & (ladder.tier == tier)]
            summ_rows.append(
                {
                    "model": model,
                    "tier": tier,
                    "n": len(g),
                    "dsc_framewise_mean": g.dsc_framewise.mean(),
                    "dsc_video_mean": g.dsc_video.mean(),
                    "dsc_video_median": g.dsc_video.median(),
                    "delta_dsc_mean": g.delta_dsc.mean(),
                    "delta_dsc_median": g.delta_dsc.median(),
                    "collapse_rate_pct": 100.0 * g.collapsed.sum() / len(g),
                }
            )
    summary = pd.DataFrame(summ_rows)
    summary_path = R / "ladder_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ---- Paired Wilcoxon: 5% vs 100% on per-patient VIDEO DSC ----
    wil_rows = []
    for model in MODELS:
        v5 = ladder[(ladder.model == model) & (ladder.tier == 5)][
            ["patient_id", "dsc_video"]
        ].rename(columns={"dsc_video": "v5"})
        v100 = ladder[(ladder.model == model) & (ladder.tier == 100)][
            ["patient_id", "dsc_video"]
        ].rename(columns={"dsc_video": "v100"})
        m = v5.merge(v100, on="patient_id", how="inner")
        assert len(m) == 29, f"wilcoxon {model}: expected 29 pairs, got {len(m)}"
        diff = m["v100"] - m["v5"]
        # scipy wilcoxon needs non-all-zero differences; guard just in case.
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(m["v100"], m["v5"])
        wil_rows.append(
            {
                "model": model,
                "n_pairs": len(m),
                "median_diff": float(np.median(diff)),  # dsc_video@100 - dsc_video@5
                "p_value": float(p),
            }
        )
    wil = pd.DataFrame(wil_rows)
    wil_path = R / "ladder_5v100_wilcoxon.csv"
    wil.to_csv(wil_path, index=False)

    # ---- Verification prints ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== ladder_summary.csv ===")
    print(summary.to_string(index=False))
    print("\n=== ladder_5v100_wilcoxon.csv (video DSC, 100% - 5%) ===")
    print(wil.to_string(index=False))

    print("\n=== collapse_rate_pct (model x tier) ===")
    piv = summary.pivot(index="model", columns="tier", values="collapse_rate_pct")
    piv = piv.reindex(MODELS)
    print(piv.to_string())

    # Sanity check: tier-0 sam2 video mean from per_sequence vs summary.
    ps = pd.read_csv(R / "sam2_video" / "per_sequence_metrics.csv")["dice_mean"].mean()
    got = summary[(summary.model == "sam2") & (summary.tier == 0)]["dsc_video_mean"].iloc[0]
    print("\n=== SANITY: sam2 tier-0 video mean DSC ===")
    print(f"  per_sequence sam2_video dice_mean mean : {ps:.6f}")
    print(f"  ladder summary dsc_video_mean          : {got:.6f}")
    print(f"  match (<1e-9)                          : {abs(ps - got) < 1e-9}")

    print("\nWritten:")
    for p in [per_patient_path, summary_path, wil_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
