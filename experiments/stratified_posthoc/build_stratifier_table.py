"""
Build unified stratifier table for Exp 7 (post-hoc stratified analysis).

Reads from ORIGINAL data paths (data/raw/extracted/...) for metadata + splits,
discovers per-image metrics CSVs under experiments/, and emits a single
long-format table with one row per (image, model) pair plus all stratifier
covariates (TIRADS, size_mm, age, sex).

Stanford AIMI is reduced to per-nodule median (frames within a cine-clip share
the same nodule_id; we collapse to median DSC/HD95 per nodule).

Full-supervised CNN baselines are read from the BOX-CONDITIONED evaluations
(oracle box as an extra input channel), matching Table 1 / Supp S1, so the
stratified figure compares like-for-like (all models prompted) against the LoRA
foundation models. There are no image-only reads.

Output:
    experiments/exp7_stratified/results/stratifier_table_long.csv

Usage:
    python build_stratifier_table.py [--fewshot-fraction 0.05]
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Original data paths (NOT copies under experiments/exp7_stratified/metadata/)
PATH_THYROIDXL_META = REPO_ROOT / "data/raw/extracted/ThyroidXL/stats/id2info_eng.json"
PATH_STANFORD_META = REPO_ROOT / "data/raw/extracted/Stanford/metadata.csv"
PATH_DDTI_XML_DIR = REPO_ROOT / "data/raw/extracted/DDTI-Pedraza_Digital_Database_Thyroid_SPIE"
PATH_SPLITS = REPO_ROOT / "data/splits"

# Per-experiment results roots
PATH_EXP3 = REPO_ROOT / "experiments/exp3_fewshot_lora/results"
PATH_EXP9 = REPO_ROOT / "experiments/exp1_fullsupervised/foundation_lora/results"

# Box-conditioned full-supervised CNN sources. The main paper defines the
# full-supervised CNN baselines as BOX-CONDITIONED (oracle box as an extra
# input channel; Methods "Unified protocol", Table 1 / Supp S1). We read the
# canonical box-cond per-image CSVs so the stratified figure matches Table 1 and
# gives a like-for-like (all-prompted) comparison against the LoRA foundation
# models. U-Net/TransUNet in-domain per-image DSC/HD95 come from the box-cond
# CNN evaluator (src==tgt cells); nnU-Net from its box-cond details CSVs.
#
# The U-Net/TransUNet per-image CSVs are NOT shipped in this repository: they
# carry gated (ThyroidXL / Stanford AIMI) image filenames. Regenerate them by
# running the box-conditioned CNN evaluation on your locally placed data; drop
# the resulting {name}_src{ds}_tgt{ds}_per_image.csv files into the directory
# below and re-run this script.
PATH_BOXCOND_CNN = REPO_ROOT / "experiments/exp1_fullsupervised/traditional_boxcond/results/boxcond_cnn_per_image"
PATH_NNUNET_BOXCOND = REPO_ROOT / "experiments/exp1_fullsupervised/traditional_boxcond/results/nnunet"
_BOXCOND_CNN_NAME = {"unet": "unet_resnet50",
                     "transunet": "transunet_r50_vitb16"}

# Output
PATH_OUT = REPO_ROOT / "experiments/exp7_stratified/results/stratifier_table_long.csv"

# 2026-07 reconfig (Exp 7 stratified re-run):
#   - datasets restricted to ThyroidXL + Stanford AIMI (DDTI dropped from
#     the main table/figure);
#   - SAM (ViT-H) dropped entirely;
#   - foundation shown at few-shot f=0.05 (Exp 3) AND full-supervision
#     f=1.0 (Exp 9), instead of a single f=0.25 regime;
#   - zero-shot foundation no longer wired into the main long table.
DATASETS = ["thyroidxl"]  # only ThyroidXL has enough patients per bin to stratify
FOUNDATION_MODELS = ["sam2", "medsam", "medsam2", "sam3"]  # SAM (ViT-H) dropped

# HD95 imputation floor for non-finite values (empty prediction or empty GT):
# image diagonal in pixels sqrt(2)*512, matching the paper policy.
HD95_IMPUTE = float(np.sqrt(2.0) * 512.0)  # 724.0773...


# ---------------------------------------------------------------------------
# Metadata loaders (one per dataset)
# ---------------------------------------------------------------------------
def load_thyroidxl_metadata() -> pd.DataFrame:
    """Return DataFrame with one row per patient. Columns: patient_id, age,
    sex, tirads, size_mm. Age outliers (>110) are dropped."""
    raw = json.loads(PATH_THYROIDXL_META.read_text())
    rows = []
    for pid, v in raw.items():
        n1 = v.get("nodule_1") or {}
        age = v.get("age")
        try:
            age = int(age) if age is not None else None
        except (TypeError, ValueError):
            age = None
        if age is not None and age > 110:
            age = None  # outlier (one record had age=2005)
        gender = v.get("gender")
        sex = {1: "M", 2: "F"}.get(gender)
        tirads = n1.get("TIRADS")
        w = n1.get("Width")
        h = n1.get("Height")
        # max in-plane axis (mm). Width/Height already in mm.
        size_mm: Optional[float] = None
        vals = [x for x in (w, h) if x is not None]
        if vals:
            size_mm = float(max(vals))
        rows.append(dict(patient_id=str(pid), age=age, sex=sex,
                         tirads=tirads, size_mm=size_mm))
    df = pd.DataFrame(rows)
    df["dataset"] = "thyroidxl"
    return df


def load_stanford_metadata() -> pd.DataFrame:
    """Return DataFrame with one row per nodule. Columns: patient_id (=annot_id),
    age, sex, tirads, size_mm (max in-plane = max(size_x, size_y) in mm;
    size_z is depth/AP, excluded for cross-dataset comparability).
    Also keeps composition sub-score for Appendix D."""
    df_raw = pd.read_csv(PATH_STANFORD_META)
    df_raw = df_raw.rename(columns={"annot_id": "patient_id",
                                    "ti-rads_level": "tirads",
                                    "ti-rads_composition": "composition"})
    # max in-plane = max(size_x, size_y) in cm -> mm
    df_raw["size_mm"] = df_raw[["size_x", "size_y"]].max(axis=1) * 10.0
    df_raw["sex"] = df_raw["sex"].map({"Female": "F", "Male": "M"})
    out = df_raw[["patient_id", "age", "sex", "tirads", "size_mm",
                  "composition"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["dataset"] = "stanford_aimi"
    return out


def load_ddti_metadata() -> pd.DataFrame:
    """Parse 390 DDTI XMLs into a DataFrame. Columns: patient_id (=case number),
    age, sex, tirads (raw Pedraza values incl. 4a/4b/4c). No size in mm
    (no pixel spacing in XML). Missing values preserved as NaN."""
    rows = []
    for xml_path in sorted(PATH_DDTI_XML_DIR.glob("*.xml")):
        case_id = xml_path.stem
        root = ET.parse(xml_path).getroot()

        def text(field: str) -> Optional[str]:
            el = root.find(field)
            return el.text.strip() if (el is not None and el.text and el.text.strip()) else None

        age_s = text("age")
        try:
            age = int(age_s) if age_s else None
        except ValueError:
            age = None
        sex_raw = text("sex")
        sex = {"F": "F", "M": "M"}.get(sex_raw)  # 'u' and missing -> None
        tirads = text("tirads")  # raw: '2','3','4a','4b','4c','5'
        rows.append(dict(patient_id=str(case_id), age=age, sex=sex,
                         tirads=tirads, size_mm=np.nan, composition=None))
    df = pd.DataFrame(rows)
    df["dataset"] = "ddti"
    return df


# ---------------------------------------------------------------------------
# Splits + patient_id extraction
# ---------------------------------------------------------------------------
def load_test_splits() -> pd.DataFrame:
    """Concatenate test splits for the included datasets. Standardizes
    patient_id to match metadata join key (preserves leading zeros, trailing
    underscores, etc.)."""
    frames = []
    for ds in DATASETS:
        csv = PATH_SPLITS / f"{ds}_test.csv"
        df = pd.read_csv(csv)
        df["dataset"] = ds
        df["patient_id"] = df["patient_id"].astype(str)
        frames.append(df[["dataset", "patient_id", "filename"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-image metrics discovery
# ---------------------------------------------------------------------------
# Schema A: unet/transunet image-only  -> case_id, filename, dice, ..., hd95
# Schema B: nnunet                     -> case_id, original_filename, dice, ..., hd95
# Schema C: foundation / box-cond CNN  -> filename, patient_id, dice, ..., hd95, skipped
SCHEMA_FILENAME_COL = {"A": "filename", "B": "original_filename", "C": "filename"}


def _read_per_image_csv(csv_path: Path, schema: str) -> pd.DataFrame:
    """Read a per-image CSV and standardize to (filename, dice, hd95).
    Drops rows flagged 'skipped' (foundation models occasionally skip
    inference when the box prompt collapses).

    Non-finite HD95 (empty prediction or empty GT) is imputed to the image
    diagonal HD95_IMPUTE = sqrt(2)*512 px BEFORE any downstream aggregation,
    matching the paper HD95 policy."""
    df = pd.read_csv(csv_path)
    fcol = SCHEMA_FILENAME_COL[schema]
    if fcol not in df.columns:
        raise ValueError(f"Expected column '{fcol}' not in {csv_path}")
    if "skipped" in df.columns:
        df = df.loc[~df["skipped"].astype(bool)].reset_index(drop=True)
    out = df[[fcol, "dice", "hd95"]].rename(columns={fcol: "filename"})
    out["hd95"] = pd.to_numeric(out["hd95"], errors="coerce")
    out.loc[~np.isfinite(out["hd95"]), "hd95"] = HD95_IMPUTE
    return out


def discover_per_image_metrics(fewshot_fraction: float = 0.05) -> pd.DataFrame:
    """Walk Exp1/3/9 result trees and return long DataFrame:
        dataset, model, regime, filename, dice, hd95.

    Regimes emitted (2026-07 reconfig):
        - traditional (unet/transunet/nnunet) full-supervised  -> 'fullsup'
        - foundation few-shot LoRA at ``fewshot_fraction``      -> 'fewshot_f{f}'
        - foundation full-supervised (Exp 9, f=1.0)            -> 'fullsup'
    Zero-shot (Exp 2) and SAM (ViT-H) are intentionally excluded here.
    """
    rows = []

    # ---- Exp 1: full-supervised traditional (BOX-CONDITIONED, matches Table 1) ----
    # The main paper's full-supervised CNN baselines are box-conditioned; we read
    # those per-image CSVs (not any image-only outputs) so the stratified figure
    # is consistent with Table 1 / Supp S1 and compares like-for-like (all models
    # prompted) against the LoRA foundation models.
    for ds in DATASETS:
        # unet (in-domain box-cond, schema C: filename, dice, hd95, skipped)
        p = PATH_BOXCOND_CNN / f"{_BOXCOND_CNN_NAME['unet']}_src{ds}_tgt{ds}_per_image.csv"
        if p.exists():
            df = _read_per_image_csv(p, "C")
            df["model"], df["regime"], df["dataset"] = "unet", "fullsup", ds
            rows.append(df)
        # transunet (in-domain box-cond, schema C)
        p = PATH_BOXCOND_CNN / f"{_BOXCOND_CNN_NAME['transunet']}_src{ds}_tgt{ds}_per_image.csv"
        if p.exists():
            df = _read_per_image_csv(p, "C")
            df["model"], df["regime"], df["dataset"] = "transunet", "fullsup", ds
            rows.append(df)
        # nnunet box-cond details (schema B: original_filename, dice, hd95)
        p = PATH_NNUNET_BOXCOND / f"nnunet_boxcond_results_{ds}_details.csv"
        if p.exists():
            df = _read_per_image_csv(p, "B")
            df["model"], df["regime"], df["dataset"] = "nnunet", "fullsup", ds
            rows.append(df)

    # ---- Exp 3: few-shot LoRA foundation (f=fewshot_fraction) ----
    for ds in DATASETS:
        for m in FOUNDATION_MODELS:
            p = PATH_EXP3 / f"fewshot_{m}_{ds}_f{fewshot_fraction}/per_image_results.csv"
            if p.exists():
                df = _read_per_image_csv(p, "C")
                df["model"], df["regime"], df["dataset"] = (
                    m, f"fewshot_f{fewshot_fraction}", ds)
                rows.append(df)

    # ---- Exp 9: full-supervised foundation (f=1.0) ----
    for ds in DATASETS:
        for m in FOUNDATION_MODELS:
            p = PATH_EXP9 / f"fullsup_{m}_{ds}/per_image_results.csv"
            if p.exists():
                df = _read_per_image_csv(p, "C")
                df["model"], df["regime"], df["dataset"] = m, "fullsup", ds
                rows.append(df)

    if not rows:
        raise RuntimeError("No per-image CSVs discovered. Check paths.")
    out = pd.concat(rows, ignore_index=True)
    return out[["dataset", "model", "regime", "filename", "dice", "hd95"]]


# ---------------------------------------------------------------------------
# Patient-id extraction per dataset
# ---------------------------------------------------------------------------
def extract_patient_id(dataset: str, filename: str) -> str:
    """Recover the metadata join-key from an image filename."""
    if dataset == "thyroidxl":
        # '00001400_81A1A425_0.png' -> '00001400'
        return filename.split("_")[0]
    if dataset == "stanford_aimi":
        # '202__frame_0001.png' -> '202_' (annot_id keeps trailing underscore)
        m = re.match(r"^(\d+_)_frame_\d+\.png$", filename)
        if m:
            return m.group(1)
        # Fallback: take everything before '_frame_'
        if "_frame_" in filename:
            return filename.split("_frame_")[0]
        return filename.split("_")[0] + "_"
    if dataset == "ddti":
        # '299_1.png' -> '299'
        return filename.split("_")[0]
    raise ValueError(dataset)


# ---------------------------------------------------------------------------
# Stanford per-nodule reduction
# ---------------------------------------------------------------------------
def reduce_stanford_to_per_nodule(df: pd.DataFrame) -> pd.DataFrame:
    """For Stanford AIMI rows: collapse ~90 frames per cine-clip to a single
    per-nodule median DSC/HD95. Returns dataframe with the same long-format
    schema (plus n_frames per-nodule audit column), with one row per
    (model, regime, patient_id) instead of per frame."""
    is_stanford = df["dataset"] == "stanford_aimi"
    other = df.loc[~is_stanford].copy()
    stanford = df.loc[is_stanford].copy()

    grouped = (stanford
               .groupby(["dataset", "model", "regime", "patient_id"],
                        as_index=False)
               .agg(dice=("dice", "median"),
                    hd95=("hd95", "median"),
                    n_frames=("dice", "size")))
    # Re-attach stratifier columns (constant within nodule -> first()).
    strat_cols = ["tirads", "size_mm", "age", "sex", "composition"]
    present = [c for c in strat_cols if c in stanford.columns]
    if present:
        nodule_meta = (stanford
                       .groupby(["dataset", "model", "regime", "patient_id"],
                                as_index=False)[present]
                       .first())
        grouped = grouped.merge(nodule_meta,
                                on=["dataset", "model", "regime", "patient_id"],
                                how="left")
    # Pad columns to match other (filename becomes a synthetic per-nodule key).
    grouped["filename"] = grouped["patient_id"] + "__per_nodule"
    # Preserve n_frames as an explicit column in the unified output (NaN for
    # non-Stanford rows since per-image counts are 1 by construction).
    if "n_frames" not in other.columns:
        other["n_frames"] = 1
    all_cols = list(dict.fromkeys(other.columns.tolist() +
                                  grouped.columns.tolist()))
    for col in all_cols:
        if col not in other.columns:
            other[col] = np.nan
        if col not in grouped.columns:
            grouped[col] = np.nan
    return pd.concat([other[all_cols], grouped[all_cols]], ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fewshot-fraction", type=float, default=0.05,
                        help="Few-shot LoRA fraction to include from Exp 3 "
                             "(default 0.05, label-efficiency headline).")
    args = parser.parse_args()

    print("[1/5] Loading raw metadata from original paths...")
    meta_frames = [load_thyroidxl_metadata()]
    print(f"   ThyroidXL : {len(meta_frames[0]):5d} patients")
    if "stanford_aimi" in DATASETS:
        meta_stf = load_stanford_metadata()
        print(f"   Stanford  : {len(meta_stf):5d} nodules")
        meta_frames.append(meta_stf)

    meta_all = pd.concat(meta_frames, ignore_index=True)
    meta_all["patient_id"] = meta_all["patient_id"].astype(str)

    print("\n[2/5] Loading test splits...")
    splits = load_test_splits()
    print(f"   Total test images across {len(DATASETS)} datasets: {len(splits)}")

    print("\n[3/5] Discovering per-image metrics CSVs...")
    metrics = discover_per_image_metrics(fewshot_fraction=args.fewshot_fraction)
    print(f"   Discovered {len(metrics):,} per-image metric rows "
          f"across {metrics[['dataset','model','regime']].drop_duplicates().shape[0]} "
          f"(dataset,model,regime) tuples.")

    print("\n[4/5] Joining: splits -> per-image -> metadata...")
    # Step 4a: attach patient_id to metrics from filename (vectorized per
    # dataset to avoid per-row apply over hundreds of thousands of rows).
    pid_parts = []
    for ds, grp in metrics.groupby("dataset", sort=False):
        pids = grp["filename"].map(lambda fn: extract_patient_id(ds, fn))
        pid_parts.append(pd.Series(pids.values, index=grp.index))
    metrics["patient_id"] = pd.concat(pid_parts).reindex(metrics.index)
    # Step 4b: attach stratifier metadata via (dataset, patient_id).
    long = metrics.merge(meta_all,
                         on=["dataset", "patient_id"],
                         how="left", validate="many_to_one")
    n_unmatched = long["age"].isna().sum() + long["sex"].isna().sum()
    print(f"   Long-format rows after merge: {len(long):,}")
    print(f"   Rows with missing stratifier (age|sex NaN): {n_unmatched:,}")

    print("\n[5/5] Collapsing Stanford to per-nodule median...")
    long = reduce_stanford_to_per_nodule(long)
    n_per_ds = long.groupby("dataset").size().to_dict()
    print(f"   Final long-table rows per dataset: {n_per_ds}")

    PATH_OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(PATH_OUT, index=False)
    print(f"\nWrote {PATH_OUT.relative_to(REPO_ROOT)} "
          f"({len(long):,} rows, {long.shape[1]} cols)")


if __name__ == "__main__":
    main()
