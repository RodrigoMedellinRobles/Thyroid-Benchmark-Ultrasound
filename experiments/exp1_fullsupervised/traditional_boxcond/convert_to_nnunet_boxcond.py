#!/usr/bin/env python3
"""Convert processed PNGs to nnU-Net v2 BOX-CONDITIONED format (2 channels).

Following MedSAM (Ma et al. 2024, Nat Commun 15:654), the bounding-box prompt is
transformed into a binary mask and concatenated with the image as an additional
input channel, using the default 2D nnU-Net configuration. Here:

  channel 0 (_0000.png): grayscale ultrasound image  → ZScoreNormalization
  channel 1 (_0001.png): rasterised tight oracle box → 'nonorm' (NoNormalization)

so the box channel stays {0,1} (never z-scored) and the image channel is
normalised identically to the nnU-Net image-only baseline (its within-model
twin). The box is baked TIGHT from the GT mask (frozen, not epoch-jittered — a
documented divergence vs the U-Net/TransUNet box-cond path). Test cases also get
the oracle tight box so nnU-Net can be prompted at eval.

Box-conditioned datasets use IDs 011-014 to avoid clashing with the image-only
Dataset001-004.

Usage:
    python experiments/exp1_fullsupervised/traditional_boxcond/convert_to_nnunet_boxcond.py \
        --datasets ddti tn3k thyroidxl stanford_aimi
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Box-conditioned IDs (image-only uses 001-004).
DATASET_IDS = {"ddti": "011", "tn3k": "012", "thyroidxl": "013", "stanford_aimi": "014"}
DATASET_NAMES = {"ddti": "DDTI", "tn3k": "TN3K",
                 "thyroidxl": "ThyroidXL", "stanford_aimi": "StanfordAIMI"}


def read_csv_filenames(csv_path: Path) -> list[str]:
    with open(csv_path) as f:
        return [row["filename"] for row in csv.DictReader(f)]


def save_grayscale(src: Path, dst: Path) -> None:
    Image.open(src).convert("L").save(dst)


def binary_mask(src: Path) -> np.ndarray:
    return (np.array(Image.open(src).convert("L")) > 127).astype(np.uint8)


def save_mask_from(arr: np.ndarray, dst: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(dst)


def box_channel_from_mask(mask: np.ndarray) -> np.ndarray:
    """Rasterise the tight axis-aligned bbox of the mask as a {0,1} uint8 image.

    Empty mask → full-image box (matches thyroidbench.boxes.masks_to_boxes
    so the fallback semantics are identical across the CNN and nnU-Net paths).
    """
    H, W = mask.shape
    chan = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        chan[:, :] = 1  # full-image fallback
        return chan
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    chan[y1:y2 + 1, x1:x2 + 1] = 1
    return chan


def _emit_case(cid: str, src_img: Path, src_mask: Path,
               img_dir: Path, lbl_dir: Path) -> None:
    mask = binary_mask(src_mask)
    save_grayscale(src_img, img_dir / f"{cid}_0000.png")          # channel 0: image
    save_mask_from(box_channel_from_mask(mask), img_dir / f"{cid}_0001.png")  # channel 1: box
    save_mask_from(mask, lbl_dir / f"{cid}.png")                  # label


def convert_dataset(key: str, data_root: Path) -> None:
    did, dname = DATASET_IDS[key], DATASET_NAMES[key]
    dir_name = f"Dataset{did}_{dname}_boxcond"
    out_root = data_root / "nnunet_raw" / dir_name
    proc = data_root / "processed" / key
    splits = data_root / "splits"

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    train_fnames = read_csv_filenames(splits / f"{key}_train.csv")
    val_fnames   = read_csv_filenames(splits / f"{key}_val.csv")
    test_fnames  = read_csv_filenames(splits / f"{key}_test.csv")
    trainval = sorted(set(train_fnames) | set(val_fnames))
    testset  = sorted(set(test_fnames))

    case_mapping: dict[str, str] = {}
    idx = 1
    for fname in trainval:
        src_img, src_mask = proc / "images" / fname, proc / "masks" / fname
        if not src_img.exists() or not src_mask.exists():
            print(f"  Warning: missing {fname}, skipping"); continue
        cid = f"case_{idx:05d}"; case_mapping[cid] = fname
        _emit_case(cid, src_img, src_mask, out_root / "imagesTr", out_root / "labelsTr")
        idx += 1

    test_start_idx = idx
    for fname in testset:
        src_img, src_mask = proc / "images" / fname, proc / "masks" / fname
        if not src_img.exists() or not src_mask.exists():
            print(f"  Warning: missing test {fname}, skipping"); continue
        cid = f"case_{idx:05d}"; case_mapping[cid] = fname
        _emit_case(cid, src_img, src_mask, out_root / "imagesTs", out_root / "labelsTs")
        idx += 1

    dataset_json = {
        "channel_names": {"0": "US", "1": "nonorm"},   # box channel skips z-score
        "labels": {"background": 0, "nodule": 1},
        "numTraining": len(trainval),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    (out_root / "case_mapping.json").write_text(json.dumps(case_mapping, indent=2))

    rev = {v: k for k, v in case_mapping.items()}
    train_cases = sorted(rev[f] for f in train_fnames if f in rev)
    val_cases   = sorted(rev[f] for f in val_fnames   if f in rev)
    (out_root / "splits_final.json").write_text(
        json.dumps([{"train": train_cases, "val": val_cases}], indent=2))

    print(f"  {dir_name}: {len(trainval)} train+val  {idx - test_start_idx} test → {out_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_IDS), required=True)
    p.add_argument("--data_root", default="data")
    args = p.parse_args()
    data_root = Path(args.data_root).resolve()
    for ds in args.datasets:
        print(f"Converting {ds} (box-conditioned)...")
        convert_dataset(ds, data_root)
    print("Done.")


if __name__ == "__main__":
    main()
