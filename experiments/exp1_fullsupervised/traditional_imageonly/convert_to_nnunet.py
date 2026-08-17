#!/usr/bin/env python3
"""Converts processed PNG data (512x512) to nnU-Net v2 format for 4 datasets."""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_IDS = {
    "ddti": "001",
    "tn3k": "002",
    "thyroidxl": "003",
    "stanford_aimi": "004",
}
DATASET_NAMES = {
    "ddti": "DDTI",
    "tn3k": "TN3K",
    "thyroidxl": "ThyroidXL",
    "stanford_aimi": "StanfordAIMI",
}


def read_csv_filenames(csv_path: Path) -> list[str]:
    with open(csv_path) as f:
        return [row["filename"] for row in csv.DictReader(f)]


def save_grayscale(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("L")
    img.save(dst)


def save_mask(src: Path, dst: Path) -> None:
    mask = np.array(Image.open(src).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    Image.fromarray(binary).save(dst)


def convert_dataset(key: str, data_root: Path) -> None:
    did = DATASET_IDS[key]
    dname = DATASET_NAMES[key]
    dir_name = f"Dataset{did}_{dname}"
    out_root = data_root / "nnunet_raw" / dir_name
    proc = data_root / "processed" / key
    splits = data_root / "splits"

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    train_fnames = read_csv_filenames(splits / f"{key}_train.csv")
    val_fnames   = read_csv_filenames(splits / f"{key}_val.csv")
    test_fnames  = read_csv_filenames(splits / f"{key}_test.csv")

    # train+val → imagesTr/labelsTr; test → imagesTs/labelsTs
    trainval = sorted(set(train_fnames) | set(val_fnames))
    testset  = sorted(set(test_fnames))

    case_mapping: dict[str, str] = {}
    idx = 1

    for fname in trainval:
        src_img  = proc / "images" / fname
        src_mask = proc / "masks"  / fname
        if not src_img.exists() or not src_mask.exists():
            print(f"  Warning: missing {fname}, skipping")
            continue
        cid = f"case_{idx:05d}"
        case_mapping[cid] = fname
        save_grayscale(src_img,  out_root / "imagesTr" / f"{cid}_0000.png")
        save_mask(src_mask,       out_root / "labelsTr" / f"{cid}.png")
        idx += 1

    test_start_idx = idx
    for fname in testset:
        src_img  = proc / "images" / fname
        src_mask = proc / "masks"  / fname
        if not src_img.exists() or not src_mask.exists():
            print(f"  Warning: missing test {fname}, skipping")
            continue
        cid = f"case_{idx:05d}"
        case_mapping[cid] = fname
        save_grayscale(src_img,  out_root / "imagesTs" / f"{cid}_0000.png")
        save_mask(src_mask,       out_root / "labelsTs" / f"{cid}.png")
        idx += 1

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "US"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": len(trainval),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    # case_mapping.json
    (out_root / "case_mapping.json").write_text(json.dumps(case_mapping, indent=2))

    # splits_final.json — our exact train/val split for fold 0
    rev = {v: k for k, v in case_mapping.items()}
    train_cases = sorted(rev[f] for f in train_fnames if f in rev)
    val_cases   = sorted(rev[f] for f in val_fnames   if f in rev)
    splits_final = [{"train": train_cases, "val": val_cases}]
    (out_root / "splits_final.json").write_text(json.dumps(splits_final, indent=2))

    n_tr = len(trainval)
    n_ts = idx - test_start_idx
    print(f"  {dir_name}: {n_tr} train+val  {n_ts} test → {out_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   choices=list(DATASET_IDS.keys()), required=True)
    p.add_argument("--data_root", default="data")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    for ds in args.datasets:
        print(f"Converting {ds}...")
        convert_dataset(ds, data_root)
    print("Done.")


if __name__ == "__main__":
    main()
