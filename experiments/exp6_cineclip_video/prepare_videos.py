#!/usr/bin/env python3
"""
prepare_videos.py

Converts Stanford AIMI test PNGs to per-patient JPEG directories for Exp 5 video API.

Output structure per patient:
  data/cineclip_videos/{patient_id}/00000.jpg, 00001.jpg, ...   (zero-indexed numeric stems)
  data/cineclip_videos/{patient_id}/frame_index_map.json        (maps numeric_idx -> original_frame_num)

SAM2's `load_video_frames_from_jpg_images` sorts via `int(splitext(p)[0])`, so filenames
MUST be purely numeric — `frame_NNNN.jpg` would crash with ValueError. The sidecar JSON
preserves the mapping back to the original Stanford AIMI frame numbers for mask lookup
and traceability.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel of LAB. Input/output: uint8 RGB. Identical to foundation_base.py."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2RGB).astype(np.uint8)


def extract_frame_number(filename: str) -> int:
    """Return integer frame number from '202__frame_0003.png' → 3."""
    parts = filename.split("__frame_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename: {filename}")
    return int(parts[1].rsplit(".", 1)[0])


def parse_args():
    p = argparse.ArgumentParser(description="Prepare per-patient JPEG dirs for Exp 5 video API")
    p.add_argument("--split_csv", default="data/splits/stanford_aimi_test.csv")
    p.add_argument("--processed_dir", default="data/processed/stanford_aimi/images")
    p.add_argument("--output_dir", default="data/cineclip_videos")
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--img_size", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    split_csv   = Path(args.split_csv)
    proc_dir    = Path(args.processed_dir)
    output_dir  = Path(args.output_dir)

    if not split_csv.exists():
        print(f"ERROR: {split_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not proc_dir.is_dir():
        print(f"ERROR: {proc_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Group filenames by patient_id (CSV already contains only test set)
    patient_files: dict[str, list[str]] = {}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            patient_files.setdefault(row["patient_id"].strip(), []).append(row["filename"].strip())

    total_patients = len(patient_files)
    total_frames   = 0

    for pid, filenames in sorted(patient_files.items()):
        # Sort by original frame number to guarantee temporal order
        filenames_sorted = sorted(filenames, key=extract_frame_number)
        patient_out = output_dir / pid
        patient_out.mkdir(parents=True, exist_ok=True)

        # Clean previous outputs that used the old `frame_NNNN.jpg` naming pattern
        # (incompatible with SAM2's numeric-only filename parser).
        for old in patient_out.glob("frame_*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass

        index_map: dict[str, int] = {}  # numeric_filename -> original frame_num
        n_written = 0
        for video_idx, fname in enumerate(filenames_sorted):
            src = proc_dir / fname
            if not src.exists():
                print(f"  WARN: {src} missing — skipping", file=sys.stderr)
                continue

            img_rgb = np.array(Image.open(src).convert("RGB"), dtype=np.uint8)
            img_clahe = apply_clahe(img_rgb)
            if img_clahe.shape[0] != args.img_size or img_clahe.shape[1] != args.img_size:
                img_clahe = cv2.resize(img_clahe, (args.img_size, args.img_size),
                                       interpolation=cv2.INTER_LINEAR)

            frame_num = extract_frame_number(fname)  # original Stanford AIMI numbering
            # Use zero-padded numeric stem so SAM2's int(splitext()) parser works.
            numeric_name = f"{video_idx:05d}.jpg"
            dst = patient_out / numeric_name
            ok = cv2.imwrite(str(dst), cv2.cvtColor(img_clahe, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ok:
                print(f"  ERROR: write failed {dst}", file=sys.stderr)
            else:
                index_map[numeric_name] = frame_num
                n_written += 1

        # Sidecar: maps video frame index back to original Stanford AIMI frame number
        with open(patient_out / "frame_index_map.json", "w") as f:
            json.dump(index_map, f, indent=2)

        print(f"patient_id={pid}  n_frames={n_written}  output_dir={patient_out}")
        total_frames += n_written

    print(f"\nTotal: n_patients={total_patients}  n_frames={total_frames}  output_dir={output_dir}")


if __name__ == "__main__":
    main()
