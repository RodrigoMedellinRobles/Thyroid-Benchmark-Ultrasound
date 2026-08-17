#!/usr/bin/env python3
"""Preprocessing script for Stanford AIMI Cine-clip dataset.

Reads from HDF5, mask threshold >127 → 255, grayscale → 3-channel RGB,
resize+pad to 512x512. On completion, this script also:
  - appends/overwrites the 'stanford_aimi' row in data/processed/preprocessing_summary.csv
  - saves a sample overview figure under figures/data_prep/
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preprocess_utils import save_overview_figure, update_summary_csv

ROOT = Path(__file__).resolve().parents[2]  # repo root (00_setup/_lib/ -> repo root)
HDF5_PATH = ROOT / 'data' / 'raw' / 'extracted' / 'Stanford' / 'dataset.hdf5'
OUT_IMAGES = ROOT / 'data' / 'processed' / 'stanford_aimi' / 'images'
OUT_MASKS  = ROOT / 'data' / 'processed' / 'stanford_aimi' / 'masks'
TARGET_SIZE = 512


def resize_pad(img: Image.Image, size: int, resample) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), resample=resample)
    out = Image.new(img.mode, (size, size), 0)
    out.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='Process only first N frames (0=all)')
    args = ap.parse_args()

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    if not HDF5_PATH.exists():
        print(f'ERROR: HDF5 not found at {HDF5_PATH}')
        print('       Stanford AIMI is access-gated and must be placed by hand.')
        print('       See 00_setup/01_place_gated_datasets.ipynb for the exact path.')
        sys.exit(1)

    processed = skipped = errors = 0

    with h5py.File(HDF5_PATH, 'r') as f:
        total = f['image'].shape[0]
        if args.smoke:
            total = min(total, args.smoke)
        print(f'Frames to process: {total}  smoke={args.smoke or "off"}')

        for i in tqdm(range(total), desc='Stanford AIMI'):
            annot_id  = f['annot_id'][i].decode().strip()
            frame_num = int(f['frame_num'][i].decode().strip())
            fname     = f'{annot_id}_frame_{frame_num:04d}.png'
            out_img   = OUT_IMAGES / fname
            out_mask  = OUT_MASKS  / fname

            if out_img.exists() and out_mask.exists():
                skipped += 1
                continue
            try:
                # Image: grayscale → RGB
                img_arr = f['image'][i]   # (802, 1054) uint8
                img_rgb = np.stack([img_arr] * 3, axis=-1)
                img_pil = resize_pad(Image.fromarray(img_rgb, 'RGB'), TARGET_SIZE, Image.Resampling.LANCZOS)
                img_pil.save(out_img, 'PNG')

                # Mask: threshold >127 → 255
                msk_arr = f['mask'][i]    # (802, 1054) uint8
                msk_bin = ((msk_arr > 127).astype(np.uint8) * 255)
                msk_pil = resize_pad(Image.fromarray(msk_bin, 'L'), TARGET_SIZE, Image.Resampling.NEAREST)
                msk_pil.save(out_mask, 'PNG')
                processed += 1
            except Exception as e:
                print(f'ERROR frame {i} ({fname}): {e}')
                errors += 1

    print(f'\nProcessed={processed}  Skipped={skipped}  Errors={errors}')
    out_imgs  = sorted(OUT_IMAGES.glob('*.png'))
    out_masks = sorted(OUT_MASKS.glob('*.png'))
    print(f'Output: {len(out_imgs)} images / {len(out_masks)} masks')
    print('Verification OK' if {p.name for p in out_imgs} == {p.name for p in out_masks} else 'MISMATCH!')

    summary_path = update_summary_csv(
        dataset='stanford_aimi',
        n_images=len(out_imgs),
        n_masks=len(out_masks),
        target_size=TARGET_SIZE,
        mask_threshold='>127 → 255 (HDF5 source)',
        observation='Cine-clip frames (~90 frames/patient); patient-level split is critical to avoid leakage',
    )
    print(f'Summary CSV → {summary_path}')

    samples = []
    for p in out_imgs[:3]:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        m   = np.array(Image.open(OUT_MASKS / p.name).convert('L'), dtype=np.uint8)
        samples.append((img, (m > 0).astype(np.uint8)))
    fig_path = save_overview_figure(
        dataset='stanford_aimi',
        samples=samples,
        title_lines=[
            f"Stanford AIMI — preprocess overview (n={len(out_imgs)})",
            f"HDF5 → grayscale→RGB | mask >127 → 255 | resize+pad to {TARGET_SIZE}×{TARGET_SIZE}",
        ],
    )
    print(f'Overview figure → {fig_path}')


if __name__ == '__main__':
    main()
