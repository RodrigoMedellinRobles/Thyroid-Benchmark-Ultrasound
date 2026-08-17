#!/usr/bin/env python3
"""Preprocessing script for ThyroidXL dataset.

Resizes images and masks to 512x512 keeping aspect ratio with black padding.
On completion, this script also:
  - appends/overwrites the 'thyroidxl' row in data/processed/preprocessing_summary.csv
  - saves a sample overview figure under figures/data_prep/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preprocess_utils import save_overview_figure, update_summary_csv

ROOT = Path(__file__).resolve().parents[2]  # repo root (00_setup/_lib/ -> repo root)
SPLITS = ['train', 'test']
IN_BASE = ROOT / 'data' / 'raw' / 'extracted' / 'ThyroidXL'
OUT_IMAGES = ROOT / 'data' / 'processed' / 'thyroidxl' / 'images'
OUT_MASKS  = ROOT / 'data' / 'processed' / 'thyroidxl' / 'masks'
TARGET_SIZE = 512
PLACEMENT_NOTEBOOK = '00_setup/01_place_gated_datasets.ipynb'


def resize_pad(img: Image.Image, size: int, resample) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), resample=resample)
    out = Image.new(img.mode, (size, size), 0)
    out.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return out


def collect_pairs(smoke: int):
    # ThyroidXL is access-gated and is not shipped. Path.glob() on a missing
    # directory yields nothing rather than raising, which would let this script
    # report "Verification OK" over zero images. Fail loudly instead.
    if not IN_BASE.is_dir():
        print(f'ERROR: ThyroidXL not found at {IN_BASE}')
        print('       It is access-gated and must be placed by hand. See')
        print('       %s' % PLACEMENT_NOTEBOOK)
        sys.exit(1)
    pairs = []
    for split in SPLITS:
        img_dir  = IN_BASE / split / 'images'
        mask_dir = IN_BASE / split / 'masks'
        if not img_dir.is_dir() or not mask_dir.is_dir():
            print(f'ERROR: expected {img_dir} and {mask_dir} to exist.')
            print('       %s shows the exact directory tree.' % PLACEMENT_NOTEBOOK)
            sys.exit(1)
        for f in sorted(img_dir.glob('*.png')):
            m = mask_dir / f.name
            if m.exists():
                pairs.append((f, m))
            else:
                print(f'WARNING: no mask for {f.name}')
            if smoke and len(pairs) >= smoke:
                return pairs
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='Process only first N pairs (0=all)')
    args = ap.parse_args()

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.smoke)
    if not pairs:
        print('ERROR: found 0 image/mask pairs for ThyroidXL. '
              'Nothing was processed.')
        sys.exit(1)
    print(f'Pairs found: {len(pairs)}  smoke={args.smoke or "off"}')

    processed = skipped = errors = 0
    for img_path, mask_path in tqdm(pairs, desc='ThyroidXL'):
        out_img  = OUT_IMAGES / img_path.name
        out_mask = OUT_MASKS  / mask_path.name
        if out_img.exists() and out_mask.exists():
            skipped += 1
            continue
        try:
            img  = resize_pad(Image.open(img_path).convert('RGB'), TARGET_SIZE, Image.Resampling.LANCZOS)
            mask = resize_pad(Image.open(mask_path).convert('L'),  TARGET_SIZE, Image.Resampling.NEAREST)
            img.save(out_img, 'PNG')
            mask.save(out_mask, 'PNG')
            processed += 1
        except Exception as e:
            print(f'ERROR {img_path.name}: {e}')
            errors += 1

    print(f'\nProcessed={processed}  Skipped={skipped}  Errors={errors}')
    out_imgs  = sorted(OUT_IMAGES.glob('*.png'))
    out_masks = sorted(OUT_MASKS.glob('*.png'))
    print(f'Output: {len(out_imgs)} images / {len(out_masks)} masks')
    assert len(out_imgs) == len(out_masks), 'MISMATCH images vs masks!'
    assert {p.name for p in out_imgs} == {p.name for p in out_masks}, 'FILENAME MISMATCH!'
    print('Verification OK')

    # Update shared summary CSV (mean/std/n_train are filled later by 06_compute_dataset_stats.py)
    summary_path = update_summary_csv(
        dataset='thyroidxl',
        n_images=len(out_imgs),
        n_masks=len(out_masks),
        target_size=TARGET_SIZE,
        mask_threshold='direct copy (already binary in raw HDF5)',
        observation='Brightest dataset (most neck anatomy in field of view); resize+pad LANCZOS/NEAREST',
    )
    print(f'Summary CSV → {summary_path}')

    # Save overview figure with first N processed pairs
    samples = []
    for p in out_imgs[:3]:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        m   = np.array(Image.open(OUT_MASKS / p.name).convert('L'), dtype=np.uint8)
        m_bin = (m > 0).astype(np.uint8)
        samples.append((img, m_bin))
    fig_path = save_overview_figure(
        dataset='thyroidxl',
        samples=samples,
        title_lines=[
            f"ThyroidXL — preprocess overview (n={len(out_imgs)})",
            f"resize+pad to {TARGET_SIZE}×{TARGET_SIZE} | LANCZOS image / NEAREST mask | source: HDF5",
        ],
    )
    print(f'Overview figure → {fig_path}')


if __name__ == '__main__':
    main()
