#!/usr/bin/env python3
"""Preprocessing script for TN3K dataset.

Grayscale L → RGB, mask threshold >127 → 255, resize 512x512 with padding.
On completion, this script also:
  - appends/overwrites the 'tn3k' row in data/processed/preprocessing_summary.csv
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
PICTURE_ROOT = ROOT / 'data' / 'raw' / 'extracted' / 'TRFE-Net-for-thyroid-nodule-segmentation-main' / 'picture'
OUT_IMAGES = ROOT / 'data' / 'processed' / 'tn3k' / 'images'
OUT_MASKS  = ROOT / 'data' / 'processed' / 'tn3k' / 'masks'
TARGET_SIZE = 512
IMG_EXTS = {'.png', '.jpg', '.jpeg'}


def resize_pad(img: Image.Image, size: int, resample) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), resample=resample)
    out = Image.new(img.mode, (size, size), 0)
    out.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return out


def collect_pairs(smoke: int):
    # NOTE: trainval and test share identical filenames (0000.jpg..0613.jpg)
    # Prefix output with split name to avoid collision: trainval_0000.png, test_0000.png
    pairs = []
    for split_name, img_subdir, mask_subdir in [
        ('trainval', 'trainval/trainval-image', 'trainval/trainval-mask'),
        ('test',     'test/test-image',          'test/test-mask'),
    ]:
        img_dir  = PICTURE_ROOT / img_subdir
        mask_dir = PICTURE_ROOT / mask_subdir
        imgs = sorted(f for f in img_dir.iterdir() if f.suffix in IMG_EXTS)
        orphan_masks = len([f for f in mask_dir.iterdir() if f.suffix in IMG_EXTS]) - len(imgs)
        if orphan_masks > 0:
            print(f'WARNING [{split_name}]: {orphan_masks} orphan mask(s) detected — iterating over images')
        for f in imgs:
            m = mask_dir / f.name
            if not m.exists():
                for ext in IMG_EXTS:
                    alt = mask_dir / (f.stem + ext)
                    if alt.exists():
                        m = alt
                        break
                else:
                    print(f'WARNING: no mask for {f.name}')
                    continue
            pairs.append((f, m, split_name))
            if smoke and len(pairs) >= smoke:
                return pairs
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.smoke)
    if not pairs:
        print('ERROR: found 0 image/mask pairs for TN3K. '
              'Nothing was processed.')
        sys.exit(1)
    print(f'Pairs found: {len(pairs)}  smoke={args.smoke or "off"}')

    processed = skipped = errors = 0
    for img_path, mask_path, split_name in tqdm(pairs, desc='TN3K'):
        out_name = f'{split_name}_{img_path.stem}.png'
        out_img  = OUT_IMAGES / out_name
        out_mask = OUT_MASKS  / out_name
        if out_img.exists() and out_mask.exists():
            skipped += 1
            continue
        try:
            # Image: L → RGB
            img = Image.open(img_path).convert('L').convert('RGB')
            img = resize_pad(img, TARGET_SIZE, Image.Resampling.LANCZOS)
            img.save(out_img, 'PNG')
            # Mask: threshold >127 → 255 (JPEG artifacts land at 1-10; real foreground at 245-255)
            m = np.array(Image.open(mask_path))
            m = ((m > 127).astype(np.uint8) * 255)
            mask = resize_pad(Image.fromarray(m, mode='L'), TARGET_SIZE, Image.Resampling.NEAREST)
            mask.save(out_mask, 'PNG')
            processed += 1
        except Exception as e:
            print(f'ERROR {img_path.name}: {e}')
            errors += 1

    print(f'\nProcessed={processed}  Skipped={skipped}  Errors={errors}')
    out_imgs  = sorted(OUT_IMAGES.glob('*.png'))
    out_masks = sorted(OUT_MASKS.glob('*.png'))
    print(f'Output: {len(out_imgs)} images / {len(out_masks)} masks')
    assert {p.name for p in out_imgs} == {p.name for p in out_masks}, 'FILENAME MISMATCH!'
    print('Verification OK')

    summary_path = update_summary_csv(
        dataset='tn3k',
        n_images=len(out_imgs),
        n_masks=len(out_masks),
        target_size=TARGET_SIZE,
        mask_threshold='>127 → 255 (filters JPEG compression noise at 1-10)',
        observation='Mid-range brightness; trainval+test merged with split prefix to avoid filename collision',
    )
    print(f'Summary CSV → {summary_path}')

    samples = []
    for p in out_imgs[:3]:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        m   = np.array(Image.open(OUT_MASKS / p.name).convert('L'), dtype=np.uint8)
        samples.append((img, (m > 0).astype(np.uint8)))
    fig_path = save_overview_figure(
        dataset='tn3k',
        samples=samples,
        title_lines=[
            f"TN3K — preprocess overview (n={len(out_imgs)})",
            f"L→RGB | mask >127 → 255 | resize+pad to {TARGET_SIZE}×{TARGET_SIZE} | source: TRFE-Net repo",
        ],
    )
    print(f'Overview figure → {fig_path}')


if __name__ == '__main__':
    main()
