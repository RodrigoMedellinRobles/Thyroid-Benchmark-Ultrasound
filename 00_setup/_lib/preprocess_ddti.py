#!/usr/bin/env python3
"""Preprocessing script for DDTI dataset.

Parses XML polygons → binary masks, then resizes to 512x512 (LANCZOS image,
NEAREST mask). Polygons under MIN_MASK_AREA pixels are dropped as likely
annotation noise. On completion, this script also:
  - appends/overwrites the 'ddti' row in data/processed/preprocessing_summary.csv
  - saves a sample overview figure under figures/data_prep/
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preprocess_utils import save_overview_figure, update_summary_csv

ROOT = Path(__file__).resolve().parents[2]  # repo root (00_setup/_lib/ -> repo root)
DDTI_ROOT = ROOT / 'data' / 'raw' / 'extracted' / 'DDTI-Pedraza_Digital_Database_Thyroid_SPIE'
OUT_IMAGES = ROOT / 'data' / 'processed' / 'ddti' / 'images'
OUT_MASKS  = ROOT / 'data' / 'processed' / 'ddti' / 'masks'
TARGET_SIZE = 512
MIN_MASK_AREA = 100


def parse_xml(xml_path: Path):
    """Returns dict {view_int: [polygon_points_list, ...]}"""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {}
    result = {}
    for mark in root.findall('mark'):
        img_el = mark.find('image')
        svg_el = mark.find('svg')
        if img_el is None or svg_el is None or not svg_el.text:
            continue
        try:
            view = int(img_el.text)
            shapes = json.loads(svg_el.text)
        except (ValueError, json.JSONDecodeError):
            continue
        polygons = [s.get('points', []) for s in shapes if s.get('points')]
        if polygons:
            result.setdefault(view, []).extend(polygons)
    return result


def render_mask(polygons, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for pts in polygons:
        if len(pts) < 3:
            continue
        arr = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
        cv2.fillPoly(mask, [arr], 255)
    return mask


def resize_pad_arr(arr: np.ndarray, size: int, interp) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(arr, (nw, nh), interpolation=interp)
    if arr.ndim == 3:
        out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    else:
        out = np.zeros((size, size), dtype=arr.dtype)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    out[top:top+nh, left:left+nw] = resized
    return out


def collect_pairs(smoke: int):
    pairs = []
    for jpg in sorted(DDTI_ROOT.glob('*.jpg')):
        stem = jpg.stem
        if '_' not in stem:
            continue
        case_id, view_str = stem.rsplit('_', 1)
        if not view_str.isdigit():
            continue
        xml = DDTI_ROOT / f'{case_id}.xml'
        if xml.exists():
            pairs.append((jpg, xml, case_id, int(view_str)))
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
        print('ERROR: found 0 image/mask pairs for DDTI. '
              'Nothing was processed.')
        sys.exit(1)
    print(f'Candidate pairs: {len(pairs)}  smoke={args.smoke or "off"}')

    stats = dict(processed=0, skipped=0, no_polygon=0, small_mask=0, errors=0)
    for jpg, xml, case_id, view in tqdm(pairs, desc='DDTI'):
        out_name = f'{case_id}_{view}.png'
        out_img  = OUT_IMAGES / out_name
        out_mask = OUT_MASKS  / out_name
        if out_img.exists() and out_mask.exists():
            stats['skipped'] += 1
            continue
        try:
            annotations = parse_xml(xml)
            if view not in annotations:
                stats['no_polygon'] += 1
                continue
            img_arr = np.array(Image.open(jpg).convert('RGB'))
            mask = render_mask(annotations[view], img_arr.shape)
            if mask.sum() // 255 < MIN_MASK_AREA:
                stats['small_mask'] += 1
                continue
            img_out  = resize_pad_arr(img_arr, TARGET_SIZE, cv2.INTER_LANCZOS4)
            mask_out = resize_pad_arr(mask,    TARGET_SIZE, cv2.INTER_NEAREST)
            cv2.imwrite(str(out_img),  cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_mask), mask_out)
            stats['processed'] += 1
        except Exception as e:
            print(f'ERROR {case_id}_{view}: {e}')
            stats['errors'] += 1

    print(f'\n{stats}')
    out_imgs  = sorted(OUT_IMAGES.glob('*.png'))
    out_masks = sorted(OUT_MASKS.glob('*.png'))
    print(f'Output: {len(out_imgs)} images / {len(out_masks)} masks')
    print('Verification OK' if {p.name for p in out_imgs} == {p.name for p in out_masks} else 'MISMATCH!')

    summary_path = update_summary_csv(
        dataset='ddti',
        n_images=len(out_imgs),
        n_masks=len(out_masks),
        target_size=TARGET_SIZE,
        mask_threshold=f'XML polygons → fillPoly (min area {MIN_MASK_AREA} px²)',
        observation='Oldest dataset (SPIE 2015); darkest images; XML polygon annotations',
    )
    print(f'Summary CSV → {summary_path}')

    samples = []
    for p in out_imgs[:3]:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        m   = np.array(Image.open(OUT_MASKS / p.name).convert('L'), dtype=np.uint8)
        samples.append((img, (m > 0).astype(np.uint8)))
    fig_path = save_overview_figure(
        dataset='ddti',
        samples=samples,
        title_lines=[
            f"DDTI — preprocess overview (n={len(out_imgs)})",
            f"XML polygons → fillPoly | min area {MIN_MASK_AREA} px² | resize+pad to {TARGET_SIZE}×{TARGET_SIZE}",
        ],
    )
    print(f'Overview figure → {fig_path}')


if __name__ == '__main__':
    main()
