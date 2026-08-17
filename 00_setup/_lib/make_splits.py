#!/usr/bin/env python3
"""Generate train/val/test split CSVs for all ThyroidBench datasets.
Output: data/splits/{dataset}_{split}.csv  columns: filename, patient_id, split
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parents[2]  # repo root (00_setup/_lib/ -> repo root)
SPLITS_DIR = ROOT / 'data' / 'splits'
SEED = 42
rng = np.random.default_rng(SEED)

IMG_EXTS = {'.png', '.jpg', '.jpeg'}


def save_split(df: pd.DataFrame, dataset: str, split: str):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    p = SPLITS_DIR / f'{dataset}_{split}.csv'
    df[['filename', 'patient_id', 'split']].to_csv(p, index=False)
    print(f'  {p.name}: {len(df)} rows')


# ── ThyroidXL ────────────────────────────────────────────────────────────────
def process_thyroidxl():
    print('ThyroidXL...')
    train_ann = json.loads((ROOT / 'data/raw/extracted/ThyroidXL/train/train_annotations.json').read_text())
    test_ann  = json.loads((ROOT / 'data/raw/extracted/ThyroidXL/test/test_annotations.json').read_text())

    records = []
    train_pids = sorted(train_ann['info'].keys())
    shuffled = train_pids.copy()
    rng.shuffle(shuffled)
    val_pids = set(shuffled[:int(len(shuffled) * 0.10)])

    for pid, rec in train_ann['info'].items():
        split = 'val' if pid in val_pids else 'train'
        for fn in rec.get('images', []):
            records.append({'filename': fn, 'patient_id': pid, 'split': split})

    for pid, rec in test_ann['info'].items():
        for fn in rec.get('images', []):
            records.append({'filename': fn, 'patient_id': pid, 'split': 'test'})

    df = pd.DataFrame(records)
    for s in ['train', 'val', 'test']:
        save_split(df[df['split'] == s], 'thyroidxl', s)
    return df


# ── TN3K ─────────────────────────────────────────────────────────────────────
def process_tn3k():
    print('TN3K...')
    picture = ROOT / 'data/raw/extracted/TRFE-Net-for-thyroid-nodule-segmentation-main/picture'
    tv_img  = picture / 'trainval/trainval-image'
    tv_mask = picture / 'trainval/trainval-mask'
    te_img  = picture / 'test/test-image'
    te_mask = picture / 'test/test-mask'

    records = []
    # NOTE: trainval and test share identical filenames → output uses split prefix
    # Output naming: trainval_{stem}.png  /  test_{stem}.png
    tv_files = sorted(f for f in tv_img.iterdir() if f.suffix in IMG_EXTS and (tv_mask / f.name).exists())
    shuffled = list(tv_files)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * 0.10)
    val_set = {f.name for f in shuffled[:n_val]}

    for f in tv_files:
        split = 'val' if f.name in val_set else 'train'
        records.append({'filename': f'trainval_{f.stem}.png', 'patient_id': f.stem, 'split': split})

    for f in sorted(f for f in te_img.iterdir() if f.suffix in IMG_EXTS):
        mask = te_mask / f.name
        if mask.exists():
            records.append({'filename': f'test_{f.stem}.png', 'patient_id': f.stem, 'split': 'test'})

    df = pd.DataFrame(records)
    for s in ['train', 'val', 'test']:
        save_split(df[df['split'] == s], 'tn3k', s)
    return df


# ── DDTI ─────────────────────────────────────────────────────────────────────
def ddti_renderable_cases():
    """Return dict {case_id: [view_int, ...]} for renderable pairs."""
    ddti = ROOT / 'data/raw/extracted/DDTI-Pedraza_Digital_Database_Thyroid_SPIE'
    renderable = {}
    for jpg in ddti.glob('*.jpg'):
        stem = jpg.stem
        if '_' not in stem:
            continue
        case_id, view_str = stem.rsplit('_', 1)
        if not view_str.isdigit():
            continue
        xml = ddti / f'{case_id}.xml'
        if not xml.exists():
            continue
        # Confirm polygon exists for this view
        try:
            root = ET.parse(xml).getroot()
            for mark in root.findall('mark'):
                img_el = mark.find('image')
                svg_el = mark.find('svg')
                if img_el is None or svg_el is None or not svg_el.text:
                    continue
                if int(img_el.text) == int(view_str):
                    import json as _json
                    shapes = _json.loads(svg_el.text)
                    has_pts = any(len(s.get('points', [])) >= 3 for s in shapes)
                    if has_pts:
                        renderable.setdefault(case_id, []).append(int(view_str))
        except Exception:
            continue
    return renderable


def process_ddti():
    print('DDTI...')
    renderable = ddti_renderable_cases()
    cases = sorted(renderable.keys())
    shuffled = cases.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_cases = set(shuffled[:int(n * 0.70)])
    val_cases   = set(shuffled[int(n * 0.70):int(n * 0.85)])
    test_cases  = set(shuffled[int(n * 0.85):])

    records = []
    for case_id, views in renderable.items():
        split = 'train' if case_id in train_cases else 'val' if case_id in val_cases else 'test'
        for v in views:
            records.append({'filename': f'{case_id}_{v}.png', 'patient_id': case_id, 'split': split})

    df = pd.DataFrame(records)
    for s in ['train', 'val', 'test']:
        save_split(df[df['split'] == s], 'ddti', s)
    return df


# ── Stanford AIMI ─────────────────────────────────────────────────────────────
def process_stanford_aimi():
    print('Stanford AIMI...')
    hdf5 = ROOT / 'data/raw/extracted/Stanford/dataset.hdf5'
    with h5py.File(hdf5, 'r') as f:
        annot_ids  = [a.decode().strip() for a in f['annot_id'][:]]
        frame_nums = [int(v.decode().strip()) for v in f['frame_num'][:]]

    all_nodules = sorted(set(annot_ids))
    shuffled = all_nodules.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_set = set(shuffled[:int(n * 0.70)])
    val_set   = set(shuffled[int(n * 0.70):int(n * 0.85)])
    test_set  = set(shuffled[int(n * 0.85):])

    records = []
    for aid, fnum in zip(annot_ids, frame_nums):
        split = 'train' if aid in train_set else 'val' if aid in val_set else 'test'
        records.append({'filename': f'{aid}_frame_{fnum:04d}.png', 'patient_id': aid, 'split': split})

    df = pd.DataFrame(records)
    for s in ['train', 'val', 'test']:
        save_split(df[df['split'] == s], 'stanford_aimi', s)
    return df


# ── Summary ───────────────────────────────────────────────────────────────────
def summarize(dfs: dict):
    print('\n' + '='*60)
    print(f'{"Dataset":20} {"Split":8} {"Patients":>10} {"Images":>8}')
    print('='*60)
    for name, df in dfs.items():
        if df is None:
            continue
        total_pat = df['patient_id'].nunique()
        for s in ['train', 'val', 'test']:
            sub = df[df['split'] == s]
            print(f'{name:20} {s:8} {sub["patient_id"].nunique():10} {len(sub):8}')
        print(f'{"":20} {"TOTAL":8} {total_pat:10} {len(df):8}')
        print('-'*60)


def main():
    dfs = {}
    for name, fn in [('thyroidxl', process_thyroidxl), ('tn3k', process_tn3k),
                     ('ddti', process_ddti), ('stanford_aimi', process_stanford_aimi)]:
        try:
            dfs[name] = fn()
        except FileNotFoundError as e:
            print(f'ERROR {name}: {e}')
            if name in ('thyroidxl', 'stanford_aimi'):
                print('       This dataset is access-gated and is not shipped. Place your '
                      'approved copy first;')
                print('       00_setup/01_place_gated_datasets.ipynb shows the exact path '
                      'and validates it.')
            else:
                print('       Download it first with 00_setup/00_get_open_datasets.ipynb.')
            dfs[name] = None
        except Exception as e:
            print(f'ERROR {name}: {e}')
            dfs[name] = None
    summarize(dfs)
    print('\nDone.')


if __name__ == '__main__':
    main()
