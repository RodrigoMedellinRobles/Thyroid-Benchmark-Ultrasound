"""
Data layer for the thyroid nodule segmentation benchmark.

Merges three former modules into one flat module:
  * PyTorch ``ThyroidDataset`` + augmentation pipeline (was data/dataset.py)
  * dataloader factory for zero-/few-/full-supervised regimes (was data/dataloader.py)
  * patient-level split validation utilities (was data/splits.py)

Supports: thyroidxl, tn3k, ddti, stanford_aimi datasets.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

# Per-dataset normalization statistics (fallback if JSON not found)
DATASET_STATS = {
    "tn3k": {
        "mean": [0.2759, 0.2761, 0.2759],
        "std": [0.2366, 0.2365, 0.2365],
    },
    "ddti": {
        "mean": [0.1593, 0.1594, 0.1593],
        "std": [0.2208, 0.2209, 0.2208],
    },
    "thyroidxl": {
        "mean": [0.3159, 0.3160, 0.3159],
        "std": [0.2562, 0.2562, 0.2562],
    },
    "stanford_aimi": {
        "mean": [0.1520, 0.1521, 0.1520],
        "std": [0.2092, 0.2092, 0.2091],
    },
}

# __file__ = <repo>/thyroidbench/data.py → parent×2 = repo root
_stats_json_path = Path(__file__).resolve().parent.parent / "data" / "preprocessing_stats.json"
if _stats_json_path.exists():
    try:
        with open(_stats_json_path, "r") as f:
            loaded_stats = json.load(f)
        for ds_name, ds_stats in loaded_stats.items():
            if ds_name in DATASET_STATS:
                DATASET_STATS[ds_name] = {
                    "mean": ds_stats["mean"],
                    "std": ds_stats["std"],
                }
        logger.info("Loaded dataset statistics from %s", _stats_json_path)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load stats from %s: %s. Using fallback values.", _stats_json_path, e)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel of LAB colorspace. Input/output: uint8 RGB."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.uint8)


def get_augmentation(
    mode: str,
    img_size: int = 512,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> A.Compose:
    """Albumentations pipeline for mode ∈ {'train', 'lora', 'val', 'test'}.
    CLAHE is applied upstream in __getitem__, not here.
    """
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]

    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif mode == "lora":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class ThyroidDataset(Dataset):
    """
    Dataset for thyroid nodule segmentation.

    Args:
        dataset_name: One of 'thyroidxl', 'tn3k', 'ddti', 'stanford_aimi'.
        split_csv: Path to CSV with columns: filename, patient_id, split.
        data_root: Root of processed data (parent of {dataset_name}/images/).
        transform: albumentations.Compose; defaults to Resize+Normalize+ToTensorV2.
        mean: Per-channel mean for normalization. Loaded from DATASET_STATS if None.
        std: Per-channel std for normalization. Loaded from DATASET_STATS if None.
    """

    VALID_DATASETS = {"thyroidxl", "tn3k", "ddti", "stanford_aimi"}

    def __init__(
        self,
        dataset_name: str,
        split_csv: Path,
        data_root: Path,
        transform: Optional[A.Compose] = None,
        mean: Optional[list] = None,
        std: Optional[list] = None,
    ):
        if dataset_name not in self.VALID_DATASETS:
            raise ValueError(f"dataset_name must be one of {self.VALID_DATASETS}, got '{dataset_name}'")

        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)

        df = pd.read_csv(self.split_csv)
        self.filenames: list[str] = df["filename"].tolist()
        self.patient_ids: list[str] = df["patient_id"].astype(str).tolist()

        self.image_dir = self.data_root / dataset_name / "images"
        self.mask_dir  = self.data_root / dataset_name / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif dataset_name in DATASET_STATS:
            self.mean = DATASET_STATS[dataset_name]["mean"]
            self.std = DATASET_STATS[dataset_name]["std"]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            logger.warning("Dataset '%s' not in DATASET_STATS — using 0.5/0.5 normalization.", dataset_name)

        self.transform = transform or A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

        logger.info(
            "ThyroidDataset: %s — %d samples (%d patients)",
            dataset_name, len(self), len(set(self.patient_ids)),
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Union[object, str]]:
        filename   = self.filenames[idx]
        patient_id = self.patient_ids[idx]

        image_path = self.image_dir / filename
        mask_path  = self.mask_dir  / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as msk:
            mask = (np.array(msk.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

        # CLAHE before augmentation pipeline (must be on uint8)
        image = apply_clahe(image)

        transformed  = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor  = transformed["mask"].float().unsqueeze(0)

        return {
            "image":      image_tensor,
            "mask":       mask_tensor,
            "patient_id": patient_id,
            "filename":   filename,
        }


# ---------------------------------------------------------------------------
# Dataloader factory (was data/dataloader.py)
# Supports zero-shot, few-shot (patient-level sampling), and full-supervised.
# ---------------------------------------------------------------------------


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker's numpy/python RNG deterministically.

    PyTorch sets ``torch.initial_seed()`` per worker from the loader's
    ``generator`` (which we seed below), so deriving the numpy/random seeds from
    it makes albumentations augmentation reproducible across runs even with
    ``num_workers>0`` (seed=42).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

DatasetName = Literal["thyroidxl", "tn3k", "ddti", "stanford_aimi"]
Split = Literal["train", "val", "test"]
Regime = Literal["zero_shot", "few_shot", "full_supervised"]
FewShotFraction = Literal[0.05, 0.10, 0.25, 0.50]

_VALID_FRACTIONS = {0.05, 0.10, 0.25, 0.50}


def get_dataloader(
    dataset_name: DatasetName,
    split: Split,
    data_root: Union[str, Path],
    regime: Regime,
    few_shot_fraction: Optional[float] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 512,
    seed: int = 42,
) -> Optional[DataLoader]:
    """
    Build a DataLoader for a dataset/split/regime combination.

    Returns None for zero-shot train split (no training in zero-shot).
    Few-shot sampling is patient-level (seed=42) to prevent data leakage.
    """
    data_root = Path(data_root)

    # Zero-shot has no training phase
    if regime == "zero_shot" and split == "train":
        logger.info("Zero-shot regime: skipping train dataloader for %s", dataset_name)
        return None

    csv_path = data_root / "splits" / f"{dataset_name}_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    # Select augmentation with per-dataset normalization stats
    ds_stats = DATASET_STATS.get(dataset_name, {})
    mean = ds_stats.get("mean")
    std  = ds_stats.get("std")

    if split == "train":
        aug_mode = "lora" if regime == "few_shot" else "train"
    else:
        aug_mode = "val"
    transform = get_augmentation(aug_mode, img_size=img_size, mean=mean, std=std)

    # Build full dataset from CSV
    # Processed images live under data_root/processed/{dataset}/images/
    dataset = ThyroidDataset(
        dataset_name=dataset_name,
        split_csv=csv_path,
        data_root=data_root / "processed",
        transform=transform,
        mean=mean,
        std=std,
    )

    # Few-shot: sample a fraction of patients, keep all their images
    if regime == "few_shot" and split == "train":
        if few_shot_fraction is None or few_shot_fraction not in _VALID_FRACTIONS:
            raise ValueError(
                f"few_shot_fraction must be one of {_VALID_FRACTIONS}, got {few_shot_fraction}"
            )
        unique_patients = np.unique(dataset.patient_ids)
        n_total = len(unique_patients)
        n_sample = max(1, int(n_total * few_shot_fraction))

        rng = np.random.default_rng(seed)
        sampled_patients = set(rng.choice(unique_patients, size=n_sample, replace=False).tolist())

        indices = [i for i, pid in enumerate(dataset.patient_ids) if pid in sampled_patients]
        effective_dataset: Union[ThyroidDataset, Subset] = Subset(dataset, indices)

        logger.info(
            "%s few-shot train (%.0f%%): %d/%d patients → %d images",
            dataset_name, few_shot_fraction * 100, n_sample, n_total, len(indices),
        )
    else:
        effective_dataset = dataset
        logger.info(
            "%s %s (%s): %d images, %d patients",
            dataset_name, split, regime, len(dataset), len(set(dataset.patient_ids)),
        )

    shuffle = split == "train"
    drop_last = split == "train" and len(effective_dataset) > batch_size

    # Seed the shuffle stream so batch order is reproducible run-to-run; the
    # box jitter generator (trainer._box_generator) is keyed on batch_idx, so a
    # deterministic shuffle is what makes the seeded jitter meaningful (CRIT-1).
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    return DataLoader(
        effective_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        generator=loader_generator,
        worker_init_fn=_seed_worker,
    )


def get_all_dataloaders(
    dataset_name: DatasetName,
    data_root: Union[str, Path],
    regime: Regime,
    few_shot_fraction: Optional[float] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 512,
    seed: int = 42,
) -> dict[str, Optional[DataLoader]]:
    """Get train/val/test DataLoaders for a dataset and supervision regime."""
    return {
        split: get_dataloader(
            dataset_name=dataset_name,
            split=split,
            data_root=data_root,
            regime=regime,
            few_shot_fraction=few_shot_fraction,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            seed=seed,
        )
        for split in ("train", "val", "test")
    }


def get_few_shot_dataloaders(
    dataset_name: DatasetName,
    data_root: Union[str, Path],
    fraction: float,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 512,
    seed: int = 42,
) -> dict[str, Optional[DataLoader]]:
    """Shortcut for few-shot DataLoaders with LoRA default batch size (4)."""
    return get_all_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        regime="few_shot",
        few_shot_fraction=fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Patient-level split validation utilities (was data/splits.py)
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ["thyroidxl", "tn3k", "ddti", "stanford_aimi"]


def validate_no_leakage(splits_dir: Path, dataset_name: str) -> Dict:
    """
    Validate no real data leakage between splits.

    Primary check: filename overlap (definitive — same file in 2 splits = leak).
    Secondary check: patient_id overlap, skipped for datasets where patient_id
    is a local index (TN3K: trainval_ and test_ prefix images both start at 0).
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset_name must be one of {SUPPORTED_DATASETS}")

    splits_dir = Path(splits_dir)
    dfs = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{dataset_name}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        dfs[split] = pd.read_csv(path)

    # Primary check: filename overlap (real leakage)
    filenames = {split: set(df["filename"]) for split, df in dfs.items()}
    filename_leakage = sorted(
        filenames["train"] & filenames["val"]
        | filenames["train"] & filenames["test"]
        | filenames["val"] & filenames["test"]
    )

    # Secondary check: patient_id overlap (only meaningful if patient_id is globally unique)
    # TN3K patient_id is a local index per subset — skip patient check for it
    pid_leakage: list = []
    if dataset_name != "tn3k":
        patients = {split: set(df["patient_id"].astype(str)) for split, df in dfs.items()}
        pid_leakage = sorted(
            patients["train"] & patients["val"]
            | patients["train"] & patients["test"]
            | patients["val"] & patients["test"]
        )

    leakage = filename_leakage  # filenames are the ground truth
    # For TN3K, patient_id is a per-subset local index (filename proxy), not a true global patient ID,
    # so this count is the number of distinct images, not the number of distinct patients.
    patients_per_split = {split: df["patient_id"].nunique() for split, df in dfs.items()}
    unit = "images" if dataset_name == "tn3k" else "patients"

    result = {
        "ok": len(leakage) == 0 and len(pid_leakage) == 0,
        "train_patients": patients_per_split["train"],
        "val_patients": patients_per_split["val"],
        "test_patients": patients_per_split["test"],
        "leakage": leakage,
        "patient_id_leakage": pid_leakage,
    }

    if result["ok"]:
        logger.info(
            "No leakage: %s — train=%d val=%d test=%d (%s)",
            dataset_name, result["train_patients"], result["val_patients"], result["test_patients"], unit,
        )
    else:
        if leakage:
            logger.error("FILENAME LEAKAGE in %s: %d files shared", dataset_name, len(leakage))
        if pid_leakage:
            logger.error("PATIENT_ID LEAKAGE in %s: %d patients shared", dataset_name, len(pid_leakage))

    return result


def validate_all_datasets(splits_dir: Path) -> bool:
    """Validate all 4 datasets and print summary table. Returns True if all pass."""
    splits_dir = Path(splits_dir)
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    results = {}
    for dataset in SUPPORTED_DATASETS:
        try:
            results[dataset] = validate_no_leakage(splits_dir, dataset)
        except Exception as e:
            logger.error("Error validating %s: %s", dataset, e)
            results[dataset] = {"ok": False, "train_patients": 0, "val_patients": 0, "test_patients": 0, "leakage": []}

    print(f"\n{'Dataset':<20} {'Train':>7} {'Val':>7} {'Test':>7} {'Status'}")
    print("-" * 55)
    for ds in SUPPORTED_DATASETS:
        r = results[ds]
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{ds:<20} {r['train_patients']:>7} {r['val_patients']:>7} {r['test_patients']:>7} {status}")
    print()

    return all(r["ok"] for r in results.values())


def get_split_stats(splits_dir: Path, dataset_name: str) -> Dict:
    """Return image/grouping statistics for a dataset's splits.

    The grouping unit is not the same for every dataset, so the counts are
    labelled rather than all called "patients":

    * ThyroidXL, DDTI  -- ``patient_id`` is a real patient identifier.
    * Stanford AIMI    -- ``patient_id`` holds the nodule id (``annot_id``); the
      distributed HDF5 carries no patient field, so the nodule is the finest
      grouping available and the one the splits use.
    * TN3K             -- ``patient_id`` is a per-subset filename index, not a
      global id, so the same value in train and test refers to different cases.
      TN3K is one image per case, so its unit is the image and any patient-level
      count over it would be meaningless. ``validate_no_leakage`` makes the same
      distinction.
    """
    splits_dir = Path(splits_dir)
    dfs = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{dataset_name}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        dfs[split] = pd.read_csv(path)

    combined = pd.concat(dfs.values(), ignore_index=True)

    unit = {"tn3k": "images", "stanford_aimi": "nodules"}.get(dataset_name, "patients")
    stats = {
        "dataset": dataset_name,
        "grouping_unit": unit,
        "total_images": len(combined),
        "images_per_split": {s: len(df) for s, df in dfs.items()},
    }
    if unit == "images":
        # patient_id collides across subsets here; counting it would invent groups.
        return stats

    imgs_per_group = combined.groupby("patient_id").size()
    stats[f"total_{unit}"] = combined["patient_id"].nunique()
    stats[f"{unit}_per_split"] = {s: df["patient_id"].nunique() for s, df in dfs.items()}
    stats[f"images_per_{unit[:-1]}"] = {
        "min": int(imgs_per_group.min()),
        "mean": float(imgs_per_group.mean()),
        "max": int(imgs_per_group.max()),
    }
    return stats
