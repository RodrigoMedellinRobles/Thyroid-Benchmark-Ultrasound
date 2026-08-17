"""
Base class and shared utilities for zero-shot foundation model inference.
All wrappers: predict(image_uint8_rgb_HxWx3, box_xyxy) → logit_HxW_float32
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image


class FoundationPredictor(ABC):

    @abstractmethod
    def predict(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        image: HxWx3 uint8 RGB numpy
        box:   [x1, y1, x2, y2] float32 pixel coords
        returns: HxW float32 logit map (pre-sigmoid; threshold at 0 or via compute_metrics)
        """

    def close(self) -> None:
        """Release GPU resources. Override if needed."""


def get_gt_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """[x1,y1,x2,y2] float32 from binary mask. None if mask empty."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def jitter_box(
    box: np.ndarray,
    rng: np.random.Generator,
    jitter_frac: float = 0.1,
    img_h: int = 512,
    img_w: int = 512,
) -> np.ndarray:
    """±jitter_frac * box_dim random noise per coordinate, clipped to image bounds."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([
        float(np.clip(x1 + rng.uniform(-jitter_frac, jitter_frac) * w, 0, img_w - 1)),
        float(np.clip(y1 + rng.uniform(-jitter_frac, jitter_frac) * h, 0, img_h - 1)),
        float(np.clip(x2 + rng.uniform(-jitter_frac, jitter_frac) * w, 0, img_w - 1)),
        float(np.clip(y2 + rng.uniform(-jitter_frac, jitter_frac) * h, 0, img_h - 1)),
    ], dtype=np.float32)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel of LAB. Input/output: uint8 RGB."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2RGB).astype(np.uint8)


class ZeroShotDataset:
    """
    Returns raw uint8 RGB images + binary masks at img_size×img_size.
    No tensor normalization — each model wrapper handles its own preprocessing.
    """

    def __init__(
        self,
        dataset_name: str,
        split_csv: Path,
        data_root: Path,
        img_size: int = 512,
    ):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.img_size = img_size

        df = pd.read_csv(Path(split_csv))
        self.filenames: list[str] = df["filename"].tolist()
        self.patient_ids: list[str] = df["patient_id"].astype(str).tolist()

        self.image_dir = self.data_root / dataset_name / "images"
        self.mask_dir  = self.data_root / dataset_name / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fn  = self.filenames[idx]
        pid = self.patient_ids[idx]

        with Image.open(self.image_dir / fn) as img:
            image = np.array(img.convert("RGB"), dtype=np.uint8)
        with Image.open(self.mask_dir / fn) as msk:
            mask = (np.array(msk.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

        image = apply_clahe(image)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return image, mask, fn, pid
