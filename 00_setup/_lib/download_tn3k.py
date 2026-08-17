#!/usr/bin/env python3
"""
Download the TN3K thyroid-nodule segmentation dataset (TRFE-Net data bundle).

The dataset consists of a zip archive containing the TRFE-Net repository plus
the TN3K image/annotation pairs. After extraction and normalization the final
layout under <out>/extracted/ becomes::

    <out>/extracted/TRFE-Net-for-thyroid-nodule-segmentation-main/picture/
        trainval/  (images + masks)
        test/      (images + masks)

Primary source: Google Drive (file id: 1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F)
Fallback sources:
  - Hugging Face dataset: haifan-gong/TN3K
  - Baidu Pan (code: trfe)

Usage:
    python download_tn3k.py                          # default: Google Drive
    python download_tn3k.py --source gdrive
    python download_tn3k.py --source manual --zip /path/to/TN3K.zip
    python download_tn3k.py --out /absolute/path
    python download_tn3k.py --dry-run                # resolve URL, print target path, no fetch
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependencies – we catch ImportError and give clear instructions.
# ---------------------------------------------------------------------------

try:
    import tqdm
except ImportError:  # pragma: no cover
    sys.exit("Missing required dependency 'tqdm'. Install with: pip install tqdm")

# gdown is only required when --source gdrive (default)
_gdown_available = True
try:
    import gdown
except ImportError:
    _gdown_available = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_tn3k")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT = "thyroidbench"
REQUIRED_SUBDIR = "picture"  # must exist after extraction under the normalised top dir
TARGET_TOP_DIR = "TRFE-Net-for-thyroid-nodule-segmentation-main"

# Google Drive file id for the TRFE-Net zip bundle
GDRIVE_FILE_ID = "1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when extraction or layout verification fails."""


class DownloadError(Exception):
    """Raised after repeated download failures."""


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """
    Extract *zip_path* (a .zip archive) into *dest* while guarding against
    Zip Slip path traversal.  Raises :class:`ExtractionError` on dangerous
    entries or extraction failures.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ExtractionError(
            f"File {zip_path} is not a valid ZIP archive. "
            "The TN3K dataset is distributed only as a .zip file."
        )

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # First pass: validate all paths (reject absolute names or '..')
        for info in zf.infolist():
            sanitised = Path(info.filename).as_posix()
            if sanitised.startswith("/") or ".." in sanitised:
                raise ExtractionError(
                    f"Zip entry '{info.filename}' is unsafe (path traversal). Aborting."
                )
        members = zf.infolist()
        logger.info("Extracting %d entries from %s -> %s", len(members), zip_path.name, dest)
        for info in tqdm.tqdm(members, desc="Extracting", unit="file"):
            zf.extract(info, dest)
        logger.info("Extraction complete.")


def _normalize_top_dir(extract_root: Path, target_name: str) -> Path:
    """
    Rename the single top-level directory inside *extract_root* to *target_name*
    if it differs. Returns the path to the normalized top directory.
    """
    target_path = extract_root / target_name
    if target_path.is_dir():
        return target_path

    children = [d for d in extract_root.iterdir() if d.is_dir()]
    if len(children) == 0:
        raise ExtractionError(
            f"Expected a single top-level directory under {extract_root}, "
            "but found none. Extracted contents:\n"
            + "\n".join(str(p) for p in extract_root.iterdir())
        )
    if len(children) > 1:
        raise ExtractionError(
            f"Expected a single top-level directory under {extract_root}, "
            f"but found {len(children)}:\n"
            + "\n".join(str(p) for p in children)
        )

    old_dir = children[0]
    new_path = old_dir.rename(extract_root / target_name)
    logger.info("Renamed top directory: %s -> %s", old_dir.name, target_name)
    return new_path


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


def download_tn3k(
    out_dir: Path,
    source: str = "gdrive",
    zip_path: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """
    Download and extract the TN3K dataset.

    Returns the path to the normalized extraction root
    (``out_dir / extracted / TARGET_TOP_DIR``).
    """
    extract_root = out_dir / "extracted"
    target_root = extract_root / TARGET_TOP_DIR
    picture_dir = target_root / REQUIRED_SUBDIR

    # Early return if already complete and not forced
    if picture_dir.is_dir() and not force:
        logger.info("Dataset already present at %s. Use --force to re-download.", picture_dir)
        return target_root

    # Dry-run
    if dry_run:
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}" if source == "gdrive" else "(manual)"
        logger.info("Dry run – download source: %s", url)
        logger.info("Target extraction directory: %s", target_root)
        logger.info("No files downloaded.")
        return target_root

    # Gather the archive
    if source == "gdrive":
        if not _gdown_available:
            raise ImportError(
                "gdown is required for Google Drive downloads. Install with: pip install gdown"
            )
        tmp_zip = out_dir / "tn3k_download.zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading from Google Drive (file id: %s) ...", GDRIVE_FILE_ID)
        result = gdown.download(id=GDRIVE_FILE_ID, output=str(tmp_zip), quiet=False)
        if result is None:
            raise DownloadError(
                "gdown failed to download the file. Check the file id or connectivity.\n"
                "Alternative download sources: Hugging Face haifan-gong/TN3K, "
                "or Baidu Pan (code: trfe)."
            )
        archive = Path(result)
    else:  # manual
        if zip_path is None:
            raise ValueError("--zip <path> is required when --source=manual")
        if not zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        archive = zip_path
        logger.info("Using manually provided archive: %s", archive)

    # Extract
    if force and target_root.exists():
        logger.info("--force set: removing existing %s", target_root)
        shutil.rmtree(target_root)
    _safe_extract(archive, extract_root)

    # Normalize top directory
    normalized = _normalize_top_dir(extract_root, TARGET_TOP_DIR)

    # Verify expected subdirectory
    picture_actual = normalized / REQUIRED_SUBDIR
    if not picture_actual.is_dir():
        tree = []
        for root, _dirs, files in os.walk(normalized):
            level = root.replace(str(normalized), "").count(os.sep)
            tree.append(f"{'  ' * level}{os.path.basename(root)}/")
            for name in files:
                tree.append(f"{'  ' * (level + 1)}{name}")
        raise ExtractionError(
            f"Expected '{REQUIRED_SUBDIR}/' directory inside the extracted folder.\n"
            f"Extracted root: {normalized}\n"
            f"First 60 lines of tree:\n" + "\n".join(tree[:60]) + "\n\n"
            "Please verify the archive structure against this module's docstring."
        )

    # Cleanup temporary download when using gdrive
    if source == "gdrive" and archive.exists():
        archive.unlink()
        logger.info("Removed temporary download: %s", archive.name)

    logger.info("TN3K dataset ready at: %s", target_root)
    return target_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the TN3K thyroid-nodule segmentation dataset.",
    )
    parser.add_argument(
        "--out", type=str, default="data/raw",
        help="Output directory. Default: data/raw (relative to cwd). "
             "Dataset lands under <out>/extracted/TRFE-Net-.../picture/.",
    )
    parser.add_argument(
        "--source", choices=["gdrive", "manual"], default="gdrive",
        help="'gdrive' (default) uses Google Drive; 'manual' requires --zip <path>.",
    )
    parser.add_argument("--zip", type=str, default=None,
                        help="Path to a manually downloaded ZIP (with --source=manual).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract even if the dataset exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve the source, print target path, and exit without fetching.")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    try:
        download_tn3k(
            out_dir=out,
            source=args.source,
            zip_path=Path(args.zip) if args.zip else None,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (DownloadError, ExtractionError, FileNotFoundError, ImportError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
