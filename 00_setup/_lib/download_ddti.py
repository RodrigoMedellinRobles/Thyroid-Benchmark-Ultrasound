#!/usr/bin/env python3
"""
Download the DDTI (Digital Database of Thyroid Ultrasound Images) dataset.

The dataset (Pedraza et al., SPIE 2015) consists of ultrasound images alongside
XML polygon annotation files. After extraction and normalisation the expected
layout is::

    <out>/extracted/DDTI-Pedraza_Digital_Database_Thyroid_SPIE/
        <original_images_and_xml_files>

Primary source: official CIMA@LAB page (Universidad Nacional de Colombia).
Because the direct ZIP URL can change, a configurable ``--url`` override is
provided. An alternative source is the Kaggle mirror
``dasmehdixtr/ddti-thyroid-ultrasound-images`` (requires the Kaggle API).
Notebook ``00_setup/00_get_open_datasets.ipynb`` runs this for you.

Usage:
    python download_ddti.py                          # default documented URL
    python download_ddti.py --url <direct_zip_url>
    python download_ddti.py --zip /path/to/ddti.zip
    python download_ddti.py --out /absolute/path
    python download_ddti.py --dry-run                # resolve URL, print target path, no fetch
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
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    sys.exit("Missing required dependency 'requests'. Install with: pip install requests")

try:
    import tqdm
except ImportError:  # pragma: no cover
    sys.exit("Missing required dependency 'tqdm'. Install with: pip install tqdm")

# rarfile is optional – only needed if the archive is .rar (not typical for DDTI)
try:
    import rarfile
except ImportError:
    rarfile = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_ddti")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT = "thyroidbench"
TARGET_TOP_DIR = "DDTI-Pedraza_Digital_Database_Thyroid_SPIE"

# Default direct ZIP URL (unstable – subject to change).
# The official CIMA@LAB page and the alternative sources are listed above.
DEFAULT_ZIP_URL = "http://cimalab.intec.co/applications/thyroid/files/DDTI.zip"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when extraction or layout verification fails."""


class DownloadError(Exception):
    """Raised after repeated download failures."""


def _download_http(url: str, dest: Path, retries: int = 4, chunk_size: int = 8192) -> None:
    """Download *url* to *dest* with streaming + exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1.0,  # sleep 1, 2, 4, 8 seconds
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info("Downloading %s -> %s", url, dest)
    try:
        resp = session.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise DownloadError(
            f"HTTP download failed for {url}: {exc}\n"
            "The DDTI dataset is open-access but its direct URL is unstable.\n"
            "Please download it manually from the official CIMA@LAB page:\n"
            "  http://cimalab.intec.co/applications/thyroid/\n"
            "or use the Kaggle mirror: dasmehdixtr/ddti-thyroid-ultrasound-images.\n"
            "Then re-run with --zip <path>."
        ) from exc

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        progress = tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
        progress.close()


def _safe_extract(archive_path: Path, dest: Path) -> None:
    """
    Extract *archive_path* (.zip, .tar.*, or .rar) into *dest* while guarding
    against path traversal. Raises :class:`ExtractionError` on unsafe entries
    or unsupported types.
    """
    dest.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()

    # ── ZIP ────────────────────────────────────────────────────────────
    if ext == ".zip" or zipfile.is_zipfile(archive_path):
        if not zipfile.is_zipfile(archive_path):
            raise ExtractionError(f"File {archive_path} is not a valid ZIP archive.")
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                sanitised = Path(info.filename).as_posix()
                if sanitised.startswith("/") or ".." in sanitised:
                    raise ExtractionError(
                        f"Zip entry '{info.filename}' is unsafe (path traversal). Aborting."
                    )
            members = zf.infolist()
            logger.info("Extracting %d entries (ZIP) -> %s", len(members), dest)
            for info in tqdm.tqdm(members, desc="Extracting", unit="file"):
                zf.extract(info, dest)
            logger.info("Extraction complete.")

    # ── TAR.* ──────────────────────────────────────────────────────────
    elif ext in (".gz", ".bz2", ".xz", ".tar") or archive_path.name.endswith(".tar.gz"):
        import tarfile
        if not tarfile.is_tarfile(archive_path):
            raise ExtractionError(f"File {archive_path} is not a valid tar archive.")
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            for member in members:
                name = Path(member.name).as_posix()
                if name.startswith("/") or ".." in name:
                    raise ExtractionError(
                        f"Tar entry '{member.name}' is unsafe (path traversal). Aborting."
                    )
            logger.info("Extracting %d entries (tar) -> %s", len(members), dest)
            for member in tqdm.tqdm(members, desc="Extracting", unit="file"):
                tf.extract(member, dest)
            logger.info("Extraction complete.")

    # ── RAR ────────────────────────────────────────────────────────────
    elif ext == ".rar":
        if rarfile is None:
            raise ExtractionError(
                "RAR archive detected but 'rarfile' is not installed.\n"
                "Install it with: pip install rarfile"
            )
        if not rarfile.is_rarfile(archive_path):
            raise ExtractionError(f"File {archive_path} is not a valid RAR archive.")
        with rarfile.RarFile(archive_path) as rf:
            members = rf.infolist()
            for info in members:
                sanitised = Path(info.filename).as_posix()
                if sanitised.startswith("/") or ".." in sanitised:
                    raise ExtractionError(
                        f"Rar entry '{info.filename}' is unsafe (path traversal). Aborting."
                    )
            logger.info("Extracting %d entries (RAR) -> %s", len(members), dest)
            for info in tqdm.tqdm(members, desc="Extracting", unit="file"):
                rf.extract(info, dest)
            logger.info("Extraction complete.")

    else:
        raise ExtractionError(
            f"Unsupported archive type '{ext}' for file {archive_path}.\n"
            "DDTI is typically distributed as a .zip file."
        )


def _normalize_top_dir(extract_root: Path, target_name: str) -> Path:
    """Rename the single top-level dir inside *extract_root* to *target_name*."""
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


def download_ddti(
    out_dir: Path,
    url: Optional[str] = None,
    zip_path: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    """
    Download and extract the DDTI dataset. Returns the path to the normalized
    extraction root (``out_dir / extracted / TARGET_TOP_DIR``).
    """
    extract_root = out_dir / "extracted"
    target_root = extract_root / TARGET_TOP_DIR

    # Early return if already complete (heuristic: XML files present)
    xml_present = list(target_root.glob("*.xml")) if target_root.is_dir() else []
    if xml_present and not force:
        logger.info(
            "Dataset already present at %s (found %d XML files). Use --force to re-download.",
            target_root, len(xml_present),
        )
        return target_root

    # Dry-run
    if dry_run:
        display_url = url or DEFAULT_ZIP_URL
        logger.info("Dry run – download URL: %s", display_url)
        logger.info("Target extraction directory: %s", target_root)
        logger.info("No files downloaded.")
        return target_root

    # Determine the archive source
    if zip_path is not None:
        if not zip_path.is_file():
            raise FileNotFoundError(f"Provided archive not found: {zip_path}")
        archive = zip_path
        logger.info("Using pre-downloaded archive: %s", archive)
    else:
        actual_url = url or DEFAULT_ZIP_URL
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_zip = out_dir / "ddti_download.zip"
        _download_http(actual_url, tmp_zip)
        archive = tmp_zip

    # Extract
    if force and target_root.exists():
        logger.info("--force set: removing existing %s", target_root)
        shutil.rmtree(target_root)
    _safe_extract(archive, extract_root)

    # Normalize top directory
    normalized = _normalize_top_dir(extract_root, TARGET_TOP_DIR)

    # Verify expected XML annotations
    xml_files = list(normalized.rglob("*.xml"))
    if not xml_files:
        tree = []
        for root_, _dirs, files in os.walk(normalized):
            level = root_.replace(str(normalized), "").count(os.sep)
            tree.append(f"{'  ' * level}{os.path.basename(root_)}/")
            for name in files:
                tree.append(f"{'  ' * (level + 1)}{name}")
        raise ExtractionError(
            f"No XML files found inside {normalized}.\n"
            f"Extracted tree (first 60 lines):\n" + "\n".join(tree[:60]) + "\n\n"
            "The DDTI dataset should contain XML polygon annotations. "
            "The expected layout is documented in this module's docstring."
        )

    # Clean up downloaded archive (unless user-provided)
    if zip_path is None and archive.exists():
        archive.unlink()
        logger.info("Removed temporary download: %s", archive.name)

    logger.info("DDTI dataset ready at: %s (found %d XML files)", normalized, len(xml_files))
    return normalized


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the DDTI thyroid ultrasound dataset.",
    )
    parser.add_argument(
        "--out", type=str, default="data/raw",
        help="Output directory. Default: data/raw (relative to cwd). "
             "Dataset lands under <out>/extracted/DDTI-.../.",
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="Direct URL to the DDTI ZIP archive. If omitted, the default "
             "(potentially unstable) URL is used.",
    )
    parser.add_argument("--zip", type=str, default=None,
                        help="Path to a manually downloaded archive (ZIP, tar.gz, or RAR).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract even if the dataset exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve the URL, print target path, and exit without fetching.")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    try:
        download_ddti(
            out_dir=out,
            url=args.url,
            zip_path=Path(args.zip) if args.zip else None,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (DownloadError, ExtractionError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
