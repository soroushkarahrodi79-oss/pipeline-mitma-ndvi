"""
CDSE product downloader.

Usage
-----
  # Download + auto-extract (default)
  python src/download.py --output-dir data/raw/scenes

  # Download from a specific metadata file
  python src/download.py --metadata data/raw/acquisitions_20260522_20260529.json --output-dir data/raw/scenes

  # Download + extract + immediately preprocess into data/processed/
  python src/download.py --output-dir data/raw/scenes --preprocess

  # Dry-run: show what would be downloaded without writing anything
  python src/download.py --output-dir data/raw/scenes --dry-run
"""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Ensure src/ is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, CONFIG_DIR, AcquisitionError, load_yaml_config, setup_logging

logger = setup_logging("download")

CDSE_TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"
CHUNK_SIZE       = 8 * 1024 * 1024    # 8 MB chunks
MIN_FREE_BYTES   = 500 * 1024 * 1024  # refuse to start if less than 500 MB free


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_cdse_token(username: str, password: str) -> str:
    """Acquire a short-lived OAuth2 Bearer token from CDSE identity service."""
    try:
        resp = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id":  "cdse-public",
                "username":   username,
                "password":   password,
            },
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token", "")
        if not token:
            raise AcquisitionError("CDSE returned an empty access token.")
        logger.info("CDSE authentication successful (token length: %d).", len(token))
        return token
    except requests.RequestException as exc:
        raise AcquisitionError(f"CDSE authentication failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _start_download_request(url: str, token: str, resume_pos: int = 0) -> requests.Response:
    """Open a streaming GET request, optionally with Range for resume."""
    headers = {"Authorization": f"Bearer {token}"}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"
    resp = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    if resp.status_code == 401:
        raise AcquisitionError("CDSE token expired during download. Re-authenticate and retry.")
    if resp.status_code == 429:
        raise requests.RequestException("CDSE rate limit (429). Will retry.")
    resp.raise_for_status()
    return resp


def _check_disk_space(output_dir: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(output_dir).free
    if free < required_bytes + MIN_FREE_BYTES:
        raise AcquisitionError(
            f"Insufficient disk space on {output_dir.anchor}. "
            f"Required: {required_bytes / 1024**3:.1f} GB + 500 MB buffer. "
            f"Available: {free / 1024**3:.2f} GB. "
            f"Free up space or use --output-dir pointing to a drive with more room."
        )


def _format_speed(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1024 * 1024:
        return f"{bytes_per_sec / 1024**2:.1f} MB/s"
    return f"{bytes_per_sec / 1024:.0f} KB/s"


def download_product(
    product_id: str,
    product_name: str,
    token: str,
    output_dir: Path,
    dry_run: bool = False,
) -> Tuple[Path, int]:
    """
    Stream-download a single Sentinel-2 .SAFE archive (as .zip).

    Returns (output_path, total_bytes_written).
    Supports resume: if a partial file exists, continues from where it left off.
    """
    url       = f"{CDSE_DOWNLOAD_BASE}({product_id})/$value"
    safe_name = product_name.replace(".SAFE", "")
    out_path  = output_dir / f"{safe_name}.zip"

    if dry_run:
        logger.info("[DRY-RUN] Would download %s → %s", product_id, out_path)
        return out_path, 0

    # Check total size with a HEAD-like initial request
    resp = _start_download_request(url, token)
    total_bytes = int(resp.headers.get("Content-Length", 0))
    resp.close()

    resume_pos = 0
    if out_path.exists():
        resume_pos = out_path.stat().st_size
        if total_bytes > 0 and resume_pos >= total_bytes:
            logger.info("Already complete: %s (%d MB)", out_path.name, total_bytes // 1024**2)
            return out_path, total_bytes

    _check_disk_space(output_dir, max(total_bytes - resume_pos, 0))

    size_str = f"{total_bytes / 1024**2:.0f} MB" if total_bytes else "unknown size"
    logger.info(
        "Downloading %s (%s)%s",
        product_name,
        size_str,
        f" — resuming from {resume_pos // 1024**2} MB" if resume_pos else "",
    )

    resp  = _start_download_request(url, token, resume_pos=resume_pos)
    mode  = "ab" if resume_pos > 0 else "wb"
    written = resume_pos
    t0 = t_last = time.monotonic()
    bytes_last = 0

    with open(out_path, mode) as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            fh.write(chunk)
            written     += len(chunk)
            bytes_last  += len(chunk)
            now          = time.monotonic()
            elapsed      = now - t_last
            if elapsed >= 10:
                speed = bytes_last / elapsed
                pct   = 100 * written / total_bytes if total_bytes else 0
                logger.info(
                    "  %s  %d / %d MB  %.0f%%  %s",
                    out_path.name,
                    written // 1024**2,
                    total_bytes // 1024**2,
                    pct,
                    _format_speed(speed),
                )
                t_last     = now
                bytes_last = 0

    total_elapsed = time.monotonic() - t0
    avg_speed = (written - resume_pos) / max(total_elapsed, 1)
    logger.info(
        "Completed %s — %d MB in %.0f s (%s)",
        out_path.name,
        written // 1024**2,
        total_elapsed,
        _format_speed(avg_speed),
    )
    return out_path, written


def download_from_metadata(
    metadata_path: str,
    output_dir: Path,
    config_path: Optional[str] = None,
    dry_run: bool = False,
) -> List[Path]:
    """
    Download all online products listed in a metadata JSON file.

    The metadata JSON is generated by ingestion.collect_new_acquisitions.
    """
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    products: List[Dict] = json.loads(meta_file.read_text(encoding="utf-8"))
    if not products:
        logger.info("Metadata file contains no products.")
        return []

    online   = [p for p in products if p.get("online", True)]
    offline  = [p for p in products if not p.get("online", True)]
    if offline:
        logger.warning("%d product(s) are offline and will be skipped.", len(offline))

    logger.info("Products to download: %d  (dry_run=%s)", len(online), dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)

    config   = load_yaml_config(config_path)
    api_cfg  = config.get("api", {})
    username = api_cfg.get("username", "")
    password = api_cfg.get("password", "")

    if not username or not password:
        raise AcquisitionError(
            "CDSE credentials missing from config. "
            "Set api.username and api.password in config/thresholds.yaml."
        )

    token        = get_cdse_token(username, password)
    token_time   = time.monotonic()
    downloaded   = []

    for i, product in enumerate(online, 1):
        pid   = product.get("product_id", "")
        name  = product.get("title", pid)
        cloud = product.get("cloudcoverpercentage")
        logger.info(
            "[%d/%d] %s  cloud=%.1f%%",
            i, len(online), name[:60], cloud if cloud is not None else 0,
        )

        # Refresh token every 8 minutes (tokens expire in ~10 min)
        if time.monotonic() - token_time > 480:
            logger.info("Refreshing CDSE token...")
            token      = get_cdse_token(username, password)
            token_time = time.monotonic()

        try:
            path, _ = download_product(pid, name, token, output_dir, dry_run=dry_run)
            downloaded.append(path)
        except AcquisitionError as exc:
            logger.error("Failed to download %s: %s", name, exc)
        except requests.RequestException as exc:
            logger.error("Network error downloading %s: %s", name, exc)

    logger.info("Download complete: %d / %d products.", len(downloaded), len(online))
    return downloaded


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_archive(zip_path: Path, output_dir: Path) -> Path:
    """
    Extract a .zip Sentinel-2 archive to output_dir.
    Returns the path to the extracted .SAFE folder.
    Skips extraction if the .SAFE folder already exists and is non-empty.
    """
    safe_name = zip_path.stem + ".SAFE"
    safe_path = output_dir / safe_name

    if safe_path.exists() and any(safe_path.iterdir()):
        logger.info("Already extracted: %s", safe_path)
        return safe_path

    logger.info("Extracting %s → %s ...", zip_path.name, output_dir)
    t0 = time.monotonic()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    elapsed = time.monotonic() - t0
    logger.info("Extracted in %.0f s → %s", elapsed, safe_path)
    return safe_path


def extract_all(zip_paths: List[Path], output_dir: Path) -> List[Path]:
    """Extract a list of downloaded .zip archives. Returns list of .SAFE paths."""
    safe_paths = []
    for zip_path in zip_paths:
        if not zip_path.exists():
            logger.warning("Zip file not found, skipping extraction: %s", zip_path)
            continue
        try:
            safe_paths.append(extract_archive(zip_path, output_dir))
        except (zipfile.BadZipFile, Exception) as exc:
            logger.error("Extraction failed for %s: %s", zip_path.name, exc)
    return safe_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _latest_metadata_file() -> Optional[Path]:
    candidates = sorted(RAW_DIR.glob("acquisitions_*.json"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 .SAFE archives from CDSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save .zip archives (e.g. D:\\Sentinel2\\raw). Needs ~1 GB per scene.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to a metadata JSON. Defaults to the latest acquisitions_*.json in data/raw/.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "thresholds.yaml"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without writing any files.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep .zip files without extracting to .SAFE folders.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="After extraction, run preprocess_batch on all .SAFE scenes.",
    )
    args = parser.parse_args()

    metadata = args.metadata
    if metadata is None:
        latest = _latest_metadata_file()
        if latest is None:
            print("ERROR: No metadata JSON found in data/raw/. Run --job weekly first.", file=sys.stderr)
            sys.exit(1)
        metadata = str(latest)
        logger.info("Using latest metadata file: %s", metadata)

    output_dir = Path(args.output_dir)

    # Pre-flight disk space check (rough estimate: 1.2 GB per product)
    try:
        meta_count = len(json.loads(Path(metadata).read_text()))
        estimated  = meta_count * 1_300 * 1024 * 1024
        check_dir  = output_dir.parent if not output_dir.exists() else output_dir
        free       = shutil.disk_usage(check_dir).free
        print(f"Products to download : {meta_count}")
        print(f"Estimated total size : {estimated / 1024**3:.1f} GB")
        print(f"Free space on target : {free / 1024**3:.1f} GB")
        if not args.dry_run and free < estimated + MIN_FREE_BYTES:
            print(f"\nERROR: Not enough space on {output_dir.anchor}. "
                  f"Need {estimated/1024**3:.1f} GB + 500 MB buffer, have {free/1024**3:.1f} GB free.",
                  file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        pass  # output_dir doesn't exist yet; will be created

    try:
        zip_paths = download_from_metadata(
            metadata_path=metadata,
            output_dir=output_dir,
            config_path=args.config,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            sys.exit(0)

        print(f"\nDownloaded {len(zip_paths)} file(s):")
        for p in zip_paths:
            size_mb = p.stat().st_size // 1024**2 if p.exists() else 0
            print(f"  {p}  ({size_mb} MB)")

        # --- extraction ---
        if not args.no_extract and zip_paths:
            print("\nExtracting archives...")
            safe_paths = extract_all(zip_paths, output_dir)
            print(f"\nExtracted {len(safe_paths)} .SAFE folder(s):")
            for sp in safe_paths:
                print(f"  {sp}")

            # --- optional preprocessing ---
            if args.preprocess and safe_paths:
                print("\nPreprocessing scenes...")
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import preprocess as _preprocess
                from common import PROCESSED_DIR
                processed = _preprocess.preprocess_batch(
                    [str(sp) for sp in safe_paths],
                    output_dir=str(PROCESSED_DIR),
                    workers=2,
                )
                print(f"\nPreprocessed {len(processed)} scene(s) → {PROCESSED_DIR}")
                for pp in processed:
                    print(f"  {pp}")

    except (AcquisitionError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
