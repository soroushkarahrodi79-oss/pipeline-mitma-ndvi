"""
Historical baseline builder for the Hayedo de Montejo monitoring system.

Downloads and processes Sentinel-2 L2A scenes for the scientific phenological
window (July 1 – September 30) for each year 2021–present. Builds a
multi-year statistical baseline suitable for Z-score anomaly detection.

SCIENTIFIC RATIONALE
--------------------
The phenological window July–September corresponds to the period of maximum
canopy development in Fagus sylvatica at this altitude (1150–1450 m, 41°N).
During this period:
  - Leaf area index is at maximum
  - Photosynthetic activity is near peak
  - NDVI values are most stable and directly comparable between years
  - Inter-year variability primarily reflects climate and health signals,
    not phenological timing differences

BASELINE METHODOLOGY
--------------------
For each year, up to N_BEST_SCENES_PER_YEAR scenes with the lowest cloud
cover are selected from the July–September window. A median composite is
built per year. The multi-year stack is then used to compute:
  - Pixel-wise mean and standard deviation (for Z-score)
  - Pixel-wise percentiles (5th, 25th, 50th, 75th, 95th)
  - Temporal trend (Sen's slope estimator — robust to outliers)

A minimum of 3 years is required for a statistically meaningful baseline.
The system will operate in degraded mode (threshold-based detection) until
this minimum is met.

Usage (CLI):
    python src/baseline_builder.py --years 2021 2022 2023 2024 2025
    python src/baseline_builder.py --year 2024          # single year
    python src/baseline_builder.py --status             # show what exists
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    COMPOSITES_DIR, PROCESSED_DIR, RAW_DIR, CONFIG_DIR,
    OUTPUTS_ANOMALIES_DIR,
    AcquisitionError, InsufficientDataError,
    load_yaml_config, save_json, setup_logging,
)

logger = setup_logging("baseline_builder")

# Scientific configuration
PHENOLOGY_START_MONTH = 7    # July
PHENOLOGY_END_MONTH   = 9    # September
N_BEST_SCENES         = 4    # scenes per year (lowest cloud cover)
MAX_CLOUD_COVER       = 15.0
MIN_BASELINE_YEARS    = 3
PER_SCENE_GUARD_GB    = 2.5  # min free GB before a scene (zip + extracted SAFE coexist briefly)
CDSE_CATALOGUE        = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD         = "https://download.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL        = ("https://identity.dataspace.copernicus.eu/auth/realms/"
                         "CDSE/protocol/openid-connect/token")
BASELINE_DIR          = COMPOSITES_DIR / "baseline"
SCENES_STAGING_DIR    = RAW_DIR / "baseline_staging"


def _dir_bytes(p) -> int:
    from pathlib import Path as _P
    p = _P(p)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _save_year_report(report: dict) -> None:
    """Persist the per-year ingestion audit (STEP 5 reporting + checkpoint trail)."""
    try:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        report = dict(report)
        report["download_mb"] = round(report.get("download_bytes", 0) / 1024**2, 1)
        report["processing_mb"] = round(report.get("processing_bytes", 0) / 1024**2, 1)
        save_json(report, str(BASELINE_DIR / f"annual_ndvi_{report['year']}_report.json"))
    except Exception as exc:
        logger.warning("Could not save year report: %s", exc)


# ---------------------------------------------------------------------------
# CDSE helpers (minimal, self-contained)
# ---------------------------------------------------------------------------

def _get_token(username: str, password: str) -> str:
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={"grant_type": "password", "client_id": "cdse-public",
              "username": username, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _query_year_scenes(
    year: int,
    aoi_wkt: str,
    cloud_max: float = MAX_CLOUD_COVER,
    max_results: int = 50,
) -> List[Dict]:
    """Return available scenes for phenological window, sorted by cloud cover."""
    start = f"{year}-{PHENOLOGY_START_MONTH:02d}-01"
    end   = f"{year}-{PHENOLOGY_END_MONTH:02d}-30"

    odata_filter = (
        f"Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start ge {start}T00:00:00.000Z "
        f"and ContentDate/Start le {end}T23:59:59.999Z "
        f"and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {cloud_max})"
    )

    params = {
        "$filter":  odata_filter,
        "$top":     max_results,
        "$orderby": "ContentDate/Start asc",
        "$expand":  "Attributes",
    }
    try:
        resp = requests.get(CDSE_CATALOGUE, params=params, timeout=30)
        resp.raise_for_status()
        entries = resp.json().get("value", [])
    except Exception as exc:
        raise AcquisitionError(f"CDSE catalogue query failed for {year}: {exc}") from exc

    scenes = []
    for e in entries:
        cloud = next(
            (float(a["Value"]) for a in e.get("Attributes", []) if a.get("Name") == "cloudCover"),
            None,
        )
        scenes.append({
            "product_id":         e["Id"],
            "title":              e["Name"],
            "beginposition":      e.get("ContentDate", {}).get("Start", ""),
            "cloudcoverpercentage": cloud,
            "online":             e.get("Online", True),
            "download_url":       f"{CDSE_DOWNLOAD}({e['Id']})/$value",
            # Footprint geometry for client-side geospatial AOI validation (STEP 4)
            "geo_footprint":      e.get("GeoFootprint"),
            "footprint":          e.get("Footprint"),
        })

    # Sort by cloud cover ascending, then by date
    scenes.sort(key=lambda s: (s["cloudcoverpercentage"] or 99, s["beginposition"]))
    return scenes


def _download_scene(product_id: str, title: str, token: str, out_dir: Path) -> Optional[Path]:
    """Stream-download a SAFE archive. Returns path or None on failure."""
    safe_stem = title.replace(".SAFE", "")
    zip_path  = out_dir / f"{safe_stem}.zip"
    safe_path = out_dir / f"{safe_stem}.SAFE"

    if safe_path.exists() and any(safe_path.iterdir()):
        logger.info("Already extracted: %s", safe_path.name)
        return safe_path

    if not zip_path.exists():
        url = f"{CDSE_DOWNLOAD}({product_id})/$value"
        logger.info("Downloading %s ...", title[:60])
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=60,
                allow_redirects=True,
            )
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            t0 = time.monotonic()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
            elapsed = time.monotonic() - t0
            logger.info(
                "Downloaded %d MB in %.0f s (%.1f MB/s)",
                written // 1024**2, elapsed, written / 1024**2 / max(elapsed, 1),
            )
        except Exception as exc:
            logger.error("Download failed for %s: %s", title, exc)
            zip_path.unlink(missing_ok=True)
            return None

    # Extract
    import zipfile
    try:
        logger.info("Extracting %s ...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink()   # delete zip to reclaim space immediately
        logger.info("Extracted to %s", safe_path)
        return safe_path
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        zip_path.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------

def build_annual_composite_for_year(
    year: int,
    config_path: Optional[str] = None,
    aoi_path: Optional[str] = None,
    max_scenes: int = N_BEST_SCENES,
    keep_processed: bool = False,
) -> Optional[Path]:
    """
    Download, preprocess, and composite the best scenes for a given year.

    1. Query CDSE for phenological window scenes
    2. Download up to max_scenes (lowest cloud cover)
    3. Preprocess each scene (8-band TIF, AOI-clipped)
    4. Compute NDVI for each preprocessed scene
    5. Build median composite
    6. Save as data/composites/baseline/annual_ndvi_{year}.tif
    7. Clean up staging data

    Returns the path to the annual composite TIF, or None if no data available.
    """
    composite_path = BASELINE_DIR / f"annual_ndvi_{year}.tif"
    if composite_path.exists():
        logger.info("Annual composite for %d already exists: %s", year, composite_path)
        return composite_path

    config     = load_yaml_config(config_path)
    api_cfg    = config.get("api", {})
    username   = api_cfg.get("username", "")
    password   = api_cfg.get("password", "")
    cloud_max  = api_cfg.get("cloud_cover_max", MAX_CLOUD_COVER)

    from common import load_geojson
    from ingestion import _geojson_to_wkt
    import aoi_validation as _aoiv
    aoi_geojson = load_geojson(aoi_path)
    aoi_wkt     = _geojson_to_wkt(aoi_geojson)
    aoi_geom    = _aoiv.load_aoi_geom(aoi_path)

    # Per-year audit accumulator (STEP 5 reporting)
    report = {"year": year, "scenes_discovered": 0, "scenes_accepted": 0,
              "scenes_rejected": 0, "rejections": [], "accepted_titles": [],
              "download_bytes": 0, "processing_bytes": 0, "composite_path": None}

    # --- query scenes ---
    logger.info("Querying CDSE for %d phenological window (Jul-Sep)...", year)
    scenes = _query_year_scenes(year, aoi_wkt, cloud_max=cloud_max)
    online = [s for s in scenes if s.get("online", True)]
    report["scenes_discovered"] = len(scenes)

    if not online:
        logger.warning("No cloud-free scenes found for %d in Jul-Sep window.", year)
        _save_year_report(report)
        return None

    # --- STEP 4: geospatial AOI pre-validation (do NOT trust names) ---
    accepted_scenes = []
    for s in online:
        ok, reason = _aoiv.validate_scene_pre_download(s, aoi_geom)
        if ok:
            accepted_scenes.append(s)
        else:
            report["scenes_rejected"] += 1
            report["rejections"].append({"title": s["title"], "stage": "pre_download",
                                          "reason": reason})
            logger.warning("REJECTED (pre-download) %s: %s", s["title"][:55], reason)

    selected = accepted_scenes[:max_scenes]
    logger.info(
        "Selected %d/%d AOI-validated scenes for %d (cloud: %s%%)",
        len(selected), len(online), year,
        ", ".join(f"{s['cloudcoverpercentage']:.1f}" for s in selected if s['cloudcoverpercentage'] is not None),
    )

    # --- check disk space ---
    staging = SCENES_STAGING_DIR / str(year)
    staging.mkdir(parents=True, exist_ok=True)

    # Scenes are downloaded, processed and DELETED one at a time, so the true
    # peak footprint is ~one scene + its processed product, not all scenes at
    # once. The per-scene disk_guard in the loop is the real safeguard; this is
    # only a minimum-headroom gate so a year does not start with no room at all.
    min_headroom_bytes = int((PER_SCENE_GUARD_GB + 0.5) * 1024**3)
    free_bytes         = shutil.disk_usage(staging).free
    if free_bytes < min_headroom_bytes:
        raise AcquisitionError(
            f"Insufficient disk space to start year {year}. "
            f"Need >= {(PER_SCENE_GUARD_GB+0.5):.1f} GB headroom for one scene. "
            f"Free: {free_bytes/1024**3:.2f} GB."
        )

    # --- authenticate ---
    if not username or not password:
        raise AcquisitionError("CDSE credentials missing. Set api.username and api.password in thresholds.yaml.")
    token = _get_token(username, password)
    token_time = time.monotonic()

    # --- download and process ---
    import preprocess as _preprocess
    import indices as _indices

    ndvi_paths = []
    for i, scene in enumerate(selected, 1):
        # Refresh token every 8 min
        if time.monotonic() - token_time > 480:
            token = _get_token(username, password)
            token_time = time.monotonic()

        logger.info(
            "[%d/%d] %s (cloud=%.1f%%)",
            i, len(selected), scene["title"][:60],
            scene["cloudcoverpercentage"] or 0,
        )

        # STEP 6: per-scene disk guard (skip if not enough headroom for one scene)
        ok, free = _aoiv.disk_guard(PER_SCENE_GUARD_GB)
        if not ok:
            logger.error("Disk guard: only %.2f GB free (< %.1f GB needed for a scene); "
                         "skipping remaining downloads for %d.", free, PER_SCENE_GUARD_GB, year)
            report["rejections"].append({"title": scene["title"], "stage": "disk_guard",
                                         "reason": f"only {free} GB free"})
            break

        safe_path = _download_scene(scene["product_id"], scene["title"], token, staging)
        if safe_path is None:
            report["scenes_rejected"] += 1
            report["rejections"].append({"title": scene["title"], "stage": "download",
                                         "reason": "download/extraction failed"})
            continue
        report["download_bytes"] += _dir_bytes(safe_path)

        try:
            proc_path = _preprocess.preprocess_scene(str(safe_path), output_dir=str(staging / "processed"))
            # STEP 4: post-download geospatial validation of the real raster extent
            ok, reason = _aoiv.validate_raster_post_download(str(proc_path), aoi_geom)
            if not ok:
                report["scenes_rejected"] += 1
                report["rejections"].append({"title": scene["title"], "stage": "post_download",
                                             "reason": reason})
                logger.warning("REJECTED (post-download) %s: %s", scene["title"][:55], reason)
                _aoiv.quarantine_path(str(proc_path), f"{scene['title']}: {reason}",
                                      subdir=f"aoi_rejected/{year}")
            else:
                idx_result = _indices.compute_scene_indices(
                    str(proc_path), indices=["NDVI"], output_dir=str(staging / "indices"),
                )
                if "NDVI" in idx_result:
                    ndvi_paths.append(str(idx_result["NDVI"]))
                    report["scenes_accepted"] += 1
                    report["accepted_titles"].append(scene["title"])
                    report["processing_bytes"] += _dir_bytes(proc_path)
        except Exception as exc:
            logger.error("Processing failed for %s: %s", safe_path.name, exc)
            report["rejections"].append({"title": scene["title"], "stage": "processing",
                                         "reason": str(exc)[:160]})
        finally:
            # Free space: delete extracted SAFE
            try:
                shutil.rmtree(safe_path, ignore_errors=True)
            except Exception:
                pass

    if not ndvi_paths:
        logger.warning("No valid NDVI files produced for year %d.", year)
        shutil.rmtree(staging, ignore_errors=True)
        _save_year_report(report)
        return None

    # --- build annual composite ---
    import compositing as _compositing
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    composite = _compositing.build_median_composite(ndvi_paths, out_path=str(composite_path))
    logger.info("Annual composite for %d saved: %s", year, composite)
    report["composite_path"] = str(composite)
    _save_year_report(report)

    # Save metadata
    meta = {
        "year":           year,
        "scenes_queried": len(scenes),
        "scenes_used":    len(ndvi_paths),
        "scenes":         [{"title": s["title"], "cloud": s["cloudcoverpercentage"]} for s in selected],
        "composite_path": str(composite),
        "built_at":       datetime.now(timezone.utc).isoformat(),
        "phenology_window": f"{year}-07-01 to {year}-09-30",
        "method":         "median composite of lowest-cloud-cover scenes",
    }
    save_json(meta, str(BASELINE_DIR / f"annual_ndvi_{year}_metadata.json"))

    # Clean staging
    if not keep_processed:
        shutil.rmtree(staging, ignore_errors=True)

    return composite


# ---------------------------------------------------------------------------
# Baseline statistics
# ---------------------------------------------------------------------------

def build_baseline_statistics(
    annual_composite_paths: List[str],
    out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Build a multi-year statistical baseline from annual composite rasters.

    Computes pixel-wise:
      - mean, standard deviation (for Z-score anomaly detection)
      - percentiles: 5, 10, 25, 50, 75, 90, 95
      - Sen's slope (robust trend estimator)
      - coefficient of variation

    Requires at least MIN_BASELINE_YEARS composites.

    Returns dict mapping statistic name → output TIF path.
    """
    if len(annual_composite_paths) < MIN_BASELINE_YEARS:
        raise InsufficientDataError(
            f"Baseline requires >= {MIN_BASELINE_YEARS} annual composites "
            f"(have {len(annual_composite_paths)})."
        )

    if out_dir is None:
        out_dir = BASELINE_DIR / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    stacks, years, profile = [], [], None
    for path in sorted(annual_composite_paths):
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float64")
            nd  = src.nodata or -9999.0
            stacks.append(np.where(arr == nd, np.nan, arr))
            profile = dict(src.profile)
            # Extract year from filename
            stem = Path(path).stem
            for tok in stem.split("_"):
                if len(tok) == 4 and tok.isdigit() and 2000 <= int(tok) <= 2100:
                    years.append(int(tok))
                    break

    n_years = len(stacks)
    stack   = np.stack(stacks, axis=0)   # (T, H, W)

    def _write(array: np.ndarray, name: str) -> Path:
        p = out_dir / f"baseline_{name}.tif"
        profile.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw", driver="GTiff")
        out = np.where(np.isnan(array), -9999.0, array).astype("float32")
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(out, 1)
        return p

    outputs = {}

    # Basic statistics
    outputs["mean"]   = _write(np.nanmean(stack,   axis=0), "mean")
    raw_std           = np.nanstd(stack, axis=0)
    # Enforce minimum std of 0.01 to prevent division by zero in Z-score
    outputs["std"]    = _write(np.where(raw_std < 0.01, 0.01, raw_std), "std")
    outputs["median"] = _write(np.nanmedian(stack, axis=0), "median")
    outputs["cv"]     = _write(
        np.where(np.nanmean(stack, axis=0) > 0,
                 np.nanstd(stack, axis=0) / np.nanmean(stack, axis=0), np.nan),
        "cv",
    )

    # Percentiles
    for pct in (5, 10, 25, 75, 90, 95):
        outputs[f"p{pct:02d}"] = _write(np.nanpercentile(stack, pct, axis=0), f"p{pct:02d}")

    # Sen's slope (robust linear trend — resistant to outliers unlike OLS)
    # Sen's slope = median of all pairwise slopes (Theil 1950, Sen 1968)
    if n_years >= 3:
        n_pairs = n_years * (n_years - 1) // 2
        slopes  = np.full((n_pairs, stack.shape[1], stack.shape[2]), np.nan)
        idx = 0
        for i in range(n_years):
            for j in range(i + 1, n_years):
                dt = years[j] - years[i] if len(years) == n_years else (j - i)
                if dt > 0:
                    slopes[idx] = (stack[j] - stack[i]) / dt
                idx += 1
        outputs["sens_slope"] = _write(np.nanmedian(slopes, axis=0), "sens_slope")

    # Number of valid years per pixel
    outputs["n_valid"] = _write(
        np.sum(np.isfinite(stack), axis=0).astype("float32"), "n_valid"
    )

    logger.info("Baseline statistics built from %d annual composites (%s)", n_years, years)

    # Metadata record
    meta = {
        "n_years":         n_years,
        "years":           years,
        "composites_used": annual_composite_paths,
        "statistics":      {k: str(v) for k, v in outputs.items()},
        "built_at":        datetime.now(timezone.utc).isoformat(),
        "method": {
            "mean_std":    "pixel-wise, NaN propagation excluded",
            "percentiles": "5, 10, 25, 50, 75, 90, 95",
            "trend":       "Sen's slope estimator (Theil 1950 / Sen 1968) — robust to outliers",
            "std_floor":   "0.01 NDVI units applied to prevent Z-score inflation in stable pixels",
        },
    }
    save_json(meta, str(out_dir / "baseline_metadata.json"))

    return outputs


def get_baseline_statistics() -> Dict[str, Optional[Path]]:
    """
    Return paths to baseline statistic TIFs if they exist, else None.
    """
    stats_dir = BASELINE_DIR / "statistics"
    names = ["mean", "std", "median", "p05", "p10", "p25", "p75", "p90", "p95", "sens_slope"]
    return {
        name: (p if (p := stats_dir / f"baseline_{name}.tif").exists() else None)
        for name in names
    }


def get_annual_composites() -> List[str]:
    """Return sorted list of existing annual composite paths in the baseline directory."""
    return sorted(str(p) for p in BASELINE_DIR.glob("annual_ndvi_????.tif"))


def baseline_summary() -> Dict:
    """Return a structured summary of the current baseline status."""
    annual    = get_annual_composites()
    stats     = get_baseline_statistics()
    has_stats = all(v is not None for v in [stats.get("mean"), stats.get("std")])

    years = []
    for p in annual:
        stem = Path(p).stem
        for tok in stem.split("_"):
            if len(tok) == 4 and tok.isdigit():
                years.append(int(tok))
    return {
        "n_annual_composites": len(annual),
        "years_available":     sorted(years),
        "statistics_built":    has_stats,
        "z_score_ready":       len(annual) >= MIN_BASELINE_YEARS and has_stats,
        "annual_composite_paths": annual,
        "statistics_paths":    {k: str(v) for k, v in stats.items() if v},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Historical baseline builder for Hayedo de Montejo")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=list(range(2021, datetime.now(timezone.utc).year)),
        help="Years to process (default: 2021 to last year)",
    )
    parser.add_argument("--year", type=int, help="Single year to process")
    parser.add_argument("--status", action="store_true", help="Show baseline status and exit")
    parser.add_argument("--build-stats", action="store_true",
                        help="Build/rebuild statistics from existing annual composites")
    parser.add_argument("--config", default=str(CONFIG_DIR / "thresholds.yaml"))
    parser.add_argument("--aoi",    default=str(CONFIG_DIR / "aoi.geojson"))
    parser.add_argument("--max-scenes", type=int, default=N_BEST_SCENES,
                        help="Max scenes to download per year")
    args = parser.parse_args()

    if args.status:
        summary = baseline_summary()
        print(json.dumps(summary, indent=2))
        return

    if args.build_stats:
        annual = get_annual_composites()
        if len(annual) < MIN_BASELINE_YEARS:
            print(f"Need at least {MIN_BASELINE_YEARS} annual composites. Have: {len(annual)}")
            sys.exit(1)
        stats = build_baseline_statistics(annual)
        print(f"Built {len(stats)} statistic layers.")
        return

    years = [args.year] if args.year else args.years
    print(f"Processing {len(years)} year(s): {years}")

    for year in years:
        print(f"\n--- Year {year} ---")
        try:
            path = build_annual_composite_for_year(
                year,
                config_path=args.config,
                aoi_path=args.aoi,
                max_scenes=args.max_scenes,
            )
            if path:
                print(f"  Composite: {path}")
            else:
                print(f"  WARNING: No data produced for {year}")
        except AcquisitionError as exc:
            print(f"  ERROR: {exc}")
        except Exception as exc:
            logger.exception("Unexpected error for year %d", year)
            print(f"  FATAL: {exc}")

    # Try to build statistics after all years
    annual = get_annual_composites()
    if len(annual) >= MIN_BASELINE_YEARS:
        print(f"\nBuilding baseline statistics from {len(annual)} annual composites...")
        stats = build_baseline_statistics(annual)
        print(f"Statistics built: {list(stats.keys())}")
    else:
        needed = MIN_BASELINE_YEARS - len(annual)
        print(f"\nNeed {needed} more year(s) before building statistics.")


if __name__ == "__main__":
    main()
