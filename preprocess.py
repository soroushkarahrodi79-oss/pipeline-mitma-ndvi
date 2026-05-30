from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

from common import (
    PROCESSED_DIR, CONFIG_DIR,
    BAND_BLUE, BAND_RED, BAND_NIR,
    BAND_RED_EDGE, BAND_NIR_8A, BAND_SWIR1, BAND_SWIR2, BAND_MASK,
    PREPROCESSED_BAND_COUNT,
    VALID_SCL_CLASSES,
    load_geojson, setup_logging,
)

logger = setup_logging("preprocess")

_SCL_PATTERN = "*SCL_20m.jp2"
_BUFFER_M    = 500   # spatial buffer around the AOI (metres)


# ---------------------------------------------------------------------------
# AOI window helper
# ---------------------------------------------------------------------------

def _get_aoi_window(src: rasterio.DatasetReader, aoi_geojson: dict) -> Optional[rasterio.windows.Window]:
    """
    Return a rasterio Window clipped to the AOI bounding box + buffer.
    Transforms from WGS84 to the source raster CRS.
    Returns None if the transformation fails.
    """
    try:
        geom   = aoi_geojson["features"][0]["geometry"]
        coords = geom["coordinates"][0]
        lons   = [c[0] for c in coords]
        lats   = [c[1] for c in coords]

        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs,
            min(lons), min(lats), max(lons), max(lats),
        )
        left -= _BUFFER_M;  bottom -= _BUFFER_M
        right += _BUFFER_M; top    += _BUFFER_M

        # Clamp to raster extent
        left   = max(left,   src.bounds.left)
        bottom = max(bottom, src.bounds.bottom)
        right  = min(right,  src.bounds.right)
        top    = min(top,    src.bounds.top)

        win = window_from_bounds(left, bottom, right, top, src.transform)
        # Round to integer pixel boundaries
        win = win.round_lengths().round_offsets()
        logger.debug("AOI window: col_off=%d row_off=%d width=%d height=%d",
                     win.col_off, win.row_off, win.width, win.height)
        return win
    except Exception as exc:
        logger.warning("Could not compute AOI window: %s — reading full tile.", exc)
        return None


# ---------------------------------------------------------------------------
# Band I/O
# ---------------------------------------------------------------------------

def _find_band_file(scene_root: Path, pattern: str) -> Optional[Path]:
    for candidate in scene_root.rglob(pattern):
        return candidate
    return None


def _read_band_windowed(
    band_path: Path,
    window: Optional[rasterio.windows.Window] = None,
) -> Tuple[np.ndarray, dict]:
    with rasterio.open(band_path) as src:
        if window is not None:
            data    = src.read(1, window=window).astype("float32")
            profile = dict(src.profile)
            profile.update(
                height=window.height,
                width=window.width,
                transform=src.window_transform(window),
            )
        else:
            data    = src.read(1).astype("float32")
            profile = dict(src.profile)
    return data, profile


def _resample_to_match(
    source: np.ndarray,
    source_profile: dict,
    target_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    destination = np.full(
        (target_profile["height"], target_profile["width"]), np.nan, dtype="float32"
    )
    reproject(
        source,
        destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=resampling,
        src_nodata=0,
        dst_nodata=np.nan,
    )
    return destination


def normalize_reflectance(band: np.ndarray) -> np.ndarray:
    return np.where(band <= 0, np.nan, band / 10000.0)


# ---------------------------------------------------------------------------
# Scene preprocessing
# ---------------------------------------------------------------------------

def preprocess_scene(
    scene_path: str,
    output_dir: Optional[str] = None,
    aoi_path: Optional[str] = None,
) -> Path:
    """
    Preprocess a Sentinel-2 L2A .SAFE scene to an 8-band float32 GeoTIFF.
    Reads only the AOI window to avoid OOM on machines with limited RAM.
    """
    scene_root = Path(scene_path)
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AOI for windowed reading
    aoi_geojson = None
    if aoi_path is not None:
        aoi_geojson = load_geojson(aoi_path)
    else:
        default_aoi = CONFIG_DIR / "aoi.geojson"
        if default_aoi.exists():
            aoi_geojson = load_geojson(str(default_aoi))

    # --- locate required bands ---
    red_path  = _find_band_file(scene_root, "*B04_10m.jp2")
    nir_path  = _find_band_file(scene_root, "*B08_10m.jp2")
    scl_path  = _find_band_file(scene_root, _SCL_PATTERN)
    blue_path = _find_band_file(scene_root, "*B02_10m.jp2")
    if not red_path or not nir_path or not scl_path:
        raise FileNotFoundError(
            f"Required Sentinel-2 bands (B04, B08, SCL) not found in {scene_path}"
        )

    # Compute the AOI window from the reference 10m band
    window_10m = None
    if aoi_geojson:
        with rasterio.open(red_path) as src:
            window_10m = _get_aoi_window(src, aoi_geojson)

    logger.info(
        "Preprocessing %s — window: %s",
        scene_root.name,
        f"{window_10m.width}×{window_10m.height} px" if window_10m else "full tile",
    )

    # --- read 10m bands ---
    red,  profile  = _read_band_windowed(red_path,  window_10m)
    nir,  _        = _read_band_windowed(nir_path,  window_10m)
    blue           = normalize_reflectance(
        _read_band_windowed(blue_path, window_10m)[0]
    ) if blue_path else np.full_like(red, np.nan)

    # --- read SCL (20m) and resample to 10m window ---
    scl_raw, scl_p = _read_band_windowed(scl_path)  # read full 20m tile first
    # Compute 20m window from same AOI bbox
    if aoi_geojson:
        with rasterio.open(scl_path) as src:
            window_20m = _get_aoi_window(src, aoi_geojson)
        if window_20m:
            scl_raw, scl_p = _read_band_windowed(scl_path, window_20m)

    scl = _resample_to_match(scl_raw, scl_p, profile, resampling=Resampling.nearest)

    red = normalize_reflectance(red)
    nir = normalize_reflectance(nir)

    valid_mask = (
        np.isin(scl, VALID_SCL_CLASSES) & np.isfinite(red) & np.isfinite(nir)
    ).astype("float32")
    valid_mask = np.where(valid_mask == 1, 1.0, np.nan)

    # --- optional 20m bands → resample to 10m ---
    def _load_optional_20m(pattern: str) -> np.ndarray:
        path = _find_band_file(scene_root, pattern)
        if path is None:
            logger.warning("Band %s not found — writing NaN layer.", pattern)
            return np.full_like(red, np.nan)
        if aoi_geojson:
            with rasterio.open(path) as src:
                w20 = _get_aoi_window(src, aoi_geojson)
            arr, arr_p = _read_band_windowed(path, w20)
        else:
            arr, arr_p = _read_band_windowed(path)
        arr = _resample_to_match(arr, arr_p, profile, resampling=Resampling.bilinear)
        return normalize_reflectance(arr)

    red_edge = _load_optional_20m("*B05_20m.jp2")
    nir_8a   = _load_optional_20m("*B8A_20m.jp2")
    swir1    = _load_optional_20m("*B11_20m.jp2")
    swir2    = _load_optional_20m("*B12_20m.jp2")

    # --- write 8-band output ---
    output_path = output_dir / f"{scene_root.name}_preprocessed.tif"
    profile.update(
        driver="GTiff",
        count=PREPROCESSED_BAND_COUNT,
        dtype="float32",
        compress="lzw",
        nodata=np.nan,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(blue,       BAND_BLUE)
        dst.write(red,        BAND_RED)
        dst.write(nir,        BAND_NIR)
        dst.write(red_edge,   BAND_RED_EDGE)
        dst.write(nir_8a,     BAND_NIR_8A)
        dst.write(swir1,      BAND_SWIR1)
        dst.write(swir2,      BAND_SWIR2)
        dst.write(valid_mask, BAND_MASK)

    logger.info(
        "Saved preprocessed scene (%d bands, %dx%d px) to %s",
        PREPROCESSED_BAND_COUNT, profile["width"], profile["height"], output_path,
    )
    return output_path


def preprocess_batch(
    scene_paths: List[str],
    output_dir: Optional[str] = None,
    aoi_path: Optional[str] = None,
    workers: int = 2,
) -> List[Path]:
    processed: List[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(preprocess_scene, p, output_dir, aoi_path): p
            for p in scene_paths
        }
        for future in as_completed(future_to_path):
            scene = future_to_path[future]
            try:
                processed.append(future.result())
            except Exception as exc:
                logger.warning("Scene preprocessing failed for %s: %s", scene, exc)
    return processed


if __name__ == "__main__":
    logger.info("Module preprocess loaded. Use preprocess_batch for production processing.")
