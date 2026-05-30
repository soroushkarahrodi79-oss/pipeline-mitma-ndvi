from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio

from common import (
    PROCESSED_DIR,
    BAND_BLUE, BAND_RED, BAND_NIR,
    BAND_RED_EDGE, BAND_NIR_8A, BAND_SWIR1, BAND_SWIR2, BAND_MASK,
    setup_logging,
)

logger = setup_logging("indices")


# ---------------------------------------------------------------------------
# Per-pixel index computations
# ---------------------------------------------------------------------------

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    return np.clip(np.where(denom == 0, np.nan, (nir - red) / denom), -1.0, 1.0)


def compute_evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, np.nan, 2.5 * ((nir - red) / denom))


def compute_ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index — B08 vs B11."""
    denom = nir + swir1
    return np.clip(np.where(denom == 0, np.nan, (nir - swir1) / denom), -1.0, 1.0)


def compute_ndre(red_edge: np.ndarray, nir_8a: np.ndarray) -> np.ndarray:
    """Red-Edge NDVI — B05 vs B8A.  Sensitive to chlorophyll stress 2–3 weeks before NDVI."""
    denom = nir_8a + red_edge
    return np.clip(np.where(denom == 0, np.nan, (nir_8a - red_edge) / denom), -1.0, 1.0)


def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio — B08 vs B12.  Disturbance and post-fire recovery."""
    denom = nir + swir2
    return np.clip(np.where(denom == 0, np.nan, (nir - swir2) / denom), -1.0, 1.0)


def compute_msavi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Modified SAVI — reduced soil-brightness influence; useful at forest edges and gaps."""
    inner = (2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)
    return np.where(inner < 0, np.nan, (2.0 * nir + 1.0 - np.sqrt(np.maximum(inner, 0.0))) / 2.0)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_index_raster(
    index_array: np.ndarray,
    profile: Dict,
    out_path: str,
    nodata: float = -9999.0,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isnan(index_array), nodata, index_array).astype("float32")
    profile = dict(profile)
    profile.update(count=1, dtype="float32", nodata=nodata, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    return out_path


# ---------------------------------------------------------------------------
# Scene-level computation
# ---------------------------------------------------------------------------

def compute_scene_indices(
    scene_path: str,
    indices: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Compute requested spectral indices for a preprocessed 8-band scene.

    Supported indices: NDVI, EVI, NDMI, NDRE, NBR, MSAVI2
    Defaults to all six when indices=None.
    """
    if indices is None:
        indices = ["NDVI", "EVI", "NDMI", "NDRE", "NBR", "MSAVI2"]
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    scene_file = Path(scene_path)

    with rasterio.open(scene_file) as src:
        blue      = src.read(BAND_BLUE).astype("float32")
        red       = src.read(BAND_RED).astype("float32")
        nir       = src.read(BAND_NIR).astype("float32")
        red_edge  = src.read(BAND_RED_EDGE).astype("float32")
        nir_8a    = src.read(BAND_NIR_8A).astype("float32")
        swir1     = src.read(BAND_SWIR1).astype("float32")
        swir2     = src.read(BAND_SWIR2).astype("float32")
        mask      = src.read(BAND_MASK).astype("float32")
        profile   = src.profile

    def _masked(arr: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(mask), np.nan, arr)

    output_paths: Dict[str, Path] = {}
    stem = scene_file.stem

    _index_map = {
        "NDVI":   lambda: _masked(compute_ndvi(red, nir)),
        "EVI":    lambda: _masked(compute_evi(red, nir, blue)),
        "NDMI":   lambda: _masked(compute_ndmi(nir, swir1)),
        "NDRE":   lambda: _masked(compute_ndre(red_edge, nir_8a)),
        "NBR":    lambda: _masked(compute_nbr(nir, swir2)),
        "MSAVI2": lambda: _masked(compute_msavi2(red, nir)),
    }

    for name in indices:
        if name not in _index_map:
            logger.warning("Unknown index '%s' — skipping.", name)
            continue
        result = _index_map[name]()
        out_path = output_dir / f"{stem}_{name}.tif"
        output_paths[name] = write_index_raster(result, profile, str(out_path))
        logger.info("Computed %s for %s", name, scene_file.name)

    return output_paths


def compute_batch_indices(
    scene_paths: List[str],
    indices: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    workers: int = 4,
) -> List[Dict[str, Path]]:
    results: List[Dict[str, Path]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(compute_scene_indices, p, indices, output_dir): p
            for p in scene_paths
        }
        for future in as_completed(future_to_path):
            scene = future_to_path[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.warning("Failed to compute indices for %s: %s", scene, exc)
    return results


if __name__ == "__main__":
    logger.info("Module indices loaded. Use compute_batch_indices for batch index generation.")
