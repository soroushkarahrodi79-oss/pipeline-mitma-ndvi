from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from common import (
    OUTPUTS_ANOMALIES_DIR,
    InsufficientDataError,
    PipelineError,
    setup_logging,
)

logger = setup_logging("anomaly_detection")


def load_raster(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile


def _nodata_to_nan(array: np.ndarray, profile: dict) -> np.ndarray:
    nodata = profile.get("nodata", -9999.0)
    if nodata is not None:
        return np.where(array == nodata, np.nan, array)
    return array


# ---------------------------------------------------------------------------
# Baseline builders
# ---------------------------------------------------------------------------

def compute_baseline_median(baseline_paths: List[str], out_path: Optional[str] = None) -> Path:
    """Single-layer median baseline for threshold-based detection (legacy method)."""
    if not baseline_paths:
        raise InsufficientDataError("Baseline requires at least one composite raster.")
    stacks, profile = [], None
    for path in baseline_paths:
        arr, profile = load_raster(path)
        stacks.append(_nodata_to_nan(arr, profile))
    baseline = np.nanmedian(np.stack(stacks, axis=0), axis=0)
    out_path = Path(out_path) if out_path else OUTPUTS_ANOMALIES_DIR / "baseline_ndvi_median.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(baseline), -9999.0, baseline).astype("float32"), 1)
    logger.info("Saved baseline median to %s", out_path)
    return out_path


def build_phenological_baseline(
    baseline_paths: List[str],
    out_mean_path: Optional[str] = None,
    out_std_path: Optional[str] = None,
    min_years: int = 3,
) -> Tuple[Path, Path]:
    """
    Compute pixel-wise mean and std from ≥min_years annual composites.
    Returns (mean_path, std_path) — both saved as single-band float32 GeoTIFFs.
    """
    if len(baseline_paths) < min_years:
        raise InsufficientDataError(
            f"Phenological baseline requires ≥{min_years} annual composites "
            f"(available: {len(baseline_paths)})."
        )
    stacks, profile = [], None
    for path in baseline_paths:
        arr, profile = load_raster(path)
        stacks.append(_nodata_to_nan(arr, profile))

    stack = np.stack(stacks, axis=0)
    mean = np.nanmean(stack, axis=0).astype("float32")
    std  = np.nanstd(stack,  axis=0).astype("float32")
    std  = np.where(std < 0.01, 0.01, std)     # floor: avoids division near zero

    base = OUTPUTS_ANOMALIES_DIR
    base.mkdir(parents=True, exist_ok=True)
    mean_path = Path(out_mean_path) if out_mean_path else base / "phenological_baseline_mean.tif"
    std_path  = Path(out_std_path)  if out_std_path  else base / "phenological_baseline_std.tif"

    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw")
    for arr, dst_path in [(mean, mean_path), (std, std_path)]:
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(np.where(np.isnan(arr), -9999.0, arr).astype("float32"), 1)

    logger.info(
        "Saved phenological baseline (mean=%s, std=%s) from %d composites.",
        mean_path, std_path, len(baseline_paths),
    )
    return mean_path, std_path


# ---------------------------------------------------------------------------
# Anomaly detectors
# ---------------------------------------------------------------------------

def detect_anomaly(
    current_path: str,
    baseline_path: str,
    threshold: float = -0.15,
) -> Dict[str, object]:
    """Threshold-based anomaly: flag pixels where current − baseline ≤ threshold."""
    current, profile  = load_raster(current_path)
    baseline, _       = load_raster(baseline_path)
    current  = _nodata_to_nan(current,  profile)
    baseline = _nodata_to_nan(baseline, profile)

    delta         = current - baseline
    anomaly_mask  = (delta <= threshold) & np.isfinite(delta)
    n_valid       = int(np.count_nonzero(np.isfinite(current)))
    n_anomaly     = int(np.count_nonzero(anomaly_mask))
    pixel_area_ha = abs(profile["transform"][0] * profile["transform"][4]) / 10_000.0

    mask_path = OUTPUTS_ANOMALIES_DIR / f"anomaly_mask_{Path(current_path).stem}.tif"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mask(anomaly_mask.astype("uint8"), profile, mask_path)

    return {
        "method": "threshold",
        "current_path": current_path,
        "baseline_path": baseline_path,
        "anomaly_threshold": threshold,
        "anomaly_pixels": n_anomaly,
        "anomaly_area_ha": float(n_anomaly * pixel_area_ha),
        "anomaly_fraction": float(n_anomaly / n_valid) if n_valid else 0.0,
        "delta_mean":   float(np.nanmean(delta))   if np.any(np.isfinite(delta)) else float("nan"),
        "delta_median": float(np.nanmedian(delta)) if np.any(np.isfinite(delta)) else float("nan"),
        "anomaly_mask_path": str(mask_path),
        "persistent": False,
    }


def detect_anomaly_zscore(
    current_path: str,
    baseline_mean_path: str,
    baseline_std_path: str,
    z_threshold: float = -2.0,
) -> Dict[str, object]:
    """
    Z-score anomaly: flag pixels where Z = (current − mean) / std ≤ z_threshold.
    A Z < −2.0 is statistically irrefutable evidence of sub-historical behaviour.
    """
    current, profile = load_raster(current_path)
    mean, _          = load_raster(baseline_mean_path)
    std, _           = load_raster(baseline_std_path)

    current = _nodata_to_nan(current, profile)
    mean    = _nodata_to_nan(mean,    profile)
    std     = _nodata_to_nan(std,     profile)

    z_score      = (current - mean) / std
    anomaly_mask = (z_score <= z_threshold) & np.isfinite(z_score)
    n_valid      = int(np.count_nonzero(np.isfinite(current)))
    n_anomaly    = int(np.count_nonzero(anomaly_mask))
    pixel_area_ha = abs(profile["transform"][0] * profile["transform"][4]) / 10_000.0

    mask_path = OUTPUTS_ANOMALIES_DIR / f"anomaly_mask_{Path(current_path).stem}.tif"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mask(anomaly_mask.astype("uint8"), profile, mask_path)

    return {
        "method": "zscore",
        "current_path": current_path,
        "z_threshold": z_threshold,
        "anomaly_pixels": n_anomaly,
        "anomaly_area_ha": float(n_anomaly * pixel_area_ha),
        "anomaly_fraction": float(n_anomaly / n_valid) if n_valid else 0.0,
        "z_mean":   float(np.nanmean(z_score))   if np.any(np.isfinite(z_score)) else float("nan"),
        "z_median": float(np.nanmedian(z_score)) if np.any(np.isfinite(z_score)) else float("nan"),
        "delta_mean": float(np.nanmean(current - mean)) if np.any(np.isfinite(current)) else float("nan"),
        "anomaly_mask_path": str(mask_path),
        "persistent": False,
    }


def detect_multivariate_anomaly(
    index_paths: Dict[str, str],
    contamination: float = 0.05,
    out_mask_suffix: str = "multivariate",
) -> Dict[str, object]:
    """
    Isolation Forest over a multi-index feature space.
    Detects combinations of unusual spectral signals even when no single
    index breaches its individual threshold.

    Requires scikit-learn (pip install scikit-learn).
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise PipelineError(
            "scikit-learn is required for isolation_forest detection. "
            "Install it with: pip install scikit-learn"
        ) from exc

    arrays, profile = {}, None
    for name, path in index_paths.items():
        arr, profile = load_raster(path)
        arrays[name] = _nodata_to_nan(arr, profile).flatten()

    if profile is None:
        raise PipelineError("No valid index rasters provided.")

    h, w = profile["height"], profile["width"]
    feature_matrix = np.column_stack(list(arrays.values()))        # (H*W, n_indices)
    valid_rows     = np.all(np.isfinite(feature_matrix), axis=1)   # pixels where ALL indices finite

    predictions = np.zeros(h * w, dtype="uint8")
    if valid_rows.sum() >= 10:
        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(feature_matrix[valid_rows])
        labels = clf.predict(feature_matrix[valid_rows])           # +1 normal, -1 anomaly
        predictions[valid_rows] = np.where(labels == -1, 1, 0)

    anomaly_raster = predictions.reshape(h, w)
    n_anomaly      = int(predictions.sum())
    n_valid        = int(valid_rows.sum())
    pixel_area_ha  = abs(profile["transform"][0] * profile["transform"][4]) / 10_000.0

    mask_path = OUTPUTS_ANOMALIES_DIR / f"anomaly_mask_{out_mask_suffix}.tif"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mask(anomaly_raster, profile, mask_path)

    logger.info(
        "Isolation Forest: %d anomaly pixels (%.1f%% of valid area) across %d indices.",
        n_anomaly, 100 * n_anomaly / n_valid if n_valid else 0, len(index_paths),
    )
    return {
        "method": "isolation_forest",
        "indices_used": list(index_paths.keys()),
        "contamination": contamination,
        "anomaly_pixels": n_anomaly,
        "anomaly_area_ha": float(n_anomaly * pixel_area_ha),
        "anomaly_fraction": float(n_anomaly / n_valid) if n_valid else 0.0,
        "anomaly_mask_path": str(mask_path),
        "persistent": False,
    }


# ---------------------------------------------------------------------------
# Persistence evaluation
# ---------------------------------------------------------------------------

def evaluate_persistent_anomalies(
    anomaly_masks: List[str],
    min_persistence: int = 2,
) -> Optional[Path]:
    if len(anomaly_masks) < min_persistence:
        logger.info("Not enough anomaly observations to evaluate persistence (%d < %d).",
                    len(anomaly_masks), min_persistence)
        return None
    stacks, profile = [], None
    for path in anomaly_masks:
        mask, profile = load_raster(path)
        stacks.append(mask == 1)
    combined   = np.sum(np.stack(stacks, axis=0), axis=0)
    persistent = (combined >= min_persistence).astype("uint8")

    persistent_path = OUTPUTS_ANOMALIES_DIR / "persistent_anomaly_mask.tif"
    persistent_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(persistent_path, "w", **profile) as dst:
        dst.write(persistent, 1)
    logger.info("Saved persistent anomaly mask to %s", persistent_path)
    return persistent_path


# ---------------------------------------------------------------------------
# Unified report entry point
# ---------------------------------------------------------------------------

def anomaly_report(
    current_path: str,
    baseline_path: str,
    previous_masks: Optional[List[str]] = None,
    threshold: float = -0.15,
    min_persistence: int = 2,
    method: str = "threshold",
    baseline_mean_path: Optional[str] = None,
    baseline_std_path: Optional[str] = None,
    z_threshold: float = -2.0,
) -> Dict[str, object]:
    """
    Unified anomaly detection entry point.

    method="threshold"  — classic ΔNDVI fixed-threshold (always available)
    method="zscore"     — phenological Z-score (requires baseline_mean_path + baseline_std_path)
    """
    if method == "zscore" and baseline_mean_path and baseline_std_path:
        report = detect_anomaly_zscore(
            current_path,
            baseline_mean_path,
            baseline_std_path,
            z_threshold=z_threshold,
        )
    else:
        if method == "zscore":
            logger.warning(
                "Z-score method requested but phenological baseline not available; "
                "falling back to threshold-based detection."
            )
        report = detect_anomaly(current_path, baseline_path, threshold=threshold)

    if previous_masks:
        persistent_mask = evaluate_persistent_anomalies(
            previous_masks + [str(report["anomaly_mask_path"])],
            min_persistence=min_persistence,
        )
        report["persistent_anomaly_mask"] = str(persistent_mask) if persistent_mask else None
        report["persistent"] = persistent_mask is not None

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_mask(mask_array: np.ndarray, profile: dict, out_path: Path) -> None:
    p = dict(profile)
    p.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(mask_array.astype("uint8"), 1)


if __name__ == "__main__":
    logger.info("Module anomaly_detection loaded.")
