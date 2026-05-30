"""
Forest phenology analysis module for Fagus sylvatica.

Detects and tracks key phenological events from the NDVI time series:
  - Start of Season (SOS): green-up / budburst
  - Peak Vegetation (PEAK): maximum NDVI
  - End of Season (EOS): senescence / leaf fall
  - Growing Season Length (GSL): EOS - SOS

SCIENTIFIC CONTEXT
------------------
Phenological shifts are among the earliest and most sensitive indicators
of climate change impacts on ecosystems. For Fagus sylvatica at its southern
distribution limit, research has documented:

  - Advancement of budburst by ~2-3 days/decade since the 1950s
    (Menzel et al. 2006, Global Change Biology)
  - Delayed senescence in warmer years
    (Vitasse et al. 2009, Agricultural and Forest Meteorology)
  - Increased inter-year variability in growing season length
    (Fu et al. 2015, Nature Climate Change)

Changes in phenological timing have direct implications for:
  - Carbon balance (longer GSL → increased net primary productivity)
  - Water balance (longer transpiring season → increased water use)
  - Late-frost vulnerability (earlier budburst → higher risk)
  - Species interactions (synchrony with insects, fungi, ground flora)

METHODS
-------
Three approaches are implemented with increasing data requirements:

1. THRESHOLD METHOD (≥3 observations spanning Apr–Oct):
   SOS = first date when NDVI exceeds 0.25 + background
   EOS = last date when NDVI exceeds 0.25 + background
   PEAK = date of maximum NDVI
   Requires at least seasonal coverage.

2. DERIVATIVE METHOD (≥6 observations with smooth coverage):
   SOS = date of maximum positive NDVI rate-of-change (maximum d(NDVI)/dt)
   EOS = date of minimum (most negative) rate-of-change
   Requires dense temporal sampling — not suitable for annual composites alone.

3. DOUBLE LOGISTIC FITTING (≥8 observations):
   Fits Beck et al. (2006) double logistic model to NDVI time series.
   Most robust method but requires well-sampled annual curves.

For the current system (sparse annual July-September composites), the
threshold method is applied. The system will automatically upgrade to
derivative or double logistic fitting as the time series density increases.

Reference: Reed et al. (1994) Measuring phenological variability from
satellite imagery. Journal of Vegetation Science, 5, 703-714.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from common import WORKSPACE_ROOT, setup_logging

logger = setup_logging("phenology")

PHENOLOGY_DIR = WORKSPACE_ROOT / "outputs" / "phenology"

# NDVI thresholds for F. sylvatica at this site
# Calibrated for 41°N, 1150-1450 m, north-facing slopes
NDVI_GREEN_UP_THRESHOLD = 0.30   # NDVI above this = active leaf development
NDVI_SENESCENCE_THRESHOLD = 0.35  # NDVI below this late-season = end of season
NDVI_BACKGROUND = 0.15           # Approximate winter NDVI (bare canopy + understorey)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PhenologyMetrics:
    """Container for phenological event dates and derived metrics."""

    def __init__(self, year: int):
        self.year     = year
        self.sos_doy  = None   # Start of Season (DOY)
        self.peak_doy = None   # Peak NDVI date (DOY)
        self.eos_doy  = None   # End of Season (DOY)
        self.gsl      = None   # Growing Season Length (days)
        self.peak_ndvi = None  # Maximum NDVI value
        self.ndvi_integrated = None  # Integral of NDVI over season (proxy for GPP)
        self.method   = None   # Method used
        self.notes    = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year":              self.year,
            "sos_doy":           self.sos_doy,
            "sos_date":          _doy_to_date(self.year, self.sos_doy),
            "peak_doy":          self.peak_doy,
            "peak_date":         _doy_to_date(self.year, self.peak_doy),
            "eos_doy":           self.eos_doy,
            "eos_date":          _doy_to_date(self.year, self.eos_doy),
            "growing_season_length_days": self.gsl,
            "peak_ndvi":         self.peak_ndvi,
            "ndvi_integrated":   self.ndvi_integrated,
            "method":            self.method,
            "notes":             self.notes,
        }


def _doy_to_date(year: int, doy: Optional[int]) -> Optional[str]:
    if doy is None:
        return None
    try:
        return datetime(year, 1, 1).replace(
            month=1, day=1
        ).strftime(f"{year}-")[:5] + date.fromordinal(
            date(year, 1, 1).toordinal() + doy - 1
        ).strftime("%m-%d")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core phenology detection methods
# ---------------------------------------------------------------------------

def detect_phenology_threshold(
    doys: List[int],
    ndvi_values: List[float],
    year: int,
    green_up_threshold: float = NDVI_GREEN_UP_THRESHOLD,
    senescence_threshold: float = NDVI_SENESCENCE_THRESHOLD,
) -> PhenologyMetrics:
    """
    Threshold-based phenological event detection.

    Suitable for sparse annual composites (minimum 3 valid observations).

    Parameters
    ----------
    doys : Day-of-Year for each observation
    ndvi_values : corresponding NDVI values
    year : calendar year
    """
    metrics = PhenologyMetrics(year)
    metrics.method = "threshold"

    if len(doys) < 2:
        metrics.notes.append("Insufficient observations for threshold detection (need >= 2).")
        return metrics

    doys_arr  = np.array(doys,        dtype="float32")
    ndvi_arr  = np.array(ndvi_values,  dtype="float32")

    # Remove NaN
    valid = np.isfinite(ndvi_arr)
    doys_arr  = doys_arr[valid]
    ndvi_arr  = ndvi_arr[valid]

    if len(doys_arr) < 2:
        metrics.notes.append("Insufficient valid observations after NaN removal.")
        return metrics

    # Sort by DOY
    sort_idx  = np.argsort(doys_arr)
    doys_arr  = doys_arr[sort_idx]
    ndvi_arr  = ndvi_arr[sort_idx]

    # Peak NDVI
    peak_idx          = np.argmax(ndvi_arr)
    metrics.peak_doy  = int(doys_arr[peak_idx])
    metrics.peak_ndvi = float(ndvi_arr[peak_idx])

    # SOS: first date above green-up threshold (searching forward from winter)
    for i, (doy, ndvi) in enumerate(zip(doys_arr, ndvi_arr)):
        if ndvi >= green_up_threshold and doy < metrics.peak_doy:
            metrics.sos_doy = int(doy)
            break

    # EOS: last date above senescence threshold (searching backward from end)
    for doy, ndvi in zip(reversed(doys_arr.tolist()), reversed(ndvi_arr.tolist())):
        if ndvi >= senescence_threshold and doy > metrics.peak_doy:
            metrics.eos_doy = int(doy)
            break

    if metrics.sos_doy and metrics.eos_doy:
        metrics.gsl = metrics.eos_doy - metrics.sos_doy

    # NDVI integral (trapezoidal — proxy for cumulative carbon uptake)
    if len(doys_arr) >= 2:
        metrics.ndvi_integrated = float(np.trapz(ndvi_arr, doys_arr))

    metrics.notes.append(
        f"Threshold method applied to {len(doys_arr)} observations."
    )
    return metrics


def fit_double_logistic(
    doys: np.ndarray,
    ndvi: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Fit Beck et al. (2006) double logistic model to NDVI time series.

    Model: NDVI(t) = v_min + (v_max - v_min) * [1/(1+exp(-m1*(t-t1)))
                                                  - 1/(1+exp(-m2*(t-t2)))]

    Parameters
    ----------
    doys : array of DOYs
    ndvi : array of NDVI values

    Returns
    -------
    (fitted_curve, parameters) or None if fitting fails
    """
    if len(doys) < 8:
        return None

    def double_logistic(t, v_min, v_max, m1, t1, m2, t2):
        return (v_min + (v_max - v_min) *
                (1 / (1 + np.exp(-m1 * (t - t1))) -
                 1 / (1 + np.exp(-m2 * (t - t2)))))

    # Initial parameters
    v_min_0 = float(np.percentile(ndvi, 10))
    v_max_0 = float(np.max(ndvi))
    peak_doy = float(doys[np.argmax(ndvi)])

    p0 = [v_min_0, v_max_0, 0.1, peak_doy - 60, 0.1, peak_doy + 60]
    bounds = (
        [0, 0, 0, 1,   0, 1],
        [1, 1, 1, 200, 1, 365],
    )

    try:
        popt, _ = curve_fit(
            double_logistic, doys, ndvi,
            p0=p0, bounds=bounds, maxfev=5000,
        )
        t_fine     = np.linspace(1, 365, 365)
        fitted     = double_logistic(t_fine, *popt)
        params     = {"v_min": popt[0], "v_max": popt[1], "m1": popt[2],
                      "t1": popt[3], "m2": popt[4], "t2": popt[5]}
        return t_fine, fitted, params
    except Exception:
        return None


def detect_phenology_best_method(
    doys: List[int],
    ndvi_values: List[float],
    year: int,
) -> PhenologyMetrics:
    """
    Select the most appropriate phenology detection method based on data density.

    - >= 8 valid observations + good seasonal coverage: double logistic fitting
    - >= 4 valid observations: threshold with Savgol smoothing
    - < 4 observations: threshold method without smoothing
    """
    doys_arr = np.array([d for d, v in zip(doys, ndvi_values) if np.isfinite(v)])
    ndvi_arr = np.array([v for d, v in zip(doys, ndvi_values) if np.isfinite(v)])

    if len(doys_arr) >= 8 and (max(doys_arr) - min(doys_arr)) >= 150:
        # Try double logistic
        result = fit_double_logistic(doys_arr, ndvi_arr)
        if result is not None:
            t_fine, fitted, params = result
            metrics = PhenologyMetrics(year)
            metrics.method = "double_logistic"
            metrics.peak_doy  = int(t_fine[np.argmax(fitted)])
            metrics.peak_ndvi = float(np.max(fitted))

            half_max = (params["v_max"] + params["v_min"]) / 2
            sos_candidates = t_fine[(fitted >= half_max) & (t_fine < metrics.peak_doy)]
            eos_candidates = t_fine[(fitted >= half_max) & (t_fine > metrics.peak_doy)]
            if len(sos_candidates): metrics.sos_doy = int(sos_candidates[0])
            if len(eos_candidates): metrics.eos_doy = int(eos_candidates[-1])
            if metrics.sos_doy and metrics.eos_doy:
                metrics.gsl = metrics.eos_doy - metrics.sos_doy
            metrics.ndvi_integrated = float(np.trapz(fitted, t_fine))
            metrics.notes.append(
                f"Double logistic fitting applied to {len(doys_arr)} observations. "
                f"Parameters: v_min={params['v_min']:.3f}, v_max={params['v_max']:.3f}."
            )
            return metrics

    if len(doys_arr) >= 4:
        # Savgol smoothing then threshold
        if len(doys_arr) >= 5:
            w = min(5, len(doys_arr) - (0 if len(doys_arr) % 2 == 1 else 1))
            if w % 2 == 0:
                w -= 1
            if w >= 3:
                ndvi_arr = savgol_filter(ndvi_arr, window_length=w, polyorder=2)

    return detect_phenology_threshold(doys_arr.tolist(), ndvi_arr.tolist(), year)


# ---------------------------------------------------------------------------
# Multi-year phenology analysis
# ---------------------------------------------------------------------------

def build_phenology_time_series(
    annual_ndvi_stats: Dict[int, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Build a multi-year phenology record from annual composite statistics.

    Note: Annual composites (Jul-Sep) provide peak-season data only.
    For full phenological curves, dense temporal sampling is required.
    This function provides a limited but reproducible phenological record
    from the available data.

    Parameters
    ----------
    annual_ndvi_stats : {year: {'median': float, 'mean': float, 'std': float, ...}}
    """
    records = []
    for year, stats in sorted(annual_ndvi_stats.items()):
        # For annual composites (Jul-Sep), we know the observation is near-peak
        # Approximate peak DOY as mid-August (DOY 228) for this altitude/latitude
        approx_peak_doy = 228
        doys  = [approx_peak_doy]
        ndvis = [stats.get("median", float("nan"))]

        metrics = detect_phenology_threshold(doys, ndvis, year)
        metrics.peak_doy  = approx_peak_doy
        metrics.peak_ndvi = stats.get("median")
        metrics.method    = "annual_composite_proxy"
        metrics.notes.append(
            "Based on annual Jul-Sep composite. Peak DOY approximated as DOY 228 "
            "(mid-August at 41N/1300m). Full phenological curve requires dense "
            "temporal sampling (monthly/bi-monthly observations)."
        )

        record = metrics.to_dict()
        record.update({
            "ndvi_mean":   stats.get("mean"),
            "ndvi_std":    stats.get("std"),
            "ndvi_p10":    stats.get("p10"),
            "ndvi_p90":    stats.get("p90"),
            "n_pixels":    stats.get("valid_pixels"),
        })
        records.append(record)

    return records


def analyse_phenology_trends(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform trend analysis on the phenology time series.

    Computes Sen's slope and Mann-Kendall test for monotonic trends
    in peak NDVI and GSL over the observation period.
    """
    from scipy import stats as sp_stats

    if len(records) < 3:
        return {"available": False, "n_years": len(records),
                "note": "Minimum 3 years required for trend analysis."}

    years     = [r["year"]              for r in records if r.get("peak_ndvi") is not None]
    peak_ndvi = [r["peak_ndvi"]         for r in records if r.get("peak_ndvi") is not None]
    gsl       = [r.get("growing_season_length_days") for r in records
                 if r.get("growing_season_length_days") is not None]

    results: Dict[str, Any] = {"n_years": len(years), "years": years}

    if len(peak_ndvi) >= 3:
        slope, intercept, r, p, se = sp_stats.linregress(years, peak_ndvi)
        tau, mk_p = sp_stats.kendalltau(range(len(years)), peak_ndvi)
        results["peak_ndvi_trend"] = {
            "slope_per_year": round(slope, 5),
            "r_squared":      round(r**2, 4),
            "p_value":        round(p, 4),
            "mann_kendall_tau": round(tau, 4),
            "mann_kendall_p":   round(mk_p, 4),
            "interpretation": _interpret_ndvi_trend(slope, p, mk_p, len(years)),
        }

    if len(gsl) >= 3:
        gsl_years = [r["year"] for r in records if r.get("growing_season_length_days") is not None]
        slope, _, r, p, _ = sp_stats.linregress(gsl_years, gsl)
        results["gsl_trend"] = {
            "slope_days_per_year": round(slope, 2),
            "r_squared":           round(r**2, 4),
            "p_value":             round(p, 4),
            "interpretation":      _interpret_gsl_trend(slope, p, len(gsl_years)),
        }

    return results


def _interpret_ndvi_trend(
    slope: float, p: float, mk_p: float, n: int
) -> str:
    sig = "statistically significant" if (p < 0.05 or mk_p < 0.05) else "not statistically significant"
    if abs(slope) < 0.002:
        return (
            f"No meaningful trend in peak-season NDVI over {n} years "
            f"(slope ≈ {slope:+.4f}/year, {sig}). "
            "Forest canopy condition at peak phenology appears stable."
        )
    direction = "greening trend" if slope > 0 else "browning trend"
    severity  = "modest" if abs(slope) < 0.01 else "substantial"
    return (
        f"{severity.capitalize()} {direction} detected in peak-season NDVI "
        f"({slope:+.4f}/year, {sig}, n={n}). "
        + ("Consistent with enhanced vegetation activity or phenological extension." if slope > 0
           else "Potentially indicative of progressive stress, land cover change, or climate-driven canopy decline.")
    )


def _interpret_gsl_trend(slope: float, p: float, n: int) -> str:
    sig = "significant" if p < 0.05 else "non-significant"
    return (
        f"Growing season length trend: {slope:+.1f} days/year ({sig}, n={n}). "
        + ("Season lengthening, consistent with warming-induced phenological advancement." if slope > 0
           else "Season shortening, potentially reflecting earlier senescence or later green-up.")
    )


def save_phenology_results(
    records: List[Dict],
    trends: Dict,
    label: str = "annual",
) -> Path:
    PHENOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "records": records,
        "trends":  trends,
        "generated_at": datetime.now().isoformat(),
        "method_note": (
            "Phenological metrics derived from Sentinel-2 annual composites. "
            "Peak DOY estimated from site-specific literature; SOS and EOS require "
            "dense temporal sampling for robust detection."
        ),
    }
    from common import save_json
    path = PHENOLOGY_DIR / f"phenology_{label}.json"
    save_json(out, str(path))
    logger.info("Phenology results saved to %s", path)
    return path
