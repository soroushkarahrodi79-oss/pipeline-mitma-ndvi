"""
AEMET OpenData climate integration module.

Integrates monthly meteorological data from AEMET (Agencia Estatal de Meteorología)
for climate-vegetation relationship analysis at the Hayedo de Montejo.

SCIENTIFIC RATIONALE
--------------------
Climate-vegetation relationships are essential for distinguishing:
  1. Stress signals driven by climate (drought, temperature anomalies) from
  2. Structural changes (disturbance, succession, management)

For Fagus sylvatica at the southern distribution limit, the most critical
climate drivers are:
  - Summer water deficit (June–August precipitation vs PET)
  - Spring temperature (influences budburst timing)
  - Late frost events (causes crown damage affecting NDVI in May–June)
  - Cumulative drought severity (SPEI, SPI indices)

STATION SELECTION
-----------------
Primary station: Montejo de la Sierra (AEMET code 2864A or nearest available)
Secondary station: Somosierra (2462, altitude ~1437 m — similar altitudinal range)
Tertiary station: Buitrago del Lozoya (2853, lower altitude — for comparison)

Note: Stations near the hayedo may have data gaps. The module falls back to
the nearest available station with acceptable data coverage.

AEMET OpenData API
------------------
Registration: https://opendata.aemet.es/centrodedescargas/altaUsuario
API endpoint: https://opendata.aemet.es/opendata/api
Rate limit: 10 req/min for free tier

INDICES COMPUTED
----------------
  - SPI (Standardised Precipitation Index) — drought duration/severity
    Reference: McKee et al. 1993
  - Temperature anomaly vs. 1991-2020 climatological normal
    Reference: WMO standard normal period
  - Potential stress coupling index: combines NDVI anomaly with precipitation anomaly
"""

import json
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy import stats as sp_stats
from scipy.signal import savgol_filter

from common import WORKSPACE_ROOT, CONFIG_DIR, save_json, setup_logging

logger = setup_logging("climate")

CLIMATE_DIR  = WORKSPACE_ROOT / "outputs" / "climate"
AEMET_BASE   = "https://opendata.aemet.es/opendata/api"

# AEMET station codes near Sierra del Rincón
# Station IDs may need verification at https://opendata.aemet.es
STATIONS = {
    "montejo_de_la_sierra": {
        "code":      "2864A",
        "name":      "Montejo de la Sierra",
        "altitude":  1125,
        "lat":       41.217,
        "lon":       -3.517,
        "priority":  1,
    },
    "somosierra": {
        "code":      "2462",
        "name":      "Somosierra",
        "altitude":  1437,
        "lat":       41.108,
        "lon":       -3.585,
        "priority":  2,
    },
    "buitrago": {
        "code":      "2853",
        "name":      "Buitrago del Lozoya",
        "altitude":  961,
        "lat":       40.993,
        "lon":       -3.651,
        "priority":  3,
    },
}

# WMO 1991–2020 climatological normals (approximate, for the region)
# Source: AEMET climatological normals — these should be replaced with
# actual station data when available
CLIMATOLOGICAL_NORMALS_MONTHLY = {
    # month: (mean_temp_C, mean_precip_mm) — approximate for the hayedo area
    1:  (-0.5,  70),
    2:  (0.5,   60),
    3:  (3.0,   60),
    4:  (5.5,   70),
    5:  (9.5,   70),
    6:  (14.0,  40),
    7:  (17.5,  20),
    8:  (17.0,  25),
    9:  (13.5,  45),
    10: (8.5,   75),
    11: (3.5,   90),
    12: (0.5,   85),
}


# ---------------------------------------------------------------------------
# AEMET API client
# ---------------------------------------------------------------------------

class AEMETClient:
    """Thin client for the AEMET OpenData REST API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = requests.Session()
        self._session.params = {"api_key": api_key}

    def _get(self, endpoint: str, max_retries: int = 3) -> Optional[Dict]:
        url = f"{AEMET_BASE}{endpoint}"
        for attempt in range(max_retries):
            try:
                resp = self._session.get(url, timeout=20)
                if resp.status_code == 429:
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                meta = resp.json()
                if meta.get("estado") != 200:
                    logger.warning("AEMET API error: %s", meta.get("descripcion", ""))
                    return None
                data_url = meta.get("datos")
                if not data_url:
                    return None
                data_resp = self._session.get(data_url, timeout=30)
                data_resp.raise_for_status()
                return data_resp.json()
            except Exception as exc:
                logger.warning("AEMET request attempt %d failed: %s", attempt + 1, exc)
                time.sleep(5)
        return None

    def get_monthly_station_data(
        self,
        station_code: str,
        year_start: int,
        year_end: int,
    ) -> Optional[List[Dict]]:
        """
        Fetch monthly climate summaries for a station over a year range.
        AEMET endpoint: /api/valores/climatologicos/mensualesanuales/datos/
        """
        endpoint = (
            f"/valores/climatologicos/mensualesanuales/datos/aniosanio/"
            f"{year_start}/{year_end}/estacion/{station_code}"
        )
        return self._get(endpoint)

    def get_daily_station_data(
        self,
        station_code: str,
        date_from: str,
        date_to: str,
    ) -> Optional[List[Dict]]:
        """
        Fetch daily climate observations.
        date_from/date_to: YYYY-MM-DDTHH:MM:SSUTC
        """
        endpoint = (
            f"/observacion/convencional/datos/estacion/{station_code}/"
            f"datos/fechaini/{date_from}/fechafin/{date_to}"
        )
        return self._get(endpoint)


# ---------------------------------------------------------------------------
# Data extraction and cleaning
# ---------------------------------------------------------------------------

def _safe_float(value: Any, scale: float = 1.0) -> Optional[float]:
    """Convert AEMET string values to float, handling commas and missing codes."""
    if value is None or str(value).strip() in ("", "Ip", "-", "---", "nan"):
        return None
    try:
        return float(str(value).replace(",", ".")) * scale
    except (ValueError, TypeError):
        return None


def parse_monthly_records(raw: List[Dict]) -> List[Dict]:
    """
    Parse AEMET monthly records into a clean list of dicts.

    Key fields extracted:
      - year, month
      - tm_mes: mean monthly temperature (°C)
      - p_mes: monthly precipitation (mm)
      - ta_max: absolute monthly maximum temperature (°C)
      - ta_min: absolute monthly minimum temperature (°C)
    """
    records = []
    for r in raw:
        year  = _safe_float(r.get("anio"))
        month = _safe_float(r.get("mes"))
        if year is None or month is None:
            continue
        records.append({
            "year":   int(year),
            "month":  int(month),
            "date":   f"{int(year)}-{int(month):02d}",
            "temp_mean":   _safe_float(r.get("tm_mes")),
            "temp_max":    _safe_float(r.get("ta_max")),
            "temp_min":    _safe_float(r.get("ta_min")),
            "precip_mm":   _safe_float(r.get("p_mes")),
            "n_precip_days": _safe_float(r.get("np_001")),
            "station_code":  r.get("indicativo", ""),
        })
    records.sort(key=lambda x: (x["year"], x["month"]))
    return records


# ---------------------------------------------------------------------------
# Climate index computation
# ---------------------------------------------------------------------------

def compute_spi(
    precip_series: List[Optional[float]],
    months: List[Tuple[int, int]],
    timescale: int = 3,
) -> List[Optional[float]]:
    """
    Compute Standardised Precipitation Index (SPI) at a given timescale.

    SPI is computed by fitting a Gamma distribution to the precipitation
    accumulation over [timescale] months, then transforming to standard normal.

    Parameters
    ----------
    precip_series : list of monthly precipitation values (mm), may contain None
    months : list of (year, month) tuples corresponding to precip_series
    timescale : accumulation period in months (3 = SPI-3, 12 = SPI-12)

    Returns
    -------
    List of SPI values (None where insufficient data)

    Reference: McKee et al. (1993) The relationship of drought frequency
    and duration to time scales. 8th Conference on Applied Climatology.
    """
    n = len(precip_series)
    spi_values = [None] * n

    for i in range(timescale - 1, n):
        window = precip_series[i - timescale + 1: i + 1]
        if any(v is None for v in window):
            continue
        acc = sum(window)

        # Collect all historical accumulations for same month-of-year
        target_month = months[i][1]
        historical = []
        for j in range(timescale - 1, n):
            if months[j][1] == target_month:
                hw = precip_series[j - timescale + 1: j + 1]
                if all(v is not None for v in hw):
                    historical.append(sum(hw))

        if len(historical) < 10:
            continue  # insufficient historical record

        # Fit Gamma distribution and compute SPI
        try:
            hist_arr = np.array(historical, dtype="float64") + 0.01  # avoid zeros
            shape_p, loc_p, scale_p = sp_stats.gamma.fit(hist_arr, floc=0)
            cdf = sp_stats.gamma.cdf(max(acc + 0.01, 0.01), shape_p, loc=loc_p, scale=scale_p)
            cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
            spi_values[i] = float(sp_stats.norm.ppf(cdf))
        except Exception:
            pass

    return spi_values


def compute_temperature_anomaly(
    temp_series: List[Optional[float]],
    months: List[Tuple[int, int]],
    normal_period_years: Tuple[int, int] = (1991, 2020),
) -> List[Optional[float]]:
    """
    Compute monthly temperature anomaly vs. climatological normal period.

    Parameters
    ----------
    temp_series : list of monthly mean temperatures (°C)
    months : list of (year, month) tuples
    normal_period_years : (start_year, end_year) for computing the normal

    Returns
    -------
    List of anomaly values (°C), positive = warmer than normal
    """
    # Compute monthly normals from the reference period
    normals = {}
    for i, (yr, mo) in enumerate(months):
        if temp_series[i] is None:
            continue
        if normal_period_years[0] <= yr <= normal_period_years[1]:
            normals.setdefault(mo, []).append(temp_series[i])

    monthly_normal = {mo: np.mean(vals) for mo, vals in normals.items()}

    anomalies = []
    for i, (yr, mo) in enumerate(months):
        if temp_series[i] is None or mo not in monthly_normal:
            anomalies.append(None)
        else:
            anomalies.append(round(temp_series[i] - monthly_normal[mo], 2))

    return anomalies


def compute_ndvi_climate_correlation(
    ndvi_values: List[float],
    ndvi_dates: List[date],
    climate_records: List[Dict],
    climate_variable: str = "precip_mm",
    lag_months: int = 0,
) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlations between NDVI observations
    and a climate variable, with optional lag.

    Parameters
    ----------
    ndvi_values : NDVI observations (forest-masked median or mean)
    ndvi_dates : observation dates corresponding to NDVI values
    climate_records : parsed monthly climate records
    climate_variable : 'precip_mm', 'temp_mean', 'spi_3', etc.
    lag_months : number of months to lag the climate variable (0 = synchronous)

    Returns
    -------
    dict with correlation, p-value, n_pairs, and interpretation
    """
    # Build climate lookup by (year, month)
    clim_lookup = {
        (r["year"], r["month"]): r.get(climate_variable)
        for r in climate_records
        if r.get(climate_variable) is not None
    }

    ndvi_matched, clim_matched = [], []
    for ndvi_val, obs_date in zip(ndvi_values, ndvi_dates):
        yr, mo = obs_date.year, obs_date.month
        # Apply lag
        lag_mo = mo - lag_months
        lag_yr = yr
        while lag_mo < 1:
            lag_mo += 12
            lag_yr -= 1
        clim_val = clim_lookup.get((lag_yr, lag_mo))
        if clim_val is not None and ndvi_val is not None:
            ndvi_matched.append(ndvi_val)
            clim_matched.append(clim_val)

    if len(ndvi_matched) < 3:
        return {"n_pairs": len(ndvi_matched), "pearson_r": None,
                "pearson_p": None, "spearman_r": None, "interpretation": "Insufficient data."}

    pearson_r,  pearson_p  = sp_stats.pearsonr(ndvi_matched, clim_matched)
    spearman_r, spearman_p = sp_stats.spearmanr(ndvi_matched, clim_matched)

    if abs(pearson_r) > 0.7 and pearson_p < 0.05:
        strength = "strong"
    elif abs(pearson_r) > 0.4 and pearson_p < 0.10:
        strength = "moderate"
    else:
        strength = "weak or non-significant"

    direction = "positive" if pearson_r > 0 else "negative"

    var_labels = {
        "precip_mm": "precipitation",
        "temp_mean": "mean temperature",
        "temp_max":  "maximum temperature",
        "spi_3":     "3-month SPI drought index",
    }
    var_label = var_labels.get(climate_variable, climate_variable)
    lag_str   = f" (lag {lag_months}M)" if lag_months > 0 else ""

    interpretation = (
        f"{strength.capitalize()} {direction} correlation between NDVI and "
        f"{var_label}{lag_str} (r={pearson_r:.3f}, p={pearson_p:.3f}, n={len(ndvi_matched)}). "
    )
    if strength == "strong" and direction == "positive" and "precip" in climate_variable:
        interpretation += (
            "Vegetation activity at this site is strongly coupled to precipitation, "
            "consistent with water-limited conditions characteristic of sub-Mediterranean "
            "beech forests near their distributional limit."
        )
    elif strength == "strong" and direction == "negative" and "temp" in climate_variable:
        interpretation += (
            "Higher temperatures are associated with reduced canopy activity, "
            "suggesting heat/drought stress — a vulnerability signal of particular "
            "concern given projected warming trends under climate change scenarios."
        )

    return {
        "n_pairs":      len(ndvi_matched),
        "pearson_r":    round(pearson_r,  4),
        "pearson_p":    round(pearson_p,  4),
        "spearman_r":   round(spearman_r, 4),
        "spearman_p":   round(spearman_p, 4),
        "climate_var":  climate_variable,
        "lag_months":   lag_months,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Main data access functions
# ---------------------------------------------------------------------------

def fetch_climate_data(
    api_key: str,
    year_start: int = 2020,
    year_end: Optional[int] = None,
    station_priority: str = "montejo_de_la_sierra",
    config_path: Optional[str] = None,
) -> Optional[List[Dict]]:
    """
    Fetch and cache AEMET monthly climate data.

    Data is cached in data/climate/aemet_{station}_{year_start}_{year_end}.json.
    Returns parsed clean records, or None on failure.
    """
    if year_end is None:
        year_end = datetime.now(timezone.utc).year

    station = STATIONS.get(station_priority, STATIONS["montejo_de_la_sierra"])
    station_code = station["code"]

    CLIMATE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CLIMATE_DIR / f"aemet_{station_code}_{year_start}_{year_end}.json"

    if cache_file.exists():
        logger.info("Loading cached climate data from %s", cache_file)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if not api_key:
        logger.warning("AEMET API key not configured — climate integration disabled. "
                       "Register at https://opendata.aemet.es/centrodedescargas/altaUsuario")
        return None

    client = AEMETClient(api_key)
    raw = client.get_monthly_station_data(station_code, year_start, year_end)
    if raw is None:
        # Try fallback stations
        for station_name, st in sorted(STATIONS.items(), key=lambda x: x[1]["priority"]):
            if station_name == station_priority:
                continue
            logger.info("Trying fallback station: %s (%s)", station_name, st["code"])
            raw = client.get_monthly_station_data(st["code"], year_start, year_end)
            if raw:
                station_code = st["code"]
                break

    if raw is None:
        logger.error("All AEMET stations failed. Climate integration unavailable.")
        return None

    records = parse_monthly_records(raw)

    # Compute derived indices
    precip = [r.get("precip_mm") for r in records]
    months = [(r["year"], r["month"]) for r in records]
    spi_3  = compute_spi(precip, months, timescale=3)
    spi_12 = compute_spi(precip, months, timescale=12)
    temps  = [r.get("temp_mean") for r in records]
    temp_anomaly = compute_temperature_anomaly(temps, months)

    for i, r in enumerate(records):
        r["spi_3"]          = spi_3[i]
        r["spi_12"]         = spi_12[i]
        r["temp_anomaly_c"] = temp_anomaly[i]
        r["drought_status"] = _classify_spi(spi_3[i])

    # Cache
    save_json(records, str(cache_file))
    logger.info("Fetched and cached %d monthly records from AEMET (%s)", len(records), station_code)
    return records


def _classify_spi(spi: Optional[float]) -> str:
    if spi is None:
        return "unknown"
    if spi >= 2.0:   return "extremely_wet"
    if spi >= 1.5:   return "very_wet"
    if spi >= 1.0:   return "moderately_wet"
    if spi >= -0.99: return "near_normal"
    if spi >= -1.49: return "moderately_dry"
    if spi >= -1.99: return "severely_dry"
    return "extremely_dry"


def get_climate_context_for_period(
    climate_records: List[Dict],
    year: int,
    month_start: int,
    month_end: int,
) -> Dict[str, Any]:
    """
    Extract and summarise climate context for a specific year/period.

    Returns a structured summary suitable for ecological interpretation.
    """
    period_records = [
        r for r in climate_records
        if r["year"] == year and month_start <= r["month"] <= month_end
    ]
    if not period_records:
        return {"available": False, "year": year, "period": f"{month_start:02d}-{month_end:02d}"}

    temps   = [r["temp_mean"]   for r in period_records if r["temp_mean"]   is not None]
    precips = [r["precip_mm"]   for r in period_records if r["precip_mm"]   is not None]
    spis    = [r["spi_3"]       for r in period_records if r["spi_3"]       is not None]
    anomalies = [r["temp_anomaly_c"] for r in period_records if r["temp_anomaly_c"] is not None]

    # Seasonal totals
    total_precip  = sum(precips) if precips else None
    mean_temp     = np.mean(temps) if temps else None
    min_spi       = min(spis) if spis else None
    mean_anomaly  = np.mean(anomalies) if anomalies else None

    # Compare to climatological normals
    normal_precip = sum(
        CLIMATOLOGICAL_NORMALS_MONTHLY.get(m, (0, 0))[1]
        for m in range(month_start, month_end + 1)
    )
    precip_anomaly_pct = (
        100 * (total_precip - normal_precip) / normal_precip
        if total_precip is not None and normal_precip > 0 else None
    )

    drought_status = _classify_spi(min_spi)

    # Ecological interpretation
    eco_note = _interpret_climate_context(
        total_precip, precip_anomaly_pct, mean_temp, mean_anomaly, drought_status
    )

    return {
        "available":           True,
        "year":                year,
        "period":              f"{year}-{month_start:02d} to {year}-{month_end:02d}",
        "total_precip_mm":     round(total_precip, 1) if total_precip is not None else None,
        "normal_precip_mm":    round(normal_precip, 1),
        "precip_anomaly_pct":  round(precip_anomaly_pct, 1) if precip_anomaly_pct is not None else None,
        "mean_temp_c":         round(mean_temp, 2) if mean_temp is not None else None,
        "temp_anomaly_mean_c": round(mean_anomaly, 2) if mean_anomaly is not None else None,
        "min_spi_3":           round(min_spi, 2) if min_spi is not None else None,
        "drought_status":      drought_status,
        "ecological_context":  eco_note,
    }


def _interpret_climate_context(
    total_precip: Optional[float],
    precip_anomaly_pct: Optional[float],
    mean_temp: Optional[float],
    temp_anomaly: Optional[float],
    drought_status: str,
) -> str:
    notes = []

    if drought_status in ("severely_dry", "extremely_dry"):
        notes.append(
            "The observation period coincides with severe to extreme drought conditions "
            "(SPI-3 ≤ -1.5). Under these conditions, Fagus sylvatica activates stomatal "
            "closure, reducing photosynthesis and canopy conductance. NDVI depression "
            "during this period has a clear climate-driven explanation."
        )
    elif drought_status == "moderately_dry":
        notes.append(
            "Moderately dry conditions (SPI-3 between -1.0 and -1.5) may be contributing "
            "to any observed reduction in canopy activity."
        )

    if temp_anomaly is not None and temp_anomaly > 2.0:
        notes.append(
            f"Mean temperature is {temp_anomaly:+.1f}°C above the 1991-2020 normal. "
            "Elevated temperatures increase vapour pressure deficit and evapotranspiration demand, "
            "potentially exceeding the water supply capacity of the beech forest root system."
        )
    elif temp_anomaly is not None and temp_anomaly < -1.5:
        notes.append(
            f"Below-normal temperatures ({temp_anomaly:+.1f}°C anomaly) may delay phenological "
            "development or indicate a late-frost event affecting canopy condition."
        )

    if precip_anomaly_pct is not None and precip_anomaly_pct < -40:
        notes.append(
            f"Precipitation is {abs(precip_anomaly_pct):.0f}% below the seasonal normal, "
            "representing a significant water deficit. Historical studies show NDVI in the "
            "Hayedo de Montejo responds strongly to antecedent precipitation in the "
            "June-August period (water-limited ecosystem response)."
        )
    elif precip_anomaly_pct is not None and precip_anomaly_pct > 50:
        notes.append(
            f"Precipitation is {precip_anomaly_pct:.0f}% above normal. "
            "Favourable water conditions are expected to support above-average canopy activity."
        )

    if not notes:
        notes.append(
            "Climate conditions during this period are within normal ranges. "
            "Any vegetation anomalies are unlikely to have a straightforward climate explanation "
            "and warrant investigation of other drivers (disturbance, management, phenology)."
        )

    return " ".join(notes)
