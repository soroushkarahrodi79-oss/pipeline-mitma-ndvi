"""
Forest Health Index (FHI) — Phase 8.

A transparent, reproducible, weighted composite index that integrates the
independent spectral, climatic and phenological indicators of the monitoring
system into a single ecological condition score for the Hayedo de Montejo
(Fagus sylvatica, Sierra del Rincon, 41N, 1150-1450 m a.s.l.).

DESIGN PRINCIPLES
-----------------
1. TRANSPARENCY  - every sub-score, weight and the exact arithmetic that
   produces the final value is returned in the output dictionary. Nothing is
   hidden inside the number.
2. REPRODUCIBILITY - the index is a deterministic function of the input
   statistics and the weight vector declared in config/thresholds.yaml
   (section `forest_health_index`). The same inputs always yield the same FHI.
3. ECOLOGICAL GROUNDING - each sub-indicator is normalised to a [0, 1] health
   score using thresholds calibrated for F. sylvatica at this site (the same
   thresholds documented in ecology.py and validation.py), not arbitrary
   min-max scaling.
4. GRACEFUL DEGRADATION - when an indicator is unavailable (e.g. no AEMET key,
   no NDRE band), its weight is removed and the remaining weights are
   renormalised. A `data_completeness` field reports how much of the intended
   weight was actually available, so a high FHI computed from one indicator is
   never mistaken for a high FHI computed from all five.

THE FIVE SUB-INDICATORS AND THEIR WEIGHTS
-----------------------------------------
  NDVI  - Canopy vigour / photosynthetic capacity .............. 0.30
  NDMI  - Canopy water status (drought stress) ................. 0.25
  NDRE  - Chlorophyll / pre-symptomatic physiological stress ... 0.20
  CLIM  - Climatic water balance (SPI-3 + temperature anomaly) . 0.15
  PHEN  - Phenological integrity (peak-NDVI trajectory) ........ 0.10
                                                        sum  =  1.00

WEIGHT RATIONALE (ecological justification)
-------------------------------------------
NDVI (0.30) - The most directly validated and most robust state variable of
   canopy condition (LAI/GPP proxy). It receives the largest weight because it
   integrates the cumulative outcome of all stressors on the canopy and is the
   least noisy of the spectral indices.

NDMI (0.25) - Water status is the dominant proximate stressor for beech at its
   dry southern range margin. F. sylvatica has a shallow root system (~80% of
   fine roots in the upper 40 cm) and closes stomata early under soil-moisture
   deficit. Because drought is the leading documented driver of decline at this
   ecotone, NDMI is weighted almost as highly as NDVI.

NDRE (0.20) - The red-edge chlorophyll signal leads NDVI by 2-4 weeks and is
   the system's early-warning channel. It is weighted substantially because
   pre-symptomatic detection is a primary management objective, but below
   NDVI/NDMI because it is noisier and has weaker site-specific calibration.

CLIM (0.15) - SPI-3 and temperature anomaly describe the external climatic
   forcing rather than the state of the canopy itself, and are measured at an
   off-site station. They are included to provide driver/attribution context
   but down-weighted relative to the three direct canopy measurements.

PHEN (0.10) - The phenological trajectory (peak-NDVI trend / deviation) is a
   slow integrative variable. With the present sparse July-September composites
   it carries the largest uncertainty, so it receives the smallest weight; it
   is retained because long-term phenological drift is the signal of greatest
   relevance to the climate-change narrative of the TFM.

OUTPUT
------
A composite score on a 0-100 scale and one of five condition classes:
  Excellent (85-100) | Good (70-85) | Moderate (55-70) | Warning (40-55) |
  Critical (<40)

Reference framing: the FHI follows the general logic of multi-criteria forest
condition indices (e.g. the ICP-Forests crown-condition framework and the
composite vitality indices of Lausch et al. 2017, "Understanding forest health
with remote sensing", Remote Sensing) adapted to the Sentinel-2 indicator set
available here.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import WORKSPACE_ROOT, setup_logging, save_json
import ecology

logger = setup_logging("forest_health_index")

FHI_DIR = WORKSPACE_ROOT / "outputs" / "forest_health_index"

# ---------------------------------------------------------------------------
# Default configuration (overridable from config/thresholds.yaml)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ndvi":      0.30,
    "ndmi":      0.25,
    "ndre":      0.20,
    "climate":   0.15,
    "phenology": 0.10,
}

# Condition classes: (lower_bound_inclusive, label, icon, short interpretation)
FHI_CLASSES: List[Tuple[float, str, str, str]] = [
    (85.0, "Excellent", "🟢", "Canopy condition at or above the expected optimum for the phenological phase; no stress signals across indicators."),
    (70.0, "Good",      "🟢", "Healthy canopy within the expected range; minor or isolated stress signals, no management concern."),
    (55.0, "Moderate",  "🟡", "Canopy condition below optimum; one or more sub-indicators show incipient stress that warrants monitoring."),
    (40.0, "Warning",   "🟠", "Multiple convergent stress signals; the stand is under measurable pressure and field verification is advised."),
    (0.0,  "Critical",  "🔴", "Severe, convergent deterioration across indicators; urgent field assessment and management response required."),
]


# ---------------------------------------------------------------------------
# Sub-indicator scoring functions  (each returns a health score in [0, 1])
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_ndvi(ndvi_median: Optional[float], phase: str) -> Dict[str, Any]:
    """
    Phenology-relative NDVI health score.

    The observed NDVI is scored against the expected range for the current
    phenological phase (ecology.NDVI_EXPECTED). A value at or above the upper
    expected bound scores 1.0; at the lower bound scores ~0.55 (i.e. "still
    within normal but at the bottom of it"); progressively below the lower
    bound the score falls towards 0. This phenology-relative design prevents a
    naturally low dormant-season NDVI from being read as poor health.
    """
    if ndvi_median is None or ndvi_median != ndvi_median:
        return {"available": False, "score": None}

    low, high = ecology.NDVI_EXPECTED.get(phase, (0.0, 1.0))
    span = max(high - low, 1e-6)

    if ndvi_median >= high:
        score = 1.0
    elif ndvi_median >= low:
        # within expected band -> map to [0.55, 1.0]
        score = 0.55 + 0.45 * (ndvi_median - low) / span
    else:
        # below expected band -> map deficit to [0, 0.55]
        deficit = (low - ndvi_median) / span
        score = _clip01(0.55 * (1.0 - deficit))

    return {
        "available":     True,
        "score":         round(_clip01(score), 4),
        "observed":      round(ndvi_median, 4),
        "expected_low":  round(low, 3),
        "expected_high": round(high, 3),
        "phase":         phase,
    }


def score_ndmi(ndmi_mean: Optional[float]) -> Dict[str, Any]:
    """
    Canopy water-status health score from NDMI, using the site thresholds in
    ecology.NDMI_THRESHOLDS (well_hydrated 0.10, mild 0.00, moderate -0.10).
    Piecewise-linear anchors: >=0.20 -> 1.0, 0.10 -> 0.85, 0.00 -> 0.55,
    -0.10 -> 0.30, <=-0.20 -> 0.0.
    """
    if ndmi_mean is None or ndmi_mean != ndmi_mean:
        return {"available": False, "score": None}
    anchors = [(-0.20, 0.0), (-0.10, 0.30), (0.00, 0.55), (0.10, 0.85), (0.20, 1.0)]
    score = _interp_anchors(ndmi_mean, anchors)
    return {"available": True, "score": round(score, 4), "observed": round(ndmi_mean, 4)}


def score_ndre(ndre_mean: Optional[float]) -> Dict[str, Any]:
    """
    Chlorophyll / physiological health score from NDRE, using
    ecology.NDRE_THRESHOLDS (healthy 0.22, mild 0.16).
    Anchors: >=0.35 -> 1.0, 0.22 -> 0.80, 0.16 -> 0.50, 0.10 -> 0.20, <=0.05 -> 0.0.
    """
    if ndre_mean is None or ndre_mean != ndre_mean:
        return {"available": False, "score": None}
    anchors = [(0.05, 0.0), (0.10, 0.20), (0.16, 0.50), (0.22, 0.80), (0.35, 1.0)]
    score = _interp_anchors(ndre_mean, anchors)
    return {"available": True, "score": round(score, 4), "observed": round(ndre_mean, 4)}


def score_climate(climate_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Climatic water-balance health score from the SPI-3 drought class and the
    temperature anomaly. SPI maps the drought severity to a base score; a
    strong warm anomaly (> +2 C) applies a small penalty because elevated VPD
    compounds water stress for beech.
    """
    if not climate_context or not climate_context.get("available"):
        return {"available": False, "score": None}

    drought_status = climate_context.get("drought_status", "near_normal")
    base = {
        "extremely_wet":  1.00,
        "very_wet":       1.00,
        "moderately_wet": 0.95,
        "near_normal":    0.85,
        "moderately_dry": 0.60,
        "severely_dry":   0.35,
        "extremely_dry":  0.15,
        "unknown":        None,
    }.get(drought_status, None)

    if base is None:
        return {"available": False, "score": None}

    temp_anom = climate_context.get("temp_anomaly_mean_c")
    penalty = 0.0
    if temp_anom is not None and temp_anom > 2.0:
        penalty = min(0.15, 0.05 * (temp_anom - 2.0))

    score = _clip01(base - penalty)
    return {
        "available":       True,
        "score":           round(score, 4),
        "drought_status":  drought_status,
        "min_spi_3":       climate_context.get("min_spi_3"),
        "temp_anomaly_c":  temp_anom,
        "warm_penalty":    round(penalty, 4),
    }


def score_phenology(
    phenology_records: Optional[List[Dict[str, Any]]],
    current_peak_ndvi: Optional[float],
) -> Dict[str, Any]:
    """
    Phenological-integrity score.

    Two complementary signals are combined when available:
      (a) the deviation of the current peak NDVI from the historical mean peak,
          expressed in historical standard deviations (a z-score), and
      (b) the sign/strength of the multi-year peak-NDVI trend.

    With < 2 historical years there is no baseline, so the score is reported as
    unavailable rather than guessed.
    """
    if not phenology_records or len(phenology_records) < 2:
        return {"available": False, "score": None,
                "note": "Fewer than 2 historical years; phenological baseline not yet established."}

    peaks = [r.get("peak_ndvi") for r in phenology_records
             if r.get("peak_ndvi") is not None]
    if len(peaks) < 2:
        return {"available": False, "score": None}

    import numpy as np
    hist = np.array(peaks, dtype="float64")
    mu, sigma = float(hist.mean()), float(hist.std(ddof=1) if len(hist) > 1 else 0.0)

    ref_peak = current_peak_ndvi if current_peak_ndvi is not None else peaks[-1]
    if sigma > 1e-4:
        z = (ref_peak - mu) / sigma
        # map z in [-2, +1] -> [0, 1]; >=+1 -> 1.0, <=-2 -> 0.0
        score = _clip01((z + 2.0) / 3.0)
    else:
        # no variance in baseline; fall back to ratio against the mean
        score = _clip01(0.55 + 0.45 * (ref_peak - mu) / max(mu, 1e-6))
        z = None

    return {
        "available":        True,
        "score":            round(score, 4),
        "current_peak":     round(ref_peak, 4) if ref_peak is not None else None,
        "historical_mean":  round(mu, 4),
        "historical_std":   round(sigma, 4),
        "z_vs_baseline":    round(z, 3) if z is not None else None,
        "n_years":          len(peaks),
    }


def _interp_anchors(x: float, anchors: List[Tuple[float, float]]) -> float:
    """Piecewise-linear interpolation through (input, score) anchor points."""
    lo_x, lo_s = anchors[0]
    hi_x, hi_s = anchors[-1]
    if x <= lo_x:
        return lo_s
    if x >= hi_x:
        return hi_s
    for (x0, s0), (x1, s1) in zip(anchors[:-1], anchors[1:]):
        if x0 <= x <= x1:
            frac = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return _clip01(s0 + frac * (s1 - s0))
    return _clip01((lo_s + hi_s) / 2.0)


# ---------------------------------------------------------------------------
# Class assignment
# ---------------------------------------------------------------------------

def classify_fhi(score_0_100: float) -> Dict[str, str]:
    for lower, label, icon, interp in FHI_CLASSES:
        if score_0_100 >= lower:
            return {"class": label, "icon": icon, "class_interpretation": interp,
                    "class_lower_bound": lower}
    last = FHI_CLASSES[-1]
    return {"class": last[1], "icon": last[2], "class_interpretation": last[3],
            "class_lower_bound": last[0]}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_forest_health_index(
    observation_date: date,
    ndvi_stats: Optional[Dict[str, float]] = None,
    ndmi_stats: Optional[Dict[str, float]] = None,
    ndre_stats: Optional[Dict[str, float]] = None,
    climate_context: Optional[Dict[str, Any]] = None,
    phenology_records: Optional[List[Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute the composite Forest Health Index.

    Parameters
    ----------
    observation_date : date of the observation (sets the phenological phase).
    ndvi_stats / ndmi_stats / ndre_stats : index statistics dicts as produced
        by compositing.summarize_composite (need 'median' for NDVI, 'mean' for
        NDMI/NDRE). Any may be None.
    climate_context : output of climate.get_climate_context_for_period, or None.
    phenology_records : output of phenology.build_phenology_time_series, or None.
    weights : optional override of DEFAULT_WEIGHTS (e.g. from config).

    Returns
    -------
    A fully transparent dictionary containing the final score (0-100), the
    condition class, every sub-score, the effective (renormalised) weights, the
    weighted contributions, and a data-completeness fraction.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    phase = ecology.get_phenological_phase(observation_date)

    subscores = {
        "ndvi":      score_ndvi((ndvi_stats or {}).get("median"), phase),
        "ndmi":      score_ndmi((ndmi_stats or {}).get("mean")),
        "ndre":      score_ndre((ndre_stats or {}).get("mean")),
        "climate":   score_climate(climate_context),
        "phenology": score_phenology(phenology_records,
                                     (ndvi_stats or {}).get("median")),
    }

    # --- renormalise weights over the available sub-indicators ---
    available = {k: v for k, v in subscores.items() if v.get("available")}
    intended_weight = sum(weights.get(k, 0.0) for k in subscores)
    available_weight = sum(weights.get(k, 0.0) for k in available)

    contributions: Dict[str, Dict[str, float]] = {}
    effective_weights: Dict[str, float] = {}
    weighted_sum = 0.0

    if available_weight > 0:
        for k, sub in available.items():
            w_eff = weights.get(k, 0.0) / available_weight
            effective_weights[k] = round(w_eff, 4)
            contrib = w_eff * sub["score"]
            weighted_sum += contrib
            contributions[k] = {
                "sub_score":         sub["score"],
                "nominal_weight":    round(weights.get(k, 0.0), 4),
                "effective_weight":  round(w_eff, 4),
                "weighted_contribution": round(contrib, 4),
            }

    fhi_0_100 = round(100.0 * weighted_sum, 2) if available_weight > 0 else None
    data_completeness = round(available_weight / intended_weight, 4) if intended_weight else 0.0

    result: Dict[str, Any] = {
        "fhi_score":          fhi_0_100,
        "observation_date":   observation_date.isoformat(),
        "phenological_phase": phase,
        "sub_scores":         subscores,
        "effective_weights":  effective_weights,
        "contributions":      contributions,
        "nominal_weights":    {k: round(v, 4) for k, v in weights.items()},
        "data_completeness":  data_completeness,
        "n_indicators_used":  len(available),
        "indicators_used":    sorted(available.keys()),
        "methodology_version": "FHI-1.0",
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }

    if fhi_0_100 is not None:
        result.update(classify_fhi(fhi_0_100))
        result["interpretation"] = _narrative(result)
        result["confidence_note"] = _confidence_note(data_completeness, len(available),
                                                      phenology_records)
    else:
        result.update({"class": "N/D", "icon": "⚪",
                       "class_interpretation": "No indicators available to compute the index.",
                       "interpretation": "The Forest Health Index could not be computed because "
                                         "no sub-indicator statistics were available for this observation.",
                       "confidence_note": "No data."})

    return result


def _narrative(result: Dict[str, Any]) -> str:
    cls = result["class"]
    score = result["fhi_score"]
    parts = [
        f"The integrated Forest Health Index for {result['observation_date']} is "
        f"{score:.1f}/100, placing the stand in the '{cls}' condition class. "
    ]
    # identify the strongest and weakest contributing indicators
    contribs = result.get("contributions", {})
    if contribs:
        subs = {k: v["sub_score"] for k, v in contribs.items()}
        weakest = min(subs, key=subs.get)
        strongest = max(subs, key=subs.get)
        labels = {"ndvi": "canopy vigour (NDVI)", "ndmi": "canopy water status (NDMI)",
                  "ndre": "chlorophyll status (NDRE)", "climate": "climatic water balance",
                  "phenology": "phenological trajectory"}
        if subs[weakest] < 0.55:
            parts.append(
                f"The condition is limited primarily by {labels.get(weakest, weakest)} "
                f"(sub-score {subs[weakest]:.2f}), the weakest of the {len(subs)} "
                f"contributing indicators. ")
        else:
            parts.append(
                f"All contributing indicators are in or above their normal ranges; "
                f"{labels.get(strongest, strongest)} is the strongest signal "
                f"(sub-score {subs[strongest]:.2f}). ")
    return "".join(parts)


def _confidence_note(completeness: float, n_used: int,
                     phenology_records: Optional[List]) -> str:
    notes = []
    if completeness < 0.999:
        missing_pct = 100 * (1 - completeness)
        notes.append(
            f"The index was computed from {n_used} of 5 intended indicators "
            f"({missing_pct:.0f}% of the nominal weight was unavailable and the "
            f"remaining weights were renormalised). Treat the value as indicative.")
    if not phenology_records or len(phenology_records) < 3:
        notes.append(
            "The phenological component rests on fewer than 3 baseline years; "
            "its contribution carries high uncertainty until the historical "
            "series lengthens.")
    if not notes:
        notes.append("All five sub-indicators were available; the index reflects "
                     "the full intended methodology.")
    return " ".join(notes)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_fhi(result: Dict[str, Any], label: str) -> Path:
    FHI_DIR.mkdir(parents=True, exist_ok=True)
    path = FHI_DIR / f"fhi_{label}.json"
    save_json(result, str(path))
    logger.info("Forest Health Index saved: %s (score=%s, class=%s)",
                path, result.get("fhi_score"), result.get("class"))
    return path


def fhi_markdown_section(result: Dict[str, Any]) -> str:
    """Render a thesis-ready Markdown section describing the FHI result."""
    if result.get("fhi_score") is None:
        return ("## Forest Health Index\n\n"
                "The Forest Health Index could not be computed for this observation "
                "(no sub-indicator statistics available).\n")

    rows = ""
    labels = {"ndvi": "Canopy vigour (NDVI)", "ndmi": "Water status (NDMI)",
              "ndre": "Chlorophyll (NDRE)", "climate": "Climatic balance (SPI/Tanom)",
              "phenology": "Phenological trajectory"}
    for k in ["ndvi", "ndmi", "ndre", "climate", "phenology"]:
        sub = result["sub_scores"].get(k, {})
        if sub.get("available"):
            c = result["contributions"][k]
            rows += (f"| {labels[k]} | {sub['score']:.2f} | {c['nominal_weight']:.2f} | "
                     f"{c['effective_weight']:.2f} | {c['weighted_contribution']:.3f} |\n")
        else:
            rows += f"| {labels[k]} | n/a | {result['nominal_weights'].get(k, 0):.2f} | - | - |\n"

    return f"""## Forest Health Index (FHI)

**FHI = {result['fhi_score']:.1f} / 100 -> {result['icon']} {result['class']}**
*(methodology {result['methodology_version']}; data completeness {result['data_completeness']*100:.0f}%)*

{result['class_interpretation']}

{result['interpretation']}

| Sub-indicator | Health sub-score (0-1) | Nominal weight | Effective weight | Contribution |
|---------------|------------------------|----------------|------------------|--------------|
{rows}| **Composite** | | **1.00** | **1.00** | **{result['fhi_score']/100:.3f}** |

> **Confidence:** {result['confidence_note']}

**Classification scale:** Excellent (85-100) - Good (70-85) - Moderate (55-70) - Warning (40-55) - Critical (<40).
The weighting scheme and its ecological rationale are documented in `src/forest_health_index.py`.
"""


if __name__ == "__main__":
    # Self-test with synthetic, healthy-forest inputs (no I/O dependencies).
    demo = compute_forest_health_index(
        observation_date=date(2025, 8, 15),
        ndvi_stats={"median": 0.78, "mean": 0.77, "std": 0.05},
        ndmi_stats={"mean": 0.18},
        ndre_stats={"mean": 0.30},
        climate_context={"available": True, "drought_status": "near_normal",
                         "min_spi_3": -0.4, "temp_anomaly_mean_c": 0.6},
        phenology_records=[{"year": 2021, "peak_ndvi": 0.80},
                           {"year": 2022, "peak_ndvi": 0.76},
                           {"year": 2023, "peak_ndvi": 0.79},
                           {"year": 2024, "peak_ndvi": 0.77}],
    )
    print("FHI demo score:", demo["fhi_score"], "class:", demo["class"])
    print("indicators used:", demo["indicators_used"],
          "completeness:", demo["data_completeness"])
    # Markdown contains class icons (emoji); write to a UTF-8 file rather than
    # printing, to stay compatible with non-UTF-8 consoles.
    _demo_path = FHI_DIR / "_selftest_fhi_section.md"
    FHI_DIR.mkdir(parents=True, exist_ok=True)
    _demo_path.write_text(fhi_markdown_section(demo), encoding="utf-8")
    print("markdown section written to:", _demo_path)
