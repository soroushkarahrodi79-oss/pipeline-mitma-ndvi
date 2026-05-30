"""
Thesis Output Package — Phase 11 (`--thesis-report`).

Assembles the full body of monitoring evidence into a single, self-contained,
publication-quality academic document suitable for direct inclusion in a
master's thesis (Trabajo Fin de Master). The language is formal and academic;
every quantitative statement is drawn from the actual outputs of the pipeline.

Structure (the ten required sections):
  1. Executive summary
  2. Methodology summary
  3. Results
  4. Trend analysis
  5. Phenology analysis
  6. Climate context
  7. Forest Health Index
  8. Discussion
  9. Limitations
 10. Conclusions

The orchestrator `run_thesis_report` gathers the latest available products
from disk (annual or rolling composite, forest mask, spectral statistics,
phenology series, climate context), computes the Forest Health Index, renders
the thesis figure set, and writes the document plus a companion figures
folder. It re-uses existing modules and does not re-download or re-process
satellite data.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio

from common import (
    WORKSPACE_ROOT, CONFIG_DIR, PROCESSED_DIR, COMPOSITES_DIR,
    load_yaml_config, setup_logging, save_json,
)
import compositing
import ecology
import forest_boundary
import forest_health_index as fhi_mod
import phenology as phenology_module
import cartography
import defensibility
import validation

logger = setup_logging("thesis_report")

THESIS_DIR = WORKSPACE_ROOT / "outputs" / "thesis"

SITE = "Hayedo de Montejo"
RESERVE = "Reserva de la Biosfera Sierra del Rincon"
SPECIES = "Fagus sylvatica L."
COORDS = "41 deg 13'-41 deg 21' N / 3 deg 25'-3 deg 36' W"
ALTITUDE = "1150-1450 m a.s.l."
SENSOR = "Sentinel-2 MSI L2A (ESA Copernicus / CDSE), 10 m"


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _scan_glob(directory: Path, pattern: str) -> List[str]:
    return sorted(str(p) for p in directory.glob(pattern))


def _latest_processed_index(suffix: str) -> List[str]:
    return sorted(str(p) for p in PROCESSED_DIR.glob(f"*_{suffix}.tif"))


def _build_forest_mask(composite_path: str, config: Dict):
    if not config.get("forest_boundary", {}).get("apply_mask", True):
        return None, "disabled"
    try:
        polygon = forest_boundary.get_forest_polygon(
            try_overpass=config.get("forest_boundary", {}).get("try_overpass", True))
        status = polygon["features"][0]["properties"].get("boundary_status", "unknown")
        with rasterio.open(composite_path) as src:
            profile = src.profile
        mask = forest_boundary.create_forest_mask_array(profile, polygon)
        if int(mask.sum()) == 0:
            logger.warning("Forest polygon has ZERO overlap with the composite extent "
                           "(imagery does not cover the forest; re-acquire for the AOI).")
            return mask, f"{status}_NO_OVERLAP"
        return mask, status
    except Exception as exc:
        logger.warning("Forest mask unavailable: %s", exc)
        return None, "unavailable"


def _index_stats(suffix: str, forest_mask) -> Optional[Dict[str, float]]:
    paths = _latest_processed_index(suffix)
    if not paths:
        return None
    vals: List[float] = []
    for p in paths:
        try:
            with rasterio.open(p) as src:
                arr = src.read(1).astype("float32")
                nd = src.nodata
                if nd is not None:
                    arr[arr == nd] = np.nan
                if forest_mask is not None and forest_mask.shape == arr.shape:
                    arr[~forest_mask] = np.nan
                vals.extend(arr[np.isfinite(arr)].tolist())
        except Exception:
            continue
    if not vals:
        return None
    a = np.array(vals, dtype="float32")
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "valid_pixels": int(a.size)}


def gather_thesis_inputs(config_path: Optional[str] = None,
                         year: Optional[int] = None) -> Dict[str, Any]:
    """Collect the latest available products into a single inputs dictionary."""
    config = load_yaml_config(config_path)

    # Prefer an annual scientific composite; fall back to the latest rolling one.
    annual = (_scan_glob(COMPOSITES_DIR / "baseline", "annual_ndvi_????.tif") or
              _scan_glob(COMPOSITES_DIR, "annual_ndvi_????.tif"))
    rolling = _scan_glob(COMPOSITES_DIR, "rolling_ndvi_*.tif")

    if year is not None:
        match = [p for p in annual if p.endswith(f"{year}.tif")]
        primary = match[0] if match else (annual[-1] if annual else (rolling[-1] if rolling else None))
    else:
        primary = annual[-1] if annual else (rolling[-1] if rolling else None)

    if primary is None:
        raise FileNotFoundError(
            "No NDVI composite found in data/composites. Run an annual or monthly "
            "job first so a composite exists to report on.")

    is_annual = "annual" in Path(primary).stem
    if year is None:
        if is_annual:
            try:
                year = int(Path(primary).stem.split("_")[-1])
            except Exception:
                year = date.today().year
        else:
            year = date.today().year
    obs_date = date(year, 8, 15) if is_annual else date.today()

    forest_mask, boundary_status = _build_forest_mask(primary, config)

    ndvi_stats = compositing.summarize_composite(primary, forest_mask=forest_mask)
    ndvi_stats["boundary_status"] = boundary_status
    ndmi_stats = _index_stats("NDMI", forest_mask)
    ndre_stats = _index_stats("NDRE", forest_mask)
    nbr_stats = _index_stats("NBR", forest_mask)

    # Trend (annual series)
    trend_slope = None
    trend_path = None
    if len(annual) >= 3:
        try:
            tp = COMPOSITES_DIR / "thesis_ndvi_trend.tif"
            trend_path = compositing.compute_trend(annual, output_path=str(tp))
            trend_slope = compositing.summarize_trend(str(trend_path))
        except Exception as exc:
            logger.warning("Trend computation failed: %s", exc)

    # Phenology
    phenology_records = phenology_trends = None
    if annual:
        try:
            annual_stats = {int(Path(p).stem.split("_")[-1]):
                            compositing.summarize_composite(p, forest_mask=forest_mask)
                            for p in annual}
            phenology_records = phenology_module.build_phenology_time_series(annual_stats)
            phenology_trends = phenology_module.analyse_phenology_trends(phenology_records)
        except Exception as exc:
            logger.warning("Phenology analysis failed: %s", exc)

    # Climate (cached only — no network dependency for the thesis build)
    climate_context = _load_cached_climate(config, year)

    # Ecology + FHI
    eco = ecology.full_ecological_assessment(
        observation_date=obs_date, ndvi_stats=ndvi_stats,
        ndmi_stats=ndmi_stats, ndre_stats=ndre_stats, nbr_stats=nbr_stats)

    weights = config.get("forest_health_index", {}).get("weights") or fhi_mod.DEFAULT_WEIGHTS
    fhi = fhi_mod.compute_forest_health_index(
        observation_date=obs_date, ndvi_stats=ndvi_stats, ndmi_stats=ndmi_stats,
        ndre_stats=ndre_stats, climate_context=climate_context,
        phenology_records=phenology_records, weights=weights)

    return {
        "config": config, "year": year, "obs_date": obs_date,
        "primary_composite": primary, "is_annual": is_annual,
        "boundary_status": boundary_status, "forest_mask": forest_mask,
        "ndvi_stats": ndvi_stats, "ndmi_stats": ndmi_stats,
        "ndre_stats": ndre_stats, "nbr_stats": nbr_stats,
        "trend_slope": trend_slope, "trend_path": str(trend_path) if trend_path else None,
        "phenology_records": phenology_records, "phenology_trends": phenology_trends,
        "climate_context": climate_context, "ecology": eco, "fhi": fhi,
        "n_annual": len(annual),
    }


def _load_cached_climate(config: Dict, year: int) -> Optional[Dict]:
    try:
        import climate as climate_module
        ae = config.get("aemet", {})
        key = ae.get("api_key", "")
        records = climate_module.fetch_climate_data(
            api_key=key, year_start=year - 1, year_end=year,
            station_priority=ae.get("primary_station", "montejo_de_la_sierra"))
        if records:
            return climate_module.get_climate_context_for_period(
                records, year, month_start=6, month_end=9)
    except Exception as exc:
        logger.info("Climate context not available for thesis build: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Document sections (academic prose)
# ---------------------------------------------------------------------------

def _fmt(v, d=3, u=""):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{d}f}{u}"


# English phenological-phase names for the academic (English) document, keyed by
# the phase identifiers used in ecology.get_phenological_phase.
_PHASE_EN = {
    "dormancy": "winter dormancy",
    "budburst": "budburst / leaf-out",
    "leaf_expansion": "leaf expansion",
    "full_canopy": "full canopy (peak photosynthetic activity)",
    "senescence_early": "early senescence",
    "senescence_late": "advanced senescence (autumn colouration)",
    "post_senescence": "post-senescence (bare canopy)",
}


def _phase_en(eco: Dict) -> str:
    return _PHASE_EN.get(eco.get("phenological_phase", ""), "current")


def _sec_executive(d: Dict) -> str:
    fhi = d["fhi"]
    eco = d["ecology"]
    cls = fhi.get("class", "n/a")
    score = fhi.get("fhi_score")
    return f"""## 1. Executive summary

This report presents the condition of the {SITE} beech forest ({SPECIES}; {COORDS};
{ALTITUDE}), one of the southernmost mature populations of *Fagus sylvatica* in Europe
and a component of the {RESERVE}, as derived from {SENSOR} observations processed through
a reproducible, config-driven monitoring pipeline.

For the reference period ({d['obs_date'].isoformat()}), the integrated **Forest Health
Index is {_fmt(score,1)}/100, corresponding to the '{cls}' condition class**. The
forest-masked canopy NDVI is {_fmt(d['ndvi_stats'].get('median'),3)} (median), with a
canopy water status (NDMI) of {_fmt((d['ndmi_stats'] or {}).get('mean'),3)} and a
red-edge chlorophyll signal (NDRE) of {_fmt((d['ndre_stats'] or {}).get('mean'),3)}.
The composite operational condition flag is **{eco.get('overall_status','n/a')}**.

{fhi.get('interpretation','')} {fhi.get('confidence_note','')}
"""


def _sec_methodology(d: Dict) -> str:
    return f"""## 2. Methodology summary

**Data.** {SENSOR} surface-reflectance scenes were selected with a cloud-cover ceiling of
{d['config'].get('api', {}).get('cloud_cover_max', 15)} % and masked to Scene Classification
Layer (SCL) classes 4 (vegetation) and 5 (bare soil/rock). All canopy statistics are
computed within a forest polygon (status: *{d['boundary_status']}*), so the reported values
characterise the beech stand rather than the bounding box.

**Indicators.** Five spectral indices are derived per scene: NDVI (canopy vigour), NDMI
(canopy water content, SWIR-based), NDRE (red-edge chlorophyll, an early-warning channel
leading NDVI by 2-4 weeks), NBR (structural disturbance) and EVI/MSAVI2 (auxiliary). A
fixed July 1-September 30 per-pixel median composite defines the annual scientific product;
a separate rolling 30-45 day composite supports operational monitoring. These two temporal
layers are kept strictly separate.

**Derived analytics.** Inter-annual change is quantified with the robust Sen's-slope
estimator and the Mann-Kendall test; anomalies are flagged with a phenologically-normalised
z-score against the multi-year baseline with a >=2-observation persistence guard;
phenological metrics follow Reed et al. (1994); drought context uses the Standardised
Precipitation Index (SPI; McKee et al. 1993) from AEMET station data. The composite
**Forest Health Index** integrates NDVI, NDMI, NDRE, the climatic water balance and the
phenological trajectory under a transparent, renormalising weighting scheme
(methodology {d['fhi'].get('methodology_version','FHI-1.0')}). The full per-indicator
scientific basis, uncertainty budget and false-positive analysis are documented in the
companion Scientific Validation Document.
"""


def _sec_results(d: Dict) -> str:
    rows = ""
    for name, s in [("NDVI", d["ndvi_stats"]), ("NDMI", d["ndmi_stats"]),
                    ("NDRE", d["ndre_stats"]), ("NBR", d["nbr_stats"])]:
        if s:
            rows += (f"| {name} | {_fmt(s.get('mean'),4)} | {_fmt(s.get('median'),4)} | "
                     f"{_fmt(s.get('std'),4)} | {s.get('valid_pixels','n/a'):,} |\n")
    return f"""## 3. Results

Canopy spectral statistics for the reference composite ({Path(d['primary_composite']).name},
forest-masked) are summarised below.

| Index | Mean | Median | SD | Valid pixels |
|-------|------|--------|----|--------------|
{rows}
The canopy NDVI distribution spans the 10th-90th percentile range
{_fmt(d['ndvi_stats'].get('p10'),3)}-{_fmt(d['ndvi_stats'].get('p90'),3)}, indicating the
internal heterogeneity of the stand. The ecological engine classifies the current NDVI as
**{d['ecology'].get('ndvi_health',{}).get('status','n/a')}** for the
{_phase_en(d['ecology'])} phenological phase. Figure *map_ndvi_composite.png*
shows the spatial distribution of canopy vigour across the forest polygon.
"""


def _sec_trend(d: Dict) -> str:
    ts = d["trend_slope"]
    pt = d.get("phenology_trends") or {}
    pk = pt.get("peak_ndvi_trend") or {}
    base = (f"The per-pixel NDVI trend (Sen's slope) over the {d['n_annual']} available "
            f"annual composites has a median of {_fmt(ts,5)} NDVI/year."
            if ts is not None else
            "An inter-annual trend could not be computed: at least three annual composites "
            "are required, and fewer are currently available.")
    pk_txt = ""
    if pk:
        pk_txt = (f"\n\nThe peak-season NDVI trajectory yields a slope of "
                  f"{_fmt(pk.get('slope_per_year'),5)} NDVI/year "
                  f"(R^2 = {_fmt(pk.get('r_squared'),3)}, p = {_fmt(pk.get('p_value'),3)}, "
                  f"Mann-Kendall tau = {_fmt(pk.get('mann_kendall_tau'),3)}). "
                  f"{pk.get('interpretation','')}")
    return f"""## 4. Trend analysis

{base}{pk_txt}

Trends at this series length must be read with caution: with fewer than eight annual
composites the Mann-Kendall test has limited statistical power, and pixel-wise estimates
are affected by spatial autocorrelation. These constraints are quantified in Section 9 and
in the Defensibility Report. The Iberian Peninsula background greening trend
(~+0.002 NDVI/year, MODIS 2000-2020) is the null hypothesis against which any site-specific
trend should be compared.
"""


def _sec_phenology(d: Dict) -> str:
    recs = d.get("phenology_records") or []
    table = ""
    for r in recs[-6:]:
        table += (f"| {r['year']} | {_fmt(r.get('peak_ndvi'),3)} | "
                  f"{r.get('growing_season_length_days','n/a')} | {r.get('method','n/a')} |\n")
    body = (f"""Peak-season metrics for the available years:

| Year | Peak NDVI | GSL (days) | Method |
|------|-----------|-----------|--------|
{table}""" if recs else
            "No multi-year phenological record is yet available; it will populate as annual "
            "composites accumulate.")
    return f"""## 5. Phenology analysis

For *F. sylvatica* at this latitude and elevation, budburst (SOS) typically occurs around
DOY 90-120, peak canopy around DOY 200-240, and leaf fall (EOS) around DOY 280-310.
{body}

The present annual July-September composites capture the peak-season value robustly but
cannot resolve SOS/EOS, which require intra-seasonal (April-November) observations. Figure
*phenology_time_series.png* plots the peak-NDVI trajectory. Documented phenological shifts
are among the most sensitive biological indicators of climate-change impact and constitute
direct site-level evidence for the adaptive-management argument of the Reserve.
"""


def _sec_climate(d: Dict) -> str:
    c = d.get("climate_context")
    if not c or not c.get("available"):
        return """## 6. Climate context

AEMET climate records were not available for this build, so the canopy signals are reported
without an independent meteorological attribution. Enabling the AEMET integration
(`aemet.api_key` in the configuration) adds monthly temperature, precipitation, SPI-3/SPI-12
drought indices and anomalies versus the 1991-2020 normal, which are required to distinguish
climate-driven stress from structural disturbance. This is flagged as a limitation.
"""
    return f"""## 6. Climate context

For the analysed period, total precipitation was {_fmt(c.get('total_precip_mm'),1,' mm')}
({_fmt(c.get('precip_anomaly_pct'),1,' %')} versus the seasonal normal), mean temperature
{_fmt(c.get('mean_temp_c'),1,' deg C')} ({_fmt(c.get('temp_anomaly_mean_c'),2,' deg C')}
anomaly versus 1991-2020), and the minimum SPI-3 was {_fmt(c.get('min_spi_3'),2)}
(drought class: {c.get('drought_status','n/a')}).

{c.get('ecological_context','')}
"""


def _sec_fhi(d: Dict) -> str:
    section = fhi_mod.fhi_markdown_section(d["fhi"])
    # Number the heading to keep the ten-section academic ordering.
    return section.replace("## Forest Health Index (FHI)",
                           "## 7. Forest Health Index (FHI)", 1)


def _sec_discussion(d: Dict) -> str:
    fhi = d["fhi"]
    eco = d["ecology"]
    return f"""## 8. Discussion

The convergence (or divergence) of independent indicators is the core of the interpretation.
In this period the canopy vigour, water status and chlorophyll channels combine into an FHI
of {_fmt(fhi.get('fhi_score'),1)}/100 ('{fhi.get('class','n/a')}'). Because NDRE responds to
physiological stress 2-4 weeks before NDVI, the agreement or disagreement between NDRE and
NDVI is diagnostic: concordant decline points to an established stress episode, whereas an
NDRE-only decline flags an emerging, pre-symptomatic condition warranting closer observation.

The position of this population at the dry southern margin of the beech distribution makes
water balance the dominant control on its condition, which is why NDMI carries a high weight
in the index. Where the climatic record is available, attributing canopy water stress to a
concurrent SPI-3 drought strengthens the causal interpretation; where it is absent, stress
signals remain spectrally robust but causally open. The current ecological flag
({eco.get('overall_status','n/a')}) should therefore be read together with the climate
section and, ultimately, with field observations, which remote sensing complements rather
than replaces.
"""


def _sec_limitations(d: Dict) -> str:
    report = defensibility.generate_defensibility_report()
    s = report["summary"]
    crit = [f for grp in report["findings"].values() for f in grp
            if f["severity"] in ("critical", "major") and f["status"] != "resolved"]
    bullets = "\n".join(f"- **{f['weakness']}** — {f['impact']}" for f in crit[:6])
    return f"""## 9. Limitations

This study is subject to well-characterised limitations, audited in full in the companion
TFM Defensibility Report (defensibility readiness score:
{s['defensibility_readiness_score']}/100). The most material are:

{bullets}

These limitations do not invalidate the monitoring signal, but they bound the strength of
the inferential claims: the system is presented as a reproducible, defensible *early-warning
and condition-tracking* tool, not as a calibrated, field-validated inventory. Each limitation
is paired with a concrete remediation pathway in the Defensibility Report.
"""


def _sec_conclusions(d: Dict) -> str:
    fhi = d["fhi"]
    return f"""## 10. Conclusions

1. A reproducible, config-driven Sentinel-2 pipeline characterises the condition of the
   {SITE} beech forest through five complementary spectral indicators, climatic context and
   a transparent composite Forest Health Index.

2. For the reference period the forest is in the **'{fhi.get('class','n/a')}'** condition
   class (FHI {_fmt(fhi.get('fhi_score'),1)}/100), computed from
   {fhi.get('n_indicators_used','n/a')} of five indicators
   ({fhi.get('data_completeness',0)*100:.0f}% data completeness).

3. The system delivers operational early warning (via the pre-symptomatic NDRE channel and
   persistence-guarded anomaly detection) while keeping the annual scientific product
   strictly separate for defensible inter-annual analysis.

4. The principal avenues to strengthen the work toward full scientific defensibility are
   field validation of the spectral thresholds, replacement of the provisional forest
   polygon with official cartography, topographic/BRDF correction, and extension of the
   temporal baseline. With these, the observatory moves from a robust monitoring prototype
   to a fully validated environmental decision-support system for the {RESERVE}.

---
*{SENSOR} · {SITE} · {COORDS} · generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.*
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_document(d: Dict[str, Any]) -> str:
    header = f"""# Environmental Monitoring of the {SITE}
## A Sentinel-2 Remote Sensing Assessment of a Southern-Margin *{SPECIES}* Forest

**Site:** {SITE} — {RESERVE}
**Coordinates:** {COORDS} · **Altitude:** {ALTITUDE}
**Sensor:** {SENSOR}
**Reference period:** {d['obs_date'].isoformat()}  ·  **Report generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

---
"""
    sections = [
        _sec_executive(d), _sec_methodology(d), _sec_results(d), _sec_trend(d),
        _sec_phenology(d), _sec_climate(d), _sec_fhi(d), _sec_discussion(d),
        _sec_limitations(d), _sec_conclusions(d),
    ]
    return header + "\n\n".join(sections) + "\n"


def run_thesis_report(config_path: Optional[str] = None,
                      year: Optional[int] = None) -> Path:
    """End-to-end: gather inputs, render figures, write the thesis document set."""
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    d = gather_thesis_inputs(config_path, year)

    label = str(d["year"])
    out_dir = THESIS_DIR / label
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figures
    try:
        index_paths = {}
        for key in ["NDMI", "NDRE", "NBR"]:
            paths = _latest_processed_index(key)
            if paths:
                index_paths[key] = paths[-1]
        cartography.render_standard_figures(
            composite_path=d["primary_composite"],
            index_paths=index_paths,
            trend_path=d.get("trend_path"),
            phenology_records=d.get("phenology_records"),
            fhi=d["fhi"],
            forest_mask=d.get("forest_mask"),
            observation_label=d["obs_date"].isoformat(),
            out_dir=str(fig_dir))
    except Exception as exc:
        logger.warning("Figure rendering failed (document will still be written): %s", exc)

    # Companion documents
    try:
        validation.save_validation_document(observation_date=d["obs_date"].isoformat())
    except Exception as exc:
        logger.warning("Validation document generation failed: %s", exc)
    try:
        defensibility.save_defensibility_report()
    except Exception as exc:
        logger.warning("Defensibility report generation failed: %s", exc)

    # Main document
    document = build_document(d)
    doc_path = out_dir / f"thesis_report_{label}.md"
    doc_path.write_text(document, encoding="utf-8")

    # Machine-readable companion
    save_json({k: v for k, v in d.items() if k not in ("forest_mask", "config")},
              str(out_dir / f"thesis_data_{label}.json"))

    logger.info("Thesis report written: %s", doc_path)
    return doc_path


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print("Thesis report:", run_thesis_report(year=yr))
