"""
Management and Conservation Decision-Support module — Phase 9.

Translates the technical outputs of the monitoring system (spectral indices,
anomaly detection, climate context, Forest Health Index) into structured,
management-oriented intelligence for the managers of the Reserva de la Biosfera
Sierra del Rincon and the Hayedo de Montejo Natura 2000 / CETS framework.

Unlike the per-indicator ecological engine in ecology.py, this module reasons
at the level of *management decisions*. For every execution it produces six
decision blocks:

  1. Forest condition assessment   - the headline state and what it means.
  2. Potential stress factors      - ranked, with attribution confidence.
  3. Drought risk interpretation   - current risk and short-term outlook.
  4. Visitor pressure implications - access/use guidance tied to fragility.
  5. Conservation implications     - Natura 2000 / CETS / habitat 9120-9150.
  6. Monitoring recommendations    - what to observe next and when.

The output is designed to be read by a protected-area manager who is not a
remote-sensing specialist, while remaining fully traceable to the underlying
quantitative evidence.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import WORKSPACE_ROOT, setup_logging, save_json

logger = setup_logging("management")

MANAGEMENT_DIR = WORKSPACE_ROOT / "outputs" / "management"

# Habitat context: the Hayedo de Montejo qualifies as EU Habitats Directive
# priority/Annex-I beech-forest habitat types.
HABITAT_CODES = {
    "9120": "Atlantic acidophilous beech forests with Ilex and sometimes Taxus",
    "9150": "Medio-European limestone beech forests (Cephalanthero-Fagion)",
}


# ---------------------------------------------------------------------------
# Block 1 — Forest condition assessment
# ---------------------------------------------------------------------------

def _condition_assessment(fhi: Optional[Dict], eco: Dict) -> Dict[str, Any]:
    status = eco.get("overall_status", "N/D")
    if fhi and fhi.get("fhi_score") is not None:
        headline = (f"Forest Health Index {fhi['fhi_score']:.0f}/100 "
                    f"({fhi['class']}). {fhi.get('class_interpretation', '')}")
        condition_class = fhi["class"]
    else:
        headline = (f"Composite condition flag: {status}. "
                    "Forest Health Index not available this run.")
        condition_class = status

    return {
        "headline":         headline,
        "condition_class":  condition_class,
        "operational_flag": status,
        "summary": (
            "The assessment integrates Sentinel-2 spectral indicators (canopy vigour, "
            "water status, chlorophyll), the available climatic context and the "
            "phenological trajectory. It describes the condition of the beech canopy "
            "relative to the expected state for the current phenological phase, not an "
            "absolute judgement independent of season."),
    }


# ---------------------------------------------------------------------------
# Block 2 — Potential stress factors (ranked, with attribution confidence)
# ---------------------------------------------------------------------------

def _stress_factors(eco: Dict, climate: Optional[Dict], anomaly: Dict) -> List[Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []

    water = eco.get("water_stress") or {}
    chloro = eco.get("chlorophyll_health") or {}
    disturb = eco.get("disturbance") or {}
    ndvi_h = eco.get("ndvi_health") or {}

    if water.get("level") in ("MODERATE", "SEVERE"):
        conf = "high" if (climate and climate.get("drought_status") in
                          ("severely_dry", "extremely_dry")) else "moderate"
        factors.append({
            "factor": "Water deficit / drought stress",
            "evidence": f"NDMI = {water.get('ndmi_mean')}, level {water.get('level')}",
            "severity": "high" if water.get("level") == "SEVERE" else "moderate",
            "attribution_confidence": conf,
            "note": ("Corroborated by SPI-3 drought class." if conf == "high"
                     else "Spectral water-stress signal; corroborate with station rainfall."),
        })

    if chloro.get("status") == "STRESSED":
        factors.append({
            "factor": "Physiological / chlorophyll decline (pre-symptomatic)",
            "evidence": f"NDRE = {chloro.get('ndre_mean')}",
            "severity": "moderate",
            "attribution_confidence": "moderate",
            "note": ("Red-edge stress typically precedes visible NDVI decline by 2-4 "
                     "weeks; possible early drought, nutrient limitation or pathogen onset."),
        })

    if disturb.get("level") in ("MODERATE", "HIGH_SEVERITY"):
        factors.append({
            "factor": "Structural disturbance (defoliation, windthrow, fire or harvest)",
            "evidence": f"NBR = {disturb.get('nbr_mean')}, level {disturb.get('level')}",
            "severity": "high" if disturb.get("level") == "HIGH_SEVERITY" else "moderate",
            "attribution_confidence": "moderate",
            "note": "Requires field verification to separate management activity from damage.",
        })

    if ndvi_h.get("status") == "ANOMALOUS" and anomaly and anomaly.get("anomaly_area_ha", 0) > 0:
        factors.append({
            "factor": "Spatially explicit canopy anomaly vs. historical baseline",
            "evidence": (f"{anomaly.get('anomaly_area_ha', 0):.1f} ha anomalous "
                         f"({anomaly.get('anomaly_fraction', 0)*100:.1f}% of forest); "
                         f"persistent={anomaly.get('persistent', False)}"),
            "severity": "high" if anomaly.get("persistent") else "moderate",
            "attribution_confidence": "moderate" if anomaly.get("persistent") else "low",
            "note": "Persistence over >=2 observations strongly raises the likelihood of a real event.",
        })

    if not factors:
        factors.append({
            "factor": "No significant stress factor detected",
            "evidence": "All indicators within expected ranges for the phenological phase.",
            "severity": "none",
            "attribution_confidence": "n/a",
            "note": "Maintain routine monitoring cadence.",
        })

    # rank: severity then confidence
    sev_rank = {"high": 0, "moderate": 1, "low": 2, "none": 3}
    factors.sort(key=lambda f: sev_rank.get(f["severity"], 4))
    return factors


# ---------------------------------------------------------------------------
# Block 3 — Drought risk interpretation
# ---------------------------------------------------------------------------

def _drought_risk(climate: Optional[Dict], eco: Dict) -> Dict[str, Any]:
    water = eco.get("water_stress") or {}
    if not climate or not climate.get("available"):
        spectral = water.get("level", "UNKNOWN")
        risk = {"SEVERE": "high", "MODERATE": "elevated"}.get(spectral, "indeterminate")
        return {
            "current_risk_level": risk,
            "basis": "spectral_only",
            "interpretation": (
                "No AEMET climate record was available this run, so drought risk is "
                f"inferred from the canopy water signal alone (NDMI level: {spectral}). "
                "Activate the AEMET integration to obtain SPI-based, attributable risk."),
            "outlook": "Not assessable without meteorological context.",
        }

    ds = climate.get("drought_status", "near_normal")
    spi = climate.get("min_spi_3")
    pa = climate.get("precip_anomaly_pct")
    risk_map = {
        "extremely_dry": "very high", "severely_dry": "high",
        "moderately_dry": "elevated", "near_normal": "low",
        "moderately_wet": "low", "very_wet": "very low", "extremely_wet": "very low",
    }
    risk = risk_map.get(ds, "indeterminate")

    coupling = ""
    if water.get("level") in ("MODERATE", "SEVERE") and ds in ("severely_dry", "extremely_dry"):
        coupling = ("The canopy water signal (NDMI) and the meteorological drought index "
                    "(SPI-3) are mutually consistent: the stress is climate-driven and "
                    "attributable with high confidence.")
    elif water.get("level") in ("MODERATE", "SEVERE") and ds in ("near_normal", "moderately_wet"):
        coupling = ("The canopy shows water stress that the rainfall record does NOT explain. "
                    "Investigate non-climatic causes (drainage, root pathogens, mixed pixels) "
                    "before attributing to drought.")

    return {
        "current_risk_level": risk,
        "basis": "spi_and_spectral",
        "min_spi_3": spi,
        "precip_anomaly_pct": pa,
        "drought_class": ds,
        "interpretation": (
            f"SPI-3 drought class '{ds}' (min SPI-3 = {spi}) implies {risk} short-term "
            f"drought risk for the beech canopy. {coupling}"),
        "outlook": (
            "If the dry signal persists into the July-August soil-moisture minimum, expect "
            "early stomatal closure and possible premature leaf drop; escalate to the "
            "intensive-monitoring protocol." if risk in ("high", "very high")
            else "Risk is currently contained; maintain the standard seasonal cadence."),
    }


# ---------------------------------------------------------------------------
# Block 4 — Visitor pressure implications
# ---------------------------------------------------------------------------

def _visitor_pressure(observation_date: date, eco: Dict, drought: Dict) -> Dict[str, Any]:
    month = observation_date.month
    autumn_peak = month in (10, 11)   # the hayedo's famous autumn-colour visitor surge
    risk = drought.get("current_risk_level", "indeterminate")
    status = eco.get("overall_status", "NORMAL")

    guidance: List[str] = []
    if autumn_peak:
        guidance.append(
            "Seasonal context: October-November is the peak visitor period (autumn "
            "leaf colour). Concentrated footfall on the guided-access trails coincides "
            "with the senescence phase when the soil and root collar are most vulnerable "
            "to compaction.")
    if risk in ("high", "very high") or status == "ALERTA":
        guidance.append(
            "Because the stand is currently under measurable stress, consider reinforcing "
            "the existing regulated-access regime: keep visitors strictly on authorised "
            "routes, and avoid authorising additional group quotas until the next "
            "observation confirms recovery.")
    else:
        guidance.append(
            "Current canopy condition does not by itself justify additional visitor "
            "restrictions beyond the standing regulated-access rules of the Reserva.")

    guidance.append(
        "Remote sensing cannot measure trampling or soil compaction directly; pair these "
        "implications with on-the-ground visitor counts and the path-condition survey.")

    return {
        "seasonal_visitor_peak": autumn_peak,
        "linked_drought_risk": risk,
        "guidance": guidance,
    }


# ---------------------------------------------------------------------------
# Block 5 — Conservation implications
# ---------------------------------------------------------------------------

def _conservation(fhi: Optional[Dict], eco: Dict, trend_slope: Optional[float]) -> Dict[str, Any]:
    implications: List[str] = []

    implications.append(
        "The Hayedo de Montejo is one of the southernmost mature beech forests in Europe "
        "and corresponds to EU Habitats Directive Annex-I beech-forest habitat "
        f"({', '.join(HABITAT_CODES.keys())}). Its favourable conservation status is a "
        "direct legal and ecological obligation of the Reserva.")

    status = eco.get("overall_status", "NORMAL")
    if status == "ALERTA":
        implications.append(
            "The current ALERTA flag means the monitored canopy condition is, this period, "
            "departing from its reference state — a signal that must be logged against the "
            "habitat's conservation-status reporting (Art. 17 Habitats Directive).")

    if trend_slope is not None:
        if trend_slope < -0.005:
            implications.append(
                f"The multi-year NDVI trend ({trend_slope:+.4f}/yr) is negative and "
                "ecologically meaningful. Sustained, it would constitute quantitative "
                "evidence of canopy decline relevant to the unfavourable-status threshold "
                "for the habitat and should trigger a management review.")
        elif trend_slope > 0.005:
            implications.append(
                f"The multi-year NDVI trend ({trend_slope:+.4f}/yr) is positive, consistent "
                "with stable-to-improving canopy condition over the monitoring window.")

    implications.append(
        "Regeneration is the key long-term conservation concern for this relict population: "
        "recurrent summer drought threatens beech seedling establishment more than mature-tree "
        "survival. Pair canopy monitoring with periodic regeneration plots.")

    return {
        "habitat_context": HABITAT_CODES,
        "implications": implications,
    }


# ---------------------------------------------------------------------------
# Block 6 — Monitoring recommendations
# ---------------------------------------------------------------------------

def _monitoring(eco: Dict, drought: Dict, baseline_summary: Optional[Dict],
                fhi: Optional[Dict]) -> List[str]:
    recs: List[str] = []
    status = eco.get("overall_status", "NORMAL")
    risk = drought.get("current_risk_level", "low")

    if status == "ALERTA" or risk in ("high", "very high"):
        recs.append("Shorten the monitoring interval: process the next Sentinel-2 pass as soon "
                    "as it is available rather than waiting for the monthly cycle.")
        recs.append("Schedule a targeted field visit to the anomalous/stressed sector to "
                    "ground-truth the spectral signal within 2-3 weeks.")
    else:
        recs.append("Maintain the standard monthly operational cadence during the Apr-Oct "
                    "active season.")

    n_annual = (baseline_summary or {}).get("n_annual_composites", 0)
    if n_annual < 3:
        recs.append(f"Priority: extend the historical baseline (currently {n_annual} annual "
                    "composites). Robust Z-score anomaly detection and a reliable phenological "
                    "component of the FHI both require >=3, ideally >=8, years.")

    if fhi and fhi.get("data_completeness", 1.0) < 0.9:
        recs.append("Improve FHI completeness by enabling the missing indicator(s) "
                    f"(used this run: {', '.join(fhi.get('indicators_used', []))}).")

    recs.append("Log this assessment in the Reserva's monitoring register so the FHI time "
                "series can itself be trend-tested over coming seasons.")
    return recs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_management_report(
    observation_date: date,
    ecological_assessment: Dict[str, Any],
    fhi: Optional[Dict[str, Any]] = None,
    climate_context: Optional[Dict[str, Any]] = None,
    anomaly: Optional[Dict[str, Any]] = None,
    trend_slope: Optional[float] = None,
    baseline_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the full six-block management/conservation decision-support payload."""
    eco = ecological_assessment or {}
    anomaly = anomaly or {}

    drought = _drought_risk(climate_context, eco)

    report = {
        "observation_date":      observation_date.isoformat(),
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "1_condition_assessment": _condition_assessment(fhi, eco),
        "2_stress_factors":       _stress_factors(eco, climate_context, anomaly),
        "3_drought_risk":         drought,
        "4_visitor_pressure":     _visitor_pressure(observation_date, eco, drought),
        "5_conservation":         _conservation(fhi, eco, trend_slope),
        "6_monitoring_recommendations": _monitoring(eco, drought, baseline_summary, fhi),
        "audience_note": (
            "This decision-support brief is written for protected-area managers. Every "
            "statement is traceable to the quantitative indicators in the technical report; "
            "remote-sensing signals complement but do not replace field inventories."),
    }
    return report


def management_markdown(report: Dict[str, Any]) -> str:
    """Render the management report as a manager-facing Markdown brief."""
    ca = report["1_condition_assessment"]
    sf = report["2_stress_factors"]
    dr = report["3_drought_risk"]
    vp = report["4_visitor_pressure"]
    cons = report["5_conservation"]

    sf_lines = "\n".join(
        f"- **{f['factor']}** (severity: {f['severity']}, attribution: {f['attribution_confidence']})\n"
        f"  - Evidence: {f['evidence']}\n  - {f['note']}"
        for f in sf)

    cons_lines = "\n".join(f"- {x}" for x in cons["implications"])
    vp_lines = "\n".join(f"- {g}" for g in vp["guidance"])
    mon_lines = "\n".join(f"{i+1}. {r}" for i, r in enumerate(report["6_monitoring_recommendations"]))

    return f"""# Management & Conservation Decision-Support Brief
## Hayedo de Montejo — Reserva de la Biosfera Sierra del Rincon
**Observation date:** {report['observation_date']}

> {report['audience_note']}

---

## 1. Forest condition assessment

**{ca['headline']}**

{ca['summary']}

## 2. Potential stress factors (ranked)

{sf_lines}

## 3. Drought risk interpretation

**Current risk level: {dr['current_risk_level']}** (basis: {dr['basis']})

{dr['interpretation']}

**Outlook:** {dr['outlook']}

## 4. Visitor pressure implications

{vp_lines}

## 5. Conservation implications

{cons_lines}

## 6. Monitoring recommendations

{mon_lines}

---
*Generated by the Hayedo de Montejo monitoring system — management decision-support module.*
"""


def save_management_report(report: Dict[str, Any], label: str) -> Path:
    MANAGEMENT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MANAGEMENT_DIR / f"management_{label}.json"
    md_path = MANAGEMENT_DIR / f"management_{label}.md"
    save_json(report, str(json_path))
    md_path.write_text(management_markdown(report), encoding="utf-8")
    logger.info("Management decision-support brief saved: %s", md_path)
    return md_path


if __name__ == "__main__":
    demo = generate_management_report(
        observation_date=date(2025, 8, 15),
        ecological_assessment={"overall_status": "AVISO",
                               "water_stress": {"level": "MODERATE", "ndmi_mean": -0.02},
                               "chlorophyll_health": {"status": "MILD_STRESS", "ndre_mean": 0.19},
                               "disturbance": {"level": "UNDISTURBED", "nbr_mean": 0.3},
                               "ndvi_health": {"status": "BELOW_EXPECTED"}},
        climate_context={"available": True, "drought_status": "severely_dry",
                         "min_spi_3": -1.7, "precip_anomaly_pct": -45,
                         "temp_anomaly_mean_c": 2.3},
        trend_slope=-0.006,
        baseline_summary={"n_annual_composites": 2},
    )
    p = save_management_report(demo, label="_selftest")
    print("management brief written to:", p)
    print("drought risk:", demo["3_drought_risk"]["current_risk_level"],
          "| stress factors:", len(demo["2_stress_factors"]))
