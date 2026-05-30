"""
Research Defensibility Audit — Phase 12.

This module adopts the adversarial stance of a critical TFM (Master's thesis)
committee member and systematically attacks the project's weakest points. For
every weakness it states the RISK (what is wrong), the IMPACT (what it does to
the validity of the conclusions), and a concrete SOLUTION (how to remove or
mitigate it). Weaknesses are organised by the five domains a committee will
probe: scientific, statistical, GIS, remote sensing, and reproducibility.

The output — the "TFM Defensibility Report" — is meant to be read BEFORE the
defence so the candidate can either fix each issue or pre-empt it with an
honest, prepared answer. Several of these weaknesses are already partially
documented in src/validation.py and in the data-quality report section; this
module consolidates them into a single self-critical audit and adds a
prioritised remediation roadmap.

Each finding carries a `status` so the audit doubles as a live checklist:
  "open"        - not yet addressed
  "mitigated"   - partially handled / acknowledged in the methodology
  "resolved"    - fully addressed in the current system
And a `severity`:
  "critical" | "major" | "moderate" | "minor"
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import WORKSPACE_ROOT, setup_logging, save_json

logger = setup_logging("defensibility")

DEFENSIBILITY_DIR = WORKSPACE_ROOT / "outputs" / "defensibility"


# ---------------------------------------------------------------------------
# The weakness catalogue
# ---------------------------------------------------------------------------

def _catalogue() -> Dict[str, List[Dict[str, str]]]:
    return {
        "Scientific weaknesses": [
            {
                "weakness": "Spectral thresholds are calibrated from literature, not from in-situ data at this site.",
                "risk": "The NDVI/NDMI/NDRE/NBR class boundaries (e.g. 'NDMI < -0.10 = severe stress') are transferred from general European beech studies, not validated against measured leaf water potential, LAI or crown condition at the Hayedo de Montejo.",
                "impact": "A committee can argue every 'stress' classification is unproven for this stand; the ecological interpretations are plausible but not locally verified.",
                "solution": "Establish >=3 permanent field plots; measure LAI (hemispherical photos / LAI-2200), crown defoliation (ICP-Forests protocol) and predawn leaf water potential in one summer campaign, and regress against the Sentinel-2 indices to recalibrate thresholds. Until then, frame thresholds explicitly as 'literature-derived, pending local validation'.",
                "severity": "major", "status": "mitigated",
            },
            {
                "weakness": "No independent ground-truth / accuracy assessment of any product.",
                "risk": "Anomaly maps, the forest mask and the FHI are presented without a confusion matrix or validation sample.",
                "impact": "Without an accuracy figure, the products have unknown reliability; this is the single most common reason remote-sensing theses are challenged.",
                "solution": "Draw an independent validation sample (stratified random points), photo-interpret on high-resolution imagery (PNOA 0.25 m) and report overall accuracy + kappa for the forest mask, and a validation of at least the anomaly hotspots in the field.",
                "severity": "critical", "status": "open",
            },
            {
                "weakness": "The Forest Health Index weights are expert-judgement, not empirically derived.",
                "risk": "The 0.30/0.25/0.20/0.15/0.10 weighting is justified ecologically but not optimised or sensitivity-tested.",
                "impact": "A reviewer can claim the headline FHI value is an artefact of arbitrary weights.",
                "solution": "Run a documented sensitivity analysis (vary each weight +/-50%, report the change in class) and, once field data exist, fit the weights to an observed crown-condition response. The module already exposes every weight transparently, which supports this analysis.",
                "severity": "moderate", "status": "mitigated",
            },
            {
                "weakness": "Causal attribution of NDVI change to 'stress' may confound phenology and condition.",
                "risk": "A low July NDVI can mean drought stress OR a shifted phenological peak; the two are not fully separated.",
                "impact": "Conclusions about 'decline' could be re-explained as phenological timing, weakening the climate-impact narrative.",
                "solution": "Acquire intra-seasonal observations (Apr-Nov) to model the full phenological curve, and report condition relative to the phenology-corrected expectation (the FHI already scores NDVI phenology-relative, which partly addresses this).",
                "severity": "major", "status": "mitigated",
            },
        ],
        "Statistical weaknesses": [
            {
                "weakness": "Baseline and trend series are far too short for the inferential tests applied.",
                "risk": "Z-score baselines and Sen's-slope/Mann-Kendall trends are computed on 3-5 annual composites; with n=3 the Mann-Kendall test has essentially no power and sigma is estimated with ~58% standard error.",
                "impact": "Any p-value or 'significant trend' statement is statistically indefensible at this series length.",
                "solution": "Extend the record with MODIS MOD13Q1 (2000-present, 250 m) and/or Landsat to build a 20+ year context; report trends only with explicit power/uncertainty caveats until n>=8 of native Sentinel-2 composites exist.",
                "severity": "critical", "status": "mitigated",
            },
            {
                "weakness": "Pixel-wise tests ignore strong spatial autocorrelation.",
                "risk": "Adjacent 10 m pixels have NDVI correlation >0.8; treating them as independent inflates Type-I error massively in anomaly and trend maps.",
                "impact": "The 'area of significant anomaly' is overstated; spatial significance claims are invalid.",
                "solution": "Apply a spatial-autocorrelation correction (effective sample size via Moran's I, or block bootstrap) before reporting significant areas; alternatively report effect sizes, not p-values, per pixel.",
                "severity": "major", "status": "open",
            },
            {
                "weakness": "Gaussian assumption behind the z-score is violated.",
                "risk": "Pixel NDVI stacks are skewed by frost/drought outliers, so Z<-2 does not correspond to the nominal 2.3% tail probability.",
                "impact": "Stated anomaly probabilities are approximate, not exact.",
                "solution": "Use non-parametric anomaly definitions (empirical percentiles) or fit a skew-appropriate distribution; report the z-score as a relative index, not a calibrated probability.",
                "severity": "moderate", "status": "mitigated",
            },
            {
                "weakness": "Temporal autocorrelation inflates confidence in inter-annual comparisons.",
                "risk": "Multi-year drought cycles make consecutive annual composites non-independent.",
                "impact": "OLS p-values on the trend are anti-conservative.",
                "solution": "Use Mann-Kendall with the Hamed-Rao autocorrelation correction and report it as the primary trend test.",
                "severity": "moderate", "status": "open",
            },
        ],
        "GIS weaknesses": [
            {
                "weakness": "The forest boundary polygon was provisional and positionally uncertain.",
                "risk": "The mask was originally derived from literature coordinates / OSM with ~50-100 m positional error, not official cartography.",
                "impact": "Edge pixels (~15%) mixed beech with scrub/pasture, biasing every forest-masked statistic and the FHI.",
                "solution": "RESOLVED: replaced with the official MFE50 layer (CNIG, Madrid prov. 28; Fagus sylvatica code 71; ~234 ha; boundary_status=OFFICIAL). This also revealed and corrected a ~6 km mislocation of the search AOI. Remaining step: an independent positional accuracy check of the MFE50 polygon against PNOA 0.25 m imagery.",
                "severity": "critical", "status": "resolved",
            },
            {
                "weakness": "No topographic correction on steep, north-facing terrain.",
                "risk": "Slopes >20 deg and low sun angles cast shadows that suppress NDVI by 0.03-0.15 without any change in vegetation.",
                "impact": "Spatial NDVI patterns and some 'anomalies' may be illumination artefacts, not ecology.",
                "solution": "Apply a C-correction or SCS+C using a DEM (PNOA-LiDAR 2 m / MDT05); document the before/after effect on the affected slopes.",
                "severity": "major", "status": "open",
            },
            {
                "weakness": "Mixed-resolution bands resampled to 10 m introduce scale inconsistency.",
                "risk": "NDMI/NDRE/NBR use 20 m bands bilinearly resampled to 10 m, so their effective resolution is coarser than NDVI's.",
                "impact": "Combining indices at a nominal 10 m overstates the spatial detail of the moisture/chlorophyll products.",
                "solution": "State the true effective resolution per index; avoid over-interpreting sub-20 m spatial patterns in NDMI/NDRE/NBR.",
                "severity": "minor", "status": "mitigated",
            },
        ],
        "Remote sensing weaknesses": [
            {
                "weakness": "No BRDF normalisation across acquisitions.",
                "risk": "Sun-sensor geometry varies seasonally; NDVI can shift by 0.05-0.12 from geometry alone, especially at 41 deg N on north slopes.",
                "impact": "Systematic seasonal bias contaminates the inter-annual trend signal.",
                "solution": "Apply a BRDF/c-factor normalisation (e.g. the Roy et al. 2016 c-factor) or restrict comparisons to a narrow, geometry-matched window.",
                "severity": "major", "status": "open",
            },
            {
                "weakness": "Residual cloud/shadow after SCL masking.",
                "risk": "Sen2Cor SCL omits/commits ~10-15% of cloud and shadow in complex mountain terrain.",
                "impact": "Composite statistics and anomalies can be driven by undetected cloud shadow rather than vegetation.",
                "solution": "Add a secondary cloud mask (s2cloudless / CloudScore+) and a temporal outlier filter in the compositing step; report the valid-pixel fraction per scene (already tracked).",
                "severity": "major", "status": "mitigated",
            },
            {
                "weakness": "NDVI saturation in dense summer canopy.",
                "risk": "At LAI 5-7 NDVI plateaus above ~0.80 and loses sensitivity.",
                "impact": "Differences between healthy and very-healthy stands are compressed; recovery may be undetectable at peak season.",
                "solution": "Use EVI/NDRE (already computed) as complementary high-LAI indicators and state NDVI's saturation explicitly.",
                "severity": "moderate", "status": "mitigated",
            },
            {
                "weakness": "Single-station, off-site climate forcing.",
                "risk": "AEMET station altitude/exposure differs from the forest; orographic precipitation and fog drip are unrepresented.",
                "impact": "SPI-based drought attribution may misstate the water balance actually experienced at the stand.",
                "solution": "Use a lapse-rate / gridded product (e.g. ERA5-Land, Spain02) at the forest elevation, or install an on-site micro-met station; report station representativeness.",
                "severity": "moderate", "status": "open",
            },
        ],
        "Reproducibility weaknesses": [
            {
                "weakness": "Credentials and secrets were stored in the config file.",
                "risk": "config/thresholds.yaml previously contained the CDSE password and a live Discord webhook in plain text.",
                "impact": "Security exposure, and the run was not reproducible by a third party without sharing secrets; committee/repository review would flag this.",
                "solution": "RESOLVED: all secrets moved to a .gitignored .env, loaded via environment variables in common.load_yaml_config (env overrides blank placeholders); .env.example template shipped and thresholds.yaml scrubbed. Residual action: rotate the previously exposed CDSE password and Discord webhook, since they existed in plain text.",
                "severity": "critical", "status": "resolved",
            },
            {
                "weakness": "No fixed random seeds / environment lock for stochastic components.",
                "risk": "The isolation-forest anomaly option and any ML step are not seeded; package versions drift.",
                "impact": "Exact numerical results may not reproduce across machines.",
                "solution": "Set and record random seeds; the pinned requirements.txt already locks versions — add a captured environment hash and the input scene IDs to each run's archive.",
                "severity": "moderate", "status": "mitigated",
            },
            {
                "weakness": "Provenance of derived products is only partially recorded.",
                "risk": "Reports cite composites but not always the exact contributing scene IDs, processing baseline (Nxxxx) and config hash.",
                "impact": "A result cannot be regenerated bit-for-bit from the report alone.",
                "solution": "Embed the full input manifest (scene IDs, dates, cloud%, config hash, code git commit) in every run archive and report header.",
                "severity": "moderate", "status": "mitigated",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Scoring / roadmap
# ---------------------------------------------------------------------------

_SEV_WEIGHT = {"critical": 4, "major": 3, "moderate": 2, "minor": 1}
_STATUS_FACTOR = {"open": 1.0, "mitigated": 0.5, "resolved": 0.0}


def _summarise(catalogue: Dict[str, List[Dict]]) -> Dict[str, Any]:
    all_findings = [f for group in catalogue.values() for f in group]
    n = len(all_findings)
    by_sev: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    residual = 0.0
    for f in all_findings:
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
        by_status[f["status"]] = by_status.get(f["status"], 0) + 1
        residual += _SEV_WEIGHT[f["severity"]] * _STATUS_FACTOR[f["status"]]
    max_residual = sum(_SEV_WEIGHT[f["severity"]] for f in all_findings)
    # Defensibility readiness: 100 = all addressed, lower = more open risk
    readiness = round(100 * (1 - residual / max_residual), 1) if max_residual else 100.0

    # Top remediation priorities: open + highest severity first
    priorities = sorted(
        [f for f in all_findings if f["status"] != "resolved"],
        key=lambda f: (-_SEV_WEIGHT[f["severity"]], 0 if f["status"] == "open" else 1),
    )[:6]

    return {
        "n_findings": n,
        "by_severity": by_sev,
        "by_status": by_status,
        "defensibility_readiness_score": readiness,
        "top_priorities": [
            {"weakness": p["weakness"], "severity": p["severity"],
             "status": p["status"], "solution": p["solution"]}
            for p in priorities
        ],
    }


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

def generate_defensibility_report() -> Dict[str, Any]:
    catalogue = _catalogue()
    summary = _summarise(catalogue)
    return {
        "title": "TFM Defensibility Report — Hayedo de Montejo Monitoring System",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "findings": catalogue,
    }


def defensibility_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# TFM Defensibility Report",
        "## Hayedo de Montejo — Sentinel-2 Forest Monitoring System",
        "",
        f"*Generated {report['generated_at'][:10]} · adopting the stance of a critical thesis committee.*",
        "",
        "> This audit attacks the project's weakest points on purpose. Each finding gives the "
        "**risk**, the **impact** on the validity of the conclusions, and a concrete **solution**. "
        "Use it to fix issues before the defence or to prepare honest, evidence-based answers.",
        "",
        "---",
        "",
        "## Readiness summary",
        "",
        f"- **Defensibility readiness score:** {s['defensibility_readiness_score']}/100 "
        "(100 = every weakness resolved; lower = more open critical risk).",
        f"- **Total findings:** {s['n_findings']}",
        f"- **By severity:** " + ", ".join(f"{k}: {v}" for k, v in sorted(
            s['by_severity'].items(), key=lambda x: -_SEV_WEIGHT.get(x[0], 0))),
        f"- **By status:** " + ", ".join(f"{k}: {v}" for k, v in s['by_status'].items()),
        "",
        "### Top remediation priorities (do these first)",
        "",
    ]
    for i, p in enumerate(s["top_priorities"], 1):
        lines.append(f"{i}. **[{p['severity'].upper()} / {p['status']}]** {p['weakness']}")
        lines.append(f"   - *Fix:* {p['solution']}")
    lines += ["", "---", ""]

    for domain, findings in report["findings"].items():
        lines += [f"## {domain}", ""]
        for j, f in enumerate(findings, 1):
            lines += [
                f"### {j}. {f['weakness']}",
                f"*Severity:* **{f['severity']}** · *Status:* **{f['status']}**",
                "",
                f"- **Risk:** {f['risk']}",
                f"- **Impact:** {f['impact']}",
                f"- **Solution:** {f['solution']}",
                "",
            ]
        lines.append("---")
        lines.append("")

    lines += [
        "## Closing note for the candidate",
        "",
        "The strongest defence is not a flawless project but a candidate who already knows every "
        "flaw and has a credible plan for each. The most urgent items above are: (1) move secrets "
        "out of the config file, (2) replace the provisional forest polygon with official MFE50 "
        "cartography, and (3) obtain at least a minimal field validation sample. Addressing these "
        "three converts the majority of critical risk into defensible, documented limitations.",
        "",
        "*Generated by the Hayedo de Montejo monitoring system — defensibility audit module.*",
    ]
    return "\n".join(lines)


def save_defensibility_report() -> Path:
    DEFENSIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_defensibility_report()
    save_json(report, str(DEFENSIBILITY_DIR / "tfm_defensibility_report.json"))
    md_path = DEFENSIBILITY_DIR / "tfm_defensibility_report.md"
    md_path.write_text(defensibility_markdown(report), encoding="utf-8")
    logger.info("TFM Defensibility Report saved: %s (readiness=%.1f)",
                md_path, report["summary"]["defensibility_readiness_score"])
    return md_path


if __name__ == "__main__":
    p = save_defensibility_report()
    print("Defensibility report:", p)
