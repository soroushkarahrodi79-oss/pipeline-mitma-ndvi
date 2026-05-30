"""
Scientific Validation Document Generator — Phase 7.

Produces a structured, peer-review-quality validation of every remote sensing
indicator used in the Hayedo de Montejo monitoring system.

For each indicator this module documents:
  1. Physical and ecological meaning
  2. Known limitations and conditions where it fails
  3. Uncertainty sources and propagation
  4. Specific interpretation for Fagus sylvatica beech forests
  5. Conditions that produce false positives / false negatives
  6. Minimum data requirements for valid interpretation
  7. Literature references

This document serves as the Methodological Annex for the TFM and should be
cited whenever results are presented to a thesis committee.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from common import WORKSPACE_ROOT, setup_logging

logger = setup_logging("validation")

VALIDATION_DIR = WORKSPACE_ROOT / "outputs" / "validation"

# ---------------------------------------------------------------------------
# Indicator scientific profiles
# ---------------------------------------------------------------------------

INDICATORS: Dict[str, Dict[str, Any]] = {

    "NDVI": {
        "full_name": "Normalized Difference Vegetation Index",
        "formula":   "(B08 - B04) / (B08 + B04)   [NIR: 842 nm, Red: 665 nm]",
        "range":     "−1.0 to +1.0; forest canopy typically 0.40–0.85",
        "introduced_by": "Rouse et al. (1974), NASA Technical Report",
        "ecological_meaning": """
NDVI measures the contrast between near-infrared reflectance (strongly scattered by
healthy leaf mesophyll) and red reflectance (strongly absorbed by chlorophyll). High
NDVI values indicate dense, actively photosynthesising canopy with high chlorophyll
content and leaf area. For *Fagus sylvatica*, NDVI is tightly coupled to:
  - Leaf Area Index (LAI): r² typically 0.82–0.94 in deciduous broadleaf forests
  - Gross Primary Production (GPP): NDVI explains ~70% of GPP variance at monthly scale
  - Foliar nitrogen content: indirect proxy via canopy greenness
At the Hayedo de Montejo (41°N, 1150–1450 m), the expected seasonal NDVI range is:
  Dormancy (Jan–Mar): 0.10–0.30 (bare branches with woody reflectance)
  Budburst (late Mar–mid Apr): 0.25–0.50 (rapid increase during leaf flush)
  Leaf expansion (mid Apr–late May): 0.40–0.68 (increasing with LAI)
  Full canopy (Jun–Sep): 0.62–0.85 (maximum photosynthetic activity)
  Senescence (Oct–Nov): 0.30–0.60 (chlorophyll degradation, leaf drop)""",

        "limitations": [
            "SATURATION at high LAI (>4 m²/m²): NDVI plateaus above ~0.80, "
            "becoming insensitive to further canopy development. This is partly "
            "why EVI was developed. For dense beech canopies (LAI 5–7 in peak summer), "
            "NDVI underestimates relative differences between healthy and very healthy stands.",

            "SOIL BACKGROUND EFFECT: At the forest edges and in canopy gaps, soil "
            "reflectance contaminates the NDVI signal. This bias depends on soil moisture "
            "and type, introducing spatial variability unrelated to canopy health. "
            "MSAVI2 partially corrects this by incorporating a soil adjustment factor.",

            "ATMOSPHERIC CONTAMINATION: Even with L2A atmospheric correction (Sen2Cor), "
            "thin aerosol residuals and adjacency effects near forest edges may bias NDVI "
            "by ±0.02–0.05 units. This is particularly relevant in summer when haze is common.",

            "BIDIRECTIONAL REFLECTANCE (BRDF): NDVI values vary by ±0.05–0.12 depending "
            "on solar zenith angle and view geometry. Sentinel-2 acquisitions at different "
            "times of year have different sun angles, introducing systematic seasonal bias "
            "in the time series. This is especially significant on north-facing slopes at "
            "41°N, where low winter sun creates long shadows.",

            "SHADOW EFFECTS: On steep north-facing terrain (>20° slope), adjacent "
            "topographic shadows reduce incoming radiation and systematically suppress "
            "observed NDVI by 0.03–0.15 units relative to flat reference surfaces. "
            "No topographic (C-correction/SCS+C) has been applied in this system.",
        ],

        "uncertainty_sources": [
            "Atmospheric correction residuals: ±0.02–0.03 NDVI",
            "SCL cloud mask commission/omission: ~10–15% in complex mountain terrain",
            "S2A/S2B spectral response function differences: ~0.005–0.01 NDVI",
            "Forest boundary polygon positional error (±50–100 m): biases edge statistics by ~5%",
            "Adjacency effects near forest edges: up to ±0.04 in edge pixels",
        ],

        "false_positives": """
Conditions that produce NDVI anomaly signals NOT caused by beech forest stress:

  1. EARLY/LATE PHENOLOGICAL YEAR: A delayed budburst (common after cold springs)
     produces low NDVI in May that mimics stress. Always interpret NDVI relative
     to the expected phenological phase for the observation date.

  2. CLOUD SHADOW CONTAMINATION: Residual cloud shadows after SCL masking reduce
     NDVI locally by 0.05–0.25. Persistent cloud shadows in mountain valleys can
     affect composites across multiple observations.

  3. TOPOGRAPHIC SHADOW: In autumn/winter, the low-angle sun creates extensive
     shadows on north-facing slopes that are NOT caused by vegetation condition.

  4. UNDERSTOREY SIGNAL: In early spring before canopy closure, the NDVI signal
     includes the understorey (herbs, shrubs), which may be phenologically different
     from the beech canopy.

  5. MIXED PIXEL EFFECT: With a provisional forest boundary, edge pixels mix beech
     forest with surrounding scrubland, causing apparent NDVI anomalies when
     adjacent land cover changes (e.g., agricultural activities).""",

        "minimum_requirements": [
            "Minimum 3 cloud-free observations per season for robust compositing",
            "SCL valid pixel fraction > 60% within forest polygon",
            "At least 3 annual composites for Z-score baseline",
            "Forest polygon with < 100 m positional uncertainty for edge-insensitive statistics",
        ],

        "beech_specific": """
For *Fagus sylvatica* at this site, NDVI has two interpretive contexts:
  1. SEASONAL: Track phenological progression. A healthy hayedo shows NDVI >0.70
     during July–September. Values below 0.60 in peak summer are anomalous.
  2. INTER-ANNUAL: Compare same-season values across years to detect trends.
     A declining trend of >0.01 NDVI/year sustained over 5+ years would indicate
     progressive canopy deterioration (comparable to documented decline rates in
     southern beech populations under climate warming: Dorado-Liñán et al. 2022).""",

        "references": [
            "Rouse et al. (1974) — original NDVI paper",
            "Tucker et al. (2005) — NDVI time series and vegetation monitoring",
            "Myneni et al. (2002) — LAI/NDVI relationship in broadleaf forests",
            "Dorado-Linán et al. (2022) — Beech decline at southern distribution limit",
        ],
    },

    "NDMI": {
        "full_name": "Normalized Difference Moisture Index",
        "formula":   "(B08 - B11) / (B08 + B11)   [NIR: 842 nm, SWIR1: 1610 nm]",
        "range":     "−1.0 to +1.0; healthy forest typically 0.10–0.40",
        "introduced_by": "Gao (1996), Remote Sensing of Environment",
        "ecological_meaning": """
NDMI exploits the sensitivity of SWIR1 reflectance (1610 nm) to liquid water content
in plant tissue. The SWIR1 band is strongly absorbed by water molecules within leaves,
so well-hydrated leaves reflect less in SWIR1 relative to NIR. NDMI therefore measures:
  - Equivalent Water Thickness (EWT, g/cm²) in the canopy
  - Plant available water content (correlation with leaf relative water content: r ~0.75)
  - Canopy interception of precipitation
  - Forest floor moisture through sparse canopy gaps
For beech forests, NDMI is a critical drought stress indicator because:
  *Fagus sylvatica* has a shallow root system (80% of biomass in upper 40 cm of soil)
  making it highly sensitive to soil moisture deficits. Stomatal closure begins when
  leaf relative water content drops below ~80%, reducing photosynthesis before NDVI
  shows any visible decline (Aranda et al. 2007).""",

        "limitations": [
            "B11 (SWIR1) is at 20 m resolution, resampled to 10 m by bilinear interpolation. "
            "This introduces spatial smoothing at the 20–10 m transition scale.",
            "SWIR1 is sensitive to atmospheric water vapour as well as leaf water. "
            "Residual atmospheric effects after Sen2Cor correction can bias NDMI by ~0.01–0.03.",
            "Canopy structure effects: dense closed canopies may show low NDMI variation "
            "even under moderate drought because the shading reduces SWIR penetration.",
            "NDMI integrates water over the entire illuminated canopy depth. It cannot "
            "distinguish between upper canopy stress and understorey conditions.",
        ],

        "false_positives": """
  1. POST-RAIN SATURATION: After heavy precipitation, surface wetness on leaves
     temporarily elevates NDMI above the typical range without indicating long-term
     improved water status. Avoid single post-precipitation acquisitions.

  2. AUTUMN LEAF SENESCENCE: During leaf colour change, chlorophyll degrades while
     water content is maintained, causing NDMI to decline while the plant is not
     under stress — this is a normal physiological process.

  3. SNOW/ICE RESIDUALS: Even after SCL filtering, partial snow cover in crevices
     or north-facing microsites suppresses NDMI, creating false drought signals
     in spring acquisitions.""",

        "beech_specific": """
Threshold values calibrated for *F. sylvatica* in montane conditions (41°N, 1150–1450 m):
  NDMI > 0.15:  Well-hydrated — no drought concern
  0.05–0.15:    Mild water stress — monitor in subsequent observations
  -0.05–0.05:   Moderate stress — stomatal closure expected, NDVI decline imminent
  < -0.05:      Severe stress — emergency response if sustained >2 observations
Studies on southern beech populations show that NDMI < 0.00 during July correlates
with >20% reduction in tree ring increment in the subsequent year (Rozas et al. 2015).""",

        "references": [
            "Gao (1996) — original NDMI paper",
            "Ceccato et al. (2001) — plant water content remote sensing",
            "Aranda et al. (2007) — water stress responses in F. sylvatica",
            "Rozas et al. (2015) — Drought and beech growth in northern Spain",
        ],
    },

    "NDRE": {
        "full_name": "Red-Edge Normalized Difference Vegetation Index",
        "formula":   "(B8A - B05) / (B8A + B05)   [NIR-narrow: 865 nm, Red-edge: 705 nm]",
        "range":     "−1.0 to +1.0; healthy canopy typically 0.20–0.55",
        "introduced_by": "Gitelson & Merzlyak (1994), Journal of Photochemistry and Photobiology",
        "ecological_meaning": """
The red-edge region (680–750 nm) is the spectral inflection point between the strong
red chlorophyll absorption and the high NIR reflectance plateau. The position and
amplitude of this feature is highly sensitive to chlorophyll content per unit leaf area
(Cab), which is the single most important biochemical indicator of photosynthetic
capacity and plant nitrogen status.

NDRE using Sentinel-2 bands B05 (705 nm) and B8A (865 nm) specifically:
  - Saturates at a much higher chlorophyll content than standard NDVI
  - Responds to chlorophyll changes that are invisible in NDVI
  - Detects physiological stress 2–4 weeks BEFORE structural changes appear in NDVI
  - Provides complementary information to NDVI in dense forest canopies

Key scientific value: NDRE is an early-warning indicator. When NDVI is still normal
but NDRE is declining, the plant is experiencing sub-clinical stress that will likely
manifest as NDVI decline in subsequent observations. This 2–4 week lead time is
critical for management response.""",

        "limitations": [
            "B05 (705 nm) and B8A (865 nm) are at 20 m spatial resolution, "
            "resampled to 10 m. Edge effects at forest boundaries are more pronounced.",
            "NDRE is not as well-calibrated as NDVI — fewer published threshold values "
            "for specific species/sites. Thresholds used in this system are derived from "
            "general European beech literature and require site-specific validation.",
            "High sensitivity means NDRE is also noisier — small atmospheric effects "
            "or sensor calibration differences have relatively larger impacts on NDRE "
            "than on NDVI.",
        ],

        "false_positives": """
  1. NUTRIENT LIMITATION (N, Mg): Deficiencies in nitrogen or magnesium cause
     chlorophyll decline (yellowish leaves) that manifests as low NDRE without
     being caused by drought or temperature stress. In the hayedo context, soil
     acidification from atmospheric N deposition may cause this signal.

  2. LATE-SEASON YELLOWING: Natural autumn senescence begins with chlorophyll
     breakdown (NDRE decline) while water content is maintained. Low autumn NDRE
     is normal phenological behaviour, not stress.

  3. UNDERSTOREY SIGNAL: Like NDVI, NDRE integrates contributions from the forest
     floor if the canopy is not fully closed. Understorey species may have different
     chlorophyll status than the dominant beech.""",

        "beech_specific": """
For *F. sylvatica* in this system:
  NDRE > 0.35: Optimal chlorophyll content — no concern
  0.25–0.35:   Normal range for leaf expansion phase
  0.18–0.25:   Mild physiological stress — schedule follow-up observation
  < 0.18:      Significant chlorophyll deficit — investigate cause within 2 weeks
The unique value of NDRE for the TFM: it can provide evidence of sub-clinical stress
that would not appear in standard NDVI analysis, strengthening the scientific
narrative of the monitoring system's sensitivity.""",

        "references": [
            "Gitelson & Merzlyak (1994) — original red-edge spectral indices",
            "Delegido et al. (2011) — Sentinel-2 red-edge bands and chlorophyll",
            "Frampton et al. (2013) — Evaluating Sentinel-2 indices for forest monitoring",
        ],
    },

    "NBR": {
        "full_name": "Normalized Burn Ratio",
        "formula":   "(B08 - B12) / (B08 + B12)   [NIR: 842 nm, SWIR2: 2190 nm]",
        "range":     "−1.0 to +1.0; undisturbed forest typically 0.20–0.55",
        "introduced_by": "Key & Benson (2006), FIREMON: Fire Effects Monitoring",
        "ecological_meaning": """
NBR was originally developed for fire severity assessment, but in non-fire contexts
it serves as a powerful indicator of general canopy disturbance. B12 (SWIR2, 2190 nm)
is sensitive to both leaf water content and carbon compounds in vegetation and soil.
Changes in NBR reflect:
  - Fire damage (char, loss of canopy and leaf water)
  - Windthrow events (canopy opening, increased soil and slash reflectance)
  - Defoliation by insects or pathogens (reduced NIR reflectance)
  - Selective logging or storm damage
  - Progressive canopy thinning from competition or decline
For beech forests specifically, NBR is valuable for detecting:
  - Post-storm damage (common in the Sierra del Rincón due to north-facing exposure)
  - Beech bark disease progression (Neonectria + Cryptococcus felgens complex)
  - Phytophthora root rot (Phytophthora plurivora — increasingly common in Spain)""",

        "limitations": [
            "B12 (SWIR2) at 20 m resolution, resampled to 10 m.",
            "Not sensitive to subtle, diffuse stress — primarily detects structural changes.",
            "Confusion with soil moisture patterns: dry soils in gaps and clearings "
            "increase SWIR2 reflectance, decreasing NBR without actual disturbance.",
        ],

        "false_positives": """
  1. DRY YEARS: Low soil moisture in drought years increases soil SWIR2 reflectance
     in canopy gaps, decreasing NBR even when the forest is structurally intact.

  2. HARVEST OR MANAGEMENT INTERVENTIONS: Any authorised management activity
     (thinning, path maintenance) produces NBR reduction that could be interpreted
     as disturbance.""",

        "beech_specific": """
Monitoring thresholds for *F. sylvatica* (adapted from USGS burn severity guidelines):
  NBR > 0.20: Undisturbed intact canopy
  0.10–0.20:  Low-severity disturbance (minor structural change)
  0.00–0.10:  Moderate disturbance (requires field investigation)
  < 0.00:     High-severity disturbance (urgent response needed)
NBR is particularly important for the TFM because it provides independent validation
of structural changes detected by NDVI, separating physiological stress (NDVI change
without NBR change) from structural damage (both NDVI and NBR change).""",

        "references": [
            "Key & Benson (2006) — NBR for burn severity",
            "Dalmayne et al. (2013) — NBR for non-fire disturbance detection",
        ],
    },

    "SPI": {
        "full_name": "Standardised Precipitation Index",
        "formula":   "Normal quantile transform of Gamma-fitted precipitation accumulation",
        "range":     "Approximately −3 to +3; |SPI| > 2 = extreme event",
        "introduced_by": "McKee et al. (1993), 8th Conference on Applied Climatology",
        "ecological_meaning": """
SPI expresses precipitation departure from the long-term normal in standard deviation
units, after fitting a Gamma distribution to the historical precipitation series. It is:
  - Scale-flexible: SPI-3 captures short-term drought; SPI-12 captures hydrological drought
  - Probabilistic: SPI = -1.5 means the precipitation is in the lowest ~7% of historical values
  - Ecologically relevant: soil moisture deficit (and thus plant water stress) is closely
    coupled to SPI-3 in the first growing season months
For *F. sylvatica* at the southern distribution limit:
  - SPI-3 for June-August is the most relevant indicator of summer water deficit
  - SPI-12 captures the cumulative multi-year drought that causes progressive decline
  - Studies show SPI-6 (winter-spring accumulation) predicts beech productivity
    more strongly than summer SPI alone (Cavin & Jump 2017)""",

        "limitations": [
            "SPI is computed from station data, not from the forest itself. "
            "The nearest AEMET station may be at a different elevation and exposure "
            "than the hayedo (potential gradient of 50–200 mm precipitation/100 m altitude).",
            "SPI assumes stationarity of the precipitation distribution — this assumption "
            "is increasingly violated under climate change, which shifts the mean and variance "
            "of precipitation in the Mediterranean.",
            "Short station records (< 30 years) reduce the statistical reliability "
            "of the Gamma distribution fit.",
        ],

        "false_positives": """
  1. HIGH-ALTITUDE GRADIENT: SPI computed from valley stations may underestimate
     the actual water deficit on mountain slopes if orographic precipitation creates
     a different climatology at the forest elevation.

  2. REDISTRIBUTION EFFECTS: Fog drip and lateral water movement in the sierra can
     partially compensate for precipitation deficit, making the actual soil moisture
     at the forest higher than SPI would suggest.""",

        "beech_specific": """
SPI interpretation for the Hayedo de Montejo:
  SPI ≥ -1.0:   Normal to wet — no drought concern
  -1.5 to -1.0: Moderate drought — monitor NDMI response
  -2.0 to -1.5: Severe drought — NDMI decline expected; alert protocols
  < -2.0:        Extreme drought — high risk of permanent canopy damage
At this site, extreme drought events (SPI < -2.0) are projected to increase from
~1 per decade to ~3 per decade by 2100 under RCP8.5 (Sánchez-Salguero et al. 2017).""",

        "references": [
            "McKee et al. (1993) — SPI definition",
            "Mishra & Singh (2010) — Review of drought indices",
            "Cavin & Jump (2017) — Drought and beech vitality at distribution limits",
            "Sánchez-Salguero et al. (2017) — Climate projections for Iberian beech",
        ],
    },

    "CLIMATE_ANOMALY": {
        "full_name": "Climatic Anomalies (Temperature & Precipitation vs. 1991-2020 Normal)",
        "formula":   "anomaly = observed_monthly_value - WMO_normal(month); precip also as % departure",
        "range":     "Temperature: approx -5 to +5 C; Precipitation: -100% to +200% of normal",
        "introduced_by": "WMO climatological normal framework (WMO No. 1203, 2017)",
        "ecological_meaning": """
Climatic anomalies express how far the temperature and precipitation of the observation
period depart from the long-term climatological normal (the WMO standard 1991-2020 period).
They are the external forcing that drives, and therefore helps attribute, the vegetation
signals measured spectrally:
  - A positive temperature anomaly raises vapour-pressure deficit and evapotranspiration
    demand; for shallow-rooted Fagus sylvatica this can exceed water supply and trigger
    stomatal closure even when soil moisture is only moderately low.
  - A negative precipitation anomaly during the spring-summer recharge period propagates,
    with a lag of weeks to months, into the soil-moisture deficit that the canopy
    experiences (captured spectrally by NDMI).
  - A negative spring temperature anomaly can indicate a late-frost event capable of
    damaging newly flushed leaves, producing a NDVI/NDRE depression in May-June that is
    climatic, not pathological, in origin.
The principal value of the climatic anomaly for this system is ATTRIBUTION: it allows a
canopy stress signal to be assigned (or not) to a concurrent climatic cause, separating
climate-driven stress from structural disturbance.""",

        "limitations": [
            "STATION REPRESENTATIVENESS: anomalies are computed from the nearest AEMET "
            "station, which differs in altitude and exposure from the forest. Orographic "
            "precipitation gradients (50-200 mm/100 m) and fog drip mean the anomaly at the "
            "station is not exactly the anomaly experienced by the stand.",
            "NORMAL-PERIOD DEPENDENCE: the magnitude of every anomaly depends on the chosen "
            "reference period. Using 1991-2020 (already a warming baseline) understates "
            "warm anomalies relative to a mid-20th-century reference.",
            "APPROXIMATE REGIONAL NORMALS: where station normals are incomplete the system "
            "falls back to approximate regional monthly normals, adding uncertainty of a few "
            "tenths of a degree and several mm to the anomaly values.",
            "MONTHLY AGGREGATION: monthly anomalies can mask damaging short-lived extremes "
            "(a single heatwave or a late-frost night) that drive the ecological response.",
        ],

        "uncertainty_sources": [
            "Station-to-forest altitudinal transfer: +/- 0.5-1.0 C and +/- 10-30 mm",
            "Incomplete station normals replaced by regional approximations: +/- 0.3 C, +/- 10 mm",
            "Data gaps in AEMET monthly series reducing the robustness of the normal",
        ],

        "false_positives": """
  1. SINGLE-STATION OUTLIER: an instrument error or a localized convective storm at the
     station can create an apparent anomaly that the forest never experienced.
  2. ALTITUDE MISMATCH: a warm anomaly recorded at a valley station may be partly an
     inversion effect not present at the forest elevation.
  3. NORMAL-PERIOD ARTEFACT: comparing against a normal that itself contained anomalous
     years can make an ordinary period appear anomalous.""",

        "beech_specific": """
For the Hayedo de Montejo the ecologically critical climatic anomalies are:
  - Summer (Jun-Aug) precipitation deficit: the dominant driver of canopy water stress at
    this dry range margin; a deficit > 40% of normal is a strong stress predictor.
  - Summer temperature anomaly > +2 C: compounds the water deficit via increased
    evaporative demand and is associated with reduced ring increment the following year.
  - Late-spring (Apr-May) cold anomaly: flags potential late-frost damage to new foliage.
Under RCP8.5 the frequency of hot-dry summer anomalies is projected to rise markedly at
this site (Sanchez-Salguero et al. 2017), making the anomaly series central to the
climate-vulnerability narrative of the TFM.""",

        "references": [
            "WMO (2017) — WMO Guidelines on the Calculation of Climate Normals (No. 1203)",
            "Sanchez-Salguero et al. (2017) — Climate projections for Iberian beech",
            "Allen et al. (2010) — Global overview of drought- and heat-induced tree mortality",
        ],
    },

    "Z_SCORE": {
        "full_name": "Phenologically-Normalised Z-score Anomaly",
        "formula":   "Z = (NDVI_current - μ_historical) / σ_historical",
        "range":     "Approximately −4 to +4; |Z| > 2 indicates significant anomaly",
        "introduced_by": "Adapted from Zeng et al. (2013) and Peters et al. (2002)",
        "ecological_meaning": """
The Z-score converts absolute NDVI values into standardised departures from the
pixel-specific historical mean, expressed in units of standard deviation (σ). This:
  - Removes the effect of persistent spatial heterogeneity (e.g., a consistently
    lower-NDVI area near a rocky outcrop will always appear yellow on a raw NDVI map
    but will appear normal on a Z-score anomaly map)
  - Focuses the analysis on CHANGES relative to local historical behaviour
  - Enables direct comparison between phenologically similar periods across years
  - Provides a statistically principled threshold: Z < -2.0 has a theoretical
    probability of ~2.3% under normal conditions (assuming Gaussian distribution)
For beech forest monitoring, the Z-score answers: "How unusual is this observation
relative to what this specific forest patch has done historically at this time of year?"
This is a fundamentally different and more ecologically meaningful question than
"Is this NDVI value below a fixed threshold?".""",

        "limitations": [
            "GAUSSIAN ASSUMPTION: NDVI distributions within a forest pixel stack are "
            "often moderately skewed (skewness ±0.2–0.8) due to phenological outliers "
            "(frost events, very dry years). This means the theoretical Z-score probabilities "
            "are approximate, not exact. Mann-Whitney or Kolmogorov-Smirnov tests should "
            "be used for hypothesis testing rather than Z-score thresholds alone.",

            "STATIONARITY: The Z-score baseline assumes that the mean and variance of "
            "NDVI at each pixel are stationary over the baseline period. Under progressive "
            "climate change, this assumption fails — a warming climate systematically "
            "shifts the baseline upward (greening) or downward (browning), causing the "
            "Z-score to underestimate anomalies relative to a more distant historical mean.",

            "MINIMUM SAMPLE SIZE: With only 3 annual composites, σ is estimated from "
            "only 3 observations. The standard error of σ with n=3 is ~58% of σ itself, "
            "making the Z-score highly uncertain. Reliable Z-scores require n ≥ 10.",

            "SPATIAL CORRELATION: Adjacent forest pixels have highly correlated NDVI "
            "time series (correlation ≥ 0.8 at 20m distance). Treating each pixel as "
            "independent in significance tests leads to massive Type I error inflation "
            "(false positives). Spatial autocorrelation adjustment is required for "
            "formal hypothesis testing.",
        ],

        "false_positives": """
  1. NEW-YEAR ANOMALY: If the most recent year has an extreme value that was not
     represented in the (short) baseline, the Z-score will be extreme even if the
     observation is within natural variability at longer time scales.

  2. BASELINE DOMINATED BY DROUGHT YEARS: If the baseline period (e.g., 2021-2025)
     happened to include multiple drought years, the "normal" baseline NDVI will be
     artificially low, making subsequent recovery appear anomalously high (positive Z).""",

        "beech_specific": """
Z-score interpretation for *F. sylvatica* with ≥5 year baseline:
  Z ≥ 1.0:   Above-average canopy activity — favourable conditions
  0 to 1.0:  Near-normal — within expected inter-annual variability
  -1.0 to 0: Slightly below normal — monitor
  -2.0 to -1.0: Moderate anomaly — warranting investigation
  < -2.0:     Statistically significant anomaly — strong evidence of unusual condition
With only 3 years of baseline (minimum required), treat Z < -1.5 as "anomalous"
rather than the standard -2.0 threshold, to account for baseline uncertainty.""",

        "references": [
            "Peters et al. (2002) — Drought monitoring with satellite data",
            "Zeng et al. (2013) — Standardised anomaly index for vegetation",
            "Anderson et al. (2016) — Satellite-based drought monitoring review",
        ],
    },

    "TREND_ANALYSIS": {
        "full_name": "Pixel-wise Annual NDVI Trend (Sen's Slope + Mann-Kendall)",
        "formula":   "slope = median((NDVI_j - NDVI_i)/(year_j - year_i)) for all pairs i < j",
        "range":     "NDVI/year, typically −0.05 to +0.05",
        "introduced_by": "Theil (1950), Sen (1968); Mann-Kendall: Mann (1945)",
        "ecological_meaning": """
The Sen's slope estimator computes the median of all pairwise slopes between
observations, providing a robust linear trend that is resistant to outliers.
Combined with the Mann-Kendall non-parametric monotonic trend test, it provides:
  - The MAGNITUDE of change (NDVI units per year)
  - The SIGNIFICANCE of the trend (p-value from Mann-Kendall τ)
  - Resistance to single-year outliers (unlike OLS regression)
Ecological interpretation at the landscape scale:
  Positive slope (greening): potential LAI increase, CO2 fertilisation,
    management improvement, or phenological extension
  Negative slope (browning): canopy decline, progressive stress,
    management degradation, or phenological shortening
For the TFM and CETS, a statistically significant (p < 0.05) negative trend of
>0.005 NDVI/year sustained over the monitoring period would constitute quantitative
evidence of ecosystem deterioration requiring management action.""",

        "limitations": [
            "TEMPORAL AUTOCORRELATION: Annual NDVI composites from the same phenological "
            "window are positively autocorrelated (adjacent years tend to be similar due "
            "to multi-year drought cycles). OLS p-values are invalid; Mann-Kendall is more "
            "appropriate but still affected by autocorrelation in short series.",

            "MINIMUM SAMPLE SIZE: With n=3 observations, the Mann-Kendall test has only 3 "
            "possible outcomes (τ = -1, 0, or 1) and essentially no statistical power. "
            "Minimum n=8 is recommended for reliable trend detection at p < 0.05.",

            "CONFOUNDED SIGNALS: A trend in the annual composite can reflect changes in "
            "phenological timing (earlier/later peak) rather than changes in forest condition. "
            "A forest with earlier peak greening may appear to have higher July NDVI simply "
            "because it has already reached peak while in previous years it had not yet done so.",

            "PIXEL INDEPENDENCE: Sen's slope computed pixel-by-pixel on correlated spatial "
            "data inflates the apparent number of significant pixels. Spatial filtering "
            "or field-based validation is required before interpreting spatial trend patterns.",
        ],

        "false_positives": """
  1. PHENOLOGICAL ADVANCEMENT: Under warming climate, earlier budburst means the
     July-September window captures later-season values (lower NDVI due to earlier
     senescence onset), producing a spurious negative trend.

  2. CLOUD MASKING INCONSISTENCY: Inconsistent cloud mask quality across years creates
     apparent NDVI trends caused by changing pixel valid fractions, not real vegetation change.

  3. BASELINE LENGTH EFFECT: With only 3-5 years, a single extreme year dominates the
     trend estimate. A single drought year at year 3 creates a strong negative trend
     that disappears when year 4 (recovery) is included.""",

        "beech_specific": """
Minimum requirements for defensible trend statements in the TFM:
  - ≥ 8 annual composites for Mann-Kendall significance at p < 0.05 with 80% power
  - Phenological consistency: all composites from same window (Jul 1 – Sep 30)
  - Correction for BRDF effects to remove systematic seasonal bias
  - Acknowledgment of Iberian Peninsula greening trend (MODIS ~+0.002 NDVI/year
    2000-2020) as the null hypothesis against which the site trend is compared""",

        "references": [
            "Sen (1968) — Sen's slope estimator",
            "Mann (1945) — Kendall rank correlation",
            "Fensholt & Proud (2012) — Assessing global vegetation greening trends",
        ],
    },

    "PHENOLOGY_METRICS": {
        "full_name": "Phenological Event Detection (SOS, PEAK, EOS, GSL)",
        "formula":   "Threshold-based or double logistic curve fitting to NDVI time series",
        "range":     "SOS: DOY 80-130; PEAK: DOY 200-250; EOS: DOY 270-320 (at 41°N, 1300m)",
        "introduced_by": "Reed et al. (1994), Journal of Vegetation Science",
        "ecological_meaning": """
Phenological metrics track the timing of key biological events in the annual cycle:
  SOS (Start of Season): Green-up / budburst — marks beginning of carbon assimilation
  PEAK: Maximum canopy development — peak of ecosystem productivity
  EOS (End of Season): Onset of senescence — end of primary growing season
  GSL (Growing Season Length): EOS - SOS — total duration of active carbon uptake

These metrics are among the most sensitive biological indicators of climate change
because phenological timing in temperate trees is primarily driven by temperature
accumulation (Growing Degree Days) and photoperiod.

For the TFM/CETS argument, documented phenological shifts provide:
  - Direct evidence of climate change impacts at this specific site
  - Comparison material against pan-European phenological datasets
  - Carbon balance implications (longer GSL → greater ecosystem carbon uptake)
  - Frost vulnerability assessment (earlier SOS → higher late-frost exposure)""",

        "limitations": [
            "SPARSE TEMPORAL SAMPLING: Annual Jul-Sep composites provide only the peak-season "
            "value, making it impossible to detect SOS or EOS from these alone. Full "
            "phenological analysis requires monthly observations throughout Apr-Nov.",

            "INTER-YEAR CONFOUNDING: The annual composite averages over a 90-day window. "
            "If peak NDVI occurs on DOY 200 in one year and DOY 230 in another, the same "
            "Jul-Sep composite will capture different stages of the phenological curve, "
            "making inter-year NDVI comparisons ambiguous.",

            "ALTITUDE GRADIENT: Phenology within the ~300m altitudinal range of the hayedo "
            "spans approximately 10-15 days (higher elevation = later budburst). A single "
            "composite does not resolve this within-forest phenological gradient.",
        ],

        "false_positives": """
  1. EARLY SENESCENCE DROUGHT: Severe summer drought can trigger premature leaf
     drop in August, appearing as early EOS and short GSL — this IS a stress signal,
     not a false positive, but it must be distinguished from normal year-to-year variation.

  2. PHENOLOGICAL DELAY VS. DAMAGE: A late SOS can mean either (a) cold spring
     (normal climatic variation) or (b) frost damage to emerging leaves (stress signal
     requiring different management response).""",

        "beech_specific": """
Expected phenological calendar for *F. sylvatica* at Hayedo de Montejo (41°N, 1300m):
  SOS (Budburst):   DOY 90-120  (late March to early May)
                    → Sensitive to spring temperature anomalies
  PEAK:             DOY 200-240 (mid-July to late August)
                    → Controlled by summer water availability
  EOS (Leaf fall):  DOY 280-310 (early to mid-November)
                    → Controlled by photoperiod and autumn temperature
  GSL:              ~180-210 days under current climate
Projected change under RCP8.5 (+4°C by 2100): SOS 15-25 days earlier,
EOS unchanged or slightly delayed → GSL increase of 15-25 days, but with
dramatically increased late-frost exposure and summer drought stress.""",

        "references": [
            "Reed et al. (1994) — Phenological event detection from NDVI",
            "Menzel et al. (2006) — European phenological changes",
            "Vitasse et al. (2009) — Iberian beech phenology",
            "Perez-Ramos et al. (2020) — Phenology climate change southern Spain",
        ],
    },
}


# ---------------------------------------------------------------------------
# Document generator
# ---------------------------------------------------------------------------

def generate_validation_document(
    current_stats: Dict[str, Any] = None,
    observation_date: str = None,
) -> str:
    """
    Generate the complete scientific validation document as Markdown.

    Parameters
    ----------
    current_stats : dict mapping indicator name to computed statistics
    observation_date : ISO date string of the current observation
    """
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    obs   = observation_date or now

    lines = [
        "# DOCUMENTO DE VALIDACIÓN CIENTÍFICA",
        "## Indicadores de Teledetección — Hayedo de Montejo",
        "",
        f"**Versión:** 1.0 | **Fecha:** {now} | **Observación analizada:** {obs}",
        "",
        "> Este documento describe la base científica, limitaciones y restricciones",
        "> interpretativas de cada indicador utilizado en el sistema de monitorización.",
        "> Constituye el Anexo Metodológico del TFM y debe ser consultado antes de",
        "> interpretar cualquier resultado cuantitativo del sistema.",
        "",
        "---",
        "",
        "## Índice de Indicadores",
        "",
    ]

    for i, key in enumerate(INDICATORS, 1):
        ind = INDICATORS[key]
        lines.append(f"{i}. [{ind['full_name']}](#{key.lower()}) — {key}")

    lines += ["", "---", ""]

    for key, ind in INDICATORS.items():
        lines += [
            f"## {ind['full_name']}",
            f"**Símbolo:** `{key}` | **Introducido por:** {ind['introduced_by']}",
            "",
            f"**Fórmula:** `{ind['formula']}`",
            f"**Rango de valores:** {ind['range']}",
            "",
            "### 1. Significado ecológico",
            "",
            ind["ecological_meaning"].strip(),
            "",
            "### 2. Limitaciones conocidas",
            "",
        ]
        for i, lim in enumerate(ind.get("limitations", []), 1):
            lines.append(f"{i}. {lim.strip()}")
        lines.append("")

        lines += [
            "### 3. Incertidumbre",
            "",
        ]
        for src in ind.get("uncertainty_sources", ["Documentada en limitaciones."]):
            lines.append(f"- {src}")
        lines.append("")

        lines += [
            "### 4. Interpretación específica para *Fagus sylvatica*",
            "",
            ind.get("beech_specific", "Ver sección de significado ecológico.").strip(),
            "",
            "### 5. Falsos positivos y condiciones de interpretación errónea",
            "",
            ind.get("false_positives", "Documentados en limitaciones.").strip(),
            "",
        ]

        refs = ind.get("references", [])
        if refs:
            lines += ["### 6. Referencias científicas", ""]
            for ref in refs:
                lines.append(f"- {ref}")
            lines.append("")

        # Add current observation context if stats available
        if current_stats and key in current_stats:
            s = current_stats[key]
            lines += [
                "### 7. Valores observados en esta ejecución",
                "",
                f"| Métrica | Valor |",
                f"|---------|-------|",
            ]
            for mk, mv in s.items():
                if isinstance(mv, float):
                    lines.append(f"| {mk} | {mv:.4f} |")
                elif mv is not None:
                    lines.append(f"| {mk} | {mv} |")
            lines.append("")

        lines += ["---", ""]

    lines += [
        "## Nota sobre el polígono forestal provisional",
        "",
        "Todas las estadísticas de índices espectrales se calculan sobre los píxeles",
        "contenidos dentro del polígono forestal definido en `config/forest_boundary.geojson`.",
        "",
        "El polígono actual es **PROVISIONAL** — derivado de coordenadas de la literatura",
        "y verificación visual, con una incertidumbre posicional estimada de ±50–100 m.",
        "",
        "**Impacto:** Los píxeles de borde (~15% del total) pueden incluir vegetación de",
        "matorral o herbácea que no corresponde al hayedo sensu stricto. Esto diluta las",
        "estadísticas espectrales hacia valores menos característicos del haya.",
        "",
        "**Solución:** Reemplazar con el polígono oficial del MFE50 (Mapa Forestal de España",
        "1:50.000), disponible en el CNIG para la provincia de Madrid (unidades cartográficas",
        "con especie principal SPE1='Fs' — *Fagus sylvatica*).",
        "",
        "---",
        "",
        "*Documento generado automáticamente por el pipeline de monitorización del Hayedo de Montejo v3.0*",
        f"*Generado el {now}*",
    ]

    return "\n".join(lines)


def save_validation_document(current_stats=None, observation_date=None) -> Path:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    content = generate_validation_document(current_stats, observation_date)
    path = VALIDATION_DIR / "scientific_validation_document.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Scientific validation document saved: %s", path)
    return path
