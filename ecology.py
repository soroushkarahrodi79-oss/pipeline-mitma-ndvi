"""
Ecological interpretation engine for Fagus sylvatica monitoring.

Translates spectral index statistics into ecologically meaningful assessments
calibrated for the Hayedo de Montejo (Sierra del Rincón, 41°N, ~1200–1450 m a.s.l.).

Scientific basis:
  - Phenological calendar: derived from literature on F. sylvatica phenology
    at comparable altitudes in the Iberian Peninsula (Menzel et al. 2006;
    Vitasse et al. 2009; Chmielewski & Rötzer 2001)
  - Index thresholds: calibrated from published European beech spectral studies
    and ESA Sentinel-2 vegetation monitoring guidelines
  - Drought stress indicators: based on NDMI-NDVI combined approach
    (Gao 1996; Ceccato et al. 2001)
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Phenological calendar for F. sylvatica at ~41°N, ~1300 m a.s.l.
# DOY = Day of Year
# ---------------------------------------------------------------------------

PHENOLOGY = {
    "dormancy":        (1,   80),    # Jan–Mar: complete dormancy
    "budburst":        (81,  110),   # late Mar – mid Apr: bud swelling & break
    "leaf_expansion":  (111, 149),   # mid Apr – late May: rapid leaf area increase
    "full_canopy":     (150, 258),   # Jun–mid Sep: peak photosynthetic activity
    "senescence_early":(259, 289),   # mid Sep – mid Oct: chlorophyll degradation begins
    "senescence_late": (290, 319),   # mid Oct – mid Nov: leaf coloration & drop
    "post_senescence": (320, 365),   # late Nov – Dec: bare canopy
}

PHENOLOGY_LABELS = {
    "dormancy":         "Dormancia invernal",
    "budburst":         "Brotación / apertura de yemas",
    "leaf_expansion":   "Expansión foliar",
    "full_canopy":      "Dosel completo — máxima actividad fotosintética",
    "senescence_early": "Inicio de senescencia",
    "senescence_late":  "Senescencia avanzada — coloración otoñal",
    "post_senescence":  "Post-senescencia — dosel desnudo",
}

# Expected NDVI ranges per phenological phase (healthy forest)
NDVI_EXPECTED = {
    "dormancy":         (0.10, 0.30),
    "budburst":         (0.25, 0.50),
    "leaf_expansion":   (0.40, 0.68),
    "full_canopy":      (0.62, 0.85),
    "senescence_early": (0.48, 0.72),
    "senescence_late":  (0.30, 0.60),
    "post_senescence":  (0.12, 0.32),
}

# NDMI thresholds for water stress assessment
NDMI_THRESHOLDS = {
    "well_hydrated":    0.10,
    "mild_stress":      0.00,
    "moderate_stress": -0.10,
    # below -0.10 = severe drought stress
}

# NDRE thresholds for chlorophyll / plant health
NDRE_THRESHOLDS = {
    "healthy":          0.22,
    "mild_stress":      0.16,
    # below 0.16 = significant physiological stress
}

# NBR thresholds for disturbance / fire impact
NBR_THRESHOLDS = {
    "undisturbed":      0.20,
    "low_severity":     0.10,
    "moderate":         0.00,
    # below 0.00 = high severity disturbance
}


# ---------------------------------------------------------------------------
# Core interpretation functions
# ---------------------------------------------------------------------------

def get_phenological_phase(observation_date: date) -> str:
    doy = observation_date.timetuple().tm_yday
    for phase, (start, end) in PHENOLOGY.items():
        if start <= doy <= end:
            return phase
    return "dormancy"


def assess_ndvi_health(ndvi_median: float, phase: str) -> Dict[str, Any]:
    """
    Compare observed NDVI against the expected range for the phenological phase.
    Returns a structured health assessment.
    """
    low, high = NDVI_EXPECTED.get(phase, (0.0, 1.0))
    expected_mid = (low + high) / 2

    if ndvi_median >= high:
        status = "EXCELLENT"
        interpretation = (
            f"El NDVI observado ({ndvi_median:.3f}) supera el rango superior esperado "
            f"({low:.2f}–{high:.2f}) para la fase de {PHENOLOGY_LABELS.get(phase, phase)}. "
            f"Indica una vitalidad excepcional del dosel de haya, posiblemente asociada "
            f"a condiciones hídricas y lumínicas favorables en este período."
        )
    elif ndvi_median >= low:
        deviation_pct = 100 * (ndvi_median - expected_mid) / expected_mid
        status = "NORMAL"
        interpretation = (
            f"El NDVI observado ({ndvi_median:.3f}) se encuentra dentro del rango "
            f"esperado ({low:.2f}–{high:.2f}) para la fase de {PHENOLOGY_LABELS.get(phase, phase)}. "
            f"La vitalidad del dosel es consistente con el comportamiento fenológico "
            f"histórico de Fagus sylvatica en este emplazamiento."
        )
    elif ndvi_median >= low * 0.8:
        status = "BELOW_EXPECTED"
        deficit_pct = 100 * (low - ndvi_median) / low
        interpretation = (
            f"El NDVI observado ({ndvi_median:.3f}) se sitúa un {deficit_pct:.1f}% por debajo "
            f"del límite inferior del rango esperado ({low:.2f}–{high:.2f}) para "
            f"{PHENOLOGY_LABELS.get(phase, phase)}. Este déficit moderado puede indicar: "
            f"(1) retraso fenológico respecto al promedio histórico; "
            f"(2) estrés hídrico o térmico subclínico; "
            f"(3) presencia de zonas de claros o bordes en el AOI analizado. "
            f"Se recomienda seguimiento en la siguiente adquisición disponible."
        )
    else:
        status = "ANOMALOUS"
        deficit_pct = 100 * (low - ndvi_median) / low
        interpretation = (
            f"El NDVI observado ({ndvi_median:.3f}) presenta un déficit significativo "
            f"({deficit_pct:.1f}% por debajo del mínimo esperado de {low:.2f}) "
            f"para la fase de {PHENOLOGY_LABELS.get(phase, phase)}. Este nivel de déficit "
            f"es ecológicamente relevante y puede ser indicativo de: "
            f"(1) estrés hídrico severo o evento de sequía; "
            f"(2) perturbación reciente del dosel (defoliación, viento, plagas); "
            f"(3) limitación por bajas temperaturas tardías (helada de primavera). "
            f"Se requiere inspección de campo y validación con datos meteorológicos."
        )

    return {
        "status":           status,
        "ndvi_observed":    round(ndvi_median, 4),
        "ndvi_expected_low":  round(low, 3),
        "ndvi_expected_high": round(high, 3),
        "interpretation":   interpretation,
    }


def assess_water_stress(ndmi_mean: float) -> Dict[str, Any]:
    """Assess canopy water content and drought stress from NDMI."""
    if ndmi_mean >= NDMI_THRESHOLDS["well_hydrated"]:
        level = "NO_STRESS"
        label = "Sin estrés hídrico detectable"
        detail = (
            f"NDMI = {ndmi_mean:.3f} indica contenido hídrico foliar normal. "
            f"El dosel de haya no muestra señales de estrés por déficit hídrico en esta observación."
        )
    elif ndmi_mean >= NDMI_THRESHOLDS["mild_stress"]:
        level = "MILD"
        label = "Estrés hídrico leve"
        detail = (
            f"NDMI = {ndmi_mean:.3f} sugiere una reducción moderada del contenido hídrico foliar. "
            f"Puede corresponder a estrés hídrico incipiente o a variación estacional normal. "
            f"Monitorizar evolución en próximas adquisiciones."
        )
    elif ndmi_mean >= NDMI_THRESHOLDS["moderate_stress"]:
        level = "MODERATE"
        label = "Estrés hídrico moderado"
        detail = (
            f"NDMI = {ndmi_mean:.3f} indica una reducción significativa del contenido hídrico foliar "
            f"consistente con condiciones de estrés por sequía. "
            f"En Fagus sylvatica este nivel de estrés puede anticipar cierre estomático y "
            f"reducción de la productividad primaria neta. "
            f"Se recomienda contrastar con datos de precipitación acumulada y SPI/SPEI del período."
        )
    else:
        level = "SEVERE"
        label = "Estrés hídrico severo"
        detail = (
            f"NDMI = {ndmi_mean:.3f} refleja un contenido hídrico foliar marcadamente reducido, "
            f"coherente con condiciones de sequía severa. "
            f"Fagus sylvatica es una especie particularmente sensible a la sequía estival dado "
            f"su sistema radicular superficial. En condiciones de estrés persistente puede "
            f"producirse defoliación temprana, aborto foliar y, en eventos extremos, mortalidad. "
            f"Se requiere evaluación de campo urgente y análisis de datos climáticos recientes."
        )

    return {
        "level":       level,
        "label":       label,
        "ndmi_mean":   round(ndmi_mean, 4),
        "detail":      detail,
    }


def assess_chlorophyll_health(ndre_mean: float) -> Dict[str, Any]:
    """Assess chlorophyll content and early physiological stress from NDRE."""
    if ndre_mean >= NDRE_THRESHOLDS["healthy"]:
        status = "HEALTHY"
        detail = (
            f"NDRE = {ndre_mean:.3f} indica contenido de clorofila foliar normal. "
            f"No se detectan señales de estrés fisiológico en la banda del red-edge."
        )
    elif ndre_mean >= NDRE_THRESHOLDS["mild_stress"]:
        status = "MILD_STRESS"
        detail = (
            f"NDRE = {ndre_mean:.3f} muestra una reducción leve del contenido de clorofila. "
            f"El índice NDRE es sensible a cambios fisiológicos 2–3 semanas antes de que sean "
            f"detectables por NDVI, lo que constituye una señal de alerta temprana relevante."
        )
    else:
        status = "STRESSED"
        detail = (
            f"NDRE = {ndre_mean:.3f} refleja una degradación significativa del contenido de clorofila. "
            f"Este nivel puede asociarse a senescencia acelerada, déficit nutricional (N, Mg) "
            f"o a una respuesta de estrés hídrico o térmico prolongado. "
            f"La señal NDRE precede típicamente al deterioro visible del dosel en 2–4 semanas."
        )

    return {
        "status":    status,
        "ndre_mean": round(ndre_mean, 4),
        "detail":    detail,
    }


def assess_disturbance(nbr_mean: float) -> Dict[str, Any]:
    """Assess disturbance level (fire, windthrow, defoliation) from NBR."""
    if nbr_mean >= NBR_THRESHOLDS["undisturbed"]:
        level = "UNDISTURBED"
        detail = (
            f"NBR = {nbr_mean:.3f}. No se detectan señales de perturbación reciente del dosel. "
            f"El bosque mantiene su integridad estructural según este indicador."
        )
    elif nbr_mean >= NBR_THRESHOLDS["low_severity"]:
        level = "LOW_SEVERITY"
        detail = (
            f"NBR = {nbr_mean:.3f} indica una perturbación de baja intensidad. "
            f"Puede corresponder a apertura de claros naturales, aprovechamiento forestal puntual "
            f"o episodios de defoliación parcial."
        )
    elif nbr_mean >= NBR_THRESHOLDS["moderate"]:
        level = "MODERATE"
        detail = (
            f"NBR = {nbr_mean:.3f} sugiere una perturbación de intensidad moderada. "
            f"Se recomienda revisión cartográfica de posibles alteraciones del dosel."
        )
    else:
        level = "HIGH_SEVERITY"
        detail = (
            f"NBR = {nbr_mean:.3f} refleja una perturbación de alta intensidad. "
            f"Este nivel es indicativo de eventos severos como incendio forestal, "
            f"defoliación masiva o daños mecánicos extensivos. Requiere verificación inmediata."
        )

    return {
        "level":    level,
        "nbr_mean": round(nbr_mean, 4),
        "detail":   detail,
    }


def generate_management_recommendations(
    ndvi_health: Dict,
    water_stress: Dict,
    chlorophyll: Dict,
    disturbance: Dict,
    phase: str,
) -> List[str]:
    """Generate prioritised management recommendations based on all indicators."""
    recs = []

    if disturbance["level"] in ("MODERATE", "HIGH_SEVERITY"):
        recs.append(
            "URGENTE — Verificar la integridad del dosel mediante inspección de campo en las "
            "zonas identificadas como perturbadas. Documentar la causa y extensión del daño."
        )

    if water_stress["level"] == "SEVERE":
        recs.append(
            "PRIORITARIO — Contrastar el estrés hídrico severo detectado con datos de "
            "precipitación acumulada (AEMET) e índices de sequía (SPEI, SPI) del período "
            "junio–agosto. Considerar el impacto sobre la regeneración natural del hayedo."
        )
    elif water_stress["level"] == "MODERATE":
        recs.append(
            "Monitorizar la evolución del estrés hídrico en adquisiciones sucesivas. "
            "Si persiste durante más de 3 observaciones consecutivas, iniciar protocolo "
            "de evaluación de impacto por sequía."
        )

    if chlorophyll["status"] == "STRESSED":
        recs.append(
            "La señal de estrés clorofílico (NDRE) anticipa un posible deterioro del dosel. "
            "Programar muestreo de suelos y análisis foliar para detectar posibles "
            "deficiencias nutricionales o sintomatología de plagas/enfermedades."
        )

    if ndvi_health["status"] == "ANOMALOUS":
        recs.append(
            "El déficit NDVI supera los umbrales de anomalía estadística para la fenofase actual. "
            "Ampliar el análisis temporal incorporando series históricas de NDVI (Landsat, MODIS) "
            "para contextualizar el evento en el marco de la variabilidad interanual del sitio."
        )

    if not recs:
        recs.append(
            "Los indicadores espectrales no revelan señales de perturbación activa ni de estrés "
            "ecológico significativo. Mantener el programa de monitorización regular y ampliar "
            "la serie temporal para establecer la variabilidad natural de referencia del sitio."
        )

    # Always recommend baseline establishment
    recs.append(
        "Continuar acumulando composites anuales (ventana julio–septiembre) para construir "
        "la línea de base histórica. La robustez estadística del sistema de detección de "
        "anomalías requiere un mínimo de 3 años de referencia."
    )

    return recs


def full_ecological_assessment(
    observation_date: date,
    ndvi_stats: Dict[str, float],
    ndmi_stats: Optional[Dict[str, float]] = None,
    ndre_stats: Optional[Dict[str, float]] = None,
    nbr_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Perform a complete multi-indicator ecological assessment.

    Parameters
    ----------
    observation_date : date
        The date of the satellite observation.
    ndvi_stats : dict
        Must contain 'median', 'mean', 'std', 'valid_pixels'.
    ndmi_stats, ndre_stats, nbr_stats : dict, optional
        Same structure; if None the corresponding assessment is skipped.

    Returns
    -------
    dict
        Structured assessment with ecological interpretation and recommendations.
    """
    phase       = get_phenological_phase(observation_date)
    phase_label = PHENOLOGY_LABELS.get(phase, phase)
    doy         = observation_date.timetuple().tm_yday

    ndvi_health   = assess_ndvi_health(ndvi_stats.get("median", float("nan")), phase)
    water_stress  = assess_water_stress(ndmi_stats.get("mean", float("nan"))) if ndmi_stats else None
    chlorophyll   = assess_chlorophyll_health(ndre_stats.get("mean", float("nan"))) if ndre_stats else None
    disturbance   = assess_disturbance(nbr_stats.get("mean", float("nan"))) if nbr_stats else None

    recs = generate_management_recommendations(
        ndvi_health,
        water_stress  or {"level": "UNKNOWN"},
        chlorophyll   or {"status": "UNKNOWN"},
        disturbance   or {"level": "UNKNOWN"},
        phase,
    )

    return {
        "observation_date":  observation_date.isoformat(),
        "day_of_year":       doy,
        "phenological_phase": phase,
        "phenological_label": phase_label,
        "ndvi_health":        ndvi_health,
        "water_stress":       water_stress,
        "chlorophyll_health": chlorophyll,
        "disturbance":        disturbance,
        "overall_status":     _derive_overall_status(ndvi_health, water_stress, chlorophyll, disturbance),
        "management_recommendations": recs,
        "scientific_note": (
            "Evaluación generada automáticamente mediante análisis multiespectal Sentinel-2 (10 m). "
            "Los umbrales aplicados están calibrados para Fagus sylvatica en condiciones de la "
            "Reserva de la Biosfera Sierra del Rincón (41°N, 1150–1450 m s.n.m.). "
            "Este análisis debe interpretarse en conjunción con datos meteorológicos, "
            "inventarios forestales y observaciones de campo."
        ),
    }


def _derive_overall_status(ndvi, water, chlorophyll, disturbance) -> str:
    """Derive a single summary status from all indicator assessments."""
    flags = []
    if ndvi and ndvi.get("status") == "ANOMALOUS":
        flags.append("ALERTA")
    if water and water.get("level") in ("MODERATE", "SEVERE"):
        flags.append("AVISO")
    if chlorophyll and chlorophyll.get("status") == "STRESSED":
        flags.append("AVISO")
    if disturbance and disturbance.get("level") in ("MODERATE", "HIGH_SEVERITY"):
        flags.append("ALERTA")

    if "ALERTA" in flags:
        return "ALERTA"
    if "AVISO" in flags:
        return "AVISO"
    return "NORMAL"
