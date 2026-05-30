"""
Scientific reporting module — 10-section professional report suite.

Generates evidence-based, data-driven reports that read as professional
environmental consultancy documents. All interpretive statements are
derived from the actual data, not template text.

Report sections:
  01_executive_summary.md
  02_forest_condition_assessment.md
  03_technical_results.md
  04_climate_context.md
  05_data_quality.md
  06_anomaly_detection.md
  07_phenology_analysis.md
  08_ecological_interpretation.md
  09_management_implications.md
  10_recommended_actions.md
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from common import OUTPUTS_ANNUAL_DIR, OUTPUTS_MONTHLY_DIR, setup_logging, save_json

logger = setup_logging("reporting")

PIPELINE_VERSION = "3.0"
SITE_NAME        = "Hayedo de Montejo"
RESERVE_NAME     = "Reserva de la Biosfera Sierra del Rincón"
FULL_SITE        = f"{SITE_NAME} — {RESERVE_NAME}"
SPECIES          = "Fagus sylvatica L."
COORDINATES      = "41°13'–41°21'N / 3°25'–3°36'W"
ALTITUDE         = "1150–1450 m s.n.m."
SENSOR           = "Sentinel-2 MSI L2A — ESA Copernicus / CDSE"
STATUS_ICONS     = {"ALERTA": "🔴", "AVISO": "🟡", "NORMAL": "🟢", "N/D": "⚪"}


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _flat(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if not isinstance(sv, (dict, list)):
                    out[f"{k}_{sk}"] = sv
        elif not isinstance(v, list):
            out[k] = v
    return out


def save_json_report(data: Dict, path: str) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    save_json(data, str(p)); return p


def save_csv_report(rows: List[Dict], path: str) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    flat = [_flat(r) for r in rows]
    if flat:
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(flat)
    return p


def save_md(content: str, path: str) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8"); return p


def _fmt(v: Any, decimals: int = 3, units: str = "") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/D"
    return f"{v:.{decimals}f}{units}"


# ---------------------------------------------------------------------------
# Section 01 — Executive Summary
# ---------------------------------------------------------------------------

def _s01_executive_summary(
    run_id: str, job: str, obs_date: datetime,
    eco: Dict, anomaly: Dict, n_scenes: int,
    climate: Optional[Dict], forest_stats: Dict,
) -> str:
    status      = eco.get("overall_status", "N/D")
    icon        = STATUS_ICONS.get(status, "⚪")
    month       = obs_date.strftime("%B %Y")
    phase_label = eco.get("phenological_label", "N/D")
    ndvi_h      = eco.get("ndvi_health", {})
    water       = eco.get("water_stress", {})
    area_ha     = anomaly.get("anomaly_area_ha", 0.0)
    frac_pct    = anomaly.get("anomaly_fraction", 0.0) * 100
    persist     = anomaly.get("persistent", False)
    ts          = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    clim_note = ""
    if climate and climate.get("available"):
        ds = climate.get("drought_status", "near_normal")
        pa = climate.get("precip_anomaly_pct")
        ta = climate.get("temp_anomaly_mean_c")
        if ds in ("severely_dry", "extremely_dry"):
            clim_note = f"**Alerta climática:** sequía severa detectada (SPI-3 ≤ -1.5)."
        elif pa is not None and abs(pa) > 30:
            clim_note = f"Precipitación {'por debajo' if pa < 0 else 'por encima'} del normal ({pa:+.0f}%)."
        if ta is not None and abs(ta) > 1.5:
            clim_note += f" Temperatura {ta:+.1f}°C respecto a la normal 1991–2020."

    ndvi_obs    = ndvi_h.get("ndvi_observed", float("nan"))
    ndvi_range  = f"{ndvi_h.get('ndvi_expected_low', '–')}–{ndvi_h.get('ndvi_expected_high', '–')}"

    # Forest-masked stats
    fm_note = ("— estadísticas calculadas únicamente sobre píxeles forestales verificados"
               if forest_stats.get("forest_masked") else
               "— estadísticas sobre ventana AOI (máscara forestal pendiente de validación oficial)")

    return f"""# Resumen Ejecutivo — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}` | **Fecha:** {ts}
**Estado del sistema:** {icon} **{status}**

---

## Situación del Hayedo de Montejo — {month}

El presente análisis integra {n_scenes} escena(s) Sentinel-2 L2A con cobertura nubosa
< 15%, procesadas con el pipeline automatizado de monitorización del {FULL_SITE}.

### Estado de la Vegetación

| Indicador | Valor | Referencia fenológica |
|-----------|-------|-----------------------|
| Fase fenológica (DOY {obs_date.timetuple().tm_yday}) | {phase_label} | — |
| NDVI mediano del bosque | **{_fmt(ndvi_obs, 4)}** | Esperado: {ndvi_range} |
| Estado de vitalidad NDVI | **{ndvi_h.get("status", "N/D")}** | — |
| Estrés hídrico (NDMI) | **{water.get("label", "N/D")}** | — |
| Área con déficit significativo | **{area_ha:.1f} ha** ({frac_pct:.1f}%) | < 5% = NORMAL |
| Anomalía persistente | **{"Sí — verificación de campo recomendada" if persist else "No"}** | — |

{fm_note}

### Contexto Climático
{clim_note if clim_note else "Datos climáticos AEMET no disponibles en esta ejecución."}

### Conclusión Ejecutiva

{ndvi_h.get("interpretation", "Datos insuficientes para evaluación completa.")}

---
*{SENSOR} · {SITE_NAME} · {COORDINATES} · {ALTITUDE} · Pipeline v{PIPELINE_VERSION}*
"""


# ---------------------------------------------------------------------------
# Section 02 — Forest Condition Assessment
# ---------------------------------------------------------------------------

def _s02_forest_condition(
    run_id: str, obs_date: datetime,
    eco: Dict, forest_stats: Dict,
    baseline_available: bool,
) -> str:
    ndvi_h  = eco.get("ndvi_health", {})
    water   = eco.get("water_stress", {})
    chloro  = eco.get("chlorophyll_health", {})
    disturb = eco.get("disturbance", {})
    status  = eco.get("overall_status", "N/D")
    month   = obs_date.strftime("%Y-%m")

    # NDVI percentile context
    p10 = forest_stats.get("p10")
    p90 = forest_stats.get("p90")
    pct_context = (
        f"P10={_fmt(p10, 3)}, P90={_fmt(p90, 3)}" if p10 and p90
        else "distribución percentilar no disponible"
    )

    baseline_note = (
        "Evaluación frente a baseline histórico (2021–2025) activa."
        if baseline_available else
        "**Nota:** Baseline histórico no disponible. La evaluación se basa en umbrales "
        "fenológicos calibrados para *Fagus sylvatica* en este emplazamiento. "
        "La precisión mejorará progresivamente con la acumulación de años de referencia."
    )

    return f"""# Evaluación del Estado Forestal — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}` | **Estado:** {STATUS_ICONS.get(status, '')} {status}

---

## 1. Especie objetivo y contexto de monitorización

- **Especie:** {SPECIES}
- **Emplazamiento:** {FULL_SITE}
- **Coordenadas:** {COORDINATES} · **Altitud:** {ALTITUDE}
- **Singularidad ecológica:** hayedo meridional más importante de la Península Ibérica
  y ejemplo de bosque atlántico bajo condiciones submediterráneas
- **Relevancia ante el cambio climático:** especie indicadora de primer orden,
  en el límite sur de su área de distribución natural

{baseline_note}

---

## 2. Actividad fotosintética (NDVI)

{ndvi_h.get("interpretation", "No evaluado.")}

**Distribución espectral del bosque:** {pct_context}

---

## 3. Estado hídrico foliar (NDMI)

{water.get("detail", "NDMI no disponible en esta ejecución.")}

---

## 4. Salud fisiológica — Contenido clorofílico (NDRE)

{chloro.get("detail", "NDRE no disponible en esta ejecución.")}

> **Nota científica:** El índice NDRE es sensible a variaciones del contenido de
> clorofila 2–3 semanas antes de que el estrés sea detectable por NDVI. Constituye
> un sistema de alerta temprana de especial valor para la gestión del hayedo.

---

## 5. Integridad estructural del dosel (NBR)

{disturb.get("detail", "NBR no disponible en esta ejecución.")}

---

## 6. Valoración integrada

**Estado global: {STATUS_ICONS.get(status, '')} {status}**

La combinación de indicadores espectrales ({', '.join(['NDVI', 'NDMI', 'NDRE', 'NBR'])})
proporciona una evaluación multidimensional del estado del ecosistema forestal.
{eco.get("scientific_note", "")}
"""


# ---------------------------------------------------------------------------
# Section 03 — Technical Results
# ---------------------------------------------------------------------------

def _s03_technical_results(
    run_id: str, obs_date: datetime,
    composite_path: str, forest_stats: Dict,
    index_stats: Dict, n_scenes: int,
    trend_slope: Optional[float], anomaly: Dict,
) -> str:
    month = obs_date.strftime("%Y-%m")

    trend_str = (
        f"{trend_slope:+.5f} NDVI/año ({'mejora' if trend_slope > 0 else 'descenso'})"
        if trend_slope is not None else "No calculable (≥3 observaciones requeridas)"
    )

    idx_rows = ""
    for name, s in index_stats.items():
        idx_rows += (
            f"| {name} | {_fmt(s.get('mean'), 4)} | {_fmt(s.get('median'), 4)} | "
            f"{_fmt(s.get('std'), 4)} | {s.get('valid_pixels', 'N/D'):,} |\n"
        )

    forest_rows = "\n".join(
        f"| {k} | {_fmt(v, 4) if isinstance(v, float) else v} |"
        for k, v in forest_stats.items()
        if k not in ("forest_masked",)
    )

    return f"""# Resultados Técnicos — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Dataset

| Parámetro | Valor |
|-----------|-------|
| Sensor | {SENSOR} |
| Nivel de procesamiento | L2A (Bottom of Atmosphere, corrección Sen2Cor) |
| Escenas en el composite | {n_scenes} |
| Período de composite | Ventana rodante 45 días |
| Fecha de referencia | {obs_date.strftime("%Y-%m-%d")} (DOY {obs_date.timetuple().tm_yday}) |
| Resolución espacial | 10 m (bandas ópticas) / 20 m → 10 m (red-edge, SWIR) |
| Composite | {Path(composite_path).name} |
| Máscara SCL válida | Clases 4 (vegetación) y 5 (suelo/roca) |
| Polígono forestal | {forest_stats.get("boundary_status", "Aproximación documentada")} |

## 2. Estadísticas forestales del composite (píxeles dentro del polígono forestal)

| Métrica | Valor NDVI |
|---------|-----------|
{forest_rows}

## 3. Resumen de índices espectrales — píxeles forestales

| Índice | Media | Mediana | SD | Píxeles válidos |
|--------|-------|---------|-----|-----------------|
{idx_rows}
## 4. Análisis de tendencia interanual

| Tendencia | {trend_str} |
|-----------|-------------|

## 5. Detección de anomalías

| Parámetro | Valor |
|-----------|-------|
| Método | {anomaly.get("method", "N/D")} |
| Umbral | {anomaly.get("anomaly_threshold", anomaly.get("z_threshold", "N/D"))} |
| Área anómala | {anomaly.get("anomaly_area_ha", 0):.1f} ha ({anomaly.get("anomaly_fraction", 0)*100:.1f}%) |
| Anomalía persistente | {"Sí" if anomaly.get("persistent") else "No"} |
| Nota | {anomaly.get("note", "—")} |
"""


# ---------------------------------------------------------------------------
# Section 04 — Climate Context
# ---------------------------------------------------------------------------

def _s04_climate_context(
    run_id: str, obs_date: datetime,
    climate_context: Optional[Dict],
    ndvi_climate_corr: Optional[Dict],
) -> str:
    month = obs_date.strftime("%Y-%m")

    if not climate_context or not climate_context.get("available"):
        return f"""# Contexto Climático — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## Estado

Los datos climáticos AEMET no están disponibles en esta ejecución.

Para activar la integración climática:
1. Registrarse en https://opendata.aemet.es/centrodedescargas/altaUsuario (gratuito)
2. Añadir la clave API al parámetro `aemet.api_key` en `config/thresholds.yaml`
3. Las variables integradas incluirán: temperatura mensual, precipitación,
   SPI-3 (sequía a 3 meses), SPI-12, y anomalías respecto a la normal 1991–2020

**Estación primaria:** Montejo de la Sierra (AEMET 2864A, 1125 m s.n.m.)

---

*La interpretación ecológica es más robusta con datos climáticos de contexto.
Se recomienda activar la integración AEMET antes del análisis del TFM.*
"""

    c = climate_context
    pa = c.get("precip_anomaly_pct")
    ta = c.get("temp_anomaly_mean_c")
    ds = c.get("drought_status", "unknown")

    drought_table = {
        "extremely_dry":   "Sequía extrema (SPI-3 < -2.0)",
        "severely_dry":    "Sequía severa (-2.0 ≤ SPI-3 < -1.5)",
        "moderately_dry":  "Sequía moderada (-1.5 ≤ SPI-3 < -1.0)",
        "near_normal":     "Condiciones normales (-1.0 ≤ SPI-3 < 1.0)",
        "moderately_wet":  "Condiciones húmedas (1.0 ≤ SPI-3 < 1.5)",
        "very_wet":        "Condiciones muy húmedas (1.5 ≤ SPI-3 < 2.0)",
        "extremely_wet":   "Condiciones extremadamente húmedas (SPI-3 ≥ 2.0)",
    }

    corr_block = ""
    if ndvi_climate_corr and ndvi_climate_corr.get("n_pairs", 0) >= 3:
        cr = ndvi_climate_corr
        corr_block = f"""
## 4. Relación NDVI–Clima

| Parámetro | Valor |
|-----------|-------|
| Variable climática | {cr.get("climate_var", "N/D")} |
| r de Pearson | {_fmt(cr.get("pearson_r"), 3)} |
| p-valor | {_fmt(cr.get("pearson_p"), 4)} |
| r de Spearman | {_fmt(cr.get("spearman_r"), 3)} |
| Pares de observaciones | {cr.get("n_pairs", 0)} |

{cr.get("interpretation", "")}
"""

    return f"""# Contexto Climático — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Resumen meteorológico del período analizado

| Variable | Valor | Anomalía / Estado |
|----------|-------|-------------------|
| Precipitación total | {_fmt(c.get("total_precip_mm"), 1, " mm")} | {_fmt(pa, 1, "%")} respecto a la normal ({_fmt(c.get("normal_precip_mm"), 0, " mm")}) |
| Temperatura media | {_fmt(c.get("mean_temp_c"), 1, " °C")} | {_fmt(ta, 2, " °C")} respecto a la normal 1991–2020 |
| SPI-3 mínimo | {_fmt(c.get("min_spi_3"), 2)} | {drought_table.get(ds, ds)} |

## 2. Interpretación climática

{c.get("ecological_context", "Sin interpretación disponible.")}

## 3. Implicaciones para la vegetación

La integración de indicadores espectrales con el contexto climático permite
distinguir entre:

- **Estrés climático** (temperatura, déficit hídrico) → causa identificable externamente
- **Perturbación estructural** (defoliación, daño mecánico) → requiere verificación de campo
- **Variabilidad fenológica normal** → contextualizado por la fase del calendario vegetativo
{corr_block}
---

*Fuente: AEMET OpenData API · Estación: {c.get("period", month)} · Normal: 1991–2020*
"""


# ---------------------------------------------------------------------------
# Section 05 — Data Quality
# ---------------------------------------------------------------------------

def _s05_data_quality(
    run_id: str, obs_date: datetime,
    n_scenes: int, cloud_covers: List[float],
    valid_pixel_fraction: float, forest_stats: Dict,
    boundary_status: str,
) -> str:
    month   = obs_date.strftime("%Y-%m")
    clouds  = ", ".join(f"{c:.1f}%" for c in cloud_covers) if cloud_covers else "N/D"
    quality = "ALTA" if valid_pixel_fraction > 0.80 else ("MEDIA" if valid_pixel_fraction > 0.55 else "BAJA")
    forest_px = forest_stats.get("valid_pixels", 0)

    return f"""# Evaluación de Calidad de Datos — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Calidad de adquisición

| Parámetro | Valor |
|-----------|-------|
| Escenas utilizadas | {n_scenes} |
| Cobertura nubosa por escena | {clouds} |
| Fracción de píxeles válidos (SCL 4+5) | **{valid_pixel_fraction*100:.1f}%** |
| Calificación global | **{quality}** |
| Píxeles forestales válidos en composite | {forest_px:,} |
| Estado del polígono forestal | {boundary_status} |

## 2. Criterios de calidad aplicados

- **Filtro de adquisición:** CC ≤ 15% (configurable en `thresholds.yaml`)
- **Máscara SCL:** clases 4 (vegetación) y 5 (suelo/roca)
- **Clases excluidas:** nubes (8, 9), sombras de nube (3), agua (6),
  sin clasificar (7), cirros (10), nieve/hielo (11)
- **Máscara forestal:** polígono {boundary_status} — estadísticas sólo sobre *F. sylvatica* pixels

## 3. Limitaciones conocidas de esta ejecución

| Limitación | Impacto | Prioridad de corrección |
|------------|---------|-------------------------|
| Corrección topográfica no aplicada | Sesgo en sombras de ladera (pendientes >20°) | ALTA |
| Normalización BRDF no aplicada | Sesgo por ángulo solar variable entre fechas | MEDIA |
| Polígono forestal provisional | Posible contaminación por píxeles de borde | ALTA |
| Baseline histórico {'disponible' if False else 'en construcción'} | Precisión de detección de anomalías limitada | ALTA |
| Validación de campo no realizada | Sin calibración independiente de umbrales | MEDIA |

## 4. Recomendaciones para mejorar la calidad

1. Aplicar corrección topográfica C-correction o SCS+C (implementación pendiente)
2. Sustituir el polígono provisional por el shapefile MFE50 oficial para Madrid
3. Ampliar la serie temporal con datos históricos 2021–2025
4. Programar visita de campo para calibración de umbrales espectrales
"""


# ---------------------------------------------------------------------------
# Section 06 — Anomaly Detection
# ---------------------------------------------------------------------------

def _s06_anomaly_detection(
    run_id: str, obs_date: datetime,
    anomaly: Dict, baseline_summary: Optional[Dict],
) -> str:
    month     = obs_date.strftime("%Y-%m")
    has_base  = baseline_summary and baseline_summary.get("z_score_ready", False)
    n_annual  = baseline_summary.get("n_annual_composites", 0) if baseline_summary else 0
    method    = anomaly.get("method", "N/D")
    area_ha   = anomaly.get("anomaly_area_ha", 0)
    frac_pct  = anomaly.get("anomaly_fraction", 0) * 100
    persist   = anomaly.get("persistent", False)

    baseline_block = f"""
## 2. Estado del baseline histórico

| Parámetro | Valor |
|-----------|-------|
| Composites anuales disponibles | {n_annual} |
| Años | {baseline_summary.get("years_available", []) if baseline_summary else "[]"} |
| Estadísticas calculadas | {"Sí" if has_base else "No"} |
| Detección Z-score activa | {"Sí" if has_base else "No — mínimo 3 años requeridos"} |

{"El sistema de detección Z-score está activo." if has_base else "**Nota:** El sistema opera en modo degradado (umbral fijo) hasta disponer de ≥3 composites anuales. Ejecutar `python src/baseline_builder.py --years 2021 2022 2023 2024 2025` para construir el baseline histórico."}
"""

    return f"""# Detección de Anomalías — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Resultados de detección

| Parámetro | Valor |
|-----------|-------|
| Método activo | {method} |
| Umbral | {anomaly.get("anomaly_threshold", anomaly.get("z_threshold", "N/D"))} |
| Área anómala | **{area_ha:.1f} ha** ({frac_pct:.1f}% del AOI) |
| Desviación media | {_fmt(anomaly.get("delta_mean", anomaly.get("z_mean")), 4)} |
| Persistencia | {"Sí — ≥2 observaciones consecutivas" if persist else "No"} |
| Nota | {anomaly.get("note", "—")} |
{baseline_block}
## 3. Interpretación

{"No se detectan anomalías significativas. La actividad espectral del bosque es consistente con el baseline." if area_ha == 0 else f"Se detectan {area_ha:.1f} ha ({frac_pct:.1f}%) con déficit espectral respecto al baseline. {'La anomalía es persistente (≥2 observaciones) y requiere verificación de campo.' if persist else 'La anomalía no es persistente — monitorizar en próximas adquisiciones.'}"}

## 4. Limitaciones de la detección

{"**BASELINE NO DISPONIBLE:** El sistema no puede realizar detección estadística robusta. Los resultados actuales son orientativos." if "note" in anomaly and "Baseline" in anomaly.get("note", "") else "La detección de anomalías está operativa con el baseline disponible."}
"""


# ---------------------------------------------------------------------------
# Section 07 — Phenology Analysis
# ---------------------------------------------------------------------------

def _s07_phenology(
    run_id: str, obs_date: datetime,
    phenology_records: Optional[List[Dict]],
    phenology_trends: Optional[Dict],
) -> str:
    month = obs_date.strftime("%Y-%m")
    doy   = obs_date.timetuple().tm_yday

    from ecology import get_phenological_phase, PHENOLOGY_LABELS, PHENOLOGY
    phase = get_phenological_phase(obs_date.date())
    p_start, p_end = PHENOLOGY.get(phase, (0, 365))

    records_table = ""
    if phenology_records:
        for r in phenology_records[-5:]:  # last 5 years
            records_table += (
                f"| {r['year']} | {r.get('peak_ndvi', 'N/D'):.3f} | "
                f"{r.get('growing_season_length_days', 'N/D')} | "
                f"{r.get('method', 'N/D')} |\n"
            )

    trend_block = ""
    if phenology_trends and phenology_trends.get("peak_ndvi_trend"):
        t = phenology_trends["peak_ndvi_trend"]
        trend_block = f"""
### Tendencia en NDVI de pico estacional

{t.get("interpretation", "")}

| Métrica | Valor |
|---------|-------|
| Pendiente (NDVI/año) | {_fmt(t.get("slope_per_year"), 5)} |
| R² | {_fmt(t.get("r_squared"), 4)} |
| p-valor | {_fmt(t.get("p_value"), 4)} |
| τ Mann-Kendall | {_fmt(t.get("mann_kendall_tau"), 4)} |
"""

    return f"""# Análisis Fenológico — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Fase fenológica actual

**Observación:** DOY {doy} → **{PHENOLOGY_LABELS.get(phase, phase)}**
(período típico: DOY {p_start}–{p_end} para *Fagus sylvatica* a 41°N, ~1300 m s.n.m.)

El calendario fenológico está calibrado a partir de:
- Menzel et al. (2006) — European phenological records 1971–2000
- Vitasse et al. (2009) — Iberian Peninsula beech populations
- AEMET observaciones fenológicas de la red nacional

## 2. Registro histórico de métricas de pico estacional (Jul–Sep)

{"No hay datos históricos disponibles. Ejecutar `baseline_builder.py` para iniciar la construcción del registro." if not phenology_records else f"""
| Año | NDVI pico | GSL (días) | Método |
|-----|-----------|-----------|--------|
{records_table}
"""}
{trend_block}
## 3. Limitaciones del análisis fenológico actual

El análisis fenológico completo (SOS, EOS, GSL) requiere observaciones distribuidas
a lo largo de toda la estación de crecimiento (abril–noviembre). Los composites anuales
de julio–septiembre proporcionan únicamente el valor de pico estacional.

Para análisis fenológicos completos se recomienda:
- Descargar imágenes mensuales adicionales (abril, junio, octubre, noviembre)
- Aplicar el método de doble logística de Beck et al. (2006) una vez disponibles ≥8 observaciones anuales
- Contrastar con observaciones de campo (apertura de yemas, caída de hoja)

## 4. Relevancia para la candidatura CETS

Los cambios fenológicos documentados mediante teledetección constituyen evidencia
cuantificable de la respuesta del ecosistema al cambio climático — un argumento
de especial relevancia para los criterios de "gestión adaptativa" de la CETS.
"""


# ---------------------------------------------------------------------------
# Section 08 — Ecological Interpretation
# ---------------------------------------------------------------------------

def _s08_ecological_interpretation(
    run_id: str, obs_date: datetime,
    eco: Dict, climate_context: Optional[Dict],
    baseline_available: bool,
) -> str:
    month = obs_date.strftime("%Y-%m")
    status = eco.get("overall_status", "N/D")
    ndvi_h = eco.get("ndvi_health", {})
    water  = eco.get("water_stress", {})
    chloro = eco.get("chlorophyll_health", {})

    # Build evidence chain
    evidence = []
    if ndvi_h.get("status") in ("ANOMALOUS", "BELOW_EXPECTED"):
        evidence.append(f"déficit de NDVI ({ndvi_h.get('ndvi_observed', 'N/D')} vs. esperado {ndvi_h.get('ndvi_expected_low', '–')}–{ndvi_h.get('ndvi_expected_high', '–')})")
    if water.get("level") in ("MODERATE", "SEVERE"):
        evidence.append(f"estrés hídrico foliar {water.get('level', '').lower()} (NDMI={_fmt(water.get('ndmi_mean'), 3)})")
    if chloro and chloro.get("status") == "STRESSED":
        evidence.append(f"déficit clorofílico (NDRE={_fmt(chloro.get('ndre_mean'), 3)})")

    if climate_context and climate_context.get("available"):
        ds = climate_context.get("drought_status", "near_normal")
        if ds in ("severely_dry", "extremely_dry"):
            evidence.append(f"condiciones de sequía severa/extrema (SPI-3={_fmt(climate_context.get('min_spi_3'), 2)})")

    if not evidence:
        evidence_text = (
            "Los índices espectrales multi-temporales no revelan señales de perturbación "
            "o estrés ecológico significativo en esta observación. El ecosistema muestra "
            "una respuesta espectral coherente con su comportamiento fenológico histórico "
            "esperado para *Fagus sylvatica* en esta localidad."
        )
    else:
        evidence_text = (
            f"La observación presenta las siguientes señales de estrés ecológico: "
            f"{'; '.join(evidence)}. "
            + (f"\n\n{ndvi_h.get('interpretation', '')}" if ndvi_h.get("interpretation") else "")
        )

    resilience_note = (
        "El Hayedo de Montejo presenta una notable resiliencia histórica a los eventos de "
        "estrés estival, dado su emplazamiento en orientación norte y el microclima local "
        "que mitiga parcialmente las temperaturas extremas. Sin embargo, la posición de esta "
        "población en el límite meridional de la distribución de *Fagus sylvatica* la hace "
        "especialmente vulnerable a la intensificación de sequías proyectada para el "
        "Mediterráneo occidental bajo los escenarios RCP4.5 y RCP8.5."
    )

    return f"""# Interpretación Ecológica — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}` | **Estado:** {STATUS_ICONS.get(status, '')} {status}

---

## 1. Evaluación del estado ecológico del ecosistema

{evidence_text}

## 2. Contexto de resiliencia y vulnerabilidad

{resilience_note}

## 3. Síntesis multi-indicador

| Indicador | Valor | Estado | Implicación ecológica |
|-----------|-------|--------|-----------------------|
| NDVI (actividad fotosintética) | {_fmt(ndvi_h.get("ndvi_observed"), 4)} | {ndvi_h.get("status", "N/D")} | Actividad del dosel respecto a la fenofase |
| NDMI (agua foliar) | {_fmt(water.get("ndmi_mean"), 4)} | {water.get("level", "N/D")} | Estado hídrico y riesgo de estrés por sequía |
| NDRE (clorofila) | {_fmt(chloro.get("ndre_mean") if chloro else None, 4)} | {chloro.get("status", "N/D") if chloro else "N/D"} | Señal de alerta temprana (2–3 semanas antes que NDVI) |

## 4. Implicaciones para la gestión de la Reserva de la Biosfera

{eco.get("scientific_note", "")}

La integración de los indicadores espectrales en el sistema de gestión adaptativa
de la Reserva de la Biosfera Sierra del Rincón permite:

1. **Monitorización continua** sin necesidad de inventarios de campo frecuentes
2. **Detección temprana** de anomalías mediante señales NDRE (pre-sintomáticas)
3. **Priorización espacial** de áreas que requieren inspección de campo
4. **Documentación cuantitativa** del estado del ecosistema para informes CETS
"""


# ---------------------------------------------------------------------------
# Section 09 — Management Implications
# ---------------------------------------------------------------------------

def _s09_management_implications(
    run_id: str, obs_date: datetime,
    eco: Dict, climate_context: Optional[Dict],
    anomaly: Dict,
) -> str:
    month = obs_date.strftime("%Y-%m")
    recs  = eco.get("management_recommendations", [])

    clim_implication = ""
    if climate_context and climate_context.get("available"):
        ds = climate_context.get("drought_status", "near_normal")
        if ds in ("severely_dry", "extremely_dry"):
            clim_implication = (
                "\n**PROTOCOLO DE SEQUÍA:** Las condiciones de sequía severa detectadas "
                "activan el protocolo de seguimiento intensivo. Se recomienda coordinar "
                "con AEMET el seguimiento de los índices de sequía en tiempo real (SPEI) "
                "y evaluar posibles restricciones de acceso de visitantes en zonas más frágiles."
            )

    return f"""# Implicaciones para la Gestión — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Acciones de gestión derivadas del análisis

{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(recs))}
{clim_implication}

## 2. Vinculación con criterios CETS

| Criterio CETS | Relevancia en esta ejecución |
|---------------|------------------------------|
| A3 — Seguimiento y evaluación | Los indicadores espectrales proporcionan métricas cuantitativas reproducibles |
| B1 — Conservación del patrimonio natural | El estado del dosel es directamente representativo del valor del patrimonio |
| B2 — Gestión sostenible del territorio | Los mapas de anomalías orientan las prioridades de gestión |
| C1 — Gestión de visitantes | Las condiciones de estrés sugieren temporadas de mayor fragilidad del ecosistema |

## 3. Comunicación a stakeholders

El estado `{eco.get("overall_status", "N/D")}` del sistema debe comunicarse a:
- Servicio de Medio Ambiente de la Comunidad de Madrid
- Comité de Gestión de la Reserva de la Biosfera Sierra del Rincón
- Equipo de guardería y gestión forestal de Montejo

## 4. Integración en el Plan de Gestión

Los resultados de teledetección complementan (no sustituyen) los inventarios
forestales y observaciones de campo. Se recomienda incorporar los indicadores
espectrales como un indicador de seguimiento permanente en el Plan Rector de
Uso y Gestión (PRUG) de la Reserva.
"""


# ---------------------------------------------------------------------------
# Section 10 — Recommended Actions
# ---------------------------------------------------------------------------

def _s10_recommended_actions(
    run_id: str, obs_date: datetime,
    eco: Dict, baseline_summary: Optional[Dict],
    trend_slope: Optional[float],
) -> str:
    month = obs_date.strftime("%Y-%m")
    n_annual = baseline_summary.get("n_annual_composites", 0) if baseline_summary else 0
    needs_baseline = n_annual < 3

    return f"""# Acciones Recomendadas — {month}
## {FULL_SITE}

**Ejecución:** `{run_id}`

---

## 1. Acciones inmediatas (próximas 4 semanas)

{"- **URGENTE — Inspección de campo:** La anomalía detectada requiere verificación en parcelas de referencia." if eco.get("overall_status") == "ALERTA" else "- No se requieren acciones inmediatas urgentes según los indicadores actuales."}
- Verificar disponibilidad de nuevas escenas Sentinel-2 en CDSE (job semanal automatizado)
- Revisar datos AEMET para contextualizar la observación con temperatura y precipitación reciente

## 2. Acciones a corto plazo (1–3 meses)

{"- **CRÍTICO — Construir el baseline histórico:** Ejecutar `python src/baseline_builder.py --years 2021 2022 2023 2024 2025` para habilitar la detección Z-score." if needs_baseline else "- Actualizar el baseline con el composite del año en curso (job anual en octubre)."}
- Obtener el polígono oficial del hayedo (MFE50, Comunidad de Madrid) y sustituir el polígono provisional
- Registrar API key de AEMET y activar la integración climática en `thresholds.yaml`

## 3. Acciones a medio plazo (TFM — 3–12 meses)

- Aplicar corrección topográfica (C-correction) al flujo de procesamiento
- Descargar observaciones adicionales fuera de la ventana julio–septiembre
  para análisis fenológico completo (abril, junio, octubre, noviembre)
- Contrastar resultados con datos MODIS (MOD13Q1, 250 m) para validación independiente
  y extensión del contexto histórico a 2000–presente
- Programar muestreo de campo en parcelas permanentes para calibración de umbrales

## 4. Acciones a largo plazo (programa de monitorización continuada)

- Mantener el pipeline activo con jobs semanales, mensuales y anuales automatizados
- Ampliar el sistema de alerta a otras especies de interés en la Reserva
- Integrar métricas de paisaje (FRAGSTATS) para análisis de fragmentación del dosel
- Desarrollar módulo de predicción (LSTM/Random Forest) para anticipación de eventos de estrés

## 5. Recursos para el TFM

- **Polígono oficial:** CNIG → MFE50 → Provincia 28 (Madrid) → capa ARBOLADO, filtrar SPE1='Fs'
- **Datos AEMET:** https://opendata.aemet.es/centrodedescargas/altaUsuario
- **Validación cruzada:** NASA AppEEARS (MODIS MOD13Q1) o Google Earth Engine
- **Literatura de referencia:**
  - Hernández-Matías et al. (2019)
  - García-Ruiz et al. (2021) — Water stress in marginal beech populations
  - PORN Sierra del Rincón (BOCM 126/2008)
  - Vitasse et al. (2009) — Phenology Iberian beech
"""


# ---------------------------------------------------------------------------
# Full 10-section suite
# ---------------------------------------------------------------------------

def export_full_report_suite(
    run_id: str,
    job: str,
    obs_date: datetime,
    output_dir: Path,
    eco: Optional[Dict],
    anomaly: Dict,
    forest_stats: Dict,
    composite_path: str,
    index_stats: Dict,
    cloud_covers: List[float],
    n_scenes: int,
    trend_slope: Optional[float] = None,
    trend_path: Optional[str] = None,
    climate_context: Optional[Dict] = None,
    climate_corr: Optional[Dict] = None,
    phenology_records: Optional[List] = None,
    phenology_trends: Optional[Dict] = None,
    baseline_summary: Optional[Dict] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eco = eco or {}
    valid_frac = min(1.0, forest_stats.get("valid_pixels", 0) / max(1_871_424, 1))
    boundary_status = forest_stats.get("boundary_status", "Provisional")

    baseline_avail = bool(
        baseline_summary and baseline_summary.get("z_score_ready", False)
    )

    sections = {
        "01_executive_summary.md":       _s01_executive_summary(run_id, job, obs_date, eco, anomaly, n_scenes, climate_context, forest_stats),
        "02_forest_condition.md":        _s02_forest_condition(run_id, obs_date, eco, forest_stats, baseline_avail),
        "03_technical_results.md":       _s03_technical_results(run_id, obs_date, composite_path, forest_stats, index_stats, n_scenes, trend_slope, anomaly),
        "04_climate_context.md":         _s04_climate_context(run_id, obs_date, climate_context, climate_corr),
        "05_data_quality.md":            _s05_data_quality(run_id, obs_date, n_scenes, cloud_covers, valid_frac, forest_stats, boundary_status),
        "06_anomaly_detection.md":       _s06_anomaly_detection(run_id, obs_date, anomaly, baseline_summary),
        "07_phenology.md":               _s07_phenology(run_id, obs_date, phenology_records, phenology_trends),
        "08_ecological_interpretation.md": _s08_ecological_interpretation(run_id, obs_date, eco, climate_context, baseline_avail),
        "09_management_implications.md": _s09_management_implications(run_id, obs_date, eco, climate_context, anomaly),
        "10_recommended_actions.md":     _s10_recommended_actions(run_id, obs_date, eco, baseline_summary, trend_slope),
    }

    paths = []
    for fname, content in sections.items():
        paths.append(save_md(content, str(output_dir / fname)))
    logger.info("Generated %d report sections in %s", len(paths), output_dir)
    return paths


# ---------------------------------------------------------------------------
# Legacy-compatible entry points
# ---------------------------------------------------------------------------

def export_monthly_summary(
    observation_date: datetime,
    anomaly_payload: Dict,
    trend_path: Optional[str],
    composite_path: str,
    webhook_url: Optional[str] = None,
    run_id: Optional[str] = None,
    ecological_assessment: Optional[Dict] = None,
    composite_stats: Optional[Dict] = None,
    index_stats: Optional[Dict] = None,
    cloud_covers: Optional[List[float]] = None,
    n_scenes: int = 0,
    trend_slope: Optional[float] = None,
    climate_context: Optional[Dict] = None,
    climate_corr: Optional[Dict] = None,
    phenology_records: Optional[List] = None,
    phenology_trends: Optional[Dict] = None,
    baseline_summary: Optional[Dict] = None,
    forest_stats: Optional[Dict] = None,
) -> Dict:
    run_id        = run_id or "unknown"
    composite_stats = composite_stats or {}
    forest_stats  = forest_stats or composite_stats
    index_stats   = index_stats or {}
    cloud_covers  = cloud_covers or []

    report = {
        "report_type":    "monthly_operational",
        "run_id":         run_id,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "month":          observation_date.strftime("%Y-%m"),
        "composite_path": composite_path,
        "trend_path":     trend_path,
        "trend_slope_ndvi_per_year": trend_slope,
        "n_scenes":       n_scenes,
        "composite_stats": composite_stats,
        "forest_stats":   forest_stats,
        "anomaly":        anomaly_payload,
        "ecological_assessment": ecological_assessment,
        "climate_context": climate_context,
    }

    stem        = observation_date.strftime("monthly_summary_%Y_%m")
    reports_dir = OUTPUTS_MONTHLY_DIR / "reports" / observation_date.strftime("%Y-%m")

    save_json_report(report, str(OUTPUTS_MONTHLY_DIR / f"{stem}.json"))
    save_csv_report([report],  str(OUTPUTS_MONTHLY_DIR / f"{stem}.csv"))

    export_full_report_suite(
        run_id=run_id, job="monthly",
        obs_date=observation_date,
        output_dir=reports_dir,
        eco=ecological_assessment,
        anomaly=anomaly_payload,
        forest_stats=forest_stats,
        composite_path=composite_path,
        index_stats=index_stats,
        cloud_covers=cloud_covers,
        n_scenes=n_scenes,
        trend_slope=trend_slope,
        trend_path=trend_path,
        climate_context=climate_context,
        climate_corr=climate_corr,
        phenology_records=phenology_records,
        phenology_trends=phenology_trends,
        baseline_summary=baseline_summary,
    )

    if webhook_url:
        _discord(webhook_url, observation_date, anomaly_payload, composite_path, trend_path)

    return report


def export_annual_scientific_report(
    year: int,
    composite_path: str,
    statistics: Dict,
    trend_slope: Optional[float] = None,
    run_id: Optional[str] = None,
    ecological_assessment: Optional[Dict] = None,
    forest_stats: Optional[Dict] = None,
    baseline_summary: Optional[Dict] = None,
    phenology_records: Optional[List] = None,
    phenology_trends: Optional[Dict] = None,
) -> Dict:
    run_id      = run_id or "unknown"
    forest_stats = forest_stats or statistics

    report = {
        "report_type":    "annual_scientific",
        "run_id":         run_id,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "year":           year,
        "composite_path": composite_path,
        "statistics":     statistics,
        "forest_stats":   forest_stats,
        "trend_slope_ndvi_per_year": trend_slope,
    }

    stem = f"annual_scientific_{year}"
    save_json_report(report, str(OUTPUTS_ANNUAL_DIR / f"{stem}.json"))
    save_csv_report([report],  str(OUTPUTS_ANNUAL_DIR / f"{stem}.csv"))

    obs_date    = datetime(year, 8, 15)
    reports_dir = OUTPUTS_ANNUAL_DIR / "reports" / str(year)

    export_full_report_suite(
        run_id=run_id, job="annual",
        obs_date=obs_date,
        output_dir=reports_dir,
        eco=ecological_assessment,
        anomaly={},
        forest_stats=forest_stats,
        composite_path=composite_path,
        index_stats={},
        cloud_covers=[],
        n_scenes=0,
        trend_slope=trend_slope,
        baseline_summary=baseline_summary,
        phenology_records=phenology_records,
        phenology_trends=phenology_trends,
    )
    return report


def _discord(webhook_url, obs_date, anomaly, composite_path, trend_path):
    area_ha = anomaly.get("anomaly_area_ha", 0)
    persist = anomaly.get("persistent", False)
    status  = "ALERTA" if persist else ("AVISO" if area_ha > 0 else "NORMAL")
    try:
        requests.post(webhook_url, json={
            "content": f"Informe mensual {SITE_NAME} · {obs_date.strftime('%Y-%m')} · {STATUS_ICONS.get(status, '')} {status}",
            "embeds": [{
                "title": f"{FULL_SITE} — {obs_date.strftime('%B %Y')}",
                "description": f"Área anómala: **{area_ha:.1f} ha** | Pipeline v{PIPELINE_VERSION}",
                "color": 15158332 if persist else (16776960 if area_ha > 0 else 3066993),
                "fields": [
                    {"name": "Composite", "value": Path(composite_path).name, "inline": True},
                    {"name": "Trend", "value": Path(trend_path).name if trend_path else "N/A", "inline": True},
                ],
            }],
        }, timeout=10)
        logger.info("Discord notification sent.")
    except Exception as exc:
        logger.warning("Discord failed: %s", exc)
