# Forest Monitoring Pipeline

A modular production-grade Sentinel-2 pipeline for deciduous beech forest monitoring in Sierra del Rinc?n, Spain.

## Overview

This system separates two temporal layers:

- **Scientific annual layer**: July 1 � September 30 median NDVI composite for thesis-ready output.
- **Operational monitoring layer**: rolling 30�45 day NDVI composite, trend, and anomaly detection during April�October.

The pipeline is designed to be config-driven, reproducible, and scheduler-friendly.

## Repository layout

- `src/` - pipeline modules
- `config/` - `aoi.geojson`, `thresholds.yaml` (secrets live in `.env`, not here)
- `scripts/` - maintenance utilities (`build_mfe50_boundary.py`, `audit_and_quarantine.py`)
- `run_pipeline.ps1` - lightweight wrapper for Windows
- `n8n/pipeline_workflow.json` - n8n workflow export

### Consolidated output structure (single inspectable tree)

All working data and artifacts live under `outputs/` so nothing is scattered:

- `outputs/raw_downloads/` - **all** raw Sentinel SAFE / ZIP / acquisition metadata (`baseline_staging/<year>/` during ingestion)
- `outputs/processed/` - per-scene 8-band TIFs and spectral-index TIFs
- `outputs/composites/` - rolling / annual composites; `composites/baseline/` holds annual baseline composites, per-year reports and statistics
- `outputs/reports/` - consolidated report landing area
- `outputs/logs/` - runtime logs
- `outputs/quarantine/` - flagged / obsolete / corrupt / AOI-rejected artifacts (never deleted silently; each carries an audit log)
- plus the research outputs: `annual/`, `monthly/`, `anomalies/`, `validation/`, `forest_health_index/`, `management/`, `figures/`, `thesis/`, `defensibility/`, `climate/`

Directory locations are defined once in `src/common.py`; the baseline staging and
composite paths derive from them automatically.

### Ingestion safeguards (production hardening)

- **Secrets** are read from a git-ignored `.env` via environment variables (see `.env.example`).
- **AOI validation** (`src/aoi_validation.py`): every scene is accepted only if its real
  footprint geometry intersects the corrected official AOI (not by name), and every produced
  raster is re-checked against the AOI after download; failures are quarantined and logged.
- **Disk guards**: a per-scene free-space check skips downloads when headroom is low; scenes
  are downloaded, processed and deleted one at a time to bound peak disk use.
- **Checkpointing / resumability**: a year whose composite already exists is skipped, so an
  interrupted baseline run resumes without re-downloading completed years. Each year writes
  `composites/baseline/annual_ndvi_<year>_report.json` (scenes discovered/accepted/rejected,
  download/processing sizes).
- **Audit & quarantine**: `scripts/audit_and_quarantine.py` produces a storage audit and moves
  obsolete-AOI / incomplete / corrupt artifacts to `outputs/quarantine/` with a full log.

## Installation

```powershell
python -m pip install -r requirements.txt
```

If you use the local Python interpreter directly:

```powershell
C:/Users/Dell/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pip install -r requirements.txt
```

## Configuration

Edit `config/thresholds.yaml` before running:

- `api.username` / `api.password` for Copernicus API access
- `api.cloud_cover_max` for acquisition filtering
- `operational.rolling_window_days` for rolling composite length
- `operational.active_season_start_month` / `end_month`
- `operational.anomaly_threshold` and `persistence_observations`
- `reporting.discord_webhook_url` for optional notifications

Verify `config/aoi.geojson` and replace the polygon with the exact Sierra del Rinc?n AOI if necessary.

## Run examples

### Using the wrapper script

```powershell
.\run_pipeline.ps1 -Job weekly
.\run_pipeline.ps1 -Job monthly
.\run_pipeline.ps1 -Job annual -Year 2025
```

### Direct Python commands

```powershell
python src/pipeline.py --job weekly
python src/pipeline.py --job monthly
python src/pipeline.py --job annual --year 2025
```

The wrapper is a lightweight convenience script that forwards the job request to `src/pipeline.py`.

### Research / thesis export modes (Phases 7-12)

These modes operate on the products already on disk and do not download or
re-process satellite data:

```powershell
python src/pipeline.py --thesis-report           # full academic report package (Phase 11)
python src/pipeline.py --thesis-report --year 2025
python src/pipeline.py --validation              # Scientific Validation Document (Phase 7)
python src/pipeline.py --defensibility           # TFM Defensibility Report (Phase 12)
```

The monthly and annual jobs additionally generate the Forest Health Index,
the management decision-support brief and publication-quality figures on every
run.

## Research-grade outputs (Phases 7-12)

The system goes beyond raw indices to produce decision-support and thesis-ready
material:

- **Scientific validation** (`src/validation.py`) — per-indicator ecological
  meaning, limitations, uncertainty, beech-specific interpretation and
  false-positive analysis for NDVI, NDMI, NDRE, NBR, SPI, climate anomalies,
  z-score, trend and phenology. Output: `outputs/validation/`.
- **Forest Health Index** (`src/forest_health_index.py`) — a transparent,
  reproducible, weighted composite (NDVI 0.30 / NDMI 0.25 / NDRE 0.20 /
  climate 0.15 / phenology 0.10; weights and rationale in the module and in
  `config/thresholds.yaml`). Produces a 0-100 score and an
  Excellent/Good/Moderate/Warning/Critical class, with full sub-score and
  weight breakdown and a data-completeness confidence field. Output:
  `outputs/forest_health_index/`.
- **Management & conservation brief** (`src/management.py`) — six manager-facing
  blocks: condition assessment, ranked stress factors, drought-risk
  interpretation, visitor-pressure implications, conservation implications
  (Natura 2000 habitats 9120/9150) and monitoring recommendations. Output:
  `outputs/management/`.
- **Publication cartography** (`src/cartography.py`) — 300-dpi maps with
  titles, subtitles, colour legends, metric scale bars, north arrows and a
  provenance footer, plus phenology and FHI figures. Output: `outputs/figures/`
  and per-run `figures/` folders.
- **Thesis report package** (`src/thesis_report.py`) — a single academic
  document with the ten required sections (executive summary, methodology,
  results, trend, phenology, climate, FHI, discussion, limitations,
  conclusions) plus its figure set. Output: `outputs/thesis/<year>/`.
- **TFM Defensibility audit** (`src/defensibility.py`) — an adversarial,
  committee-style audit across scientific, statistical, GIS, remote-sensing and
  reproducibility domains, each finding with risk / impact / solution, a
  severity/status flag, a readiness score and a prioritised remediation
  roadmap. Output: `outputs/defensibility/`.

## Scheduling

### Cron examples

Weekly job (every Monday at 02:00):

```cron
0 2 * * 1 cd /path/to/Pipeline && python src/pipeline.py --job weekly >> logs/cron_weekly.log 2>&1
```

Monthly job (first day of month at 03:00):

```cron
0 3 1 * * cd /path/to/Pipeline && python src/pipeline.py --job monthly >> logs/cron_monthly.log 2>&1
```

Annual job (October 1 at 04:00):

```cron
0 4 1 10 * cd /path/to/Pipeline && python src/pipeline.py --job annual --year $(date +\%Y) >> logs/cron_annual.log 2>&1
```

### n8n workflow

Import `n8n/pipeline_workflow.json` into n8n to create scheduled nodes for weekly, monthly, and annual execution.

## Next step

1. Populate `config/thresholds.yaml` with your Copernicus credentials and desired thresholds.
2. Validate or replace `config/aoi.geojson` for the exact Sierra del Rinc?n forest boundary.
3. Run the appropriate job via the wrapper script or direct Python command.
