"""
Main pipeline orchestrator — Hayedo de Montejo Monitoring System v3.0

Job types:
  weekly   — catalogue query, new acquisition metadata
  monthly  — rolling composite, anomaly detection, climate context, ecological assessment
  annual   — scientific phenological-window composite, baseline update, phenology trends
  baseline — download and process historical data (2021–present)
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CONFIG_DIR, RAW_DIR, PROCESSED_DIR, COMPOSITES_DIR,
    OUTPUTS_ANOMALIES_DIR, OUTPUTS_MONTHLY_DIR, OUTPUTS_ANNUAL_DIR,
    LOG_DIR, QUARANTINE_DIR,
    InsufficientDataError, PipelineError,
    ensure_directories, load_yaml_config, save_json, setup_logging, fmt_date,
)
import ingestion
import compositing
import anomaly_detection
import reporting
import archive
import ecology
import forest_boundary
import baseline_builder
import climate as climate_module
import phenology as phenology_module
import forest_health_index as fhi_module
import management as management_module
import cartography
import validation
import defensibility
import thesis_report

logger = setup_logging("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _date_in_season(d: date, start: int, end: int) -> bool:
    return start <= d.month <= end


def _scan(directory: Path, suffix: str) -> List[str]:
    return sorted(str(p) for p in directory.glob(f"*_{suffix}.tif"))


def _scan_by_glob(directory: Path, pattern: str) -> List[str]:
    return sorted(str(p) for p in directory.glob(pattern))


def _extract_date_from_path(path: str) -> date:
    stem = Path(path).stem
    for token in stem.split("_"):
        if len(token) == 8 and token.isdigit():
            return datetime.strptime(token, "%Y%m%d").date()
    raise ValueError(f"Unable to parse date from {path}")


def _collect_index_stats_forest_masked(
    index_suffix: str,
    forest_mask_arr,
) -> Dict[str, float]:
    """Compute statistics for an index, restricted to forest pixels."""
    import numpy as np, rasterio
    paths = _scan(PROCESSED_DIR, index_suffix)
    if not paths:
        return {}
    vals = []
    for p in paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype("float32")
            nd  = src.nodata
            if nd is not None:
                arr[arr == nd] = float("nan")
            if forest_mask_arr is not None and forest_mask_arr.shape == arr.shape:
                arr[~forest_mask_arr] = float("nan")
            vals.extend(arr[np.isfinite(arr)].tolist())
    if not vals:
        return {}
    a = np.array(vals, dtype="float32")
    return {
        "mean":         float(np.mean(a)),
        "median":       float(np.median(a)),
        "std":          float(np.std(a)),
        "p10":          float(np.percentile(a, 10)),
        "p90":          float(np.percentile(a, 90)),
        "valid_pixels": len(a),
        "forest_masked": forest_mask_arr is not None,
    }


def _load_cloud_covers(metadata_dir: Path) -> List[float]:
    candidates = sorted(metadata_dir.glob("acquisitions_*.json"))
    if not candidates:
        return []
    try:
        data = json.loads(candidates[-1].read_text())
        return [p.get("cloudcoverpercentage") or 0.0 for p in data]
    except Exception:
        return []


def _build_forest_mask(composite_path: str, config: Dict) -> tuple:
    """
    Build and return (forest_mask_array, polygon_geojson, boundary_status).
    Returns (None, None, 'unavailable') on failure.
    """
    if not config.get("forest_boundary", {}).get("apply_mask", True):
        return None, None, "disabled"

    try:
        import rasterio
        polygon = forest_boundary.get_forest_polygon(
            try_overpass=config.get("forest_boundary", {}).get("try_overpass", True),
        )
        status = polygon["features"][0]["properties"].get("boundary_status", "unknown")
        with rasterio.open(composite_path) as src:
            profile = src.profile
        mask_arr = forest_boundary.create_forest_mask_array(profile, polygon)
        n_px = int(mask_arr.sum())
        if n_px == 0:
            logger.warning(
                "Forest polygon (%s) has ZERO overlap with the composite extent. "
                "The imagery does not cover the forest -- re-run ingestion/preprocessing "
                "for the corrected AOI (config/aoi.geojson) before trusting forest stats.",
                status)
            status = f"{status}_NO_OVERLAP"
        else:
            logger.info("Forest mask applied (%s polygon, %d pixels ~ %.1f ha).",
                        status, n_px, n_px * 100 / 10000)
        return mask_arr, polygon, status
    except Exception as exc:
        logger.warning("Forest mask unavailable: %s — using all pixels.", exc)
        return None, None, "unavailable"


def _run_job_with_archive(job_fn, job_name, run_id, **kwargs) -> bool:
    try:
        job_fn(**kwargs)
        archive.close_run(run_id, status="success")
        return True
    except InsufficientDataError as exc:
        logger.warning("[%s] Skipped — insufficient data: %s", job_name, exc)
        archive.close_run(run_id, status="skipped", error=str(exc))
        return True
    except Exception as exc:
        logger.exception("[%s] FATAL: %s", job_name, exc)
        archive.close_run(run_id, status="error", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Job: weekly
# ---------------------------------------------------------------------------

def weekly_job(config_path=None, aoi_path=None):
    config = load_yaml_config(config_path)
    window = config.get("operational", {}).get("weekly_window_days", 7)
    meta_path = ingestion.collect_new_acquisitions(
        window_days=window, config_path=config_path, aoi_path=aoi_path,
    )
    logger.info("Weekly job completed. Metadata: %s", meta_path)


# ---------------------------------------------------------------------------
# Job: monthly
# ---------------------------------------------------------------------------

def monthly_job(config_path=None, aoi_path=None, run_date=None, run_id=None):
    if run_date is None:
        run_date = date.today()
    config  = load_yaml_config(config_path)
    op_cfg  = config.get("operational", {})
    ad_cfg  = config.get("anomaly_detection", {})
    ae_cfg  = config.get("aemet", {})

    if not _date_in_season(run_date, op_cfg.get("active_season_start_month", 4),
                           op_cfg.get("active_season_end_month", 10)):
        logger.info("Date %s outside active season; skipping monthly job.", run_date)
        return

    weekly_job(config_path=config_path, aoi_path=aoi_path)

    index_paths = _scan(PROCESSED_DIR, "NDVI")
    if not index_paths:
        raise InsufficientDataError("No NDVI index files found for rolling composite.")

    run_dt      = datetime.combine(run_date, datetime.min.time())
    window_days = op_cfg.get("rolling_window_days", 45)
    rolling_p   = COMPOSITES_DIR / f"rolling_ndvi_{fmt_date(run_dt)}.tif"

    rolling_composite = compositing.build_rolling_composite(
        index_paths, reference_date=run_dt, window_days=window_days,
        out_path=str(rolling_p),
    )
    n_scenes = len(compositing.select_recent_observations(index_paths, run_dt, window_days))

    # --- forest mask ---
    forest_mask_arr, polygon, boundary_status = _build_forest_mask(str(rolling_composite), config)

    # --- composite stats (forest-masked) ---
    composite_stats = compositing.summarize_composite(str(rolling_composite), forest_mask=forest_mask_arr)
    composite_stats["boundary_status"] = boundary_status

    # --- trend ---
    trend_path, trend_slope = None, None
    if len(index_paths) >= 3:
        try:
            trend_out  = COMPOSITES_DIR / f"ndvi_trend_{fmt_date(run_dt)}.tif"
            trend_path = compositing.compute_trend(index_paths, output_path=str(trend_out))
            trend_slope = compositing.summarize_trend(str(trend_path))
        except Exception as exc:
            logger.warning("Trend computation failed: %s", exc)

    # --- anomaly detection ---
    method       = ad_cfg.get("method", "threshold")
    threshold    = op_cfg.get("anomaly_threshold",        -0.15)
    min_persist  = op_cfg.get("persistence_observations",     2)
    z_thresh     = ad_cfg.get("z_threshold",              -2.0)
    min_years    = ad_cfg.get("min_baseline_years",           3)

    # Prefer baseline composites from the dedicated baseline directory
    baseline_candidates = (
        _scan_by_glob(COMPOSITES_DIR / "baseline", "annual_ndvi_????.tif") or
        _scan_by_glob(COMPOSITES_DIR, "annual_ndvi_????.tif")
    )
    baseline_path = baseline_mean = baseline_std = None

    if baseline_candidates:
        baseline_path = str(anomaly_detection.compute_baseline_median(baseline_candidates))
        if method == "zscore" and len(baseline_candidates) >= min_years:
            mean_p, std_p = anomaly_detection.build_phenological_baseline(
                baseline_candidates, min_years=min_years,
            )
            baseline_mean = str(mean_p)
            baseline_std  = str(std_p)
        elif method == "zscore":
            logger.warning("Z-score needs >= %d composites (%d available); fallback to threshold.",
                           min_years, len(baseline_candidates))
    else:
        logger.warning("No annual composites for baseline.")

    anomaly_payload = {"note": "Baseline unavailable."} if not baseline_path else {}
    if baseline_path:
        prev_masks = _scan_by_glob(OUTPUTS_ANOMALIES_DIR, "anomaly_mask_*.tif")
        anomaly_payload = anomaly_detection.anomaly_report(
            current_path=str(rolling_composite),
            baseline_path=baseline_path,
            previous_masks=prev_masks,
            threshold=threshold,
            min_persistence=min_persist,
            method=method,
            baseline_mean_path=baseline_mean,
            baseline_std_path=baseline_std,
            z_threshold=z_thresh,
        )

    # --- index stats (forest-masked) ---
    index_stats: Dict[str, Dict] = {}
    for idx_name in ["NDVI", "NDMI", "NDRE", "NBR"]:
        s = _collect_index_stats_forest_masked(idx_name, forest_mask_arr)
        if s:
            index_stats[idx_name] = s

    # --- ecological assessment ---
    eco = ecology.full_ecological_assessment(
        observation_date=run_date,
        ndvi_stats=index_stats.get("NDVI", composite_stats),
        ndmi_stats=index_stats.get("NDMI"),
        ndre_stats=index_stats.get("NDRE"),
        nbr_stats=index_stats.get("NBR"),
    )

    # --- climate integration ---
    climate_context = climate_corr = None
    aemet_key = ae_cfg.get("api_key", "")
    if aemet_key:
        try:
            climate_records = climate_module.fetch_climate_data(
                api_key=aemet_key,
                year_start=run_date.year - 1,
                year_end=run_date.year,
                station_priority=ae_cfg.get("primary_station", "montejo_de_la_sierra"),
            )
            if climate_records:
                climate_context = climate_module.get_climate_context_for_period(
                    climate_records, run_date.year,
                    month_start=max(1, run_date.month - 2),
                    month_end=run_date.month,
                )
                # NDVI-precipitation correlation if enough data
                if index_stats.get("NDVI"):
                    ndvi_dates = [
                        _extract_date_from_path(p)
                        for p in _scan(PROCESSED_DIR, "NDVI")
                    ]
                    ndvi_meds  = [index_stats["NDVI"].get("median")] * len(ndvi_dates)
                    climate_corr = climate_module.compute_ndvi_climate_correlation(
                        ndvi_meds, ndvi_dates, climate_records, "precip_mm",
                    )
        except Exception as exc:
            logger.warning("Climate integration failed: %s", exc)

    # --- phenology (from historical record) ---
    phenology_records = phenology_trends = None
    try:
        annual_stats = {
            int(Path(p).stem.split("_")[-1]): compositing.summarize_composite(p, forest_mask=forest_mask_arr)
            for p in baseline_candidates
        }
        if annual_stats:
            phenology_records = phenology_module.build_phenology_time_series(annual_stats)
            phenology_trends  = phenology_module.analyse_phenology_trends(phenology_records)
            phenology_module.save_phenology_results(phenology_records, phenology_trends)
    except Exception as exc:
        logger.warning("Phenology analysis failed: %s", exc)

    # --- baseline summary ---
    bsummary = baseline_builder.baseline_summary()

    # --- Forest Health Index (Phase 8) ---
    fhi = None
    try:
        weights = config.get("forest_health_index", {}).get("weights") or fhi_module.DEFAULT_WEIGHTS
        fhi = fhi_module.compute_forest_health_index(
            observation_date=run_date,
            ndvi_stats=index_stats.get("NDVI", composite_stats),
            ndmi_stats=index_stats.get("NDMI"),
            ndre_stats=index_stats.get("NDRE"),
            climate_context=climate_context,
            phenology_records=phenology_records,
            weights=weights,
        )
        fhi_module.save_fhi(fhi, label=run_date.strftime("%Y-%m"))
        logger.info("FHI = %s (%s)", fhi.get("fhi_score"), fhi.get("class"))
    except Exception as exc:
        logger.warning("FHI computation failed: %s", exc)

    # --- management / conservation decision support (Phase 9) ---
    try:
        mgmt = management_module.generate_management_report(
            observation_date=run_date,
            ecological_assessment=eco,
            fhi=fhi,
            climate_context=climate_context,
            anomaly=anomaly_payload,
            trend_slope=trend_slope,
            baseline_summary=bsummary,
        )
        management_module.save_management_report(mgmt, label=run_date.strftime("%Y-%m"))
    except Exception as exc:
        logger.warning("Management report failed: %s", exc)

    # --- scientific validation document (Phase 7) ---
    try:
        validation.save_validation_document(observation_date=run_date.isoformat())
    except Exception as exc:
        logger.warning("Validation document failed: %s", exc)

    # --- publication-quality figures (Phase 10) ---
    try:
        idx_paths = {k: _scan(PROCESSED_DIR, k)[-1] for k in ["NDMI", "NDRE", "NBR"]
                     if _scan(PROCESSED_DIR, k)}
        cartography.render_standard_figures(
            composite_path=str(rolling_composite),
            index_paths=idx_paths,
            trend_path=str(trend_path) if trend_path else None,
            phenology_records=phenology_records,
            fhi=fhi,
            forest_mask=forest_mask_arr,
            observation_label=run_date.isoformat(),
            out_dir=str(OUTPUTS_MONTHLY_DIR / "figures" / run_date.strftime("%Y-%m")),
        )
    except Exception as exc:
        logger.warning("Figure rendering failed: %s", exc)

    # --- report ---
    cloud_covers = _load_cloud_covers(RAW_DIR)
    webhook = config.get("reporting", {}).get("discord_webhook_url", "")

    report = reporting.export_monthly_summary(
        observation_date=run_dt,
        anomaly_payload=anomaly_payload,
        trend_path=str(trend_path) if trend_path else None,
        composite_path=str(rolling_composite),
        webhook_url=webhook,
        run_id=run_id,
        ecological_assessment=eco,
        composite_stats=composite_stats,
        forest_stats=composite_stats,
        index_stats=index_stats,
        cloud_covers=cloud_covers,
        n_scenes=n_scenes,
        trend_slope=trend_slope,
        climate_context=climate_context,
        climate_corr=climate_corr,
        phenology_records=phenology_records,
        phenology_trends=phenology_trends,
        baseline_summary=bsummary,
    )

    # --- archive ---
    if run_id:
        out_files = [str(rolling_composite)]
        if trend_path:
            out_files.append(str(trend_path))
        archive.archive_outputs(run_id, out_files)
        rd = OUTPUTS_MONTHLY_DIR / "reports" / run_date.strftime("%Y-%m")
        if rd.exists():
            archive.archive_reports(run_id, [str(p) for p in rd.glob("*.md")])

    logger.info("Monthly job complete for %s.", run_date.strftime("%Y-%m"))


# ---------------------------------------------------------------------------
# Job: annual
# ---------------------------------------------------------------------------

def annual_job(config_path=None, aoi_path=None, year=None, run_id=None):
    if year is None:
        year = date.today().year
    logger.info("Annual scientific composite for year %d (Jul–Sep)", year)
    config = load_yaml_config(config_path)

    start_date = date(year, 7, 1)
    end_date   = date(year, 9, 30)

    ndvi_paths = [
        p for p in _scan(PROCESSED_DIR, "NDVI")
        if start_date <= _extract_date_from_path(p) <= end_date
    ]
    if not ndvi_paths:
        raise InsufficientDataError(
            f"No NDVI observations in scientific window {start_date}–{end_date}."
        )

    output_path      = COMPOSITES_DIR / f"annual_ndvi_{year}.tif"
    annual_composite = compositing.build_median_composite(ndvi_paths, out_path=str(output_path))

    forest_mask_arr, _, boundary_status = _build_forest_mask(str(annual_composite), config)
    stats = compositing.summarize_composite(str(annual_composite), forest_mask=forest_mask_arr)
    stats["boundary_status"] = boundary_status

    trend_slope = None
    if len(ndvi_paths) >= 3:
        try:
            tp          = compositing.compute_trend(ndvi_paths, output_path=str(COMPOSITES_DIR / f"annual_ndvi_trend_{year}.tif"))
            trend_slope = compositing.summarize_trend(str(tp))
        except Exception as exc:
            logger.warning("Trend computation failed: %s", exc)

    obs_date = datetime(year, 8, 15)
    eco = ecology.full_ecological_assessment(
        observation_date=obs_date.date(), ndvi_stats=stats,
    )

    # Phenology from all available annual composites
    baseline_candidates = (
        _scan_by_glob(COMPOSITES_DIR / "baseline", "annual_ndvi_????.tif") or
        _scan_by_glob(COMPOSITES_DIR, "annual_ndvi_????.tif")
    )
    phenology_records = phenology_trends = None
    if baseline_candidates:
        try:
            annual_stats = {
                int(Path(p).stem.split("_")[-1]): compositing.summarize_composite(p, forest_mask=forest_mask_arr)
                for p in baseline_candidates
            }
            phenology_records = phenology_module.build_phenology_time_series(annual_stats)
            phenology_trends  = phenology_module.analyse_phenology_trends(phenology_records)
            phenology_module.save_phenology_results(phenology_records, phenology_trends, label=str(year))
        except Exception as exc:
            logger.warning("Phenology analysis failed: %s", exc)

    bsummary = baseline_builder.baseline_summary()

    # --- Forest Health Index + management + figures (Phases 8-10) ---
    ndmi_stats = _collect_index_stats_forest_masked("NDMI", forest_mask_arr) or None
    ndre_stats = _collect_index_stats_forest_masked("NDRE", forest_mask_arr) or None
    fhi = None
    try:
        weights = config.get("forest_health_index", {}).get("weights") or fhi_module.DEFAULT_WEIGHTS
        fhi = fhi_module.compute_forest_health_index(
            observation_date=obs_date.date(), ndvi_stats=stats,
            ndmi_stats=ndmi_stats, ndre_stats=ndre_stats,
            phenology_records=phenology_records, weights=weights,
        )
        fhi_module.save_fhi(fhi, label=str(year))
        logger.info("Annual FHI = %s (%s)", fhi.get("fhi_score"), fhi.get("class"))
    except Exception as exc:
        logger.warning("Annual FHI failed: %s", exc)
    try:
        mgmt = management_module.generate_management_report(
            observation_date=obs_date.date(), ecological_assessment=eco, fhi=fhi,
            trend_slope=trend_slope, baseline_summary=bsummary)
        management_module.save_management_report(mgmt, label=str(year))
    except Exception as exc:
        logger.warning("Annual management report failed: %s", exc)
    try:
        validation.save_validation_document(observation_date=obs_date.date().isoformat())
        idx_paths = {k: _scan(PROCESSED_DIR, k)[-1] for k in ["NDMI", "NDRE", "NBR"]
                     if _scan(PROCESSED_DIR, k)}
        cartography.render_standard_figures(
            composite_path=str(annual_composite), index_paths=idx_paths,
            phenology_records=phenology_records, fhi=fhi, forest_mask=forest_mask_arr,
            observation_label=str(year),
            out_dir=str(OUTPUTS_ANNUAL_DIR / "figures" / str(year)))
    except Exception as exc:
        logger.warning("Annual figures/validation failed: %s", exc)

    report = reporting.export_annual_scientific_report(
        year=year,
        composite_path=str(annual_composite),
        statistics=stats,
        trend_slope=trend_slope,
        run_id=run_id,
        ecological_assessment=eco,
        forest_stats=stats,
        baseline_summary=bsummary,
        phenology_records=phenology_records,
        phenology_trends=phenology_trends,
    )

    if run_id:
        archive.archive_outputs(run_id, [str(annual_composite)])
        rd = OUTPUTS_ANNUAL_DIR / "reports" / str(year)
        if rd.exists():
            archive.archive_reports(run_id, [str(p) for p in rd.glob("*.md")])

    logger.info("Annual scientific report written for %d.", year)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ensure_directories()
    parser = argparse.ArgumentParser(
        description="Hayedo de Montejo — Forest Monitoring Pipeline v3.0"
    )
    parser.add_argument("--job", choices=["weekly", "monthly", "annual", "baseline"])
    parser.add_argument("--year",   type=int)
    parser.add_argument("--years",  nargs="+", type=int)
    parser.add_argument("--config", default=str(CONFIG_DIR / "thresholds.yaml"))
    parser.add_argument("--aoi",    default=str(CONFIG_DIR / "aoi.geojson"))
    parser.add_argument("--thesis-report", action="store_true",
                        help="Generate the full academic thesis report package (Phase 11).")
    parser.add_argument("--validation", action="store_true",
                        help="Generate the Scientific Validation Document (Phase 7).")
    parser.add_argument("--defensibility", action="store_true",
                        help="Generate the TFM Defensibility Report (Phase 12).")
    args = parser.parse_args()

    # --- standalone export modes (no satellite processing) ---
    if args.thesis_report:
        path = thesis_report.run_thesis_report(config_path=args.config, year=args.year)
        logger.info("Thesis report package written: %s", path)
        print(f"Thesis report: {path}")
        sys.exit(0)
    if args.validation:
        path = validation.save_validation_document()
        print(f"Scientific Validation Document: {path}")
        sys.exit(0)
    if args.defensibility:
        path = defensibility.save_defensibility_report()
        print(f"TFM Defensibility Report: {path}")
        sys.exit(0)

    if not args.job:
        parser.error("one of --job, --thesis-report, --validation or --defensibility is required")

    run_id = archive.new_run_id(args.job)
    rdir   = archive.init_run(run_id, args.job, args.config)
    logger.info("Run ID: %s  ->  %s", run_id, rdir)

    if args.job == "baseline":
        years = args.years or list(range(2021, date.today().year))
        if args.year:
            years = [args.year]
        logger.info("Building historical baseline for years: %s", years)
        for yr in years:
            try:
                baseline_builder.build_annual_composite_for_year(
                    yr, config_path=args.config, aoi_path=args.aoi,
                )
            except Exception as exc:
                logger.error("Baseline year %d failed: %s", yr, exc)
        annual = baseline_builder.get_annual_composites()
        if len(annual) >= 3:
            baseline_builder.build_baseline_statistics(annual)
            logger.info("Baseline statistics rebuilt from %d annual composites.", len(annual))
        archive.close_run(run_id, status="success")
        return

    def _weekly():  weekly_job(config_path=args.config, aoi_path=args.aoi)
    def _monthly(): monthly_job(config_path=args.config, aoi_path=args.aoi, run_id=run_id)
    def _annual():  annual_job(config_path=args.config, aoi_path=args.aoi, year=args.year, run_id=run_id)

    job_map = {"weekly": _weekly, "monthly": _monthly, "annual": _annual}
    success = _run_job_with_archive(job_map[args.job], args.job, run_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
