"""
Publication-quality cartography and figure module — Phase 10.

Produces thesis-ready raster maps and scientific figures from the monitoring
system's GeoTIFF outputs, with consistent legends, scale bars, north arrows,
academic titles and high-resolution PNG export (300 dpi by default).

The intent is that any figure produced here can be dropped directly into a
master's thesis (TFM) without further editing: every map carries a title, a
subtitle with the site and sensor provenance, a labelled colour legend, a
metric scale bar derived from the raster transform, a north arrow, and a
provenance footer (CRS, date, pipeline version).

Design choices
--------------
- Diverging colormaps (RdYlGn) for vegetation indices so that "red = stressed,
  green = healthy" reads intuitively for non-specialist managers.
- Index-specific, fixed value ranges so colours are comparable across dates
  (a green pixel always means the same NDVI, regardless of the scene).
- Scale bar length chosen automatically as a round number (~1/4 of map width).
- Robust to missing matplotlib backend: all rendering uses the Agg backend.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / file-only backend
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.patches import FancyArrow, Rectangle

import rasterio

from common import WORKSPACE_ROOT, setup_logging

logger = setup_logging("cartography")

FIGURES_DIR = WORKSPACE_ROOT / "outputs" / "figures"

SITE_NAME = "Hayedo de Montejo"
RESERVE   = "Reserva de la Biosfera Sierra del Rincon"
SENSOR    = "Sentinel-2 MSI L2A (ESA Copernicus / CDSE)"
PIPELINE_VERSION = "3.0"

# Index display profiles: (long name, cmap, vmin, vmax, units/label)
INDEX_PROFILES: Dict[str, Dict[str, Any]] = {
    "NDVI": {"name": "Normalized Difference Vegetation Index",
             "cmap": "RdYlGn", "vmin": 0.0, "vmax": 1.0, "label": "NDVI"},
    "NDMI": {"name": "Normalized Difference Moisture Index",
             "cmap": "RdYlBu", "vmin": -0.3, "vmax": 0.4, "label": "NDMI"},
    "NDRE": {"name": "Red-Edge NDVI (chlorophyll)",
             "cmap": "RdYlGn", "vmin": 0.0, "vmax": 0.5, "label": "NDRE"},
    "NBR":  {"name": "Normalized Burn Ratio",
             "cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.6, "label": "NBR"},
    "TREND": {"name": "Per-pixel NDVI trend (Sen's slope)",
              "cmap": "RdBu", "vmin": -0.03, "vmax": 0.03, "label": "NDVI / year"},
    "ZSCORE": {"name": "NDVI anomaly (z-score vs. baseline)",
               "cmap": "RdBu", "vmin": -3.0, "vmax": 3.0, "label": "z-score (sigma)"},
}


# ---------------------------------------------------------------------------
# Cartographic furniture
# ---------------------------------------------------------------------------

def _add_north_arrow(ax, x: float = 0.94, y: float = 0.92) -> None:
    """Draw a simple north arrow in axes (0-1) coordinates."""
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - 0.09),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=10),
        ha="center", va="center", fontsize=13, fontweight="bold",
    )


def _add_scale_bar(ax, pixel_size_m: float, img_width_px: int,
                   crs_is_metric: bool) -> None:
    """
    Draw a metric scale bar sized to ~1/4 of the map width, rounded to a nice
    number. pixel_size_m is the ground size of one pixel in metres.
    """
    if not crs_is_metric or pixel_size_m <= 0:
        return
    target_m = pixel_size_m * img_width_px * 0.25
    nice = _nice_round(target_m)
    bar_px = nice / pixel_size_m
    frac = bar_px / img_width_px

    x0 = 0.05
    y0 = 0.06
    # bar
    ax.add_patch(Rectangle((x0, y0), frac, 0.012, transform=ax.transAxes,
                           facecolor="black", edgecolor="black", zorder=10))
    label = f"{nice/1000:.0f} km" if nice >= 1000 else f"{nice:.0f} m"
    ax.text(x0 + frac / 2, y0 + 0.03, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, fontweight="bold")


def _nice_round(value: float) -> float:
    """Round to 1/2/5 x 10^n."""
    if value <= 0:
        return 100.0
    exp = np.floor(np.log10(value))
    base = value / (10 ** exp)
    if base < 1.5:
        nice = 1
    elif base < 3.5:
        nice = 2
    elif base < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


def _crs_is_metric(crs) -> Tuple[bool, str]:
    try:
        if crs is None:
            return False, "unknown CRS"
        epsg = crs.to_epsg()
        # geographic CRS (EPSG:4326 etc.) are degree-based
        is_geographic = getattr(crs, "is_geographic", False)
        return (not is_geographic), (f"EPSG:{epsg}" if epsg else crs.to_string())
    except Exception:
        return False, "unknown CRS"


# ---------------------------------------------------------------------------
# Core single-raster map
# ---------------------------------------------------------------------------

def render_index_map(
    raster_path: str,
    index_key: str,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    observation_label: str = "",
    forest_mask: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Render a single index/anomaly/trend raster as a publication-quality map.

    Parameters
    ----------
    raster_path : path to a single-band GeoTIFF.
    index_key   : one of INDEX_PROFILES (NDVI, NDMI, NDRE, NBR, TREND, ZSCORE).
    forest_mask : optional boolean array (True = forest) to clip the display to
                  the beech polygon.
    """
    profile = INDEX_PROFILES.get(index_key.upper(), INDEX_PROFILES["NDVI"])
    raster_path = str(raster_path)

    try:
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
    except Exception as exc:
        logger.warning("Could not open raster %s: %s", raster_path, exc)
        return None

    if nodata is not None:
        arr[arr == nodata] = np.nan
    if forest_mask is not None and forest_mask.shape == arr.shape:
        arr = np.where(forest_mask, arr, np.nan)

    if not np.isfinite(arr).any():
        logger.warning("Raster %s has no valid pixels to plot.", raster_path)
        return None

    metric, crs_label = _crs_is_metric(crs)
    pixel_size_m = abs(transform.a) if metric else 0.0

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap(profile["cmap"]).copy()
    cmap.set_bad(color="#d9d9d9")  # grey for nodata / outside forest

    im = ax.imshow(arr, cmap=cmap, vmin=profile["vmin"], vmax=profile["vmax"],
                   extent=extent, origin="upper", interpolation="nearest")

    # Titles
    main_title = title or f"{profile['name']} — {SITE_NAME}"
    sub = subtitle or f"{RESERVE} · {SENSOR}"
    if observation_label:
        sub = f"{observation_label} · {sub}"
    ax.set_title(main_title, fontsize=14, fontweight="bold", pad=14)
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#444444")

    ax.set_xlabel("Easting (m)" if metric else "Longitude")
    ax.set_ylabel("Northing (m)" if metric else "Latitude")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=8)

    # Colour legend
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
    cbar.set_label(profile["label"], fontsize=10)

    _add_north_arrow(ax)
    _add_scale_bar(ax, pixel_size_m, arr.shape[1], metric)

    # Provenance footer
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    valid_px = int(np.isfinite(arr).sum())
    footer = (f"CRS: {crs_label}  |  valid pixels: {valid_px:,}  |  "
              f"generated {ts}  |  pipeline v{PIPELINE_VERSION}")
    fig.text(0.5, 0.015, footer, ha="center", fontsize=7, color="#666666")

    fig.tight_layout(rect=(0, 0.03, 1, 0.98))

    if out_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(raster_path).stem
        out_path = FIGURES_DIR / f"map_{index_key.lower()}_{stem}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Map written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Non-spatial scientific figures
# ---------------------------------------------------------------------------

def plot_phenology_time_series(
    phenology_records: List[Dict[str, Any]],
    out_path: Optional[str] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot the multi-year peak-NDVI series with a fitted linear trend."""
    if not phenology_records or len(phenology_records) < 2:
        logger.info("Not enough phenology records for a time-series figure.")
        return None

    years = [r["year"] for r in phenology_records if r.get("peak_ndvi") is not None]
    peaks = [r["peak_ndvi"] for r in phenology_records if r.get("peak_ndvi") is not None]
    if len(years) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years, peaks, "o-", color="#1b7837", linewidth=2, markersize=7,
            label="Peak-season NDVI (Jul-Sep)")

    if len(years) >= 3:
        coef = np.polyfit(years, peaks, 1)
        ax.plot(years, np.polyval(coef, years), "--", color="#762a83",
                label=f"Linear trend: {coef[0]:+.4f} NDVI/yr")

    ax.set_title(f"Peak-season NDVI trajectory — {SITE_NAME}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Peak NDVI")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.text(0.5, 0.01, f"{RESERVE} · {SENSOR} · pipeline v{PIPELINE_VERSION}",
             ha="center", fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    if out_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "phenology_time_series.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Phenology figure written: %s", out_path)
    return out_path


def plot_fhi_breakdown(fhi: Dict[str, Any], out_path: Optional[str] = None,
                       dpi: int = 300) -> Optional[Path]:
    """Horizontal bar chart of FHI sub-scores and a gauge-style headline."""
    if not fhi or fhi.get("fhi_score") is None:
        return None

    labels_map = {"ndvi": "Canopy vigour\n(NDVI)", "ndmi": "Water status\n(NDMI)",
                  "ndre": "Chlorophyll\n(NDRE)", "climate": "Climatic balance\n(SPI/Tanom)",
                  "phenology": "Phenology\ntrajectory"}
    keys, scores, colors = [], [], []
    for k in ["ndvi", "ndmi", "ndre", "climate", "phenology"]:
        sub = fhi["sub_scores"].get(k, {})
        if sub.get("available"):
            keys.append(labels_map[k])
            s = sub["score"]
            scores.append(s)
            colors.append(plt.get_cmap("RdYlGn")(s))

    if not keys:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(keys, scores, color=colors, edgecolor="#333333")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Health sub-score (0 = critical, 1 = optimal)")
    ax.set_title(f"Forest Health Index breakdown — FHI = {fhi['fhi_score']:.0f}/100 "
                 f"({fhi['class']})", fontsize=12, fontweight="bold")
    for i, s in enumerate(scores):
        ax.text(s + 0.01, i, f"{s:.2f}", va="center", fontsize=9)
    ax.axvline(0.55, color="#888888", linestyle=":", linewidth=1)
    ax.grid(True, axis="x", alpha=0.3)
    fig.text(0.5, 0.01, f"{SITE_NAME} · methodology {fhi.get('methodology_version','FHI-1.0')} "
             f"· data completeness {fhi.get('data_completeness',1)*100:.0f}%",
             ha="center", fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    if out_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "fhi_breakdown.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("FHI figure written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: render the standard figure set for a run
# ---------------------------------------------------------------------------

def render_standard_figures(
    composite_path: Optional[str],
    index_paths: Optional[Dict[str, str]] = None,
    trend_path: Optional[str] = None,
    phenology_records: Optional[List[Dict]] = None,
    fhi: Optional[Dict] = None,
    forest_mask: Optional[np.ndarray] = None,
    observation_label: str = "",
    out_dir: Optional[str] = None,
) -> List[Path]:
    """Render the full thesis figure set for one execution; returns written paths."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if composite_path and Path(composite_path).exists():
        p = render_index_map(composite_path, "NDVI",
                             out_path=str(out_dir / "map_ndvi_composite.png"),
                             observation_label=observation_label,
                             forest_mask=forest_mask)
        if p: written.append(p)

    for key, path in (index_paths or {}).items():
        if path and Path(path).exists() and key.upper() in INDEX_PROFILES:
            p = render_index_map(path, key.upper(),
                                 out_path=str(out_dir / f"map_{key.lower()}.png"),
                                 observation_label=observation_label,
                                 forest_mask=forest_mask)
            if p: written.append(p)

    if trend_path and Path(trend_path).exists():
        p = render_index_map(trend_path, "TREND",
                             out_path=str(out_dir / "map_ndvi_trend.png"),
                             observation_label=observation_label,
                             forest_mask=forest_mask)
        if p: written.append(p)

    if phenology_records:
        p = plot_phenology_time_series(phenology_records,
                                       out_path=str(out_dir / "phenology_time_series.png"))
        if p: written.append(p)

    if fhi:
        p = plot_fhi_breakdown(fhi, out_path=str(out_dir / "fhi_breakdown.png"))
        if p: written.append(p)

    logger.info("Rendered %d standard figures into %s", len(written), out_dir)
    return written


if __name__ == "__main__":
    logger.info("Cartography module loaded. Use render_index_map / render_standard_figures.")
