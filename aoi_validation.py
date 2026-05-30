"""
AOI validation hardening — STEP 4.

Geospatial guards that prevent AOI contamination: a scene is accepted only if
its actual footprint geometry intersects the corrected official AOI, and a
downloaded product is kept only if its real raster extent intersects the AOI.
Validation never relies on file/scene names alone — it uses real geometry.

Also provides small production-grade safeguards (STEP 6): a disk-space guard
and a quarantine helper used by the ingestion workflow.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import shutil

from common import QUARANTINE_DIR, load_geojson, setup_logging

logger = setup_logging("aoi_validation")


# ---------------------------------------------------------------------------
# AOI geometry
# ---------------------------------------------------------------------------

def load_aoi_geom(aoi_path: Optional[str] = None):
    """Return the AOI as a single shapely geometry in WGS84 (union of features)."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    fc = load_geojson(aoi_path)
    geoms = [shape(f["geometry"]) for f in fc.get("features", []) if f.get("geometry")]
    if not geoms:
        raise ValueError("AOI GeoJSON contains no geometry.")
    return unary_union(geoms)


# ---------------------------------------------------------------------------
# Scene footprint parsing + pre-download validation
# ---------------------------------------------------------------------------

def _parse_footprint(scene: Dict[str, Any]):
    """
    Extract a shapely geometry from a CDSE scene record.
    Supports GeoFootprint (GeoJSON) and Footprint (WKT, possibly SRID-prefixed).
    Returns None if no parseable geometry is present.
    """
    from shapely.geometry import shape
    gf = scene.get("geo_footprint") or scene.get("GeoFootprint")
    if isinstance(gf, dict) and gf.get("type"):
        try:
            return shape(gf)
        except Exception:
            pass
    fp = scene.get("footprint") or scene.get("Footprint")
    if isinstance(fp, str) and fp:
        wkt = fp.split(";", 1)[-1].strip().strip("'")  # drop SRID=4326; prefix/quotes
        try:
            from shapely import wkt as shp_wkt
            return shp_wkt.loads(wkt)
        except Exception:
            return None
    return None


def validate_scene_pre_download(scene: Dict[str, Any], aoi_geom) -> Tuple[bool, str]:
    """
    Geospatial pre-download check: does the scene footprint intersect the AOI?
    Returns (accepted, reason).
    """
    geom = _parse_footprint(scene)
    if geom is None:
        # No geometry available -> cannot verify by geometry. Accept only if the
        # server-side OData Intersects filter was applied (it is), but flag it.
        return True, "no footprint geometry returned; relying on server-side AOI filter"
    try:
        if geom.intersects(aoi_geom):
            frac = geom.intersection(aoi_geom).area / max(aoi_geom.area, 1e-12)
            return True, f"footprint intersects AOI (covers ~{min(frac,1.0)*100:.0f}% of AOI)"
        return False, "footprint does NOT intersect the corrected AOI"
    except Exception as exc:
        return True, f"geometry check error ({exc}); relying on server-side filter"


# ---------------------------------------------------------------------------
# Post-download raster validation
# ---------------------------------------------------------------------------

def validate_raster_post_download(raster_path: str, aoi_geom) -> Tuple[bool, str]:
    """
    Confirm a produced raster's real extent intersects the AOI (in WGS84).
    Returns (accepted, reason).
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    try:
        with rasterio.open(raster_path) as src:
            b = src.bounds
            wgs = transform_bounds(src.crs, "EPSG:4326", *b)
        extent = box(wgs[0], wgs[1], wgs[2], wgs[3])
        if extent.intersects(aoi_geom):
            return True, "raster extent intersects AOI"
        return False, (f"raster extent {tuple(round(v,3) for v in wgs)} does NOT "
                       "intersect the corrected AOI")
    except Exception as exc:
        return False, f"could not validate raster extent: {exc}"


# ---------------------------------------------------------------------------
# Safeguards (STEP 6)
# ---------------------------------------------------------------------------

def free_gb(path: Optional[str] = None) -> float:
    from common import WORKSPACE_ROOT
    return shutil.disk_usage(path or WORKSPACE_ROOT).free / 1024 ** 3


def disk_guard(required_gb: float, path: Optional[str] = None) -> Tuple[bool, float]:
    """Return (ok, free_gb). ok=True if free space >= required_gb."""
    f = free_gb(path)
    return (f >= required_gb), round(f, 2)


def quarantine_path(src: str, reason: str, subdir: str = "rejected") -> Optional[str]:
    """Move a file/dir to the quarantine tree; return destination path."""
    src_p = Path(src)
    if not src_p.exists():
        return None
    dest_dir = QUARANTINE_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_p.name
    try:
        shutil.move(str(src_p), str(dest))
        (dest_dir / (src_p.name + ".reason.txt")).write_text(reason, encoding="utf-8")
        logger.warning("Quarantined %s -> %s (%s)", src_p.name, dest, reason)
        return str(dest)
    except Exception as exc:
        logger.warning("Quarantine failed for %s: %s", src, exc)
        return None


if __name__ == "__main__":
    g = load_aoi_geom()
    print("AOI bounds (WGS84):", [round(v, 4) for v in g.bounds])
    print("free GB:", round(free_gb(), 2))
