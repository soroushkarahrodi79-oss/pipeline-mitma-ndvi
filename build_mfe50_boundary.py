"""
Build the official Hayedo de Montejo forest boundary from the MFE50 (Madrid,
prov. 28) shapefile, replacing the provisional config/forest_boundary.geojson.

Selects MFE50 polygons whose dominant species (SP1) is Fagus sylvatica and that
fall within / intersect the AOI of the Reserva de la Biosfera Sierra del Rincon,
dissolves them into a single official polygon, reprojects to WGS84 and writes the
GeoJSON in the structure expected by src/forest_boundary.py.
"""
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
ZIP = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Dell\Downloads\MFE50_28_tcm30-200078.zip"
OUT = ROOT / "config" / "forest_boundary.geojson"

# AOI bbox (WGS84), corrected to contain the real beech stand (~41.12N, -3.50E).
AOI_WGS84 = box(-3.57, 41.07, -3.45, 41.17)

def main():
    work = tempfile.mkdtemp()
    zipfile.ZipFile(ZIP).extractall(work)
    shp = os.path.join(work, "mfe50_28.shp")

    # MFE50 encodes species as numeric codes in SP1/SP2/SP3 (the text fields are
    # empty in this build). Fagus sylvatica = code 71.
    FAGUS_CODE = 71.0
    cols = ["SP1", "SP2", "SP3", "Shape_Area", "geometry"]
    gdf = gpd.read_file(shp, columns=cols)
    src_crs = gdf.crs
    print("CRS:", src_crs, "| total polygons:", len(gdf))

    # Fagus sylvatica is extremely rare in Madrid province: every polygon where
    # beech appears as dominant (SP1), co-dominant (SP2) or significant (SP3)
    # species belongs to the Hayedo de Montejo. We therefore select by species
    # code directly. (The provisional AOI bbox was found to be mislocated ~6 km
    # north of the real stand, so it is NOT used to gate the selection.)
    is_fagus = ((gdf["SP1"] == FAGUS_CODE) | (gdf["SP2"] == FAGUS_CODE) |
                (gdf["SP3"] == FAGUS_CODE))
    near = gdf[is_fagus].copy()
    n_dom = int((near["SP1"] == FAGUS_CODE).sum())
    print(f"beech polygons in province (SP1/2/3=71): {len(near)} "
          f"(dominant SP1=71: {n_dom})")
    if len(near) == 0:
        raise SystemExit("No Fagus polygons found — check the species code / shapefile.")

    # Sanity check: confirm the cluster sits at the known site (~41.12N, -3.50E)
    # and warn if it falls outside the recorded AOI bbox.
    union_4326 = gpd.GeoSeries([near.geometry.union_all()], crs=src_crs).to_crs(4326).iloc[0]
    cen = union_4326.centroid
    print(f"selection centroid (WGS84): lon={cen.x:.4f}, lat={cen.y:.4f}")
    if not union_4326.intersects(AOI_WGS84):
        print("WARNING: selected beech does NOT intersect the recorded AOI bbox "
              "-> the provisional AOI is mislocated and should be corrected.")

    area_ha = round(near.geometry.area.sum() / 10000.0, 1)
    print("selected beech area (ha):", area_ha)

    dissolved = unary_union(near.geometry.values)
    gpoly = gpd.GeoSeries([dissolved], crs=src_crs).to_crs("EPSG:4326").iloc[0]

    feature = {
        "type": "Feature",
        "properties": {
            "name": "Hayedo de Montejo - official forest boundary (MFE50)",
            "species": "Fagus sylvatica (MFE50 species code 71; dominant + co-dominant stands)",
            "source": "MFE50 (Mapa Forestal de Espana 1:50.000), province 28 (Madrid), CNIG/MITECO",
            "source_file": Path(ZIP).name,
            "selection": "All MFE50 polygons with Fagus sylvatica (code 71) as SP1/SP2/SP3 in Madrid province (uniquely = Hayedo de Montejo)",
            "n_source_polygons": int(len(near)),
            "n_dominant_beech_polygons": int((near["SP1"] == 71.0).sum()),
            "area_ha": area_ha,
            "original_crs": str(src_crs),
            "boundary_status": "OFFICIAL",
            "boundary_detail": "MFE50",
            "positional_uncertainty_m": "~25 (MFE50 1:50.000 nominal)",
        },
        "geometry": json.loads(gpd.GeoSeries([gpoly]).to_json())["features"][0]["geometry"],
    }
    fc = {"type": "FeatureCollection", "features": [feature]}

    # Back up the provisional polygon before overwriting.
    if OUT.exists():
        backup = OUT.with_suffix(".provisional.geojson")
        if not backup.exists():
            backup.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
            print("provisional boundary backed up to:", backup.name)

    OUT.write_text(json.dumps(fc, ensure_ascii=True, indent=2), encoding="utf-8")
    print("OFFICIAL boundary written:", OUT)
    print("geometry type:", feature["geometry"]["type"])

if __name__ == "__main__":
    main()
