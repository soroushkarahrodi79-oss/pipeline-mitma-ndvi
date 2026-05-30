from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common import (
    RAW_DIR, CONFIG_DIR,
    AcquisitionError,
    load_geojson, load_yaml_config, save_json, setup_logging,
)

logger = setup_logging("ingestion")

CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"


def _geojson_to_wkt(geojson: Dict) -> str:
    features = geojson.get("features", [])
    if not features:
        raise ValueError("AOI GeoJSON must contain at least one feature.")
    ring = features[0]["geometry"]["coordinates"][0]
    points = ", ".join(f"{lon} {lat}" for lon, lat in ring)
    return f"POLYGON(({points}))"


def build_cdse_filter(
    aoi_wkt: str,
    start_date: str,
    end_date: str,
    cloud_cover_max: float = 15.0,
) -> str:
    return (
        f"Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start ge {start_date}T00:00:00.000Z "
        f"and ContentDate/Start le {end_date}T23:59:59.999Z "
        f"and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {cloud_cover_max})"
    )


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_cdse_page(params: Dict) -> Dict:
    response = requests.get(CDSE_CATALOGUE_URL, params=params, timeout=30)
    if response.status_code == 429:
        raise requests.RequestException("CDSE rate limit exceeded (HTTP 429).")
    response.raise_for_status()
    return response.json()


def _parse_cdse_entry(entry: Dict) -> Dict:
    product_id = entry.get("Id", "")
    name = entry.get("Name", "")
    start = entry.get("ContentDate", {}).get("Start", "")
    cloud_cover = None
    for attr in entry.get("Attributes", []):
        if attr.get("Name") == "cloudCover":
            cloud_cover = float(attr.get("Value", 0.0))
    # CDSE download API requires UUID without quotes: Products(UUID)/$value
    download_url = f"{CDSE_DOWNLOAD_BASE}({product_id})/$value" if product_id else None
    return {
        "product_id": product_id,
        "title": name,
        "beginposition": start,
        "cloudcoverpercentage": cloud_cover,
        "download_url": download_url,
        "online": entry.get("Online", True),
        "s3_path": entry.get("S3Path", ""),
        "queried_at": datetime.utcnow().isoformat() + "Z",
    }


def query_cdse_api(
    aoi_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config_path: Optional[str] = None,
    page_size: int = 100,
) -> List[Dict]:
    if aoi_path is None:
        aoi_path = CONFIG_DIR / "aoi.geojson"
    geojson = load_geojson(str(aoi_path))
    config = load_yaml_config(config_path)
    cloud_cover_max = config.get("api", {}).get("cloud_cover_max", 15.0)
    aoi_wkt = _geojson_to_wkt(geojson)
    odata_filter = build_cdse_filter(aoi_wkt, start_date, end_date, cloud_cover_max)

    logger.info("Querying CDSE catalogue %s → %s (cloud ≤ %.0f%%)", start_date, end_date, cloud_cover_max)

    all_entries: List[Dict] = []
    params: Dict = {
        "$filter": odata_filter,
        "$top": page_size,
        "$orderby": "ContentDate/Start desc",
        "$expand": "Attributes",   # required to populate Attributes[] in the response
    }

    while True:
        try:
            payload = _fetch_cdse_page(params)
        except requests.RequestException as exc:
            raise AcquisitionError(f"CDSE catalogue query failed: {exc}") from exc

        entries = payload.get("value", [])
        all_entries.extend(_parse_cdse_entry(e) for e in entries)
        logger.info("Retrieved %d products (total so far: %d)", len(entries), len(all_entries))

        next_link = payload.get("@odata.nextLink")
        if not next_link or len(entries) < page_size:
            break
        # nextLink is a full URL; replace params with the pre-built URL on next iteration
        params = {"$filter": odata_filter, "$top": page_size, "$skip": len(all_entries)}

    return all_entries


def save_acquisition_metadata(metadata: List[Dict], start_date: str, end_date: str) -> Path:
    target = RAW_DIR / f"acquisitions_{start_date.replace('-', '')}_{end_date.replace('-', '')}.json"
    return save_json(metadata, str(target))


def collect_new_acquisitions(
    window_days: int = 7,
    config_path: Optional[str] = None,
    aoi_path: Optional[str] = None,
    auth: Optional[Dict] = None,  # kept for API compatibility; search needs no auth
) -> Path:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=window_days)
    acquisitions = query_cdse_api(
        aoi_path=aoi_path,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        config_path=config_path,
    )
    if not acquisitions:
        logger.info("No new acquisitions found in the last %d days.", window_days)
    return save_acquisition_metadata(acquisitions, start_date.isoformat(), end_date.isoformat())


if __name__ == "__main__":
    config = load_yaml_config()
    window = config.get("operational", {}).get("weekly_window_days", 7)
    collect_new_acquisitions(window_days=window)
