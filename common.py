import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Consolidated output structure (single, inspectable tree under outputs/).
# All raw downloads, intermediate products, composites, reports, logs and
# quarantined artifacts live here so nothing is scattered across the repo.
# ---------------------------------------------------------------------------
RAW_DIR = OUTPUTS_DIR / "raw_downloads"      # all raw Sentinel SAFE / ZIP / metadata
PROCESSED_DIR = OUTPUTS_DIR / "processed"    # per-scene 8-band TIFs + index TIFs
COMPOSITES_DIR = OUTPUTS_DIR / "composites"  # median / rolling / annual composites
REPORTS_DIR = OUTPUTS_DIR / "reports"        # consolidated report landing area
LOG_DIR = OUTPUTS_DIR / "logs"               # runtime logs
QUARANTINE_DIR = OUTPUTS_DIR / "quarantine"  # flagged / obsolete / corrupt artifacts

# DATA_DIR retained for backward reference to any legacy data/ tree (not written to).
DATA_DIR = WORKSPACE_ROOT / "data"

OUTPUTS_ANNUAL_DIR = OUTPUTS_DIR / "annual"
OUTPUTS_MONTHLY_DIR = OUTPUTS_DIR / "monthly"
OUTPUTS_ANOMALIES_DIR = OUTPUTS_DIR / "anomalies"
CONFIG_DIR = WORKSPACE_ROOT / "config"

# Preprocessed TIF band layout (8 bands total)
BAND_BLUE = 1       # B02  10 m
BAND_RED = 2        # B04  10 m
BAND_NIR = 3        # B08  10 m
BAND_RED_EDGE = 4   # B05  20 m → resampled to 10 m
BAND_NIR_8A = 5     # B8A  20 m → resampled to 10 m
BAND_SWIR1 = 6      # B11  20 m → resampled to 10 m
BAND_SWIR2 = 7      # B12  20 m → resampled to 10 m
BAND_MASK = 8       # SCL-derived validity mask (1 = valid, NaN = invalid)
PREPROCESSED_BAND_COUNT = 8

# SCL classes accepted as valid surface observations for FOREST monitoring.
# Class 4 = vegetation, 5 = bare soil/rock
# Excluded: 0 (no data), 1 (saturated), 2 (dark features), 3 (cloud shadow),
#           6 (water — distorts NDVI stats over forest),
#           7 (unclassified — unreliable), 8 (cloud medium), 9 (cloud high),
#           10 (thin cirrus), 11 (snow/ice — distorts summer vegetation signals)
VALID_SCL_CLASSES = [4, 5]


class PipelineError(Exception):
    """Base exception for controlled pipeline failures."""


class InsufficientDataError(PipelineError):
    """Not enough valid observations to continue processing."""


class AcquisitionError(PipelineError):
    """Failure acquiring or downloading data from Copernicus."""


def ensure_directories() -> None:
    for path in [
        RAW_DIR, PROCESSED_DIR, COMPOSITES_DIR, REPORTS_DIR, QUARANTINE_DIR,
        OUTPUTS_ANNUAL_DIR, OUTPUTS_MONTHLY_DIR, OUTPUTS_ANOMALIES_DIR,
        LOG_DIR, CONFIG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "forest_monitoring", run_id: Optional[str] = None) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    run_id = run_id or str(uuid.uuid4())[:8]
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] %(name)s [run={run_id}]: %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logfile = LOG_DIR / f"{name}.log"
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Secret handling
# ---------------------------------------------------------------------------
# Secrets are NEVER stored in thresholds.yaml. They are read from environment
# variables (optionally populated from a .gitignored .env file) and overlaid on
# the parsed config at load time. The mapping below is (ENV_VAR -> config path).
SECRET_ENV_MAP = {
    "CDSE_USERNAME":       ("api", "username"),
    "CDSE_PASSWORD":       ("api", "password"),
    "AEMET_API_KEY":       ("aemet", "api_key"),
    "DISCORD_WEBHOOK_URL": ("reporting", "discord_webhook_url"),
}


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader (no external dependency). Existing os.environ wins."""
    if not dotenv_path.exists():
        return
    try:
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def _overlay_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay secret env vars onto the config (env value wins when present)."""
    _load_dotenv(WORKSPACE_ROOT / ".env")
    for env_var, (section, key) in SECRET_ENV_MAP.items():
        value = os.environ.get(env_var)
        if value:
            config.setdefault(section, {})
            if isinstance(config[section], dict):
                config[section][key] = value
    return config


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    ensure_directories()
    if config_path is None:
        config_path = CONFIG_DIR / "thresholds.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return _overlay_secrets(config)


def load_geojson(geojson_path: Optional[str] = None) -> Dict[str, Any]:
    if geojson_path is None:
        geojson_path = CONFIG_DIR / "aoi.geojson"
    else:
        geojson_path = Path(geojson_path)
    with open(geojson_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, target_path: str) -> Path:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
    return target_path


def save_yaml(data: Any, target_path: str) -> Path:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return target_path


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", ""))


def fmt_date(value: datetime, pattern: str = "%Y%m%d") -> str:
    return value.strftime(pattern)
