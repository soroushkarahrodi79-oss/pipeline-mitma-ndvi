"""
Execution archive manager.

Every pipeline run creates a timestamped folder under runs/{YYYY}/
containing a full snapshot of the configuration, outputs, reports, and logs.
This enables complete reproducibility and longitudinal comparison of results.

Folder structure per run:
  runs/
    2026/
      2026-05-30_1441_monthly_a3f2b1/
        execution_metadata.json
        config_snapshot.yaml
        outputs/          (copies of all generated rasters and data files)
        reports/          (all 7 markdown report sections)
        logs/             (copy of the run log)
"""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from common import WORKSPACE_ROOT, LOG_DIR, save_json, setup_logging

logger = setup_logging("archive")

RUNS_DIR = WORKSPACE_ROOT / "runs"


def new_run_id(job: str) -> str:
    """Generate a unique, time-sortable run identifier."""
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    short = str(uuid.uuid4())[:6]
    return f"{ts}_{job}_{short}"


def run_dir(run_id: str) -> Path:
    year = run_id[:4]
    return RUNS_DIR / year / run_id


def init_run(run_id: str, job: str, config_path: str) -> Path:
    """
    Initialise a run archive folder.
    Returns the path to the run directory.
    """
    rdir = run_dir(run_id)
    for sub in ["outputs", "reports", "logs"]:
        (rdir / sub).mkdir(parents=True, exist_ok=True)

    # Copy config snapshot
    try:
        shutil.copy2(config_path, rdir / "config_snapshot.yaml")
    except Exception as exc:
        logger.warning("Could not copy config snapshot: %s", exc)

    # Write execution metadata
    meta = {
        "run_id":        run_id,
        "job":           job,
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "finished_at":   None,
        "status":        "running",
        "config_path":   config_path,
        "outputs":       [],
        "reports":       [],
    }
    save_json(meta, str(rdir / "execution_metadata.json"))
    logger.info("Run archive initialised: %s", rdir)
    return rdir


def archive_outputs(run_id: str, output_paths: List[str]) -> None:
    """Copy generated output files (TIF, JSON, CSV, MD) into the run archive."""
    rdir    = run_dir(run_id) / "outputs"
    rdir.mkdir(parents=True, exist_ok=True)
    archived = []
    for path_str in output_paths:
        src = Path(path_str)
        if not src.exists():
            continue
        dst = rdir / src.name
        try:
            shutil.copy2(src, dst)
            archived.append(str(dst))
        except Exception as exc:
            logger.warning("Could not archive %s: %s", src, exc)
    _update_meta(run_id, {"outputs": archived})


def archive_reports(run_id: str, report_paths: List[str]) -> None:
    """Copy generated report files into the run archive."""
    rdir = run_dir(run_id) / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    archived = []
    for path_str in report_paths:
        src = Path(path_str)
        if not src.exists():
            continue
        dst = rdir / src.name
        try:
            shutil.copy2(src, dst)
            archived.append(str(dst))
        except Exception as exc:
            logger.warning("Could not archive report %s: %s", src, exc)
    _update_meta(run_id, {"reports": archived})


def archive_logs(run_id: str) -> None:
    """Copy the current pipeline log into the run archive."""
    rdir = run_dir(run_id) / "logs"
    rdir.mkdir(parents=True, exist_ok=True)
    for log_file in LOG_DIR.glob("*.log"):
        try:
            shutil.copy2(log_file, rdir / log_file.name)
        except Exception as exc:
            logger.warning("Could not archive log %s: %s", log_file, exc)


def close_run(run_id: str, status: str = "success", error: Optional[str] = None) -> None:
    """Mark the run as finished and write final metadata."""
    updates: Dict[str, Any] = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status":      status,
    }
    if error:
        updates["error"] = error
    _update_meta(run_id, updates)
    archive_logs(run_id)
    logger.info("Run %s closed with status: %s", run_id, status)


def list_runs(job: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return a list of all archived run metadata, newest first."""
    runs = []
    for meta_file in sorted(RUNS_DIR.rglob("execution_metadata.json"), reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if job is None or meta.get("job") == job:
                runs.append(meta)
        except Exception:
            pass
    return runs


def latest_run(job: str) -> Optional[Dict[str, Any]]:
    """Return metadata for the most recent completed run of a given job type."""
    for run in list_runs(job):
        if run.get("status") == "success":
            return run
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_meta(run_id: str, updates: Dict[str, Any]) -> None:
    meta_path = run_dir(run_id) / "execution_metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(updates)
    save_json(meta, str(meta_path))
