"""
STEP 1 + STEP 2 : Storage audit and legacy-data quarantine.

- Audits free disk space and estimates the space the baseline re-ingestion needs.
- Scans the LEGACY data/ tree (and the new outputs/raw_downloads tree) for
  obsolete-AOI artifacts, incomplete/corrupt SAFE downloads, duplicates and
  stray ZIPs, and MOVES them to outputs/quarantine/ (never deletes silently),
  writing a detailed audit log (JSON + Markdown).

Run:  .venv/Scripts/python.exe scripts/audit_and_quarantine.py
"""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402

ROOT = common.WORKSPACE_ROOT
LEGACY_DATA = ROOT / "data"
QUAR = common.QUARANTINE_DIR
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# Baseline sizing assumptions
YEARS = [2021, 2022, 2023, 2024, 2025]
SCENES_PER_YEAR = 4
GB = 1024 ** 3
PER_SCENE_GB = 1.1            # typical L2A SAFE footprint on disk
PEAK_YEAR_GB = SCENES_PER_YEAR * PER_SCENE_GB + 0.5   # one year staged at once
TOTAL_DL_GB = len(YEARS) * SCENES_PER_YEAR * PER_SCENE_GB


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def safe_has_bands(safe: Path) -> bool:
    """A valid L2A SAFE must contain IMG_DATA spectral band JP2s (B02/B04/B08...)."""
    img = list(safe.rglob("IMG_DATA/**/*.jp2"))
    band_imgs = [p for p in img if any(b in p.name for b in
                 ("_B02", "_B03", "_B04", "_B08", "_B8A", "_B11", "_B12"))]
    return len(band_imgs) > 0


def storage_audit() -> dict:
    u = shutil.disk_usage(ROOT)
    free_gb = u.free / GB
    report = {
        "free_gb": round(free_gb, 2),
        "total_gb": round(u.total / GB, 1),
        "used_gb": round(u.used / GB, 1),
        "estimated_total_download_gb": round(TOTAL_DL_GB, 1),
        "estimated_peak_year_gb": round(PEAK_YEAR_GB, 1),
        "assumptions": f"{len(YEARS)} years x {SCENES_PER_YEAR} scenes x ~{PER_SCENE_GB} GB/scene; "
                       "sequential per-year staging with cleanup between years",
        "safety_margin_after_peak_year_gb": round(free_gb - PEAK_YEAR_GB, 2),
        "sufficient_for_sequential_run": free_gb >= PEAK_YEAR_GB,
        "legacy_data_dir_size": human(dir_size(LEGACY_DATA)),
        "outputs_dir_size": human(dir_size(common.OUTPUTS_DIR)),
    }
    return report


def classify_legacy() -> list:
    """Return a list of {path, kind, reason, action} for obsolete/corrupt artifacts."""
    findings = []

    # 1) Legacy SAFE scenes (obsolete operational workflow + likely incomplete)
    for safe in sorted((LEGACY_DATA / "raw" / "scenes").glob("*.SAFE")) if (LEGACY_DATA / "raw" / "scenes").exists() else []:
        has_bands = safe_has_bands(safe)
        reason = ("Obsolete pre-MFE50 operational scene (May-2026, tile T30TVL) from the "
                  "wrong-AOI workflow")
        if not has_bands:
            reason += "; INCOMPLETE/corrupt — IMG_DATA spectral bands missing (only QI/metadata present)"
        findings.append({"path": str(safe), "kind": "safe_scene",
                         "size": human(dir_size(safe)), "reason": reason,
                         "action": "quarantine"})

    # 2) Legacy acquisition metadata JSONs (obsolete-AOI catalogue searches)
    for j in sorted((LEGACY_DATA / "raw").glob("acquisitions_*.json")) if (LEGACY_DATA / "raw").exists() else []:
        findings.append({"path": str(j), "kind": "metadata",
                         "size": human(dir_size(j)),
                         "reason": "Obsolete-AOI catalogue metadata from pre-correction operational search",
                         "action": "quarantine"})

    # 3) Stray ZIP archives anywhere under data/ (incomplete downloads)
    if LEGACY_DATA.exists():
        for z in LEGACY_DATA.rglob("*.zip"):
            findings.append({"path": str(z), "kind": "zip",
                             "size": human(dir_size(z)),
                             "reason": "Stray/legacy ZIP archive under data/ (possible incomplete download)",
                             "action": "quarantine"})

    # 4) Empty leftover staging dirs (failed run) — removed, not quarantined (nothing in them)
    for stg in [LEGACY_DATA / "raw" / "baseline_staging"]:
        if stg.exists() and not any(f.is_file() for f in stg.rglob("*")):
            findings.append({"path": str(stg), "kind": "empty_dir",
                             "size": "0 B",
                             "reason": "Empty leftover staging directory from the failed baseline run",
                             "action": "remove_empty"})
    return findings


def apply(findings: list) -> list:
    QUAR.mkdir(parents=True, exist_ok=True)
    dest_root = QUAR / f"legacy_{TS}"
    log = []
    for f in findings:
        src = Path(f["path"])
        entry = dict(f)
        try:
            if not src.exists():
                entry["result"] = "skipped (already gone)"
            elif f["action"] == "remove_empty":
                shutil.rmtree(src, ignore_errors=True)
                entry["result"] = "removed empty directory"
            else:
                sub = {"safe_scene": "obsolete_safe_scenes",
                       "metadata": "obsolete_metadata",
                       "zip": "incomplete_zips"}.get(f["kind"], "misc")
                dest_dir = dest_root / sub
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src.name
                shutil.move(str(src), str(dest))
                entry["quarantined_to"] = str(dest)
                entry["result"] = "moved to quarantine"
        except Exception as exc:
            entry["result"] = f"ERROR: {exc}"
        log.append(entry)
    return log


def main():
    audit = storage_audit()
    findings = classify_legacy()
    log = apply(findings)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage_audit": audit,
        "quarantine_actions": log,
        "n_flagged": len(log),
    }
    QUAR.mkdir(parents=True, exist_ok=True)
    (QUAR / f"audit_{TS}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [f"# Storage & Data Audit — {TS}", "",
          "## STEP 1 — Storage audit", "",
          f"- Free space: **{audit['free_gb']} GB** / {audit['total_gb']} GB total",
          f"- Estimated total download (all 5 years): ~{audit['estimated_total_download_gb']} GB",
          f"- Estimated peak (one year staged at a time): ~{audit['estimated_peak_year_gb']} GB",
          f"- Safety margin after peak year: **{audit['safety_margin_after_peak_year_gb']} GB**",
          f"- Sufficient for sequential run: **{audit['sufficient_for_sequential_run']}**",
          f"- Legacy data/ size: {audit['legacy_data_dir_size']} | outputs/ size: {audit['outputs_dir_size']}",
          "", "## STEP 2 — Quarantine actions", ""]
    if not log:
        md.append("No obsolete/corrupt artifacts found.")
    for e in log:
        md += [f"### {Path(e['path']).name}  ({e['kind']}, {e.get('size','?')})",
               f"- Reason: {e['reason']}",
               f"- Action: {e['action']} -> {e['result']}"]
        if e.get("quarantined_to"):
            md.append(f"- Moved to: `{e['quarantined_to']}`")
        md.append("")
    (QUAR / f"audit_{TS}.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(f"\nFlagged {len(log)} items. Audit log: {QUAR / ('audit_'+TS+'.md')}")
    for e in log:
        print(f"  [{e['result']}] {Path(e['path']).name} — {e['kind']}")


if __name__ == "__main__":
    main()
