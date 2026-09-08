"""Past experiments browser — scans experiment manifest files."""
from __future__ import annotations

import json
from pathlib import Path


def discover_experiments(experiments_root: Path | None = None) -> list[dict]:
    """Scan experiments directory and return list of manifest dicts."""
    if experiments_root is None:
        experiments_root = Path(__file__).resolve().parent.parent.parent.parent / "experiments"
    if not experiments_root.exists():
        return []
    results = []
    for d in sorted(experiments_root.iterdir()):
        if not d.is_dir() or d.name.endswith(".tmp"):
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["_dir"] = str(d)
            data["_manifest_ok"] = True
            results.append(data)
        except Exception as e:
            results.append({
                "_dir": str(d),
                "_manifest_ok": False,
                "_error": str(e),
                "experiment_id": d.name,
            })
    return sorted(results, key=lambda x: x.get("created_utc", ""), reverse=True)


def verify_checksums(experiment_dir: Path) -> list[str]:
    """Return list of checksum failure messages (empty = all OK)."""
    import hashlib
    checksums_path = experiment_dir / "checksums.sha256"
    if not checksums_path.exists():
        return ["checksums.sha256 not found"]
    failures = []
    for line in checksums_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            failures.append(f"Malformed line: {line!r}")
            continue
        expected_hash, fname = parts
        fpath = experiment_dir / fname
        if not fpath.exists():
            failures.append(f"Missing file: {fname}")
            continue
        h = hashlib.sha256()
        with fpath.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != expected_hash:
            failures.append(f"Checksum mismatch: {fname}")
    return failures
