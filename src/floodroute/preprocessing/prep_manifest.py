"""Preprocessing manifest generation and I/O (Stage 3).

Preprocessing manifests are distinct from dataset acquisition manifests
(data/manifests/*.yaml).  They record processing provenance for every
output written under data/processed/: source checksums, operation
parameters, software versions, output CRS/bounds/checksums, and
timestamps.  They are written as JSON under
data/processed/preprocessing_manifests/.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floodroute.preprocessing.config import PREPROCESSING_MANIFEST_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Software-version snapshot
# ---------------------------------------------------------------------------


def _software_versions() -> dict[str, str]:
    """Return a dict of relevant package versions for reproducibility."""
    import floodroute

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "floodroute": floodroute.__version__,
    }
    for pkg in ("fiona", "geopandas", "shapely", "pyproj", "rasterio", "numpy"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


# ---------------------------------------------------------------------------
# File checksum
# ---------------------------------------------------------------------------


def compute_file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Preprocessing manifest
# ---------------------------------------------------------------------------


def build_preprocessing_manifest(
    *,
    output_id: str,
    operation: str,
    parameters: dict[str, Any],
    source_dataset_ids: list[str],
    source_checksums: dict[str, str],
    output_path: Path,
    output_crs: str,
    output_bounds: dict[str, float],
    feature_count: int | None = None,
    raster_width: int | None = None,
    raster_height: int | None = None,
    raster_nodata: float | None = None,
    raster_dtype: str | None = None,
    geometry_repairs: int = 0,
    validation_status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a preprocessing manifest dict for one output file.

    Parameters
    ----------
    output_id:
        Short identifier for the output (used as the manifest filename stem).
    operation:
        Human-readable name of the processing operation.
    parameters:
        Processing parameters (municipality codes, resampling method, etc.).
    source_dataset_ids:
        List of dataset_ids from data/manifests/ that were consumed.
    source_checksums:
        Mapping of dataset_id -> SHA-256 of the source file.
    output_path:
        Absolute path to the output file (used to compute output checksum).
    output_crs:
        CRS of the output (e.g. "EPSG:4326" or "EPSG:32651").
    output_bounds:
        Bounding box dict with keys west/south/east/north (geographic) or
        xmin/ymin/xmax/ymax (projected).
    feature_count:
        Number of vector features written (None for raster outputs).
    raster_width, raster_height:
        Pixel dimensions for raster outputs (None for vector).
    raster_nodata:
        Nodata sentinel value for raster outputs.
    raster_dtype:
        NumPy dtype string for raster outputs (e.g. "float32").
    geometry_repairs:
        Number of geometries repaired with make_valid().
    validation_status:
        Explicit validation result: "passed", "failed", or None if validation
        was not performed.  Written as a top-level manifest field so that
        downstream consumers can inspect validation state without parsing extra.
    extra:
        Any additional metadata to include verbatim.
    """
    output_checksum = compute_file_sha256(output_path)

    manifest: dict[str, Any] = {
        "schema_version": PREPROCESSING_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "preprocessing",
        "output_id": output_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source_datasets": [
            {"dataset_id": did, "sha256": source_checksums.get(did, "unknown")}
            for did in source_dataset_ids
        ],
        "operation": operation,
        "parameters": parameters,
        "software": _software_versions(),
        "output": {
            "path": str(output_path),
            "crs": output_crs,
            "bounds": output_bounds,
            "sha256": output_checksum,
        },
    }

    if validation_status is not None:
        manifest["validation_status"] = validation_status

    out = manifest["output"]
    if feature_count is not None:
        out["feature_count"] = feature_count
    if geometry_repairs:
        out["geometry_repairs"] = geometry_repairs
    if raster_width is not None:
        out["width"] = raster_width
        out["height"] = raster_height
    if raster_nodata is not None:
        out["nodata"] = raster_nodata
    if raster_dtype is not None:
        out["dtype"] = raster_dtype
    if extra:
        manifest["extra"] = extra

    return manifest


def write_preprocessing_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Write a preprocessing manifest dict to *manifest_path* as indented JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
        fh.write("\n")


def read_preprocessing_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a preprocessing manifest from a JSON file."""
    with manifest_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != PREPROCESSING_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported preprocessing manifest schema_version "
            f"'{data.get('schema_version')}' in {manifest_path}"
        )
    if data.get("manifest_type") != "preprocessing":
        raise ValueError(
            f"Expected manifest_type='preprocessing', got "
            f"'{data.get('manifest_type')}' in {manifest_path}"
        )
    return data


# ---------------------------------------------------------------------------
# Source-checksum helper
# ---------------------------------------------------------------------------


def source_checksum_from_manifest_dir(
    dataset_id: str,
    manifests_dir: Path,
) -> str:
    """Return the sha256 recorded in the dataset acquisition manifest, or 'unknown'."""
    try:
        import yaml

        manifest_path = manifests_dir / f"{dataset_id}.yaml"
        if not manifest_path.exists():
            return "unknown"
        with manifest_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return str(data.get("sha256", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Validate existing preprocessing manifests
# ---------------------------------------------------------------------------


def validate_preprocessing_manifests(manifests_dir: Path) -> list[dict[str, Any]]:
    """Load and validate all preprocessing manifests in *manifests_dir*.

    Returns a list of manifest dicts.  Raises ValueError on the first
    schema error.
    """
    results: list[dict[str, Any]] = []
    for path in sorted(manifests_dir.glob("*.json")):
        m = read_preprocessing_manifest(path)
        results.append(m)
    return results
