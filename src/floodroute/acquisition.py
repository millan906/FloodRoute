"""Safe dataset acquisition utilities for FloodRoute.

Provides checksum calculation, format-aware validation, and a controlled
download workflow. Network access is explicit and opt-in — no bulk downloads.
Downloaded content is never executed.

Stage 2 additions: validate_pbf_file, validate_xlsx_file, validate_geotiff_file.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from floodroute.logging_config import get_logger
from floodroute.manifest import AccessMethod, DatasetManifest

logger = get_logger("acquisition")

_DOWNLOADABLE = {AccessMethod.direct_download, AccessMethod.api}
_MANUAL = {
    AccessMethod.manual_download,
    AccessMethod.formal_request,
    AccessMethod.local_authority,
    AccessMethod.derived_later,
    AccessMethod.researcher_assembled,  # curated registry; no automated download path
}
_CHUNK = 65_536  # 64 KiB


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ManualAcquisitionRequired(Exception):
    """Raised when a dataset must be acquired by a human, not automated."""

    def __init__(self, message: str, instructions: str) -> None:
        super().__init__(message)
        self.instructions = instructions


class ChecksumMismatch(Exception):
    """Raised when a downloaded file's SHA-256 does not match the manifest."""


class FileAlreadyExists(Exception):
    """Raised when the destination file exists and force=False."""


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of a bytes object."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_local_path(manifest: DatasetManifest, data_dir: Path) -> Path | None:
    """Return the absolute local path for a manifest's data file, or None."""
    if manifest.local_path is None:
        return None
    p = Path(manifest.local_path)
    return p if p.is_absolute() else data_dir / p


def _default_dest(manifest: DatasetManifest, data_dir: Path) -> Path:
    """Derive a destination path when manifest.local_path is unset."""
    ext = {
        "geotiff": ".tif",
        "tiff": ".tif",
        "geojson": ".geojson",
        "shapefile": ".zip",
        "pbf": ".osm.pbf",
        "csv": ".csv",
        "zip": ".zip",
        "excel": ".xlsx",
    }.get((manifest.expected_format or "").lower(), "")
    return data_dir / "raw" / f"{manifest.dataset_id}{ext}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _stream_download(url: str, dest: Path, *, timeout: int) -> None:
    """Stream a URL to dest in chunks. Separated for easy test-patching."""
    req = urllib.request.Request(url, headers={"User-Agent": "floodroute/0.1 data-acquisition"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
        while chunk := resp.read(_CHUNK):
            fh.write(chunk)


def download_dataset(
    manifest: DatasetManifest,
    data_dir: Path,
    *,
    force: bool = False,
    timeout: int = 30,
) -> Path:
    """Download a dataset described by manifest into data_dir.

    Rules enforced:
    - Manual/formal/local-authority datasets raise ManualAcquisitionRequired.
    - source_url must be non-null.
    - Existing files raise FileAlreadyExists unless force=True.
    - Writes first to a .part temp file, then renames atomically.
    - SHA-256 is verified when manifest.sha256 is set.
    - Cleans up .part file on any failure.
    """
    if manifest.access_method in _MANUAL:
        raise ManualAcquisitionRequired(
            f"Dataset '{manifest.dataset_id}' requires manual acquisition.",
            _manual_instructions(manifest),
        )

    if manifest.source_url is None:
        raise ValueError(f"Dataset '{manifest.dataset_id}' has no source_url; cannot download.")

    dest = resolve_local_path(manifest, data_dir) or _default_dest(manifest, data_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        raise FileAlreadyExists(f"File already exists: {dest}. Supply force=True to overwrite.")

    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s → %s", manifest.source_url, dest)

    try:
        _stream_download(manifest.source_url, part, timeout=timeout)
        actual = compute_sha256(part)

        if manifest.sha256 is not None and actual != manifest.sha256:
            raise ChecksumMismatch(
                f"SHA-256 mismatch for '{manifest.dataset_id}': "
                f"expected {manifest.sha256}, got {actual}"
            )

        shutil.move(str(part), str(dest))
        logger.info("Download complete: %s  sha256=%s", dest.name, actual)
        return dest

    except Exception:
        if part.exists():
            part.unlink()
            logger.debug("Cleaned up temp file: %s", part)
        raise


def _manual_instructions(manifest: DatasetManifest) -> str:
    lines = [
        f"Manual acquisition required for: {manifest.dataset_id}",
        f"  Title         : {manifest.title}",
        f"  Category      : {manifest.category.value}",
        f"  Access method : {manifest.access_method.value}",
    ]
    if manifest.landing_page_url:
        lines.append(f"  Landing page  : {manifest.landing_page_url}")
    if manifest.source_url:
        lines.append(f"  Source URL    : {manifest.source_url}")
    if manifest.local_path:
        lines.append(f"  Expected path : {manifest.local_path}")
    lines.append(f"  After placing the file, run: floodroute verify-dataset {manifest.dataset_id}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format-aware validation
# ---------------------------------------------------------------------------

# Required columns for known CSV categories
_CSV_REQUIRED: dict[str, list[str]] = {
    "evacuation_shelters": [
        "shelter_id",
        "official_name",
        "facility_type",
        "designation_authority",
        "designation_reference",
        "barangay_psgc",
        "address",
        "latitude",
        "longitude",
        "entrance_latitude",
        "entrance_longitude",
        "documented_capacity",
        "capacity_unit",
        "operational_status",
        "validation_date",
        "source_reference",
        "notes",
    ],
    "historical_incidents": [
        "incident_id",
        "event_name",
        "event_date",
        "barangay_psgc",
        "location_description",
        "latitude",
        "longitude",
        "flood_depth_m",
        "road_status",
        "closure_start",
        "closure_end",
        "evacuation_facility",
        "affected_families",
        "affected_persons",
        "source_reference",
        "validation_status",
        "notes",
    ],
}


def validate_csv_file(path: Path, category: str | None = None) -> dict[str, Any]:
    """Validate a CSV file: existence, header, required columns, row count."""
    import csv

    if not path.exists():
        return {"valid": False, "error": "File not found"}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return {"valid": False, "error": "CSV has no header row"}
            row_count = sum(1 for _ in reader)
    except Exception as exc:
        return {"valid": False, "error": str(exc)}

    required = _CSV_REQUIRED.get(category or "", [])
    missing = [c for c in required if c not in header]
    return {
        "valid": not missing,
        "header": header,
        "row_count": row_count,
        "missing_columns": missing,
    }


def validate_zip_file(path: Path) -> dict[str, Any]:
    """Validate a ZIP archive: readable, members listed, no path traversal."""
    if not path.exists():
        return {"valid": False, "error": "File not found"}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = zf.namelist()
    except zipfile.BadZipFile as exc:
        return {"valid": False, "error": f"Bad ZIP file: {exc}"}

    unsafe = [m for m in members if m.startswith(("/", "\\")) or ".." in m.split("/")]
    if unsafe:
        return {
            "valid": False,
            "error": "Unsafe path-traversal entries detected in ZIP",
            "unsafe_entries": unsafe,
        }
    return {"valid": True, "member_count": len(members), "members": members}


def validate_pbf_file(path: Path) -> dict[str, Any]:
    """Validate an OSM PBF file: existence, minimum size, OSMHeader signature.

    An OSM PBF starts with a 4-byte big-endian int32 (BlobHeader length),
    followed immediately by a BlobHeader whose 'type' field is "OSMHeader".
    Checking for the literal b"OSMHeader" in the first 32 bytes is a
    reliable signature test without requiring a full protobuf parser.
    """
    if not path.exists():
        return {"valid": False, "error": "File not found"}
    size = path.stat().st_size
    if size < 32:
        return {"valid": False, "error": f"File too small to be a valid PBF ({size} bytes)"}
    with path.open("rb") as fh:
        header = fh.read(32)
    if b"OSMHeader" not in header:
        return {"valid": False, "error": "Missing OSMHeader signature — not a valid OSM PBF"}
    return {"valid": True, "file_size_bytes": size}


def validate_xlsx_file(path: Path) -> dict[str, Any]:
    """Validate an Excel workbook: readable as OOXML ZIP, extract sheet names.

    XLSX files are ZIP archives.  Sheet names live in xl/workbook.xml inside
    a <sheets> element.  Parsing uses only stdlib (zipfile + xml.etree).
    """
    if not path.exists():
        return {"valid": False, "error": "File not found"}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if "xl/workbook.xml" not in names:
                return {"valid": False, "error": "xl/workbook.xml missing — not a valid XLSX"}
            wb_xml = zf.read("xl/workbook.xml")
    except zipfile.BadZipFile as exc:
        return {"valid": False, "error": f"Not a ZIP/XLSX file: {exc}"}

    try:
        root = ET.fromstring(wb_xml)
        # Strip namespace for portable tag matching
        ns = root.tag.partition("{")[2].partition("}")[0]
        prefix = f"{{{ns}}}" if ns else ""
        sheets_el = root.find(f"{prefix}sheets")
        sheet_names = (
            [el.get("name", "") for el in sheets_el.findall(f"{prefix}sheet")]
            if sheets_el is not None
            else []
        )
    except ET.ParseError as exc:
        return {"valid": False, "error": f"Could not parse xl/workbook.xml: {exc}"}

    return {
        "valid": True,
        "file_size_bytes": path.stat().st_size,
        "sheet_names": sheet_names,
        "sheet_count": len(sheet_names),
    }


# TIFF magic-byte sequences (little-endian, big-endian, BigTIFF variants)
_TIFF_MAGIC: set[bytes] = {
    b"II\x2a\x00",  # TIFF little-endian
    b"MM\x00\x2a",  # TIFF big-endian
    b"II\x2b\x00",  # BigTIFF little-endian
    b"MM\x00\x2b",  # BigTIFF big-endian
}


def validate_geotiff_file(path: Path) -> dict[str, Any]:
    """Validate a GeoTIFF: existence, minimum size, TIFF magic bytes.

    Checks the first 4 bytes against the four known TIFF/BigTIFF magic
    sequences.  Does not require GDAL; cannot verify projection metadata.
    """
    if not path.exists():
        return {"valid": False, "error": "File not found"}
    size = path.stat().st_size
    if size < 4:
        return {"valid": False, "error": f"File too small to be a valid TIFF ({size} bytes)"}
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic not in _TIFF_MAGIC:
        return {"valid": False, "error": f"Invalid TIFF magic bytes: {magic!r}"}
    return {"valid": True, "file_size_bytes": size}


def validate_geotiff_raster(
    path: Path,
    *,
    expected_crs: str | None = None,
    municipality_bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Semantic GeoTIFF validation using rasterio.

    Checks: TIFF magic, CRS, raster dimensions, bounds, nodata value, and
    optionally whether the raster covers a municipality bounding box.

    Args:
        path: Path to the GeoTIFF file.
        expected_crs: If given (e.g. 'EPSG:4326'), verify the file CRS matches.
        municipality_bbox: (west, south, east, north) in WGS84 degrees.
            If given, verify the raster bounds overlap this box.

    Returns:
        Dict with keys: valid (bool), and on success: crs, bounds (west, south,
        east, north), width, height, count (band count), nodata, pixel_size_deg,
        and optionally covers_municipality (bool).
    """
    # Magic-byte pre-check (no rasterio import needed for this gate)
    magic_result = validate_geotiff_file(path)
    if not magic_result["valid"]:
        return magic_result

    try:
        import rasterio  # noqa: PLC0415
        from rasterio.crs import CRS  # noqa: PLC0415
    except ImportError:
        return {
            "valid": False,
            "error": "rasterio is not installed; cannot perform semantic validation",
        }

    try:
        with rasterio.open(path) as ds:
            crs = ds.crs.to_epsg() if ds.crs else None
            bounds = ds.bounds  # BoundingBox(left, bottom, right, top)
            width, height = ds.width, ds.height
            count = ds.count
            nodata = ds.nodata
            transform = ds.transform
    except Exception as exc:
        return {"valid": False, "error": f"rasterio failed to open file: {exc}"}

    crs_str = f"EPSG:{crs}" if crs else "unknown"
    if expected_crs is not None:
        try:
            expected = CRS.from_string(expected_crs)
            actual = CRS.from_epsg(crs) if crs else None
            if actual is None or not actual.equals(expected):
                return {
                    "valid": False,
                    "error": f"CRS mismatch: expected {expected_crs}, got {crs_str}",
                }
        except Exception as exc:
            return {"valid": False, "error": f"CRS comparison failed: {exc}"}

    result: dict[str, Any] = {
        "valid": True,
        "file_size_bytes": path.stat().st_size,
        "crs": crs_str,
        "bounds": {
            "west": bounds.left,
            "south": bounds.bottom,
            "east": bounds.right,
            "north": bounds.top,
        },
        "width": width,
        "height": height,
        "band_count": count,
        "nodata": nodata,
        "pixel_size_deg": abs(transform.a),
    }

    if municipality_bbox is not None:
        west, south, east, north = municipality_bbox
        covers = (
            bounds.left <= west
            and bounds.right >= east
            and bounds.bottom <= south
            and bounds.top >= north
        )
        result["covers_municipality"] = covers
        if not covers:
            result["valid"] = False
            result["error"] = (
                f"Raster bounds {dict(result['bounds'])} do not fully cover "
                f"municipality bbox (W={west}, S={south}, E={east}, N={north})"
            )

    return result


def validate_file(manifest: DatasetManifest, data_dir: Path) -> dict[str, Any]:
    """Dispatch format-aware validation for the manifest's local file."""
    path = resolve_local_path(manifest, data_dir)
    if path is None:
        return {"valid": False, "error": "No local_path defined in manifest"}
    if not path.exists():
        return {"valid": False, "error": f"File not found: {path}"}

    fmt = (manifest.expected_format or "").lower()
    if fmt == "csv":
        return validate_csv_file(path, category=manifest.category.value)
    if fmt in ("zip", "shapefile"):
        return validate_zip_file(path)
    if fmt == "pbf":
        return validate_pbf_file(path)
    if fmt in ("excel", "xlsx"):
        return validate_xlsx_file(path)
    if fmt in ("geotiff", "tiff"):
        return validate_geotiff_file(path)

    size = path.stat().st_size
    return {
        "valid": True,
        "file_size_bytes": size,
        "note": f"Format '{fmt}' — existence and size only.",
    }
