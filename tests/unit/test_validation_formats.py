"""Unit tests for Stage 2 format-aware validation functions.

Tests for validate_pbf_file, validate_xlsx_file, validate_geotiff_file,
and validate_geotiff_raster. All tests use in-memory fixtures or tmp_path
— no real datasets are downloaded.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from floodroute.acquisition import (
    validate_geotiff_file,
    validate_geotiff_raster,
    validate_pbf_file,
    validate_xlsx_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal OSM PBF: 4-byte big-endian BlobHeader length (13) + BlobHeader
# containing type="OSMHeader" (field 1, wire-type 2) + datasize=0 (field 3),
# padded to 40 bytes so the size check (≥ 32) passes.
# Bytes: \x00\x00\x00\x0d | \x0a\x09 O S M H e a d e r | \x18\x00 | padding
_MOCK_PBF = b"\x00\x00\x00\x0d\x0a\x09OSMHeader\x18\x00" + b"\x00" * 23


def _write_mock_xlsx(path: Path, sheets: list[str] | None = None) -> None:
    """Create a minimal valid OOXML workbook ZIP at *path*."""
    sheets = sheets or ["Sheet1"]
    sheet_els = "".join(
        f'<sheet name="{s}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, s in enumerate(sheets)
    )
    wb_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/ml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheet_els}</sheets>"
        "</workbook>"
    )
    ct_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", ct_xml)
        zf.writestr("xl/workbook.xml", wb_xml)


# ---------------------------------------------------------------------------
# validate_pbf_file
# ---------------------------------------------------------------------------


def test_pbf_valid_magic(tmp_path: Path) -> None:
    f = tmp_path / "valid.osm.pbf"
    f.write_bytes(_MOCK_PBF)
    result = validate_pbf_file(f)
    assert result["valid"] is True
    assert result["file_size_bytes"] == len(_MOCK_PBF)


def test_pbf_missing_file(tmp_path: Path) -> None:
    result = validate_pbf_file(tmp_path / "nonexistent.pbf")
    assert result["valid"] is False
    assert "not found" in result["error"].lower()


def test_pbf_too_small(tmp_path: Path) -> None:
    f = tmp_path / "tiny.pbf"
    f.write_bytes(b"\x00\x01\x02")
    result = validate_pbf_file(f)
    assert result["valid"] is False
    assert "small" in result["error"].lower()


def test_pbf_wrong_magic(tmp_path: Path) -> None:
    f = tmp_path / "fake.pbf"
    f.write_bytes(b"\x00" * 40)  # 40 null bytes — no OSMHeader
    result = validate_pbf_file(f)
    assert result["valid"] is False
    assert "OSMHeader" in result["error"]


def test_pbf_random_binary_rejected(tmp_path: Path) -> None:
    f = tmp_path / "random.pbf"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 50)  # ZIP magic, not PBF
    result = validate_pbf_file(f)
    assert result["valid"] is False


# ---------------------------------------------------------------------------
# validate_xlsx_file
# ---------------------------------------------------------------------------


def test_xlsx_valid_single_sheet(tmp_path: Path) -> None:
    f = tmp_path / "wb.xlsx"
    _write_mock_xlsx(f, sheets=["Population"])
    result = validate_xlsx_file(f)
    assert result["valid"] is True
    assert result["sheet_names"] == ["Population"]
    assert result["sheet_count"] == 1


def test_xlsx_valid_multiple_sheets(tmp_path: Path) -> None:
    f = tmp_path / "wb.xlsx"
    _write_mock_xlsx(f, sheets=["Region VI", "Antique", "Readme"])
    result = validate_xlsx_file(f)
    assert result["valid"] is True
    assert result["sheet_count"] == 3
    assert "Antique" in result["sheet_names"]


def test_xlsx_missing_file(tmp_path: Path) -> None:
    result = validate_xlsx_file(tmp_path / "missing.xlsx")
    assert result["valid"] is False
    assert "not found" in result["error"].lower()


def test_xlsx_bad_zip(tmp_path: Path) -> None:
    f = tmp_path / "bad.xlsx"
    f.write_bytes(b"this is not a ZIP file at all")
    result = validate_xlsx_file(f)
    assert result["valid"] is False
    assert "zip" in result["error"].lower() or "xlsx" in result["error"].lower()


def test_xlsx_zip_without_workbook(tmp_path: Path) -> None:
    """A valid ZIP that lacks xl/workbook.xml is not a valid XLSX."""
    f = tmp_path / "notxlsx.xlsx"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("somefile.txt", "hello")
    result = validate_xlsx_file(f)
    assert result["valid"] is False
    assert "workbook.xml" in result["error"]


def test_xlsx_file_size_reported(tmp_path: Path) -> None:
    f = tmp_path / "wb.xlsx"
    _write_mock_xlsx(f)
    result = validate_xlsx_file(f)
    assert result["valid"] is True
    assert result["file_size_bytes"] == f.stat().st_size


# ---------------------------------------------------------------------------
# validate_geotiff_file
# ---------------------------------------------------------------------------


def test_geotiff_valid_little_endian(tmp_path: Path) -> None:
    f = tmp_path / "le.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 100)
    result = validate_geotiff_file(f)
    assert result["valid"] is True


def test_geotiff_valid_big_endian(tmp_path: Path) -> None:
    f = tmp_path / "be.tif"
    f.write_bytes(b"MM\x00\x2a" + b"\x00" * 100)
    result = validate_geotiff_file(f)
    assert result["valid"] is True


def test_geotiff_valid_bigtiff_le(tmp_path: Path) -> None:
    f = tmp_path / "big_le.tif"
    f.write_bytes(b"II\x2b\x00" + b"\x00" * 100)
    result = validate_geotiff_file(f)
    assert result["valid"] is True


def test_geotiff_valid_bigtiff_be(tmp_path: Path) -> None:
    f = tmp_path / "big_be.tif"
    f.write_bytes(b"MM\x00\x2b" + b"\x00" * 100)
    result = validate_geotiff_file(f)
    assert result["valid"] is True


def test_geotiff_missing_file(tmp_path: Path) -> None:
    result = validate_geotiff_file(tmp_path / "nonexistent.tif")
    assert result["valid"] is False
    assert "not found" in result["error"].lower()


def test_geotiff_too_small(tmp_path: Path) -> None:
    f = tmp_path / "tiny.tif"
    f.write_bytes(b"II\x2a")  # 3 bytes — less than magic minimum
    result = validate_geotiff_file(f)
    assert result["valid"] is False
    assert "small" in result["error"].lower()


def test_geotiff_wrong_magic(tmp_path: Path) -> None:
    f = tmp_path / "fake.tif"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 100)  # ZIP, not TIFF
    result = validate_geotiff_file(f)
    assert result["valid"] is False
    assert "magic" in result["error"].lower()


def test_geotiff_file_size_reported(tmp_path: Path) -> None:
    f = tmp_path / "raster.tif"
    payload = b"II\x2a\x00" + b"\x00" * 200
    f.write_bytes(payload)
    result = validate_geotiff_file(f)
    assert result["valid"] is True
    assert result["file_size_bytes"] == len(payload)


# ---------------------------------------------------------------------------
# validate_geotiff_raster (rasterio-based semantic validation)
# ---------------------------------------------------------------------------


def _make_mock_rasterio_dataset(
    *,
    epsg: int = 4326,
    west: float = 121.0,
    south: float = 10.0,
    east: float = 122.0,
    north: float = 11.0,
    width: int = 3600,
    height: int = 3600,
    count: int = 1,
    nodata: float = -9999.0,
    pixel_size: float = 0.000277778,
):
    """Return a mock rasterio dataset object."""
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    ds = MagicMock()
    ds.crs = CRS.from_epsg(epsg)
    ds.bounds = MagicMock()
    ds.bounds.left = west
    ds.bounds.bottom = south
    ds.bounds.right = east
    ds.bounds.top = north
    ds.width = width
    ds.height = height
    ds.count = count
    ds.nodata = nodata
    ds.transform = from_bounds(west, south, east, north, width, height)
    ds.__enter__ = lambda s: s
    ds.__exit__ = MagicMock(return_value=False)
    return ds


def test_geotiff_raster_valid_basic(tmp_path: Path) -> None:
    f = tmp_path / "dem.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    mock_ds = _make_mock_rasterio_dataset()
    with patch("rasterio.open", return_value=mock_ds):
        result = validate_geotiff_raster(f)
    assert result["valid"] is True
    assert result["crs"] == "EPSG:4326"
    assert result["width"] == 3600
    assert result["band_count"] == 1


def test_geotiff_raster_missing_file(tmp_path: Path) -> None:
    result = validate_geotiff_raster(tmp_path / "nonexistent.tif")
    assert result["valid"] is False
    assert "not found" in result["error"].lower()


def test_geotiff_raster_wrong_magic(tmp_path: Path) -> None:
    f = tmp_path / "bad.tif"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    result = validate_geotiff_raster(f)
    assert result["valid"] is False
    assert "magic" in result["error"].lower()


def test_geotiff_raster_crs_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "dem.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    mock_ds = _make_mock_rasterio_dataset(epsg=32651)  # UTM zone 51N
    with patch("rasterio.open", return_value=mock_ds):
        result = validate_geotiff_raster(f, expected_crs="EPSG:4326")
    assert result["valid"] is False
    assert "CRS mismatch" in result["error"]


def test_geotiff_raster_crs_matches(tmp_path: Path) -> None:
    f = tmp_path / "dem.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    mock_ds = _make_mock_rasterio_dataset(epsg=4326)
    with patch("rasterio.open", return_value=mock_ds):
        result = validate_geotiff_raster(f, expected_crs="EPSG:4326")
    assert result["valid"] is True


def test_geotiff_raster_covers_municipality(tmp_path: Path) -> None:
    """Raster that fully contains the municipality bbox passes."""
    f = tmp_path / "dem.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    # Tile covers 10°N–11°N, 121°E–122°E; municipality is within
    mock_ds = _make_mock_rasterio_dataset(west=121.0, south=10.0, east=122.0, north=11.0)
    sibalom_bbox = (121.964, 10.683, 122.207, 10.845)
    with patch("rasterio.open", return_value=mock_ds):
        result = validate_geotiff_raster(f, municipality_bbox=sibalom_bbox)
    assert result["valid"] is False  # 122.207 > east=122.0 — needs N10_E122 too
    assert "covers_municipality" in result


def test_geotiff_raster_two_tiles_together_cover_sibalom(tmp_path: Path) -> None:
    """Validate coverage reasoning: N10_E121 alone does not cover Sibalom's eastern extent."""
    f = tmp_path / "dem.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    # N10_E122 covers 10°N–11°N, 122°E–123°E — includes Sibalom east
    mock_ds = _make_mock_rasterio_dataset(west=122.0, south=10.0, east=123.0, north=11.0)
    sibalom_east_part = (122.0, 10.683, 122.207, 10.845)
    with patch("rasterio.open", return_value=mock_ds):
        result = validate_geotiff_raster(f, municipality_bbox=sibalom_east_part)
    assert result["valid"] is True
    assert result["covers_municipality"] is True


def test_geotiff_raster_rasterio_open_fails(tmp_path: Path) -> None:
    f = tmp_path / "broken.tif"
    f.write_bytes(b"II\x2a\x00" + b"\x00" * 200)
    with patch("rasterio.open", side_effect=Exception("corrupt file")):
        result = validate_geotiff_raster(f)
    assert result["valid"] is False
    assert "rasterio failed" in result["error"]
