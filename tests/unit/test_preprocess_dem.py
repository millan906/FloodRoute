"""Unit tests for DEM preprocessing (Stage 3).

Uses tiny synthetic rasters (5×5 pixels) created in tmp_path.
Never touches real research data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from floodroute.preprocessing.dem import (
    DemProcessingError,
    OutputExistsError,
    build_dem_vrt,
    clip_and_reproject_dem,
    read_raster_metadata,
)

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

_WGS84 = CRS.from_epsg(4326)
_UTM51N = CRS.from_epsg(32651)


def _make_tile(
    path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    width: int = 5,
    height: int = 5,
    fill: float = 10.0,
    nodata: float | None = None,
    crs: CRS = _WGS84,
) -> Path:
    """Write a tiny synthetic float32 GeoTIFF."""
    transform = from_bounds(west, south, east, north, width, height)
    data = np.full((1, height, width), fill, dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Tests: build_dem_vrt
# ---------------------------------------------------------------------------


class TestBuildDemVrt:
    """Tests for build_dem_vrt()."""

    def test_vrt_created(self, tmp_path: Path) -> None:
        """VRT file is created when two valid tiles are supplied."""
        t1 = _make_tile(tmp_path / "tile1.tif", west=120.0, south=10.0, east=121.0, north=11.0)
        t2 = _make_tile(tmp_path / "tile2.tif", west=121.0, south=10.0, east=122.0, north=11.0)
        vrt_path = tmp_path / "mosaic.vrt"

        result = build_dem_vrt([t1, t2], vrt_path)

        assert result == vrt_path
        assert vrt_path.exists()
        content = vrt_path.read_text()
        assert "VRTDataset" in content
        assert "tile1.tif" in content
        assert "tile2.tif" in content

    def test_vrt_is_readable_by_rasterio(self, tmp_path: Path) -> None:
        """rasterio can open the generated VRT and reports correct combined dimensions."""
        t1 = _make_tile(tmp_path / "t1.tif", west=120.0, south=10.0, east=121.0, north=11.0)
        t2 = _make_tile(tmp_path / "t2.tif", west=121.0, south=10.0, east=122.0, north=11.0)
        vrt_path = tmp_path / "mosaic.vrt"
        build_dem_vrt([t1, t2], vrt_path)

        with rasterio.open(vrt_path) as ds:
            # Combined width should be sum of both tile widths
            assert ds.width == 10  # 5 + 5
            assert ds.height == 5

    def test_missing_tile_raises(self, tmp_path: Path) -> None:
        """DemProcessingError raised when a tile file is absent."""
        t1 = _make_tile(tmp_path / "tile1.tif", west=120.0, south=10.0, east=121.0, north=11.0)
        missing = tmp_path / "nonexistent.tif"
        with pytest.raises(DemProcessingError, match="not found"):
            build_dem_vrt([t1, missing], tmp_path / "mosaic.vrt")

    def test_wrong_number_of_tiles_raises(self, tmp_path: Path) -> None:
        """DemProcessingError raised when tile count != 2."""
        t1 = _make_tile(tmp_path / "tile1.tif", west=120.0, south=10.0, east=121.0, north=11.0)
        with pytest.raises(DemProcessingError, match="exactly two"):
            build_dem_vrt([t1], tmp_path / "mosaic.vrt")


# ---------------------------------------------------------------------------
# Tests: clip_and_reproject_dem
# ---------------------------------------------------------------------------


class TestClipAndReprojectDem:
    """Tests for clip_and_reproject_dem()."""

    def _two_tile_vrt(self, tmp_path: Path) -> Path:
        t1 = _make_tile(
            tmp_path / "t1.tif",
            west=121.9,
            south=10.6,
            east=122.0,
            north=10.8,
            width=5,
            height=5,
            fill=50.0,
        )
        t2 = _make_tile(
            tmp_path / "t2.tif",
            west=122.0,
            south=10.6,
            east=122.1,
            north=10.8,
            width=5,
            height=5,
            fill=80.0,
        )
        vrt_path = tmp_path / "mosaic.vrt"
        build_dem_vrt([t1, t2], vrt_path)
        return vrt_path

    def test_clip_and_reproject_creates_output(self, tmp_path: Path) -> None:
        """Output GeoTIFF is created with the requested CRS."""
        vrt = self._two_tile_vrt(tmp_path)
        out = tmp_path / "clipped.tif"
        clip_geom = box(121.92, 10.62, 121.98, 10.76)

        result = clip_and_reproject_dem(vrt, clip_geom, out, dst_crs="EPSG:32651", nodata=-9999.0)

        assert out.exists()
        assert result["crs"] == "EPSG:32651"
        assert result["width"] > 0
        assert result["height"] > 0
        assert result["nodata"] == -9999.0

    def test_nodata_preserved_in_output(self, tmp_path: Path) -> None:
        """Output raster has -9999.0 nodata sentinel set."""
        vrt = self._two_tile_vrt(tmp_path)
        out = tmp_path / "clipped_nd.tif"
        clip_geom = box(121.92, 10.62, 121.98, 10.76)

        clip_and_reproject_dem(vrt, clip_geom, out, dst_crs="EPSG:32651", nodata=-9999.0)

        with rasterio.open(out) as ds:
            assert ds.nodata == -9999.0

    def test_overwrite_protection(self, tmp_path: Path) -> None:
        """OutputExistsError raised when output exists and force=False."""
        vrt = self._two_tile_vrt(tmp_path)
        out = tmp_path / "existing.tif"
        out.write_text("placeholder")

        with pytest.raises(OutputExistsError):
            clip_and_reproject_dem(
                vrt, box(121.92, 10.62, 121.98, 10.76), out, dst_crs="EPSG:32651", force=False
            )

    def test_force_overwrites(self, tmp_path: Path) -> None:
        """force=True allows overwriting an existing output."""
        vrt = self._two_tile_vrt(tmp_path)
        out = tmp_path / "overwrite_me.tif"
        out.write_text("old content")

        result = clip_and_reproject_dem(
            vrt,
            box(121.92, 10.62, 121.98, 10.76),
            out,
            dst_crs="EPSG:32651",
            force=True,
            nodata=-9999.0,
        )

        assert out.exists()
        assert result["crs"] == "EPSG:32651"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        """dry_run=True returns metadata without writing the output file."""
        vrt = self._two_tile_vrt(tmp_path)
        out = tmp_path / "dry_run_out.tif"

        result = clip_and_reproject_dem(
            vrt,
            box(121.92, 10.62, 121.98, 10.76),
            out,
            dst_crs="EPSG:32651",
            dry_run=True,
            nodata=-9999.0,
        )

        assert not out.exists(), "dry_run should not write the output file"
        assert result["crs"] == "EPSG:32651"
        assert result["width"] > 0

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        """DemProcessingError raised when source raster does not exist."""
        with pytest.raises(DemProcessingError, match="not found"):
            clip_and_reproject_dem(
                tmp_path / "nonexistent.tif",
                box(121.9, 10.6, 122.0, 10.8),
                tmp_path / "out.tif",
                dst_crs="EPSG:32651",
            )

    def test_invalid_resampling_raises(self, tmp_path: Path) -> None:
        """DemProcessingError raised for an unrecognised resampling method name."""
        vrt = self._two_tile_vrt(tmp_path)
        with pytest.raises(DemProcessingError, match="resampling"):
            clip_and_reproject_dem(
                vrt,
                box(121.92, 10.62, 121.98, 10.76),
                tmp_path / "out.tif",
                dst_crs="EPSG:32651",
                resampling="unicorn",
            )


# ---------------------------------------------------------------------------
# Tests: read_raster_metadata
# ---------------------------------------------------------------------------


class TestReadRasterMetadata:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        t = _make_tile(tmp_path / "meta.tif", 121.0, 10.0, 122.0, 11.0)
        meta = read_raster_metadata(t)
        for key in ("crs", "width", "height", "nodata", "bounds", "dtype"):
            assert key in meta, f"Missing key: {key}"

    def test_missing_raster_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DemProcessingError, match="not found"):
            read_raster_metadata(tmp_path / "does_not_exist.tif")
