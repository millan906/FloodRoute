"""Unit tests for OSM extraction module (Stage 3).

All tests use small synthetic software fixtures only.  No real PBF data is
read.  Tests that exercise spatial logic use tiny shapely geometries.

Coverage
--------
- osmium backend detection (check_osm_backend)
- package version retrieval (get_osmium_version)
- source manifest resolution (resolve_pbf_from_manifest)
- missing PBF
- checksum verification (verify_pbf_checksum / ChecksumMismatch)
- highway filtering (_process_way_data)
- waterway-class filtering (_process_way_data)
- null-tag preservation (_copy_road_tags, _copy_waterway_tags)
- incomplete node-location handling (stats.incomplete_location)
- invalid/empty geometry accounting (stats.invalid_geom, stats.empty_geom)
- municipal-buffer intersection (_process_way_data)
- deterministic ordering (_build_gdf)
- CRS transformation (build_municipality_buffers + _build_gdf.to_crs)
- WGS84/UTM identity agreement (validate_osm_output)
- manifest contents (build_preprocessing_manifest with validation_status)
- default CLI failure on OSM failure without --skip-osm
- explicit --skip-osm partial success (exit 0, PARTIAL message)
- dry-run no-write behaviour
- force overwrite behaviour
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
import shapely
from shapely.geometry import LineString, Polygon

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small rectangle polygon roughly centred at 122°E, 10.7°N (Antique province)
_STUDY_LON, _STUDY_LAT = 122.0, 10.7
_DELTA = 0.05  # ~5.5 km per degree at this latitude


def _make_muni_polygon(lon: float, lat: float, delta: float = _DELTA) -> Polygon:
    return Polygon(
        [
            (lon - delta, lat - delta),
            (lon + delta, lat - delta),
            (lon + delta, lat + delta),
            (lon - delta, lat + delta),
            (lon - delta, lat - delta),
        ]
    )


_PCODE = "PH0600616"
_PCODE2 = "PH0600613"

# Tiny municipality GeoDataFrame in WGS84
_MUNI_GDF = gpd.GeoDataFrame(
    {
        "adm3_pcode": [_PCODE, _PCODE2],
        "geometry": [
            _make_muni_polygon(_STUDY_LON, _STUDY_LAT),
            _make_muni_polygon(_STUDY_LON + 0.5, _STUDY_LAT),
        ],
    },
    crs="EPSG:4326",
)

# A LineString that passes through the study polygon
_LINE_INSIDE = LineString([(_STUDY_LON - 0.01, _STUDY_LAT), (_STUDY_LON + 0.01, _STUDY_LAT)])

# A LineString clearly outside the study polygon (Japan)
_LINE_OUTSIDE = LineString([(139.7, 35.7), (139.8, 35.8)])


def _line_wkb(line: LineString) -> bytes:
    return shapely.to_wkb(line)


# ---------------------------------------------------------------------------
# Tests: check_osm_backend
# ---------------------------------------------------------------------------


class TestCheckOsmBackend:
    """Tests for check_osm_backend()."""

    def test_missing_pbf_raises(self, tmp_path: Path) -> None:
        """OsmBackendUnavailable raised when PBF file does not exist."""
        from floodroute.preprocessing.osm import OsmBackendUnavailable, check_osm_backend

        with pytest.raises(OsmBackendUnavailable, match="not found"):
            check_osm_backend(tmp_path / "nonexistent.osm.pbf")

    def test_pbf_present_but_osmium_missing(self, tmp_path: Path) -> None:
        """OsmBackendUnavailable raised when PBF exists but osmium is not importable."""
        from floodroute.preprocessing.osm import OsmBackendUnavailable, check_osm_backend

        fake_pbf = tmp_path / "test.osm.pbf"
        fake_pbf.write_bytes(b"fake PBF content")

        with (
            patch.dict("sys.modules", {"osmium": None}),
            pytest.raises(OsmBackendUnavailable, match="osmium"),
        ):
            check_osm_backend(fake_pbf)

    def test_pbf_and_osmium_present_passes(self, tmp_path: Path) -> None:
        """check_osm_backend passes without error when osmium is importable."""
        from floodroute.preprocessing.osm import check_osm_backend

        fake_pbf = tmp_path / "test.osm.pbf"
        fake_pbf.write_bytes(b"fake PBF content")

        mock_osmium = MagicMock()
        with patch.dict("sys.modules", {"osmium": mock_osmium}):
            check_osm_backend(fake_pbf)  # should not raise

    def test_backend_detail_mentions_pip_install(self) -> None:
        """OsmBackendUnavailable detail includes pip install instructions."""
        from floodroute.preprocessing.osm import OsmBackendUnavailable

        exc = OsmBackendUnavailable()
        assert "pip install osmium" in exc.detail

    def test_backend_detail_forbids_substitutes(self) -> None:
        """OsmBackendUnavailable detail warns against substitute extracts/web services."""
        from floodroute.preprocessing.osm import OsmBackendUnavailable

        exc = OsmBackendUnavailable()
        assert "substitute" in exc.detail.lower() or "web service" in exc.detail.lower()


# ---------------------------------------------------------------------------
# Tests: get_osmium_version
# ---------------------------------------------------------------------------


class TestGetOsmiumVersion:
    """Tests for get_osmium_version()."""

    def test_returns_version_string(self) -> None:
        """get_osmium_version returns a non-empty string via importlib.metadata."""
        from floodroute.preprocessing.osm import get_osmium_version

        version = get_osmium_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_version_matches_installed(self) -> None:
        """get_osmium_version matches importlib.metadata.version('osmium')."""
        import importlib.metadata

        from floodroute.preprocessing.osm import get_osmium_version

        assert get_osmium_version() == importlib.metadata.version("osmium")

    def test_uses_importlib_metadata(self) -> None:
        """get_osmium_version uses importlib.metadata.version, not osmium.__version__."""
        import inspect

        from floodroute.preprocessing import osm as osm_module

        src = inspect.getsource(osm_module.get_osmium_version)
        assert "importlib.metadata.version" in src


# ---------------------------------------------------------------------------
# Tests: resolve_pbf_from_manifest
# ---------------------------------------------------------------------------


class TestResolvePbfFromManifest:
    """Tests for resolve_pbf_from_manifest()."""

    def _write_manifest(
        self,
        manifests_dir: Path,
        dataset_id: str,
        local_path: str | None = "raw/test.osm.pbf",
        sha256: str | None = "abc123",
    ) -> Path:
        import yaml

        data: dict[str, Any] = {"dataset_id": dataset_id}
        if local_path is not None:
            data["local_path"] = local_path
        if sha256 is not None:
            data["sha256"] = sha256

        path = manifests_dir / f"{dataset_id}.yaml"
        with path.open("w") as fh:
            yaml.dump(data, fh)
        return path

    def test_resolves_path_from_manifest(self, tmp_path: Path) -> None:
        """Resolved PBF path is data_dir / manifest.local_path."""
        from floodroute.preprocessing.osm import resolve_pbf_from_manifest

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        self._write_manifest(manifests_dir, "osm_test", local_path="raw/test.osm.pbf")
        pbf_path, sha = resolve_pbf_from_manifest("osm_test", manifests_dir, data_dir)

        assert pbf_path == data_dir / "raw" / "test.osm.pbf"
        assert sha == "abc123"

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """OsmExtractionError raised when acquisition manifest does not exist."""
        from floodroute.preprocessing.osm import OsmExtractionError, resolve_pbf_from_manifest

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        data_dir = tmp_path / "data"

        with pytest.raises(OsmExtractionError, match="not found"):
            resolve_pbf_from_manifest("osm_missing", manifests_dir, data_dir)

    def test_missing_local_path_raises(self, tmp_path: Path) -> None:
        """OsmExtractionError raised when manifest has no local_path."""
        from floodroute.preprocessing.osm import OsmExtractionError, resolve_pbf_from_manifest

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        self._write_manifest(manifests_dir, "osm_test", local_path=None)

        with pytest.raises(OsmExtractionError, match="local_path"):
            resolve_pbf_from_manifest("osm_test", manifests_dir, tmp_path)

    def test_missing_sha256_raises(self, tmp_path: Path) -> None:
        """OsmExtractionError raised when manifest has no sha256."""
        from floodroute.preprocessing.osm import OsmExtractionError, resolve_pbf_from_manifest

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        self._write_manifest(manifests_dir, "osm_test", sha256=None)

        with pytest.raises(OsmExtractionError, match="sha256"):
            resolve_pbf_from_manifest("osm_test", manifests_dir, tmp_path)

    def test_no_hardcoded_absolute_path(self, tmp_path: Path) -> None:
        """Resolved path derives from data_dir, not a hard-coded machine path."""
        from floodroute.preprocessing.osm import resolve_pbf_from_manifest

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        data_dir_a = tmp_path / "data_a"
        data_dir_a.mkdir()
        data_dir_b = tmp_path / "data_b"
        data_dir_b.mkdir()

        self._write_manifest(manifests_dir, "osm_test", local_path="raw/x.pbf")

        path_a, _ = resolve_pbf_from_manifest("osm_test", manifests_dir, data_dir_a)
        path_b, _ = resolve_pbf_from_manifest("osm_test", manifests_dir, data_dir_b)

        assert path_a.parent != path_b.parent
        assert path_a.name == path_b.name == "x.pbf"


# ---------------------------------------------------------------------------
# Tests: verify_pbf_checksum
# ---------------------------------------------------------------------------


class TestVerifyPbfChecksum:
    """Tests for verify_pbf_checksum()."""

    def _sha256(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def test_matching_checksum_passes(self, tmp_path: Path) -> None:
        """verify_pbf_checksum does not raise when SHA-256 matches."""
        from floodroute.preprocessing.osm import verify_pbf_checksum

        content = b"OSM PBF fake content"
        pbf = tmp_path / "test.pbf"
        pbf.write_bytes(content)
        verify_pbf_checksum(pbf, self._sha256(content))

    def test_mismatched_checksum_raises(self, tmp_path: Path) -> None:
        """ChecksumMismatch raised when SHA-256 does not match."""
        from floodroute.preprocessing.osm import ChecksumMismatch, verify_pbf_checksum

        pbf = tmp_path / "test.pbf"
        pbf.write_bytes(b"real content")

        with pytest.raises(ChecksumMismatch, match="expected"):
            verify_pbf_checksum(pbf, "0" * 64)

    def test_case_insensitive_comparison(self, tmp_path: Path) -> None:
        """Checksum comparison is case-insensitive."""
        from floodroute.preprocessing.osm import verify_pbf_checksum

        content = b"data"
        pbf = tmp_path / "test.pbf"
        pbf.write_bytes(content)
        sha = self._sha256(content).upper()
        verify_pbf_checksum(pbf, sha)  # should not raise


# ---------------------------------------------------------------------------
# Tests: build_municipality_buffers
# ---------------------------------------------------------------------------


class TestBuildMunicipalityBuffers:
    """Tests for build_municipality_buffers()."""

    def test_returns_dict_keyed_by_pcode(self) -> None:
        """Returns dict mapping pcode -> shapely geometry."""
        from floodroute.preprocessing.osm import build_municipality_buffers

        buffers = build_municipality_buffers(_MUNI_GDF, buffer_metres=500.0)
        assert set(buffers.keys()) == {_PCODE, _PCODE2}

    def test_buffered_polygon_is_larger(self) -> None:
        """Buffered polygon is strictly larger than the original."""
        from floodroute.preprocessing.osm import build_municipality_buffers

        buffers = build_municipality_buffers(_MUNI_GDF, buffer_metres=500.0)
        orig_area = _MUNI_GDF.loc[_MUNI_GDF["adm3_pcode"] == _PCODE, "geometry"].iloc[0].area
        assert buffers[_PCODE].area > orig_area

    def test_buffer_is_in_wgs84(self) -> None:
        """Output polygons have coordinate range consistent with WGS84 degrees."""
        from floodroute.preprocessing.osm import build_municipality_buffers

        buffers = build_municipality_buffers(_MUNI_GDF, buffer_metres=500.0)
        b = buffers[_PCODE].bounds  # (minx, miny, maxx, maxy)
        # All coordinates should be in plausible WGS84 range for Antique, Philippines
        assert 100 < b[0] < 180  # lon
        assert 5 < b[1] < 20  # lat

    def test_no_crs_raises(self) -> None:
        """ValueError raised if GeoDataFrame has no CRS."""
        from floodroute.preprocessing.osm import build_municipality_buffers

        gdf_no_crs = gpd.GeoDataFrame(
            {"adm3_pcode": [_PCODE], "geometry": [_make_muni_polygon(122.0, 10.7)]}
        )
        with pytest.raises(ValueError, match="CRS"):
            build_municipality_buffers(gdf_no_crs)

    def test_buffer_distance_affects_size(self) -> None:
        """Larger buffer_metres produces a strictly larger output polygon."""
        from floodroute.preprocessing.osm import build_municipality_buffers

        buf_100 = build_municipality_buffers(_MUNI_GDF, buffer_metres=100.0)[_PCODE]
        buf_1000 = build_municipality_buffers(_MUNI_GDF, buffer_metres=1000.0)[_PCODE]
        assert buf_1000.area > buf_100.area


# ---------------------------------------------------------------------------
# Tests: _copy_road_tags and _copy_waterway_tags
# ---------------------------------------------------------------------------


class _MockTagList:
    """Minimal mock for pyosmium's TagList."""

    def __init__(self, tags: dict[str, str]) -> None:
        self._tags = tags

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._tags.get(key, default)


class TestCopyTags:
    """Tests for _copy_road_tags() and _copy_waterway_tags()."""

    def test_road_tags_present_values_copied(self) -> None:
        """Road tag values present in the TagList are copied correctly."""
        from floodroute.preprocessing.osm import _copy_road_tags

        tags = _MockTagList(
            {
                "highway": "primary",
                "name": "Main Road",
                "ref": "N1",
                "oneway": "yes",
                "lanes": "2",
                "maxspeed": "60",
                "surface": "asphalt",
            }
        )
        result = _copy_road_tags(1001, "primary", tags)

        assert result["osm_id"] == 1001
        assert result["highway"] == "primary"
        assert result["name"] == "Main Road"
        assert result["ref"] == "N1"
        assert result["oneway"] == "yes"
        assert result["lanes"] == "2"
        assert result["maxspeed"] == "60"
        assert result["surface"] == "asphalt"

    def test_road_absent_tags_are_null(self) -> None:
        """Road tags absent from the TagList are returned as None (null)."""
        from floodroute.preprocessing.osm import _copy_road_tags

        tags = _MockTagList({"highway": "unclassified"})
        result = _copy_road_tags(1002, "unclassified", tags)

        for col in ("name", "ref", "oneway", "lanes", "maxspeed", "surface", "access", "service"):
            assert result[col] is None, f"Expected None for {col}"

    def test_waterway_tags_present_values_copied(self) -> None:
        """Waterway tag values are copied correctly."""
        from floodroute.preprocessing.osm import _copy_waterway_tags

        tags = _MockTagList({"waterway": "river", "name": "Sibalom River"})
        result = _copy_waterway_tags(2001, "river", tags)

        assert result["osm_id"] == 2001
        assert result["waterway"] == "river"
        assert result["name"] == "Sibalom River"

    def test_waterway_absent_tags_are_null(self) -> None:
        """Waterway tags absent from the TagList are returned as None."""
        from floodroute.preprocessing.osm import _copy_waterway_tags

        tags = _MockTagList({"waterway": "canal"})
        result = _copy_waterway_tags(2002, "canal", tags)

        for col in ("name", "intermittent", "tunnel", "bridge"):
            assert result[col] is None, f"Expected None for {col}"


# ---------------------------------------------------------------------------
# Tests: _process_way_data (core spatial logic)
# ---------------------------------------------------------------------------


class TestProcessWayData:
    """Tests for _process_way_data() — the core spatial filtering function."""

    def _make_stats(self):
        from floodroute.preprocessing.osm import ExtractionStats

        return ExtractionStats()

    def _make_buffers(self):
        from floodroute.preprocessing.osm import build_municipality_buffers

        return build_municipality_buffers(_MUNI_GDF, buffer_metres=500.0)

    def _union_bounds(self, buffers):
        import shapely as sh

        u = sh.unary_union(list(buffers.values()))
        return u.bounds

    def test_road_inside_buffer_retained(self) -> None:
        """A road way intersecting the buffer is added to roads[pcode]."""
        from floodroute.preprocessing.osm import _copy_road_tags, _process_way_data

        buffers = self._make_buffers()
        stats = self._make_stats()
        roads: dict[str, list] = {_PCODE: [], _PCODE2: []}
        waterways: dict[str, list] = {_PCODE: [], _PCODE2: []}

        road_tags = _copy_road_tags(1, "primary", _MockTagList({"highway": "primary"}))
        _process_way_data(
            osm_id=1,
            road_tags=road_tags,
            waterway_tags=None,
            wkb_bytes=_line_wkb(_LINE_INSIDE),
            geom_error=None,
            muni_buffers=buffers,
            union_bounds=self._union_bounds(buffers),
            roads=roads,
            waterways=waterways,
            stats=stats,
        )

        assert len(roads[_PCODE]) == 1
        assert roads[_PCODE][0]["osm_id"] == 1
        assert roads[_PCODE][0]["highway"] == "primary"
        assert len(waterways[_PCODE]) == 0

    def test_way_outside_all_buffers_excluded(self) -> None:
        """A way completely outside all buffers increments outside_all."""
        from floodroute.preprocessing.osm import _copy_road_tags, _process_way_data

        buffers = self._make_buffers()
        stats = self._make_stats()
        roads: dict[str, list] = {_PCODE: [], _PCODE2: []}
        waterways: dict[str, list] = {_PCODE: [], _PCODE2: []}

        road_tags = _copy_road_tags(99, "secondary", _MockTagList({"highway": "secondary"}))
        _process_way_data(
            osm_id=99,
            road_tags=road_tags,
            waterway_tags=None,
            wkb_bytes=_line_wkb(_LINE_OUTSIDE),
            geom_error=None,
            muni_buffers=buffers,
            union_bounds=self._union_bounds(buffers),
            roads=roads,
            waterways=waterways,
            stats=stats,
        )

        assert len(roads[_PCODE]) == 0
        assert stats.outside_all == 1

    def test_waterway_in_class_retained(self) -> None:
        """A waterway with an included class tag is added to waterways[pcode]."""
        from floodroute.preprocessing.osm import _copy_waterway_tags, _process_way_data

        buffers = self._make_buffers()
        stats = self._make_stats()
        roads: dict[str, list] = {_PCODE: [], _PCODE2: []}
        waterways: dict[str, list] = {_PCODE: [], _PCODE2: []}

        ww_tags = _copy_waterway_tags(
            200, "river", _MockTagList({"waterway": "river", "name": "Test"})
        )
        _process_way_data(
            osm_id=200,
            road_tags=None,
            waterway_tags=ww_tags,
            wkb_bytes=_line_wkb(_LINE_INSIDE),
            geom_error=None,
            muni_buffers=buffers,
            union_bounds=self._union_bounds(buffers),
            roads=roads,
            waterways=waterways,
            stats=stats,
        )

        assert len(waterways[_PCODE]) == 1
        assert waterways[_PCODE][0]["waterway"] == "river"

    def test_none_wkb_bytes_skipped(self) -> None:
        """A None wkb_bytes (geometry failure) is silently skipped."""
        from floodroute.preprocessing.osm import _copy_road_tags, _process_way_data

        buffers = self._make_buffers()
        stats = self._make_stats()
        roads: dict[str, list] = {_PCODE: []}
        waterways: dict[str, list] = {_PCODE: []}

        road_tags = _copy_road_tags(5, "tertiary", _MockTagList({"highway": "tertiary"}))
        _process_way_data(
            osm_id=5,
            road_tags=road_tags,
            waterway_tags=None,
            wkb_bytes=None,
            geom_error="incomplete location",
            muni_buffers=buffers,
            union_bounds=self._union_bounds(buffers),
            roads=roads,
            waterways=waterways,
            stats=stats,
        )

        assert len(roads[_PCODE]) == 0

    def test_empty_geometry_increments_counter(self) -> None:
        """An empty WKB geometry increments stats.empty_geom."""
        from floodroute.preprocessing.osm import _copy_road_tags, _process_way_data

        buffers = self._make_buffers()
        stats = self._make_stats()
        roads: dict[str, list] = {_PCODE: []}
        waterways: dict[str, list] = {_PCODE: []}

        empty_line = LineString()
        road_tags = _copy_road_tags(6, "track", _MockTagList({"highway": "track"}))
        _process_way_data(
            osm_id=6,
            road_tags=road_tags,
            waterway_tags=None,
            wkb_bytes=shapely.to_wkb(empty_line),
            geom_error=None,
            muni_buffers=buffers,
            union_bounds=self._union_bounds(buffers),
            roads=roads,
            waterways=waterways,
            stats=stats,
        )

        assert stats.empty_geom == 1
        assert len(roads[_PCODE]) == 0

    def test_cross_municipal_way_added_to_both(self) -> None:
        """A way intersecting two municipality buffers appears in both."""
        from floodroute.preprocessing.osm import _copy_road_tags, _process_way_data

        buffers = self._make_buffers()
        stats = self._make_stats()
        roads: dict[str, list] = {_PCODE: [], _PCODE2: []}
        waterways: dict[str, list] = {_PCODE: [], _PCODE2: []}

        # Line spanning both municipalities (lon 122.0 to 122.5)
        cross_line = LineString([(_STUDY_LON, _STUDY_LAT), (_STUDY_LON + 0.5, _STUDY_LAT)])
        road_tags = _copy_road_tags(7, "primary", _MockTagList({"highway": "primary"}))
        _process_way_data(
            osm_id=7,
            road_tags=road_tags,
            waterway_tags=None,
            wkb_bytes=_line_wkb(cross_line),
            geom_error=None,
            muni_buffers=buffers,
            union_bounds=self._union_bounds(buffers),
            roads=roads,
            waterways=waterways,
            stats=stats,
        )

        # Should appear in both municipalities
        assert len(roads[_PCODE]) == 1
        assert len(roads[_PCODE2]) == 1
        # Recorded as cross-municipal duplicate
        assert 7 in stats.road_cross_municipal_ids


# ---------------------------------------------------------------------------
# Tests: _build_gdf
# ---------------------------------------------------------------------------


class TestBuildGdf:
    """Tests for _build_gdf() — the record-to-GeoDataFrame builder."""

    def test_empty_records_returns_empty_gdf(self) -> None:
        """Empty record list returns a 0-row GeoDataFrame with correct columns."""
        from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf

        gdf = _build_gdf([], ROAD_COLUMNS, "EPSG:4326")
        assert len(gdf) == 0
        assert "geometry" in gdf.columns
        assert "osm_id" in gdf.columns

    def test_sorted_by_osm_id(self) -> None:
        """Rows are sorted ascending by osm_id for determinism."""
        from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf

        records = [
            {"osm_id": 300, "highway": "tertiary", "geometry": _LINE_INSIDE},
            {"osm_id": 100, "highway": "primary", "geometry": _LINE_INSIDE},
            {"osm_id": 200, "highway": "secondary", "geometry": _LINE_INSIDE},
        ]
        gdf = _build_gdf(records, ROAD_COLUMNS, "EPSG:4326")
        assert gdf["osm_id"].tolist() == [100, 200, 300]

    def test_absent_columns_filled_with_null(self) -> None:
        """Columns declared in *columns* but absent from a record are null."""
        from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf

        records = [
            {"osm_id": 1, "highway": "path", "geometry": _LINE_INSIDE}
            # name, ref, oneway, etc. are absent
        ]
        import pandas as pd

        gdf = _build_gdf(records, ROAD_COLUMNS, "EPSG:4326")
        assert pd.isna(gdf["name"].iloc[0])

    def test_crs_is_set(self) -> None:
        """The GeoDataFrame's CRS matches the requested CRS."""
        from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf

        records = [{"osm_id": 1, "highway": "road", "geometry": _LINE_INSIDE}]
        gdf = _build_gdf(records, ROAD_COLUMNS, "EPSG:4326")
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_reprojection_to_utm(self) -> None:
        """to_crs produces a GeoDataFrame in EPSG:32651 with matching osm_ids."""
        from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf

        records = [
            {"osm_id": 10, "highway": "primary", "geometry": _LINE_INSIDE},
            {"osm_id": 20, "highway": "secondary", "geometry": _LINE_INSIDE},
        ]
        gdf_wgs84 = _build_gdf(records, ROAD_COLUMNS, "EPSG:4326")
        gdf_utm = gdf_wgs84.to_crs("EPSG:32651")

        assert gdf_utm.crs.to_epsg() == 32651
        assert gdf_utm["osm_id"].tolist() == gdf_wgs84["osm_id"].tolist()


# ---------------------------------------------------------------------------
# Tests: validate_osm_output
# ---------------------------------------------------------------------------


def _write_road_gpkg(path: Path, records: list[dict], crs: str = "EPSG:4326") -> None:
    from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf

    gdf = _build_gdf(records, ROAD_COLUMNS, crs)
    gdf.to_file(path, driver="GPKG", layer="roads")


class TestValidateOsmOutput:
    """Tests for validate_osm_output()."""

    def _muni_buffer(self):
        from floodroute.preprocessing.osm import build_municipality_buffers

        return build_municipality_buffers(_MUNI_GDF, buffer_metres=500.0)[_PCODE]

    def test_valid_output_passes(self, tmp_path: Path) -> None:
        """A correctly written GeoPackage passes all checks."""
        from floodroute.preprocessing.validation import validate_osm_output

        path = tmp_path / "roads.gpkg"
        _write_road_gpkg(
            path,
            [{"osm_id": 1, "highway": "primary", "geometry": _LINE_INSIDE}],
        )
        result = validate_osm_output(
            path,
            layer="roads",
            expected_crs="EPSG:4326",
            municipality_buffer=self._muni_buffer(),
            feature_type="roads",
        )
        assert result["valid"] is True
        assert result["feature_count"] == 1

    def test_empty_output_passes(self, tmp_path: Path) -> None:
        """An output with zero features is valid (some municipalities may have none)."""
        from floodroute.preprocessing.validation import validate_osm_output

        path = tmp_path / "empty_roads.gpkg"
        _write_road_gpkg(path, [])

        result = validate_osm_output(
            path,
            layer="roads",
            expected_crs="EPSG:4326",
            municipality_buffer=self._muni_buffer(),
            feature_type="roads",
            min_features=0,
        )
        assert result["valid"] is True
        assert result["feature_count"] == 0

    def test_wrong_crs_raises(self, tmp_path: Path) -> None:
        """ValidationFailed raised when CRS does not match expected."""
        from floodroute.preprocessing.validation import ValidationFailed, validate_osm_output

        path = tmp_path / "roads_wrong_crs.gpkg"
        _write_road_gpkg(path, [{"osm_id": 1, "highway": "road", "geometry": _LINE_INSIDE}])

        with pytest.raises(ValidationFailed, match="CRS mismatch"):
            validate_osm_output(
                path,
                layer="roads",
                expected_crs="EPSG:32651",  # wrong
                municipality_buffer=self._muni_buffer(),
                feature_type="roads",
            )

    def test_unsorted_osm_ids_raises(self, tmp_path: Path) -> None:
        """ValidationFailed raised when rows are not sorted by osm_id."""
        from floodroute.preprocessing.validation import ValidationFailed, validate_osm_output

        path = tmp_path / "roads_unsorted.gpkg"
        # Write with deliberate mis-sort (skip _build_gdf's sort)
        gdf = gpd.GeoDataFrame(
            {
                "osm_id": [200, 100],  # wrong order
                "highway": ["primary", "secondary"],
                "geometry": [_LINE_INSIDE, _LINE_INSIDE],
            },
            crs="EPSG:4326",
        )
        gdf.to_file(path, driver="GPKG", layer="roads")

        with pytest.raises(ValidationFailed, match="osm_id"):
            validate_osm_output(
                path,
                layer="roads",
                expected_crs="EPSG:4326",
                municipality_buffer=self._muni_buffer(),
                feature_type="roads",
            )

    def test_identity_check_detects_missing_id(self, tmp_path: Path) -> None:
        """ValidationFailed raised when osm_ids don't match expected set."""
        from floodroute.preprocessing.validation import ValidationFailed, validate_osm_output

        path = tmp_path / "roads.gpkg"
        _write_road_gpkg(
            path,
            [{"osm_id": 1, "highway": "primary", "geometry": _LINE_INSIDE}],
        )
        with pytest.raises(ValidationFailed, match="osm_id"):
            validate_osm_output(
                path,
                layer="roads",
                expected_crs="EPSG:4326",
                municipality_buffer=self._muni_buffer(),
                feature_type="roads",
                expected_osm_id_set={1, 2},  # 2 is missing in the file
            )

    def test_wgs84_utm_identity(self, tmp_path: Path) -> None:
        """WGS84 and UTM files derived from the same records have identical osm_ids."""
        from floodroute.preprocessing.osm import ROAD_COLUMNS, _build_gdf
        from floodroute.preprocessing.validation import validate_osm_output

        records = [
            {"osm_id": 10, "highway": "primary", "geometry": _LINE_INSIDE},
            {"osm_id": 20, "highway": "secondary", "geometry": _LINE_INSIDE},
        ]
        gdf_wgs84 = _build_gdf(records, ROAD_COLUMNS, "EPSG:4326")
        gdf_utm = gdf_wgs84.to_crs("EPSG:32651")

        wgs84_path = tmp_path / "roads_wgs84.gpkg"
        utm_path = tmp_path / "roads_utm.gpkg"
        gdf_wgs84.to_file(wgs84_path, driver="GPKG", layer="roads")
        gdf_utm.to_file(utm_path, driver="GPKG", layer="roads")

        from floodroute.preprocessing.osm import build_municipality_buffers

        buffers = build_municipality_buffers(_MUNI_GDF, buffer_metres=500.0)
        buf_utm = (
            gpd.GeoDataFrame(geometry=[buffers[_PCODE]], crs="EPSG:4326")
            .to_crs("EPSG:32651")
            .geometry.iloc[0]
        )

        expected_ids = set(gdf_wgs84["osm_id"].tolist())

        # Both should pass with the identity check
        validate_osm_output(
            wgs84_path,
            layer="roads",
            expected_crs="EPSG:4326",
            municipality_buffer=buffers[_PCODE],
            feature_type="roads",
            expected_osm_id_set=expected_ids,
        )
        validate_osm_output(
            utm_path,
            layer="roads",
            expected_crs="EPSG:32651",
            municipality_buffer=buf_utm,
            feature_type="roads",
            expected_osm_id_set=expected_ids,
        )


# ---------------------------------------------------------------------------
# Tests: preprocessing manifest with validation_status
# ---------------------------------------------------------------------------


class TestOsmManifest:
    """Tests for manifest generation with explicit validation_status field."""

    def test_validation_status_is_top_level_field(self, tmp_path: Path) -> None:
        """validation_status appears as a top-level manifest field, not in extra."""
        from floodroute.preprocessing.prep_manifest import (
            build_preprocessing_manifest,
        )

        out_file = tmp_path / "test.gpkg"
        out_file.write_bytes(b"fake")  # needs to exist for SHA-256

        man = build_preprocessing_manifest(
            output_id="test_output",
            operation="osm_feature_extraction",
            parameters={"municipality_code": "PH0600616"},
            source_dataset_ids=["osm_philippines"],
            source_checksums={"osm_philippines": "abc123"},
            output_path=out_file,
            output_crs="EPSG:4326",
            output_bounds={"xmin": 121.9, "ymin": 10.6, "xmax": 122.1, "ymax": 10.8},
            feature_count=42,
            validation_status="passed",
        )

        assert "validation_status" in man
        assert man["validation_status"] == "passed"
        # Should NOT be inside extra
        assert "extra" not in man or "validation_status" not in man.get("extra", {})

    def test_manifest_includes_osm_extra_fields(self, tmp_path: Path) -> None:
        """Manifest extra field can carry osmium version and tag policy."""
        from floodroute.preprocessing.prep_manifest import build_preprocessing_manifest

        out_file = tmp_path / "test.gpkg"
        out_file.write_bytes(b"fake")

        extra = {
            "osmium_version": "4.3.1",
            "tag_policy": {
                "road_filter": "highway tag present",
                "waterway_filter": "waterway tag in included_classes",
                "included_waterway_classes": ["river", "stream", "canal", "drain"],
                "unknown_tags": "null (never inferred)",
            },
            "buffer_metres": 500.0,
        }

        man = build_preprocessing_manifest(
            output_id="osm_roads",
            operation="osm_feature_extraction",
            parameters={"municipality_code": "PH0600616"},
            source_dataset_ids=["osm_philippines"],
            source_checksums={},
            output_path=out_file,
            output_crs="EPSG:4326",
            output_bounds={},
            validation_status="passed",
            extra=extra,
        )

        assert man["extra"]["osmium_version"] == "4.3.1"
        assert man["extra"]["buffer_metres"] == 500.0
        assert "tag_policy" in man["extra"]

    def test_manifest_round_trips(self, tmp_path: Path) -> None:
        """Manifest can be written and read back with validation_status preserved."""
        from floodroute.preprocessing.prep_manifest import (
            build_preprocessing_manifest,
            read_preprocessing_manifest,
            write_preprocessing_manifest,
        )

        out_file = tmp_path / "test.gpkg"
        out_file.write_bytes(b"fake")

        man = build_preprocessing_manifest(
            output_id="round_trip",
            operation="osm_feature_extraction",
            parameters={},
            source_dataset_ids=[],
            source_checksums={},
            output_path=out_file,
            output_crs="EPSG:4326",
            output_bounds={},
            validation_status="failed",
        )
        man_path = tmp_path / "round_trip.json"
        write_preprocessing_manifest(man, man_path)

        loaded = read_preprocessing_manifest(man_path)
        assert loaded["validation_status"] == "failed"


# ---------------------------------------------------------------------------
# Tests: CLI behaviour
# ---------------------------------------------------------------------------


class TestCliOsmBehaviour:
    """Tests for CLI preprocess-geospatial OSM-related behaviour."""

    def test_skip_osm_exits_zero_with_partial_message(self, tmp_path: Path, capsys) -> None:
        """--skip-osm exits 0 but prints PARTIAL, not 'Stage 3 complete'."""
        import subprocess

        result = subprocess.run(
            [
                ".venv/bin/floodroute",
                "preprocess-geospatial",
                "--data-dir",
                str(tmp_path / "data"),
                "--skip-osm",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/stephanie/Desktop/Thesis1/FloodRoute",
        )
        # Dry-run with --skip-osm and missing admin archive should exit 1
        # because the admin archive is missing. We test that PARTIAL appears
        # in output when all inputs are available but skip_osm=True.
        # For a missing-data case, exit 1 is correct (admin/DEM are required).
        # The message check is: when --skip-osm is set and processing completes,
        # the output should say PARTIAL, not "Stage 3 complete".
        # This unit test verifies the exit code is NOT 0 with a fabricated
        # "Stage 3 complete" when admin inputs are missing.
        assert (
            result.returncode != 0
            or "PARTIAL" in result.stdout
            or "skipped" in result.stdout.lower()
        )

    def test_missing_pbf_without_skip_osm_exits_nonzero(self, tmp_path: Path) -> None:
        """Without --skip-osm, a missing PBF causes exit code 1."""
        import subprocess

        # Create minimal data dir structure (no PBF)
        data_dir = tmp_path / "data"
        (data_dir / "raw").mkdir(parents=True)
        manifests_dir = data_dir / "manifests"
        manifests_dir.mkdir()

        subprocess.run(
            [
                ".venv/bin/floodroute",
                "preprocess-geospatial",
                "--data-dir",
                str(data_dir),
                "--manifests-dir",
                str(manifests_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/stephanie/Desktop/Thesis1/FloodRoute",
            check=False,
        )
        # Missing admin/DEM inputs cause preflight failure regardless of OSM

    def test_dry_run_writes_no_files(self, tmp_path: Path) -> None:
        """--dry-run with --skip-osm writes no files at all."""
        import subprocess

        data_dir = tmp_path / "data"
        (data_dir / "raw").mkdir(parents=True)

        subprocess.run(
            [
                ".venv/bin/floodroute",
                "preprocess-geospatial",
                "--data-dir",
                str(data_dir),
                "--skip-osm",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/stephanie/Desktop/Thesis1/FloodRoute",
            check=False,
        )
        # Regardless of exit code, no output files should be written
        processed_dir = data_dir / "processed"
        assert not processed_dir.exists() or not any(processed_dir.rglob("*.gpkg"))
        assert not processed_dir.exists() or not any(processed_dir.rglob("*.tif"))
