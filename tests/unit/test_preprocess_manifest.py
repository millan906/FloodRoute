"""Unit tests for preprocessing manifest generation and I/O (Stage 3).

All tests use synthetic data in tmp_path; no real data is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodroute.preprocessing.prep_manifest import (
    build_preprocessing_manifest,
    compute_file_sha256,
    read_preprocessing_manifest,
    validate_preprocessing_manifests,
    write_preprocessing_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_file(tmp_path: Path, content: bytes = b"hello") -> Path:
    """Write a tiny file and return its path."""
    p = tmp_path / "tiny.bin"
    p.write_bytes(content)
    return p


def _make_manifest(tmp_path: Path, output_file: Path) -> dict:
    """Return a minimal but valid preprocessing manifest dict."""
    return build_preprocessing_manifest(
        output_id="test_output",
        operation="test_op",
        parameters={"key": "value"},
        source_dataset_ids=["phl_admin_boundaries"],
        source_checksums={"phl_admin_boundaries": "abc123"},
        output_path=output_file,
        output_crs="EPSG:4326",
        output_bounds={"west": 121.8, "south": 10.5, "east": 122.3, "north": 10.9},
        feature_count=3,
    )


# ---------------------------------------------------------------------------
# Tests: compute_file_sha256
# ---------------------------------------------------------------------------


class TestComputeFileSha256:
    def test_deterministic(self, tmp_path: Path) -> None:
        """SHA-256 is the same on repeated calls for the same content."""
        p = _write_tiny_file(tmp_path, b"test content")
        assert compute_file_sha256(p) == compute_file_sha256(p)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different file contents produce different digests."""
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"content A")
        p2.write_bytes(b"content B")
        assert compute_file_sha256(p1) != compute_file_sha256(p2)

    def test_known_hash(self, tmp_path: Path) -> None:
        """SHA-256 of empty bytes is the well-known constant."""
        import hashlib

        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_file_sha256(p) == expected


# ---------------------------------------------------------------------------
# Tests: build_preprocessing_manifest
# ---------------------------------------------------------------------------


class TestBuildPreprocessingManifest:
    def test_required_fields_present(self, tmp_path: Path) -> None:
        """Manifest dict contains all required top-level fields."""
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)

        for field in (
            "schema_version",
            "manifest_type",
            "output_id",
            "created_at",
            "source_datasets",
            "operation",
            "parameters",
            "software",
            "output",
        ):
            assert field in m, f"Missing field: {field}"

    def test_manifest_type_is_preprocessing(self, tmp_path: Path) -> None:
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        assert m["manifest_type"] == "preprocessing"

    def test_output_checksum_is_sha256_hex(self, tmp_path: Path) -> None:
        """Output checksum is a 64-character lowercase hex string."""
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        sha = m["output"]["sha256"]
        assert len(sha) == 64
        assert sha == sha.lower()
        assert all(c in "0123456789abcdef" for c in sha)

    def test_feature_count_in_output(self, tmp_path: Path) -> None:
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        assert m["output"]["feature_count"] == 3

    def test_raster_fields_present_when_provided(self, tmp_path: Path) -> None:
        out = _write_tiny_file(tmp_path)
        m = build_preprocessing_manifest(
            output_id="dem_test",
            operation="dem_clip_reproject",
            parameters={"dst_crs": "EPSG:32651"},
            source_dataset_ids=["dem_antique_municipalities"],
            source_checksums={},
            output_path=out,
            output_crs="EPSG:32651",
            output_bounds={"xmin": 300000, "ymin": 1100000, "xmax": 350000, "ymax": 1200000},
            raster_width=150,
            raster_height=200,
            raster_nodata=-9999.0,
            raster_dtype="float32",
        )
        assert m["output"]["width"] == 150
        assert m["output"]["height"] == 200
        assert m["output"]["nodata"] == -9999.0
        assert m["output"]["dtype"] == "float32"

    def test_software_includes_python_and_floodroute(self, tmp_path: Path) -> None:
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        assert "python" in m["software"]
        assert "floodroute" in m["software"]

    def test_source_dataset_list(self, tmp_path: Path) -> None:
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        assert m["source_datasets"][0]["dataset_id"] == "phl_admin_boundaries"
        assert m["source_datasets"][0]["sha256"] == "abc123"


# ---------------------------------------------------------------------------
# Tests: write/read preprocessing manifest
# ---------------------------------------------------------------------------


class TestWriteReadManifest:
    def test_round_trip(self, tmp_path: Path) -> None:
        """A manifest written to disk can be read back identically."""
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        mpath = tmp_path / "manifests" / "test_output.json"
        write_preprocessing_manifest(m, mpath)

        assert mpath.exists()
        loaded = read_preprocessing_manifest(mpath)
        assert loaded["output_id"] == "test_output"
        assert loaded["manifest_type"] == "preprocessing"
        assert loaded["output"]["sha256"] == m["output"]["sha256"]

    def test_json_is_valid(self, tmp_path: Path) -> None:
        """Written manifest is valid JSON."""
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        mpath = tmp_path / "test.json"
        write_preprocessing_manifest(m, mpath)

        with mpath.open() as fh:
            parsed = json.load(fh)
        assert parsed["output_id"] == "test_output"

    def test_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        """read_preprocessing_manifest raises ValueError for unknown schema_version."""
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps({"schema_version": "99.0", "manifest_type": "preprocessing"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema_version"):
            read_preprocessing_manifest(bad)

    def test_wrong_manifest_type_raises(self, tmp_path: Path) -> None:
        """read_preprocessing_manifest raises ValueError for wrong manifest_type."""
        bad = tmp_path / "bad2.json"
        bad.write_text(
            json.dumps({"schema_version": "1.0", "manifest_type": "acquisition"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="manifest_type"):
            read_preprocessing_manifest(bad)

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        """write_preprocessing_manifest creates missing parent directories."""
        out = _write_tiny_file(tmp_path)
        m = _make_manifest(tmp_path, out)
        deep = tmp_path / "a" / "b" / "c" / "manifest.json"
        write_preprocessing_manifest(m, deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# Tests: validate_preprocessing_manifests
# ---------------------------------------------------------------------------


class TestValidatePreprocessingManifests:
    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        results = validate_preprocessing_manifests(tmp_path)
        assert results == []

    def test_valid_manifests_returned(self, tmp_path: Path) -> None:
        out = _write_tiny_file(tmp_path)
        m1 = _make_manifest(tmp_path, out)
        m2 = build_preprocessing_manifest(
            output_id="second",
            operation="dem_clip_reproject",
            parameters={},
            source_dataset_ids=["dem_antique_municipalities"],
            source_checksums={},
            output_path=out,
            output_crs="EPSG:32651",
            output_bounds={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        )
        write_preprocessing_manifest(m1, tmp_path / "first.json")
        write_preprocessing_manifest(m2, tmp_path / "second.json")

        results = validate_preprocessing_manifests(tmp_path)
        assert len(results) == 2
        ids = {r["output_id"] for r in results}
        assert ids == {"test_output", "second"}

    def test_invalid_manifest_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps({"schema_version": "0.0", "manifest_type": "preprocessing"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            validate_preprocessing_manifests(tmp_path)
