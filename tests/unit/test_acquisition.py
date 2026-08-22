"""Unit tests for acquisition utilities.

All network calls are mocked — no internet access occurs.
File operations use pytest tmp_path for isolation.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from floodroute.acquisition import (
    ChecksumMismatch,
    FileAlreadyExists,
    ManualAcquisitionRequired,
    compute_sha256,
    compute_sha256_bytes,
    download_dataset,
    validate_csv_file,
    validate_zip_file,
)
from floodroute.manifest import DatasetManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_manifest(**kwargs) -> DatasetManifest:
    """Build a minimal valid DatasetManifest, merging provided kwargs."""
    defaults = dict(
        schema_version="1.0",
        dataset_id="test_ds",
        title="Test Dataset",
        category="road_network",
        access_method="direct_download",
        acquisition_status="not_requested",
        provenance="observed",
        validation_status="not_validated",
        source_url="http://example.test/data.csv",
        local_path="raw/test_ds.csv",
    )
    defaults.update(kwargs)
    return DatasetManifest.model_validate(defaults)


# ---------------------------------------------------------------------------
# SHA-256 calculation
# ---------------------------------------------------------------------------


def test_compute_sha256_known_value(tmp_path):
    data = b"hello floodroute"
    f = tmp_path / "test.bin"
    f.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    assert compute_sha256(f) == expected


def test_compute_sha256_bytes_known_value():
    data = b"another test"
    assert compute_sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_compute_sha256_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert compute_sha256(f) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# Refusal to download: non-downloadable access_method
# ---------------------------------------------------------------------------


def test_refuse_download_manual_method():
    manifest = _make_manifest(access_method="manual_download")
    with pytest.raises(ManualAcquisitionRequired):
        download_dataset(manifest, Path("/tmp"))


def test_refuse_download_formal_request():
    manifest = _make_manifest(access_method="formal_request")
    with pytest.raises(ManualAcquisitionRequired):
        download_dataset(manifest, Path("/tmp"))


def test_refuse_download_local_authority():
    manifest = _make_manifest(access_method="local_authority")
    with pytest.raises(ManualAcquisitionRequired):
        download_dataset(manifest, Path("/tmp"))


def test_manual_instructions_include_dataset_id():
    manifest = _make_manifest(
        access_method="manual_download",
        landing_page_url="http://example.test/landing",
    )
    with pytest.raises(ManualAcquisitionRequired) as exc_info:
        download_dataset(manifest, Path("/tmp"))
    assert manifest.dataset_id in exc_info.value.instructions


# ---------------------------------------------------------------------------
# Refusal to download: missing source_url
# ---------------------------------------------------------------------------


def test_refuse_download_missing_url():
    manifest = _make_manifest(source_url=None)
    with pytest.raises(ValueError, match="source_url"):
        download_dataset(manifest, Path("/tmp"))


# ---------------------------------------------------------------------------
# Refusal to overwrite without --force
# ---------------------------------------------------------------------------


def test_refuse_overwrite_without_force(tmp_path):
    manifest = _make_manifest(local_path="raw/test_ds.csv")
    dest = tmp_path / "raw" / "test_ds.csv"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing content")

    with pytest.raises(FileAlreadyExists):
        download_dataset(manifest, tmp_path, force=False)


def test_force_flag_allows_overwrite(tmp_path):
    data = b"new content"
    manifest = _make_manifest(local_path="raw/test_ds.csv")
    dest = tmp_path / "raw" / "test_ds.csv"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old content")

    def fake_stream(url, part, *, timeout):
        part.write_bytes(data)

    with patch("floodroute.acquisition._stream_download", side_effect=fake_stream):
        result = download_dataset(manifest, tmp_path, force=True)

    assert result.read_bytes() == data


# ---------------------------------------------------------------------------
# Mocked successful download
# ---------------------------------------------------------------------------


def test_mocked_successful_download(tmp_path):
    data = b"synthetic road network data"
    manifest = _make_manifest(local_path="raw/test_ds.csv")

    def fake_stream(url, part, *, timeout):
        part.write_bytes(data)

    with patch("floodroute.acquisition._stream_download", side_effect=fake_stream):
        result = download_dataset(manifest, tmp_path)

    assert result.exists()
    assert result.read_bytes() == data
    # Part file must not remain
    assert not result.with_suffix(result.suffix + ".part").exists()


def test_mocked_download_checksum_verified(tmp_path):
    data = b"known data"
    expected_sha = hashlib.sha256(data).hexdigest()
    manifest = _make_manifest(local_path="raw/test_ds.csv", sha256=expected_sha)

    def fake_stream(url, part, *, timeout):
        part.write_bytes(data)

    with patch("floodroute.acquisition._stream_download", side_effect=fake_stream):
        result = download_dataset(manifest, tmp_path)

    assert result.exists()


# ---------------------------------------------------------------------------
# Checksum mismatch
# ---------------------------------------------------------------------------


def test_checksum_mismatch_raises(tmp_path):
    data = b"actual data"
    wrong_sha = "0" * 64  # definitely wrong
    manifest = _make_manifest(local_path="raw/test_ds.csv", sha256=wrong_sha)

    def fake_stream(url, part, *, timeout):
        part.write_bytes(data)

    with (
        patch("floodroute.acquisition._stream_download", side_effect=fake_stream),
        pytest.raises(ChecksumMismatch),
    ):
        download_dataset(manifest, tmp_path)


# ---------------------------------------------------------------------------
# Cleanup after failed download
# ---------------------------------------------------------------------------


def test_cleanup_after_failed_download(tmp_path):
    manifest = _make_manifest(local_path="raw/test_ds.csv")
    dest = tmp_path / "raw" / "test_ds.csv"
    part = dest.with_suffix(dest.suffix + ".part")

    def fake_stream(url, p, *, timeout):
        p.write_bytes(b"partial")
        raise OSError("Simulated network error")

    with (
        pytest.raises(OSError),
        patch("floodroute.acquisition._stream_download", side_effect=fake_stream),
    ):
        download_dataset(manifest, tmp_path)

    # Destination must not exist and .part file must be cleaned up
    assert not dest.exists()
    assert not part.exists()


# ---------------------------------------------------------------------------
# ZIP validation
# ---------------------------------------------------------------------------


def test_zip_valid(tmp_path):
    z = tmp_path / "test.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/roads.geojson", '{"type":"FeatureCollection"}')
        zf.writestr("README.txt", "test archive")

    result = validate_zip_file(z)
    assert result["valid"] is True
    assert result["member_count"] == 2
    assert "data/roads.geojson" in result["members"]


def test_zip_path_traversal_rejected(tmp_path):
    z = tmp_path / "malicious.zip"
    with zipfile.ZipFile(z, "w") as zf:
        # Manually add an entry with a path-traversal name
        info = zipfile.ZipInfo("../../etc/passwd")
        zf.writestr(info, "root:x:0:0")

    result = validate_zip_file(z)
    assert result["valid"] is False
    assert "traversal" in result["error"].lower() or "unsafe" in result["error"].lower()


def test_zip_missing_file(tmp_path):
    result = validate_zip_file(tmp_path / "nonexistent.zip")
    assert result["valid"] is False
    assert "not found" in result["error"].lower()


def test_zip_bad_archive(tmp_path):
    z = tmp_path / "bad.zip"
    z.write_bytes(b"this is not a zip file")
    result = validate_zip_file(z)
    assert result["valid"] is False


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------


def test_csv_valid_with_required_columns(tmp_path):
    from floodroute.acquisition import _CSV_REQUIRED

    required = _CSV_REQUIRED["evacuation_shelters"]
    f = tmp_path / "shelters.csv"
    f.write_text(",".join(required) + "\n")

    result = validate_csv_file(f, category="evacuation_shelters")
    assert result["valid"] is True
    assert result["missing_columns"] == []
    assert result["row_count"] == 0  # header only


def test_csv_missing_required_columns(tmp_path):
    f = tmp_path / "incomplete.csv"
    f.write_text("shelter_id,official_name\n")

    result = validate_csv_file(f, category="evacuation_shelters")
    assert result["valid"] is False
    assert len(result["missing_columns"]) > 0


def test_csv_missing_file(tmp_path):
    result = validate_csv_file(tmp_path / "missing.csv")
    assert result["valid"] is False


def test_csv_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_bytes(b"")
    result = validate_csv_file(f)
    assert result["valid"] is False
