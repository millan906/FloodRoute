"""Unit tests for the data-readiness report.

No network calls, no real case-study data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from floodroute.manifest import DatasetManifest
from floodroute.readiness import (
    build_readiness_report,
    format_report_json,
    format_report_text,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
TEMPLATES = Path(__file__).parent.parent.parent / "data" / "templates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(**kwargs) -> DatasetManifest:
    defaults = dict(
        schema_version="1.0",
        dataset_id="test_ds",
        title="Test Dataset",
        category="road_network",
        access_method="direct_download",
        acquisition_status="not_requested",
        provenance="observed",
        validation_status="not_validated",
    )
    defaults.update(kwargs)
    return DatasetManifest.model_validate(defaults)


# ---------------------------------------------------------------------------
# Readiness report with missing datasets
# ---------------------------------------------------------------------------


def test_readiness_report_all_missing(tmp_path):
    manifests = {
        "ds_a": _make_manifest(dataset_id="ds_a", local_path="raw/a.csv"),
        "ds_b": _make_manifest(dataset_id="ds_b", local_path="raw/b.csv"),
    }
    entries = build_readiness_report(manifests, tmp_path)
    assert len(entries) == 2
    for e in entries:
        assert e.file_exists is False
        assert e.checksum_matches is None


def test_readiness_report_file_present(tmp_path):
    data = b"test data"
    f = tmp_path / "raw" / "present.csv"
    f.parent.mkdir(parents=True)
    f.write_bytes(data)

    manifests = {
        "ds_present": _make_manifest(dataset_id="ds_present", local_path="raw/present.csv"),
    }
    entries = build_readiness_report(manifests, tmp_path)
    assert entries[0].file_exists is True
    assert entries[0].checksum_matches is None  # no sha256 in manifest


def test_readiness_report_checksum_ok(tmp_path):
    data = b"verified data"
    sha = hashlib.sha256(data).hexdigest()
    f = tmp_path / "raw" / "verified.csv"
    f.parent.mkdir(parents=True)
    f.write_bytes(data)

    manifests = {
        "ds_ck": _make_manifest(dataset_id="ds_ck", local_path="raw/verified.csv", sha256=sha),
    }
    entries = build_readiness_report(manifests, tmp_path)
    assert entries[0].file_exists is True
    assert entries[0].checksum_matches is True


def test_readiness_report_checksum_mismatch(tmp_path):
    f = tmp_path / "raw" / "bad.csv"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"actual data")

    manifests = {
        "ds_bad": _make_manifest(dataset_id="ds_bad", local_path="raw/bad.csv", sha256="0" * 64),
    }
    entries = build_readiness_report(manifests, tmp_path)
    assert entries[0].checksum_matches is False
    assert entries[0].blocking_issue is not None
    assert "mismatch" in entries[0].blocking_issue.lower()


def test_readiness_report_blocking_unavailable(tmp_path):
    manifests = {
        "ds_gone": _make_manifest(
            dataset_id="ds_gone",
            acquisition_status="unavailable",
        ),
    }
    entries = build_readiness_report(manifests, tmp_path)
    assert entries[0].blocking_issue is not None
    assert "unavailable" in entries[0].blocking_issue.lower()


def test_readiness_report_no_blocking_not_requested(tmp_path):
    manifests = {
        "ds_pending": _make_manifest(
            dataset_id="ds_pending",
            acquisition_status="not_requested",
            local_path="raw/ds_pending.csv",
        ),
    }
    entries = build_readiness_report(manifests, tmp_path)
    # File doesn't exist, but acquisition_status is not_requested — not a blocker
    assert entries[0].blocking_issue is None


def test_readiness_report_empty_manifests(tmp_path):
    entries = build_readiness_report({}, tmp_path)
    assert entries == []


def test_format_report_text_contains_dataset_id(tmp_path):
    manifests = {"my_ds": _make_manifest(dataset_id="my_ds")}
    entries = build_readiness_report(manifests, tmp_path)
    text = format_report_text(entries)
    assert "my_ds" in text


def test_format_report_json_is_valid(tmp_path):
    import json

    manifests = {"my_ds": _make_manifest(dataset_id="my_ds")}
    entries = build_readiness_report(manifests, tmp_path)
    parsed = json.loads(format_report_json(entries))
    assert isinstance(parsed, list)
    assert parsed[0]["dataset_id"] == "my_ds"


def test_format_report_text_no_manifests():
    text = format_report_text([])
    assert "no" in text.lower() or len(text) > 0


# ---------------------------------------------------------------------------
# Template CSV — required columns
# ---------------------------------------------------------------------------


def test_shelter_template_has_required_columns():
    from floodroute.acquisition import validate_csv_file

    if not TEMPLATES.is_dir():
        pytest.skip("data/templates/ directory not found")

    template = TEMPLATES / "evacuation_shelters_template.csv"
    if not template.exists():
        pytest.skip("evacuation_shelters_template.csv not found")

    result = validate_csv_file(template, category="evacuation_shelters")
    assert result.get("missing_columns") == [], (
        f"Template missing columns: {result.get('missing_columns')}"
    )


def test_incident_template_has_required_columns():
    from floodroute.acquisition import validate_csv_file

    if not TEMPLATES.is_dir():
        pytest.skip("data/templates/ directory not found")

    template = TEMPLATES / "historical_incidents_template.csv"
    if not template.exists():
        pytest.skip("historical_incidents_template.csv not found")

    result = validate_csv_file(template, category="historical_incidents")
    assert result.get("missing_columns") == [], (
        f"Template missing columns: {result.get('missing_columns')}"
    )
