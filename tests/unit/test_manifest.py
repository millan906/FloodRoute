"""Unit tests for the DatasetManifest model and loaders.

All fixtures are synthetic toy data in tests/fixtures/.
No real case-study data is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Valid manifest loading
# ---------------------------------------------------------------------------


def test_valid_manifest_loads():
    from floodroute.manifest import load_manifest

    m = load_manifest(FIXTURES / "manifest_valid.yaml")
    assert m.schema_version == "1.0"
    assert m.dataset_id == "test_valid_dataset"
    assert m.category.value == "road_network"
    assert m.access_method.value == "direct_download"
    assert m.acquisition_status.value == "not_requested"
    assert m.provenance == "observed"
    assert m.validation_status.value == "not_validated"
    # Null fields preserved — not replaced by defaults
    assert m.sha256 is None
    assert m.source_url is None


def test_load_all_manifests_returns_dict():
    from floodroute.manifest import load_all_manifests

    # Load from the real manifests directory
    manifests_dir = Path(__file__).parent.parent.parent / "data" / "manifests"
    if not manifests_dir.is_dir():
        pytest.skip("data/manifests/ directory not found")

    result = load_all_manifests(manifests_dir)
    assert isinstance(result, dict)
    assert len(result) >= 1
    for dataset_id, manifest in result.items():
        assert manifest.dataset_id == dataset_id


def test_load_all_manifests_empty_dir(tmp_path):
    from floodroute.manifest import load_all_manifests

    result = load_all_manifests(tmp_path)
    assert result == {}


def test_load_all_manifests_nonexistent_dir(tmp_path):
    from floodroute.manifest import load_all_manifests

    result = load_all_manifests(tmp_path / "does_not_exist")
    assert result == {}


# ---------------------------------------------------------------------------
# Invalid category / status / access_method / provenance rejection
# ---------------------------------------------------------------------------


def test_invalid_category_rejected():
    from floodroute.manifest import load_manifest

    with pytest.raises(ValidationError) as exc_info:
        load_manifest(FIXTURES / "manifest_bad_category.yaml")
    errors = exc_info.value.errors()
    assert any("category" in str(e.get("loc", "")) for e in errors)


def test_invalid_acquisition_status_rejected():
    from floodroute.manifest import load_manifest

    with pytest.raises(ValidationError) as exc_info:
        load_manifest(FIXTURES / "manifest_bad_acq_status.yaml")
    errors = exc_info.value.errors()
    assert any("acquisition_status" in str(e.get("loc", "")) for e in errors)


def test_invalid_provenance_rejected():
    from floodroute.manifest import load_manifest

    with pytest.raises(ValidationError) as exc_info:
        load_manifest(FIXTURES / "manifest_bad_provenance.yaml")
    errors = exc_info.value.errors()
    assert any("provenance" in str(e.get("loc", "")) for e in errors)


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------


def test_missing_dataset_id_rejected():
    from floodroute.manifest import load_manifest

    with pytest.raises(ValidationError) as exc_info:
        load_manifest(FIXTURES / "manifest_missing_required.yaml")
    errors = exc_info.value.errors()
    assert any("dataset_id" in str(e.get("loc", "")) for e in errors)


# ---------------------------------------------------------------------------
# Duplicate dataset IDs
# ---------------------------------------------------------------------------


def test_duplicate_dataset_ids_rejected(tmp_path):
    from floodroute.manifest import load_all_manifests

    content = (FIXTURES / "manifest_valid.yaml").read_text()
    (tmp_path / "a.yaml").write_text(content)
    (tmp_path / "b.yaml").write_text(content)  # same dataset_id

    with pytest.raises(ValueError, match="Duplicate"):
        load_all_manifests(tmp_path)
