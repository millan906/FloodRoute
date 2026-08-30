"""Unit tests for the potential facilities registry.

Tests cover:
  - Registry completeness (exactly 4 records)
  - SJDB-001 field values against CSV source
  - Coordinate absence for SJDB-002/003/004
  - ``eligible_for_optimization`` invariants (all False)
  - ``can_be_scenario_activated`` (True only for SJDB-001)
  - FacilityRecord immutability (frozen dataclass)
  - ``get_facility_by_id`` lookup
  - ``snap_is_pipeline_result`` for SJDB-001
  - Source provenance completeness
"""

from __future__ import annotations

import pytest

from floodroute.dashboard.facilities import (
    DESIGNATION_TYPE_LABELS,
    EVIDENCE_TIER_LABELS,
    FACILITY_REGISTRY,
    FACILITY_TYPE_LABELS,
    FacilityRecord,
    get_facility_by_id,
)

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_has_exactly_four_records():
    assert len(FACILITY_REGISTRY) == 4


def test_all_facility_ids_unique():
    ids = [f.facility_id for f in FACILITY_REGISTRY]
    assert len(ids) == len(set(ids))


def test_expected_facility_ids_present():
    ids = {f.facility_id for f in FACILITY_REGISTRY}
    assert ids == {"SJDB-001", "SJDB-002", "SJDB-003", "SJDB-004"}


# ---------------------------------------------------------------------------
# SJDB-001 field values (cross-checked against CSV source)
# ---------------------------------------------------------------------------


@pytest.fixture
def sjdb001() -> FacilityRecord:
    fac = get_facility_by_id("SJDB-001")
    assert fac is not None
    return fac


def test_sjdb001_name(sjdb001):
    assert sjdb001.facility_name == "Antique Regional Evacuation Center"


def test_sjdb001_type(sjdb001):
    assert sjdb001.facility_type == "evacuation_center"


def test_sjdb001_barangay(sjdb001):
    assert sjdb001.barangay_name == "San Pedro"


def test_sjdb001_coordinates_present(sjdb001):
    assert sjdb001.has_coordinates is True
    assert sjdb001.latitude == pytest.approx(10.800394, abs=1e-6)
    assert sjdb001.longitude == pytest.approx(121.948801, abs=1e-6)


def test_sjdb001_entrance_present(sjdb001):
    assert sjdb001.has_entrance is True
    assert sjdb001.entrance_lat == pytest.approx(10.800406, abs=1e-6)
    assert sjdb001.entrance_lon == pytest.approx(121.948580, abs=1e-6)
    assert sjdb001.entrance_status == "present"


def test_sjdb001_designation(sjdb001):
    assert sjdb001.designation_type == "permanent"


def test_sjdb001_operational_status(sjdb001):
    assert sjdb001.operational_status == "unknown"


def test_sjdb001_evidence_tier(sjdb001):
    assert sjdb001.evidence_tier == "B"


def test_sjdb001_snap_node(sjdb001):
    assert sjdb001.snapped_node_id == 1345
    assert sjdb001.snap_distance_m == pytest.approx(6.3, abs=0.01)


def test_sjdb001_snap_is_not_pipeline_result(sjdb001):
    """SJDB-001 snapped node is a research reference; Stage 6B skipped it."""
    assert sjdb001.snap_is_pipeline_result is False


def test_sjdb001_official_capacity_null(sjdb001):
    assert sjdb001.official_capacity is None


def test_sjdb001_source_url(sjdb001):
    assert sjdb001.source_url == "https://www.pna.gov.ph/articles/1065266"


def test_sjdb001_source_date(sjdb001):
    assert sjdb001.source_date == "2019-03-21"


def test_sjdb001_issuing_office(sjdb001):
    assert sjdb001.issuing_office is not None
    assert "OCD" in sjdb001.issuing_office


# ---------------------------------------------------------------------------
# SJDB-002 — no coordinates
# ---------------------------------------------------------------------------


@pytest.fixture
def sjdb002() -> FacilityRecord:
    fac = get_facility_by_id("SJDB-002")
    assert fac is not None
    return fac


def test_sjdb002_no_coordinates(sjdb002):
    assert sjdb002.has_coordinates is False
    assert sjdb002.latitude is None
    assert sjdb002.longitude is None


def test_sjdb002_no_entrance(sjdb002):
    assert sjdb002.has_entrance is False
    assert sjdb002.entrance_lat is None
    assert sjdb002.entrance_lon is None
    assert sjdb002.entrance_status == "missing"


def test_sjdb002_designation(sjdb002):
    assert sjdb002.designation_type == "historically_activated"


def test_sjdb002_evidence_tier(sjdb002):
    assert sjdb002.evidence_tier == "C"


# ---------------------------------------------------------------------------
# SJDB-003 — candidate only, no coordinates
# ---------------------------------------------------------------------------


@pytest.fixture
def sjdb003() -> FacilityRecord:
    fac = get_facility_by_id("SJDB-003")
    assert fac is not None
    return fac


def test_sjdb003_no_coordinates(sjdb003):
    assert sjdb003.has_coordinates is False


def test_sjdb003_designation(sjdb003):
    assert sjdb003.designation_type == "candidate_only"


def test_sjdb003_operational_status(sjdb003):
    assert sjdb003.operational_status == "project_only"


# ---------------------------------------------------------------------------
# SJDB-004 — candidate only, no coordinates
# ---------------------------------------------------------------------------


@pytest.fixture
def sjdb004() -> FacilityRecord:
    fac = get_facility_by_id("SJDB-004")
    assert fac is not None
    return fac


def test_sjdb004_no_coordinates(sjdb004):
    assert sjdb004.has_coordinates is False


def test_sjdb004_designation(sjdb004):
    assert sjdb004.designation_type == "candidate_only"


# ---------------------------------------------------------------------------
# eligible_for_optimization — always False
# ---------------------------------------------------------------------------


def test_all_facilities_ineligible_for_optimization():
    """All four registry records must be ineligible for optimization."""
    for fac in FACILITY_REGISTRY:
        assert fac.eligible_for_optimization is False, (
            f"{fac.facility_id} unexpectedly eligible"
        )


def test_candidate_only_never_eligible():
    """candidate_only designation must never make a facility eligible."""
    candidates = [f for f in FACILITY_REGISTRY if f.designation_type == "candidate_only"]
    assert len(candidates) >= 1  # at least SJDB-003, SJDB-004
    for fac in candidates:
        assert fac.eligible_for_optimization is False


def test_unknown_operational_status_not_eligible():
    """unknown operational_status must prevent eligibility."""
    unknown = [f for f in FACILITY_REGISTRY if f.operational_status == "unknown"]
    assert len(unknown) >= 1
    for fac in unknown:
        assert fac.eligible_for_optimization is False


# ---------------------------------------------------------------------------
# can_be_scenario_activated
# ---------------------------------------------------------------------------


def test_sjdb001_can_be_scenario_activated(sjdb001):
    """SJDB-001 has entrance coords and snapped_node_id — can be activated."""
    assert sjdb001.can_be_scenario_activated is True


def test_sjdb002_cannot_be_scenario_activated(sjdb002):
    """SJDB-002 has no entrance — cannot be activated."""
    assert sjdb002.can_be_scenario_activated is False


def test_sjdb003_cannot_be_scenario_activated(sjdb003):
    assert sjdb003.can_be_scenario_activated is False


def test_sjdb004_cannot_be_scenario_activated(sjdb004):
    assert sjdb004.can_be_scenario_activated is False


def test_only_sjdb001_can_be_activated():
    activatable = [f for f in FACILITY_REGISTRY if f.can_be_scenario_activated]
    assert [f.facility_id for f in activatable] == ["SJDB-001"]


# ---------------------------------------------------------------------------
# FacilityRecord immutability
# ---------------------------------------------------------------------------


def test_facility_record_is_frozen(sjdb001):
    with pytest.raises((AttributeError, TypeError)):
        sjdb001.facility_name = "Tampered Name"  # type: ignore[misc]


def test_facility_record_is_hashable(sjdb001):
    """Frozen dataclasses are hashable."""
    h = hash(sjdb001)
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# get_facility_by_id
# ---------------------------------------------------------------------------


def test_get_facility_by_id_returns_correct_record():
    fac = get_facility_by_id("SJDB-002")
    assert fac is not None
    assert fac.facility_name == "Bariri Covered Court"


def test_get_facility_by_id_returns_none_for_unknown():
    assert get_facility_by_id("SJDB-999") is None


def test_get_facility_by_id_all_ids():
    for expected_id in ("SJDB-001", "SJDB-002", "SJDB-003", "SJDB-004"):
        fac = get_facility_by_id(expected_id)
        assert fac is not None
        assert fac.facility_id == expected_id


# ---------------------------------------------------------------------------
# Source provenance completeness
# ---------------------------------------------------------------------------


def test_all_records_have_source_title():
    for fac in FACILITY_REGISTRY:
        assert fac.source_title, f"{fac.facility_id} missing source_title"


def test_all_records_have_source_date():
    for fac in FACILITY_REGISTRY:
        assert fac.source_date, f"{fac.facility_id} missing source_date"


def test_all_records_have_designation_source():
    for fac in FACILITY_REGISTRY:
        assert fac.designation_source, f"{fac.facility_id} missing designation_source"


# ---------------------------------------------------------------------------
# Label map coverage
# ---------------------------------------------------------------------------


def test_all_facility_types_in_label_map():
    for fac in FACILITY_REGISTRY:
        assert fac.facility_type in FACILITY_TYPE_LABELS, (
            f"{fac.facility_id} facility_type {fac.facility_type!r} not in FACILITY_TYPE_LABELS"
        )


def test_all_designation_types_in_label_map():
    for fac in FACILITY_REGISTRY:
        assert fac.designation_type in DESIGNATION_TYPE_LABELS, (
            f"{fac.facility_id} designation_type {fac.designation_type!r} "
            "not in DESIGNATION_TYPE_LABELS"
        )


def test_all_evidence_tiers_in_label_map():
    for fac in FACILITY_REGISTRY:
        assert fac.evidence_tier in EVIDENCE_TIER_LABELS, (
            f"{fac.facility_id} evidence_tier {fac.evidence_tier!r} not in EVIDENCE_TIER_LABELS"
        )
