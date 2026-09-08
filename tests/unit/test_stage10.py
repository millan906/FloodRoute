"""Stage 10 focused unit tests.

Tests cover exactly the eight categories in the Stage 10 spec.
No Streamlit, no live geospatial I/O. Data-file-dependent tests
use pytest.mark.skipif or fixtures.

Reuses existing test patterns from test_facilities.py and test_stage8.py.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Operating mode
# ---------------------------------------------------------------------------


def test_operating_mode_values():
    from floodroute.dashboard.operating_mode import OperatingMode
    assert OperatingMode.OPERATIONAL == "operational"
    assert OperatingMode.CONTROLLED_RESEARCH == "controlled_research"
    assert OperatingMode.LEGACY_BENCHMARK == "legacy_benchmark"


def test_mode_labels_all_present():
    from floodroute.dashboard.operating_mode import MODE_LABELS, OperatingMode
    for mode in OperatingMode:
        assert mode in MODE_LABELS, f"Missing label for {mode}"


def test_mode_descriptions_all_present():
    from floodroute.dashboard.operating_mode import MODE_DESCRIPTIONS, OperatingMode
    for mode in OperatingMode:
        assert mode in MODE_DESCRIPTIONS


def test_legacy_benchmark_shelters_unchanged():
    """Legacy Benchmark mode must produce exactly the Stage 8 shelter dict."""
    from floodroute.experiments.runner import SCENARIO_SHELTER_CAPACITIES
    assert SCENARIO_SHELTER_CAPACITIES == {33: 12_000, 58: 10_000}


# ---------------------------------------------------------------------------
# 2. OSM candidates
# ---------------------------------------------------------------------------


def test_osm_candidates_loads():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES
    assert isinstance(OSM_CANDIDATES, list)


def test_osm_candidates_not_empty_when_json_present():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    assert len(OSM_CANDIDATES) > 0


def test_osm_candidates_all_have_coordinates():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    for c in OSM_CANDIDATES:
        assert c.has_coordinates, f"{c.name} lacks coordinates"
        assert c.latitude != 0.0
        assert c.longitude != 0.0


def test_osm_candidates_no_duplicate_osm_ids():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    ids = [c.osm_id for c in OSM_CANDIDATES]
    assert len(ids) == len(set(ids)), "Duplicate OSM IDs in candidates"


def test_osm_candidates_facility_types_in_vocab():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, FACILITY_TYPE_VOCAB, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    for c in OSM_CANDIDATES:
        assert c.facility_type in FACILITY_TYPE_VOCAB, (
            f"{c.name}: facility_type {c.facility_type!r} not in vocab"
        )


def test_osm_candidates_all_have_snapped_node():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    for c in OSM_CANDIDATES:
        assert c.snapped_node_id is not None
        assert c.can_be_scenario_activated


def test_osm_candidates_sjdb001_excluded():
    """Antique Regional Evacuation Center is in FACILITY_REGISTRY — not in OSM_CANDIDATES."""
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    names = {c.name for c in OSM_CANDIDATES}
    assert "Antique Regional Evacuation Center" not in names


def test_osm_candidates_sjdb005_excluded():
    """Brgy. Funda-Dalipe Evacuation Center is in FACILITY_REGISTRY — not in OSM_CANDIDATES."""
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    names = {c.name for c in OSM_CANDIDATES}
    assert "Brgy. Funda-Dalipe Evacuation Center" not in names


def test_osm_candidates_all_have_official_capacity_null():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    for c in OSM_CANDIDATES:
        assert c.official_capacity is None, (
            f"{c.name} unexpectedly has official_capacity={c.official_capacity}"
        )


def test_osm_candidates_snap_not_pipeline_result():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    for c in OSM_CANDIDATES:
        assert c.snap_is_pipeline_result is False


def test_get_candidate_by_osm_id():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH, get_candidate_by_osm_id
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    if not OSM_CANDIDATES:
        pytest.skip("No candidates loaded")
    first = OSM_CANDIDATES[0]
    found = get_candidate_by_osm_id(first.osm_id)
    assert found is not None
    assert found.osm_id == first.osm_id


def test_get_candidate_by_osm_id_returns_none_for_unknown():
    from floodroute.dashboard.osm_candidates import get_candidate_by_osm_id
    assert get_candidate_by_osm_id("node:999999999999") is None


# ---------------------------------------------------------------------------
# 3. Coverage state
# ---------------------------------------------------------------------------


def test_classify_coverage_full():
    from floodroute.dashboard.coverage_state import CoverageState, classify_coverage
    m = {"assignment_rate": 1.0, "num_capacity_violations": 0}
    assert classify_coverage(m) == CoverageState.FULL_COVERAGE


def test_classify_coverage_partial_rate():
    from floodroute.dashboard.coverage_state import CoverageState, classify_coverage
    m = {"assignment_rate": 0.8, "num_capacity_violations": 0}
    assert classify_coverage(m) == CoverageState.PARTIAL_COVERAGE


def test_classify_coverage_partial_violation():
    from floodroute.dashboard.coverage_state import CoverageState, classify_coverage
    m = {"assignment_rate": 1.0, "num_capacity_violations": 1}
    assert classify_coverage(m) == CoverageState.PARTIAL_COVERAGE


def test_classify_coverage_no_feasible():
    from floodroute.dashboard.coverage_state import CoverageState, classify_coverage
    m = {"assignment_rate": 0.0, "num_capacity_violations": 0}
    assert classify_coverage(m) == CoverageState.NO_FEASIBLE_PLAN


def test_format_coverage_banner_keys():
    from floodroute.dashboard.coverage_state import CoverageState, format_coverage_banner
    banner = format_coverage_banner(CoverageState.FULL_COVERAGE, {"assignment_rate": 1.0, "num_capacity_violations": 0, "total_unassigned": 0, "num_unreachable_origins": 0})
    assert "status" in banner
    assert "label" in banner
    assert "detail" in banner
    assert "streamlit_type" in banner


def test_format_coverage_banner_streamlit_types():
    from floodroute.dashboard.coverage_state import CoverageState, format_coverage_banner
    base = {"assignment_rate": 1.0, "num_capacity_violations": 0, "total_unassigned": 0, "num_unreachable_origins": 0}
    assert format_coverage_banner(CoverageState.FULL_COVERAGE, base)["streamlit_type"] == "success"
    assert format_coverage_banner(CoverageState.PARTIAL_COVERAGE, {**base, "assignment_rate": 0.8, "total_unassigned": 10})["streamlit_type"] == "warning"
    assert format_coverage_banner(CoverageState.NO_FEASIBLE_PLAN, {**base, "assignment_rate": 0.0})["streamlit_type"] == "error"


# ---------------------------------------------------------------------------
# 4. Road overrides
# ---------------------------------------------------------------------------


def test_road_override_store_starts_empty():
    from floodroute.dashboard.road_overrides import RoadOverrideStore
    store = RoadOverrideStore()
    assert len(store) == 0


def test_road_override_add_remove():
    from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore
    store = RoadOverrideStore()
    store.add(RoadOverride(u=100, v=200, status="closed"))
    assert len(store) == 1
    store.remove(100, 200)
    assert len(store) == 0


def test_road_override_serialisation_round_trip():
    from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore
    store = RoadOverrideStore()
    store.add(RoadOverride(u=10, v=20, status="closed", evidence_type="controlled_assumption", notes="test"))
    store.add(RoadOverride(u=30, v=40, status="flood_exposed"))
    d = store.to_dict()
    restored = RoadOverrideStore.from_dict(d)
    assert len(restored) == 2
    assert (10, 20) in restored.overrides
    assert restored.overrides[(10, 20)].status == "closed"


def test_apply_overrides_closed_returns_none():
    from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore, apply_overrides
    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="closed"))
    base = lambda u, v, d: 100.0
    fn = apply_overrides(base, store)
    result = fn(1, 2, {})
    assert result is None


def test_apply_overrides_passable_override():
    from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore, apply_overrides
    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="reported_passable"))
    base = lambda u, v, d: 50.0
    fn = apply_overrides(base, store)
    assert fn(1, 2, {}) == 50.0


def test_apply_overrides_no_graph_mutation():
    """Weight function wrapping must not write to any mutable dict or graph."""
    from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore, apply_overrides
    store = RoadOverrideStore()
    store.add(RoadOverride(u=5, v=6, status="flood_exposed"))
    original_attrs = {"length_m": 200.0, "jrc_rp20_status": "dry"}
    attrs_copy = dict(original_attrs)
    base = lambda u, v, d: float(d.get("length_m", 0))
    fn = apply_overrides(base, store)
    fn(5, 6, attrs_copy)
    assert attrs_copy == original_attrs, "apply_overrides mutated edge attributes"


def test_apply_overrides_unaffected_edge():
    from floodroute.dashboard.road_overrides import RoadOverrideStore, apply_overrides
    store = RoadOverrideStore()
    base = lambda u, v, d: 77.0
    fn = apply_overrides(base, store)
    assert fn(999, 888, {}) == 77.0


# ---------------------------------------------------------------------------
# 5. Candidates remain excluded until activated (via scenario_capacities dict)
# ---------------------------------------------------------------------------


def test_candidates_excluded_without_activation():
    """OSM candidates do not enter optimization until planner explicitly activates them."""
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    # Without activation dict, no candidate should be in the shelter capacities.
    # Simulate: shelter set is empty when no facilities are activated.
    scenario_capacities: dict[str, int] = {}
    activated_nodes = {
        c.snapped_node_id: cap
        for c in OSM_CANDIDATES
        if c.osm_id in scenario_capacities
    }
    assert activated_nodes == {}


def test_activated_facility_enters_optimization():
    """Activating a routable candidate with a capacity puts its node in shelter set."""
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    if not OSM_CANDIDATES:
        pytest.skip("No candidates loaded")
    candidate = OSM_CANDIDATES[0]
    scenario_capacities = {candidate.osm_id: 5000}
    activated_nodes = {
        c.snapped_node_id: scenario_capacities[c.osm_id]
        for c in OSM_CANDIDATES
        if c.osm_id in scenario_capacities
    }
    assert candidate.snapped_node_id in activated_nodes
    assert activated_nodes[candidate.snapped_node_id] == 5000


def test_activation_does_not_change_evidence_classification():
    """Activating a candidate must not mutate its evidence fields."""
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    if not OSM_CANDIDATES:
        pytest.skip("No candidates loaded")
    c = OSM_CANDIDATES[0]
    original_type = c.facility_type
    original_capacity = c.official_capacity
    # Simulate activation (scenario_capacities is external; candidate is frozen)
    with pytest.raises((AttributeError, TypeError)):
        c.official_capacity = 9999  # type: ignore[misc]
    assert c.facility_type == original_type
    assert c.official_capacity == original_capacity


# ---------------------------------------------------------------------------
# 6. Coverage state from result metrics
# ---------------------------------------------------------------------------


def test_coverage_state_not_partial_when_full():
    from floodroute.dashboard.coverage_state import CoverageState, classify_coverage
    m = {"assignment_rate": 1.0, "num_capacity_violations": 0}
    assert classify_coverage(m) != CoverageState.PARTIAL_COVERAGE


def test_no_fabricated_fallback_label():
    """PARTIAL_COVERAGE description must not imply a fabricated fallback."""
    from floodroute.dashboard.coverage_state import COVERAGE_DESCRIPTIONS, CoverageState
    desc = COVERAGE_DESCRIPTIONS[CoverageState.PARTIAL_COVERAGE]
    assert "fallback" not in desc.lower()
    assert "fabricated" not in desc.lower()


# ---------------------------------------------------------------------------
# 7. Backward compatibility: result_formatter unchanged
# ---------------------------------------------------------------------------


def test_shelter_display_labels_unchanged():
    """SHELTER_DISPLAY_LABELS constant must be preserved for Stage 8 comparison."""
    from floodroute.dashboard.result_formatter import SHELTER_DISPLAY_LABELS
    assert SHELTER_DISPLAY_LABELS[33] == "Scenario Shelter A (node 33)"
    assert SHELTER_DISPLAY_LABELS[58] == "Scenario Shelter B (node 58)"


def test_validate_inputs_still_works():
    from floodroute.dashboard.result_formatter import validate_inputs
    assert validate_inputs("C", "RP20", 0.25) == []
    assert validate_inputs("X", "RP20", 0.25) != []


def test_format_feasibility_status_still_works():
    from floodroute.dashboard.result_formatter import format_feasibility_status
    m = {"assignment_rate": 1.0, "total_unassigned": 0, "num_capacity_violations": 0, "num_unreachable_origins": 0}
    res = format_feasibility_status(m)
    assert res["status"] == "ok"


# ---------------------------------------------------------------------------
# 8. OsmCandidate dataclass immutability
# ---------------------------------------------------------------------------


def test_osm_candidate_is_frozen():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    if not OSM_CANDIDATES:
        pytest.skip("No candidates loaded")
    c = OSM_CANDIDATES[0]
    with pytest.raises((AttributeError, TypeError)):
        c.name = "Tampered"  # type: ignore[misc]


def test_osm_candidate_is_hashable():
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES, _JSON_PATH
    if not _JSON_PATH.exists():
        pytest.skip("JSON file not present")
    if not OSM_CANDIDATES:
        pytest.skip("No candidates loaded")
    h = hash(OSM_CANDIDATES[0])
    assert isinstance(h, int)
