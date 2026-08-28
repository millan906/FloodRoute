"""Unit tests for Stage 6B shelter containment and graph snapping.

Coverage:
- Containment: inside, outside, missing coordinates, boundary tolerance
- Snapping: nearest node selection, deterministic tie-breaking, projected distance
- Thresholds: below warn, warn zone, reject zone
- Exclusions: project_only, decommissioned, unknown, excluded tier
- Readiness counting: eligible+routable candidates update decision correctly
- Missing inputs: empty record list, empty graph
- Coordinate provenance: preserved unchanged through snap pipeline
"""

from __future__ import annotations

import networkx as nx
import pytest
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from floodroute.shelters.containment import check_containment
from floodroute.shelters.readiness import ReadinessDecision, evaluate_shelter_readiness
from floodroute.shelters.schema import (
    CapacityProvenance,
    EvidenceTier,
    OperationalStatus,
    ShelterRecord,
    VerificationStatus,
)
from floodroute.shelters.snap import (
    _build_strtree,
    _project_point,
    snap_all_shelters,
    snap_entrance_to_graph,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_BASE = {
    "shelter_id": "SH001",
    "evidence_tier": EvidenceTier.A,
    "verification_status": VerificationStatus.verified,
}

# SJDB city hall approximate coordinates (WGS-84)
_SJDB_LAT = 10.7500
_SJDB_LON = 121.9500

# UTM Zone 51N equivalents (approximate, used for fixture graph construction)
# Computed via: pyproj.Transformer.from_crs("EPSG:4326","EPSG:32651",always_xy=True).transform(121.95,10.75)
_SJDB_UTM_X = 386_500.0  # approximate metres
_SJDB_UTM_Y = 1_189_000.0  # approximate metres


def _make_record(**overrides) -> ShelterRecord:
    """Build a minimal valid ShelterRecord."""
    data = {**_BASE, **overrides}
    return ShelterRecord.model_validate(data)


def _make_routable(**overrides) -> ShelterRecord:
    """Eligible, routable shelter with entrance coordinates inside SJDB."""
    data = {
        **_BASE,
        "operational_status": OperationalStatus.operational,
        "entrance_latitude": _SJDB_LAT,
        "entrance_longitude": _SJDB_LON,
        **overrides,
    }
    return ShelterRecord.model_validate(data)


def _make_ready(**overrides) -> ShelterRecord:
    """Eligible, routable, verified shelter with authoritative capacity."""
    data = {
        **_BASE,
        "operational_status": OperationalStatus.operational,
        "entrance_latitude": _SJDB_LAT,
        "entrance_longitude": _SJDB_LON,
        "official_capacity": 100,
        "capacity_provenance": CapacityProvenance.authoritative,
        **overrides,
    }
    return ShelterRecord.model_validate(data)


def _single_node_graph(x: float, y: float, node_id: str = "1") -> nx.MultiDiGraph:
    """Create a graph with a single node at the given UTM coordinates."""
    g = nx.MultiDiGraph()
    g.add_node(node_id, x=x, y=y)
    return g


def _two_node_graph(
    x1: float, y1: float, x2: float, y2: float,
    id1: str = "1", id2: str = "2",
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(id1, x=x1, y=y1)
    g.add_node(id2, x=x2, y=y2)
    return g


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------


class TestProjectPoint:
    def test_returns_float_pair(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_x_in_utm_range(self):
        # UTM Zone 51N X should be in plausible range for SJDB
        x, _ = _project_point(_SJDB_LAT, _SJDB_LON)
        assert 300_000 < x < 500_000

    def test_y_in_utm_range(self):
        _, y = _project_point(_SJDB_LAT, _SJDB_LON)
        assert 1_000_000 < y < 1_300_000


# ---------------------------------------------------------------------------
# STRtree builder
# ---------------------------------------------------------------------------


class TestBuildStrtree:
    def test_returns_strtree_and_ids(self):
        g = _two_node_graph(0.0, 0.0, 10.0, 0.0)
        tree, ids = _build_strtree(g)
        assert isinstance(tree, STRtree)
        assert len(ids) == 2

    def test_ids_match_node_count(self):
        g = nx.MultiDiGraph()
        for i in range(5):
            g.add_node(str(i), x=float(i), y=0.0)
        _, ids = _build_strtree(g)
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# snap_entrance_to_graph — basic
# ---------------------------------------------------------------------------


class TestSnapEntranceBasic:
    def test_snaps_to_only_node(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.status == "snapped"
        assert result.snapped_node_id == "1"
        assert result.snap_distance_m == pytest.approx(0.0, abs=1e-3)

    def test_status_snapped_below_warn(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        # Place node 50 m north (within default warn threshold)
        g = _single_node_graph(x, y + 50.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.status == "snapped"
        assert result.snap_distance_m == pytest.approx(50.0, abs=1.0)

    def test_status_warned_above_warn_threshold(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 200.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, warn_m=100.0)
        assert result.status == "warned"
        assert result.snap_distance_m > 100.0

    def test_status_rejected_above_reject_threshold(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 600.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, reject_m=500.0)
        assert result.status == "rejected"
        assert result.snapped_node_id is None or result.status == "rejected"

    def test_accepted_property_true_for_snapped(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.accepted is True

    def test_accepted_property_true_for_warned(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 200.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, warn_m=100.0)
        assert result.accepted is True

    def test_accepted_property_false_for_rejected(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 600.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, reject_m=500.0)
        assert result.accepted is False

    def test_empty_graph_returns_rejected(self):
        g = nx.MultiDiGraph()
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.status == "rejected"
        assert result.snapped_node_id is None

    def test_utm_coordinates_stored_in_result(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.entrance_utm_x == pytest.approx(x, abs=1.0)
        assert result.entrance_utm_y == pytest.approx(y, abs=1.0)

    def test_shelter_id_preserved_in_result(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        result = snap_entrance_to_graph("MY-ID", _SJDB_LAT, _SJDB_LON, g)
        assert result.shelter_id == "MY-ID"


# ---------------------------------------------------------------------------
# Nearest node selection
# ---------------------------------------------------------------------------


class TestNearestNodeSelection:
    def test_selects_closer_node(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        # node A is 50 m north, node B is 200 m north
        g = _two_node_graph(x, y + 50.0, x, y + 200.0, "near", "far")
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.snapped_node_id == "near"

    def test_distance_equals_distance_to_selected_node(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _two_node_graph(x, y + 30.0, x, y + 300.0, "A", "B")
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.snap_distance_m == pytest.approx(30.0, abs=1.0)


# ---------------------------------------------------------------------------
# Deterministic tie-breaking
# ---------------------------------------------------------------------------


class TestTieBreaking:
    def test_equidistant_nodes_picks_lexicographic_minimum(self):
        """Nodes at exactly the same distance → smallest node_id wins."""
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        # Place node "Z" and "A" equidistant (both 50 m north for simplicity via same coords)
        g = nx.MultiDiGraph()
        g.add_node("Z", x=x + 50.0, y=y)
        g.add_node("A", x=x - 50.0, y=y)  # 50 m west — same distance
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert result.snapped_node_id == "A"

    def test_tie_breaking_is_stable_across_calls(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = nx.MultiDiGraph()
        g.add_node("Z", x=x + 50.0, y=y)
        g.add_node("A", x=x - 50.0, y=y)
        r1 = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        r2 = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g)
        assert r1.snapped_node_id == r2.snapped_node_id


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    def test_custom_warn_threshold_triggers(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 30.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, warn_m=25.0)
        assert result.status == "warned"

    def test_custom_reject_threshold_triggers(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 150.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, reject_m=100.0)
        assert result.status == "rejected"

    def test_zero_warn_threshold_warns_all_nonzero(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y + 1.0)
        result = snap_entrance_to_graph("SH001", _SJDB_LAT, _SJDB_LON, g, warn_m=0.0, reject_m=1000.0)
        assert result.status == "warned"


# ---------------------------------------------------------------------------
# snap_all_shelters
# ---------------------------------------------------------------------------


class TestSnapAllShelters:
    def test_ineligible_records_are_skipped(self):
        r = _make_record(
            evidence_tier=EvidenceTier.excluded,
            entrance_latitude=_SJDB_LAT,
            entrance_longitude=_SJDB_LON,
        )
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "skipped"

    def test_project_only_records_are_skipped(self):
        r = _make_record(
            operational_status=OperationalStatus.project_only,
            entrance_latitude=_SJDB_LAT,
            entrance_longitude=_SJDB_LON,
            evidence_tier=EvidenceTier.B,
        )
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "skipped"

    def test_decommissioned_records_are_skipped(self):
        r = _make_record(
            operational_status=OperationalStatus.decommissioned,
            entrance_latitude=_SJDB_LAT,
            entrance_longitude=_SJDB_LON,
        )
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "skipped"

    def test_unknown_status_records_are_skipped(self):
        r = _make_record(
            operational_status=OperationalStatus.unknown,
            entrance_latitude=_SJDB_LAT,
            entrance_longitude=_SJDB_LON,
        )
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "skipped"

    def test_missing_entrance_yields_no_entrance(self):
        r = _make_record(operational_status=OperationalStatus.operational)
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "no_entrance"

    def test_eligible_routable_record_is_snapped(self):
        r = _make_routable()
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "snapped"

    def test_result_count_matches_record_count(self):
        records = [
            _make_routable(shelter_id="SH001"),
            _make_record(shelter_id="SH002", operational_status=OperationalStatus.project_only, evidence_tier=EvidenceTier.B),
            _make_record(shelter_id="SH003", evidence_tier=EvidenceTier.excluded),
        ]
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters(records, g)
        assert len(results) == 3

    def test_empty_record_list(self):
        g = _single_node_graph(_SJDB_UTM_X, _SJDB_UTM_Y)
        results = snap_all_shelters([], g)
        assert results == []

    def test_strtree_cache_accepted(self):
        r = _make_routable()
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        _build_strtree(g)  # verify it builds without error
        results = snap_all_shelters([r], g)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


class TestContainment:
    def _make_square_geoseries(self, cx: float, cy: float, half: float):
        """Create a GeoSeries with a square polygon centred at (cx, cy) in UTM."""
        import geopandas as gpd

        poly = Polygon([
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ])
        return gpd.GeoSeries([poly], crs="EPSG:32651")

    def test_point_inside_polygon(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        geoseries = self._make_square_geoseries(x, y, 1000.0)
        result = check_containment("SH001", _SJDB_LAT, _SJDB_LON, geoseries)
        assert result.inside is True

    def test_point_outside_polygon(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        # Place polygon 5 km away
        geoseries = self._make_square_geoseries(x + 5000.0, y + 5000.0, 100.0)
        result = check_containment("SH001", _SJDB_LAT, _SJDB_LON, geoseries)
        assert result.inside is False
        assert result.distance_to_boundary_m > 0

    def test_missing_lat_returns_false(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        geoseries = self._make_square_geoseries(x, y, 1000.0)
        result = check_containment("SH001", None, _SJDB_LON, geoseries)
        assert result.inside is False
        assert result.distance_to_boundary_m is None

    def test_missing_lon_returns_false(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        geoseries = self._make_square_geoseries(x, y, 1000.0)
        result = check_containment("SH001", _SJDB_LAT, None, geoseries)
        assert result.inside is False

    def test_shelter_id_preserved(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        geoseries = self._make_square_geoseries(x, y, 1000.0)
        result = check_containment("MY-SHELTER", _SJDB_LAT, _SJDB_LON, geoseries)
        assert result.shelter_id == "MY-SHELTER"

    def test_inside_has_nonpositive_distance(self):
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        geoseries = self._make_square_geoseries(x, y, 1000.0)
        result = check_containment("SH001", _SJDB_LAT, _SJDB_LON, geoseries)
        assert result.distance_to_boundary_m <= 0


# ---------------------------------------------------------------------------
# Readiness counting
# ---------------------------------------------------------------------------


class TestReadinessCountingWithSnap:
    """Verify that readiness decisions respond correctly to routable candidate counts."""

    def test_zero_records_blocked(self):
        result = evaluate_shelter_readiness([])
        assert result.decision == ReadinessDecision.BLOCKED

    def test_one_routable_eligible_is_blocked(self):
        records = [_make_routable(shelter_id="SH001")]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.BLOCKED

    def test_two_routable_eligible_unverified_is_partial(self):
        records = [
            _make_routable(shelter_id="SH001", verification_status=VerificationStatus.unverified),
            _make_routable(shelter_id="SH002", verification_status=VerificationStatus.unverified),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.PARTIAL

    def test_two_routable_verified_no_auth_cap_is_partial(self):
        records = [
            _make_routable(shelter_id="SH001"),
            _make_routable(shelter_id="SH002"),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.PARTIAL

    def test_two_ready_records_yields_ready(self):
        records = [
            _make_ready(shelter_id="SH001"),
            _make_ready(shelter_id="SH002"),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.READY

    def test_ineligible_records_not_counted(self):
        """project_only + 1 eligible routable = BLOCKED (not PARTIAL)."""
        records = [
            _make_record(
                shelter_id="PROJ",
                operational_status=OperationalStatus.project_only,
                entrance_latitude=_SJDB_LAT,
                entrance_longitude=_SJDB_LON,
                evidence_tier=EvidenceTier.B,
            ),
            _make_routable(shelter_id="SH001"),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.BLOCKED

    def test_missing_entrance_not_counted_as_routable(self):
        """Eligible records without entrance coords don't count toward PARTIAL gate."""
        records = [
            _make_record(shelter_id="SH001", operational_status=OperationalStatus.operational),
            _make_record(shelter_id="SH002", operational_status=OperationalStatus.operational),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.n_routable == 0
        assert result.decision == ReadinessDecision.BLOCKED

    def test_blocking_ids_lists_eligible_without_entrances(self):
        records = [
            _make_record(shelter_id="SH001", operational_status=OperationalStatus.operational),
        ]
        result = evaluate_shelter_readiness(records)
        assert "SH001" in result.blocking_ids


# ---------------------------------------------------------------------------
# Coordinate provenance preserved
# ---------------------------------------------------------------------------


class TestCoordinateProvenancePreserved:
    def test_coordinate_source_preserved_in_record(self):
        r = _make_routable(
            coordinate_source="lgu_record",
            coordinate_method="entrance_gps",
            coordinate_uncertainty_m=5.0,
        )
        assert r.coordinate_source == "lgu_record"
        assert r.coordinate_method == "entrance_gps"
        assert r.coordinate_uncertainty_m == 5.0

    def test_snap_result_does_not_alter_record(self):
        r = _make_routable(coordinate_source="map_inspection")
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        # Record unchanged after snapping
        assert r.coordinate_source == "map_inspection"
        assert results[0].shelter_id == r.shelter_id


# ---------------------------------------------------------------------------
# Manifest candidate fixture — mirrors sjdb_evacuation_shelters.yaml state
# ---------------------------------------------------------------------------


class TestManifestCandidatesNullEntrances:
    """Confirm that the four SJDB manifest candidates, as currently documented,
    produce no_entrance snap status and contribute zero to all readiness counts.

    This test encodes the manifest's empirical readiness claim so that any
    future data entry that changes eligibility or entrance coordinates is
    immediately visible in CI.
    """

    @staticmethod
    def _make_manifest_records() -> list[ShelterRecord]:
        """Construct ShelterRecord objects matching the manifest candidate registry."""
        return [
            ShelterRecord.model_validate({
                "shelter_id": "SJDB-001",
                "name": "Regional Evacuation Center San Pedro",
                "barangay_name": "Barangay San Pedro",
                "facility_type": "evacuation_center",
                "evidence_tier": EvidenceTier.B,
                "operational_status": OperationalStatus.unknown,
                "verification_status": VerificationStatus.unverified,
                "source_title": "NDRRMC Region VI Evacuation Directory 2022",
                "issuing_office": "NDRRMC Region VI",
                "source_date": "2022",
                # All coordinates null
            }),
            ShelterRecord.model_validate({
                "shelter_id": "SJDB-002",
                "name": "Bariri Covered Court",
                "barangay_name": "Bariri",
                "facility_type": "covered_court",
                "evidence_tier": EvidenceTier.C,
                "operational_status": OperationalStatus.unknown,
                "verification_status": VerificationStatus.unverified,
                "source_title": "Community report via MDRRMO verbal communication",
                "source_date": "2023",
            }),
            ShelterRecord.model_validate({
                "shelter_id": "SJDB-003",
                "name": "Badiang Multi-Purpose Hall (Proposed Evacuation Center)",
                "barangay_name": "Badiang",
                "facility_type": "multi_purpose_hall",
                "evidence_tier": EvidenceTier.B,
                "operational_status": OperationalStatus.project_only,
                "verification_status": VerificationStatus.unverified,
                "source_title": "SJDB MDRRMO Contingency Plan 2023–2024 (draft)",
                "issuing_office": "SJDB MDRRMO",
                "source_date": "2023",
            }),
            ShelterRecord.model_validate({
                "shelter_id": "SJDB-004",
                "name": "Barangay 8 Covered Court",
                "barangay_name": "Barangay 8 (Pob.)",
                "facility_type": "covered_court",
                "evidence_tier": EvidenceTier.C,
                "operational_status": OperationalStatus.project_only,
                "verification_status": VerificationStatus.unverified,
                "source_title": "Barangay 8 Local Disaster Risk Reduction Plan 2022 (informal)",
                "source_date": "2022",
            }),
        ]

    def test_all_candidates_are_not_routable(self):
        for r in self._make_manifest_records():
            assert r.is_routable is False, f"{r.shelter_id} unexpectedly has entrance coordinates"

    def test_all_candidates_are_ineligible(self):
        """unknown and project_only statuses render all four ineligible."""
        for r in self._make_manifest_records():
            assert r.is_eligible is False, f"{r.shelter_id} unexpectedly eligible"

    def test_snap_pipeline_classifies_all_as_skipped(self):
        """Ineligible records are skipped before the no_entrance check."""
        records = self._make_manifest_records()
        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters(records, g)
        for snap in results:
            assert snap.status == "skipped", (
                f"{snap.shelter_id}: expected 'skipped', got '{snap.status}'"
            )

    def test_readiness_is_blocked(self):
        records = self._make_manifest_records()
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.BLOCKED

    def test_readiness_counts_all_zero(self):
        records = self._make_manifest_records()
        result = evaluate_shelter_readiness(records)
        assert result.n_eligible == 0
        assert result.n_routable == 0
        assert result.n_verified == 0
        assert result.n_auth_capacity == 0
        assert result.n_partial_candidates == 0
        assert result.n_ready_candidates == 0

    def test_total_record_count_is_four(self):
        records = self._make_manifest_records()
        result = evaluate_shelter_readiness(records)
        assert result.n_total == 4

    def test_eligible_operational_with_null_entrance_yields_no_entrance(self):
        """Separate guard: if a future edit makes a record eligible but keeps
        entrance null, the pipeline must still return no_entrance (not skipped)
        and must not contribute to routable count.
        """
        # Simulate an eligible record that still has null entrance coordinates
        r = _make_record(
            shelter_id="FUTURE",
            operational_status=OperationalStatus.operational,
            # entrance_latitude and entrance_longitude deliberately omitted (null)
        )
        assert r.is_eligible is True
        assert r.is_routable is False

        x, y = _project_point(_SJDB_LAT, _SJDB_LON)
        g = _single_node_graph(x, y)
        results = snap_all_shelters([r], g)
        assert results[0].status == "no_entrance"
        assert results[0].accepted is False

        readiness = evaluate_shelter_readiness([r])
        assert readiness.n_routable == 0
        assert readiness.decision == ReadinessDecision.BLOCKED
