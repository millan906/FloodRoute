"""Unit tests for floodroute.dashboard.result_formatter and map_builder helpers.

All tests are pure-Python — no Streamlit, no geospatial dependencies.
"""

from __future__ import annotations

import pytest

from floodroute.dashboard.map_builder import (
    STATUS_AVAILABILITY,
    edge_color,
    edge_opacity,
    edge_weight,
    flooded_color,
)
from floodroute.dashboard.result_formatter import (
    SHELTER_DISPLAY_LABELS,
    format_barangay_card,
    format_feasibility_status,
    format_origin_assignment_status,
    format_recommendation,
    format_run_label,
    format_shelter_loads,
    validate_inputs,
)

# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


class TestValidateInputs:
    def test_valid_combination_returns_no_errors(self):
        assert validate_inputs("C", "RP20", 0.25) == []

    def test_all_valid_algorithms(self):
        for alg in ("A", "B", "C"):
            assert validate_inputs(alg, "RP10", 0.10) == []

    def test_all_valid_return_periods(self):
        for rp in ("RP10", "RP20", "RP100"):
            assert validate_inputs("A", rp, 0.50) == []

    def test_all_valid_demand_fractions(self):
        for frac in (0.10, 0.25, 0.50):
            assert validate_inputs("B", "RP100", frac) == []

    def test_invalid_algorithm_produces_error(self):
        errors = validate_inputs("D", "RP20", 0.25)
        assert len(errors) == 1
        assert "Algorithm" in errors[0]

    def test_invalid_return_period_produces_error(self):
        errors = validate_inputs("A", "RP50", 0.25)
        assert len(errors) == 1
        assert "Return period" in errors[0]

    def test_invalid_fraction_produces_error(self):
        errors = validate_inputs("A", "RP10", 0.33)
        assert len(errors) == 1
        assert "Demand fraction" in errors[0]

    def test_multiple_invalid_produces_multiple_errors(self):
        errors = validate_inputs("X", "RP999", 0.99)
        assert len(errors) == 3

    def test_empty_string_algorithm_is_invalid(self):
        errors = validate_inputs("", "RP10", 0.10)
        assert any("Algorithm" in e for e in errors)

    def test_lowercase_algorithm_is_invalid(self):
        errors = validate_inputs("c", "RP20", 0.25)
        assert any("Algorithm" in e for e in errors)


# ---------------------------------------------------------------------------
# format_feasibility_status
# ---------------------------------------------------------------------------


class TestFormatFeasibilityStatus:
    def _make_metrics(
        self,
        assignment_rate: float = 1.0,
        total_unassigned: int = 0,
        num_capacity_violations: int = 0,
        capacity_violation_shelters: str = "",
        num_unreachable_origins: int = 0,
    ) -> dict:
        return {
            "assignment_rate": assignment_rate,
            "total_unassigned": total_unassigned,
            "num_capacity_violations": num_capacity_violations,
            "capacity_violation_shelters": capacity_violation_shelters,
            "num_unreachable_origins": num_unreachable_origins,
        }

    def test_fully_assigned_no_violations_is_ok(self):
        result = format_feasibility_status(self._make_metrics())
        assert result["status"] == "ok"
        assert result["label"] == "Feasible"
        assert result["warnings"] == []

    def test_unassigned_units_produce_warning(self):
        m = self._make_metrics(assignment_rate=0.9, total_unassigned=100)
        result = format_feasibility_status(m)
        assert result["status"] == "warning"
        assert any("unassigned" in w for w in result["warnings"])

    def test_capacity_violation_produces_warning(self):
        m = self._make_metrics(
            num_capacity_violations=1, capacity_violation_shelters="33"
        )
        result = format_feasibility_status(m)
        assert result["status"] == "warning"
        assert any("33" in w for w in result["warnings"])

    def test_unreachable_origins_produce_warning(self):
        m = self._make_metrics(num_unreachable_origins=3)
        result = format_feasibility_status(m)
        assert any("disconnected" in w.lower() for w in result["warnings"])

    def test_unreachable_warning_drops_topology_gap_language(self):
        m = self._make_metrics(num_unreachable_origins=1)
        result = format_feasibility_status(m)
        combined = " ".join(result["warnings"])
        assert "OSM topology gap" not in combined

    def test_unreachable_warning_includes_field_verified_disclaimer(self):
        m = self._make_metrics(num_unreachable_origins=2)
        result = format_feasibility_status(m)
        assert any("field-verified" in w for w in result["warnings"])

    def test_unreachable_warning_includes_count(self):
        m = self._make_metrics(num_unreachable_origins=5)
        result = format_feasibility_status(m)
        assert any("5" in w for w in result["warnings"])

    def test_zero_assignment_rate_is_error(self):
        m = self._make_metrics(assignment_rate=0.0, total_unassigned=500)
        result = format_feasibility_status(m)
        assert result["status"] == "error"

    def test_multiple_warnings_all_present(self):
        m = self._make_metrics(
            assignment_rate=0.8,
            total_unassigned=200,
            num_capacity_violations=1,
            capacity_violation_shelters="58",
            num_unreachable_origins=2,
        )
        result = format_feasibility_status(m)
        assert len(result["warnings"]) == 3


# ---------------------------------------------------------------------------
# format_shelter_loads
# ---------------------------------------------------------------------------


class TestFormatShelterLoads:
    _CAPS = {33: 12_000, 58: 10_000}

    def test_returns_one_row_per_shelter(self):
        metrics = {"shelter_33_load": 8000, "shelter_58_load": 9000,
                   "shelter_33_capacity": 12000, "shelter_58_capacity": 10000}
        rows = format_shelter_loads(metrics, self._CAPS)
        assert len(rows) == 2

    def test_shelter_ids_in_ascending_order(self):
        metrics = {"shelter_33_load": 0, "shelter_58_load": 0,
                   "shelter_33_capacity": 12000, "shelter_58_capacity": 10000}
        rows = format_shelter_loads(metrics, self._CAPS)
        assert rows[0]["shelter"] == "Scenario Shelter A (node 33)"
        assert rows[1]["shelter"] == "Scenario Shelter B (node 58)"

    def test_over_capacity_flag_true_when_load_exceeds_cap(self):
        metrics = {"shelter_33_load": 13_000, "shelter_58_load": 0,
                   "shelter_33_capacity": 12000, "shelter_58_capacity": 10000}
        rows = format_shelter_loads(metrics, self._CAPS)
        assert rows[0]["over_capacity"] is True
        assert rows[1]["over_capacity"] is False

    def test_utilization_format(self):
        metrics = {"shelter_33_load": 6000, "shelter_58_load": 0,
                   "shelter_33_capacity": 12000, "shelter_58_capacity": 10000}
        rows = format_shelter_loads(metrics, self._CAPS)
        assert rows[0]["utilization"] == "50.0%"

    def test_zero_load_not_over_capacity(self):
        metrics = {"shelter_33_load": 0, "shelter_58_load": 0,
                   "shelter_33_capacity": 12000, "shelter_58_capacity": 10000}
        rows = format_shelter_loads(metrics, self._CAPS)
        assert all(not r["over_capacity"] for r in rows)

    def test_missing_load_key_defaults_to_zero(self):
        rows = format_shelter_loads({}, self._CAPS)
        assert all(r["load"] == 0 for r in rows)


# ---------------------------------------------------------------------------
# format_recommendation
# ---------------------------------------------------------------------------


class TestFormatRecommendation:
    def _m(self, rate=1.0, detour=1.0, exposed=0.0):
        return {
            "assignment_rate": rate,
            "detour_ratio": detour,
            "flood_exposed_length_m": exposed,
        }

    def test_alg_a_exposed_mentions_flood(self):
        rec = format_recommendation("A", self._m(exposed=500.0))
        assert "500" in rec or "flood" in rec.lower()

    def test_alg_a_no_exposure_says_flood_free(self):
        rec = format_recommendation("A", self._m(exposed=0.0))
        assert "flood-free" in rec.lower() or "0 m" in rec

    def test_alg_b_detour_mentions_factor(self):
        rec = format_recommendation("B", self._m(detour=1.25))
        assert "1.250" in rec or "detour" in rec.lower()

    def test_alg_b_no_detour_says_match(self):
        rec = format_recommendation("B", self._m(detour=1.0))
        assert "match" in rec.lower() or "no active" in rec.lower()

    def test_alg_c_partial_assignment_mentions_rate(self):
        rec = format_recommendation("C", self._m(rate=0.85))
        assert "85%" in rec or "capacity" in rec.lower()

    def test_alg_c_full_assignment_no_detour(self):
        rec = format_recommendation("C", self._m(rate=1.0, detour=1.0))
        assert "fully" in rec.lower() or "all demand" in rec.lower()

    def test_alg_c_full_assignment_with_detour(self):
        rec = format_recommendation("C", self._m(rate=1.0, detour=1.15))
        assert "1.150" in rec or "detour" in rec.lower()


# ---------------------------------------------------------------------------
# format_run_label
# ---------------------------------------------------------------------------


class TestFormatRunLabel:
    def test_label_format(self):
        assert format_run_label("C", "RP20", 0.25) == "C / RP20 / 25%"

    def test_all_fractions(self):
        assert "10%" in format_run_label("A", "RP10", 0.10)
        assert "50%" in format_run_label("B", "RP100", 0.50)


# ---------------------------------------------------------------------------
# map_builder colour / weight helpers
# ---------------------------------------------------------------------------


class TestFloodedColor:
    def test_none_depth_returns_orange(self):
        color = flooded_color(None)
        assert color == "#F97316"

    def test_shallow_depth_returns_yellow(self):
        color = flooded_color(0.2)
        assert color == "#FCD34D"

    def test_moderate_depth_returns_orange(self):
        color = flooded_color(0.7)
        assert color == "#F97316"

    def test_deep_depth_returns_red(self):
        color = flooded_color(1.5)
        assert color == "#EF4444"

    def test_boundary_at_0_5_is_orange(self):
        # depth == 0.5 should fall into the moderate (orange) band
        assert flooded_color(0.5) == "#F97316"

    def test_boundary_at_1_0_is_red(self):
        assert flooded_color(1.0) == "#EF4444"


class TestEdgeColor:
    def test_modelled_dry_returns_gray(self):
        color = edge_color("modelled_dry", None)
        assert color == "#9CA3AF"

    def test_flooded_shallow_returns_yellow(self):
        color = edge_color("flooded", 0.1)
        assert color == "#FCD34D"

    def test_flooded_deep_returns_red(self):
        color = edge_color("flooded", 2.0)
        assert color == "#EF4444"

    def test_no_overlap_returns_pale_gray(self):
        color = edge_color("no_overlap", None)
        assert color == "#E5E7EB"

    def test_outside_domain_returns_pale_gray(self):
        color = edge_color("outside_domain", None)
        assert color == "#E5E7EB"

    def test_none_status_returns_pale_gray(self):
        color = edge_color(None, None)
        assert color == "#E5E7EB"

    def test_unknown_status_returns_pale_gray(self):
        color = edge_color("some_future_status", None)
        assert color == "#E5E7EB"


class TestEdgeWeight:
    def test_flooded_has_highest_weight(self):
        assert edge_weight("flooded") > edge_weight("modelled_dry")

    def test_modelled_dry_heavier_than_unavailable(self):
        assert edge_weight("modelled_dry") > edge_weight("no_overlap")

    def test_none_status_returns_light_weight(self):
        w = edge_weight(None)
        assert w <= edge_weight("modelled_dry")


class TestEdgeOpacity:
    def test_flooded_and_dry_have_high_opacity(self):
        assert edge_opacity("flooded") > 0.5
        assert edge_opacity("modelled_dry") > 0.5

    def test_unavailable_has_low_opacity(self):
        assert edge_opacity("no_overlap") < 0.5
        assert edge_opacity(None) < 0.5


class TestStatusAvailability:
    def test_modelled_dry_available(self):
        assert "available" in STATUS_AVAILABILITY["modelled_dry"].lower()

    def test_flooded_available(self):
        assert "available" in STATUS_AVAILABILITY["flooded"].lower()

    def test_no_overlap_unavailable(self):
        assert "unavailable" in STATUS_AVAILABILITY["no_overlap"].lower()

    def test_outside_domain_unavailable(self):
        assert "unavailable" in STATUS_AVAILABILITY["outside_domain"].lower()

    def test_all_expected_statuses_present(self):
        for key in ("modelled_dry", "flooded", "no_overlap", "outside_domain", "missing"):
            assert key in STATUS_AVAILABILITY


# ---------------------------------------------------------------------------
# format_origin_assignment_status
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal RunResult stand-in for unit testing."""

    def __init__(self, assignments, od_costs_scenario):
        self.assignments = assignments
        self.od_costs_scenario = od_costs_scenario
        # routes not needed for format_origin_assignment_status
        self.routes = {}


class TestFormatOriginAssignmentStatus:
    def test_assigned_origin_returns_assigned_status(self):
        result = _FakeResult(
            assignments={(1, 33): 500, (2, 58): 200},
            od_costs_scenario={(1, 33): 100.0, (2, 58): 200.0},
        )
        status = format_origin_assignment_status(1, result)
        assert status["status"] == "assigned"
        assert status["shelter"] == 33
        assert status["units"] == 500

    def test_assigned_origin_picks_shelter_with_most_units(self):
        # Origin 1 split across two shelters
        result = _FakeResult(
            assignments={(1, 33): 300, (1, 58): 700},
            od_costs_scenario={(1, 33): 100.0, (1, 58): 150.0},
        )
        status = format_origin_assignment_status(1, result)
        assert status["status"] == "assigned"
        assert status["shelter"] == 58  # higher assignment
        assert status["units"] == 700

    def test_unreachable_origin_returns_unreachable(self):
        # Origin 99 not in od_costs_scenario at all
        result = _FakeResult(
            assignments={(1, 33): 100},
            od_costs_scenario={(1, 33): 100.0},
        )
        status = format_origin_assignment_status(99, result)
        assert status["status"] == "unreachable"
        assert status["shelter"] is None
        assert status["units"] == 0
        assert "path" in status["reason"].lower() or "no path" in status["reason"].lower()

    def test_capacity_exhausted_origin_returns_unassigned(self):
        # Origin 5 is reachable (in od_costs_scenario) but not assigned
        result = _FakeResult(
            assignments={(1, 33): 100},  # origin 5 not here
            od_costs_scenario={(1, 33): 100.0, (5, 33): 200.0, (5, 58): 300.0},
        )
        status = format_origin_assignment_status(5, result)
        assert status["status"] == "unassigned"
        assert status["shelter"] is None
        assert status["units"] == 0
        assert "capacity" in status["reason"].lower()

    def test_zero_unit_assignment_is_treated_as_unassigned(self):
        # Assignment exists but units == 0 (should not occur in practice,
        # but must not be classified as assigned)
        result = _FakeResult(
            assignments={(7, 33): 0},
            od_costs_scenario={(7, 33): 50.0},
        )
        # Zero-unit pair is in od_costs but has no positive assignment
        status = format_origin_assignment_status(7, result)
        assert status["status"] == "unassigned"

    def test_reason_string_is_non_empty(self):
        result = _FakeResult(assignments={}, od_costs_scenario={})
        status = format_origin_assignment_status(42, result)
        assert status["reason"] and len(status["reason"]) > 0


# ---------------------------------------------------------------------------
# Ordinary-route semantics (weight function)
# ---------------------------------------------------------------------------


class TestOrdinaryWeightSemantics:
    """Verify that the ordinary weight function uses all edges regardless of
    flood status — i.e. it does NOT apply the conservative hazard exclusion.
    """

    def _make_parallel_edge_dict(self, *edge_attrs):
        """Return a MultiDiGraph-style {key: attrs} dict."""
        return {i: attrs for i, attrs in enumerate(edge_attrs)}

    def test_traverses_flooded_edge(self):
        """A flooded edge must return a finite weight (not None)."""
        from floodroute.dashboard.app import _ordinary_weight

        d = self._make_parallel_edge_dict(
            {"length_m": 100.0, "jrc_rp20_status": "flooded"}
        )
        w = _ordinary_weight(0, 1, d)
        assert w is not None
        assert w == pytest.approx(100.0)

    def test_traverses_no_overlap_edge(self):
        from floodroute.dashboard.app import _ordinary_weight

        d = self._make_parallel_edge_dict(
            {"length_m": 50.0, "jrc_rp20_status": "no_overlap"}
        )
        w = _ordinary_weight(0, 1, d)
        assert w is not None
        assert w == pytest.approx(50.0)

    def test_traverses_outside_domain_edge(self):
        from floodroute.dashboard.app import _ordinary_weight

        d = self._make_parallel_edge_dict(
            {"length_m": 75.0, "jrc_rp20_status": "outside_domain"}
        )
        w = _ordinary_weight(0, 1, d)
        assert w is not None
        assert w == pytest.approx(75.0)

    def test_picks_minimum_length_across_parallel_edges(self):
        from floodroute.dashboard.app import _ordinary_weight

        d = self._make_parallel_edge_dict(
            {"length_m": 200.0, "jrc_rp20_status": "flooded"},
            {"length_m": 80.0, "jrc_rp20_status": "modelled_dry"},
        )
        w = _ordinary_weight(0, 1, d)
        assert w == pytest.approx(80.0)

    def test_ignores_flood_status_when_choosing_min_length(self):
        """Ordinary routing picks the shorter edge even if it is flooded."""
        from floodroute.dashboard.app import _ordinary_weight

        d = self._make_parallel_edge_dict(
            {"length_m": 50.0, "jrc_rp20_status": "flooded"},
            {"length_m": 120.0, "jrc_rp20_status": "modelled_dry"},
        )
        w = _ordinary_weight(0, 1, d)
        assert w == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Smoke test — data-backed map build
# ---------------------------------------------------------------------------

_DATA_FILES = [
    "data/processed/hazard/PH0600613_phase_b_enriched.gpkg",
    "data/processed/osm/PH0600613_waterways_wgs84.gpkg",
    "data/processed/admin/municipalities_wgs84.gpkg",
    "data/processed/hazard/PH0600613_phase_b_enriched.graphml",
]


def _data_present() -> bool:
    from pathlib import Path

    return all(Path(p).exists() for p in _DATA_FILES)


@pytest.mark.skipif(not _data_present(), reason="Stage 9 data files not present")
class TestMapBuildSmoke:
    """Data-backed smoke tests — skipped if source files are absent."""

    def setup_method(self):
        """Reset module caches and load shared graph/data for each test."""
        import os

        os.chdir("/Users/stephanie/Desktop/Thesis1/FloodRoute")

        import floodroute.dashboard.map_builder as mb

        mb._EDGES_GDF = None
        mb._WATERWAYS_GDF = None
        mb._BOUNDARY_GDF = None

        import networkx as nx

        from floodroute.experiments.algorithms import run_floodroute_assignment
        from floodroute.experiments.demand import (
            build_demands,
            load_psa_population,
            snap_barangay_origins,
        )
        from floodroute.experiments.metrics import compute_metrics
        from floodroute.experiments.runner import (
            _DEFAULT_BARANGAY_GPKG,
            _DEFAULT_GRAPHML,
            _DEFAULT_NODES_GPKG,
            _DEFAULT_POP_CSV,
            MUNICIPALITY_PSGC,
            SCENARIO_SHELTER_CAPACITIES,
        )

        self.G = nx.read_graphml(str(_DEFAULT_GRAPHML), node_type=int)
        records = load_psa_population(_DEFAULT_POP_CSV, adm3_filter=MUNICIPALITY_PSGC)
        self.origins = snap_barangay_origins(
            records,
            barangay_gpkg=_DEFAULT_BARANGAY_GPKG,
            nodes_gpkg=_DEFAULT_NODES_GPKG,
            municipality_psgc=MUNICIPALITY_PSGC,
            exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
        )
        total_pop = sum(o.population_2020 for o in self.origins)
        demands, _ = build_demands(self.origins, 0.25)
        self.result = run_floodroute_assignment(
            self.G, demands, SCENARIO_SHELTER_CAPACITIES, "RP20"
        )
        self.result.demand_fraction = 0.25
        self.metrics = compute_metrics(self.result, self.G, total_population=total_pop)
        self.caps = SCENARIO_SHELTER_CAPACITIES

    def test_map_builds_without_error(self):
        import folium

        from floodroute.dashboard.map_builder import build_analytical_map

        fmap = build_analytical_map(
            self.G, "RP20", self.caps,
            result=self.result, metrics=self.metrics,
        )
        assert isinstance(fmap, folium.Map)

    def test_alg_c_route_endpoint_matches_assigned_shelter(self):
        """The last node in the MCF route must equal the shelter Algorithm C assigned."""
        maybato = next(o for o in self.origins if o.psgc == "PH0600613023")
        origin = maybato.origin_node
        status = format_origin_assignment_status(origin, self.result)

        if status["status"] == "assigned":
            shelter = status["shelter"]
            # Find the stored MCF route
            route = self.result.routes.get((origin, shelter))
            assert route is not None, "Assigned pair must have a stored route"
            assert route[-1] == shelter, (
                f"Route endpoint {route[-1]} must equal assigned shelter {shelter}"
            )

    def test_ordinary_weight_produces_finite_path(self):
        """Ordinary weight function must allow a path to be found."""
        import networkx as nx

        from floodroute.dashboard.app import _ordinary_weight
        from floodroute.experiments.runner import SCENARIO_SHELTER_CAPACITIES

        maybato = next(o for o in self.origins if o.psgc == "PH0600613023")
        origin = maybato.origin_node
        lengths, _ = nx.single_source_dijkstra(self.G, origin, weight=_ordinary_weight)
        reachable_shelters = [
            s for s in SCENARIO_SHELTER_CAPACITIES if s in lengths
        ]
        assert len(reachable_shelters) > 0, "Maybato Norte must reach at least one shelter"

    def test_file_not_found_raises_clear_error(self):
        """Missing data file must raise FileNotFoundError with a non-empty message."""
        from pathlib import Path

        import floodroute.dashboard.map_builder as mb

        # Temporarily point to a non-existent path
        orig = mb._ENRICHED_GPKG
        try:
            mb._EDGES_GDF = None
            mb._ENRICHED_GPKG = Path("data/does_not_exist.gpkg")
            with pytest.raises(FileNotFoundError) as exc_info:
                mb._get_edges_wgs84()
            assert len(str(exc_info.value)) > 10
        finally:
            mb._ENRICHED_GPKG = orig
            mb._EDGES_GDF = None


# ---------------------------------------------------------------------------
# SHELTER_DISPLAY_LABELS constant
# ---------------------------------------------------------------------------


class TestShelterDisplayLabels:
    def test_shelter_33_label(self):
        assert SHELTER_DISPLAY_LABELS[33] == "Scenario Shelter A (node 33)"

    def test_shelter_58_label(self):
        assert SHELTER_DISPLAY_LABELS[58] == "Scenario Shelter B (node 58)"

    def test_labels_include_both_name_and_node(self):
        for node, label in SHELTER_DISPLAY_LABELS.items():
            assert "Scenario Shelter" in label
            assert f"node {node}" in label


# ---------------------------------------------------------------------------
# Route / waterway style constants
# ---------------------------------------------------------------------------


class TestRouteStyleConstants:
    def test_ordinary_route_is_dark_navy_not_medium_blue(self):
        from floodroute.dashboard.map_builder import ORDINARY_ROUTE_COLOR, WATERWAY_COLOR

        # Dark navy must not equal the lighter waterway blue
        assert ORDINARY_ROUTE_COLOR != WATERWAY_COLOR
        # Must start with # and be a valid hex color
        assert ORDINARY_ROUTE_COLOR.startswith("#")
        assert len(ORDINARY_ROUTE_COLOR) == 7

    def test_waterway_is_lighter_blue(self):
        from floodroute.dashboard.map_builder import WATERWAY_COLOR, WATERWAY_WEIGHT

        assert WATERWAY_COLOR.startswith("#")
        # Weight must be thin (≤ 1.5)
        assert WATERWAY_WEIGHT <= 1.5

    def test_floodroute_color_is_green(self):
        from floodroute.dashboard.map_builder import FLOODROUTE_COLOR

        # Green channel dominant: rough check via lowercase hex
        assert FLOODROUTE_COLOR.startswith("#")
        r = int(FLOODROUTE_COLOR[1:3], 16)
        g = int(FLOODROUTE_COLOR[3:5], 16)
        assert g > r, "FloodRoute color should be green-dominant"

    def test_reference_color_is_purple(self):
        from floodroute.dashboard.map_builder import REFERENCE_ROUTE_COLOR

        assert REFERENCE_ROUTE_COLOR.startswith("#")
        r = int(REFERENCE_ROUTE_COLOR[1:3], 16)
        b = int(REFERENCE_ROUTE_COLOR[5:7], 16)
        assert r > 0 and b > 0, "Reference color should be purple (red + blue)"

    def test_ordinary_darker_than_floodroute(self):
        """Dark navy ordinary route must be visually darker than bright green MCF."""
        from floodroute.dashboard.map_builder import FLOODROUTE_COLOR, ORDINARY_ROUTE_COLOR

        def _luminance(hex_color: str) -> float:
            r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
            return 0.299 * r + 0.587 * g + 0.114 * b

        assert _luminance(ORDINARY_ROUTE_COLOR) < _luminance(FLOODROUTE_COLOR)


# ---------------------------------------------------------------------------
# format_barangay_card
# ---------------------------------------------------------------------------


class TestFormatBarangayCard:
    _STATUS_ASSIGNED = {"status": "assigned", "reason": "Assigned 500 units to shelter node 33"}
    _STATUS_UNREACHABLE = {"status": "unreachable", "reason": "No path to any shelter"}
    _STATUS_UNASSIGNED = {"status": "unassigned", "reason": "Shelter capacity exhausted"}

    def test_assigned_formats_shelter_label_correctly(self):
        card = format_barangay_card(
            bgy_name="Test Barangay",
            population_2020=4000,
            demand=1000,
            assigned=1000,
            shelter_node=33,
            ordinary_dist_m=500.0,
            alg_dist_m=600.0,
            flood_exposed_m=0.0,
            assignment_status=self._STATUS_ASSIGNED,
        )
        assert "Scenario Shelter A" in card["shelter"]
        assert "node 33" in card["shelter"]

    def test_unassigned_shelter_shows_dash(self):
        card = format_barangay_card(
            bgy_name="Test Barangay",
            population_2020=4000,
            demand=1000,
            assigned=0,
            shelter_node=None,
            ordinary_dist_m=None,
            alg_dist_m=None,
            flood_exposed_m=None,
            assignment_status=self._STATUS_UNASSIGNED,
        )
        assert card["shelter"] == "—"

    def test_unassigned_computes_unassigned_correctly(self):
        card = format_barangay_card(
            bgy_name="Test",
            population_2020=1000,
            demand=500,
            assigned=300,
            shelter_node=58,
            ordinary_dist_m=None,
            alg_dist_m=None,
            flood_exposed_m=None,
            assignment_status=self._STATUS_ASSIGNED,
        )
        assert card["unassigned"] == "200"

    def test_distances_formatted_with_unit(self):
        card = format_barangay_card(
            bgy_name="Test",
            population_2020=1000,
            demand=250,
            assigned=250,
            shelter_node=33,
            ordinary_dist_m=1234.5,
            alg_dist_m=1567.8,
            flood_exposed_m=89.0,
            assignment_status=self._STATUS_ASSIGNED,
        )
        assert "1,234 m" in card["ordinary_dist"]  # :.0f banker-rounds 1234.5 → 1234
        assert "1,568 m" in card["alg_dist"]
        assert "89 m" in card["flood_exposed"]

    def test_none_distances_show_dash(self):
        card = format_barangay_card(
            bgy_name="Test",
            population_2020=1000,
            demand=250,
            assigned=0,
            shelter_node=None,
            ordinary_dist_m=None,
            alg_dist_m=None,
            flood_exposed_m=None,
            assignment_status=self._STATUS_UNREACHABLE,
        )
        assert card["ordinary_dist"] == "—"
        assert card["alg_dist"] == "—"
        assert card["flood_exposed"] == "—"

    def test_show_reason_false_when_assigned(self):
        card = format_barangay_card(
            bgy_name="Test",
            population_2020=1000,
            demand=250,
            assigned=250,
            shelter_node=33,
            ordinary_dist_m=None,
            alg_dist_m=None,
            flood_exposed_m=None,
            assignment_status=self._STATUS_ASSIGNED,
        )
        assert card["show_reason"] is False

    def test_show_reason_true_when_unassigned(self):
        card = format_barangay_card(
            bgy_name="Test",
            population_2020=1000,
            demand=250,
            assigned=0,
            shelter_node=None,
            ordinary_dist_m=None,
            alg_dist_m=None,
            flood_exposed_m=None,
            assignment_status=self._STATUS_UNASSIGNED,
        )
        assert card["show_reason"] is True
        assert "capacity" in card["reason"].lower()

    def test_unknown_shelter_node_falls_back_to_node_label(self):
        card = format_barangay_card(
            bgy_name="Test",
            population_2020=1000,
            demand=250,
            assigned=250,
            shelter_node=99,
            ordinary_dist_m=None,
            alg_dist_m=None,
            flood_exposed_m=None,
            assignment_status=self._STATUS_ASSIGNED,
        )
        assert "Node 99" in card["shelter"]


# ---------------------------------------------------------------------------
# _path_length_m helper
# ---------------------------------------------------------------------------


class TestPathLengthM:
    def _make_graph(self):
        import networkx as nx

        G = nx.MultiDiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_node(3)
        G.add_edge(1, 2, length_m=100.0, jrc_rp20_status="modelled_dry")
        G.add_edge(2, 3, length_m=200.0, jrc_rp20_status="flooded")
        return G

    def test_total_length_sums_edges(self):
        from floodroute.dashboard.app import _path_length_m

        G = self._make_graph()
        total_m, flood_m = _path_length_m(G, [1, 2, 3])
        assert total_m == pytest.approx(300.0)
        assert flood_m == pytest.approx(0.0)  # no return_period given

    def test_flood_exposed_with_return_period(self):
        from floodroute.dashboard.app import _path_length_m

        G = self._make_graph()
        total_m, flood_m = _path_length_m(G, [1, 2, 3], return_period="RP20")
        assert total_m == pytest.approx(300.0)
        assert flood_m == pytest.approx(200.0)  # only edge 2→3 is flooded

    def test_dry_edge_not_counted_as_flood_exposed(self):
        from floodroute.dashboard.app import _path_length_m

        G = self._make_graph()
        total_m, flood_m = _path_length_m(G, [1, 2], return_period="RP20")
        assert total_m == pytest.approx(100.0)
        assert flood_m == pytest.approx(0.0)

    def test_empty_path_returns_zeros(self):
        from floodroute.dashboard.app import _path_length_m

        G = self._make_graph()
        total_m, flood_m = _path_length_m(G, [], return_period="RP20")
        assert total_m == 0.0
        assert flood_m == 0.0

    def test_single_node_path_returns_zeros(self):
        from floodroute.dashboard.app import _path_length_m

        G = self._make_graph()
        total_m, flood_m = _path_length_m(G, [1], return_period="RP20")
        assert total_m == 0.0
        assert flood_m == 0.0

    def test_picks_minimum_length_parallel_edge(self):
        import networkx as nx

        from floodroute.dashboard.app import _path_length_m

        G = nx.MultiDiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_edge(1, 2, length_m=500.0, jrc_rp20_status="modelled_dry")
        G.add_edge(1, 2, length_m=80.0, jrc_rp20_status="modelled_dry")
        total_m, _ = _path_length_m(G, [1, 2])
        assert total_m == pytest.approx(80.0)
