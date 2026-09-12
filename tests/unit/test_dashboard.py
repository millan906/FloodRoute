"""Unit tests for floodroute.dashboard.result_formatter and map_builder helpers.

All tests are pure-Python — no Streamlit, no geospatial dependencies.
"""

from __future__ import annotations

import pytest

from floodroute.dashboard.map_builder import (
    FACILITY_ROUTE_PALETTE,
    STATUS_AVAILABILITY,
    edge_color,
    edge_opacity,
    edge_weight,
    facility_route_color,
    flooded_color,
)
from floodroute.dashboard.result_formatter import (
    SHELTER_DISPLAY_LABELS,
    classify_unassigned_cause,
    format_barangay_card,
    format_feasibility_status,
    format_origin_assignment_status,
    format_recommendation,
    format_run_label,
    format_shelter_loads,
    summarise_unassigned,
    unassigned_rows,
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

    def __init__(self, assignments, od_costs_scenario, capacities=None):
        self.assignments = assignments
        self.od_costs_scenario = od_costs_scenario
        self.capacities = capacities or {}
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
            capacities={33: 100},
        )
        status = format_origin_assignment_status(99, result)
        assert status["status"] == "unreachable"
        assert status["shelter"] is None
        assert status["units"] == 0
        assert "route" in status["reason"].lower()

    def test_capacity_exhausted_origin_returns_unassigned(self):
        # Origin 5 is reachable (in od_costs_scenario) but not assigned.
        # Both shelters 33 and 58 are at capacity so the diagnostic assertion
        # must not fire (no reachable shelter has remaining capacity).
        result = _FakeResult(
            assignments={(1, 33): 100, (2, 58): 200},
            od_costs_scenario={
                (1, 33): 100.0, (2, 58): 150.0,
                (5, 33): 200.0, (5, 58): 300.0,
            },
            capacities={33: 100, 58: 200},  # both shelters exactly at load
        )
        status = format_origin_assignment_status(5, result)
        assert status["status"] == "unassigned"
        assert status["shelter"] is None
        assert status["units"] == 0
        assert "capacity" in status["reason"].lower()

    def test_zero_unit_assignment_is_treated_as_unassigned(self):
        # Assignment exists but units == 0 (should not occur in practice,
        # but must not be classified as assigned).
        # Shelter 33 is filled to capacity by origin 1 so the assertion
        # does not fire for origin 7.
        result = _FakeResult(
            assignments={(1, 33): 100, (7, 33): 0},
            od_costs_scenario={(7, 33): 50.0, (1, 33): 30.0},
            capacities={33: 100},  # shelter at capacity; no remaining space
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

    def test_shelter_labels_param_overrides_hardcoded_labels(self):
        """Regression: shelter_labels kwarg must appear in marker tooltips, not the
        hardcoded fallback 'Scenario Shelter (node X)'."""
        from floodroute.dashboard.map_builder import build_analytical_map

        custom_labels = {
            33: "SJDB Municipal Evacuation Center",
            58: "SJDB Covered Court",
        }
        fmap = build_analytical_map(
            self.G, "RP10", self.caps,
            result=self.result,
            metrics=self.metrics,
            shelter_labels=custom_labels,
        )
        html = fmap._repr_html_()
        assert "SJDB Municipal Evacuation Center" in html, (
            "shelter_labels value for node 33 must appear in map HTML"
        )
        assert "SJDB Covered Court" in html, (
            "shelter_labels value for node 58 must appear in map HTML"
        )


# ---------------------------------------------------------------------------
# _origin_tooltip_html — pure helper
# ---------------------------------------------------------------------------


class TestOriginTooltipHtml:
    """Unit tests for _origin_tooltip_html — pure, no Folium or graph deps."""

    from floodroute.dashboard.map_builder import _origin_tooltip_html  # noqa: E402

    _NODE_INFO = {
        10: {"name": "Maybato Sur", "population_2020": 2308},
        20: {"name": "Atabay", "population_2020": 2180},
    }
    _LABELS = {33: "Salazar Elementary", 58: "Covered Court"}

    def _html(self, **kw):
        from floodroute.dashboard.map_builder import _origin_tooltip_html
        defaults = dict(
            node=10,
            demand=577,
            assigned=577,
            node_info=self._NODE_INFO,
            demand_fraction=0.25,
            assignments={(10, 33): 577},
            shelter_labels=self._LABELS,
            is_selected=False,
        )
        defaults.update(kw)
        return _origin_tooltip_html(**defaults)

    def test_header_shows_barangay_name_not_node_id_first(self):
        html = self._html()
        # Barangay name must appear before the raw node id
        assert "Maybato Sur" in html
        idx_name = html.index("Maybato Sur")
        idx_node = html.index("node 10")
        assert idx_name < idx_node, "Barangay name must precede the node id"

    def test_header_contains_pickup_point_label(self):
        assert "Barangay pickup point" in self._html()

    def test_psa_population_shown(self):
        assert "2,308" in self._html()

    def test_fraction_demand_shows_percentage_and_hamilton(self):
        html = self._html(demand_fraction=0.25, demand=577)
        assert "25%" in html
        assert "Hamilton apportionment" in html
        assert "577" in html

    def test_exact_demand_shows_exact_label_not_hamilton(self):
        html = self._html(demand_fraction=None, demand=500)
        assert "exact" in html.lower()
        assert "Hamilton" not in html

    def test_assigned_and_unassigned_totals(self):
        html = self._html(demand=577, assigned=400)
        assert "Assigned: 400" in html
        assert "Unassigned: 177" in html

    def test_fully_assigned_shows_zero_unassigned(self):
        html = self._html(demand=577, assigned=577)
        assert "Unassigned: 0" in html

    def test_single_facility_name_and_count(self):
        html = self._html(assignments={(10, 33): 577})
        assert "Salazar Elementary" in html
        assert "577" in html

    def test_two_facility_split(self):
        """Demand split between two facilities — both names and counts appear."""
        html = self._html(
            demand=577,
            assigned=577,
            assignments={(10, 33): 377, (10, 58): 200},
        )
        assert "Salazar Elementary" in html
        assert "377" in html
        assert "Covered Court" in html
        assert "200" in html
        assert "Assigned facilities:" in html

    def test_unassigned_origin_shows_no_facility_message(self):
        html = self._html(
            demand=436,
            assigned=0,
            assignments={},
        )
        assert "No facility assignment" in html
        assert "Unassigned: 436" in html

    def test_selected_marker_includes_star(self):
        html = self._html(is_selected=True)
        # ★ encoded as &#9733; or the literal character
        assert "&#9733;" in html or "★" in html

    def test_node_id_present_as_technical_detail(self):
        html = self._html(node=10)
        assert "node 10" in html

    def test_missing_node_info_falls_back_gracefully(self):
        html = self._html(node=99, node_info={})
        assert "Pickup point 99" in html

    def test_capacity_not_mentioned(self):
        """Capacity is a shelter property; the origin popup must not show it."""
        html = self._html()
        assert "capacity" not in html.lower()

    def test_other_origin_assignments_excluded(self):
        """Assignments from a different origin must not appear in this tooltip."""
        html = self._html(
            node=10,
            assignments={(10, 33): 377, (20, 33): 999},  # node 20 is a different origin
        )
        assert "999" not in html


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

# ---------------------------------------------------------------------------
# Regression: generate-plan button path must load G before run_floodroute_assignment
# ---------------------------------------------------------------------------


class TestGeneratePlanPath:
    """Regression guard for NameError: name 'G' is not defined.

    Before the fix the run-plan body executed unconditionally on every
    render — not inside a button handler — and called
    run_floodroute_assignment(G, ...) without G ever being loaded in
    that scope.  An HTTP 200 check cannot catch this because the error
    only fires when the user clicks the button (i.e. after initial load).
    These tests directly exercise the code path that was broken.
    """

    @pytest.fixture(autouse=True)
    def _require_data(self):
        from pathlib import Path

        from floodroute.experiments.runner import _DEFAULT_GRAPHML

        if not Path(_DEFAULT_GRAPHML).exists():
            pytest.skip("Graph data not present")

    def test_load_graph_returns_non_empty_multidigraph(self):
        """_load_graph() must return a MultiDiGraph with nodes and edges."""
        import networkx as nx

        from floodroute.dashboard.app import _load_graph

        G = _load_graph()
        assert isinstance(G, nx.MultiDiGraph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_generate_plan_path_does_not_raise_name_error(self):
        """Calling run_floodroute_assignment with _load_graph() output must succeed.

        This mirrors the corrected button-handler body:
            G = _load_graph()
            result = run_floodroute_assignment(G, demands, shelters, rp)
        The test would have caught the NameError that existed before the fix
        because G was not assigned before that call in the module body.
        """
        from floodroute.dashboard.app import _load_catalog, _load_graph, _load_origins
        from floodroute.experiments.algorithms import run_floodroute_assignment
        from floodroute.experiments.demand import build_demands
        from floodroute.scenario.config import ScenarioConfig

        # Exact sequence executed inside the fixed button handler
        G = _load_graph()
        origins, total_pop = _load_origins()
        demands, _ = build_demands(origins, 0.25)
        catalog = _load_catalog()
        scenario = ScenarioConfig(
            municipality="PH0600608",
            return_period="RP10",
            demand_mode="fraction",
            demand_fraction=0.25,
            selected_facility_ids=[
                e.facility_id for e in catalog.all() if e.can_be_selected
            ],
            facility_capacities={
                e.facility_id: 500 for e in catalog.all() if e.can_be_selected
            },
        )
        origin_nodes = {o.origin_node for o in origins}
        effective_shelters = catalog.resolve_node_shelters(
            scenario, exclude_nodes=origin_nodes
        )

        result = run_floodroute_assignment(G, demands, effective_shelters, "RP10")

        assert result is not None
        assert hasattr(result, "assignments")
        assert hasattr(result, "routes")
        assert sum(result.demands.values()) > 0

    def test_generate_plan_metrics_complete(self):
        """compute_metrics must return a complete dict after plan generation."""
        from floodroute.dashboard.app import _load_catalog, _load_graph, _load_origins
        from floodroute.experiments.algorithms import run_floodroute_assignment
        from floodroute.experiments.demand import build_demands
        from floodroute.experiments.metrics import compute_metrics
        from floodroute.scenario.config import ScenarioConfig

        G = _load_graph()
        origins, total_pop = _load_origins()
        demands, _ = build_demands(origins, 0.25)
        catalog = _load_catalog()
        scenario = ScenarioConfig(
            municipality="PH0600608",
            return_period="RP10",
            demand_mode="fraction",
            demand_fraction=0.25,
            selected_facility_ids=[
                e.facility_id for e in catalog.all() if e.can_be_selected
            ],
            facility_capacities={
                e.facility_id: 500 for e in catalog.all() if e.can_be_selected
            },
        )
        origin_nodes = {o.origin_node for o in origins}
        effective_shelters = catalog.resolve_node_shelters(
            scenario, exclude_nodes=origin_nodes
        )
        result = run_floodroute_assignment(G, demands, effective_shelters, "RP10")
        metrics = compute_metrics(result, G, total_population=total_pop)

        assert metrics["total_demand"] > 0
        assert 0.0 <= metrics["assignment_rate"] <= 1.0
        assert "total_assigned" in metrics
        assert "total_unassigned" in metrics


# ---------------------------------------------------------------------------
# summarise_unassigned — display reason correctness
# ---------------------------------------------------------------------------


class TestSummariseUnassigned:
    """Verify that summarise_unassigned reports the correct cause and never
    blames capacity when all unassigned demand comes from unreachable origins."""

    _DEMANDS = {1: 1284, 2: 436, 3: 500}
    # Origins 1 and 2 are unreachable; 3 is reachable and fully assigned.
    _REACHABLE_ALL_ASSIGNED = {3}

    def test_zero_unassigned_returns_empty_string(self):
        assert summarise_unassigned(0, {1, 2, 3}, self._DEMANDS) == ""

    def test_all_unreachable_mentions_no_modeled_route_and_osm(self):
        """When every unassigned unit comes from unreachable origins, the message
        must use 'no modeled route', reference the OSM-derived network, and must
        not mention 'capacity'."""
        msg = summarise_unassigned(1720, self._REACHABLE_ALL_ASSIGNED, self._DEMANDS)
        assert "no modeled route" in msg, f"Expected 'no modeled route' in: {msg!r}"
        assert "OSM-derived" in msg, f"Expected 'OSM-derived' in: {msg!r}"
        assert "capacity" not in msg.lower(), (
            f"Must not mention capacity when cause is unreachability: {msg!r}"
        )

    def test_all_unreachable_count_matches(self):
        """The message must include the exact unassigned count."""
        msg = summarise_unassigned(1720, self._REACHABLE_ALL_ASSIGNED, self._DEMANDS)
        assert "1,720" in msg, f"Expected '1,720' in: {msg!r}"

    def test_all_capacity_mentions_capacity_not_pickup(self):
        """When every unassigned unit comes from a reachable origin, the message
        must mention 'capacity' and must not mention 'pickup points'."""
        # All origins reachable, but 500 unassigned → capacity-constrained
        msg = summarise_unassigned(500, {1, 2, 3}, self._DEMANDS)
        assert "capacity" in msg.lower(), f"Expected 'capacity' in: {msg!r}"
        assert "pickup points" not in msg, (
            f"Must not mention pickup points for capacity-only cause: {msg!r}"
        )

    def test_mixed_cause_mentions_both(self):
        """Mixed scenario: some unreachable, some capacity-constrained."""
        # Origins 1 (1284) unreachable, origin 3 (500) partially unassigned (300)
        # unreachable_demand = 1284; capacity_demand = 300; total = 1584
        demands = {1: 1284, 2: 436, 3: 500}
        reachable = {2, 3}  # origin 1 unreachable
        msg = summarise_unassigned(1584, reachable, demands)
        assert "pickup points" in msg, f"Expected 'pickup points' in: {msg!r}"
        assert "capacity" in msg.lower(), f"Expected 'capacity' in: {msg!r}"
        assert "1,284" in msg, f"Expected unreachable count '1,284' in: {msg!r}"
        assert "300" in msg, f"Expected capacity count '300' in: {msg!r}"

    def test_banner_and_panel_produce_same_reason(self):
        """Coverage banner and result panel both call summarise_unassigned with
        the same arguments — they must return identical strings."""
        total_unassigned = 1720
        reachable = self._REACHABLE_ALL_ASSIGNED
        demands = self._DEMANDS
        banner_reason = summarise_unassigned(total_unassigned, reachable, demands)
        panel_reason = summarise_unassigned(total_unassigned, reachable, demands)
        assert banner_reason == panel_reason, (
            "Coverage banner and result panel must display the same reason"
        )


# ---------------------------------------------------------------------------
# facility_route_color — deterministic per-facility route colours
# ---------------------------------------------------------------------------


class TestFacilityRouteColor:
    """Tests for facility_route_color() and FACILITY_ROUTE_PALETTE."""

    def test_same_shelter_same_list_returns_same_color(self):
        """Identical inputs must produce identical output (determinism)."""
        nodes = [10, 20, 30]
        assert facility_route_color(10, nodes) == facility_route_color(10, nodes)
        assert facility_route_color(20, nodes) == facility_route_color(20, nodes)

    def test_different_shelters_get_different_colors(self):
        """Two distinct shelter nodes in the same list must receive different colors."""
        nodes = [10, 20]
        assert facility_route_color(10, nodes) != facility_route_color(20, nodes)

    def test_color_is_palette_entry(self):
        """Returned color must be a member of FACILITY_ROUTE_PALETTE."""
        nodes = [5, 10, 15]
        for s in nodes:
            assert facility_route_color(s, nodes) in FACILITY_ROUTE_PALETTE

    def test_sorted_order_determines_color(self):
        """Position in the sorted list, not the node value, determines palette index."""
        # Node 5 is index 1 in [3, 5, 7]; same color as index 1 applied directly.
        nodes = [3, 5, 7]
        expected = FACILITY_ROUTE_PALETTE[1]
        assert facility_route_color(5, nodes) == expected

    def test_unknown_shelter_returns_first_palette_entry(self):
        """A shelter absent from the list falls back to index 0."""
        assert facility_route_color(999, [1, 2, 3]) == FACILITY_ROUTE_PALETTE[0]

    def test_split_assignment_origin_yields_two_colors(self):
        """An origin split across two facilities must produce two distinct colors.

        This is a display-layer invariant: split assignments are always visible
        as two differently colored polylines.
        """
        nodes = [10, 20]
        color_a = facility_route_color(10, nodes)
        color_b = facility_route_color(20, nodes)
        assert color_a != color_b, (
            "Split-assignment routes must use different colors for each destination"
        )

    def test_palette_wraps_for_large_shelter_sets(self):
        """With more shelters than palette entries colors wrap — no IndexError."""
        n = len(FACILITY_ROUTE_PALETTE) + 3
        nodes = list(range(n))
        # Must not raise and last node must reuse a palette color
        last_color = facility_route_color(nodes[-1], nodes)
        assert last_color in FACILITY_ROUTE_PALETTE


# ---------------------------------------------------------------------------
# unassigned_rows — per-origin unassigned breakdown
# ---------------------------------------------------------------------------


class _UnassignedFakeResult:
    """Minimal RunResult-shaped stub for unassigned_rows tests."""

    def __init__(
        self,
        demands: dict,
        assignments: dict,
        od_costs_scenario: dict,
    ) -> None:
        self.demands = demands
        self.assignments = assignments
        self.od_costs_scenario = od_costs_scenario


_NODE_INFO_STUB = {
    1: {"name": "Barangay Alpha", "population_2020": 1000},
    2: {"name": "Barangay Beta",  "population_2020": 800},
    3: {"name": "Barangay Gamma", "population_2020": 600},
}


class TestUnassignedRows:
    """Tests for unassigned_rows()."""

    def test_multiple_unassigned_barangays_all_returned(self):
        """All origins with unassigned demand must appear in the result."""
        result = _UnassignedFakeResult(
            demands={1: 200, 2: 150, 3: 100},
            assignments={},
            od_costs_scenario={},  # no reachability → all "no modeled route"
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        nodes = {r["node"] for r in rows}
        assert {1, 2, 3} == nodes, "All three unassigned origins must be listed"

    def test_multiple_unassigned_sorted_by_unassigned_descending(self):
        """Rows must be ordered largest unassigned first."""
        result = _UnassignedFakeResult(
            demands={1: 200, 2: 150, 3: 100},
            assignments={},
            od_costs_scenario={},
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        unassigned_vals = [r["unassigned"] for r in rows]
        assert unassigned_vals == sorted(unassigned_vals, reverse=True)

    def test_partial_assignment_capacity_reason(self):
        """Origin that received some units but not all → 'insufficient reachable capacity'."""
        result = _UnassignedFakeResult(
            demands={1: 500},
            assignments={(1, 10): 200},  # 300 unassigned
            od_costs_scenario={(1, 10): 1000.0},  # origin 1 is reachable
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        assert len(rows) == 1
        r = rows[0]
        assert r["assigned"] == 200
        assert r["unassigned"] == 300
        assert r["reason"] == "insufficient reachable capacity"

    def test_fully_unassigned_reachable_origin_capacity_reason(self):
        """Reachable but fully unassigned origin → 'insufficient reachable capacity'."""
        result = _UnassignedFakeResult(
            demands={2: 150},
            assignments={},
            od_costs_scenario={(2, 10): 500.0},  # reachable but no units assigned
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        assert len(rows) == 1
        assert rows[0]["reason"] == "insufficient reachable capacity"

    def test_fully_unassigned_unreachable_origin_no_route_reason(self):
        """Origin with no entry in od_costs_scenario → 'No route found in current map data.'"""
        result = _UnassignedFakeResult(
            demands={3: 100},
            assignments={},
            od_costs_scenario={},  # no path at all
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        assert len(rows) == 1
        assert rows[0]["reason"] == "No route found in current map data."

    def test_zero_unassigned_scenario_returns_empty(self):
        """When all demand is satisfied, unassigned_rows must return an empty list."""
        result = _UnassignedFakeResult(
            demands={1: 200, 2: 150},
            assignments={(1, 10): 200, (2, 10): 150},
            od_costs_scenario={(1, 10): 400.0, (2, 10): 500.0},
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        assert rows == [], "No rows must be returned when all demand is assigned"

    def test_row_totals_equal_overall_unassigned(self):
        """Sum of row['unassigned'] must equal total_demand - total_assigned."""
        result = _UnassignedFakeResult(
            demands={1: 400, 2: 300, 3: 200},
            assignments={(1, 10): 250, (2, 10): 100},  # 3 fully unassigned
            od_costs_scenario={(1, 10): 1000.0, (2, 10): 1200.0},
        )
        total_demand = sum(result.demands.values())           # 900
        total_assigned = sum(result.assignments.values())     # 350
        expected_unassigned = total_demand - total_assigned   # 550

        rows = unassigned_rows(result, _NODE_INFO_STUB)
        row_sum = sum(r["unassigned"] for r in rows)
        assert row_sum == expected_unassigned, (
            f"Row total {row_sum} must equal overall unassigned {expected_unassigned}"
        )

    def test_fully_assigned_origins_excluded(self):
        """Origins with zero unassigned demand must not appear in rows."""
        result = _UnassignedFakeResult(
            demands={1: 200, 2: 150},
            assignments={(1, 10): 200, (2, 10): 50},  # origin 1 fully assigned
            od_costs_scenario={(1, 10): 400.0, (2, 10): 500.0},
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        nodes = {r["node"] for r in rows}
        assert 1 not in nodes, "Fully assigned origin 1 must not appear"
        assert 2 in nodes, "Partially unassigned origin 2 must appear"

    def test_name_fallback_when_node_not_in_node_info(self):
        """Origins absent from node_info get a 'Pickup point {node}' fallback name."""
        result = _UnassignedFakeResult(
            demands={99: 100},
            assignments={},
            od_costs_scenario={},
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        assert len(rows) == 1
        assert rows[0]["name"] == "Pickup point 99"

    def test_none_result_returns_empty(self):
        """Passing result=None must return an empty list without raising."""
        assert unassigned_rows(None, _NODE_INFO_STUB) == []

    def test_mixed_reasons_in_same_run(self):
        """A run may have both map-data and capacity reasons for different origins."""
        result = _UnassignedFakeResult(
            demands={1: 400, 2: 300},
            assignments={(1, 10): 100},     # origin 1 partially assigned
            od_costs_scenario={(1, 10): 1000.0},  # origin 2 unreachable
        )
        rows = unassigned_rows(result, _NODE_INFO_STUB)
        reasons = {r["node"]: r["reason"] for r in rows}
        assert reasons[1] == "insufficient reachable capacity"
        assert reasons[2] == "No route found in current map data."


# ---------------------------------------------------------------------------
# classify_unassigned_cause — warning variant classification
# ---------------------------------------------------------------------------


class TestClassifyUnassignedCause:
    """Tests for classify_unassigned_cause() warning variant logic."""

    _DEMANDS_AB = {1: 1284, 2: 436}  # total = 1720
    _DEMANDS_ABC = {1: 1284, 2: 436, 3: 500}  # total = 2220

    def test_topology_only_when_no_origins_reachable(self):
        """All unreachable → case == 'topology'."""
        result = classify_unassigned_cause(1720, set(), self._DEMANDS_AB)
        assert result["case"] == "topology"

    def test_topology_origin_count_correct(self):
        """topology case reports exact number of unreachable origins."""
        result = classify_unassigned_cause(1720, set(), self._DEMANDS_AB)
        assert result["unreachable_origin_count"] == 2

    def test_topology_capacity_demand_is_zero(self):
        """topology case: no capacity-constrained demand."""
        result = classify_unassigned_cause(1720, set(), self._DEMANDS_AB)
        assert result["capacity_demand"] == 0

    def test_capacity_only_when_all_origins_reachable(self):
        """All reachable but unassigned → case == 'capacity'."""
        result = classify_unassigned_cause(500, {1, 2}, self._DEMANDS_AB)
        assert result["case"] == "capacity"

    def test_capacity_only_unreachable_demand_is_zero(self):
        """capacity case: no unreachable demand."""
        result = classify_unassigned_cause(500, {1, 2}, self._DEMANDS_AB)
        assert result["unreachable_demand"] == 0
        assert result["capacity_demand"] == 500

    def test_mixed_case_when_both_causes_present(self):
        """One unreachable origin + capacity shortfall → case == 'mixed'."""
        # origin 1 unreachable (1284 demand), origins 2 and 3 reachable but 300 unassigned
        result = classify_unassigned_cause(1584, {2, 3}, self._DEMANDS_ABC)
        assert result["case"] == "mixed"
        assert result["unreachable_demand"] == 1284
        assert result["capacity_demand"] == 300

    def test_topology_single_origin(self):
        """Single unreachable origin with all demand unassigned → topology."""
        result = classify_unassigned_cause(100, set(), {5: 100})
        assert result["case"] == "topology"
        assert result["unreachable_origin_count"] == 1

    def test_capacity_only_origin_count_is_zero(self):
        """capacity case: no unreachable origins."""
        result = classify_unassigned_cause(200, {1, 2, 3}, {1: 100, 2: 200, 3: 300})
        assert result["case"] == "capacity"
        assert result["unreachable_origin_count"] == 0
