"""Tests for Stage 7 flood-aware optimization core.

Structure
---------
TestEdgeCost            — pure cost function for a single edge
TestMakeWeightFn        — MultiDiGraph weight callable
TestComputeODMatrix     — shortest-path OD matrix on synthetic graphs
TestSolveAssignment     — min-cost flow solver with synthetic OD inputs
TestSolveAssignmentExhaustive — exhaustive comparison for a small 2×2 case
TestRunFloodAssignment  — full pipeline on a tiny synthetic graph
TestSanJoseFeasibility  — controlled scenario on the real PH0600613 RP100 graph

All controlled nodes and demands/capacities in TestSanJoseFeasibility are
explicitly labelled scenario-based (not derived from population data).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from floodroute.optimization.assignment import (
    AssignmentResult,
    run_flood_assignment,
    solve_assignment,
)
from floodroute.optimization.cost import FLOOD_MULTIPLIER, edge_cost, make_weight_fn
from floodroute.optimization.routing import compute_od_matrix

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENRICHED_GRAPHML = (
    _PROJECT_ROOT / "data" / "processed" / "hazard" / "PH0600613_phase_b_enriched.graphml"
)


def _make_multi_di_graph(edges: list[dict]) -> nx.MultiDiGraph:
    """Build a minimal MultiDiGraph from a list of edge dicts.

    Each dict must have ``u``, ``v``, and optionally ``length_m``,
    ``jrc_rp100_status``.  Nodes are created implicitly.
    """
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for e in edges:
        attrs = {k: v for k, v in e.items() if k not in ("u", "v")}
        G.add_edge(e["u"], e["v"], **attrs)
    return G


# ---------------------------------------------------------------------------
# TestEdgeCost
# ---------------------------------------------------------------------------


class TestEdgeCost:
    def test_dry_edge_returns_length(self):
        attrs = {"length_m": 100.0, "jrc_rp100_status": "modelled_dry"}
        assert edge_cost(attrs) == 100.0

    def test_flooded_edge_returns_multiplied_length(self):
        attrs = {"length_m": 100.0, "jrc_rp100_status": "flooded"}
        assert edge_cost(attrs) == pytest.approx(100.0 * FLOOD_MULTIPLIER)

    def test_no_overlap_returns_none(self):
        # no_overlap means no JRC pixel has ≥ MIN_INTERIOR_LEN_M overlap with
        # the edge — NOT a confirmation of dryness.  Must be unavailable.
        attrs = {"length_m": 200.0, "jrc_rp100_status": "no_overlap"}
        assert edge_cost(attrs) is None

    def test_outside_domain_returns_none(self):
        # outside_domain means NODATA pixels — no hydraulic model output.
        # No evidence of safety → unavailable.
        attrs = {"length_m": 50.0, "jrc_rp100_status": "outside_domain"}
        assert edge_cost(attrs) is None

    def test_missing_status_returns_none(self):
        # Key absent entirely → no hydraulic confirmation → unavailable.
        attrs = {"length_m": 300.0}
        assert edge_cost(attrs) is None

    def test_none_status_value_returns_none(self):
        # Key present, value is None (e.g. after GraphML round-trip).
        # Absence of status is not evidence of safety → unavailable.
        attrs = {"length_m": 100.0, "jrc_rp100_status": None}
        assert edge_cost(attrs) is None

    def test_unknown_status_string_returns_none(self):
        # Unrecognised status (e.g. from a future raster version).
        # Conservative policy: unknown → unavailable.
        attrs = {"length_m": 100.0, "jrc_rp100_status": "future_category"}
        assert edge_cost(attrs) is None

    def test_zero_length_dry_returns_zero(self):
        attrs = {"length_m": 0.0, "jrc_rp100_status": "modelled_dry"}
        assert edge_cost(attrs) == 0.0

    def test_zero_length_flooded_returns_zero(self):
        attrs = {"length_m": 0.0, "jrc_rp100_status": "flooded"}
        assert edge_cost(attrs) == 0.0

    def test_none_length_dry_returns_zero(self):
        attrs = {"length_m": None, "jrc_rp100_status": "modelled_dry"}
        assert edge_cost(attrs) == 0.0

    def test_rp10_status_key_used_for_rp10(self):
        attrs = {
            "length_m": 100.0,
            "jrc_rp10_status": "flooded",
            "jrc_rp100_status": "modelled_dry",
        }
        assert edge_cost(attrs, return_period="RP10") == pytest.approx(100.0 * FLOOD_MULTIPLIER)
        assert edge_cost(attrs, return_period="RP100") == 100.0

    # --- Condition 3: explicit status policy ---

    def test_status_sets_are_disjoint(self):
        from floodroute.optimization.cost import (
            _ORDINARY_STATUSES,
            _PENALISED_STATUSES,
            _UNAVAILABLE_STATUSES,
        )

        assert frozenset() == _PENALISED_STATUSES & _ORDINARY_STATUSES
        assert frozenset() == _PENALISED_STATUSES & _UNAVAILABLE_STATUSES
        assert frozenset() == _ORDINARY_STATUSES & _UNAVAILABLE_STATUSES
        assert "flooded" in _PENALISED_STATUSES
        assert "modelled_dry" in _ORDINARY_STATUSES
        assert "no_overlap" in _UNAVAILABLE_STATUSES
        assert "outside_domain" in _UNAVAILABLE_STATUSES


# ---------------------------------------------------------------------------
# TestMakeWeightFn
# ---------------------------------------------------------------------------


class TestMakeWeightFn:
    """make_weight_fn returns a callable for MultiDiGraph Dijkstra."""

    def _d(self, **attrs) -> dict:
        """Wrap attrs as a single-entry MultiDiGraph edge dict {0: attrs}."""
        return {0: attrs}

    def test_dry_edge(self):
        wfn = make_weight_fn("RP100")
        d = self._d(length_m=100.0, jrc_rp100_status="modelled_dry")
        assert wfn(0, 1, d) == 100.0

    def test_flooded_edge(self):
        wfn = make_weight_fn("RP100")
        d = self._d(length_m=100.0, jrc_rp100_status="flooded")
        assert wfn(0, 1, d) == pytest.approx(100.0 * FLOOD_MULTIPLIER)

    def test_picks_minimum_across_parallel_edges(self):
        wfn = make_weight_fn("RP100")
        # Two parallel edges: one dry (100), one flooded (1000)
        d = {
            0: {"length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            1: {"length_m": 100.0, "jrc_rp100_status": "flooded"},
        }
        assert wfn(0, 1, d) == 100.0

    def test_empty_edge_dict_returns_none(self):
        # No parallel edges → no available path → None (edge skipped by Dijkstra).
        wfn = make_weight_fn("RP100")
        assert wfn(0, 1, {}) is None

    # --- Conservative policy: unavailable edges ---

    def test_unavailable_edge_never_selected(self):
        # A graph where the only path goes through a no_overlap edge and an
        # alternative goes through modelled_dry.  Dijkstra must choose the dry
        # path; the no_overlap path must never appear in the result.
        G = _make_multi_di_graph(
            [
                # Direct path via unavailable (no_overlap) edge: 10 m
                {"u": 0, "v": 2, "length_m": 10.0, "jrc_rp100_status": "no_overlap"},
                # Alternative via dry edges: 200 m (longer but available)
                {"u": 0, "v": 1, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 1, "v": 2, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            ]
        )
        costs, routes = compute_od_matrix(G, origins=[0], shelters=[2])
        assert (0, 2) in costs
        # Must use the dry path (cost 200), not the no_overlap shortcut (cost 10)
        assert costs[(0, 2)] == pytest.approx(200.0)
        assert 1 in routes[(0, 2)]  # path goes through intermediate node 1

    def test_valid_parallel_edge_usable_when_other_unavailable(self):
        # Two parallel edges between the same nodes: one no_overlap (unavailable),
        # one modelled_dry (available).  The dry edge must be selected.
        wfn = make_weight_fn("RP100")
        d = {
            0: {"length_m": 50.0, "jrc_rp100_status": "no_overlap"},
            1: {"length_m": 50.0, "jrc_rp100_status": "modelled_dry"},
        }
        assert wfn(0, 1, d) == pytest.approx(50.0)

    def test_all_unavailable_parallel_edges_returns_none(self):
        # Both parallel edges unavailable → weight function returns None →
        # Dijkstra skips this edge entirely.
        wfn = make_weight_fn("RP100")
        d = {
            0: {"length_m": 50.0, "jrc_rp100_status": "no_overlap"},
            1: {"length_m": 80.0, "jrc_rp100_status": "outside_domain"},
        }
        assert wfn(0, 1, d) is None


# ---------------------------------------------------------------------------
# TestComputeODMatrix
# ---------------------------------------------------------------------------


class TestComputeODMatrix:
    """compute_od_matrix on synthetic MultiDiGraphs."""

    def _simple_graph(self) -> nx.MultiDiGraph:
        """Four nodes: 0, 1 are origins; 2, 3 are shelters.

        Edges (all modelled_dry):
          0→2: 100m   0→3: 500m
          1→2: 200m   1→3: 300m
        """
        return _make_multi_di_graph(
            [
                {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 0, "v": 3, "length_m": 500.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 1, "v": 2, "length_m": 200.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 1, "v": 3, "length_m": 300.0, "jrc_rp100_status": "modelled_dry"},
            ]
        )

    def test_all_pairs_reachable(self):
        G = self._simple_graph()
        costs, routes = compute_od_matrix(G, origins=[0, 1], shelters=[2, 3])
        assert set(costs.keys()) == {(0, 2), (0, 3), (1, 2), (1, 3)}

    def test_costs_match_edge_lengths(self):
        G = self._simple_graph()
        costs, _ = compute_od_matrix(G, origins=[0, 1], shelters=[2, 3])
        assert costs[(0, 2)] == pytest.approx(100.0)
        assert costs[(0, 3)] == pytest.approx(500.0)
        assert costs[(1, 2)] == pytest.approx(200.0)
        assert costs[(1, 3)] == pytest.approx(300.0)

    def test_routes_start_and_end_correctly(self):
        G = self._simple_graph()
        _, routes = compute_od_matrix(G, origins=[0, 1], shelters=[2, 3])
        for (o, s), path in routes.items():
            assert path[0] == o
            assert path[-1] == s

    def test_unreachable_shelter_excluded(self):
        # Disconnected graph: origin 0 can reach shelter 2 only.
        G = _make_multi_di_graph(
            [
                {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            ]
        )
        costs, routes = compute_od_matrix(G, origins=[0], shelters=[2, 3])
        assert (0, 2) in costs
        assert (0, 3) not in costs

    def test_flooded_edges_increase_cost(self):
        G = _make_multi_di_graph(
            [
                {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "flooded"},
            ]
        )
        costs, _ = compute_od_matrix(G, origins=[0], shelters=[2])
        assert costs[(0, 2)] == pytest.approx(100.0 * FLOOD_MULTIPLIER)

    def test_origin_not_in_graph_skipped(self):
        G = _make_multi_di_graph(
            [
                {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            ]
        )
        costs, _ = compute_od_matrix(G, origins=[0, 99], shelters=[2])
        assert (0, 2) in costs
        assert all(o != 99 for (o, _) in costs)


# ---------------------------------------------------------------------------
# TestSolveAssignment
# ---------------------------------------------------------------------------


class TestSolveAssignment:
    """solve_assignment with synthetic OD inputs (no graph required)."""

    def _simple_od(self):
        """2 origins, 2 shelters, all pairs reachable."""
        od_costs = {(0, 2): 100.0, (0, 3): 500.0, (1, 2): 200.0, (1, 3): 300.0}
        od_routes = {(0, 2): [0, 2], (0, 3): [0, 3], (1, 2): [1, 2], (1, 3): [1, 3]}
        return od_costs, od_routes

    # --- basic ---

    def test_trivial_single_pair(self):
        costs = {(0, 1): 10.0}
        routes = {(0, 1): [0, 1]}
        result = solve_assignment({0: 5}, {1: 10}, costs, routes)
        assert result.assignments == {(0, 1): 5}
        assert result.total_assigned == 5
        assert result.total_unassigned == 0
        assert result.shelter_loads == {1: 5}
        assert result.capacity_violations == []
        assert result.objective_cost == pytest.approx(50.0)

    def test_zero_demand_returns_empty(self):
        result = solve_assignment({0: 0}, {1: 10}, {}, {})
        assert result.total_assigned == 0
        assert result.total_unassigned == 0
        assert result.assignments == {}

    # --- insufficient capacity ---

    def test_insufficient_capacity_yields_unassigned(self):
        costs = {(0, 1): 10.0}
        routes = {(0, 1): [0, 1]}
        result = solve_assignment({0: 10}, {1: 3}, costs, routes)
        assert result.total_assigned == 3
        assert result.total_unassigned == 7
        assert result.assignments[(0, 1)] == 3
        assert result.shelter_loads[1] == 3
        assert result.capacity_violations == []

    # --- unreachable shelters ---

    def test_no_reachable_shelter_all_unassigned(self):
        # No OD pairs → origin has nowhere to go → all unassigned.
        result = solve_assignment({0: 5}, {1: 10}, {}, {})
        assert result.total_unassigned == 5
        assert result.total_assigned == 0
        assert result.assignments == {}

    def test_one_shelter_unreachable_other_used(self):
        # Only (0→2) is reachable; shelter 3 has no path from origin 0.
        costs = {(0, 2): 50.0}
        routes = {(0, 2): [0, 2]}
        result = solve_assignment({0: 4}, {2: 10, 3: 10}, costs, routes)
        assert result.total_assigned == 4
        assert result.assignments == {(0, 2): 4}
        assert result.assignments.get((0, 3), 0) == 0

    # --- optimality ---

    def test_cheapest_shelter_preferred(self):
        # Origin 0: demand=5.  Shelter A: cheap, cap=5. Shelter B: expensive, cap=5.
        costs = {(0, "A"): 10.0, (0, "B"): 100.0}
        routes = {(0, "A"): [0, "A"], (0, "B"): [0, "B"]}
        result = solve_assignment({0: 5}, {"A": 5, "B": 5}, costs, routes)
        assert result.assignments.get((0, "A"), 0) == 5
        assert result.assignments.get((0, "B"), 0) == 0

    # --- capacity violations ---

    def test_no_capacity_violations_in_valid_result(self):
        od_costs, od_routes = self._simple_od()
        result = solve_assignment({0: 2, 1: 3}, {2: 3, 3: 2}, od_costs, od_routes)
        assert result.capacity_violations == []

    # --- routes ---

    def test_routes_populated_for_assigned_pairs(self):
        costs = {(0, 1): 10.0}
        routes = {(0, 1): [0, "mid", 1]}
        result = solve_assignment({0: 5}, {1: 10}, costs, routes)
        assert result.routes[(0, 1)] == [0, "mid", 1]

    def test_unassigned_pair_has_no_route(self):
        # Only shelter 1 is reachable; shelter 2 is not.
        costs = {(0, 1): 10.0}
        routes = {(0, 1): [0, 1]}
        result = solve_assignment({0: 3}, {1: 10, 2: 10}, costs, routes)
        assert (0, 2) not in result.routes

    # --- validation ---

    def test_node_in_both_demands_and_capacities_raises(self):
        with pytest.raises(ValueError, match="both demands and capacities"):
            solve_assignment({0: 5, 1: 3}, {1: 10}, {(0, 1): 5.0}, {(0, 1): [0, 1]})

    # --- determinism ---

    def test_identical_inputs_produce_identical_outputs(self):
        od_costs, od_routes = self._simple_od()
        demands = {0: 2, 1: 3}
        capacities = {2: 3, 3: 2}
        r1 = solve_assignment(demands, capacities, od_costs, od_routes)
        r2 = solve_assignment(demands, capacities, od_costs, od_routes)
        assert r1.assignments == r2.assignments
        assert r1.objective_cost == pytest.approx(r2.objective_cost)
        assert r1.total_assigned == r2.total_assigned

    # --- Condition 1: lexicographic guarantee ---

    def test_expensive_reachable_path_assigned_before_unassigned(self):
        # A reachable unit must never remain unassigned because its route is
        # expensive.  The DUMMY penalty must exceed any real path cost so that
        # the solver always routes to a real shelter when capacity is available.
        # Here the path cost (999_999.0) is very high but shelter capacity = 10.
        costs = {(0, 1): 999_999.0}
        routes = {(0, 1): [0, 1]}
        result = solve_assignment({0: 1}, {1: 10}, costs, routes)
        assert result.total_assigned == 1
        assert result.total_unassigned == 0
        assert result.assignments[(0, 1)] == 1

    def test_expensive_path_beats_unassigned_even_with_many_origins(self):
        # With multiple origins all sending to the same expensive shelter, the
        # solver must fill all available capacity before leaving any unassigned.
        # Origins are labelled 'o0'..'o4' to avoid colliding with shelter node 1.
        origins = [f"o{i}" for i in range(5)]
        costs = {(o, "S"): 500_000.0 for o in origins}
        routes = {(o, "S"): [o, "S"] for o in origins}
        demands = {o: 2 for o in origins}  # total demand = 10
        result = solve_assignment(demands, {"S": 10}, costs, routes)
        assert result.total_assigned == 10  # all capacity used
        assert result.total_unassigned == 0

    # --- Condition 2: integer scaling for network_simplex ---

    def test_fractional_costs_yield_correct_optimal_and_float_objective(self):
        # OD costs with sub-metre precision.  The solver scales to integers
        # (×_INT_COST_SCALE) before calling network_simplex; the cheaper path
        # (100.7 < 200.3) must still be preferred.  The reported objective_cost
        # must be computed from the original float costs, not the scaled integers.
        costs = {(0, 2): 100.7, (0, 3): 200.3}
        routes = {(0, 2): [0, 2], (0, 3): [0, 3]}
        result = solve_assignment({0: 5}, {2: 3, 3: 5}, costs, routes)
        assert result.assignments.get((0, 2), 0) == 3  # cheaper shelter filled first
        assert result.assignments.get((0, 3), 0) == 2  # remainder in second shelter
        assert result.total_assigned == 5
        # Objective uses original floats, not mm-scaled integers:
        assert result.objective_cost == pytest.approx(3 * 100.7 + 2 * 200.3)


# ---------------------------------------------------------------------------
# TestSolveAssignmentExhaustive — compare against exhaustive enumeration
# ---------------------------------------------------------------------------


class TestSolveAssignmentExhaustive:
    """Compare solver against manually proven optimal for a 2×2 case.

    Problem:
        Origin 0: demand=2    Origin 1: demand=3
        Shelter 2: cap=3      Shelter 3: cap=2

        OD costs:
            (0→2)=100  (0→3)=500
            (1→2)=200  (1→3)=300

    Feasible integer assignments (sum=5, caps respected):
        Case A: f[0,2]=0, f[1,2]=3, f[0,3]=2, f[1,3]=0  → cost=0+600+1000+0=1600
        Case B: f[0,2]=1, f[1,2]=2, f[0,3]=1, f[1,3]=1  → cost=100+400+500+300=1300
                                                            Wait: f[0,3]+f[1,3]=1+1=2 ≤ 2 ✓
                                                                  f[0,2]+f[1,2]=1+2=3 ≤ 3 ✓
        Case C: f[0,2]=2, f[1,2]=1, f[0,3]=0, f[1,3]=2  → cost=200+200+0+600=1000
                                                            f[0,3]+f[1,3]=0+2=2 ≤ 2 ✓
                                                            f[0,2]+f[1,2]=2+1=3 ≤ 3 ✓

    Optimal is Case C with cost=1000.
    Manually proven: dCost/d(f[0,2]) = 100-500 = -400 < 0  →  maximise f[0,2].
    At f[0,2]=2 (demand_0 reached), set f[0,3]=0.
    Remaining: f[1,2]+f[1,3]=3 with f[1,2]≤1 (cap_2 remaining=1), f[1,3]≤2 (cap_3).
    f[1,2]=1, f[1,3]=2.  Cost check: 200+200+600=1000 ✓.
    """

    _OD_COSTS = {(0, 2): 100.0, (0, 3): 500.0, (1, 2): 200.0, (1, 3): 300.0}
    _OD_ROUTES = {(0, 2): [0, 2], (0, 3): [0, 3], (1, 2): [1, 2], (1, 3): [1, 3]}
    _DEMANDS = {0: 2, 1: 3}
    _CAPACITIES = {2: 3, 3: 2}

    def _all_feasible_costs(self) -> list[float]:
        """Enumerate all feasible integer assignments and return their costs."""
        d, c = self._DEMANDS, self._CAPACITIES
        w = self._OD_COSTS
        feasible_costs = []
        for f02 in range(d[0] + 1):
            f03 = d[0] - f02
            for f12 in range(d[1] + 1):
                f13 = d[1] - f12
                # capacity constraints
                if f02 + f12 > c[2]:
                    continue
                if f03 + f13 > c[3]:
                    continue
                cost = f02 * w[(0, 2)] + f03 * w[(0, 3)] + f12 * w[(1, 2)] + f13 * w[(1, 3)]
                feasible_costs.append(cost)
        return feasible_costs

    def test_solver_matches_exhaustive_optimum(self):
        result = solve_assignment(
            self._DEMANDS, self._CAPACITIES, self._OD_COSTS, self._OD_ROUTES
        )
        min_cost = min(self._all_feasible_costs())
        assert result.objective_cost == pytest.approx(min_cost)

    def test_exhaustive_minimum_is_case_c(self):
        assert min(self._all_feasible_costs()) == pytest.approx(1000.0)

    def test_solver_finds_case_c_assignment(self):
        result = solve_assignment(
            self._DEMANDS, self._CAPACITIES, self._OD_COSTS, self._OD_ROUTES
        )
        assert result.assignments.get((0, 2), 0) == 2
        assert result.assignments.get((1, 2), 0) == 1
        assert result.assignments.get((1, 3), 0) == 2
        assert result.assignments.get((0, 3), 0) == 0
        assert result.total_assigned == 5
        assert result.total_unassigned == 0


# ---------------------------------------------------------------------------
# TestRunFloodAssignment — full pipeline on a tiny synthetic graph
# ---------------------------------------------------------------------------


class TestRunFloodAssignment:
    """run_flood_assignment using the same 2-origin, 2-shelter topology."""

    def _graph(self) -> nx.MultiDiGraph:
        """Bidirectional graph: origins={0,1}, shelters={2,3}.

        Edges are added in both directions so Dijkstra can find paths.
        Edge lengths match the OD costs in TestSolveAssignmentExhaustive.
        """
        edges = [
            {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 2, "v": 0, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 0, "v": 3, "length_m": 500.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 3, "v": 0, "length_m": 500.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 1, "v": 2, "length_m": 200.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 2, "v": 1, "length_m": 200.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 1, "v": 3, "length_m": 300.0, "jrc_rp100_status": "modelled_dry"},
            {"u": 3, "v": 1, "length_m": 300.0, "jrc_rp100_status": "modelled_dry"},
        ]
        return _make_multi_di_graph(edges)

    def test_optimal_assignment_matches_exhaustive(self):
        G = self._graph()
        result = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        assert result.objective_cost == pytest.approx(1000.0)
        assert result.total_assigned == 5
        assert result.total_unassigned == 0

    def test_runtime_is_positive(self):
        G = self._graph()
        result = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        assert result.runtime_s > 0.0

    def test_no_capacity_violations(self):
        G = self._graph()
        result = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        assert result.capacity_violations == []

    def test_shelter_loads_within_capacity(self):
        G = self._graph()
        result = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        for s, load in result.shelter_loads.items():
            assert load <= {2: 3, 3: 2}[s]

    def test_flooded_edge_increases_cost(self):
        # Mark the 0→2 edge (cheapest path) as flooded; cost should rise.
        G = _make_multi_di_graph(
            [
                {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "flooded"},
                {"u": 2, "v": 0, "length_m": 100.0, "jrc_rp100_status": "flooded"},
                {"u": 0, "v": 3, "length_m": 500.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 3, "v": 0, "length_m": 500.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 1, "v": 2, "length_m": 200.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 2, "v": 1, "length_m": 200.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 1, "v": 3, "length_m": 300.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 3, "v": 1, "length_m": 300.0, "jrc_rp100_status": "modelled_dry"},
            ]
        )
        dry_result = run_flood_assignment(
            self._graph(), demands={0: 2, 1: 3}, capacities={2: 3, 3: 2}
        )
        flood_result = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        assert flood_result.objective_cost > dry_result.objective_cost

    def test_determinism(self):
        G = self._graph()
        r1 = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        r2 = run_flood_assignment(G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2})
        assert r1.assignments == r2.assignments
        assert r1.objective_cost == pytest.approx(r2.objective_cost)

    def test_disconnected_shelter_is_unassigned(self):
        # Shelter 99 is isolated: no path from any origin.
        G = self._graph()
        G.add_node(99)  # isolated node, no edges
        result = run_flood_assignment(
            G, demands={0: 2, 1: 3}, capacities={2: 3, 3: 2, 99: 100}
        )
        assert result.assignments.get((0, 99), 0) == 0
        assert result.assignments.get((1, 99), 0) == 0

    def test_tie_in_costs_determinism(self):
        # Two shelters with identical cost from origin 0; result must be consistent.
        G = _make_multi_di_graph(
            [
                {"u": 0, "v": 2, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 2, "v": 0, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 0, "v": 3, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
                {"u": 3, "v": 0, "length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
            ]
        )
        r1 = run_flood_assignment(G, demands={0: 4}, capacities={2: 4, 3: 4})
        r2 = run_flood_assignment(G, demands={0: 4}, capacities={2: 4, 3: 4})
        assert r1.assignments == r2.assignments
        assert r1.total_assigned == 4


# ---------------------------------------------------------------------------
# TestSanJoseFeasibility
# ---------------------------------------------------------------------------

_SJDB_AVAILABLE = _ENRICHED_GRAPHML.exists()


@pytest.mark.skipif(
    not _SJDB_AVAILABLE,
    reason="PH0600613 phase-B enriched graph not found — skipping San Jose feasibility test",
)
class TestSanJoseFeasibility:
    """Controlled scenario on the real PH0600613 RP100 graph.

    All node IDs, demands, and capacities below are scenario-based values
    chosen for controlled feasibility verification, NOT derived from PSA
    population data or real shelter records.

    Scenario parameters (scenario_based provenance):
        Return period : RP100
        Origin nodes  : 5 (demand=80), 16 (demand=120), 72 (demand=60)
        Shelter nodes : 33 (capacity=150), 58 (capacity=100)
        Total demand  : 260
        Total capacity: 250
        Expected unassigned: 10  (demand exceeds capacity by 10)

    Conservative hazard policy: no_overlap and outside_domain edges are
    unavailable; only modelled_dry and flooded edges are routable.

    Manually verified optimal assignment (see derivation in docstring):
        (5, 58)  → 70 units   (origin 5 to shelter 58)
        (16, 33) → 120 units  (origin 16 to shelter 33)
        (72, 33) → 30 units   (origin 72 to shelter 33)
        (72, 58) → 30 units   (origin 72 to shelter 58)
        Unassigned: 10 (from origin 5)
        Shelter loads: {33: 150 (full), 58: 100 (full)}
        Approximate objective cost: ≈502,661 m-equivalent

    Derivation (conservative policy OD costs):
        (5,33)=6540.79, (5,58)=4821.23, (16,33)=152.33,
        (16,58)=1527.78, (72,33)=3234.38, (72,58)=1662.13

    Both shelters full → x+a+p=150 and shelter 58 = 100.
    Cost gradient: 1719.56·x − 1375.45·a + 1572.25·p + const
    Minimised by: x=0 (f[5,33]=0), a=120 (f[16,33]=120), p=30 (f[72,33]=30).
    Unassigning from origin 5 (savings ≈48,212/10 units) beats origins 16
    (savings ≈1,523) or 72 (savings ≈16,621).
    """

    # Scenario-based controlled parameters
    _DEMANDS = {5: 80, 16: 120, 72: 60}  # origin_node: demand_units (scenario_based)
    _CAPACITIES = {33: 150, 58: 100}  # shelter_node: capacity_units (scenario_based)
    _TOTAL_DEMAND = 260
    _TOTAL_CAPACITY = 250

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls) -> AssignmentResult:
        G = nx.read_graphml(str(_ENRICHED_GRAPHML), node_type=int)
        return run_flood_assignment(G, cls._DEMANDS, cls._CAPACITIES, return_period="RP100")

    def test_total_demand_accounting(self, result):
        assert result.total_assigned + result.total_unassigned == self._TOTAL_DEMAND

    def test_expected_unassigned(self, result):
        # Exactly 10 units must be unassigned (capacity deficit = 260 - 250).
        assert result.total_unassigned == self._TOTAL_DEMAND - self._TOTAL_CAPACITY

    def test_total_assigned(self, result):
        assert result.total_assigned == self._TOTAL_CAPACITY

    def test_shelter_loads_within_capacity(self, result):
        for s, load in result.shelter_loads.items():
            assert load <= self._CAPACITIES[s], (
                f"Shelter {s}: load {load} exceeds capacity {self._CAPACITIES[s]}"
            )

    def test_both_shelters_at_full_capacity(self, result):
        # With penalty-driven assignment, all 250 capacity units are used.
        assert result.shelter_loads[33] == 150
        assert result.shelter_loads[58] == 100

    def test_no_capacity_violations(self, result):
        assert result.capacity_violations == []

    def test_assignment_flows_sum_to_total_assigned(self, result):
        assert sum(result.assignments.values()) == result.total_assigned

    def test_optimal_assignment_matches_derivation(self, result):
        # Manually proven optimal under conservative hazard policy (see docstring).
        assert result.assignments.get((5, 58), 0) == 70
        assert result.assignments.get((16, 33), 0) == 120
        assert result.assignments.get((72, 33), 0) == 30
        assert result.assignments.get((72, 58), 0) == 30

    def test_unassigned_origin_is_origin_5(self, result):
        # Origin 5 has 80 demand but only 70 assigned → 10 unassigned.
        assigned_from_5 = sum(v for (o, s), v in result.assignments.items() if o == 5)
        assert assigned_from_5 == 70

    def test_objective_cost_approximately_correct(self, result):
        # Value from conservative-policy derivation; allow 1% tolerance for float paths.
        assert result.objective_cost == pytest.approx(502_660.82, rel=0.01)

    def test_routes_populated_for_all_assigned_pairs(self, result):
        for (o, s) in result.assignments:
            assert (o, s) in result.routes, f"No route for assigned pair ({o}, {s})"

    def test_routes_start_at_origin_end_at_shelter(self, result):
        for (o, s), path in result.routes.items():
            assert path[0] == o, f"Route ({o},{s}) starts at {path[0]}, expected {o}"
            assert path[-1] == s, f"Route ({o},{s}) ends at {path[-1]}, expected {s}"

    def test_runtime_is_positive(self, result):
        assert result.runtime_s > 0.0

    def test_determinism(self):
        G = nx.read_graphml(str(_ENRICHED_GRAPHML), node_type=int)
        r1 = run_flood_assignment(G, self._DEMANDS, self._CAPACITIES, return_period="RP100")
        r2 = run_flood_assignment(G, self._DEMANDS, self._CAPACITIES, return_period="RP100")
        assert r1.assignments == r2.assignments
        assert r1.objective_cost == pytest.approx(r2.objective_cost)

    def test_used_edges_have_proper_status(self, result):
        # Conservative policy: all edges on optimal routes must be modelled_dry
        # or flooded.  no_overlap and outside_domain must not appear.
        G = nx.read_graphml(str(_ENRICHED_GRAPHML), node_type=int)
        allowed = {"modelled_dry", "flooded"}
        for (o, s), path in result.routes.items():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                for key, attrs in G[u][v].items():
                    status = attrs.get("jrc_rp100_status")
                    assert status in allowed, (
                        f"Route ({o}→{s}) uses edge ({u},{v},{key}) "
                        f"with unavailable status '{status}'"
                    )
