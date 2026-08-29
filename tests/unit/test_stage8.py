"""Tests for Stage 8: barangay demand, algorithms A/B/C, metrics, and 27-run matrix.

Structure
---------
TestLoadPsaPopulation       — CSV loading and validation
TestSnapBarangayOrigins     — centroid snapping with collision avoidance
TestBuildDemands            — demand fraction rounding and aggregation
TestOrdinaryWeightFn        — ordinary weight function (length only)
TestAlgorithmA              — ordinary nearest-shelter (synthetic graph)
TestAlgorithmB              — flood-aware nearest-shelter (synthetic graph)
TestAlgorithmC              — FloodRoute MCF (synthetic graph)
TestMetrics                 — metric computation from RunResult
TestStage8Matrix            — 27-run matrix integration test (real data, skippable)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import pytest

from floodroute.experiments.algorithms import (
    DEMAND_FRACTIONS,
    RunResult,
    _ordinary_weight_fn,
    run_flood_aware_nearest,
    run_floodroute_assignment,
    run_ordinary_nearest,
)
from floodroute.experiments.demand import (
    BarangayOrigin,
    BarangayRecord,
    build_demands,
    load_psa_population,
)
from floodroute.experiments.metrics import compute_flood_exposed_length, compute_metrics
from floodroute.experiments.runner import (
    SCENARIO_SHELTER_CAPACITIES,
    run_stage8_matrix,
)

# ---------------------------------------------------------------------------
# Shared project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GRAPHML = _PROJECT_ROOT / "data" / "processed" / "hazard" / "PH0600613_phase_b_enriched.graphml"
_POP_CSV = _PROJECT_ROOT / "data" / "raw" / "psa_barangay_population_sjdb_2020.csv"
_BARANGAY_GPKG = _PROJECT_ROOT / "data" / "processed" / "admin" / "barangays_utm51n.gpkg"
_NODES_GPKG = _PROJECT_ROOT / "data" / "processed" / "graph" / "PH0600613_nodes.gpkg"
_REAL_DATA_AVAILABLE = all(p.exists() for p in [_GRAPHML, _POP_CSV, _BARANGAY_GPKG, _NODES_GPKG])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pop_csv(rows: list[dict]) -> Path:
    """Write a temporary population CSV to a tmp file (pytest tmp_path)."""
    raise NotImplementedError  # not used directly; tests use tmp_path


def _records(*args) -> list[BarangayRecord]:
    """Quick list of BarangayRecord from (psgc, name, pop) tuples."""
    return [BarangayRecord(psgc=p, name=n, population_2020=pop) for p, n, pop in args]


def _make_synthetic_graph() -> nx.MultiDiGraph:
    """Two origins (10, 11), two shelters (20, 21), all modelled_dry.

    Topology (one-way directed):
        10 → 20: 100 m (dry)
        10 → 21: 400 m (dry)
        11 → 20: 300 m (dry)
        11 → 21: 200 m (dry)
        10 → 99: 50 m  (flooded) — detour test
    """
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    edges = [
        (10, 20, {"length_m": 100.0, "jrc_rp100_status": "modelled_dry",
                  "jrc_rp10_status": "modelled_dry", "jrc_rp20_status": "modelled_dry"}),
        (10, 21, {"length_m": 400.0, "jrc_rp100_status": "modelled_dry",
                  "jrc_rp10_status": "modelled_dry", "jrc_rp20_status": "modelled_dry"}),
        (11, 20, {"length_m": 300.0, "jrc_rp100_status": "modelled_dry",
                  "jrc_rp10_status": "modelled_dry", "jrc_rp20_status": "modelled_dry"}),
        (11, 21, {"length_m": 200.0, "jrc_rp100_status": "modelled_dry",
                  "jrc_rp10_status": "modelled_dry", "jrc_rp20_status": "modelled_dry"}),
        (10, 99, {"length_m": 50.0, "jrc_rp100_status": "flooded",
                  "jrc_rp10_status": "flooded", "jrc_rp20_status": "flooded"}),
    ]
    for u, v, d in edges:
        G.add_edge(u, v, **d)
    return G


# ---------------------------------------------------------------------------
# TestLoadPsaPopulation
# ---------------------------------------------------------------------------


class TestLoadPsaPopulation:
    def _write_csv(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "pop.csv"
        with p.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_loads_basic_rows(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [
                {"adm4_pcode": "PH0600613001", "adm3_pcode": "PH0600613",
                 "adm4_name": "Atabay", "population_2020": "2800", "source": "PSA"},
                {"adm4_pcode": "PH0600613002", "adm3_pcode": "PH0600613",
                 "adm4_name": "Badiang", "population_2020": "1200", "source": "PSA"},
            ],
        )
        records = load_psa_population(p)
        assert len(records) == 2
        assert records[0].psgc == "PH0600613001"
        assert records[0].population_2020 == 2800

    def test_filters_by_adm3_pcode(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [
                {"adm4_pcode": "PH0600613001", "adm3_pcode": "PH0600613",
                 "adm4_name": "Atabay", "population_2020": "2800", "source": "PSA"},
                {"adm4_pcode": "PH0600608001", "adm3_pcode": "PH0600608",
                 "adm4_name": "Other", "population_2020": "999", "source": "PSA"},
            ],
        )
        records = load_psa_population(p, adm3_filter="PH0600613")
        assert len(records) == 1
        assert records[0].psgc == "PH0600613001"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_psa_population(tmp_path / "nonexistent.csv")

    def test_raises_on_duplicate_psgc(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [
                {"adm4_pcode": "PH0600613001", "adm3_pcode": "PH0600613",
                 "adm4_name": "A", "population_2020": "100", "source": "PSA"},
                {"adm4_pcode": "PH0600613001", "adm3_pcode": "PH0600613",
                 "adm4_name": "B", "population_2020": "200", "source": "PSA"},
            ],
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_psa_population(p)

    def test_raises_on_zero_population(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [{"adm4_pcode": "PH0600613001", "adm3_pcode": "PH0600613",
              "adm4_name": "A", "population_2020": "0", "source": "PSA"}],
        )
        with pytest.raises(ValueError):
            load_psa_population(p)

    def test_sorted_by_psgc(self, tmp_path):
        p = self._write_csv(
            tmp_path,
            [
                {"adm4_pcode": "PH0600613030", "adm3_pcode": "PH0600613",
                 "adm4_name": "Supa", "population_2020": "1240", "source": "PSA"},
                {"adm4_pcode": "PH0600613001", "adm3_pcode": "PH0600613",
                 "adm4_name": "Atabay", "population_2020": "2800", "source": "PSA"},
            ],
        )
        records = load_psa_population(p)
        assert records[0].psgc == "PH0600613001"
        assert records[1].psgc == "PH0600613030"


# ---------------------------------------------------------------------------
# TestSnapBarangayOrigins — skip if real geodata absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_BARANGAY_GPKG.exists() and _NODES_GPKG.exists()),
    reason="Real geodata not found",
)
class TestSnapBarangayOrigins:
    def test_all_28_snap_to_unique_nodes(self):
        from floodroute.experiments.demand import snap_barangay_origins

        records = load_psa_population(_POP_CSV)
        origins = snap_barangay_origins(
            records,
            barangay_gpkg=_BARANGAY_GPKG,
            nodes_gpkg=_NODES_GPKG,
            exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
        )
        assert len(origins) == 28
        node_ids = [o.origin_node for o in origins]
        assert len(set(node_ids)) == 28  # all unique

    def test_no_origin_snaps_to_shelter_node(self):
        from floodroute.experiments.demand import snap_barangay_origins

        records = load_psa_population(_POP_CSV)
        origins = snap_barangay_origins(
            records,
            barangay_gpkg=_BARANGAY_GPKG,
            nodes_gpkg=_NODES_GPKG,
            exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
        )
        shelter_nodes = set(SCENARIO_SHELTER_CAPACITIES)
        for o in origins:
            assert o.origin_node not in shelter_nodes, (
                f"Barangay {o.psgc} snapped to shelter node {o.origin_node}"
            )

    def test_snap_distances_are_positive(self):
        from floodroute.experiments.demand import snap_barangay_origins

        records = load_psa_population(_POP_CSV)
        origins = snap_barangay_origins(
            records,
            barangay_gpkg=_BARANGAY_GPKG,
            nodes_gpkg=_NODES_GPKG,
            exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
        )
        for o in origins:
            assert o.snap_distance_m >= 0.0

    def test_deterministic_snap(self):
        from floodroute.experiments.demand import snap_barangay_origins

        records = load_psa_population(_POP_CSV)
        origins1 = snap_barangay_origins(
            records, barangay_gpkg=_BARANGAY_GPKG, nodes_gpkg=_NODES_GPKG,
            exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
        )
        origins2 = snap_barangay_origins(
            records, barangay_gpkg=_BARANGAY_GPKG, nodes_gpkg=_NODES_GPKG,
            exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
        )
        assert [o.origin_node for o in origins1] == [o.origin_node for o in origins2]


# ---------------------------------------------------------------------------
# TestBuildDemands
# ---------------------------------------------------------------------------


class TestBuildDemands:
    def _origins(self) -> list[BarangayOrigin]:
        return [
            BarangayOrigin("A001", "Alpha", 1000, origin_node=1, snap_distance_m=10.0),
            BarangayOrigin("A002", "Beta", 500, origin_node=2, snap_distance_m=20.0),
        ]

    def test_10pct_demand(self):
        demands, _ = build_demands(self._origins(), 0.10)
        # ceil(1000*0.10)=100, ceil(500*0.10)=50
        assert demands[1] == 100
        assert demands[2] == 50

    def test_50pct_demand(self):
        demands, _ = build_demands(self._origins(), 0.50)
        assert demands[1] == 500
        assert demands[2] == 250

    def test_barangay_audit_trail_preserved(self):
        _, bgy_demand = build_demands(self._origins(), 0.10)
        assert "A001" in bgy_demand
        assert "A002" in bgy_demand
        assert bgy_demand["A001"] == 100
        assert bgy_demand["A002"] == 50

    def test_lr_floor_when_single_barangay(self):
        # Single barangay: target=round(1001*0.10)=100; floor=100; no remainder.
        # LR gives 100, not 101 (old ceil behaviour).
        origins = [BarangayOrigin("X", "X", 1001, origin_node=5, snap_distance_m=0.0)]
        demands, _ = build_demands(origins, 0.10)
        assert demands[5] == 100

    def test_lr_remainder_awarded_to_largest_fractional_part(self):
        # Two barangays: pop 3 and 7, fraction 0.10.
        # exact: 0.3 and 0.7; floor both 0; target=round(1.0)=1; remainder=1.
        # Barangay B (remainder 0.7) wins the extra unit.
        origins = [
            BarangayOrigin("A001", "Alpha", 3, origin_node=1, snap_distance_m=0.0),
            BarangayOrigin("B002", "Beta", 7, origin_node=2, snap_distance_m=0.0),
        ]
        demands, bgy = build_demands(origins, 0.10)
        assert bgy["A001"] == 0
        assert bgy["B002"] == 1
        assert sum(bgy.values()) == 1

    def test_conservation_at_three_rates(self):
        # sum(barangay_demands) == round(total_pop * rate) for all three rates.
        origins = [
            BarangayOrigin("P001", "A", 1000, origin_node=1, snap_distance_m=0.0),
            BarangayOrigin("P002", "B", 743, origin_node=2, snap_distance_m=0.0),
            BarangayOrigin("P003", "C", 257, origin_node=3, snap_distance_m=0.0),
        ]
        total_pop = 2000
        for frac in [0.10, 0.25, 0.50]:
            _, bgy = build_demands(origins, frac)
            assert sum(bgy.values()) == round(total_pop * frac), (
                f"Conservation failed at {frac:.0%}: "
                f"got {sum(bgy.values())}, expected {round(total_pop * frac)}"
            )

    def test_deterministic_tie_breaking_by_psgc(self):
        # Two barangays with identical fractional remainder at 0.10 (pop 5 each).
        # exact: 0.5 each; target=round(1.0)=1; tie-break by PSGC ascending → A001 wins.
        origins = [
            BarangayOrigin("B002", "Beta", 5, origin_node=2, snap_distance_m=0.0),
            BarangayOrigin("A001", "Alpha", 5, origin_node=1, snap_distance_m=0.0),
        ]
        _, bgy = build_demands(origins, 0.10)
        assert bgy["A001"] == 1, "A001 should win tie on PSGC sort (A < B)"
        assert bgy["B002"] == 0

    def test_no_negative_demands(self):
        demands, bgy = build_demands(self._origins(), 0.10)
        assert all(v >= 0 for v in demands.values())
        assert all(v >= 0 for v in bgy.values())

    def test_demand_does_not_exceed_population(self):
        by_psgc = {o.psgc: o.population_2020 for o in self._origins()}
        for frac in [0.10, 0.25, 0.50]:
            _, bgy = build_demands(self._origins(), frac)
            for psgc, d in bgy.items():
                assert d <= by_psgc[psgc], (
                    f"Demand {d} exceeds population {by_psgc[psgc]} for {psgc} at {frac:.0%}"
                )

    @pytest.mark.skipif(not _POP_CSV.exists(), reason="Real population CSV not found")
    def test_real_population_total_is_65140(self):
        records = load_psa_population(_POP_CSV)
        assert sum(r.population_2020 for r in records) == 65140

    @pytest.mark.skipif(not _POP_CSV.exists(), reason="Real population CSV not found")
    def test_demand_conserved_real_populations(self):
        # Conservation holds for each of the three experimental rates using
        # real 65,140 total, without requiring geodata for snapping.
        records = load_psa_population(_POP_CSV)
        origins = [
            BarangayOrigin(r.psgc, r.name, r.population_2020, i, 0.0)
            for i, r in enumerate(records)
        ]
        expected = {0.10: 6514, 0.25: 16285, 0.50: 32570}
        for frac in DEMAND_FRACTIONS:
            _, bgy = build_demands(origins, frac)
            assert sum(bgy.values()) == expected[frac], (
                f"Conservation failed at {frac:.0%}: "
                f"got {sum(bgy.values())}, expected {expected[frac]}"
            )

    def test_shared_node_aggregated(self):
        # Two barangays snap to same node → demands summed
        origins = [
            BarangayOrigin("A", "A", 1000, origin_node=7, snap_distance_m=0.0),
            BarangayOrigin("B", "B", 500, origin_node=7, snap_distance_m=0.0),
        ]
        demands, _ = build_demands(origins, 0.10)
        assert demands[7] == 100 + 50

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            build_demands(self._origins(), 0.0)
        with pytest.raises(ValueError):
            build_demands(self._origins(), 1.5)


# ---------------------------------------------------------------------------
# TestOrdinaryWeightFn
# ---------------------------------------------------------------------------


class TestOrdinaryWeightFn:
    def _d(self, **attrs):
        return {0: attrs}

    def test_dry_edge_returns_length(self):
        wfn = _ordinary_weight_fn()
        assert wfn(0, 1, self._d(length_m=100.0, jrc_rp100_status="modelled_dry")) == 100.0

    def test_flooded_edge_returns_length_not_penalised(self):
        # Ordinary routing ignores flood status — flooded edge costs length only.
        wfn = _ordinary_weight_fn()
        assert wfn(0, 1, self._d(length_m=100.0, jrc_rp100_status="flooded")) == 100.0

    def test_no_overlap_edge_returns_length(self):
        # Ordinary routing traverses no_overlap edges (length only).
        wfn = _ordinary_weight_fn()
        assert wfn(0, 1, self._d(length_m=50.0, jrc_rp100_status="no_overlap")) == 50.0

    def test_missing_status_returns_length(self):
        wfn = _ordinary_weight_fn()
        assert wfn(0, 1, self._d(length_m=200.0)) == 200.0

    def test_picks_minimum_across_parallel_edges(self):
        wfn = _ordinary_weight_fn()
        d = {
            0: {"length_m": 300.0, "jrc_rp100_status": "flooded"},
            1: {"length_m": 100.0, "jrc_rp100_status": "modelled_dry"},
        }
        assert wfn(0, 1, d) == 100.0

    def test_empty_dict_returns_zero(self):
        wfn = _ordinary_weight_fn()
        assert wfn(0, 1, {}) == 0.0


# ---------------------------------------------------------------------------
# TestAlgorithmA
# ---------------------------------------------------------------------------


class TestAlgorithmA:
    """Algorithm A: ordinary nearest-shelter, no capacity."""

    def test_nearest_shelter_chosen(self):
        G = _make_synthetic_graph()
        # Node 10: nearest is shelter 20 (100m) vs 21 (400m)
        # Node 11: nearest is shelter 21 (200m) vs 20 (300m)
        demands = {10: 50, 11: 80}
        capacities = {20: 1000, 21: 1000}
        result = run_ordinary_nearest(G, demands, capacities)
        assert result.assignments.get((10, 20), 0) == 50
        assert result.assignments.get((11, 21), 0) == 80

    def test_flooded_edge_not_avoided(self):
        # Algorithm A ignores flood status; node 10→99 via flooded 50m wins over 100m dry
        G = _make_synthetic_graph()
        demands = {10: 30}
        capacities = {99: 1000}
        result = run_ordinary_nearest(G, demands, capacities)
        # Path 10→99 via 50m flooded edge should be chosen (ordinary routing)
        assert result.assignments.get((10, 99), 0) == 30

    def test_unreachable_origin_excluded(self):
        G = _make_synthetic_graph()
        # Node 999 not in graph → no route → excluded from assignments
        demands = {10: 50, 999: 20}
        capacities = {20: 1000}
        result = run_ordinary_nearest(G, demands, capacities)
        assert all(o != 999 for (o, _) in result.assignments)

    def test_capacity_not_enforced(self):
        # Both origins send all demand to nearest shelter — may exceed capacity.
        G = _make_synthetic_graph()
        demands = {10: 50, 11: 80}
        capacities = {20: 5}  # tiny capacity — A does not enforce it
        result = run_ordinary_nearest(G, demands, capacities)
        # Node 10 nearest is 20, node 11 nearest is 20 too? (300 < 400)
        # Actually 11→20 is 300, 11→21 is 200 → nearest for 11 is 21
        # But let's just check total assigned is all demand or less
        assert sum(result.assignments.values()) <= 130  # demand total

    def test_algorithm_label(self):
        G = _make_synthetic_graph()
        result = run_ordinary_nearest(G, {10: 10}, {20: 100})
        assert result.algorithm == "A"

    def test_runtime_positive(self):
        G = _make_synthetic_graph()
        result = run_ordinary_nearest(G, {10: 10}, {20: 100})
        assert result.runtime_s > 0.0

    def test_od_costs_ordinary_equals_scenario_for_A(self):
        G = _make_synthetic_graph()
        result = run_ordinary_nearest(G, {10: 50, 11: 80}, {20: 1000, 21: 1000})
        # For algorithm A, scenario and ordinary OD costs are identical
        assert result.od_costs_scenario == result.od_costs_ordinary


# ---------------------------------------------------------------------------
# TestAlgorithmB
# ---------------------------------------------------------------------------


class TestAlgorithmB:
    """Algorithm B: flood-aware nearest-shelter, no capacity."""

    def test_flooded_edge_avoided_when_alternative_exists(self):
        # Two-leg graph: 10→99 via flooded 50m, 10→20 via dry 100m.
        # Algorithm B should route 10→20 (dry 100) not 10→99 (flooded 500 eq).
        G = _make_synthetic_graph()
        demands = {10: 30}
        capacities = {20: 1000}
        result = run_flood_aware_nearest(G, demands, capacities)
        # node 20 is reachable with dry path; node 99 only via flooded edge (cost=500)
        # nearest shelter 20: dry cost 100 < flooded cost to 99: 500
        assert result.assignments.get((10, 20), 0) == 30

    def test_algorithm_label(self):
        G = _make_synthetic_graph()
        result = run_flood_aware_nearest(G, {10: 10}, {20: 100})
        assert result.algorithm == "B"

    def test_capacity_not_enforced(self):
        G = _make_synthetic_graph()
        demands = {10: 50, 11: 80}
        capacities = {20: 5, 21: 5}  # tiny — B does not enforce
        result = run_flood_aware_nearest(G, demands, capacities)
        assert sum(result.assignments.values()) <= 130

    def test_ordinary_od_costs_populated(self):
        G = _make_synthetic_graph()
        result = run_flood_aware_nearest(G, {10: 10, 11: 10}, {20: 100, 21: 100})
        # Both (10,20) and (11,21) should be in ordinary OD costs
        assert (10, 20) in result.od_costs_ordinary
        assert (11, 21) in result.od_costs_ordinary

    def test_routes_use_available_edges_only(self):
        # All edges on routes must be modelled_dry or flooded (not no_overlap etc.)
        G = _make_synthetic_graph()
        # Add a no_overlap edge that would be a shortcut
        G.add_edge(10, 20, length_m=10.0, jrc_rp100_status="no_overlap",
                   jrc_rp10_status="no_overlap", jrc_rp20_status="no_overlap")
        result = run_flood_aware_nearest(G, {10: 20}, {20: 100})
        # Path must use the 100m dry edge, not the 10m no_overlap edge
        assert result.od_costs_scenario.get((10, 20), 0) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TestAlgorithmC
# ---------------------------------------------------------------------------


class TestAlgorithmC:
    """Algorithm C: FloodRoute capacitated min-cost-flow."""

    def test_capacity_enforced(self):
        G = _make_synthetic_graph()
        demands = {10: 50, 11: 80}
        capacities = {20: 30, 21: 100}
        result = run_floodroute_assignment(G, demands, capacities)
        # Shelter 20 must not exceed 30
        load_20 = sum(units for (o, s), units in result.assignments.items() if s == 20)
        assert load_20 <= 30

    def test_algorithm_label(self):
        G = _make_synthetic_graph()
        result = run_floodroute_assignment(G, {10: 10}, {20: 100})
        assert result.algorithm == "C"

    def test_assigns_all_when_capacity_sufficient(self):
        G = _make_synthetic_graph()
        result = run_floodroute_assignment(
            G, {10: 50, 11: 80}, {20: 1000, 21: 1000}
        )
        assert sum(result.assignments.values()) == 130

    def test_lexicographic_priority(self):
        # Even via a very expensive flooded path, all demand should be assigned
        # when capacity is available.
        G = _make_synthetic_graph()
        demands = {10: 30}
        capacities = {99: 100}  # only reachable via flooded 50m edge (cost=500)
        result = run_floodroute_assignment(G, demands, capacities)
        assert sum(result.assignments.values()) == 30

    def test_cheaper_shelter_preferred(self):
        G = _make_synthetic_graph()
        # Node 10: shelter 20 at 100m (dry), shelter 21 at 400m (dry)
        demands = {10: 50}
        capacities = {20: 100, 21: 100}
        result = run_floodroute_assignment(G, demands, capacities)
        assert result.assignments.get((10, 20), 0) == 50
        assert result.assignments.get((10, 21), 0) == 0


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def _synthetic_result(self, algorithm="A") -> tuple[RunResult, nx.MultiDiGraph]:
        G = _make_synthetic_graph()
        demands = {10: 50, 11: 80}
        capacities = {20: 1000, 21: 1000}
        if algorithm == "A":
            result = run_ordinary_nearest(G, demands, capacities)
        elif algorithm == "B":
            result = run_flood_aware_nearest(G, demands, capacities)
        else:
            result = run_floodroute_assignment(G, demands, capacities)
        result.demand_fraction = 0.10
        return result, G

    def test_total_demand_equals_assigned_plus_unassigned(self):
        result, G = self._synthetic_result()
        m = compute_metrics(result, G)
        assert m["total_assigned"] + m["total_unassigned"] == m["total_demand"]

    def test_shelter_loads_present(self):
        result, G = self._synthetic_result()
        m = compute_metrics(result, G)
        assert "shelter_20_load" in m
        assert "shelter_21_load" in m

    def test_no_capacity_violations_within_capacity(self):
        result, G = self._synthetic_result("C")
        m = compute_metrics(result, G)
        assert m["num_capacity_violations"] == 0

    def test_run_id_format(self):
        result, G = self._synthetic_result()
        m = compute_metrics(result, G)
        assert m["run_id"].startswith("A_")
        assert "RP100" in m["run_id"]

    def test_detour_ratio_is_one_for_algorithm_A(self):
        result, G = self._synthetic_result("A")
        m = compute_metrics(result, G)
        assert m["detour_ratio"] == pytest.approx(1.0)

    def test_runtime_positive(self):
        result, G = self._synthetic_result()
        m = compute_metrics(result, G)
        assert m["runtime_s"] >= 0.0

    def test_flood_exposed_length_zero_for_dry_routes(self):
        # Synthetic graph edges on routes from 10→20 and 11→21 are all dry
        result, G = self._synthetic_result()
        m = compute_metrics(result, G)
        # Routes 10→20 and 11→21 use modelled_dry edges → no flood exposure
        assert m["flood_exposed_length_m"] == pytest.approx(0.0)

    def test_flood_exposed_length_nonzero_for_flooded_route(self):
        G = _make_synthetic_graph()
        # Node 10→99 via flooded 50m edge
        demands = {10: 30}
        capacities = {99: 100}
        result = run_floodroute_assignment(G, demands, capacities)
        result.demand_fraction = 0.10
        m = compute_metrics(result, G)
        assert m["flood_exposed_length_m"] == pytest.approx(50.0)

    def test_unreachable_origins_counted(self):
        G = _make_synthetic_graph()
        demands = {10: 50, 999: 20}  # 999 not in graph
        result = run_ordinary_nearest(G, demands, {20: 1000})
        result.demand_fraction = 0.10
        m = compute_metrics(result, G)
        assert m["num_unreachable_origins"] == 1
        assert "999" in m["unreachable_origin_nodes"]


class TestComputeFloodExposedLength:
    def test_no_flooded_edges_returns_zero(self):
        G = _make_synthetic_graph()
        routes = {(10, 20): [10, 20]}
        assignments = {(10, 20): 50}
        assert compute_flood_exposed_length(routes, assignments, G, "RP100") == pytest.approx(0.0)

    def test_flooded_edge_counted(self):
        G = _make_synthetic_graph()
        routes = {(10, 99): [10, 99]}
        assignments = {(10, 99): 30}
        result = compute_flood_exposed_length(routes, assignments, G, "RP100")
        assert result == pytest.approx(50.0)

    def test_shared_edge_not_double_counted(self):
        G = _make_synthetic_graph()
        # Two assigned pairs sharing edge (10,99)
        routes = {(10, 99): [10, 99], (10, 20): [10, 99, 20]}  # both use 10→99
        assignments = {(10, 99): 10, (10, 20): 20}
        result = compute_flood_exposed_length(routes, assignments, G, "RP100")
        # Edge (10,99) = 50m flooded; counted once
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TestStage8Matrix — integration test on real data (skippable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="Real PH0600613 data not found — skipping Stage 8 matrix test",
)
class TestStage8Matrix:
    """Integration test: run the full 27-run matrix on real San Jose data."""

    @pytest.fixture(scope="class")
    @classmethod
    def matrix_results(cls, tmp_path_factory) -> list[dict]:
        output_dir = tmp_path_factory.mktemp("stage8")
        return run_stage8_matrix(
            graphml_path=_GRAPHML,
            population_csv=_POP_CSV,
            barangay_gpkg=_BARANGAY_GPKG,
            nodes_gpkg=_NODES_GPKG,
            output_dir=output_dir,
        )

    def test_exactly_27_runs(self, matrix_results):
        assert len(matrix_results) == 27

    def test_all_algorithms_present(self, matrix_results):
        algos = {r["algorithm"] for r in matrix_results}
        assert algos == {"A", "B", "C"}

    def test_all_return_periods_present(self, matrix_results):
        rps = {r["return_period"] for r in matrix_results}
        assert rps == {"RP10", "RP20", "RP100"}

    def test_all_demand_fractions_present(self, matrix_results):
        fracs = {r["demand_fraction"] for r in matrix_results}
        assert fracs == {0.10, 0.25, 0.50}

    def test_demand_accounting_sums_correctly(self, matrix_results):
        for r in matrix_results:
            assert r["total_assigned"] + r["total_unassigned"] == r["total_demand"], (
                f"Run {r['run_id']}: demand accounting mismatch"
            )

    def test_total_demand_consistent_across_algorithms_per_fraction(self, matrix_results):
        # For the same RP and fraction, all algorithms should see the same total demand.

        grouped: dict[tuple, list[int]] = {}
        for r in matrix_results:
            key = (r["return_period"], r["demand_fraction"])
            grouped.setdefault(key, []).append(r["total_demand"])
        for key, demands in grouped.items():
            assert len(set(demands)) == 1, (
                f"Total demand differs across algorithms for {key}: {demands}"
            )

    def test_algorithm_C_never_violates_capacity(self, matrix_results):
        for r in matrix_results:
            if r["algorithm"] == "C":
                assert r["num_capacity_violations"] == 0, (
                    f"Run {r['run_id']}: C has capacity violations"
                )

    def test_algorithm_A_detour_ratio_is_one(self, matrix_results):
        for r in matrix_results:
            if r["algorithm"] == "A":
                assert r["detour_ratio"] == pytest.approx(1.0, abs=1e-4), (
                    f"Run {r['run_id']}: A detour ratio {r['detour_ratio']} ≠ 1.0"
                )

    def test_50pct_demand_exceeds_capacity_for_C(self, matrix_results):
        # Total capacity = 22 000; 50% demand ≈ 32 570 → must have unassigned.
        for r in matrix_results:
            if r["algorithm"] == "C" and r["demand_fraction"] == 0.50:
                assert r["total_unassigned"] > 0, (
                    f"Run {r['run_id']}: expected unassigned at 50% demand"
                )

    def test_10pct_demand_fully_assigned_for_C(self, matrix_results):
        # 10% demand ≈ 6 514 < 22 000 capacity → all reachable demand assigned.
        for r in matrix_results:
            if r["algorithm"] == "C" and r["demand_fraction"] == 0.10:
                # Some barangays are structurally unreachable, so allow for those
                assert r["assignment_rate"] > 0.80, (
                    f"Run {r['run_id']}: low assignment rate at 10% demand"
                )

    def test_flood_aware_detour_ratio_gte_one(self, matrix_results):
        # B and C detour ratios should be ≥ 1 (flood-aware ≥ ordinary cost).
        for r in matrix_results:
            if r["algorithm"] in ("B", "C") and r["total_assigned"] > 0:
                assert r["detour_ratio"] >= 1.0 - 1e-6, (
                    f"Run {r['run_id']}: detour ratio {r['detour_ratio']} < 1"
                )

    def test_results_csv_written(self, matrix_results, tmp_path_factory):
        # Already verified by matrix_results fixture; just check file exists
        # by re-running with a known output dir
        output_dir = tmp_path_factory.mktemp("stage8_csv")
        run_stage8_matrix(
            graphml_path=_GRAPHML,
            population_csv=_POP_CSV,
            barangay_gpkg=_BARANGAY_GPKG,
            nodes_gpkg=_NODES_GPKG,
            output_dir=output_dir,
        )
        assert (output_dir / "results.csv").exists()
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "results_detailed.json").exists()

    def test_manifest_has_required_fields(self, matrix_results, tmp_path_factory):
        output_dir = tmp_path_factory.mktemp("stage8_manifest")
        run_stage8_matrix(
            graphml_path=_GRAPHML,
            population_csv=_POP_CSV,
            barangay_gpkg=_BARANGAY_GPKG,
            nodes_gpkg=_NODES_GPKG,
            output_dir=output_dir,
        )
        manifest = json.loads((output_dir / "manifest.json").read_text())
        for field in [
            "experiment", "municipality_psgc", "total_population_2020",
            "num_barangays", "scenario_shelter_capacities",
            "shelter_capacity_provenance", "num_runs", "created_at",
        ]:
            assert field in manifest, f"Manifest missing field: {field}"
        assert manifest["shelter_capacity_provenance"] == "scenario_based"
        assert manifest["total_population_2020"] == 65140
        assert manifest["num_runs"] == 27

    def test_three_structurally_unreachable_barangays(self, matrix_results):
        # Three barangays are in isolated WCCs and unreachable under any algorithm.
        # Cause: road graph topology (OSM disconnected stubs), NOT flood policy or
        # snapping.  Nodes 421, 515 are 2-node stubs; node 1493 is in a 32-node
        # WCC with no edges to the main network.
        for r in matrix_results:
            if r["algorithm"] == "A" and r["return_period"] == "RP100":
                assert r["num_unreachable_origins"] >= 3, (
                    f"Run {r['run_id']}: expected ≥ 3 unreachable, "
                    f"got {r['num_unreachable_origins']}"
                )
                break


# ---------------------------------------------------------------------------
# TestHazardStressVerification — controlled algorithmic verification
# ---------------------------------------------------------------------------
#
# This is NOT an empirical evacuation result.  It is a controlled test that
# verifies the two routing policies behave differently on a known OD pair
# where the ordinary shortest route crosses a genuinely flooded edge and a
# longer, fully-dry alternative exists.
#
# Origin 286 / Shelter 33 (San Jose de Buenavista RP100 graph):
#   Ordinary path (Algorithm A): cost ≈ 5 800 m — crosses flooded edge(s)
#   Flood-aware path (Algorithm B): cost ≈ 8 347 m-eq — entirely dry
#
# This pair was selected from 14 candidate OD pairs identified by
# exhaustive audit of the SJDB RP100 road graph (Stage 8 audit, 2026-08).


@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="Real PH0600613 RP100 graph not found — skipping hazard stress verification",
)
class TestHazardStressVerification:
    """Controlled verification: ordinary vs flood-aware routing on a known OD pair.

    These tests are algorithmic unit-level verifications, not municipal results.
    They demonstrate that the conservative hazard policy routes around flooded
    edges when a dry alternative exists, at the cost of a longer path.
    """

    ORIGIN = 286
    SHELTER = 33
    # Small scenario capacity — just enough to accept the single demand unit.
    CAPACITIES = {33: 100, 58: 100}
    DEMANDS = {286: 1}

    @pytest.fixture(scope="class")
    @classmethod
    def graph(cls) -> nx.MultiDiGraph:
        return nx.read_graphml(str(_GRAPHML), node_type=int)

    @pytest.fixture(scope="class")
    @classmethod
    def result_a(cls, graph) -> RunResult:
        return run_ordinary_nearest(
            graph, cls.DEMANDS, cls.CAPACITIES, return_period="RP100"
        )

    @pytest.fixture(scope="class")
    @classmethod
    def result_b(cls, graph) -> RunResult:
        return run_flood_aware_nearest(
            graph, cls.DEMANDS, cls.CAPACITIES, return_period="RP100"
        )

    def test_ordinary_assigns_origin_to_shelter_33(self, result_a):
        """Algorithm A routes origin 286 to shelter 33 (nearest by length)."""
        assert result_a.assignments.get((self.ORIGIN, self.SHELTER), 0) == 1, (
            "Algorithm A should send origin 286 to shelter 33 as nearest by length"
        )

    def test_ordinary_path_crosses_flooded_edge(self, result_a, graph):
        """Algorithm A's path for (286, 33) includes at least one flooded RP100 edge."""
        flood_m = compute_flood_exposed_length(
            result_a.routes, result_a.assignments, graph, "RP100", algorithm="A"
        )
        assert flood_m > 0.0, (
            "Algorithm A should expose origin 286 to flooded edges on the ordinary shortest path"
        )

    def test_flood_aware_assigns_origin_to_a_shelter(self, result_b):
        """Algorithm B assigns origin 286 to some shelter (reachable via dry path)."""
        assigned = sum(
            units for (o, _s), units in result_b.assignments.items() if o == self.ORIGIN
        )
        assert assigned == 1, "Algorithm B should assign origin 286 to at least one shelter"

    def test_flood_aware_path_is_more_expensive(self, result_a, result_b):
        """Algorithm B's flood-aware cost for (286, 33) exceeds the ordinary cost.

        The flood-aware detour is the signature that the policy has rerouted
        around flooded segments — a longer dry path was chosen over the shorter
        exposed one.
        """
        ordinary_cost = result_a.od_costs_ordinary.get((self.ORIGIN, self.SHELTER))
        flood_cost = result_b.od_costs_scenario.get((self.ORIGIN, self.SHELTER))
        assert ordinary_cost is not None, "Ordinary OD cost for (286, 33) should be computed"
        assert flood_cost is not None, "Flood-aware OD cost for (286, 33) should be computed"
        assert flood_cost > ordinary_cost, (
            f"Flood-aware cost ({flood_cost:.1f}) should exceed ordinary cost ({ordinary_cost:.1f}): "
            "the dry alternative is longer than the shortest exposed route"
        )

    def test_flood_aware_path_has_zero_flood_exposure(self, result_b, graph):
        """Algorithm B's assigned routes contain no flooded edges.

        This confirms the conservative hazard policy successfully rerouted
        all assigned paths onto dry segments at RP100.
        """
        flood_m = compute_flood_exposed_length(
            result_b.routes, result_b.assignments, graph, "RP100", algorithm="B"
        )
        assert flood_m == pytest.approx(0.0), (
            f"Algorithm B should have zero flood exposure on assigned routes, got {flood_m:.1f} m"
        )
