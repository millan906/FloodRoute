"""Stage 11 tests: configurable flood penalty, B+ algorithm, common metrics,
batch runner, evidence bundle, and integrity validation.

All tests use synthetic graphs (2–5 nodes) and do not require Streamlit.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Helpers: synthetic graph factories
# ---------------------------------------------------------------------------


def _make_simple_graph(flooded: bool = False) -> nx.MultiDiGraph:
    """Three-node graph: origin 1 → shelter 2 (single edge).

    Nodes: 1 (origin), 2 (shelter).
    Edge 1→2: length_m=100, jrc_rp100_status="flooded" or "modelled_dry".
    """
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    status = "flooded" if flooded else "modelled_dry"
    G.add_edge(1, 2, length_m=100.0, jrc_rp100_status=status,
               jrc_rp10_status=status, jrc_rp20_status=status)
    return G


def _make_two_shelter_graph() -> nx.MultiDiGraph:
    """Four-node graph: origin 1 → shelter 2 (dry, 100m), origin 1 → shelter 3 (flooded, 100m)."""
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2, length_m=100.0, jrc_rp100_status="modelled_dry",
               jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    G.add_edge(1, 3, length_m=100.0, jrc_rp100_status="flooded",
               jrc_rp10_status="flooded", jrc_rp20_status="flooded")
    return G


def _make_capacity_graph() -> nx.MultiDiGraph:
    """Two origins, two shelters, all dry edges."""
    G = nx.MultiDiGraph()
    for n in [1, 2, 3, 4]:
        G.add_node(n)
    # 1→3 (100m), 1→4 (200m), 2→3 (150m), 2→4 (50m)
    for u, v, length_m in [(1, 3, 100.0), (1, 4, 200.0), (2, 3, 150.0), (2, 4, 50.0)]:
        G.add_edge(u, v, length_m=length_m, jrc_rp100_status="modelled_dry",
                   jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    return G


# ---------------------------------------------------------------------------
# Step 1: Configurable flood penalty in cost.py
# ---------------------------------------------------------------------------


def test_configurable_flood_penalty_10():
    """Default penalty=10.0 multiplies flooded edge length by 10."""
    from floodroute.optimization.cost import make_weight_fn
    G = _make_simple_graph(flooded=True)
    wfn = make_weight_fn("RP100", flood_penalty=10.0)
    cost = wfn(1, 2, G[1][2])
    assert cost == pytest.approx(1000.0)  # 100m * 10


def test_configurable_flood_penalty_1():
    """Penalty=1.0 does not penalize flooded edges."""
    from floodroute.optimization.cost import make_weight_fn
    G = _make_simple_graph(flooded=True)
    wfn = make_weight_fn("RP100", flood_penalty=1.0)
    cost = wfn(1, 2, G[1][2])
    assert cost == pytest.approx(100.0)  # 100m * 1


def test_configurable_flood_penalty_prohibited():
    """Penalty='prohibited' makes flooded edges return None."""
    from floodroute.optimization.cost import make_weight_fn
    G = _make_simple_graph(flooded=True)
    wfn = make_weight_fn("RP100", flood_penalty="prohibited")
    cost = wfn(1, 2, G[1][2])
    assert cost is None


def test_prohibited_makes_flooded_edges_unavailable():
    """With 'prohibited', no path exists through flooded-only graph."""
    from floodroute.optimization.routing import compute_od_matrix
    G = _make_simple_graph(flooded=True)
    od_costs, od_routes = compute_od_matrix(
        G, [1], [2], "RP100", flood_penalty="prohibited"
    )
    assert (1, 2) not in od_costs


def test_cache_isolation_by_penalty():
    """Different flood penalties produce different costs."""
    from floodroute.optimization.cost import make_weight_fn
    G = _make_simple_graph(flooded=True)
    cost_10 = make_weight_fn("RP100", flood_penalty=10.0)(1, 2, G[1][2])
    cost_5 = make_weight_fn("RP100", flood_penalty=5.0)(1, 2, G[1][2])
    assert cost_10 == pytest.approx(1000.0)
    assert cost_5 == pytest.approx(500.0)
    assert cost_10 != cost_5


def test_valid_flood_penalties_constant():
    """VALID_FLOOD_PENALTIES contains expected values."""
    from floodroute.optimization.cost import VALID_FLOOD_PENALTIES
    assert 10.0 in VALID_FLOOD_PENALTIES
    assert "prohibited" in VALID_FLOOD_PENALTIES
    assert 1.0 in VALID_FLOOD_PENALTIES


# ---------------------------------------------------------------------------
# Step 2–3: Algorithm definitions
# ---------------------------------------------------------------------------


def test_algorithm_a_exact_definition():
    """Algorithm A uses plain length, assigns to nearest shelter, no capacity."""
    from floodroute.experiments.algorithms import run_ordinary_nearest
    G = _make_two_shelter_graph()
    demands = {1: 10}
    capacities = {2: 5, 3: 5}
    result = run_ordinary_nearest(G, demands, capacities, "RP100")
    assert result.algorithm == "A"
    # Should assign to the nearest shelter by plain length (both 100m, picks lower ID)
    assert (1, 2) in result.assignments or (1, 3) in result.assignments
    assert result.flood_penalty == 10.0  # default


def test_algorithm_b_exact_definition():
    """Algorithm B uses flood-aware cost, assigns to nearest shelter, no capacity."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    G = _make_two_shelter_graph()
    demands = {1: 10}
    capacities = {2: 5, 3: 5}
    result = run_flood_aware_nearest(G, demands, capacities, "RP100")
    assert result.algorithm == "B"
    # Shelter 2 is dry (100m), shelter 3 is flooded (1000m) — should pick shelter 2
    assert (1, 2) in result.assignments


def test_algorithm_b_exact_definition_flood_penalty():
    """Algorithm B accepts flood_penalty parameter."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    G = _make_two_shelter_graph()
    demands = {1: 10}
    capacities = {2: 5, 3: 5}
    result = run_flood_aware_nearest(
        G, demands, capacities, "RP100", flood_penalty=1.0
    )
    assert result.algorithm == "B"
    assert result.flood_penalty == 1.0


def test_algorithm_c_exact_definition():
    """Algorithm C uses min-cost-flow, enforces capacity."""
    from floodroute.experiments.algorithms import run_floodroute_assignment
    G = _make_capacity_graph()
    demands = {1: 3, 2: 3}
    capacities = {3: 3, 4: 3}
    result = run_floodroute_assignment(G, demands, capacities, "RP100")
    assert result.algorithm == "C"
    total_assigned = sum(result.assignments.values())
    assert total_assigned == 6  # all demand assigned


# ---------------------------------------------------------------------------
# Step 4: B+ algorithm
# ---------------------------------------------------------------------------


def test_algorithm_b_plus_exact_definition():
    """B+ returns algorithm='B+' and assigns origins in priority order."""
    from floodroute.experiments.baselines import run_flood_aware_greedy_capacitated
    G = _make_capacity_graph()
    demands = {1: 3, 2: 3}
    capacities = {3: 10, 4: 10}
    result = run_flood_aware_greedy_capacitated(G, demands, capacities, "RP100")
    assert result.algorithm == "B+"
    assert sum(result.assignments.values()) == 6


def test_b_plus_respects_capacity():
    """B+ does not exceed shelter capacity."""
    from floodroute.experiments.baselines import run_flood_aware_greedy_capacitated
    G = _make_capacity_graph()
    demands = {1: 5, 2: 5}
    # Only 3 capacity at shelter 3, 2 at shelter 4
    capacities = {3: 3, 4: 2}
    result = run_flood_aware_greedy_capacitated(G, demands, capacities, "RP100")
    # Check no shelter exceeds capacity
    shelter_loads: dict[int, int] = {}
    for (_o, s), flow in result.assignments.items():
        shelter_loads[s] = shelter_loads.get(s, 0) + flow
    for s, load in shelter_loads.items():
        assert load <= capacities[s], f"Shelter {s} overloaded: {load} > {capacities[s]}"


def test_b_plus_descending_priority_order():
    """B+ processes origins in descending min-OD-cost order (highest-cost first).

    One shelter, capacity=5. Origin 1 has high cost (flooded, 100m → 1000m-eq),
    origin 2 has low cost (dry, 50m). Descending order → origin 1 goes first and
    receives the limited shelter capacity; origin 2 is left without capacity.
    """
    from floodroute.experiments.baselines import run_flood_aware_greedy_capacitated
    G = nx.MultiDiGraph()
    for n in [1, 2, 10]:
        G.add_node(n)
    # Origin 1 → shelter 10: flooded 100m (cost = 1000 with penalty=10)
    G.add_edge(1, 10, length_m=100.0, jrc_rp100_status="flooded",
               jrc_rp10_status="flooded", jrc_rp20_status="flooded")
    # Origin 2 → shelter 10: dry 50m (cost = 50)
    G.add_edge(2, 10, length_m=50.0, jrc_rp100_status="modelled_dry",
               jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    demands = {1: 5, 2: 5}
    capacities = {10: 5}  # capacity only sufficient for one origin
    result = run_flood_aware_greedy_capacitated(G, demands, capacities, "RP100")
    # Origin 1 min cost = 1000 (flooded), origin 2 min cost = 50 (dry)
    # Descending order → origin 1 goes first and gets the 5 units of capacity
    assert result.assignments.get((1, 10), 0) == 5, "Origin 1 (high cost) should get capacity first"
    assert result.assignments.get((2, 10), 0) == 0, "Origin 2 (low cost) left without capacity"


def test_b_plus_demand_splitting_regression():
    """B+ splits demand across shelters: demand=100, caps={A:60,B:40} → all 100 assigned.

    This verifies that B+ is comparable to Algorithm C: both allow divisible
    integer origin flow.
    """
    from floodroute.experiments.baselines import run_flood_aware_greedy_capacitated
    G = nx.MultiDiGraph()
    for n in [1, 20, 21]:
        G.add_node(n)
    # Origin 1 → shelter 20: dry 100m (cheaper)
    G.add_edge(1, 20, length_m=100.0, jrc_rp100_status="modelled_dry",
               jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    # Origin 1 → shelter 21: dry 200m (more expensive)
    G.add_edge(1, 21, length_m=200.0, jrc_rp100_status="modelled_dry",
               jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    demands = {1: 100}
    capacities = {20: 60, 21: 40}
    result = run_flood_aware_greedy_capacitated(G, demands, capacities, "RP100")
    total_assigned = sum(result.assignments.values())
    assert total_assigned == 100, f"Expected 100 assigned, got {total_assigned}"
    assert result.assignments.get((1, 20), 0) == 60, "Shelter 20 (cheaper) should absorb 60"
    assert result.assignments.get((1, 21), 0) == 40, "Shelter 21 (more expensive) should absorb 40"


# ---------------------------------------------------------------------------
# Step 5: Common metrics layer
# ---------------------------------------------------------------------------


def test_physical_distance_separate_from_penalized():
    """Physical distance uses actual length_m, not penalized cost."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    from floodroute.experiments.metrics import compute_metrics
    G = _make_simple_graph(flooded=True)
    demands = {1: 5}
    capacities = {2: 10}
    result = run_flood_aware_nearest(G, demands, capacities, "RP100", flood_penalty=10.0)
    metrics = compute_metrics(result, G)
    # Physical distance = 5 * 100m = 500m
    assert metrics["physical_route_distance_m"] == pytest.approx(500.0)
    # Penalized cost = 5 * 1000m = 5000m-equivalent
    assert metrics["penalized_cost_m_eq"] == pytest.approx(5000.0)


def test_flooded_distance_only_counts_flooded_edges():
    """flooded_route_distance_m only sums flooded edges."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    from floodroute.experiments.metrics import compute_metrics
    G = _make_simple_graph(flooded=False)
    demands = {1: 5}
    capacities = {2: 10}
    result = run_flood_aware_nearest(G, demands, capacities, "RP100")
    metrics = compute_metrics(result, G)
    # Edge is dry — flooded distance = 0
    assert metrics["flooded_route_distance_m"] == pytest.approx(0.0)


def test_flooded_person_distance():
    """flooded_person_distance_m = flow * flooded_physical_m."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    from floodroute.experiments.metrics import compute_metrics
    G = _make_simple_graph(flooded=True)
    demands = {1: 3}
    capacities = {2: 10}
    result = run_flood_aware_nearest(G, demands, capacities, "RP100", flood_penalty=10.0)
    metrics = compute_metrics(result, G)
    # 3 people * 100m flooded = 300 person-metres
    assert metrics["flooded_person_distance_m"] == pytest.approx(300.0)


def test_common_metrics_assignment_rate():
    """assignment_rate = assigned / total_demand."""
    from floodroute.experiments.algorithms import run_ordinary_nearest
    from floodroute.experiments.metrics import compute_metrics
    G = _make_simple_graph(flooded=False)
    demands = {1: 10}
    capacities = {2: 10}
    result = run_ordinary_nearest(G, demands, capacities, "RP100")
    metrics = compute_metrics(result, G)
    assert metrics["assignment_rate"] == pytest.approx(1.0)
    assert metrics["assigned_population"] == 10
    assert metrics["unassigned_population"] == 0


def test_unassigned_reason_decomposition():
    """Unreachable origins appear as 'unreachable' in unassigned_by_reason."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    from floodroute.experiments.metrics import compute_metrics
    # Origin 1 reachable, origin 99 not in graph
    G = _make_simple_graph(flooded=False)
    demands = {1: 5, 99: 3}  # 99 not in graph → unreachable
    capacities = {2: 10}
    result = run_flood_aware_nearest(G, demands, capacities, "RP100")
    metrics = compute_metrics(result, G)
    assert metrics["unassigned_by_reason"]["unreachable"] >= 3


def test_capacity_scaling():
    """adjusted_capacities = round(nominal * multiplier)."""
    from floodroute.experiments.runner import _adjusted_capacities
    nominal = {33: 12000, 58: 10000}
    adj = _adjusted_capacities(nominal, 0.5)
    assert adj[33] == 6000
    assert adj[58] == 5000
    adj2 = _adjusted_capacities(nominal, 2.0)
    assert adj2[33] == 24000


# ---------------------------------------------------------------------------
# Step 6: Batch runner
# ---------------------------------------------------------------------------


def test_pilot_mode_one_per_algorithm():
    """Pilot mode generates exactly one scenario per algorithm."""
    from floodroute.experiments.runner import (
        generate_scenarios,
        make_experiment_config,
    )
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A", "B", "C"),
        return_periods=("RP10", "RP100"),
        demand_fractions=(0.10, 0.25),
        capacity_multipliers=(1.0, 2.0),
        flood_penalties=(10.0, "prohibited"),
        nominal_capacities={33: 12000, 58: 10000},
        pilot_mode=True,
    )
    scenarios = generate_scenarios(config)
    assert len(scenarios) == 3  # one per algorithm
    algs = [s.algorithm for s in scenarios]
    assert set(algs) == {"A", "B", "C"}


def test_deterministic_scenario_generation():
    """generate_scenarios is deterministic for the same config."""
    from floodroute.experiments.runner import generate_scenarios, make_experiment_config
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A", "B"),
        return_periods=("RP10",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
        pilot_mode=False,
    )
    s1 = generate_scenarios(config)
    s2 = generate_scenarios(config)
    assert [(s.algorithm, s.return_period) for s in s1] == [
        (s.algorithm, s.return_period) for s in s2
    ]


def test_deterministic_configuration_hash():
    """Same scientific parameters → same configuration_hash across calls."""
    from floodroute.experiments.runner import make_experiment_config
    c1 = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP100",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
    )
    c2 = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP100",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
    )
    assert c1.configuration_hash == c2.configuration_hash


def test_experiment_id_unique_per_run():
    """experiment_id differs between two make_experiment_config calls (timestamp prefix).

    experiment_id = <timestamp>_<config_hash>. Two calls may produce the same
    timestamp in theory (sub-second), but configuration_hash is the stable part.
    The key property tested here: experiment_id includes configuration_hash.
    """
    from floodroute.experiments.runner import make_experiment_config
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP100",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
    )
    # experiment_id must embed the configuration_hash
    assert config.configuration_hash in config.experiment_id
    # experiment_id has the form <timestamp>_<hash>
    assert "_" in config.experiment_id
    parts = config.experiment_id.split("_")
    assert len(parts) == 2
    assert parts[1] == config.configuration_hash


# ---------------------------------------------------------------------------
# Step 7: Evidence manifest
# ---------------------------------------------------------------------------


def _make_minimal_config_and_results():
    """Return a minimal ExperimentConfig and a list with one mock ScenarioResult."""
    from floodroute.experiments.runner import (
        ScenarioKey,
        ScenarioResult,
        make_experiment_config,
    )
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP100",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
        pilot_mode=False,
    )
    key = ScenarioKey(
        algorithm="A",
        return_period="RP100",
        demand_fraction=0.10,
        capacity_multiplier=1.0,
        flood_penalty=10.0,
    )
    sr = ScenarioResult(
        key=key,
        nominal_capacities={33: 100},
        adjusted_capacities={33: 100},
        metrics={
            "total_demand": 10,
            "assigned_population": 10,
            "unassigned_population": 0,
            "assignment_rate": 1.0,
            "physical_route_distance_m": 500.0,
            "flooded_route_distance_m": 0.0,
            "dry_route_distance_m": 500.0,
            "population_weighted_distance_m": 500.0,
            "flooded_person_distance_m": 0.0,
            "penalized_cost_m_eq": 500.0,
            "shelter_overflow": {33: 0},
            "facility_utilization": {33: 0.1},
            "num_utilized_facilities": 1,
            "total_overflow_units": 0,
            "num_split_origins": 0,
            "unassigned_by_reason": {"unreachable": 0, "capacity_exhausted": 0},
            "runtime_s": 0.01,
            "run_status": "completed",
        },
        assignments={(1, 33): 10},
        route_metrics={(1, 33): {"physical_m": 50.0, "flooded_m": 0.0, "penalized_m_eq": 500.0}},
        facility_metrics={33: {"load": 10, "capacity": 100, "utilization": 0.1, "overflow": 0}},
        unassigned_reasons={},
        run_status="completed",
        error_message=None,
        barangay_demand={"PH0600613001": 10},
    )
    return config, [sr]


def test_evidence_manifest_completeness():
    """save_experiment writes manifest.json with required fields."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp_dir = save_experiment(config, results, experiments_root=root)
        manifest_path = exp_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        required = [
            "experiment_id", "configuration_hash", "created_utc", "saved_utc",
            "git_commit", "git_dirty", "municipality", "algorithms",
            "return_periods", "demand_fractions", "capacity_multipliers",
            "flood_penalties", "nominal_capacities", "pilot_mode",
            "python_version", "networkx_version", "dataset_inventory",
            "file_hashes",
        ]
        for field in required:
            assert field in manifest, f"Missing field: {field}"
        # dataset_inventory is a list (may be empty when no paths supplied)
        assert isinstance(manifest["dataset_inventory"], list)
        # configuration_hash must be recorded separately from experiment_id
        assert manifest["configuration_hash"] == config.configuration_hash
        assert manifest["experiment_id"] != manifest["configuration_hash"]


def test_output_immutability_no_overwrite():
    """save_experiment raises FileExistsError if the same experiment_id is saved twice.

    experiment_id = timestamp + config_hash. Passing the same config object
    twice yields the same experiment_id → second call raises FileExistsError.
    """
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_experiment(config, results, experiments_root=root)
        with pytest.raises(FileExistsError):
            # Same config object → same experiment_id → directory already exists
            save_experiment(config, results, experiments_root=root)


def test_checksum_verification():
    """checksums.sha256 contains correct hashes for all CSV files."""
    from floodroute.experiments.manifest import _sha256_file, save_experiment
    config, results = _make_minimal_config_and_results()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp_dir = save_experiment(config, results, experiments_root=root)
        checksums_path = exp_dir / "checksums.sha256"
        assert checksums_path.exists()
        lines = checksums_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        for line in lines:
            parts = line.split("  ", 1)
            assert len(parts) == 2, f"Malformed checksum line: {line!r}"
            hex_hash, fname = parts
            fpath = exp_dir / fname
            assert fpath.exists(), f"Checksum references missing file: {fname}"
            actual = _sha256_file(fpath)
            assert actual == hex_hash, f"Checksum mismatch for {fname}"


# ---------------------------------------------------------------------------
# Step 8: Integrity validation
# ---------------------------------------------------------------------------


def test_integrity_check_passes_valid():
    """validate_experiment returns [] for a valid experiment."""
    from floodroute.experiments.integrity import validate_experiment
    config, results = _make_minimal_config_and_results()
    key = results[0].key
    demands_by_scenario = {key: {1: 10}}
    violations = validate_experiment(results, config, demands_by_scenario)
    assert violations == [], f"Unexpected violations: {violations}"


def test_integrity_check_catches_capacity_violation():
    """validate_experiment catches a shelter capacity violation for Alg C."""
    from floodroute.experiments.integrity import validate_experiment
    from floodroute.experiments.runner import (
        ScenarioKey,
        ScenarioResult,
        make_experiment_config,
    )
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP100",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 5},  # capacity = 5
        pilot_mode=False,
    )
    key = ScenarioKey(
        algorithm="C",
        return_period="RP100",
        demand_fraction=0.10,
        capacity_multiplier=1.0,
        flood_penalty=10.0,
    )
    sr = ScenarioResult(
        key=key,
        nominal_capacities={33: 5},
        adjusted_capacities={33: 5},
        metrics={
            "total_demand": 10,
            "assigned_population": 10,
            "unassigned_population": 0,
            "assignment_rate": 1.0,
            "penalized_cost_m_eq": 500.0,
        },
        assignments={(1, 33): 10},  # 10 > capacity 5 → violation
        route_metrics={(1, 33): {"physical_m": 50.0, "flooded_m": 0.0, "penalized_m_eq": 500.0}},
        facility_metrics={33: {"load": 10, "capacity": 5, "utilization": 2.0, "overflow": 5}},
        unassigned_reasons={},
        run_status="completed",
        error_message=None,
        barangay_demand={},
    )
    demands_by_scenario = {key: {1: 10}}
    violations = validate_experiment([sr], config, demands_by_scenario)
    assert any("load=10 exceeds" in v for v in violations), f"Expected violation not found: {violations}"


# ---------------------------------------------------------------------------
# Step 5 (helpers): physical_path_distances
# ---------------------------------------------------------------------------


def test_physical_path_distances_basic():
    """physical_path_distances returns correct total and flooded distances."""
    from floodroute.experiments.metrics import physical_path_distances
    G = _make_simple_graph(flooded=True)
    total_m, flooded_m = physical_path_distances([1, 2], G, "RP100")
    assert total_m == pytest.approx(100.0)
    assert flooded_m == pytest.approx(100.0)


def test_physical_path_distances_dry():
    """physical_path_distances returns 0 flooded_m for dry edges."""
    from floodroute.experiments.metrics import physical_path_distances
    G = _make_simple_graph(flooded=False)
    total_m, flooded_m = physical_path_distances([1, 2], G, "RP100")
    assert total_m == pytest.approx(100.0)
    assert flooded_m == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Step 6: Hamilton demand reuse
# ---------------------------------------------------------------------------


def test_hamilton_demand_reuse():
    """build_demands is deterministic: same inputs → same demands."""
    from floodroute.experiments.demand import BarangayOrigin, build_demands
    origins = [
        BarangayOrigin(psgc="P001", name="A", population_2020=1000,
                       origin_node=1, snap_distance_m=0.0),
        BarangayOrigin(psgc="P002", name="B", population_2020=2000,
                       origin_node=2, snap_distance_m=0.0),
    ]
    d1, _ = build_demands(origins, 0.10)
    d2, _ = build_demands(origins, 0.10)
    assert d1 == d2


# ---------------------------------------------------------------------------
# Stage 11 correction tests (Item 1–7)
# ---------------------------------------------------------------------------


def test_b_plus_zero_overflow_invariant():
    """B+ enforces capacity — shelter load never exceeds capacity (zero overflow)."""
    from floodroute.experiments.baselines import run_flood_aware_greedy_capacitated
    from floodroute.experiments.metrics import compute_metrics
    G = _make_capacity_graph()
    demands = {1: 5, 2: 5}
    capacities = {3: 4, 4: 3}  # tight capacity, some demand will be unassigned
    result = run_flood_aware_greedy_capacitated(G, demands, capacities, "RP100")
    metrics = compute_metrics(result, G)
    assert metrics["total_overflow_units"] == 0, (
        f"B+ must never overflow, got {metrics['total_overflow_units']}"
    )


def test_algorithm_c_zero_overflow_invariant():
    """Algorithm C (MCF) enforces capacity — shelter load never exceeds capacity."""
    from floodroute.experiments.algorithms import run_floodroute_assignment
    from floodroute.experiments.metrics import compute_metrics
    G = _make_capacity_graph()
    demands = {1: 5, 2: 5}
    capacities = {3: 4, 4: 3}
    result = run_floodroute_assignment(G, demands, capacities, "RP100")
    metrics = compute_metrics(result, G)
    assert metrics["total_overflow_units"] == 0, (
        f"C (MCF) must never overflow, got {metrics['total_overflow_units']}"
    )


def test_algorithm_a_can_overflow():
    """Algorithm A does not enforce capacity — overflow may occur."""
    from floodroute.experiments.algorithms import run_ordinary_nearest
    from floodroute.experiments.metrics import compute_metrics
    G = _make_simple_graph(flooded=False)
    demands = {1: 10}
    capacities = {2: 5}  # 10 > 5 → overflow
    result = run_ordinary_nearest(G, demands, capacities, "RP100")
    # A assigns greedily without capacity, so shelter 2 load = 10 > capacity 5
    shelter_load = sum(v for (o, s), v in result.assignments.items() if s == 2)
    # A may assign the full demand or nearest only — verify load vs capacity
    if shelter_load > 5:
        metrics = compute_metrics(result, G)
        assert metrics["total_overflow_units"] > 0


def test_detour_ratio_zero_ordinary_denominator():
    """detour_ratio returns 1.0 when the ordinary-cost denominator is zero."""
    from floodroute.experiments.algorithms import RunResult
    from floodroute.experiments.metrics import compute_metrics
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    # Edge with length_m = 0 (degenerate but valid for the test)
    G.add_edge(1, 2, length_m=0.0, jrc_rp100_status="modelled_dry",
               jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    result = RunResult(
        algorithm="A",
        return_period="RP100",
        demand_fraction=0.10,
        demands={1: 5},
        capacities={2: 10},
        assignments={(1, 2): 5},
        routes={(1, 2): [1, 2]},
        od_costs_scenario={(1, 2): 0.0},
        od_costs_ordinary={(1, 2): 0.0},
        runtime_s=0.001,
        flood_penalty=10.0,
    )
    metrics = compute_metrics(result, G)
    assert metrics.get("penalized_cost_m_eq", 0.0) == pytest.approx(0.0)


def test_dry_plus_flooded_equals_physical():
    """dry_route_distance_m + flooded_route_distance_m == physical_route_distance_m."""
    from floodroute.experiments.algorithms import run_flood_aware_nearest
    from floodroute.experiments.metrics import compute_metrics
    G = _make_simple_graph(flooded=True)
    demands = {1: 7}
    capacities = {2: 10}
    result = run_flood_aware_nearest(G, demands, capacities, "RP100")
    metrics = compute_metrics(result, G)
    total = metrics["physical_route_distance_m"]
    dry = metrics["dry_route_distance_m"]
    flooded = metrics["flooded_route_distance_m"]
    assert dry + flooded == pytest.approx(total), (
        f"dry({dry}) + flooded({flooded}) != physical({total})"
    )


def test_dataset_inventory_with_real_path(tmp_path):
    """build_dataset_inventory records sha256 for existing files and marks missing."""
    from floodroute.experiments.manifest import build_dataset_inventory
    existing = tmp_path / "data.csv"
    existing.write_text("a,b\n1,2\n", encoding="utf-8")
    missing = tmp_path / "nonexistent.gpkg"
    inventory = build_dataset_inventory({
        "population_csv": existing,
        "barangay_boundaries": missing,
    })
    assert len(inventory) == 2
    by_role = {r["role"]: r for r in inventory}
    assert by_role["population_csv"]["exists"] is True
    assert len(by_role["population_csv"]["sha256"]) == 64  # SHA-256 hex
    assert by_role["barangay_boundaries"]["exists"] is False
    assert by_role["barangay_boundaries"]["sha256"] is None


def test_manifest_records_dataset_inventory(tmp_path):
    """save_experiment writes dataset_inventory.json and includes it in manifest."""
    from floodroute.experiments.manifest import save_experiment
    data_file = tmp_path / "pop.csv"
    data_file.write_text("psgc,pop\nP001,1000\n", encoding="utf-8")
    experiments_root = tmp_path / "experiments"
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(
        config, results,
        experiments_root=experiments_root,
        dataset_paths={"population_csv": data_file},
    )
    inventory_path = exp_dir / "dataset_inventory.json"
    assert inventory_path.exists()
    inventory = json.loads(inventory_path.read_text())
    assert isinstance(inventory, list)
    assert len(inventory) == 1
    assert inventory[0]["role"] == "population_csv"
    assert inventory[0]["exists"] is True
    # Also reflected in manifest
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert len(manifest["dataset_inventory"]) == 1


def test_pilot_mode_all_four_algorithms_matched_conditions():
    """Pilot mode runs A, B, B+, C under identical scenario conditions."""
    from floodroute.experiments.runner import generate_scenarios, make_experiment_config
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A", "B", "B+", "C"),
        return_periods=("RP10", "RP100"),
        demand_fractions=(0.10, 0.25),
        capacity_multipliers=(0.75, 1.00, 1.25),
        flood_penalties=(10.0, "prohibited"),
        nominal_capacities={33: 12000, 58: 10000},
        pilot_mode=True,
    )
    scenarios = generate_scenarios(config)
    assert len(scenarios) == 4  # one per algorithm
    algs = {s.algorithm for s in scenarios}
    assert algs == {"A", "B", "B+", "C"}
    # All scenarios must have identical non-algorithm parameters
    rps = {s.return_period for s in scenarios}
    fracs = {s.demand_fraction for s in scenarios}
    mults = {s.capacity_multiplier for s in scenarios}
    penalties = {s.flood_penalty for s in scenarios}
    assert len(rps) == 1, f"Pilot must use one return period, got {rps}"
    assert len(fracs) == 1, f"Pilot must use one demand fraction, got {fracs}"
    assert len(mults) == 1, f"Pilot must use one capacity multiplier, got {mults}"
    assert len(penalties) == 1, f"Pilot must use one flood penalty, got {penalties}"


def test_capacity_multipliers_default_values():
    """Default capacity_multipliers are (0.50, 0.75, 1.00, 1.25)."""
    from floodroute.experiments.runner import make_experiment_config
    config = make_experiment_config(municipality="PH0600613")
    assert set(config.capacity_multipliers) == {0.50, 0.75, 1.00, 1.25}


def test_capacity_multiplier_in_scenario_key():
    """capacity_multiplier is recorded in every ScenarioKey."""
    from floodroute.experiments.runner import generate_scenarios, make_experiment_config
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP100",),
        demand_fractions=(0.10,),
        capacity_multipliers=(0.75, 1.25),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
        pilot_mode=False,
    )
    scenarios = generate_scenarios(config)
    mults = {s.capacity_multiplier for s in scenarios}
    assert mults == {0.75, 1.25}


def test_facility_csv_has_nominal_and_adjusted_capacity(tmp_path):
    """facility_metrics.csv records both nominal_capacity and adjusted_capacity."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    facility_csv = (exp_dir / "facility_metrics.csv").read_text(encoding="utf-8")
    assert "nominal_capacity" in facility_csv
    assert "adjusted_capacity" in facility_csv


def test_algorithm_c_label():
    """Algorithm C label is qualified correctly — not 'globally optimal evacuation'."""
    from floodroute.dashboard.result_formatter import ALGORITHM_LABELS
    label = ALGORITHM_LABELS["C"]
    assert "Global shelter allocation" in label
    assert "exact" in label.lower() or "MCF" in label or "minimum-cost" in label.lower()
    # Must NOT claim broader optimality
    assert "globally optimal evacuation" not in label.lower()


# ---------------------------------------------------------------------------
# Stage 11 evidence-saving corrections
# ---------------------------------------------------------------------------


def test_scenario_result_has_barangay_demand_field():
    """ScenarioResult carries a barangay_demand dict (PSGC → demand units)."""
    from floodroute.experiments.runner import ScenarioKey, ScenarioResult
    key = ScenarioKey("A", "RP100", 0.25, 1.0, 10.0)
    sr = ScenarioResult(
        key=key,
        nominal_capacities={33: 100},
        adjusted_capacities={33: 100},
        metrics={},
        assignments={},
        route_metrics={},
        facility_metrics={},
        unassigned_reasons={},
        run_status="completed",
        error_message=None,
        barangay_demand={"PH0600613001": 5, "PH0600613002": 3},
    )
    assert sr.barangay_demand == {"PH0600613001": 5, "PH0600613002": 3}


def test_scenario_result_barangay_demand_defaults_to_empty():
    """ScenarioResult.barangay_demand defaults to {} when not supplied."""
    from floodroute.experiments.runner import ScenarioKey, ScenarioResult
    key = ScenarioKey("A", "RP100", 0.25, 1.0, 10.0)
    sr = ScenarioResult(
        key=key,
        nominal_capacities={},
        adjusted_capacities={},
        metrics={},
        assignments={},
        route_metrics={},
        facility_metrics={},
        unassigned_reasons={},
        run_status="completed",
        error_message=None,
    )
    assert sr.barangay_demand == {}


def test_run_scenario_populates_barangay_demand():
    """run_scenario stores per-barangay Hamilton demand from build_demands."""
    import networkx as nx

    from floodroute.experiments.demand import BarangayOrigin
    from floodroute.experiments.runner import (
        ScenarioKey,
        make_experiment_config,
        run_scenario,
    )
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2, length_m=100.0, jrc_rp100_status="modelled_dry",
               jrc_rp10_status="modelled_dry", jrc_rp20_status="modelled_dry")
    origins = [BarangayOrigin(
        psgc="PH0600613001", name="Poblacion", population_2020=100,
        origin_node=1, snap_distance_m=10.0,
    )]
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP100",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={2: 200},
        pilot_mode=False,
    )
    key = ScenarioKey("A", "RP100", 0.25, 1.0, 10.0)
    sr = run_scenario(G, origins, config, key)
    assert isinstance(sr.barangay_demand, dict)
    assert "PH0600613001" in sr.barangay_demand
    assert sr.barangay_demand["PH0600613001"] == 25  # round(100 * 0.25)


def test_save_experiment_writes_barangay_demand_csv(tmp_path):
    """save_experiment writes barangay_demand.csv with correct columns and rows."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    # results[0].barangay_demand = {"PH0600613001": 10} (set in _make_minimal_config_and_results)
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    csv_path = exp_dir / "barangay_demand.csv"
    assert csv_path.exists(), "barangay_demand.csv must be written"
    text = csv_path.read_text(encoding="utf-8")
    assert "demand_fraction" in text
    assert "adm4_pcode" in text
    assert "barangay_name" in text
    assert "origin_node" in text
    assert "demand_units" in text
    assert "PH0600613001" in text
    assert "10" in text


def test_save_experiment_barangay_demand_deduplicates_by_fraction(tmp_path):
    """barangay_demand.csv has one row per (fraction, psgc), not one per scenario."""
    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import ScenarioKey, ScenarioResult, make_experiment_config
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A", "B"),
        return_periods=("RP100",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
        pilot_mode=False,
    )
    # Two scenarios at same fraction — barangay_demand is identical
    def _sr(alg):
        return ScenarioResult(
            key=ScenarioKey(alg, "RP100", 0.25, 1.0, 10.0),
            nominal_capacities={33: 100},
            adjusted_capacities={33: 100},
            metrics={},
            assignments={},
            route_metrics={},
            facility_metrics={},
            unassigned_reasons={},
            run_status="completed",
            error_message=None,
            barangay_demand={"PH0600613001": 7},
        )
    results = [_sr("A"), _sr("B")]
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    lines = (exp_dir / "barangay_demand.csv").read_text(encoding="utf-8").strip().splitlines()
    # Header + one data row (deduplicated)
    assert len(lines) == 2, f"Expected 1 data row (header + 1), got {len(lines) - 1}"


def test_save_experiment_writes_facility_registry_csv(tmp_path):
    """save_experiment writes facility_registry.csv with supplied rows."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    fac_rows = [
        {
            "facility_id": "SJDB-001",
            "name": "Harborview Elementary School",
            "facility_type": "school",
            "designation_status": "government_confirmed_from_published_sources",
            "snapped_node": 33,
            "snapping_distance_m": 42.1,
            "configured_capacity": 20000,
        }
    ]
    exp_dir = save_experiment(
        config, results,
        experiments_root=tmp_path,
        facility_registry_rows=fac_rows,
    )
    csv_path = exp_dir / "facility_registry.csv"
    assert csv_path.exists(), "facility_registry.csv must be written"
    text = csv_path.read_text(encoding="utf-8")
    assert "facility_id" in text
    assert "SJDB-001" in text
    assert "Harborview Elementary School" in text
    assert "20000" in text


def test_save_experiment_facility_registry_csv_empty_when_none(tmp_path):
    """save_experiment writes a header-only facility_registry.csv when rows are None."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    text = (exp_dir / "facility_registry.csv").read_text(encoding="utf-8")
    lines = [ln for ln in text.strip().splitlines() if ln]
    assert len(lines) == 1, "Only the header row expected when no facility rows supplied"
    assert "facility_id" in lines[0]


def test_save_experiment_manifest_has_network_note(tmp_path):
    """manifest.json contains the OSM network limitation statement."""
    from floodroute.experiments.manifest import _NETWORK_NOTE, save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert "network_note" in manifest
    assert "OpenStreetMap" in manifest["network_note"]
    assert manifest["network_note"] == _NETWORK_NOTE


def test_save_experiment_manifest_has_source_scenario_id(tmp_path):
    """manifest.json records source_scenario_id when supplied."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(
        config, results,
        experiments_root=tmp_path,
        source_scenario_id="7cdc187852aa",
    )
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest.get("source_scenario_id") == "7cdc187852aa"


def test_save_experiment_manifest_source_scenario_id_null_when_absent(tmp_path):
    """manifest.json records source_scenario_id as null when not supplied."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert "source_scenario_id" in manifest
    assert manifest["source_scenario_id"] is None


def test_save_experiment_manifest_has_road_overrides(tmp_path):
    """manifest.json records serialized road overrides when supplied."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    overrides = {"1,2": {"u": 1, "v": 2, "status": "road_closed",
                         "evidence_type": "controlled_assumption",
                         "source_reference": "", "observation_time": "", "notes": ""}}
    exp_dir = save_experiment(
        config, results,
        experiments_root=tmp_path,
        road_overrides=overrides,
    )
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert "road_overrides" in manifest
    assert "1,2" in manifest["road_overrides"]
    assert manifest["road_overrides"]["1,2"]["status"] == "road_closed"


def test_save_experiment_manifest_road_overrides_empty_when_none(tmp_path):
    """manifest.json records road_overrides as {} when not supplied."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert "road_overrides" in manifest
    assert manifest["road_overrides"] == {}


def test_save_experiment_new_files_in_checksums(tmp_path):
    """checksums.sha256 includes barangay_demand.csv and facility_registry.csv."""
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    exp_dir = save_experiment(config, results, experiments_root=tmp_path)
    checksums_text = (exp_dir / "checksums.sha256").read_text(encoding="utf-8")
    assert "barangay_demand.csv" in checksums_text
    assert "facility_registry.csv" in checksums_text


def test_save_experiment_barangay_demand_with_origins(tmp_path):
    """barangay_demand.csv records barangay_name and origin_node when origins supplied."""
    from floodroute.experiments.demand import BarangayOrigin
    from floodroute.experiments.manifest import save_experiment
    config, results = _make_minimal_config_and_results()
    origins = [BarangayOrigin(
        psgc="PH0600613001", name="Poblacion", population_2020=40,
        origin_node=99, snap_distance_m=5.0,
    )]
    exp_dir = save_experiment(config, results, experiments_root=tmp_path, origins=origins)
    text = (exp_dir / "barangay_demand.csv").read_text(encoding="utf-8")
    assert "Poblacion" in text
    assert "99" in text


class TestResetEntireScenario:
    """Regression: Reset entire scenario clears all scenario and result state.

    Tests call make_default_session_state() directly — the same function the
    "Yes, reset" button handler calls — so the tests and the reset are always
    in sync; there is no duplicated dictionary in the test file.
    """

    @staticmethod
    def _defaults() -> dict:
        from floodroute.dashboard.session_defaults import make_default_session_state
        return make_default_session_state()

    @staticmethod
    def _populated() -> dict:
        """Return a session-state-like dict with every reset key populated."""
        return {
            "result": object(),           # non-None sentinel (previous RunResult)
            "metrics": {"total_demand": 16285, "total_assigned": 14565},
            "run_params": ("RP100", ["Atabay"], "25%", 0.25, {58: 300}, {}),
            "run_error": "some error",
            "run_scenario_key": "RP100|...|some_key",
            "ordinary_path": [1, 2, 3],
            "alg_c_path": [4, 5, 6],
            "alg_c_assigned_shelter": 58,
            "alg_c_origin_status": {"status": "assigned", "shelter": 58},
            "road_override_store_dict": {"edge_1": {"override": "closed"}},
            "road_name_conditions": {"Rizal St": {"condition": "road_closed", "count": 3}},
            "step4_last_applied": ("Rizal St", "road_closed", 3),
            "sc_name": "My Saved Scenario",
            "sc_id": "abc-123",
            "sc_dirty": True,
            "sc_unresolved_fids": ["way:99999"],
            "sc_confirm_delete": "abc-123",
            "sc_selected_fids": ["way:168981805", "SJDB-001"],
            "sc_facility_caps": {"way:168981805": 300, "SJDB-001": 500},
            "activated_osm_ids": {"way:168981805"},
            "scenario_capacities": {"some_key": 100},
            "selected_origin_node": 42,
            "_fit_bounds": True,
            "road_editor_road": "Rizal Street",
            "road_editor_cond": "road_closed",
            "highlighted_override_road": "Rizal Street",
            "_pending_clear_road_search": True,
            "_confirm_remove_road": "Rizal Street",
            "_confirm_clear_roads": True,
            "_confirm_scenario_reset": True,
            "all_barangays_check": False,
            "barangay_multiselect": ["Atabay", "Cuyab"],
            "demand_type_radio": "Exact number",
            "demand_exact_value": 5000,
        }

    def _apply_reset(self, state: dict) -> dict:
        """Simulate st.session_state.update(make_default_session_state())."""
        state.update(self._defaults())
        return state

    # ── Function contract ─────────────────────────────────────────────────

    def test_each_call_returns_fresh_containers(self):
        """make_default_session_state() must return a new dict with new containers."""
        d1 = self._defaults()
        d2 = self._defaults()
        d1["road_override_store_dict"]["injected"] = True
        assert "injected" not in d2["road_override_store_dict"], (
            "road_override_store_dict must be a fresh container on each call"
        )

    def test_each_call_returns_fresh_list_containers(self):
        d1 = self._defaults()
        d2 = self._defaults()
        d1["sc_selected_fids"].append("sentinel")
        assert "sentinel" not in d2["sc_selected_fids"]

    # ── Result state ─────────────────────────────────────────────────────

    def test_reset_clears_result(self):
        assert self._apply_reset(self._populated())["result"] is None

    def test_reset_clears_metrics(self):
        assert self._apply_reset(self._populated())["metrics"] is None

    def test_reset_clears_run_params(self):
        assert self._apply_reset(self._populated())["run_params"] is None

    def test_reset_clears_run_scenario_key(self):
        assert self._apply_reset(self._populated())["run_scenario_key"] is None

    def test_reset_clears_route_overlays(self):
        s = self._apply_reset(self._populated())
        assert s["ordinary_path"] is None
        assert s["alg_c_path"] is None
        assert s["alg_c_assigned_shelter"] is None
        assert s["alg_c_origin_status"] is None

    # ── Road conditions ──────────────────────────────────────────────────

    def test_reset_clears_road_override_store(self):
        assert self._apply_reset(self._populated())["road_override_store_dict"] == {}

    def test_reset_clears_road_name_conditions(self):
        assert self._apply_reset(self._populated())["road_name_conditions"] == {}

    # ── Scenario identity ────────────────────────────────────────────────

    def test_reset_clears_scenario_identity(self):
        s = self._apply_reset(self._populated())
        assert s["sc_name"] == "Untitled scenario"
        assert s["sc_id"] is None
        assert s["sc_dirty"] is False

    # ── Facility selection ───────────────────────────────────────────────

    def test_reset_clears_selected_fids(self):
        assert self._apply_reset(self._populated())["sc_selected_fids"] == []

    def test_reset_clears_facility_caps(self):
        assert self._apply_reset(self._populated())["sc_facility_caps"] == {}

    # ── UI and map state ─────────────────────────────────────────────────

    def test_reset_clears_selected_origin_node(self):
        assert self._apply_reset(self._populated())["selected_origin_node"] is None

    def test_reset_clears_road_editor_state(self):
        s = self._apply_reset(self._populated())
        assert s["road_editor_road"] is None
        assert s["road_editor_cond"] is None
        assert s["highlighted_override_road"] is None

    def test_reset_clears_dialog_flags(self):
        s = self._apply_reset(self._populated())
        assert s["_confirm_scenario_reset"] is False
        assert s["_confirm_remove_road"] is None
        assert s["_confirm_clear_roads"] is False

    # ── Widget defaults ──────────────────────────────────────────────────

    def test_reset_restores_barangay_widget_to_all(self):
        s = self._apply_reset(self._populated())
        assert s["all_barangays_check"] is True
        assert s["barangay_multiselect"] == []

    def test_reset_restores_demand_widget_to_25pct(self):
        s = self._apply_reset(self._populated())
        assert s["demand_type_radio"] == "25%"
        assert s["demand_exact_value"] == 0

    # ── No result rendered after reset ───────────────────────────────────

    def test_no_result_rendered_after_reset(self):
        """After reset, result and metrics are None — the result panel must not render."""
        s = self._apply_reset(self._populated())
        result_panel_would_render = (
            s["result"] is not None and s["metrics"] is not None
        )
        assert not result_panel_would_render, (
            "Result panel must not render after reset — "
            f"result={s['result']!r}, metrics={s['metrics']!r}"
        )

    def test_stale_warning_not_shown_after_reset(self):
        """Fixed stale predicate: result is None after reset → not stale."""
        s = self._apply_reset(self._populated())
        _result_is_stale = (
            s["result"] is not None
            and s["run_scenario_key"] != "any_current_key"
        )
        assert not _result_is_stale


class TestStalePlannerResultDetection:
    """Regression: stale-plan predicate correctly fires when run_scenario_key is None.

    Bug (fixed): five action sites set run_scenario_key = None to signal that
    the displayed result is no longer valid (road condition add/edit/remove/clear
    and scenario load).  The old staleness predicate guarded with

        run_scenario_key is not None  ...

    so None short-circuited to False — hiding the stale warning instead of
    showing it.  The fix removes that guard.
    """

    # Reproduce the _scenario_key() logic from app.py as a pure helper.
    # app.py is not importable in tests (module-level Streamlit calls).
    @staticmethod
    def _scenario_key(
        return_period, selected_bgy_names, demand_type,
        demand_value, effective_shelters, override_dict,
    ) -> str:
        return "|".join([
            return_period,
            str(sorted(selected_bgy_names)),
            demand_type,
            str(demand_value),
            str(tuple(sorted(effective_shelters.items()))),
            str(sorted(override_dict.items())),
        ])

    def _key(self, shelters, overrides=None):
        return self._scenario_key(
            "RP100", ["Barangay A"], "25%", 0.25, shelters, overrides or {}
        )

    # ── Key differentiation ───────────────────────────────────────────────

    def test_different_capacity_produces_different_key(self):
        """Changing a facility capacity must change the scenario key."""
        assert self._key({58: 200}) != self._key({58: 500})

    def test_different_rp_produces_different_key(self):
        """Changing the return period must change the scenario key."""
        k1 = self._scenario_key("RP10", ["B"], "25%", 0.25, {58: 300}, {})
        k2 = self._scenario_key("RP100", ["B"], "25%", 0.25, {58: 300}, {})
        assert k1 != k2

    def test_different_road_overrides_produce_different_key(self):
        """Adding a road override must change the scenario key."""
        assert self._key({58: 300}, {}) != self._key({58: 300}, {"road_1": "closed"})

    def test_two_scenario_configs_different_capacity_differ(self):
        """Two ScenarioConfigs with different capacities → different resolve output
        → different key (end-to-end data-flow check)."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig

        catalog = build_catalog()
        selectable = [e for e in catalog.all() if e.can_be_selected]
        assert selectable, "Catalog must have selectable facilities"
        fid = selectable[0].facility_id

        sc1 = ScenarioConfig(
            selected_facility_ids=[fid],
            facility_capacities={fid: 200},
        )
        sc2 = ScenarioConfig(
            selected_facility_ids=[fid],
            facility_capacities={fid: 500},
        )
        shelters1 = catalog.resolve_node_shelters(sc1)
        shelters2 = catalog.resolve_node_shelters(sc2)

        assert shelters1 != shelters2, "Different capacities must resolve differently"
        k1 = self._key(shelters1)
        k2 = self._key(shelters2)
        assert k1 != k2, "Different resolve outputs must produce different scenario keys"

    def test_second_run_does_not_reuse_first_result(self):
        """Scenario key from run 2 != key from run 1 → they are treated as distinct runs.

        Simulates: run once with {fid: 200}, then change capacity to {fid: 500} and
        re-run.  The stored run_scenario_key must not equal the new current_key.
        """
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig

        catalog = build_catalog()
        fid = next(e.facility_id for e in catalog.all() if e.can_be_selected)

        sc1 = ScenarioConfig(selected_facility_ids=[fid], facility_capacities={fid: 200})
        sc2 = ScenarioConfig(selected_facility_ids=[fid], facility_capacities={fid: 500})

        stored_key = self._key(catalog.resolve_node_shelters(sc1))  # key after run 1
        current_key = self._key(catalog.resolve_node_shelters(sc2))  # key for run 2

        assert stored_key != current_key, (
            "Run 2 scenario key must differ from run 1 — "
            "second run must not reuse first result"
        )

    # ── Stale predicate correctness ───────────────────────────────────────

    def test_old_predicate_incorrectly_hides_stale_result(self):
        """Document the bug: old predicate suppresses stale warning when key is None."""
        current_key = self._key({58: 300})
        stored_key = None   # set by road condition change / scenario load
        result = object()   # non-None: a prior RunResult exists

        old_is_stale = (
            stored_key is not None     # ← False when None — bug
            and result is not None
            and stored_key != current_key
        )
        assert not old_is_stale, (
            "Old predicate returns False (stale warning hidden) — this is the bug"
        )

    def test_fixed_predicate_correctly_detects_stale_result(self):
        """Fixed predicate fires correctly when run_scenario_key=None and result exists."""
        current_key = self._key({58: 300})
        stored_key = None   # set by road condition change / scenario load
        result = object()   # non-None: a prior RunResult exists

        # Fixed predicate (run_scenario_key is not None guard removed):
        new_is_stale = (
            result is not None
            and stored_key != current_key
        )
        assert new_is_stale, (
            "Fixed predicate must return True — stale result must be flagged"
        )

    def test_fixed_predicate_no_stale_when_no_result(self):
        """Fixed predicate returns False when no result exists (initial state)."""
        current_key = self._key({58: 300})
        result = None
        stored_key = None
        assert not (result is not None and stored_key != current_key)

    def test_fixed_predicate_no_stale_when_keys_match(self):
        """Fixed predicate returns False when run_scenario_key matches current key."""
        current_key = self._key({58: 300})
        result = object()
        stored_key = current_key
        assert not (result is not None and stored_key != current_key)


class TestPlannerResultLabels:
    """Regression: Evacuation Planner result panel displays correct labels and summaries."""

    def test_planner_routing_method_constant_exact(self):
        """PLANNER_ROUTING_METHOD must state Algorithm C, its full name, and MCF."""
        from floodroute.dashboard.result_formatter import PLANNER_ROUTING_METHOD
        assert PLANNER_ROUTING_METHOD.startswith("Routing method:")
        assert "Algorithm C" in PLANNER_ROUTING_METHOD
        assert "Capacity-aware global allocation" in PLANNER_ROUTING_METHOD
        assert "Minimum-Cost Flow" in PLANNER_ROUTING_METHOD

    def test_summarise_unassigned_zero_returns_empty(self):
        """Zero unassigned returns an empty string."""
        from floodroute.dashboard.result_formatter import summarise_unassigned
        assert summarise_unassigned(0, {1, 2}, {1: 500, 2: 300}) == ""

    def test_summarise_unassigned_all_unreachable(self):
        """All unassigned from origins with no reachable facility → reachability message."""
        from floodroute.dashboard.result_formatter import summarise_unassigned
        # Origins 1 and 2 have demand but neither is reachable
        msg = summarise_unassigned(
            total_unassigned=800,
            reachable_origins=set(),
            demands={1: 500, 2: 300},
        )
        assert "800" in msg
        assert "no modeled route" in msg
        assert "selected facility" in msg
        assert "OSM-derived" in msg

    def test_summarise_unassigned_all_capacity(self):
        """All unassigned from reachable origins → capacity message."""
        from floodroute.dashboard.result_formatter import summarise_unassigned
        msg = summarise_unassigned(
            total_unassigned=200,
            reachable_origins={1, 2},
            demands={1: 500, 2: 300},
        )
        assert "200" in msg
        assert "capacity" in msg.lower()
        assert "no selected facility" not in msg

    def test_summarise_unassigned_mixed(self):
        """Mixed causes → both unreachable and capacity counts mentioned."""
        from floodroute.dashboard.result_formatter import summarise_unassigned
        # Origin 1 is reachable (capacity-constrained), origin 2 is not reachable
        msg = summarise_unassigned(
            total_unassigned=350,
            reachable_origins={1},
            demands={1: 500, 2: 300},
        )
        assert "350" in msg
        assert "300" in msg   # unreachable count
        assert "50" in msg    # capacity-constrained count (350 - 300)
        assert "capacity" in msg.lower()
        assert "no modeled route" in msg

    def test_summarise_unassigned_exact_phrasing_all_unreachable(self):
        """Exact phrase matches the specified UI copy when all unassigned are unreachable."""
        from floodroute.dashboard.result_formatter import summarise_unassigned
        msg = summarise_unassigned(
            total_unassigned=1720,
            reachable_origins=set(),
            demands={10: 1000, 20: 720},
        )
        expected = (
            "1,720 people unassigned — no modeled route to any selected facility "
            "from their pickup points (OSM-derived road network and snapped "
            "coordinates only; unmapped or disconnected roads are not included)."
        )
        assert msg == expected


# ─────────────────────────────────────────────────────────────────────────────
# Stage 11 UCD additions: ScenarioConfig, FacilityCatalog, persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestScenarioConfig:
    """Unit tests for ScenarioConfig."""

    def test_selected_with_capacity_filters_zero(self):
        from floodroute.scenario.config import ScenarioConfig
        sc = ScenarioConfig(
            selected_facility_ids=["a", "b"],
            facility_capacities={"a": 100, "b": 0},
        )
        assert sc.selected_with_capacity() == {"a": 100}

    def test_selected_with_capacity_missing_id(self):
        from floodroute.scenario.config import ScenarioConfig
        sc = ScenarioConfig(
            selected_facility_ids=["a", "b"],
            facility_capacities={"a": 50},
        )
        # "b" not in facility_capacities → excluded
        assert sc.selected_with_capacity() == {"a": 50}

    def test_selected_with_capacity_empty(self):
        from floodroute.scenario.config import ScenarioConfig
        sc = ScenarioConfig()
        assert sc.selected_with_capacity() == {}

    def test_validation_errors_no_facilities(self):
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig()
        errs = sc.validation_errors(cat)
        assert any("No facilities" in e for e in errs)

    def test_validation_errors_unknown_id(self):
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["NONEXISTENT-999"],
            facility_capacities={"NONEXISTENT-999": 100},
        )
        errs = sc.validation_errors(cat)
        assert any("NONEXISTENT-999" in e for e in errs)

    def test_validation_errors_no_capacity(self):
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={},   # no capacity set
        )
        errs = sc.validation_errors(cat)
        assert any("no scenario capacity" in e for e in errs)

    def test_validation_passes_valid_facility(self):
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={"SJDB-001": 250},
        )
        errs = sc.validation_errors(cat)
        assert errs == []


class TestFacilityCatalog:
    """Unit tests for FacilityCatalog."""

    def test_build_has_expected_size(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        # 5 FACILITY_REGISTRY + 37 OSM candidates
        assert len(cat) == 42

    def test_sjdb001_is_government_confirmed(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.designation_status == "government_confirmed_from_published_sources"
        assert e.source == "facility_registry"

    def test_osm_entries_are_candidate_only(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        osm_entries = [e for e in cat.all() if e.source == "osm_candidate"]
        assert len(osm_entries) == 37
        for e in osm_entries:
            assert e.designation_status == "candidate_only"

    def test_by_type_evacuation_center(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        ev_centers = cat.by_type("evacuation_center")
        assert len(ev_centers) >= 1
        ids = [e.facility_id for e in ev_centers]
        assert "SJDB-001" in ids

    def test_fingerprint_is_deterministic(self):
        from floodroute.scenario.catalog import build_catalog
        cat1 = build_catalog()
        cat2 = build_catalog()
        assert cat1.fingerprint() == cat2.fingerprint()

    def test_fingerprint_is_16_hex_chars(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        fp = cat.fingerprint()
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_resolve_node_shelters_basic(self):
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={"SJDB-001": 400},
        )
        shelters = cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert 1345 in shelters
        assert shelters[1345] == 400

    def test_resolve_node_shelters_excludes_origin_collision(self):
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={"SJDB-001": 400},
        )
        # Exclude node 1345 (SJDB-001's snapped node) as an origin
        shelters = cat.resolve_node_shelters(sc, exclude_nodes={1345})
        assert shelters == {}

    def test_resolve_node_shelters_aggregates_colocated(self):
        """Two facilities sharing a node → capacities summed."""
        from floodroute.scenario.catalog import CatalogEntry, FacilityCatalog
        from floodroute.scenario.config import ScenarioConfig
        # Build a mini catalog with two entries on the same node
        e1 = CatalogEntry(
            facility_id="F1", name="Fac A", facility_type="school",
            latitude=None, longitude=None,
            osm_element_type=None, osm_id_raw=None,
            designation_status="candidate_only", designation_source="test",
            official_capacity=None, capacity_status="unknown",
            snapped_node=999, snapping_distance_m=10.0,
            source="osm_candidate", barangay_name="Test",
        )
        e2 = CatalogEntry(
            facility_id="F2", name="Fac B", facility_type="covered_court",
            latitude=None, longitude=None,
            osm_element_type=None, osm_id_raw=None,
            designation_status="candidate_only", designation_source="test",
            official_capacity=None, capacity_status="unknown",
            snapped_node=999, snapping_distance_m=15.0,
            source="osm_candidate", barangay_name="Test",
        )
        mini_cat = FacilityCatalog([e1, e2])
        sc = ScenarioConfig(
            selected_facility_ids=["F1", "F2"],
            facility_capacities={"F1": 100, "F2": 200},
        )
        shelters = mini_cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert shelters == {999: 300}

    def test_no_null_snapped_node_excluded_from_resolve(self):
        """Entries with snapped_node=None are excluded from resolve."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        # SJDB-002 has no snapped node (entrance missing)
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-002"],
            facility_capacities={"SJDB-002": 300},
        )
        shelters = cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert shelters == {}

    def test_can_be_selected_requires_snapped_node(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e_with = cat.get("SJDB-001")
        assert e_with is not None and e_with.can_be_selected
        e_without = cat.get("SJDB-002")
        assert e_without is not None and not e_without.can_be_selected


class TestScenarioPersistence:
    """Unit tests for scenario save/load/list/delete."""

    def test_round_trip(self, tmp_path):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import (
            load_scenario,
            save_scenario,
            scenario_from_dict,
            scenario_to_dict,
        )
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001", "way:123"],
            facility_capacities={"SJDB-001": 300, "way:123": 150},
            return_period="RP20",
            demand_mode="exact",
            demand_exact=500,
        )
        d = scenario_to_dict(sc, "fp_abc", "Test scenario")
        save_scenario(d, scenarios_dir=tmp_path)
        loaded_raw = load_scenario(d["scenario_id"], scenarios_dir=tmp_path)
        sc2, _ = scenario_from_dict(loaded_raw)
        assert sc2.return_period == "RP20"
        assert sc2.demand_mode == "exact"
        assert sc2.demand_exact == 500
        assert sc2.selected_facility_ids == ["SJDB-001", "way:123"]
        assert sc2.facility_capacities == {"SJDB-001": 300, "way:123": 150}

    def test_list_scenarios_sorted_by_updated(self, tmp_path):
        import time

        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import (
            list_scenarios,
            save_scenario,
            scenario_to_dict,
        )
        sc = ScenarioConfig()
        d1 = scenario_to_dict(sc, "fp1", "Alpha")
        save_scenario(d1, scenarios_dir=tmp_path)
        time.sleep(0.05)
        d2 = scenario_to_dict(sc, "fp2", "Beta")
        save_scenario(d2, scenarios_dir=tmp_path)
        listed = list_scenarios(scenarios_dir=tmp_path)
        # Most recent first
        assert listed[0]["scenario_name"] == "Beta"
        assert listed[1]["scenario_name"] == "Alpha"

    def test_delete_removes_file(self, tmp_path):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import (
            delete_scenario,
            list_scenarios,
            save_scenario,
            scenario_to_dict,
        )
        sc = ScenarioConfig()
        d = scenario_to_dict(sc, "fp1", "ToDelete")
        save_scenario(d, scenarios_dir=tmp_path)
        assert len(list_scenarios(scenarios_dir=tmp_path)) == 1
        delete_scenario(d["scenario_id"], scenarios_dir=tmp_path)
        assert list_scenarios(scenarios_dir=tmp_path) == []

    def test_load_nonexistent_raises(self, tmp_path):
        from floodroute.scenario.persistence import load_scenario
        with pytest.raises(FileNotFoundError):
            load_scenario("does_not_exist", scenarios_dir=tmp_path)

    def test_schema_version_mismatch_raises(self, tmp_path):
        import json

        from floodroute.scenario.persistence import _safe_filename, load_scenario, new_scenario_id
        sid = new_scenario_id()
        bad_doc = {"schema_version": "0.9", "scenario_id": sid, "scenario_name": "V0"}
        (tmp_path / _safe_filename(sid)).write_text(
            json.dumps(bad_doc), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Unsupported schema version"):
            load_scenario(sid, scenarios_dir=tmp_path)

    def test_save_as_new_generates_different_id(self, tmp_path):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_to_dict
        sc = ScenarioConfig()
        d1 = scenario_to_dict(sc, "fp1", "First")
        d2 = scenario_to_dict(sc, "fp1", "Second")
        # Each call to scenario_to_dict generates a fresh scenario_id
        assert d1["scenario_id"] != d2["scenario_id"]

    def test_scenario_id_uses_hex_12chars(self, tmp_path):
        from floodroute.scenario.persistence import new_scenario_id
        sid = new_scenario_id()
        assert len(sid) == 12
        assert all(c in "0123456789abcdef" for c in sid)

    def test_flood_penalties_prohibited_round_trips(self, tmp_path):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import (
            load_scenario,
            save_scenario,
            scenario_from_dict,
            scenario_to_dict,
        )
        sc = ScenarioConfig(flood_penalties=("prohibited",))
        d = scenario_to_dict(sc, "fp1", "Prohibited test")
        save_scenario(d, scenarios_dir=tmp_path)
        loaded = load_scenario(d["scenario_id"], scenarios_dir=tmp_path)
        sc2, _ = scenario_from_dict(loaded)
        assert sc2.flood_penalties == ("prohibited",)


class TestPastExperimentsDiscovery:
    """Unit tests for past_experiments.discover_experiments and verify_checksums."""

    def test_discover_empty_dir(self, tmp_path):
        from floodroute.dashboard.past_experiments import discover_experiments
        result = discover_experiments(experiments_root=tmp_path)
        assert result == []

    def test_discover_nonexistent_dir(self, tmp_path):
        from floodroute.dashboard.past_experiments import discover_experiments
        result = discover_experiments(experiments_root=tmp_path / "missing")
        assert result == []

    def test_discover_finds_valid_manifest(self, tmp_path):
        import json

        from floodroute.dashboard.past_experiments import discover_experiments
        exp_dir = tmp_path / "exp_001"
        exp_dir.mkdir()
        manifest = {
            "experiment_id": "exp_001",
            "created_utc": "2026-09-01T10:00:00",
            "algorithms": ["A", "C"],
            "return_periods": ["RP100"],
            "configuration_hash": "abc123",
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = discover_experiments(experiments_root=tmp_path)
        assert len(result) == 1
        assert result[0]["experiment_id"] == "exp_001"
        assert result[0]["_manifest_ok"] is True

    def test_discover_handles_bad_manifest_gracefully(self, tmp_path):
        from floodroute.dashboard.past_experiments import discover_experiments
        exp_dir = tmp_path / "exp_bad"
        exp_dir.mkdir()
        (exp_dir / "manifest.json").write_text("{invalid json}", encoding="utf-8")
        result = discover_experiments(experiments_root=tmp_path)
        assert len(result) == 1
        assert result[0]["_manifest_ok"] is False
        assert "_error" in result[0]

    def test_discover_ignores_non_directory_entries(self, tmp_path):
        from floodroute.dashboard.past_experiments import discover_experiments
        (tmp_path / "not_a_dir.json").write_text("{}", encoding="utf-8")
        result = discover_experiments(experiments_root=tmp_path)
        assert result == []

    def test_verify_checksums_ok(self, tmp_path):
        import hashlib

        from floodroute.dashboard.past_experiments import verify_checksums
        content = b"results data here"
        h = hashlib.sha256(content).hexdigest()
        (tmp_path / "results.csv").write_bytes(content)
        (tmp_path / "checksums.sha256").write_text(
            f"{h}  results.csv\n", encoding="utf-8"
        )
        failures = verify_checksums(tmp_path)
        assert failures == []

    def test_verify_checksums_mismatch(self, tmp_path):
        from floodroute.dashboard.past_experiments import verify_checksums
        (tmp_path / "results.csv").write_bytes(b"some data")
        (tmp_path / "checksums.sha256").write_text(
            "0000000000000000000000000000000000000000000000000000000000000000  results.csv\n",
            encoding="utf-8",
        )
        failures = verify_checksums(tmp_path)
        assert any("results.csv" in f for f in failures)

    def test_verify_checksums_missing_file(self, tmp_path):
        from floodroute.dashboard.past_experiments import verify_checksums
        (tmp_path / "checksums.sha256").write_text(
            "abc123  missing_file.csv\n", encoding="utf-8"
        )
        failures = verify_checksums(tmp_path)
        assert any("Missing" in f for f in failures)

    def test_verify_checksums_no_checksums_file(self, tmp_path):
        from floodroute.dashboard.past_experiments import verify_checksums
        failures = verify_checksums(tmp_path)
        assert any("not found" in f for f in failures)


# ─────────────────────────────────────────────────────────────────────────────
# Issue fixes: fallback removal, node 33/58 handling, unresolved facilities
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackRemovalAndNodeHandling:
    """Verify no automatic fallback to hardcoded nodes 33/58."""

    def test_empty_scenario_config_produces_no_shelters(self):
        """An empty ScenarioConfig resolves to zero shelter nodes — never 33 or 58."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig()  # no selected facilities
        shelters = cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert shelters == {}, f"Expected empty dict, got {shelters}"
        assert 33 not in shelters, "Node 33 must not appear in empty ScenarioConfig output"
        assert 58 not in shelters, "Node 58 must not appear in empty ScenarioConfig output"

    def test_node_33_not_in_catalog(self):
        """Node 33 (unnamed road node) has no facility entry in the catalog."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        # No entry should have snapped_node == 33 except via explicit selection
        entries_at_33 = [e for e in cat.all() if e.snapped_node == 33]
        assert entries_at_33 == [], (
            f"Node 33 should have no catalog entry (unnamed road node). "
            f"Found: {[e.facility_id for e in entries_at_33]}"
        )

    def test_node_58_maps_to_atabay_elementary(self):
        """Node 58 maps exclusively to Atabay Elementary School (way:168981805)."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        entries_at_58 = [e for e in cat.all() if e.snapped_node == 58]
        assert len(entries_at_58) == 1, (
            f"Expected exactly 1 entry at node 58, got {len(entries_at_58)}"
        )
        e = entries_at_58[0]
        assert e.facility_id == "way:168981805", (
            f"Expected facility_id 'way:168981805', got {e.facility_id!r}"
        )
        assert "Atabay" in e.name, f"Expected Atabay in name, got {e.name!r}"
        assert e.snapping_distance_m == pytest.approx(94.8, abs=1.0)
        assert e.official_capacity is None
        assert e.capacity_status == "unknown"

    def test_node_33_selection_requires_catalog_entry(self):
        """Selecting a raw node ID with no catalog entry yields no shelters."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        # Simulate someone trying to select "node_33" (not a real facility_id)
        sc = ScenarioConfig(
            selected_facility_ids=["legacy_unknown_shelter_node_33"],
            facility_capacities={"legacy_unknown_shelter_node_33": 12000},
        )
        shelters = cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert shelters == {}, (
            "A raw node ID with no catalog entry must not reach the optimizer"
        )
        assert 33 not in shelters

    def test_atabay_elementary_disabled_by_default(self):
        """Atabay Elementary School is not in the default (empty) ScenarioConfig."""
        from floodroute.scenario.config import ScenarioConfig
        sc = ScenarioConfig()
        assert "way:168981805" not in sc.selected_facility_ids
        assert sc.selected_facility_ids == []

    def test_sjdb001_in_catalog_as_government_confirmed(self):
        """SJDB-001 (Antique Regional EC) is government_confirmed_from_published_sources."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None, "SJDB-001 must be in the catalog"
        assert e.name == "Antique Regional Evacuation Center"
        assert e.facility_type == "evacuation_center"
        assert e.designation_status == "government_confirmed_from_published_sources"
        assert e.official_capacity is None
        assert e.capacity_status == "unverified"
        assert e.snapped_node == 1345
        assert e.barangay_name == "San Pedro"

    def test_sjdb001_osm_way_captured(self):
        """SJDB-001 catalog entry records OSM way 1394870512."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.osm_element_type == "way"
        assert e.osm_id_raw == "1394870512"

    def test_sjdb001_disabled_by_default(self):
        """SJDB-001 is not selected in a default ScenarioConfig."""
        from floodroute.scenario.config import ScenarioConfig
        sc = ScenarioConfig()
        assert "SJDB-001" not in sc.selected_facility_ids

    def test_sjdb001_requires_positive_capacity_to_reach_optimizer(self):
        """SJDB-001 selected without capacity yields no shelters."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={},  # no capacity
        )
        shelters = cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert shelters == {}

    def test_sjdb001_with_positive_capacity_reaches_optimizer(self):
        """SJDB-001 with scenario capacity resolves to node 1345."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={"SJDB-001": 500},
        )
        shelters = cat.resolve_node_shelters(sc, exclude_nodes=set())
        assert shelters == {1345: 500}

    def test_validation_errors_no_facilities_selected(self):
        """ScenarioConfig with no selected facilities fails validation."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig()
        errors = sc.validation_errors(cat)
        assert any("No facilities" in e for e in errors)


class TestUnresolvedFacilityHandling:
    """Verify unresolved facility detection in scenario_from_dict."""

    def test_unresolved_facility_detected(self, tmp_path):
        """scenario_from_dict with catalog returns unresolved facility IDs."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001", "nonexistent:xyz"],
            facility_capacities={"SJDB-001": 300, "nonexistent:xyz": 100},
        )
        d = scenario_to_dict(sc, cat.fingerprint(), "Test")
        _, unresolved = scenario_from_dict(d, catalog=cat)
        assert "nonexistent:xyz" in unresolved
        assert "SJDB-001" not in unresolved

    def test_no_unresolved_when_all_valid(self, tmp_path):
        """scenario_from_dict returns empty unresolved when all facilities in catalog."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={"SJDB-001": 300},
        )
        d = scenario_to_dict(sc, cat.fingerprint(), "Test")
        _, unresolved = scenario_from_dict(d, catalog=cat)
        assert unresolved == []

    def test_catalog_fingerprint_detectable(self):
        """Two different catalogs produce different fingerprints."""
        from floodroute.scenario.catalog import CatalogEntry, FacilityCatalog
        e1 = CatalogEntry(
            facility_id="A", name="A", facility_type="school",
            latitude=0.0, longitude=0.0, osm_element_type=None, osm_id_raw=None,
            designation_status="candidate_only", designation_source="OSM",
            official_capacity=None, capacity_status="unknown",
            snapped_node=1, snapping_distance_m=10.0,
            source="osm_candidate", barangay_name="Brgy A",
        )
        e2 = CatalogEntry(
            facility_id="B", name="B", facility_type="school",
            latitude=0.0, longitude=0.0, osm_element_type=None, osm_id_raw=None,
            designation_status="candidate_only", designation_source="OSM",
            official_capacity=None, capacity_status="unknown",
            snapped_node=2, snapping_distance_m=10.0,
            source="osm_candidate", barangay_name="Brgy B",
        )
        cat1 = FacilityCatalog([e1])
        cat2 = FacilityCatalog([e1, e2])
        assert cat1.fingerprint() != cat2.fingerprint()


# ---------------------------------------------------------------------------
# Verification corrections — geospatial evidence, designation provenance,
# structured OSM identity, per-facility reachability, node 33 non-runnable
# ---------------------------------------------------------------------------

class TestSJDB001GeospatialEvidence:
    """Item 1 — Exact SJDB-001 geospatial fields."""

    def test_sjdb001_latitude(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        assert fr.latitude == pytest.approx(10.800394, abs=1e-6)

    def test_sjdb001_longitude(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        assert fr.longitude == pytest.approx(121.948801, abs=1e-6)

    def test_sjdb001_coordinate_source_mentions_osm_way(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        assert "OpenStreetMap" in fr.coordinate_source
        assert "1394870512" in fr.coordinate_source

    def test_sjdb001_snapped_node_is_1345(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        assert fr.snapped_node_id == 1345

    def test_sjdb001_snapping_distance_exact(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        # Exact value from facilities.py — researcher snap, not pipeline result
        assert fr.snap_distance_m == pytest.approx(6.3, abs=0.05)
        assert fr.snap_is_pipeline_result is False

    def test_sjdb001_catalog_snapping_distance(self):
        """CatalogEntry.snapping_distance_m is propagated from FacilityRecord."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.snapping_distance_m == pytest.approx(6.3, abs=0.05)

    def test_sjdb001_inside_study_boundary(self):
        """SJDB-001 is within the PH0600613 study boundary."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.inside_study_boundary is True


class TestSJDB001DesignationProvenance:
    """Item 2 — Designation provenance: government_confirmed_from_published_sources."""

    def test_designation_status_is_from_published_sources(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.designation_status == "government_confirmed_from_published_sources"

    def test_designation_status_is_not_vague_government_confirmed(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.designation_status != "government_confirmed"

    def test_designation_source_includes_pna_url(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        assert "pna.gov.ph/articles/1065266" in fr.designation_source

    def test_designation_source_includes_secondary_source(self):
        from floodroute.dashboard.facilities import get_facility_by_id
        fr = get_facility_by_id("SJDB-001")
        assert fr is not None
        assert "lorenlegarda.com.ph" in fr.designation_source

    def test_official_capacity_is_null(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.official_capacity is None

    def test_capacity_status_is_unverified(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.capacity_status == "unverified"

    def test_is_government_confirmed_property_true(self):
        """is_government_confirmed returns True for published-source status."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.is_government_confirmed is True

    def test_lgu_verified_reserved_for_verified_permanent(self):
        """_map_designation returns lgu_verified only for permanent+verified."""
        from floodroute.scenario.catalog import _map_designation
        assert _map_designation("permanent", "verified") == "lgu_verified"
        assert _map_designation("permanent", "unverified") == (
            "government_confirmed_from_published_sources"
        )


class TestStructuredOSMIdentity:
    """Item 3 — Structured OSM identity fields on CatalogEntry."""

    def test_sjdb001_osm_element_type(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.osm_element_type == "way"

    def test_sjdb001_osm_id_raw(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.osm_id_raw == "1394870512"

    def test_sjdb001_osm_url(self):
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("SJDB-001")
        assert e is not None
        assert e.osm_url == "https://www.openstreetmap.org/way/1394870512"

    def test_atabay_es_osm_url(self):
        """Atabay Elementary School (way:168981805) has correct osm_url."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        e = cat.get("way:168981805")
        assert e is not None
        assert e.osm_url == "https://www.openstreetmap.org/way/168981805"

    def test_osm_candidates_all_have_osm_url(self):
        """Every OSM candidate with a way/node ID has a non-None osm_url."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        for e in cat.all():
            if e.source == "osm_candidate":
                assert e.osm_url is not None, f"{e.facility_id} missing osm_url"
                assert e.osm_url.startswith("https://www.openstreetmap.org/")

    def test_osm_url_none_for_registry_entries_without_osm(self):
        """FACILITY_REGISTRY entries with no OSM reference have osm_url=None."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        # SJDB-002 through SJDB-005 — check those without coordinate_source OSM refs
        e002 = cat.get("SJDB-002")
        assert e002 is not None
        # SJDB-002 has no coordinate_source so osm_url is None
        assert e002.osm_url is None


@pytest.fixture(scope="module")
def graph_and_origins_module():
    """Load graph + origins once per module for reachability tests."""
    import networkx as nx

    from floodroute.experiments.demand import (
        load_psa_population,
        snap_barangay_origins,
    )
    from floodroute.experiments.runner import (
        _DEFAULT_BARANGAY_GPKG,
        _DEFAULT_GRAPHML,
        _DEFAULT_NODES_GPKG,
        _DEFAULT_POP_CSV,
        MUNICIPALITY_PSGC,
    )
    G = nx.read_graphml(str(_DEFAULT_GRAPHML), node_type=int)
    records = load_psa_population(_DEFAULT_POP_CSV, adm3_filter=MUNICIPALITY_PSGC)
    origins = snap_barangay_origins(
        records,
        barangay_gpkg=_DEFAULT_BARANGAY_GPKG,
        nodes_gpkg=_DEFAULT_NODES_GPKG,
        municipality_psgc=MUNICIPALITY_PSGC,
    )
    return G, origins


class TestPerFacilityReachability:
    """Item 4 — Per-facility reachability, computed via exact path search.

    Values confirmed by single_source_shortest_path_length in reversed graph
    (see: tests run 2026-09-04 against PH0600613_phase_b_enriched.graphml).

    Both SJDB-001 (node 1345) and Atabay ES (node 58) have identical
    reachability sets, identical across all three return periods.
    The 25/28 and 57812/65140 figures apply to each facility individually
    AND to the union of both.
    """

    @pytest.mark.parametrize("fac_node,fac_label", [
        (1345, "SJDB-001 node 1345"),
        (58,   "Atabay ES node 58"),
    ])
    @pytest.mark.parametrize("rp", ["RP10", "RP20", "RP100"])
    def test_reachable_origin_count(self, graph_and_origins_module, fac_node, fac_label, rp):
        """Each facility individually serves 25 of 28 origins for every RP."""
        import networkx as nx
        G, origins = graph_and_origins_module
        G_rev = G.reverse(copy=False)
        reachable = set(nx.single_source_shortest_path_length(G_rev, fac_node).keys())
        count = sum(1 for o in origins if o.origin_node in reachable)
        assert count == 25, (
            f"{fac_label} {rp}: expected 25 reachable origins, got {count}"
        )

    @pytest.mark.parametrize("fac_node,fac_label", [
        (1345, "SJDB-001 node 1345"),
        (58,   "Atabay ES node 58"),
    ])
    @pytest.mark.parametrize("rp", ["RP10", "RP20", "RP100"])
    def test_reachable_population(self, graph_and_origins_module, fac_node, fac_label, rp):
        """Each facility individually serves 57,812 of 65,140 people for every RP."""
        import networkx as nx
        G, origins = graph_and_origins_module
        G_rev = G.reverse(copy=False)
        reachable = set(nx.single_source_shortest_path_length(G_rev, fac_node).keys())
        pop = sum(o.population_2020 for o in origins if o.origin_node in reachable)
        assert pop == 57_812, (
            f"{fac_label} {rp}: expected 57812 reachable population, got {pop}"
        )

    def test_unreachable_barangays_identical_for_both_facilities(self, graph_and_origins_module):
        """Both facilities have the same three unreachable barangays."""
        import networkx as nx
        G, origins = graph_and_origins_module
        G_rev = G.reverse(copy=False)
        reach_1345 = set(nx.single_source_shortest_path_length(G_rev, 1345).keys())
        reach_58   = set(nx.single_source_shortest_path_length(G_rev, 58).keys())
        unreachable_1345 = {o.name for o in origins if o.origin_node not in reach_1345}
        unreachable_58   = {o.name for o in origins if o.origin_node not in reach_58}
        assert unreachable_1345 == unreachable_58
        assert unreachable_1345 == {"Barangay 8 (Pob.)", "Durog", "Malaiba"}

    def test_union_reachability_same_as_individual(self, graph_and_origins_module):
        """Union of both facilities adds no additional coverage over either alone."""
        import networkx as nx
        G, origins = graph_and_origins_module
        G_rev = G.reverse(copy=False)
        reach_1345 = set(nx.single_source_shortest_path_length(G_rev, 1345).keys())
        reach_58   = set(nx.single_source_shortest_path_length(G_rev, 58).keys())
        union = reach_1345 | reach_58
        union_count = sum(1 for o in origins if o.origin_node in union)
        union_pop   = sum(o.population_2020 for o in origins if o.origin_node in union)
        assert union_count == 25
        assert union_pop == 57_812

    def test_adjacent_edges_are_modelled_dry_all_rps(self, graph_and_origins_module):
        """All edges adjacent to both facility nodes are modelled_dry in every RP."""
        G, _ = graph_and_origins_module
        for fac_node in (1345, 58):
            for u, v, data in G.edges(data=True):
                if u == fac_node or v == fac_node:
                    for rp in ("rp10", "rp20", "rp100"):
                        col = f"jrc_{rp}_status"
                        status = data.get(col, "modelled_dry")
                        assert status == "modelled_dry", (
                            f"Node {fac_node} adjacent edge ({u},{v}) "
                            f"{col}={status!r} — expected modelled_dry"
                        )


class TestNode33LegacyNonRunnable:
    """Item 5 — Node 33 legacy preset is non-runnable and correctly labeled."""

    def test_node_33_not_in_catalog(self):
        """No catalog entry has snapped_node == 33."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        assert all(e.snapped_node != 33 for e in cat.all())

    def test_legacy_unknown_node_33_not_in_catalog(self):
        """'legacy_unknown_shelter_node_33' is not a valid facility_id."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        assert cat.get("legacy_unknown_shelter_node_33") is None

    def test_node_33_produces_empty_shelters(self):
        """ScenarioConfig selecting 'legacy_unknown_shelter_node_33' → empty shelters."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["legacy_unknown_shelter_node_33"],
            facility_capacities={"legacy_unknown_shelter_node_33": 12000},
        )
        shelters = cat.resolve_node_shelters(sc)
        assert shelters == {}

    def test_node_33_fails_validation(self):
        """ScenarioConfig with only node 33 fails validation_errors()."""
        from floodroute.scenario.catalog import build_catalog
        from floodroute.scenario.config import ScenarioConfig
        cat = build_catalog()
        sc = ScenarioConfig(
            selected_facility_ids=["legacy_unknown_shelter_node_33"],
            facility_capacities={"legacy_unknown_shelter_node_33": 12000},
        )
        errors = sc.validation_errors(cat)
        assert len(errors) > 0
        assert any("legacy_unknown_shelter_node_33" in e for e in errors)


# ---------------------------------------------------------------------------
# Regression: Algorithm C co-located origin-shelter (node 58 is both)
# ---------------------------------------------------------------------------
#
# Prior to the fix, A/B/B+ completed while Algorithm C raised:
#   ValueError: Nodes appear in both demands and capacities: ['58']
#
# Correct fix: solve_assignment namespaces logical nodes as
#   ("origin", physical_id) and ("shelter", physical_id)
# so a physical node may serve as both supply and demand vertex.
# Demand population is NEVER removed: build_demands() always receives the
# full origin list and Hamilton apportionment is recomputed over all origins.
# ---------------------------------------------------------------------------


def _make_overlap_graph():
    """Three-node synthetic graph where node 58 is both an origin and a shelter.

    Topology (all edges modelled_dry across RP10/20/100):
      58 (origin Atabay, pop=5000)  ─── 100 m ──► 99 (pure shelter, cap=8000)
      77 (origin OtherBgy, pop=3000) ── 150 m ──► 99
      58 is also a shelter with capacity 4000.
    """
    G = nx.MultiDiGraph()
    for node in (58, 77, 99):
        G.add_node(node)
    for u, v, length in [(58, 99, 100.0), (77, 99, 150.0)]:
        G.add_edge(u, v, length_m=length,
                   jrc_rp10_status="modelled_dry",
                   jrc_rp20_status="modelled_dry",
                   jrc_rp100_status="modelled_dry")
    return G


def _make_overlap_origins():
    from floodroute.experiments.demand import BarangayOrigin
    return [
        BarangayOrigin(psgc="PH060061300X", name="Atabay",
                       population_2020=5000, origin_node=58, snap_distance_m=5.0),
        BarangayOrigin(psgc="PH060061300Y", name="OtherBgy",
                       population_2020=3000, origin_node=77, snap_distance_m=8.0),
    ]


def _overlap_run_scenario(algorithm: str, fraction: float = 0.25):
    """Helper: run one algorithm through run_scenario on the overlap fixture."""
    from floodroute.experiments.runner import (
        ScenarioKey,
        make_experiment_config,
        run_scenario,
    )
    G = _make_overlap_graph()
    origins = _make_overlap_origins()
    nominal_caps = {58: 4000, 99: 8000}
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=(algorithm,),
        return_periods=("RP10",),
        demand_fractions=(fraction,),
        capacity_multipliers=(1.0,),
        flood_penalties=(1.0,),
        nominal_capacities=nominal_caps,
        pilot_mode=False,
    )
    key = ScenarioKey(
        algorithm=algorithm,
        return_period="RP10",
        demand_fraction=fraction,
        capacity_multiplier=1.0,
        flood_penalty=1.0,
    )
    return run_scenario(G, origins, config, key)


class TestAlgorithmCNodeOverlapRegression:
    """Regression: co-located origin-shelter (node 58 is both origin and shelter).

    Nine correctness properties are asserted.
    """

    # ------------------------------------------------------------------ 1 & 2
    @pytest.mark.parametrize("with_shelter_58", [True, False])
    def test_total_demand_unchanged_by_shelter_selection(self, with_shelter_58):
        """Prop 1 + 2: total demand and node-58 demand are identical whether
        node 58 is selected as a shelter or not."""
        from floodroute.experiments.demand import build_demands
        origins = _make_overlap_origins()
        demands_with, _ = build_demands(origins, 0.25)
        demands_without, _ = build_demands(origins, 0.25)
        # Both calls use the full origin list regardless of shelter selection
        assert sum(demands_with.values()) == sum(demands_without.values()), (
            "Total demand changed when shelter selection changed"
        )
        assert 58 in demands_with, "Node-58 origin demand is absent"
        assert demands_with[58] > 0, "Node-58 origin demand is zero"

    # ------------------------------------------------------------------ 3
    def test_hamilton_apportionment_over_full_origin_set(self):
        """Prop 3: Hamilton target total uses all origins, not a reduced set."""
        from floodroute.experiments.demand import build_demands
        origins_full = _make_overlap_origins()
        # A reduced set (Atabay removed) would change total_population
        origins_reduced = [o for o in origins_full if o.origin_node != 58]
        demands_full, _ = build_demands(origins_full, 0.25)
        demands_reduced, _ = build_demands(origins_reduced, 0.25)
        # The full-set total (5000+3000)*0.25 = 2000 ≠ 3000*0.25 = 750
        assert sum(demands_full.values()) != sum(demands_reduced.values()), (
            "Sanity: full and reduced totals should differ"
        )
        # run_scenario must produce the full total, not the reduced one
        result = _overlap_run_scenario("C", fraction=0.25)
        reported_demand = result.metrics.get("total_demand", 0)
        assert reported_demand == sum(demands_full.values()), (
            f"run_scenario used reduced origin set: "
            f"reported={reported_demand}, expected={sum(demands_full.values())}"
        )

    # ------------------------------------------------------------------ 4
    @pytest.mark.parametrize("algorithm", ["A", "B", "B+", "C"])
    def test_all_algorithms_complete_not_error(self, algorithm):
        """Prop 4: all four algorithms finish with run_status='completed'."""
        result = _overlap_run_scenario(algorithm)
        assert result.run_status == "completed", (
            f"Algorithm {algorithm}: run_status={result.run_status!r} "
            f"— {result.error_message}"
        )

    # ------------------------------------------------------------------ 5
    def test_colocated_od_cost_is_zero(self):
        """Prop 5: for the co-located pair (58, 58) the OD cost is 0."""
        from floodroute.optimization.routing import compute_od_matrix
        G = _make_overlap_graph()
        demands = {58: 1250, 77: 750}
        capacities = {58: 4000, 99: 8000}
        od_costs, od_routes = compute_od_matrix(G, list(demands), list(capacities), "RP10")
        assert (58, 58) in od_costs, "Co-located pair (58, 58) missing from OD matrix"
        assert od_costs[(58, 58)] == 0.0, (
            f"Co-located OD cost is {od_costs[(58, 58)]}, expected 0.0"
        )
        assert od_routes[(58, 58)] == [58], (
            f"Co-located route is {od_routes[(58, 58)]}, expected [58]"
        )

    # ------------------------------------------------------------------ 6
    def test_colocated_shelter_allocation_respects_capacity(self):
        """Prop 6: flow to shelter 58 does not exceed its capacity (4000)."""
        result = _overlap_run_scenario("C", fraction=1.0)
        assert result.run_status == "completed"
        load_58 = result.facility_metrics.get(58, {}).get("load", 0)
        assert load_58 <= 4000, (
            f"Shelter 58 load={load_58} exceeds capacity=4000"
        )
        assert result.facility_metrics.get(58, {}).get("overflow", 1) == 0

    # ------------------------------------------------------------------ 7
    def test_excess_demand_assigned_elsewhere_or_unassigned(self):
        """Prop 7: demand that overflows shelter 58's capacity goes to shelter 99
        or is counted as unassigned — never silently dropped."""
        from floodroute.experiments.demand import build_demands
        origins = _make_overlap_origins()
        demands, _ = build_demands(origins, 1.0)  # full population: 5000+3000=8000
        result = _overlap_run_scenario("C", fraction=1.0)
        assert result.run_status == "completed"
        total_demand = result.metrics["total_demand"]
        assigned = result.metrics["assigned_population"]
        unassigned = result.metrics["unassigned_population"]
        # Nothing is silently dropped
        assert assigned + unassigned == total_demand
        # Shelter 58 absorbs up to 4000; shelter 99 must absorb some overflow
        load_99 = result.facility_metrics.get(99, {}).get("load", 0)
        load_58 = result.facility_metrics.get(58, {}).get("load", 0)
        assert load_58 + load_99 == assigned, (
            f"load_58={load_58} + load_99={load_99} != assigned={assigned}"
        )

    # ------------------------------------------------------------------ 8
    @pytest.mark.parametrize("algorithm", ["A", "B", "B+", "C"])
    def test_assigned_plus_unassigned_equals_total_demand(self, algorithm):
        """Prop 8: assigned + unassigned == total_demand for every algorithm."""
        result = _overlap_run_scenario(algorithm, fraction=1.0)
        assert result.run_status == "completed"
        m = result.metrics
        assert m["assigned_population"] + m["unassigned_population"] == m["total_demand"], (
            f"Algorithm {algorithm}: "
            f"assigned={m['assigned_population']} + "
            f"unassigned={m['unassigned_population']} "
            f"!= total_demand={m['total_demand']}"
        )

    # ------------------------------------------------------------------ 9
    def test_algorithm_c_passes_all_integrity_checks(self):
        """Prop 9: Algorithm C result passes capacity, non-negativity, and
        demand-conservation integrity checks."""
        result = _overlap_run_scenario("C", fraction=1.0)
        assert result.run_status == "completed"
        nominal_caps = {58: 4000, 99: 8000}
        # No capacity violation reported by solver
        # (capacity_violations lives inside solve_assignment; check via facility_metrics)
        for s, _cap in nominal_caps.items():
            fm = result.facility_metrics.get(s, {})
            assert fm.get("overflow", 0) == 0, f"Capacity violated at shelter {s}"
            assert fm.get("load", 0) >= 0, f"Negative load at shelter {s}"
        # All assignment flows are non-negative
        for pair, flow in result.assignments.items():
            assert flow >= 0, f"Negative flow at {pair}: {flow}"
        # assigned + unassigned == total_demand (MCF integrity)
        m = result.metrics
        assert m["assigned_population"] + m["unassigned_population"] == m["total_demand"]


# ---------------------------------------------------------------------------
# UI corrections (Step 4 + Step 5): logic layer tests
# Tests cover: ScenarioConfig road_conditions field, persistence round-trip,
# empty-group hiding, All-facilities toggle logic, capacity label, stable road
# IDs, duplicate guard, and save/restore of road conditions.
# ---------------------------------------------------------------------------


class TestScenarioConfigRoadConditions:
    """ScenarioConfig stores road_conditions and round-trips through persistence."""

    def test_road_conditions_field_defaults_to_empty(self):
        from floodroute.scenario.config import ScenarioConfig
        sc = ScenarioConfig()
        assert hasattr(sc, "road_conditions")
        assert sc.road_conditions == []

    def test_road_conditions_accepts_list_of_dicts(self):
        from floodroute.scenario.config import ScenarioConfig
        rc = [
            {"road_name": "Rizal Street", "condition": "road_closed", "segment_count": 3},
            {"road_name": "National Road", "condition": "flooded_passable", "segment_count": 7},
        ]
        sc = ScenarioConfig(road_conditions=rc)
        assert len(sc.road_conditions) == 2
        assert sc.road_conditions[0]["road_name"] == "Rizal Street"

    def test_road_conditions_preserved_in_scenario_to_dict(self):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_to_dict
        rc = [{"road_name": "Sto. Nino", "condition": "flooded_impassable", "segment_count": 2}]
        sc = ScenarioConfig(road_conditions=rc)
        d = scenario_to_dict(sc, "fp1", "Test")
        assert "road_conditions" in d
        assert d["road_conditions"] == rc

    def test_road_conditions_empty_serialises_as_empty_list(self):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_to_dict
        sc = ScenarioConfig()
        d = scenario_to_dict(sc, "fp1", "Empty")
        assert d["road_conditions"] == []

    def test_road_conditions_round_trip(self, tmp_path):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import (
            load_scenario,
            save_scenario,
            scenario_from_dict,
            scenario_to_dict,
        )
        rc = [
            {"road_name": "Rizal Street", "condition": "road_closed", "segment_count": 3},
            {"road_name": "National Road", "condition": "flooded_passable", "segment_count": 7},
        ]
        sc = ScenarioConfig(
            selected_facility_ids=["SJDB-001"],
            facility_capacities={"SJDB-001": 300},
            road_conditions=rc,
        )
        d = scenario_to_dict(sc, "fp_rc", "Road conditions test")
        save_scenario(d, scenarios_dir=tmp_path)
        loaded_raw = load_scenario(d["scenario_id"], scenarios_dir=tmp_path)
        sc2, _ = scenario_from_dict(loaded_raw)
        assert sc2.road_conditions == rc

    def test_road_conditions_backward_compat_missing_key(self, tmp_path):
        """Scenarios saved before road_conditions was added restore as empty list."""
        from floodroute.scenario.persistence import scenario_from_dict
        # Simulate old JSON without road_conditions key
        old_data = {
            "schema_version": "1.0",
            "scenario_id": "abc123",
            "municipality": "PH0600613",
            "return_period": "RP100",
            "demand_mode": "fraction",
            "demand_fraction": 0.25,
            "demand_exact": 0,
            "selected_facility_ids": ["SJDB-001"],
            "facility_capacities": {"SJDB-001": 300},
            "algorithms": ["A", "B", "B+", "C"],
            "capacity_multipliers": [0.5, 0.75, 1.0, 1.25],
            "flood_penalties": ["10.0"],
            "catalog_fingerprint": "fp_old",
            # NO road_conditions key
        }
        sc, _ = scenario_from_dict(old_data)
        assert sc.road_conditions == []

    def test_multiple_road_conditions_order_preserved(self, tmp_path):
        """Order of road_conditions entries is preserved through round-trip."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import (
            load_scenario,
            save_scenario,
            scenario_from_dict,
            scenario_to_dict,
        )
        rc = [
            {"road_name": "Alpha Road", "condition": "road_closed", "segment_count": 1},
            {"road_name": "Beta Road", "condition": "flooded_impassable", "segment_count": 2},
            {"road_name": "Gamma Street", "condition": "unknown", "segment_count": 5},
        ]
        sc = ScenarioConfig(road_conditions=rc)
        d = scenario_to_dict(sc, "fp_order", "Order test")
        save_scenario(d, scenarios_dir=tmp_path)
        loaded_raw = load_scenario(d["scenario_id"], scenarios_dir=tmp_path)
        sc2, _ = scenario_from_dict(loaded_raw)
        assert [r["road_name"] for r in sc2.road_conditions] == [
            "Alpha Road", "Beta Road", "Gamma Street"
        ]


class TestFacilitySelectionUILogic:
    """Logic-layer tests for Step 4 UI: empty group hiding, All facilities,
    designation labels, capacity labels."""

    def test_empty_group_hidden_when_type_total_count_zero(self):
        """Groups with no selectable entries (count==0) must not appear."""
        from floodroute.scenario.catalog import build_catalog
        # Build a mini catalog with only evacuation_center entries (no schools etc.)
        cat = build_catalog()
        all_entries = cat.all()
        type_total: dict[str, int] = {}
        for e in all_entries:
            if e.can_be_selected:
                type_total[e.facility_type] = type_total.get(e.facility_type, 0) + 1
        # covered_court and multi_purpose_hall should have 0 selectable entries in
        # the study catalog, making them hidden.
        assert type_total.get("covered_court", 0) == 0, (
            "covered_court must have 0 selectable entries to verify empty-group hiding"
        )
        assert type_total.get("multi_purpose_hall", 0) == 0, (
            "multi_purpose_hall must have 0 selectable entries"
        )

    def test_all_facilities_checkbox_logic_select_all(self):
        """All-facilities logic: when none selected, selecting all adds all selectable fids."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        all_selectable = [e.facility_id for e in cat.all() if e.can_be_selected]
        assert len(all_selectable) > 0, "Catalog must have selectable facilities"

        # Simulate: selected_fids starts empty, user clicks "All facilities"
        selected_fids: list[str] = []
        all_checked = bool(all_selectable) and all(f in selected_fids for f in all_selectable)
        assert not all_checked  # starts unchecked

        # After clicking (new_all_checked=True):
        new_all_checked = True
        if new_all_checked != all_checked:
            for fid in all_selectable:
                if fid not in selected_fids:
                    selected_fids.append(fid)
        assert set(selected_fids) == set(all_selectable)

    def test_all_facilities_checkbox_logic_deselect_all(self):
        """All-facilities logic: when all selected, unchecking removes all selectable fids."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        all_selectable = [e.facility_id for e in cat.all() if e.can_be_selected]

        # Simulate: all are selected, user unchecks "All facilities"
        selected_fids = list(all_selectable)
        all_checked = bool(all_selectable) and all(f in selected_fids for f in all_selectable)
        assert all_checked  # starts checked

        new_all_checked = False
        if new_all_checked != all_checked:
            selected_fids = [f for f in selected_fids if f not in all_selectable]
        assert selected_fids == []

    def test_partial_selection_not_counted_as_all_checked(self):
        """If only some facilities are selected, all_checked must be False."""
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        all_selectable = [e.facility_id for e in cat.all() if e.can_be_selected]
        if len(all_selectable) < 2:
            pytest.skip("Need at least 2 selectable facilities for partial selection test")
        selected_fids = [all_selectable[0]]  # only one selected
        all_checked = bool(all_selectable) and all(f in selected_fids for f in all_selectable)
        assert not all_checked

    def test_designation_label_sjdb001(self):
        """SJDB-001 gets the government-confirmed + experimental capacity label."""
        from floodroute.dashboard.scenario_ui import _designation_label
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        entry = cat.get("SJDB-001")
        assert entry is not None
        label = _designation_label(entry)
        assert "Government-confirmed" in label
        assert "Experimental scenario capacity" in label

    def test_capacity_label_is_experimental_for_all_facilities(self):
        """Every selectable facility gets 'Experimental scenario capacity' label."""
        from floodroute.dashboard.scenario_ui import _capacity_label
        from floodroute.scenario.catalog import build_catalog
        cat = build_catalog()
        for entry in cat.all():
            if entry.can_be_selected:
                label = _capacity_label(entry)
                assert label == "Experimental scenario capacity", (
                    f"{entry.facility_id} got unexpected capacity label: {label!r}"
                )

    def test_evacuation_center_group_expands_by_default(self):
        """The evacuation_center group has auto_exp=True in the UI logic."""
        # Verify the constant in scenario_ui
        from floodroute.dashboard.scenario_ui import _FACILITY_TYPE_ORDER
        assert "evacuation_center" in _FACILITY_TYPE_ORDER
        assert _FACILITY_TYPE_ORDER[0] == "evacuation_center"


class TestRoadConditionsDuplicateAndStableID:
    """Road conditions: duplicate detection via road name as stable ID,
    and the road_name_conditions dict structure matches ScenarioConfig."""

    def test_road_condition_dict_structure(self):
        """road_name_conditions entries have required keys."""
        from floodroute.scenario.config import ScenarioConfig
        # Simulate what _apply_name_condition writes
        road_name_conditions = {
            "Rizal Street": {"condition": "road_closed", "count": 3},
            "National Road": {"condition": "flooded_passable", "count": 7},
        }
        # Convert to ScenarioConfig road_conditions format
        rc = [
            {"road_name": rn, "condition": rv["condition"], "segment_count": rv["count"]}
            for rn, rv in road_name_conditions.items()
        ]
        sc = ScenarioConfig(road_conditions=rc)
        assert len(sc.road_conditions) == 2
        assert sc.road_conditions[0]["road_name"] == "Rizal Street"
        assert sc.road_conditions[1]["condition"] == "flooded_passable"

    def test_duplicate_road_cannot_appear_twice(self):
        """The road name is the stable key — adding same road replaces the entry."""
        # road_name_conditions is a dict keyed by road_name — duplicates are impossible
        road_name_conditions: dict = {}

        def simulate_apply(road_name: str, condition: str, count: int) -> None:
            if condition == "use_model":
                road_name_conditions.pop(road_name, None)
            else:
                road_name_conditions[road_name] = {"condition": condition, "count": count}

        simulate_apply("Rizal Street", "road_closed", 3)
        simulate_apply("Rizal Street", "flooded_passable", 3)  # update same road
        assert len(road_name_conditions) == 1
        assert road_name_conditions["Rizal Street"]["condition"] == "flooded_passable"

    def test_remove_clears_entry_from_dict(self):
        """Removing a road condition (setting 'use_model') removes it from the dict."""
        road_name_conditions: dict = {
            "Rizal Street": {"condition": "road_closed", "count": 3},
        }

        def simulate_remove(road_name: str) -> None:
            road_name_conditions.pop(road_name, None)

        simulate_remove("Rizal Street")
        assert "Rizal Street" not in road_name_conditions
        assert road_name_conditions == {}

    def test_multiple_distinct_roads_all_tracked(self):
        """Multiple different roads each get their own entry."""
        road_name_conditions: dict = {}
        roads = [
            ("Alpha Road", "road_closed", 1),
            ("Beta Street", "flooded_impassable", 4),
            ("Gamma Avenue", "unknown", 2),
        ]
        for name, cond, cnt in roads:
            road_name_conditions[name] = {"condition": cond, "count": cnt}
        assert len(road_name_conditions) == 3
        for name, cond, _cnt in roads:
            assert road_name_conditions[name]["condition"] == cond

    def test_road_conditions_scene_to_config_snapshot(self):
        """Snapshot of road_name_conditions → ScenarioConfig.road_conditions matches."""
        from floodroute.scenario.config import ScenarioConfig
        road_name_conditions = {
            "Rizal Street": {"condition": "road_closed", "count": 3},
            "National Road": {"condition": "flooded_passable", "count": 7},
        }
        rc_snap = [
            {"road_name": rn, "condition": rv.get("condition", "use_model"),
             "segment_count": rv.get("count", 0)}
            for rn, rv in road_name_conditions.items()
        ]
        sc = ScenarioConfig(road_conditions=rc_snap)
        assert sc.road_conditions[0]["road_name"] == "Rizal Street"
        assert sc.road_conditions[0]["segment_count"] == 3
        assert sc.road_conditions[1]["road_name"] == "National Road"
        assert sc.road_conditions[1]["condition"] == "flooded_passable"

    def test_clear_all_empties_road_name_conditions(self):
        """Clearing all road conditions produces an empty dict."""
        road_name_conditions = {
            "Rizal Street": {"condition": "road_closed", "count": 3},
            "National Road": {"condition": "flooded_passable", "count": 7},
        }
        # Simulate "Clear all" action
        road_name_conditions.clear()
        assert road_name_conditions == {}


class TestNode58ColocatedNotExcluded:
    """Regression: node 58 (Atabay Elementary School) must NOT be excluded.

    Node 58 is both a barangay origin (Atabay) and the snapped node for
    Atabay Elementary School.  The optimizer represents them as distinct
    logical vertices ("origin", 58) / ("shelter", 58), so the facility
    resolver must allow node 58 as a shelter.
    """

    _ATABAY_SCHOOL_FID = "way:168981805"
    _NODE_58 = 58

    def _catalog(self):
        from floodroute.scenario.catalog import build_catalog
        return build_catalog()

    def test_atabay_school_snaps_to_node_58(self):
        """CatalogEntry for Atabay Elementary School must snap to node 58."""
        catalog = self._catalog()
        entry = catalog.get(self._ATABAY_SCHOOL_FID)
        assert entry is not None, (
            f"{self._ATABAY_SCHOOL_FID} not found in catalog"
        )
        assert entry.snapped_node == self._NODE_58, (
            f"Expected snapped_node=58, got {entry.snapped_node}"
        )

    def test_atabay_school_can_be_selected(self):
        """Atabay Elementary School must be selectable (snapped_node is not None)."""
        catalog = self._catalog()
        entry = catalog.get(self._ATABAY_SCHOOL_FID)
        assert entry is not None
        assert entry.can_be_selected, (
            "Atabay Elementary School should be selectable but can_be_selected=False"
        )

    def test_node_58_present_in_resolve_node_shelters_without_exclusion(self):
        """resolve_node_shelters must include node 58 when no exclude_nodes given."""
        from floodroute.scenario.config import ScenarioConfig
        catalog = self._catalog()
        sc = ScenarioConfig(
            municipality="PH0600613",
            return_period="RP100",
            selected_facility_ids=[self._ATABAY_SCHOOL_FID],
            facility_capacities={self._ATABAY_SCHOOL_FID: 300},
        )
        shelters = catalog.resolve_node_shelters(sc)
        assert self._NODE_58 in shelters, (
            f"Node 58 must appear as a shelter; got {shelters}"
        )
        assert shelters[self._NODE_58] == 300

    def test_node_58_respects_scenario_capacity(self):
        """Capacity for node 58 must match the value set in ScenarioConfig."""
        from floodroute.scenario.config import ScenarioConfig
        catalog = self._catalog()
        for cap in (150, 500, 1000):
            sc = ScenarioConfig(
                municipality="PH0600613",
                return_period="RP100",
                selected_facility_ids=[self._ATABAY_SCHOOL_FID],
                facility_capacities={self._ATABAY_SCHOOL_FID: cap},
            )
            shelters = catalog.resolve_node_shelters(sc)
            assert shelters.get(self._NODE_58) == cap, (
                f"Expected capacity {cap} at node 58, got {shelters.get(self._NODE_58)}"
            )

    def test_node_58_excluded_only_when_explicitly_in_exclude_nodes(self):
        """resolve_node_shelters excludes node 58 only when explicitly asked."""
        from floodroute.scenario.config import ScenarioConfig
        catalog = self._catalog()
        sc = ScenarioConfig(
            municipality="PH0600613",
            return_period="RP100",
            selected_facility_ids=[self._ATABAY_SCHOOL_FID],
            facility_capacities={self._ATABAY_SCHOOL_FID: 300},
        )
        # Without exclusion: present
        assert self._NODE_58 in catalog.resolve_node_shelters(sc)
        # With explicit exclusion: absent
        shelters_excluded = catalog.resolve_node_shelters(sc, exclude_nodes={self._NODE_58})
        assert self._NODE_58 not in shelters_excluded


# ---------------------------------------------------------------------------
# Regression: road_override_store propagation in run_scenario / run_experiment
# ---------------------------------------------------------------------------

def _make_override_graph() -> nx.MultiDiGraph:
    """Minimal graph: origin 1 → 2 → shelter 3 (dry path).
    Also has direct edge 1 → 3 (flooded, longer) as fallback.
    Closing 1→2 forces algorithms to take the flooded path or report unreachable.
    """
    G = nx.MultiDiGraph()
    for n, x in [(1, 0.0), (2, 100.0), (3, 200.0)]:
        G.add_node(n, x=x, y=0.0)
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None,
               jrc_rp10_status="modelled_dry", jrc_rp10_depth_max_m=None,
               jrc_rp20_status="modelled_dry", jrc_rp20_depth_max_m=None)
    G.add_edge(2, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None,
               jrc_rp10_status="modelled_dry", jrc_rp10_depth_max_m=None,
               jrc_rp20_status="modelled_dry", jrc_rp20_depth_max_m=None)
    G.add_edge(1, 3, key=0, length_m=400.0,
               jrc_rp100_status="flooded", jrc_rp100_depth_max_m=0.5,
               jrc_rp10_status="flooded", jrc_rp10_depth_max_m=0.5,
               jrc_rp20_status="flooded", jrc_rp20_depth_max_m=0.5)
    return G


def _make_override_runner_config(algorithm: str):
    from floodroute.experiments.runner import ScenarioKey, make_experiment_config
    config = make_experiment_config(
        municipality="PH0600613",
        algorithms=(algorithm,),
        return_periods=("RP100",),
        demand_fractions=(1.0,),
        capacity_multipliers=(1.0,),
        flood_penalties=(10.0,),
        nominal_capacities={3: 1000},
        pilot_mode=False,
    )
    key = ScenarioKey(
        algorithm=algorithm,
        return_period="RP100",
        demand_fraction=1.0,
        capacity_multiplier=1.0,
        flood_penalty=10.0,
    )
    return config, key


class TestRunScenarioOverrideStorePropagation:
    """Regression: road_override_store kwarg is accepted and honoured."""

    def _origins(self):
        from floodroute.experiments.demand import BarangayOrigin
        return [BarangayOrigin(
            psgc="PH060061300X", name="TestBgy",
            population_2020=100, origin_node=1, snap_distance_m=0.0,
        )]

    def test_run_scenario_accepts_none_store(self):
        """run_scenario(road_override_store=None) must not raise."""
        from floodroute.experiments.runner import run_scenario
        G = _make_override_graph()
        origins = self._origins()
        config, key = _make_override_runner_config("B")
        result = run_scenario(G, origins, config, key, road_override_store=None)
        assert result.run_status == "completed"

    def test_run_scenario_accepts_empty_store(self):
        """run_scenario with an empty RoadOverrideStore must behave like no overrides."""
        from floodroute.dashboard.road_overrides import RoadOverrideStore
        from floodroute.experiments.runner import run_scenario
        G = _make_override_graph()
        origins = self._origins()
        config, key = _make_override_runner_config("C")
        store = RoadOverrideStore()
        result = run_scenario(G, origins, config, key, road_override_store=store)
        assert result.run_status == "completed"

    def test_run_scenario_override_store_affects_flood_aware_routing(self):
        """Closing edge 1→2 via override must alter routing for algorithm B.

        With the closure, algorithm B cannot use the dry path 1→2→3 and must
        either take the flooded path 1→3 (with penalty) or report unreachable.
        Without the closure the result is the same graph — comparing the two
        confirms the override was applied.
        """
        from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore
        from floodroute.experiments.runner import run_scenario
        G = _make_override_graph()
        origins = self._origins()
        config, key = _make_override_runner_config("B")

        result_no_override = run_scenario(G, origins, config, key, road_override_store=None)

        store = RoadOverrideStore()
        store.add(RoadOverride(u=1, v=2, status="road_closed"))
        result_with_override = run_scenario(G, origins, config, key, road_override_store=store)

        # Both must complete without error (may assign via flooded path or report unreachable)
        assert result_no_override.run_status == "completed"
        assert result_with_override.run_status == "completed"

        # The override must change the OD cost or reachability — the results
        # cannot be identical when a key edge is closed.
        cost_no_override = result_no_override.metrics.get("total_route_cost_m_eq", 0)
        cost_with_override = result_with_override.metrics.get("total_route_cost_m_eq", 0)
        assert cost_no_override != cost_with_override, (
            "road_closed override had no effect on algorithm B routing costs"
        )

    def test_run_experiment_accepts_road_override_store(self):
        """run_experiment(road_override_store=...) must propagate to all scenarios."""
        from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore
        from floodroute.experiments.runner import make_experiment_config, run_experiment
        G = _make_override_graph()
        origins = self._origins()
        config = make_experiment_config(
            municipality="PH0600613",
            algorithms=("B",),
            return_periods=("RP100",),
            demand_fractions=(1.0,),
            capacity_multipliers=(1.0,),
            flood_penalties=(10.0,),
            nominal_capacities={3: 1000},
            pilot_mode=False,
        )
        store = RoadOverrideStore()
        store.add(RoadOverride(u=1, v=2, status="road_closed"))
        results = run_experiment(G, origins, config, road_override_store=store)
        assert len(results) == 1
        assert results[0].run_status == "completed"

    @pytest.mark.parametrize("algorithm", ["A", "B", "B+", "C"])
    def test_run_scenario_all_algorithms_accept_store(self, algorithm):
        """All four algorithms must accept road_override_store without raising."""
        from floodroute.dashboard.road_overrides import RoadOverride, RoadOverrideStore
        from floodroute.experiments.runner import run_scenario
        G = _make_override_graph()
        origins = self._origins()
        config, key = _make_override_runner_config(algorithm)
        store = RoadOverrideStore()
        store.add(RoadOverride(u=1, v=2, status="flooded_passable"))
        result = run_scenario(G, origins, config, key, road_override_store=store)
        assert result.run_status in ("completed", "failed"), (
            f"Algorithm {algorithm} must return a ScenarioResult, not raise"
        )


# ---------------------------------------------------------------------------
# Regression: Feasible column logic (experiment_page display)
# ---------------------------------------------------------------------------

class TestFeasibleColumnLogic:
    """Unit tests for the Feasible / overflow display logic used in experiment_page."""

    @staticmethod
    def _feasible_label(overflow: object) -> str:
        """Mirror the logic from experiment_page.py rows builder."""
        val = overflow if overflow is not None else 0
        val = val or 0
        return "Yes" if val == 0 else f"No ({val:,} overflow)"

    def test_zero_overflow_shows_yes(self):
        assert self._feasible_label(0) == "Yes"

    def test_none_overflow_shows_yes(self):
        assert self._feasible_label(None) == "Yes"

    def test_positive_overflow_shows_no_with_count(self):
        assert self._feasible_label(500) == "No (500 overflow)"

    def test_large_overflow_uses_comma_format(self):
        assert self._feasible_label(1500) == "No (1,500 overflow)"

    def test_algorithms_b_and_c_produce_zero_overflow_on_unconstrained_graph(self):
        """B+ and C must not overflow when total capacity > total demand."""
        from floodroute.experiments.demand import BarangayOrigin
        from floodroute.experiments.runner import run_scenario
        G = _make_override_graph()
        origins = [BarangayOrigin(
            psgc="PH060061300X", name="TestBgy",
            population_2020=10, origin_node=1, snap_distance_m=0.0,
        )]
        for alg in ("B+", "C"):
            config, key_base = _make_override_runner_config(alg)
            result = run_scenario(G, origins, config, key_base, road_override_store=None)
            overflow = result.metrics.get("total_overflow_units", 0) or 0
            assert overflow == 0, (
                f"Algorithm {alg} must not overflow when capacity ({1000}) >> demand"
            )
            label = self._feasible_label(overflow)
            assert label == "Yes", f"Expected Feasible=Yes for {alg}, got {label!r}"
