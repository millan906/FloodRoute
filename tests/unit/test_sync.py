"""Synchronization tests: preset round-trip, configuration hash, frozen snapshot,
Algorithm A closure handling, and B/B+/C override application.

All tests are pure-Python with synthetic graphs.  No Streamlit, no geospatial I/O.
"""
from __future__ import annotations

import json

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_override(u: int, v: int, status: str = "road_closed") -> dict:
    """Return a single road-override dict in road_override_store_dict format."""
    return {
        f"{u},{v}": {
            "u": u, "v": v, "status": status,
            "evidence_type": "controlled_assumption",
            "source_reference": "", "observation_time": "", "notes": "",
        }
    }


def _two_node_graph(flooded: bool = False) -> nx.MultiDiGraph:
    """Origin 1 → shelter 2, single edge."""
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    status = "flooded" if flooded else "modelled_dry"
    G.add_edge(
        1, 2,
        length_m=100.0,
        jrc_rp100_status=status,
        jrc_rp10_status=status,
        jrc_rp20_status=status,
    )
    return G


def _three_node_graph() -> nx.MultiDiGraph:
    """Origin 1 → shelter 2 (direct, 100m) and origin 1 → 3 → shelter 2 (detour, 300m).

    This allows tests to close edge 1→2 and verify routing switches to 1→3→2.
    """
    G = nx.MultiDiGraph()
    for n in [1, 2, 3]:
        G.add_node(n)
    dry = {"jrc_rp100_status": "modelled_dry",
           "jrc_rp10_status": "modelled_dry",
           "jrc_rp20_status": "modelled_dry"}
    G.add_edge(1, 2, length_m=100.0, **dry)   # direct route
    G.add_edge(1, 3, length_m=150.0, **dry)   # detour leg 1
    G.add_edge(3, 2, length_m=150.0, **dry)   # detour leg 2
    return G


def _make_origin(origin_node: int, psgc: str = "PH0600613001", name: str = "Test"):
    from dataclasses import dataclass

    @dataclass
    class _FakeOrigin:
        origin_node: int
        psgc: str
        name: str
        population_2020: int = 100

    return _FakeOrigin(origin_node=origin_node, psgc=psgc, name=name)


# ---------------------------------------------------------------------------
# 1. Preset round-trip: Save → scenario_to_dict → scenario_from_dict
# ---------------------------------------------------------------------------

class TestPresetRoundTrip:
    """Complete save/load cycle for ScenarioConfig including road_overrides."""

    def test_road_overrides_serialized(self, tmp_path):
        """scenario_to_dict must include road_overrides in the JSON payload."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_to_dict

        overrides = _make_override(10, 20, "flooded_impassable")
        sc = ScenarioConfig(
            return_period="RP20",
            demand_mode="fraction",
            demand_fraction=0.10,
            selected_facility_ids=["fac_1"],
            facility_capacities={"fac_1": 500},
            road_overrides=overrides,
        )
        d = scenario_to_dict(sc, "fp123", "Test Scenario")
        assert "road_overrides" in d
        assert "10,20" in d["road_overrides"]
        assert d["road_overrides"]["10,20"]["status"] == "flooded_impassable"

    def test_road_overrides_deserialized(self):
        """scenario_from_dict must populate ScenarioConfig.road_overrides from JSON."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        overrides = _make_override(5, 7, "road_closed")
        sc_in = ScenarioConfig(
            return_period="RP100",
            demand_fraction=0.25,
            road_overrides=overrides,
        )
        d = scenario_to_dict(sc_in, "fp", "name")
        sc_out, _ = scenario_from_dict(d)
        assert sc_out.road_overrides == {"5,7": dict(sorted(overrides["5,7"].items()))}

    def test_return_period_round_trips(self):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        sc = ScenarioConfig(return_period="RP10")
        d = scenario_to_dict(sc, "", "s")
        sc2, _ = scenario_from_dict(d)
        assert sc2.return_period == "RP10"

    def test_demand_fraction_round_trips(self):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        sc = ScenarioConfig(demand_mode="fraction", demand_fraction=0.50)
        d = scenario_to_dict(sc, "", "s")
        sc2, _ = scenario_from_dict(d)
        assert sc2.demand_mode == "fraction"
        assert sc2.demand_fraction == pytest.approx(0.50)

    def test_exact_demand_round_trips(self):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        sc = ScenarioConfig(demand_mode="exact", demand_exact=750)
        d = scenario_to_dict(sc, "", "s")
        sc2, _ = scenario_from_dict(d)
        assert sc2.demand_mode == "exact"
        assert sc2.demand_exact == 750

    def test_facilities_round_trip(self):
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        sc = ScenarioConfig(
            selected_facility_ids=["a", "b"],
            facility_capacities={"a": 100, "b": 200},
        )
        d = scenario_to_dict(sc, "", "s")
        sc2, _ = scenario_from_dict(d)
        assert sc2.selected_facility_ids == ["a", "b"]
        assert sc2.facility_capacities == {"a": 100, "b": 200}

    def test_empty_overrides_round_trip(self):
        """Empty road_overrides round-trips as empty dict."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        sc = ScenarioConfig(road_overrides={})
        d = scenario_to_dict(sc, "", "s")
        sc2, _ = scenario_from_dict(d)
        assert sc2.road_overrides == {}

    def test_road_overrides_normalized_on_save(self):
        """scenario_to_dict sorts override keys for canonical representation."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_to_dict

        # Provide overrides in reverse-alpha order
        overrides = {
            "9,1": {"u": 9, "v": 1, "status": "road_closed",
                    "evidence_type": "controlled_assumption",
                    "source_reference": "", "observation_time": "", "notes": ""},
            "1,9": {"u": 1, "v": 9, "status": "flooded_impassable",
                    "evidence_type": "controlled_assumption",
                    "source_reference": "", "observation_time": "", "notes": ""},
        }
        sc = ScenarioConfig(road_overrides=overrides)
        d = scenario_to_dict(sc, "", "s")
        # Keys must appear in sorted order in the serialized dict
        keys = list(d["road_overrides"].keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# 2. Override replacement on load
# ---------------------------------------------------------------------------

class TestOverrideReplacementOnLoad:
    """Loading a preset must replace active road_overrides, not merge them."""

    def test_prior_overrides_not_in_loaded_scenario(self):
        """ScenarioConfig loaded from JSON has only its own overrides, not prior ones."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        # Preset was saved with override on edge 5,6
        sc_saved = ScenarioConfig(road_overrides=_make_override(5, 6, "road_closed"))
        d = scenario_to_dict(sc_saved, "", "s")
        sc_loaded, _ = scenario_from_dict(d)

        # The loaded ScenarioConfig must have exactly one override (5,6) — not any prior
        assert "5,6" in sc_loaded.road_overrides
        assert len(sc_loaded.road_overrides) == 1

    def test_loading_empty_preset_yields_empty_overrides(self):
        """A preset saved with no overrides loads with empty road_overrides."""
        from floodroute.scenario.config import ScenarioConfig
        from floodroute.scenario.persistence import scenario_from_dict, scenario_to_dict

        sc = ScenarioConfig(road_overrides={})
        d = scenario_to_dict(sc, "", "s")
        sc2, _ = scenario_from_dict(d)
        assert sc2.road_overrides == {}

    def test_loading_preset_without_field_yields_empty_overrides(self):
        """Legacy presets without 'road_overrides' field load with empty dict."""
        from floodroute.scenario.persistence import scenario_from_dict

        # Simulate a legacy preset (no road_overrides key)
        legacy_data = {
            "schema_version": "1.0",
            "scenario_id": "abc123",
            "scenario_name": "Legacy",
            "municipality": "PH0600613",
            "return_period": "RP100",
            "demand_mode": "fraction",
            "demand_fraction": 0.25,
            "demand_exact": 0,
            "selected_facility_ids": [],
            "facility_capacities": {},
            "algorithms": ["A", "B", "B+", "C"],
            "capacity_multipliers": [0.50, 0.75, 1.00, 1.25],
            "flood_penalties": ["10.0"],
            "catalog_fingerprint": "",
            "road_conditions": [],
            # Note: no "road_overrides" key
        }
        sc, _ = scenario_from_dict(legacy_data)
        assert sc.road_overrides == {}


# ---------------------------------------------------------------------------
# 3. Hash sensitivity and canonical ordering
# ---------------------------------------------------------------------------

class TestHashSensitivity:
    """Road overrides are included in the configuration hash."""

    def _make_cfg(self, overrides=None):
        from floodroute.experiments.runner import make_experiment_config
        return make_experiment_config(
            municipality="PH0600613",
            algorithms=("A",),
            return_periods=("RP100",),
            demand_fractions=(0.25,),
            capacity_multipliers=(1.0,),
            flood_penalties=(10.0,),
            nominal_capacities={33: 100},
            road_overrides=overrides,
        )

    def test_no_overrides_vs_overrides_differ(self):
        """Config with overrides has a different hash from config without."""
        c_no = self._make_cfg(overrides=None)
        c_ov = self._make_cfg(overrides=_make_override(1, 2))
        assert c_no.configuration_hash != c_ov.configuration_hash

    def test_same_overrides_same_hash(self):
        """Same overrides → same hash on repeated calls."""
        ov = _make_override(1, 2, "road_closed")
        assert self._make_cfg(ov).configuration_hash == self._make_cfg(ov).configuration_hash

    def test_different_statuses_differ(self):
        """Changing override status changes the hash."""
        c1 = self._make_cfg(overrides=_make_override(1, 2, "road_closed"))
        c2 = self._make_cfg(overrides=_make_override(1, 2, "flooded_impassable"))
        assert c1.configuration_hash != c2.configuration_hash

    def test_canonical_ordering_invariant(self):
        """Same overrides in different insertion order produce the same hash."""
        ov_a = {
            "1,2": {"u": 1, "v": 2, "status": "road_closed",
                    "evidence_type": "controlled_assumption",
                    "source_reference": "", "observation_time": "", "notes": ""},
            "3,4": {"u": 3, "v": 4, "status": "flooded_impassable",
                    "evidence_type": "controlled_assumption",
                    "source_reference": "", "observation_time": "", "notes": ""},
        }
        ov_b = {
            "3,4": {"u": 3, "v": 4, "status": "flooded_impassable",
                    "evidence_type": "controlled_assumption",
                    "source_reference": "", "observation_time": "", "notes": ""},
            "1,2": {"u": 1, "v": 2, "status": "road_closed",
                    "evidence_type": "controlled_assumption",
                    "source_reference": "", "observation_time": "", "notes": ""},
        }
        c_a = self._make_cfg(overrides=ov_a)
        c_b = self._make_cfg(overrides=ov_b)
        assert c_a.configuration_hash == c_b.configuration_hash

    def test_road_overrides_stored_in_config(self):
        """ExperimentConfig.road_overrides contains the canonical override dict."""
        ov = _make_override(7, 8, "road_closed")
        c = self._make_cfg(overrides=ov)
        assert "7,8" in c.road_overrides
        assert c.road_overrides["7,8"]["status"] == "road_closed"

    def test_empty_overrides_default(self):
        """make_experiment_config with road_overrides=None stores empty dict."""
        c = self._make_cfg(overrides=None)
        assert c.road_overrides == {}


# ---------------------------------------------------------------------------
# 4. Immutable run/export snapshot via manifest
# ---------------------------------------------------------------------------

class TestImmutableRunSnapshot:
    """ExperimentConfig.road_overrides is used by save_experiment when no explicit
    override dict is passed, ensuring the manifest reflects run-time state."""

    def _make_cfg_with_overrides(self, overrides, tmp_path):
        from floodroute.experiments.manifest import save_experiment
        from floodroute.experiments.runner import (
            ScenarioKey,
            ScenarioResult,
            make_experiment_config,
        )

        cfg = make_experiment_config(
            municipality="PH0600613",
            algorithms=("A",),
            return_periods=("RP100",),
            demand_fractions=(0.25,),
            capacity_multipliers=(1.0,),
            flood_penalties=(10.0,),
            nominal_capacities={33: 100},
            road_overrides=overrides,
        )
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
        )
        exp_dir = save_experiment(cfg, [sr], experiments_root=tmp_path)
        return exp_dir

    def test_frozen_overrides_in_manifest(self, tmp_path):
        """manifest.json road_overrides comes from config.road_overrides, not session state."""
        ov = _make_override(11, 22, "road_closed")
        exp_dir = self._make_cfg_with_overrides(ov, tmp_path)
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert "11,22" in manifest["road_overrides"]
        assert manifest["road_overrides"]["11,22"]["status"] == "road_closed"

    def test_no_overrides_in_manifest(self, tmp_path):
        """manifest.json road_overrides is {} when no overrides were frozen."""
        exp_dir = self._make_cfg_with_overrides({}, tmp_path)
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert manifest["road_overrides"] == {}

    def test_explicit_override_param_still_works(self, tmp_path):
        """Explicit road_overrides parameter to save_experiment still takes precedence."""
        from floodroute.experiments.manifest import save_experiment
        from floodroute.experiments.runner import (
            ScenarioKey,
            ScenarioResult,
            make_experiment_config,
        )

        cfg = make_experiment_config(
            municipality="PH0600613", algorithms=("A",), return_periods=("RP100",),
            demand_fractions=(0.25,), capacity_multipliers=(1.0,), flood_penalties=(10.0,),
            nominal_capacities={33: 100}, road_overrides={},
        )
        key = ScenarioKey("A", "RP100", 0.25, 1.0, 10.0)
        sr = ScenarioResult(
            key=key, nominal_capacities={33: 100}, adjusted_capacities={33: 100},
            metrics={}, assignments={}, route_metrics={}, facility_metrics={},
            unassigned_reasons={}, run_status="completed", error_message=None,
        )
        explicit_ov = _make_override(99, 100, "flooded_impassable")
        exp_dir = save_experiment(cfg, [sr], experiments_root=tmp_path,
                                  road_overrides=explicit_ov)
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert "99,100" in manifest["road_overrides"]


# ---------------------------------------------------------------------------
# 5. Stale detection: hash changes when override changes
# ---------------------------------------------------------------------------

class TestStaleDetection:
    """Changing road_overrides between make_experiment_config calls changes the hash."""

    def test_adding_override_changes_hash(self):
        from floodroute.experiments.runner import make_experiment_config
        kwargs = dict(
            municipality="PH0600613", algorithms=("C",), return_periods=("RP100",),
            demand_fractions=(0.25,), capacity_multipliers=(1.0,), flood_penalties=(10.0,),
            nominal_capacities={33: 100},
        )
        hash_before = make_experiment_config(**kwargs, road_overrides=None).configuration_hash
        hash_after = make_experiment_config(
            **kwargs, road_overrides=_make_override(5, 6)).configuration_hash
        assert hash_before != hash_after

    def test_removing_override_changes_hash(self):
        from floodroute.experiments.runner import make_experiment_config
        kwargs = dict(
            municipality="PH0600613", algorithms=("C",), return_periods=("RP100",),
            demand_fractions=(0.25,), capacity_multipliers=(1.0,), flood_penalties=(10.0,),
            nominal_capacities={33: 100},
        )
        ov = _make_override(5, 6)
        hash_with = make_experiment_config(**kwargs, road_overrides=ov).configuration_hash
        hash_without = make_experiment_config(**kwargs, road_overrides=None).configuration_hash
        assert hash_with != hash_without


# ---------------------------------------------------------------------------
# 6. Algorithm A: closure handling without flood penalty
# ---------------------------------------------------------------------------

class TestAlgorithmAClosures:
    """Algorithm A must honour impassable overrides but must not apply flood costs."""

    def _run_scenario_a(self, G, demands, caps, overrides=None):
        """Run Algorithm A via run_scenario with given road_override_store."""
        from floodroute.dashboard.road_overrides import RoadOverrideStore
        from floodroute.experiments.runner import (
            ScenarioKey,
            make_experiment_config,
            run_scenario,
        )
        store = None
        if overrides:
            store = RoadOverrideStore.from_dict(overrides)

        cfg = make_experiment_config(
            municipality="PH0600613",
            algorithms=("A",),
            return_periods=("RP100",),
            demand_fractions=(0.25,),
            capacity_multipliers=(1.0,),
            flood_penalties=(10.0,),
            nominal_capacities=caps,
        )
        key = ScenarioKey("A", "RP100", 0.25, 1.0, 10.0)

        # Build a minimal origins list
        origins = [_make_origin(o) for o in demands]

        return run_scenario(G, origins, cfg, key, road_override_store=store)

    def test_a_ignores_flood_cost_on_flooded_edge(self):
        """Algorithm A routes via flooded edge without penalty (ordinary length only).

        population_2020=100, demand_fraction=0.25 → demand=25 assigned to shelter 2.
        """
        G = _two_node_graph(flooded=True)
        result = self._run_scenario_a(G, {1: 25}, {2: 100})
        # Flooded edge is used (A ignores flood status) — all 25 units assigned
        assert result.assignments.get((1, 2), 0) > 0

    def test_a_respects_hard_closure(self):
        """Algorithm A must not use a road_closed overridden edge.

        _three_node_graph: direct 1→2 = 100m, detour 1→3→2 = 300m.
        Closing 1→2 forces Algorithm A to use the 300m detour.
        """
        G = _three_node_graph()
        overrides = _make_override(1, 2, "road_closed")
        result = self._run_scenario_a(G, {1: 25}, {2: 100}, overrides=overrides)
        # Must still assign via detour
        assert result.assignments.get((1, 2), 0) > 0
        # Detour route (1→3→2) is 300m; direct is 100m.
        rm = result.route_metrics.get((1, 2), {})
        assert rm.get("physical_m", 0) > 100.0, (
            f"Expected detour (>100m), got physical_m={rm.get('physical_m')}"
        )

    def test_a_unreachable_when_only_route_closed(self):
        """Algorithm A reports origin unreachable when its only route is closed."""
        G = _two_node_graph()  # only one path: 1→2
        overrides = _make_override(1, 2, "road_closed")
        result = self._run_scenario_a(G, {1: 25}, {2: 100}, overrides=overrides)
        # No route → no assignment
        assert result.assignments.get((1, 2), 0) == 0

    def test_a_closure_does_not_apply_flood_weight_to_ordinary_edges(self):
        """Closure-only weight function for A uses ordinary length, not flood length.

        If the closure WFN accidentally applied flood_penalty_fn, a flooded edge
        with penalty 10 would cost 1000m instead of 100m.  Algorithm A should
        compute ordinary length (100m) for non-closed edges.
        """
        from floodroute.dashboard.road_overrides import RoadOverrideStore, apply_overrides
        from floodroute.experiments.algorithms import make_ordinary_weight_fn

        G = _two_node_graph(flooded=True)
        # Close a different edge (doesn't exist — so no effect on routing)
        overrides = _make_override(99, 100, "road_closed")
        store = RoadOverrideStore.from_dict(overrides)
        closures_wfn = apply_overrides(make_ordinary_weight_fn(), store, flood_penalty_fn=None)

        # Edge 1→2 is flooded but not overridden — closures_only should return 100m
        edge_data = dict(G[1][2])
        weight = closures_wfn(1, 2, edge_data)
        assert weight == pytest.approx(100.0), (
            f"Closures-only weight should be ordinary length 100m, got {weight}"
        )


# ---------------------------------------------------------------------------
# 7. B, B+, C: override application
# ---------------------------------------------------------------------------

class TestFloodAwareAlgorithmOverrides:
    """B, B+, and C must apply road overrides (closures) before computing OD costs."""

    def _run_scenario(self, algorithm: str, G, demands, caps, overrides=None):
        from floodroute.dashboard.road_overrides import RoadOverrideStore
        from floodroute.experiments.runner import (
            ScenarioKey,
            make_experiment_config,
            run_scenario,
        )
        store = None
        if overrides:
            store = RoadOverrideStore.from_dict(overrides)

        cfg = make_experiment_config(
            municipality="PH0600613",
            algorithms=(algorithm,),
            return_periods=("RP100",),
            demand_fractions=(0.25,),
            capacity_multipliers=(1.0,),
            flood_penalties=(10.0,),
            nominal_capacities=caps,
        )
        key = ScenarioKey(algorithm, "RP100", 0.25, 1.0, 10.0)
        origins = [_make_origin(o) for o in demands]
        return run_scenario(G, origins, cfg, key, road_override_store=store)

    @pytest.mark.parametrize("algorithm", ["B", "B+", "C"])
    def test_impassable_override_blocks_direct_route(self, algorithm):
        """Closing direct edge 1→2 forces flood-aware algorithms to use detour 1→3→2."""
        G = _three_node_graph()
        overrides = _make_override(1, 2, "road_closed")
        result = self._run_scenario(algorithm, G, {1: 25}, {2: 100}, overrides=overrides)
        assert result.assignments.get((1, 2), 0) > 0
        rm = result.route_metrics.get((1, 2), {})
        assert rm.get("physical_m", 0) > 100.0, (
            f"Algorithm {algorithm}: expected detour (>100m), got physical_m={rm.get('physical_m')}"
        )

    @pytest.mark.parametrize("algorithm", ["B", "B+", "C"])
    def test_no_override_uses_direct_route(self, algorithm):
        """Without overrides, direct route 1→2 is used."""
        G = _three_node_graph()
        result = self._run_scenario(algorithm, G, {1: 25}, {2: 100}, overrides=None)
        assert result.assignments.get((1, 2), 0) > 0
        rm = result.route_metrics.get((1, 2), {})
        assert rm.get("physical_m", 0) <= 100.0 + 1e-6, (
            f"Algorithm {algorithm}: expected direct route (<=100m), got physical_m={rm.get('physical_m')}"
        )

    @pytest.mark.parametrize("algorithm", ["B", "B+", "C"])
    def test_unreachable_when_only_route_closed(self, algorithm):
        """Closing the only route makes origin unreachable for flood-aware algorithms."""
        G = _two_node_graph()
        overrides = _make_override(1, 2, "flooded_impassable")
        result = self._run_scenario(algorithm, G, {1: 25}, {2: 100}, overrides=overrides)
        assert result.assignments.get((1, 2), 0) == 0


# ---------------------------------------------------------------------------
# 8. Algorithm C solver verification
# ---------------------------------------------------------------------------

class TestAlgorithmCSolver:
    """Verify Algorithm C uses NetworkX network_simplex via nx.min_cost_flow."""

    def test_solver_is_networkx_min_cost_flow(self):
        """solve_assignment must call nx.min_cost_flow (NetworkX network_simplex)."""
        import inspect

        from floodroute.optimization.assignment import solve_assignment
        src = inspect.getsource(solve_assignment)
        assert "min_cost_flow" in src, (
            "solve_assignment must call nx.min_cost_flow (NetworkX network_simplex)"
        )
        # Must NOT reference scipy or HiGHS
        for forbidden in ("scipy", "linprog", "HiGHS", "pulp", "ortools"):
            assert forbidden not in src, (
                f"solve_assignment must not reference '{forbidden}'"
            )

    def test_algorithm_c_is_deterministic(self):
        """Two calls with identical inputs must produce identical assignments."""
        G = _three_node_graph()
        from floodroute.experiments.runner import ScenarioKey, make_experiment_config, run_scenario
        cfg = make_experiment_config(
            municipality="PH0600613",
            algorithms=("C",),
            return_periods=("RP100",),
            demand_fractions=(0.25,),
            capacity_multipliers=(1.0,),
            flood_penalties=(10.0,),
            nominal_capacities={2: 50},
        )
        key = ScenarioKey("C", "RP100", 0.25, 1.0, 10.0)
        origins = [_make_origin(1)]

        r1 = run_scenario(G, origins, cfg, key, road_override_store=None)
        r2 = run_scenario(G, origins, cfg, key, road_override_store=None)
        assert r1.assignments == r2.assignments

    def test_all_algorithms_deterministic_with_overrides(self):
        """All algorithms produce identical assignments on repeated runs with same overrides."""
        from floodroute.dashboard.road_overrides import RoadOverrideStore
        from floodroute.experiments.runner import ScenarioKey, make_experiment_config, run_scenario

        G = _three_node_graph()
        overrides = _make_override(1, 2, "road_closed")
        store = RoadOverrideStore.from_dict(overrides)
        caps = {2: 100}

        for algorithm in ("A", "B", "B+", "C"):
            cfg = make_experiment_config(
                municipality="PH0600613",
                algorithms=(algorithm,),
                return_periods=("RP100",),
                demand_fractions=(0.25,),
                capacity_multipliers=(1.0,),
                flood_penalties=(10.0,),
                nominal_capacities=caps,
            )
            key = ScenarioKey(algorithm, "RP100", 0.25, 1.0, 10.0)
            origins = [_make_origin(1)]

            r1 = run_scenario(G, origins, cfg, key, road_override_store=store)
            r2 = run_scenario(G, origins, cfg, key, road_override_store=store)
            assert r1.assignments == r2.assignments, (
                f"Algorithm {algorithm} is not deterministic with identical inputs"
            )


# ---------------------------------------------------------------------------
# 9. make_ordinary_weight_fn export
# ---------------------------------------------------------------------------

def test_make_ordinary_weight_fn_exported():
    """make_ordinary_weight_fn must be importable from algorithms."""
    from floodroute.experiments.algorithms import make_ordinary_weight_fn
    wfn = make_ordinary_weight_fn()
    assert callable(wfn)


def test_make_ordinary_weight_fn_returns_length():
    """make_ordinary_weight_fn returns length_m for a simple edge dict."""
    from floodroute.experiments.algorithms import make_ordinary_weight_fn
    wfn = make_ordinary_weight_fn()
    edge = {0: {"length_m": 250.0, "jrc_rp100_status": "flooded"}}
    assert wfn(1, 2, edge) == pytest.approx(250.0)


def test_make_ordinary_weight_fn_ignores_flood_status():
    """Ordinary weight function returns same value regardless of flood status."""
    from floodroute.experiments.algorithms import make_ordinary_weight_fn
    wfn = make_ordinary_weight_fn()
    dry = {0: {"length_m": 100.0, "jrc_rp100_status": "modelled_dry"}}
    flooded = {0: {"length_m": 100.0, "jrc_rp100_status": "flooded"}}
    assert wfn(1, 2, dry) == wfn(1, 2, flooded)
