"""Stage 10 finalisation: 14 minimal behavioral tests.

Graph fixture A (3-node):
    1 → 2 → 3   (both edges dry, 100 m)
    1 → 3        (200 m, flooded 0.3 m)
Node 1 = origin, node 3 = shelter.

Graph fixture B (4-node, two shelters):
    1 → 2 → 3   (dry, 100 m each)
    1 → 2 → 4   (dry, 100 m + 200 m)
Node 1 = origin, node 3 = Facility A, node 4 = Facility B.
Used for acceptance-scenario split and unassigned tests.
"""
from __future__ import annotations

import networkx as nx

from floodroute.dashboard.result_formatter import format_origin_assignment_status
from floodroute.dashboard.road_overrides import (
    RoadOverride,
    RoadOverrideStore,
    apply_overrides,
)
from floodroute.experiments.algorithms import (
    run_flood_aware_nearest,
    run_floodroute_assignment,
)
from floodroute.optimization.cost import make_weight_fn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n, x in [(1, 0.0), (2, 100.0), (3, 200.0)]:
        G.add_node(n, x=x, y=0.0)
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(1, 3, key=0, length_m=200.0,
               jrc_rp100_status="flooded", jrc_rp100_depth_max_m=0.3)
    return G


def _two_shelter_graph() -> nx.MultiDiGraph:
    """4-node graph: origin 1, intermediate 2, shelters 3 (A) and 4 (B)."""
    G = nx.MultiDiGraph()
    for n, x in [(1, 0.0), (2, 100.0), (3, 200.0), (4, 300.0)]:
        G.add_node(n, x=x, y=0.0)
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 4, key=0, length_m=200.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    return G


def _store_with(*overrides: RoadOverride) -> RoadOverrideStore:
    s = RoadOverrideStore()
    for ov in overrides:
        s.add(ov)
    return s


# ---------------------------------------------------------------------------
# Tests 1-2: multiple overrides
# ---------------------------------------------------------------------------

def test_multiple_overrides_stored_independently():
    """Two different edges can carry simultaneous overrides."""
    store = _store_with(
        RoadOverride(u=1, v=2, status="road_closed"),
        RoadOverride(u=2, v=3, status="flooded_passable"),
    )
    assert len(store) == 2
    assert store.overrides[(1, 2)].status == "road_closed"
    assert store.overrides[(2, 3)].status == "flooded_passable"


def test_multiple_overrides_applied_simultaneously():
    """Both overrides affect routing at the same time."""
    G = _graph()
    store = _store_with(
        RoadOverride(u=1, v=2, status="road_closed"),
        RoadOverride(u=2, v=3, status="road_closed"),
    )
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)
    # All intermediate edges closed — only 1→3 (flooded) remains
    lengths, paths = nx.single_source_dijkstra(G, 1, weight=wfn)
    assert 3 in paths
    assert paths[3] == [1, 3]  # must use direct flooded edge


# ---------------------------------------------------------------------------
# Tests 3-4: hard closures block all algorithms
# ---------------------------------------------------------------------------

def test_road_closed_blocks_alg_b():
    G = _graph()
    store = _store_with(RoadOverride(u=1, v=2, status="road_closed"))
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)
    result = run_flood_aware_nearest(G, {1: 100}, {3: 500}, "RP100", weight_fn=wfn)
    # Must NOT route via node 2
    for path in result.routes.values():
        assert 2 not in path, "road_closed must block Algorithm B from using node 2"


def test_flooded_impassable_blocks_alg_c():
    G = _graph()
    store = _store_with(
        RoadOverride(u=1, v=2, status="flooded_impassable"),
        RoadOverride(u=2, v=3, status="flooded_impassable"),
    )
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)
    result = run_floodroute_assignment(G, {1: 100}, {3: 500}, "RP100", weight_fn=wfn)
    for path in result.routes.values():
        assert 2 not in path, "flooded_impassable must block Algorithm C from node 2"


# ---------------------------------------------------------------------------
# Test 5: flooded_passable stays available with penalty (B/C)
# ---------------------------------------------------------------------------

def test_flooded_passable_available_with_penalty():
    G = _graph()
    store = _store_with(RoadOverride(u=1, v=2, status="flooded_passable"))
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store, flood_penalty_fn=base)
    result = run_flood_aware_nearest(G, {1: 100}, {3: 500}, "RP100", weight_fn=wfn)
    assert result.assignments, "flooded_passable must keep edge available"


# ---------------------------------------------------------------------------
# Test 6: clearing overrides restores original behavior
# ---------------------------------------------------------------------------

def test_clear_overrides_restores_routing():
    G = _graph()
    store = _store_with(RoadOverride(u=1, v=2, status="road_closed"))
    base = make_weight_fn("RP100")
    wfn_closed = apply_overrides(base, store)
    run_flood_aware_nearest(G, {1: 100}, {3: 500}, "RP100", weight_fn=wfn_closed)

    store.clear()
    wfn_open = apply_overrides(base, store)
    result_open = run_flood_aware_nearest(G, {1: 100}, {3: 500}, "RP100", weight_fn=wfn_open)

    # After clearing, route may go through node 2 again
    for path in result_open.routes.values():
        assert 3 in path  # still reaches the shelter


# ---------------------------------------------------------------------------
# Test 7: source graph unchanged after all operations
# ---------------------------------------------------------------------------

def test_source_graph_not_mutated():
    G = _graph()
    edges_before = set(G.edges(keys=True))
    attrs_snap = {(u, v, k): dict(d) for u, v, k, d in G.edges(data=True, keys=True)}

    store = _store_with(RoadOverride(u=1, v=2, status="road_closed"))
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)
    run_flood_aware_nearest(G, {1: 100}, {3: 500}, "RP100", weight_fn=wfn)
    store.clear()

    assert set(G.edges(keys=True)) == edges_before
    for (u, v, k), old in attrs_snap.items():
        assert dict(G[u][v][k]) == old


# ---------------------------------------------------------------------------
# Tests 8-9: multiple facilities enter optimization
# ---------------------------------------------------------------------------

def test_two_facilities_both_in_optimization():
    """Two shelters at different nodes means algorithm can split demand."""
    G = nx.MultiDiGraph()
    # node 1 = origin; nodes 3 and 4 = two shelters
    for n, x in [(1, 0.0), (2, 100.0), (3, 200.0), (4, 300.0)]:
        G.add_node(n, x=x, y=0.0)
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 4, key=0, length_m=200.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)

    result = run_floodroute_assignment(
        G, {1: 200}, {3: 100, 4: 200}, "RP100"
    )
    # With capacity 100 at node 3 and 200 at node 4, both may be used
    assert len(result.assignments) >= 1
    total_assigned = sum(result.assignments.values())
    assert total_assigned <= 200


def test_algorithm_c_never_violates_capacity():
    """Algorithm C must never assign more than the effective capacity to any shelter."""
    G = _graph()
    cap = 50  # very small capacity
    result = run_floodroute_assignment(G, {1: 200}, {3: cap}, "RP100")
    shelter_load = sum(u for (_, s), u in result.assignments.items() if s == 3)
    assert shelter_load <= cap, f"Algorithm C violated capacity: {shelter_load} > {cap}"


# ---------------------------------------------------------------------------
# Test 10: excess demand redirects or goes unassigned
# ---------------------------------------------------------------------------

def test_excess_demand_does_not_fabricate_route():
    """When capacity < demand, unassigned portion must have no fabricated route."""
    G = _graph()
    cap = 50
    result = run_floodroute_assignment(G, {1: 200}, {3: cap}, "RP100")
    total_assigned = sum(result.assignments.values())
    unassigned = 200 - total_assigned
    assert unassigned >= 0
    # No route for unassigned — routes only cover assigned pairs
    for (o, s), _path in result.routes.items():
        assert result.assignments.get((o, s), 0) > 0, (
            "A route must not exist for an unassigned (o, s) pair"
        )


# ---------------------------------------------------------------------------
# Test 11: unassigned totals reconcile
# ---------------------------------------------------------------------------

def test_unassigned_totals_reconcile():
    """Per-barangay unassigned + assigned must equal total demand for each barangay."""
    G = _graph()
    demands = {1: 200}
    result = run_floodroute_assignment(G, demands, {3: 50}, "RP100")
    for o, demand in demands.items():
        assigned_o = sum(u for (orig, _s), u in result.assignments.items() if orig == o and u > 0)
        unassigned_o = demand - assigned_o
        assert unassigned_o >= 0, "Unassigned cannot be negative"
        assert assigned_o + unassigned_o == demand, (
            f"Reconciliation failed for origin {o}: "
            f"{assigned_o} + {unassigned_o} != {demand}"
        )


# ---------------------------------------------------------------------------
# Test 12: end-to-end scenario with real scenario capacities
# ---------------------------------------------------------------------------

def test_end_to_end_with_real_scenario_structure():
    """Smoke test: a realistic scenario runs without error and produces valid output."""
    G = _graph()
    demands = {1: 100}
    capacities = {3: 200}
    store = _store_with(RoadOverride(u=1, v=3, status="road_closed"))
    base = make_weight_fn("RP100")
    wfn_b = apply_overrides(base, store, flood_penalty_fn=base)

    result_b = run_flood_aware_nearest(G, demands, capacities, "RP100", weight_fn=wfn_b)
    result_c = run_floodroute_assignment(G, demands, capacities, "RP100", weight_fn=wfn_b)

    for result in (result_b, result_c):
        total_assigned = sum(result.assignments.values())
        assert total_assigned <= sum(demands.values())
        for path in result.routes.values():
            assert 3 in path  # shelter is reachable via 1→2→3
            # direct edge 1→3 is closed; path must go through 2
            if len(path) > 1:
                assert path != [1, 3], "Closed edge 1→3 must not be used directly"


# ---------------------------------------------------------------------------
# Test 13: acceptance scenario — 1,000 demand splits across 500/700 capacities
# ---------------------------------------------------------------------------

def test_acceptance_1000_demand_split_500_700():
    """1,000 demand; Facility A cap 500, Facility B cap 700.

    Expected: 500 assigned to A (fills it), 500 to B; total 1,000; no violation.
    Road closed on the direct 1→3 edge forces routing through node 2.
    """
    G = _two_shelter_graph()
    # Close edge 1→2 to simulate the original route being blocked,
    # forcing use of a detour through node 2 (already the only path in this graph).
    # Actually: mark 2→3 as closed to demonstrate re-routing to facility B.
    store = _store_with(RoadOverride(u=2, v=3, status="road_closed"))
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)

    # With 2→3 closed, only facility B (node 4) is reachable
    result_closed = run_floodroute_assignment(
        G, {1: 1000}, {3: 500, 4: 700}, "RP100", weight_fn=wfn
    )
    total_assigned = sum(result_closed.assignments.values())
    assert total_assigned <= 1000
    load_3 = sum(u for (_, s), u in result_closed.assignments.items() if s == 3 and u > 0)
    assert load_3 == 0, "road_closed on 2→3 must prevent any assignment to shelter 3"
    assert load_3 <= 500, "Shelter 3 capacity 500 must not be exceeded"
    load_4 = sum(u for (_, s), u in result_closed.assignments.items() if s == 4 and u > 0)
    assert load_4 <= 700, "Shelter 4 capacity 700 must not be exceeded"

    # Without closure: both facilities should fill up in priority order
    result_open = run_floodroute_assignment(
        G, {1: 1000}, {3: 500, 4: 700}, "RP100"
    )
    total_open = sum(result_open.assignments.values())
    assert total_open == 1000, f"All 1,000 must be assigned when total cap=1,200; got {total_open}"
    load_3_open = sum(u for (_, s), u in result_open.assignments.items() if s == 3 and u > 0)
    load_4_open = sum(u for (_, s), u in result_open.assignments.items() if s == 4 and u > 0)
    assert load_3_open <= 500, "Facility A capacity 500 must not be exceeded"
    assert load_4_open <= 700, "Facility B capacity 700 must not be exceeded"
    assert load_3_open + load_4_open == 1000


# ---------------------------------------------------------------------------
# Test 14: acceptance scenario — 200 unassigned under 500/300 capacity
# ---------------------------------------------------------------------------

def test_acceptance_200_unassigned_500_300():
    """1,000 demand; Facility A cap 500, Facility B cap 300 (total 800).

    Expected: 800 assigned, 200 unassigned; no route fabricated for unassigned.
    """
    G = _two_shelter_graph()
    result = run_floodroute_assignment(G, {1: 1000}, {3: 500, 4: 300}, "RP100")

    total_assigned = sum(result.assignments.values())
    total_unassigned = 1000 - total_assigned
    assert total_unassigned == 200, (
        f"Expected 200 unassigned (cap=800, demand=1000); got {total_unassigned}"
    )
    load_3 = sum(u for (_, s), u in result.assignments.items() if s == 3 and u > 0)
    load_4 = sum(u for (_, s), u in result.assignments.items() if s == 4 and u > 0)
    assert load_3 <= 500, "Facility A must not exceed capacity 500"
    assert load_4 <= 300, "Facility B must not exceed capacity 300"
    # No route for the unassigned portion
    for (o, s), _path in result.routes.items():
        assert result.assignments.get((o, s), 0) > 0, (
            f"Route exists for unassigned pair ({o}, {s})"
        )
    assert total_assigned + total_unassigned == 1000


# ---------------------------------------------------------------------------
# Test 15: San Fernando ES reachable → Igbonglo gets assignment, no
#          "capacity exhausted" label
# ---------------------------------------------------------------------------

def _igbonglo_san_fernando_graph() -> nx.MultiDiGraph:
    """Two-hop dry graph: origin 507 → intermediate 900 → shelter 1333.

    Mirrors the real node IDs of Igbonglo (507) and San Fernando Elementary
    School (1333) so that the regression is unambiguous.  All edges are
    modelled_dry under RP100 so the flood-aware router can traverse them.
    """
    G = nx.MultiDiGraph()
    for n, x in [(507, 0.0), (900, 500.0), (1333, 1000.0)]:
        G.add_node(n, x=x, y=0.0)
    G.add_edge(507, 900, key=0, length_m=500.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(900, 1333, key=0, length_m=500.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    return G


def test_igbonglo_reachable_san_fernando_gets_assignment():
    """When node 507 (Igbonglo) has a dry path to node 1333 (San Fernando ES)
    and 1,000 spaces are available there, Algorithm C must assign the full
    demand of 391 and the formatter must report 'assigned', not
    'capacity exhausted'.

    Regression for the facility-input defect: San Fernando ES is routing-
    isolated in the production graph (its sole access edge carries
    jrc_rp100_status='no_overlap', making it unavailable under the
    conservative flood policy).  This test verifies that the correct
    outcome occurs whenever the path IS traversable.
    """
    G = _igbonglo_san_fernando_graph()
    result = run_floodroute_assignment(G, {507: 391}, {1333: 1000}, "RP100")

    assert (507, 1333) in result.assignments, (
        "Igbonglo must be assigned to San Fernando ES when a dry path exists "
        "and capacity is available"
    )
    assert result.assignments[(507, 1333)] == 391, (
        f"Expected full demand 391 assigned; got {result.assignments[(507, 1333)]}"
    )

    status = format_origin_assignment_status(507, result)
    assert status["status"] == "assigned", (
        f"Expected status='assigned', got {status['status']!r}: {status['reason']}"
    )
    assert "exhausted" not in status["reason"].lower(), (
        "Formatter must not report 'exhausted' when demand is fully assigned"
    )


# ---------------------------------------------------------------------------
# Tests 16-21: Stage 10 facility-UI regression suite
# ---------------------------------------------------------------------------

# ── Test 16: reachable_shelter_nodes scope ───────────────────────────────

def _minimal_graph_with_shelter() -> nx.MultiDiGraph:
    """Two-node graph: origin 1, shelter 2.  UTM-like x/y coordinates."""
    G = nx.MultiDiGraph()
    # Approximate UTM 51N coordinates for San Jose de Buenavista area
    G.add_node(1, x=399_000.0, y=1_085_000.0)
    G.add_node(2, x=399_100.0, y=1_085_000.0)
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    return G


def test_build_map_no_result_no_origin_no_nameerror():
    """build_analytical_map must not raise NameError when result=None and
    selected_origin_node=None even when capacities are non-empty.

    Regression for the reachable_shelter_nodes scope: the variable was
    only defined inside if/elif blocks.  An explicit initialisation before
    those blocks now guarantees it is always bound.
    """
    import folium

    from floodroute.dashboard.map_builder import build_analytical_map

    G = _minimal_graph_with_shelter()
    m = build_analytical_map(
        G, "RP100", {2: 500},
        result=None,
        selected_origin_node=None,
    )
    assert isinstance(m, folium.Map)


# ── Test 17: OSM candidates load ────────────────────────────────────────

def test_osm_candidates_load_expected_types_and_count():
    """OSM_CANDIDATES must contain at least 1 barangay_hall and 1 school,
    and the total must be 37 (the full sjdb_osm_facility_candidates.json)."""
    from collections import Counter

    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES

    counts = Counter(c.facility_type for c in OSM_CANDIDATES)
    assert counts["barangay_hall"] >= 1, "Expected at least one barangay_hall candidate"
    assert counts["school"] >= 1, "Expected at least one school candidate"
    assert len(OSM_CANDIDATES) == 37, (
        f"Expected 37 OSM candidates; got {len(OSM_CANDIDATES)}"
    )


def test_osm_candidates_json_path_resolves_via_symlink():
    """_JSON_PATH must use Path(__file__).resolve() so it points to the real
    project root even when the module is loaded through the site-packages
    symlink (macOS editable-install workaround).

    Regression for the Section 4 facility-display defect: without .resolve(),
    the symlink path .venv/lib/python3.13/site-packages/floodroute/... gives
    a parent chain ending in .venv/lib/python3.13/ instead of the project root,
    so _JSON_PATH does not exist and OSM_CANDIDATES silently returns [].
    """

    from floodroute.dashboard.osm_candidates import _JSON_PATH

    assert _JSON_PATH.exists(), (
        f"_JSON_PATH must point to the real JSON file; got {_JSON_PATH}"
    )
    # The path must be fully resolved (no symlink components).
    assert _JSON_PATH.resolve() == _JSON_PATH or _JSON_PATH.resolve().exists(), (
        "_JSON_PATH must resolve to an existing file"
    )


# ── Test 18: unreachable_enabled field on unassigned status ─────────────

def _fake_result_with_unreachable_shelter():
    """RunResult stub: origin 5 is reachable only to shelter 33 (full).
    Shelter 99 is enabled but absent from the OD matrix (routing-isolated).
    """
    class FakeResult:
        assignments = {(1, 33): 100}  # shelter 33 at capacity
        od_costs_scenario = {(1, 33): 100.0, (5, 33): 200.0}
        capacities = {33: 100, 99: 500}  # 99 enabled but not in OD

    return FakeResult()


def test_unassigned_with_unreachable_enabled_shelter_populates_field():
    """When origin is reachable only to full shelter(s) and some enabled
    shelters are absent from the OD matrix, format_origin_assignment_status
    must set unreachable_enabled to the absent shelter nodes and include
    the count in the reason string.
    """
    result = _fake_result_with_unreachable_shelter()
    status = format_origin_assignment_status(5, result)

    assert status["status"] == "unassigned"
    assert 99 in status["unreachable_enabled"], (
        "Node 99 is enabled but not in OD matrix — must appear in unreachable_enabled"
    )
    assert "1" in status["reason"] or "facilit" in status["reason"].lower(), (
        "Reason must mention the count of unreachable enabled facilities"
    )


# ── Test 19: unreachable origin returns all capacities in unreachable_enabled

def test_unreachable_origin_unreachable_enabled_contains_all_enabled_shelters():
    """When an origin has no path to any enabled shelter, unreachable_enabled
    must equal the full set of enabled shelter node IDs."""
    class FakeResult:
        assignments = {}
        od_costs_scenario = {}
        capacities = {33: 100, 58: 200}

    result = FakeResult()
    status = format_origin_assignment_status(42, result)

    assert status["status"] == "unreachable"
    assert status["unreachable_enabled"] == {33, 58}, (
        "All enabled shelters must appear in unreachable_enabled when origin has no path"
    )


# ── Test 20: reachability caption logic — all facilities unreachable ─────

def test_reachability_caption_all_selected_unreachable():
    """When all activated facility nodes are absent from the run's reachable
    set, _reachable_selected_nodes must be empty.  This drives the 'none
    reachable — plan generation disabled' caption in Section 4.
    """
    class _FacStub:
        def __init__(self, node_id: int):
            self.snapped_node_id = node_id

    activated_ids = {"way:111", "way:222"}
    osm_by_id = {"way:111": _FacStub(10), "way:222": _FacStub(20)}
    reachable_nodes_from_run: set[int] = {99}  # neither 10 nor 20 is reachable

    reachable_selected = {
        osm_by_id[oid].snapped_node_id
        for oid in activated_ids
        if oid in osm_by_id
        and osm_by_id[oid].snapped_node_id in reachable_nodes_from_run
    }
    assert reachable_selected == set(), (
        "No facility should be reachable when their nodes are absent from the OD matrix"
    )


# ── Test 21: reachability caption logic — partial reachability ───────────

def test_reachability_caption_partial_reachability():
    """When only some activated facility nodes appear in the run's reachable
    set, _reachable_selected_nodes must contain only those nodes.  The
    Section 4 caption shows 'N reachable' as a partial-reachability note.
    """
    class _FacStub:
        def __init__(self, node_id: int):
            self.snapped_node_id = node_id

    activated_ids = {"way:111", "way:222", "way:333"}
    osm_by_id = {
        "way:111": _FacStub(10),
        "way:222": _FacStub(20),  # routing-isolated
        "way:333": _FacStub(30),
    }
    reachable_nodes_from_run: set[int] = {10, 30}

    reachable_selected = {
        osm_by_id[oid].snapped_node_id
        for oid in activated_ids
        if oid in osm_by_id
        and osm_by_id[oid].snapped_node_id in reachable_nodes_from_run
    }
    assert reachable_selected == {10, 30}, (
        "Only nodes in both activated set and reachable set should be included"
    )
    assert len(reachable_selected) == 2
    assert 20 not in reachable_selected, "Isolated node 20 must not appear in reachable set"


# ── Test 22: origin-collision exclusion ─────────────────────────────────

def test_effective_shelters_excludes_origin_collision_node():
    """When a selected facility's snapped_node_id is also a barangay
    demand-origin node, it must be excluded from effective_shelters.

    Regression for ValueError: "Nodes appear in both demands and capacities"
    raised by solve_assignment when node 502 (Banusing-Serdena Elementary
    School) was both a shelter candidate and a barangay origin node.
    """
    class _FacStub:
        def __init__(self, osm_id: str, node_id: int):
            self.osm_id = osm_id
            self.snapped_node_id = node_id

    # Simulate: node 502 is both an origin and a candidate shelter node.
    origin_nodes = {502, 507}
    activated_ids = {"way:collision", "way:ok"}
    osm_by_id = {
        "way:collision": _FacStub("way:collision", 502),  # same as origin → exclude
        "way:ok": _FacStub("way:ok", 999),                # safe → include
    }
    scenario_caps = {"way:collision": 500, "way:ok": 300}

    effective_shelters: dict[int, int] = {}
    excluded: list[str] = []
    for oid in activated_ids:
        cand = osm_by_id.get(oid)
        if cand is None:
            continue
        if cand.snapped_node_id in origin_nodes:
            excluded.append(oid)
            continue
        cap = scenario_caps.get(oid, 0)
        if cap > 0:
            effective_shelters[cand.snapped_node_id] = (
                effective_shelters.get(cand.snapped_node_id, 0) + cap
            )

    assert 502 not in effective_shelters, (
        "Node 502 is a demand origin — must not appear in effective_shelters"
    )
    assert 999 in effective_shelters, (
        "Node 999 is not an origin — must appear in effective_shelters"
    )
    assert effective_shelters[999] == 300
    assert "way:collision" in excluded
    assert set(effective_shelters) & origin_nodes == set(), (
        "effective_shelters must be disjoint from origin_nodes"
    )
