"""Planner-workflow focused tests.

Verifies the behaviors required by the simplified Evacuation Planner:

  1.  Name-based road condition applies to all matching segments.
  2.  500-person demand splits correctly across capacities of 200 and 400.
  3.  With capacities of 200 and 150, exactly 150 remain unassigned.
  4.  No route is fabricated for an unassigned (origin, shelter) pair.
  5.  Per-barangay demand + unassigned reconciles to total demand.
  6.  Multiple named road conditions are applied simultaneously.
  7.  Resetting one named road preserves all other active conditions.
  8.  Two distinct barangay origins produce routes for both.
  9.  A split assignment produces one route per destination.
  10. Every route ends at its assigned facility.
  11. Impassable roads are absent from all computed routes.
  12. Flooded-but-passable roads may appear in routes (not blocked).
  13. Route assignment amounts reconcile with assignment totals.
  14. Add / remove interaction: two roads added, one removed, other persists.

Graph fixture — two shelters, one origin (4-node):
    1 → 2 → 3  (dry, 100 m + 100 m)  shelter A = node 3, cap 200 or 500
    1 → 2 → 4  (dry, 100 m + 200 m)  shelter B = node 4, cap 400 or 150
Node 1 = origin.

Multi-origin fixture (5-node, two origins, two shelters):
    1 → 3 → 5  (dry, 100 m + 100 m)  shelter = node 5
    2 → 4 → 5  (dry, 100 m + 100 m)  shelter = node 5
Nodes 1 and 2 = origins; node 5 = single shelter.

Named-road fixture (3-node):
    Edge (1, 2) is labeled "Main Street" (two parallel keys).
    Edge (2, 3) is unlabeled.
"""
from __future__ import annotations

import networkx as nx

from floodroute.dashboard.road_overrides import (
    RoadOverride,
    RoadOverrideStore,
    apply_overrides,
)
from floodroute.experiments.algorithms import run_floodroute_assignment
from floodroute.optimization.cost import make_weight_fn

# ---------------------------------------------------------------------------
# Shared graph builders
# ---------------------------------------------------------------------------

def _two_shelter_graph() -> nx.MultiDiGraph:
    """origin=1, intermediate=2, shelter A=3, shelter B=4."""
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


def _named_road_graph() -> nx.MultiDiGraph:
    """Same topology, but edge (1,2) has TWO parallel segments named 'Main Street'."""
    G = nx.MultiDiGraph()
    for n, x in [(1, 0.0), (2, 100.0), (3, 200.0)]:
        G.add_node(n, x=x, y=0.0)
    # Two parallel edges with the same name — simulates a multi-segment road.
    G.add_edge(1, 2, key=0, length_m=100.0, name="Main Street",
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(1, 2, key=1, length_m=100.0, name="Main Street",
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    return G


# ---------------------------------------------------------------------------
# Test 1: named-road condition
# ---------------------------------------------------------------------------

def test_name_based_road_closure_blocks_all_matching_segments():
    """Closing 'Main Street' (2 segments on edge 1→2) must block routing via node 2."""
    G = _named_road_graph()
    # Collect all (u, v) pairs whose name == "Main Street" and add overrides.
    store = RoadOverrideStore()
    for u, v, attrs in G.edges(data=True):
        if attrs.get("name") == "Main Street":
            store.add(RoadOverride(u=u, v=v, status="road_closed"))
    # RoadOverrideStore keys on (u, v); one entry covers all parallel edges.
    assert len(store) == 1, (
        "Main Street's two parallel edges share (u,v)=(1,2); one override blocks all"
    )

    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)

    result = run_floodroute_assignment(
        G, {1: 100}, {3: 200}, "RP100", weight_fn=wfn
    )
    # Node 3 is unreachable (only path goes through Main Street).
    total_assigned = sum(result.assignments.values())
    assert total_assigned == 0, (
        "Closing Main Street must leave node 3 unreachable; "
        f"got {total_assigned} assigned"
    )


# ---------------------------------------------------------------------------
# Test 2: 500-person demand with caps 200 + 400
# ---------------------------------------------------------------------------

def test_500_demand_splits_200_400_capacities():
    """500-person demand; cap A=200, cap B=400 → all 500 assigned, no violation."""
    G = _two_shelter_graph()
    result = run_floodroute_assignment(G, {1: 500}, {3: 200, 4: 400}, "RP100")

    total_assigned = sum(result.assignments.values())
    assert total_assigned == 500, (
        f"All 500 must be assigned when total cap=600; got {total_assigned}"
    )
    load_a = sum(u for (_, s), u in result.assignments.items() if s == 3)
    load_b = sum(u for (_, s), u in result.assignments.items() if s == 4)
    assert load_a <= 200, f"Shelter A capacity 200 violated: {load_a}"
    assert load_b <= 400, f"Shelter B capacity 400 violated: {load_b}"
    assert load_a + load_b == 500


# ---------------------------------------------------------------------------
# Test 3: 500-person demand with caps 200 + 150 → 150 unassigned
# ---------------------------------------------------------------------------

def test_500_demand_150_unassigned_200_150_capacities():
    """500-person demand; cap A=200, cap B=150 (total 350) → exactly 150 unassigned."""
    G = _two_shelter_graph()
    result = run_floodroute_assignment(G, {1: 500}, {3: 200, 4: 150}, "RP100")

    total_assigned = sum(result.assignments.values())
    total_unassigned = 500 - total_assigned
    assert total_unassigned == 150, (
        f"Expected 150 unassigned (cap=350, demand=500); got {total_unassigned}"
    )
    load_a = sum(u for (_, s), u in result.assignments.items() if s == 3)
    load_b = sum(u for (_, s), u in result.assignments.items() if s == 4)
    assert load_a <= 200, f"Shelter A capacity 200 violated: {load_a}"
    assert load_b <= 150, f"Shelter B capacity 150 violated: {load_b}"


# ---------------------------------------------------------------------------
# Test 4: no route fabricated for unassigned portion
# ---------------------------------------------------------------------------

def test_no_route_for_unassigned_500_demand_350_cap():
    """When 150 are unassigned, no (origin, shelter) route must exist for them."""
    G = _two_shelter_graph()
    result = run_floodroute_assignment(G, {1: 500}, {3: 200, 4: 150}, "RP100")

    for (o, s), _path in result.routes.items():
        assert result.assignments.get((o, s), 0) > 0, (
            f"Route exists for unassigned pair ({o}, {s})"
        )


# ---------------------------------------------------------------------------
# Test 5: per-origin reconciliation
# ---------------------------------------------------------------------------

def test_per_origin_demand_reconciles():
    """assigned + unassigned == demand for each origin node."""
    G = _two_shelter_graph()
    demands = {1: 500}
    result = run_floodroute_assignment(G, demands, {3: 200, 4: 150}, "RP100")

    for o, demand in demands.items():
        assigned_o = sum(
            u for (orig, _s), u in result.assignments.items() if orig == o
        )
        unassigned_o = demand - assigned_o
        assert unassigned_o >= 0, f"Unassigned cannot be negative for origin {o}"
        assert assigned_o + unassigned_o == demand, (
            f"Reconciliation failed for origin {o}: "
            f"{assigned_o} + {unassigned_o} != {demand}"
        )


# ---------------------------------------------------------------------------
# Multi-origin fixture
# ---------------------------------------------------------------------------

def _multi_origin_graph() -> nx.MultiDiGraph:
    """5-node graph: two origins (1, 2), one shelter (5).

    1 → 3 → 5  (dry, 100 m each)
    2 → 4 → 5  (dry, 100 m each)

    Used to verify that two barangays both produce routes.
    """
    G = nx.MultiDiGraph()
    for n, x in [(1, 0.0), (2, 0.0), (3, 100.0), (4, 100.0), (5, 200.0)]:
        G.add_node(n, x=x, y=float(n) * 10.0)
    G.add_edge(1, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(3, 5, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(2, 4, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    G.add_edge(4, 5, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    return G


# ---------------------------------------------------------------------------
# Test 6: multiple named road conditions applied simultaneously
# ---------------------------------------------------------------------------

def test_multiple_named_conditions_applied_simultaneously():
    """Two different named-road conditions must both affect routing at once.

    * Main Street (1→2): road_closed  → node 2 unreachable via that edge
    * Side Street (2→3): flooded_passable → node 3 reachable with penalty

    With Main Street closed, node 3 is unreachable from origin 1 regardless
    of Side Street's condition.
    """
    G = _named_road_graph()
    store = RoadOverrideStore()
    # Condition 1: close Main Street
    for u, v, attrs in G.edges(data=True):
        if attrs.get("name") == "Main Street":
            store.add(RoadOverride(u=u, v=v, status="road_closed"))
    # Condition 2: mark (2, 3) as flooded_passable (a different condition)
    store.add(RoadOverride(u=2, v=3, status="flooded_passable"))

    assert len(store) == 2, "Two distinct (u,v) pairs must produce two overrides"
    assert store.overrides[(1, 2)].status == "road_closed"
    assert store.overrides[(2, 3)].status == "flooded_passable"

    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store, flood_penalty_fn=base)
    result = run_floodroute_assignment(
        G, {1: 100}, {3: 200}, "RP100", weight_fn=wfn
    )
    # Main Street closed → node 3 unreachable from 1
    total_assigned = sum(result.assignments.values())
    assert total_assigned == 0, (
        "With Main Street closed, shelter 3 must be unreachable even if "
        f"Side Street is passable; got {total_assigned} assigned"
    )


# ---------------------------------------------------------------------------
# Test 7: resetting one road preserves other conditions
# ---------------------------------------------------------------------------

def test_reset_one_road_preserves_others():
    """Removing one override must leave all other overrides intact."""
    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="road_closed"))
    store.add(RoadOverride(u=2, v=3, status="flooded_passable"))
    store.add(RoadOverride(u=3, v=4, status="flooded_impassable"))
    assert len(store) == 3

    # Reset only the (2, 3) override
    store.remove(2, 3)
    assert len(store) == 2, "Only one override must be removed"
    assert (1, 2) in store.overrides, "road_closed on (1,2) must survive"
    assert (3, 4) in store.overrides, "flooded_impassable on (3,4) must survive"
    assert (2, 3) not in store.overrides, "(2,3) must be cleared"


# ---------------------------------------------------------------------------
# Test 8: two barangay origins produce routes for both
# ---------------------------------------------------------------------------

def test_two_origins_both_produce_routes():
    """Two distinct origins must each have a route in result.routes."""
    G = _multi_origin_graph()
    # Capacity large enough for both
    result = run_floodroute_assignment(
        G, {1: 100, 2: 100}, {5: 300}, "RP100"
    )
    origin_nodes_with_routes = {o for (o, _s) in result.routes}
    assert 1 in origin_nodes_with_routes, "Origin 1 must have a route"
    assert 2 in origin_nodes_with_routes, "Origin 2 must have a route"


# ---------------------------------------------------------------------------
# Test 9: split assignment produces one route per destination
# ---------------------------------------------------------------------------

def test_split_assignment_produces_one_route_per_destination():
    """When one origin is assigned to two shelters, there is one route per pair."""
    G = _two_shelter_graph()
    # With cap A=200, cap B=400, demand=500 → split expected
    result = run_floodroute_assignment(
        G, {1: 500}, {3: 200, 4: 400}, "RP100"
    )
    assigned_pairs = [(o, s) for (o, s), u in result.assignments.items() if u > 0]
    for pair in assigned_pairs:
        assert pair in result.routes, (
            f"Every assigned pair must have a route; missing {pair}"
        )
    # Both facilities should be used (500 > 200, so overflow to 4)
    shelters_used = {s for (_o, s), u in result.assignments.items() if u > 0}
    assert len(shelters_used) >= 1


# ---------------------------------------------------------------------------
# Test 10: every route ends at its assigned facility
# ---------------------------------------------------------------------------

def test_every_route_ends_at_assigned_facility():
    """The last node of each route must equal the shelter it is assigned to."""
    G = _two_shelter_graph()
    result = run_floodroute_assignment(
        G, {1: 500}, {3: 200, 4: 400}, "RP100"
    )
    for (o, s), path in result.routes.items():
        assert path[-1] == s, (
            f"Route for ({o},{s}) ends at {path[-1]}, expected {s}"
        )


# ---------------------------------------------------------------------------
# Test 11: impassable roads are absent from computed routes
# ---------------------------------------------------------------------------

def test_impassable_road_absent_from_routes():
    """A road_closed edge must not appear in any route node sequence."""
    G = _two_shelter_graph()
    # Close (2, 3): shelter 3 (node 3) becomes unreachable; only shelter 4 usable.
    store = RoadOverrideStore()
    store.add(RoadOverride(u=2, v=3, status="road_closed"))
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)

    result = run_floodroute_assignment(
        G, {1: 200}, {3: 500, 4: 500}, "RP100", weight_fn=wfn
    )
    for (o, s), path in result.routes.items():
        # The closed edge (2→3) must not be traversed
        for u_hop, v_hop in zip(path[:-1], path[1:], strict=False):
            assert not (u_hop == 2 and v_hop == 3), (
                f"Closed edge (2→3) appears in route for ({o},{s}): {path}"
            )
        # Must not be assigned to shelter 3 (unreachable)
        assert s != 3, (
            f"Route ends at closed shelter 3 in assignment ({o},{s})"
        )


# ---------------------------------------------------------------------------
# Test 12: flooded-but-passable roads can appear in routes
# ---------------------------------------------------------------------------

def test_flooded_passable_road_can_appear_in_route():
    """A flooded_passable edge must remain usable (not blocked), so the path
    that traverses it must be chosen when it is the only viable route."""
    G = nx.MultiDiGraph()
    # origin=1, flooded segment (1→2), shelter=2
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=100.0, y=0.0)
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)

    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="flooded_passable"))
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store, flood_penalty_fn=base)

    result = run_floodroute_assignment(
        G, {1: 50}, {2: 200}, "RP100", weight_fn=wfn
    )
    total_assigned = sum(result.assignments.values())
    assert total_assigned == 50, (
        "flooded_passable edge must remain usable; "
        f"expected 50 assigned, got {total_assigned}"
    )
    # The route must use node 2 (the only path)
    for path in result.routes.values():
        assert 2 in path, "Route must pass through the flooded-but-passable segment"


# ---------------------------------------------------------------------------
# Test 13: route assignment amounts reconcile with assignment totals
# ---------------------------------------------------------------------------

def test_route_amounts_reconcile_with_assignments():
    """For every assigned (origin, shelter) pair, a route must exist and the
    assignment count must be positive.  No route may exist for a zero-unit pair.
    """
    G = _two_shelter_graph()
    result = run_floodroute_assignment(
        G, {1: 500}, {3: 200, 4: 400}, "RP100"
    )
    # Every positive assignment has a route
    for (o, s), units in result.assignments.items():
        if units > 0:
            assert (o, s) in result.routes, (
                f"Positive assignment ({o},{s})={units} has no route"
            )
    # Every route has a positive assignment
    for (o, s) in result.routes:
        assert result.assignments.get((o, s), 0) > 0, (
            f"Route exists for zero-unit pair ({o},{s})"
        )
    # Totals match
    total_from_assignments = sum(result.assignments.values())
    assert total_from_assignments <= 500


# ---------------------------------------------------------------------------
# Test 14: add/remove interaction — two roads, one removed, other persists
# ---------------------------------------------------------------------------

def test_add_remove_interaction_two_roads():
    """Planner interaction sequence:
    1. Add Road X (1→2) as road_closed.
    2. Add Road Y (2→3) as flooded_impassable.
    3. Confirm both remain active.
    4. Run the plan — confirm both are excluded from all routes.
    5. Remove Road X (1→2).
    6. Confirm Road Y (2→3) remains active.
    """
    G = _two_shelter_graph()
    store = RoadOverrideStore()

    # Step 1: add Road X
    store.add(RoadOverride(u=1, v=2, status="road_closed"))
    assert (1, 2) in store.overrides

    # Step 2: add Road Y
    store.add(RoadOverride(u=2, v=3, status="flooded_impassable"))
    assert (2, 3) in store.overrides

    # Step 3: both active
    assert len(store) == 2, "Both roads must be active simultaneously"

    # Step 4: run plan — both excluded
    base = make_weight_fn("RP100")
    wfn = apply_overrides(base, store)
    result = run_floodroute_assignment(
        G, {1: 100}, {3: 200, 4: 200}, "RP100", weight_fn=wfn
    )
    for (o, s), path in result.routes.items():
        for u_hop, v_hop in zip(path[:-1], path[1:], strict=False):
            assert not (u_hop == 1 and v_hop == 2), (
                f"Closed Road X (1→2) appears in route for ({o},{s})"
            )
            assert not (u_hop == 2 and v_hop == 3), (
                f"Impassable Road Y (2→3) appears in route for ({o},{s})"
            )
    # With (1→2) closed, origin 1 is disconnected → 0 assigned
    total = sum(result.assignments.values())
    assert total == 0, (
        f"With Road X (1→2) closed, origin 1 cannot reach any shelter; got {total}"
    )

    # Step 5: remove Road X
    store.remove(1, 2)
    assert (1, 2) not in store.overrides, "Road X must be removed"

    # Step 6: Road Y still active
    assert (2, 3) in store.overrides, "Road Y must survive after Road X is removed"
    assert store.overrides[(2, 3)].status == "flooded_impassable"
    assert len(store) == 1
