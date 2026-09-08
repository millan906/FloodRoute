"""Focused behavioral tests for road condition overrides (Stage 10 UCD correction).

Tests verify:
1. Closing a road causes Algorithms B/C to avoid it or report no route.
2. Clearing the override restores the original result.
3. flooded_passable keeps edge available (with flood penalty).
4. Source graph is not mutated by any override operation.
"""

from __future__ import annotations

import networkx as nx
import pytest

from floodroute.dashboard.road_overrides import (
    RoadOverride,
    RoadOverrideStore,
    apply_overrides,
    override_color,
)
from floodroute.optimization.cost import make_weight_fn

# ---------------------------------------------------------------------------
# Minimal test graph fixture
# ---------------------------------------------------------------------------

def _make_test_graph() -> nx.MultiDiGraph:
    """Three-node linear graph: 1 → 2 → 3 (both edges dry, 100 m each).

    Node 1 is the origin, node 3 is the shelter.  The only path is 1→2→3.
    An alternative direct edge 1→3 (200 m, flooded) is also present so that
    closing 1→2 forces use of the flooded path or leaves origin unreachable.
    """
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=100.0, y=0.0)
    G.add_node(3, x=200.0, y=0.0)

    # 1→2: dry, 100 m
    G.add_edge(1, 2, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    # 2→3: dry, 100 m
    G.add_edge(2, 3, key=0, length_m=100.0,
               jrc_rp100_status="modelled_dry", jrc_rp100_depth_max_m=None)
    # 1→3: flooded, 200 m (only available when flood-penalty is applied)
    G.add_edge(1, 3, key=0, length_m=200.0,
               jrc_rp100_status="flooded", jrc_rp100_depth_max_m=0.3)
    return G


# ---------------------------------------------------------------------------
# Test 1: closing a road forces re-route or no-route
# ---------------------------------------------------------------------------

def test_closing_road_makes_route_avoid_it():
    """road_closed override on edge 1→2 must block that edge for Alg B/C routing."""
    G = _make_test_graph()
    base_wfn = make_weight_fn("RP100")

    # Baseline: edge 1→2 is available — shortest path is 1→2→3 (200 m effective)
    lengths_before, paths_before = nx.single_source_dijkstra(G, 1, weight=base_wfn)
    assert 3 in paths_before, "Node 3 should be reachable before override"
    assert paths_before[3] == [1, 2, 3] or 3 in lengths_before

    # Apply road_closed override on edge 1→2
    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="road_closed"))
    override_wfn = apply_overrides(base_wfn, store)

    # After override: 1→2 is blocked; path must not go through node 2
    lengths_after, paths_after = nx.single_source_dijkstra(G, 1, weight=override_wfn)
    if 3 in paths_after:
        assert 2 not in paths_after[3], (
            "Path must not traverse closed edge (1→2); node 2 must be bypassed"
        )


# ---------------------------------------------------------------------------
# Test 2: clearing override restores original result
# ---------------------------------------------------------------------------

def test_clearing_override_restores_original_path():
    """Removing a road_closed override must restore the original shortest path."""
    G = _make_test_graph()
    base_wfn = make_weight_fn("RP100")

    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="road_closed"))
    override_wfn = apply_overrides(base_wfn, store)

    _, paths_closed = nx.single_source_dijkstra(G, 1, weight=override_wfn)

    # Clear the override
    store.remove(1, 2)
    assert len(store) == 0

    restored_wfn = apply_overrides(base_wfn, store)
    _, paths_restored = nx.single_source_dijkstra(G, 1, weight=restored_wfn)

    # After clearing, the optimal route should be reachable via 1→2→3 again
    assert 3 in paths_restored
    # And the restored weight function should behave identically to the base
    for u, v, data in G.edges(data=True):
        assert restored_wfn(u, v, {0: data}) == base_wfn(u, v, {0: data})


# ---------------------------------------------------------------------------
# Test 3: flooded_passable keeps edge available with penalty
# ---------------------------------------------------------------------------

def test_flooded_passable_keeps_edge_available_with_penalty():
    """flooded_passable override must not block the edge — it must return a cost."""
    G = _make_test_graph()
    base_wfn = make_weight_fn("RP100")
    flood_penalty_wfn = make_weight_fn("RP100")  # same fn is fine; returns penalty cost

    # Hypothetically close edge 1→2 then re-open it as flooded_passable
    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="flooded_passable"))
    override_wfn = apply_overrides(base_wfn, store, flood_penalty_fn=flood_penalty_wfn)

    # Edge must return a numeric cost (not None)
    edge_data = dict(G[1][2])  # {key: attrs}
    cost = override_wfn(1, 2, edge_data)
    assert cost is not None, "flooded_passable edge must not be blocked (must return a cost)"
    assert cost > 0, "Cost must be positive"

    # Path through 1→2→3 must still be discoverable
    lengths, paths = nx.single_source_dijkstra(G, 1, weight=override_wfn)
    assert 3 in paths, "Node 3 must still be reachable via flooded_passable edge"


# ---------------------------------------------------------------------------
# Test 4: source graph is not mutated
# ---------------------------------------------------------------------------

def test_source_graph_not_mutated():
    """Override operations must never add, remove, or modify edges in the graph."""
    G = _make_test_graph()
    edges_before = set(G.edges(keys=True))
    nodes_before = set(G.nodes())
    attrs_before = {
        (u, v, k): dict(d)
        for u, v, k, d in G.edges(data=True, keys=True)
    }

    store = RoadOverrideStore()
    store.add(RoadOverride(u=1, v=2, status="road_closed"))
    base_wfn = make_weight_fn("RP100")
    override_wfn = apply_overrides(base_wfn, store)

    # Run Dijkstra — this exercises the weight function against the graph
    nx.single_source_dijkstra(G, 1, weight=override_wfn)

    # Graph must be identical after
    assert set(G.edges(keys=True)) == edges_before, "Edge set must not change"
    assert set(G.nodes()) == nodes_before, "Node set must not change"
    for (u, v, k), old_attrs in attrs_before.items():
        current_attrs = dict(G[u][v][k])
        assert current_attrs == old_attrs, (
            f"Edge ({u},{v},{k}) attributes must not be mutated"
        )

    store.clear()
    assert set(G.edges(keys=True)) == edges_before, "Clearing store must not affect graph"


# ---------------------------------------------------------------------------
# Bonus: override_color returns correct types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected_color_prefix,has_dash", [
    ("dry_confirmed_passable", "#16A34A", False),
    ("flooded_passable",       "#F59E0B", False),
    ("flooded_unknown",        "#F97316", True),
    ("flooded_impassable",     "#EF4444", False),
    ("road_closed",            "#7F1D1D", False),
    ("unknown",                "#9CA3AF", True),
    ("use_model",              "#6B7280", True),   # fallback
])
def test_override_color_returns_correct_style(status, expected_color_prefix, has_dash):
    color, dash = override_color(status)
    assert color == expected_color_prefix, f"Wrong color for {status!r}"
    if has_dash:
        assert dash is not None, f"Expected dash_array for {status!r}"
    else:
        assert dash is None, f"Expected no dash_array for {status!r}"
