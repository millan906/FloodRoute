"""OD-matrix computation: flood-aware shortest paths from origins to shelters.

For each origin node, a single Dijkstra sweep produces the shortest path cost
and route to every reachable node.  Origin–shelter pairs that are disconnected
in the flood-penalised graph are simply omitted from the output dicts; the
assignment solver treats absent pairs as unreachable (zero flow allowed).

Complexity: O(|origins| × (|E| log |V|))  — acceptable for Stage 7 feasibility
scenarios with a small number of controlled origin and shelter nodes.
"""

from __future__ import annotations

import networkx as nx

from floodroute.optimization.cost import make_weight_fn


def compute_od_matrix(
    G: nx.MultiDiGraph,
    origins: list,
    shelters: list,
    return_period: str = "RP100",
) -> tuple[dict[tuple, float], dict[tuple, list]]:
    """Flood-aware shortest-path costs and routes from every origin to every shelter.

    Parameters
    ----------
    G:
        Road graph (``nx.MultiDiGraph``) with JRC flood attributes on edges.
        Each edge must carry ``length_m``; optionally ``jrc_rp<N>_status``.
    origins:
        Sequence of origin node IDs present in *G*.
    shelters:
        Sequence of shelter node IDs present in *G*.
    return_period:
        Flood scenario for edge-cost computation: ``'RP10'``, ``'RP20'``,
        or ``'RP100'``.

    Returns
    -------
    od_costs : dict[(origin, shelter), float]
        Flood-aware path cost in metres-equivalent.  Only reachable pairs
        are included.
    od_routes : dict[(origin, shelter), list]
        Ordered list of node IDs on the shortest path.  Only reachable pairs
        are included.
    """
    weight_fn = make_weight_fn(return_period)
    shelter_set = set(shelters)
    od_costs: dict[tuple, float] = {}
    od_routes: dict[tuple, list] = {}

    for o in origins:
        if o not in G:
            continue
        try:
            lengths, paths = nx.single_source_dijkstra(G, o, weight=weight_fn)
        except nx.NetworkXError:
            continue
        for s in shelter_set:
            if s in lengths:
                od_costs[(o, s)] = lengths[s]
                od_routes[(o, s)] = paths[s]

    return od_costs, od_routes
