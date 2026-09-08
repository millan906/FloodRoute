"""Algorithm B+: CASPER-inspired flood-aware sequential greedy assignment.

B+ is a greedy heuristic inspired by sequential assignment methods (e.g.
the CASPER family).  It enforces shelter capacity and allows demand splitting,
making it directly comparable with Algorithm C (MCF).  It does not implement
iterative re-allocation or agent-based features of CASPER itself and should
be described as CASPER-inspired, not as CASPER.

Comparability with Algorithm C
-------------------------------
Both B+ and C:
- allow divisible integer origin flow (demand splitting across shelters)
- use identical OD pairs
- use identical shelter capacities (adjusted by capacity_multiplier)
- use identical flood policy
- account for all unassigned demand

The intended difference is:
- B+: sequential greedy allocation (heuristic, not globally optimal)
- C:  simultaneous global minimum-cost-flow allocation (exact, optimal for
      the formulated MCF with scalar OD costs)

Algorithm
---------
Priority order is fixed before assignment begins:

1. For each origin, find its minimum flood-aware OD cost to *any* reachable
   shelter (using the scenario flood penalty).
2. Sort origins by that minimum cost in **descending** order (highest-cost
   origins — most distant or most flood-exposed — get first access to shelter
   capacity).  Ties are broken by adm4_pcode (PSGC) ascending; when no psgc
   mapping is supplied, by origin node ID ascending.
3. For each origin in priority order, iterate reachable shelters in ascending
   OD-cost order and assign:

       flow = min(origin_remaining_demand, shelter_remaining_capacity)

   Continue until the origin's demand is fully assigned or no reachable
   shelter has remaining capacity.
4. If an origin has no path to any shelter → "unreachable".
   If demand remains after all shelters are exhausted → "capacity_exhausted".
"""

from __future__ import annotations

import time

import networkx as nx

from floodroute.experiments.algorithms import RunResult, _ordinary_weight_fn
from floodroute.optimization.cost import FloodPolicy
from floodroute.optimization.routing import compute_od_matrix


def run_flood_aware_greedy_capacitated(
    G: nx.MultiDiGraph,
    demands: dict[int, int],
    capacities: dict[int, int],
    return_period: str = "RP100",
    flood_penalty: FloodPolicy = 10.0,
    weight_fn=None,
    origin_psgc: dict[int, str] | None = None,
) -> RunResult:
    """Algorithm B+: CASPER-inspired flood-aware sequential greedy assignment.

    Demand may be split across shelters (comparable to Algorithm C).

    Parameters
    ----------
    G:
        Road graph (``nx.MultiDiGraph``) with JRC flood attributes.
    demands:
        ``{origin_node: demand_units}`` — integer demand at each origin.
    capacities:
        ``{shelter_node: capacity_units}`` — shelter capacity limits.
    return_period:
        Flood scenario: ``'RP10'``, ``'RP20'``, or ``'RP100'``.
    flood_penalty:
        Flood-penalty policy.  Float multiplier or ``"prohibited"``.
        Default ``10.0`` matches production FloodRoute behaviour.
    weight_fn:
        Optional weight function override.  When supplied, overrides
        both ``flood_penalty`` and ``return_period`` for OD routing.
    origin_psgc:
        Optional ``{origin_node: psgc_string}`` mapping for deterministic
        tie-breaking by adm4_pcode ascending.  Falls back to zero-padded
        node ID when not supplied.

    Returns
    -------
    RunResult
        ``algorithm="B+"``.  Origins may be split across multiple shelters.
    """
    t0 = time.perf_counter()
    ordinary_wfn = _ordinary_weight_fn()

    # Flood-aware OD matrix
    od_costs_flood, od_routes_flood = compute_od_matrix(
        G,
        list(demands),
        list(capacities),
        return_period,
        weight_fn=weight_fn,
        flood_penalty=flood_penalty,
    )

    # Ordinary OD matrix (for detour ratio / reporting)
    od_costs_ord: dict[tuple, float] = {}
    for o in sorted(demands):
        if o not in G:
            continue
        try:
            lengths, _ = nx.single_source_dijkstra(G, o, weight=ordinary_wfn)
        except nx.NetworkXError:
            continue
        for s in capacities:
            if s in lengths:
                od_costs_ord[(o, s)] = lengths[s]

    shelters = sorted(capacities)

    def _min_od_cost(o: int) -> float:
        costs = [od_costs_flood[(o, s)] for s in shelters if (o, s) in od_costs_flood]
        return min(costs) if costs else float("inf")

    def _psgc_key(o: int) -> str:
        """Secondary sort key: adm4_pcode if available, else zero-padded node ID."""
        if origin_psgc is not None and o in origin_psgc:
            return origin_psgc[o]
        return f"{o:020d}"

    # Descending min OD cost; ties broken by psgc/node ascending.
    # High-cost origins (most distant / most flood-exposed) get first access
    # to shelter capacity.
    priority_order = sorted(
        demands.keys(),
        key=lambda o: (-_min_od_cost(o), _psgc_key(o)),
    )

    # Greedy sequential assignment with demand splitting
    remaining_capacity: dict[int, int] = dict(capacities)
    assignments: dict[tuple, int] = {}
    routes: dict[tuple, list] = {}

    for o in priority_order:
        remaining_demand = demands.get(o, 0)
        if remaining_demand <= 0:
            continue

        # Reachable shelters sorted by ascending OD cost (cheapest first)
        reachable = sorted(
            [(od_costs_flood[(o, s)], s) for s in shelters if (o, s) in od_costs_flood],
            key=lambda x: (x[0], x[1]),
        )
        if not reachable:
            # Origin unreachable from all shelters
            continue

        for _cost, s in reachable:
            if remaining_demand <= 0:
                break
            avail = remaining_capacity.get(s, 0)
            if avail <= 0:
                continue
            flow = min(remaining_demand, avail)
            key_pair = (o, s)
            assignments[key_pair] = assignments.get(key_pair, 0) + flow
            routes[key_pair] = od_routes_flood[(o, s)]
            remaining_capacity[s] -= flow
            remaining_demand -= flow

    runtime_s = time.perf_counter() - t0

    return RunResult(
        algorithm="B+",
        return_period=return_period,
        demand_fraction=0.0,
        demands=demands,
        capacities=capacities,
        assignments=assignments,
        routes=routes,
        od_costs_scenario=od_costs_flood,
        od_costs_ordinary=od_costs_ord,
        runtime_s=runtime_s,
        flood_penalty=flood_penalty,
    )
