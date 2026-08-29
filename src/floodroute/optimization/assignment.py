"""Capacitated minimum-cost shelter assignment via NetworkX min_cost_flow.

Formulation
-----------
Given a set of origin nodes with integer demand and shelter nodes with integer
capacity, and a flood-aware OD cost matrix for reachable pairs, the assignment
problem is:

    minimise   Σ_{o,s} f[o,s] · cost[o,s]
    subject to Σ_s f[o,s] ≤ demand[o]       for each origin o
               Σ_o f[o,s] ≤ capacity[s]     for each shelter s
               f[o,s] = 0                    if (o,s) unreachable
               f[o,s] ≥ 0

Demand that cannot be assigned (because total capacity is exhausted or because
an origin has no reachable shelter) is reported as ``total_unassigned``.

Min-cost flow network
---------------------
A small auxiliary DiGraph is built with:
- SOURCE   → origin_i   : cap=demand[i],   weight=0
- origin_i → shelter_j  : cap=demand[i],   weight=int_cost[i,j]  (reachable pairs)
- origin_i → DUMMY      : cap=demand[i],   weight=DUMMY_PENALTY (see below)
- shelter_j → SINK      : cap=capacity[j], weight=0
- DUMMY    → SINK       : cap=Σ demand[i], weight=0

All edge weights are integers (see "Integer scaling" below).

Integer scaling
---------------
NetworkX's ``network_simplex`` is not guaranteed to work correctly with
floating-point edge weights (see NetworkX source, line ~400: "This algorithm
is not guaranteed to work if edge weights … are floating point numbers").

OD path costs (float metres from Dijkstra) are therefore converted to integer
weights before the flow network is built:

    int_cost = int(round(cost_m × _INT_COST_SCALE))

where ``_INT_COST_SCALE = 1000`` (metre → millimetre precision).

The reported ``AssignmentResult.objective_cost`` is computed from the
**original float OD costs** (not from the scaled integers), preserving
sub-millimetre accuracy in the output.

DUMMY penalty and lexicographic guarantee
-----------------------------------------
The lexicographic objective — first maximise assigned demand, then minimise
routing cost — is enforced by the DUMMY penalty.  Let:
  - D  = total_demand
  - C  = max(int_cost) over all reachable (o,s) pairs

  DUMMY_PENALTY = D × C + 1

**Proof of lexicographic correctness (in integer arithmetic):**
Consider any feasible flow F with k ≥ 1 units routed to DUMMY, and an
alternative feasible flow F' in which those k units use real paths instead
(possible only when capacity is available).  The cost difference is:

  cost(F) - cost(F')
    ≥ k × (DUMMY_PENALTY - C)
    = k × (D × C + 1 - C)
    = k × ((D - 1) × C + 1)
    ≥ k × 1  > 0        [since D ≥ 1, C ≥ 0]

So cost(F) > cost(F') whenever F' is feasible.  The solver therefore routes
to DUMMY only when no real capacity remains — i.e., only when leaving demand
unassigned is the unique feasible choice.  Reachable evacuees are never left
unassigned merely because their route is expensive.

The ``+1`` is computed in pure integer arithmetic (no float addition), so it
is exact regardless of the magnitude of D or C.

Determinism
-----------
Edge insertion order in the flow network follows ``sorted()`` on all dict keys
so that identical inputs always produce identical DiGraph structures and solver
outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx

from floodroute.optimization.routing import compute_od_matrix

# Internal sentinel node names.  These must not collide with any user-supplied
# node ID (user nodes may be any type; these are strings with a rare prefix).
_SOURCE = "__FR_SOURCE__"
_SINK = "__FR_SINK__"
_DUMMY = "__FR_DUMMY__"

# OD path costs in metres (float) are multiplied by this factor and rounded to
# produce integer edge weights for network_simplex.  1 mm resolution.
_INT_COST_SCALE: int = 1000


@dataclass
class AssignmentResult:
    """Result of a flood-aware capacitated shelter assignment.

    All demand values are integer units matching the input ``demands`` dict.
    """

    assignments: dict[tuple, int]
    """Mapping of (origin_node, shelter_node) → assigned demand units."""

    total_assigned: int
    """Sum of all assigned units across all origin–shelter pairs."""

    total_unassigned: int
    """Demand that could not be assigned (capacity exceeded or unreachable)."""

    shelter_loads: dict
    """Mapping of shelter_node → total units assigned to that shelter."""

    capacity_violations: list[str]
    """Human-readable descriptions of capacity violations (normally empty)."""

    objective_cost: float
    """Weighted assignment cost: Σ f[o,s] · path_cost[o,s] (metres-equivalent).
    Computed from original float OD costs; does not include penalty for
    unassigned demand."""

    runtime_s: float
    """Wall-clock time for the full pipeline (routing + solver), in seconds."""

    routes: dict[tuple, list]
    """Mapping of (origin_node, shelter_node) → ordered list of node IDs on the
    flood-aware shortest path.  Only populated for pairs with f[o,s] > 0."""


def solve_assignment(
    demands: dict,
    capacities: dict,
    od_costs: dict[tuple, float],
    od_routes: dict[tuple, list],
) -> AssignmentResult:
    """Solve a capacitated min-cost assignment using NetworkX min_cost_flow.

    Parameters
    ----------
    demands:
        ``{origin_node: demand_units}`` — non-negative integers.  Every key
        must be absent from *capacities* (origins and shelters are distinct).
    capacities:
        ``{shelter_node: capacity_units}`` — non-negative integers.
    od_costs:
        ``{(origin, shelter): path_cost_m}`` — reachable pairs only.  Path
        costs are in metres (float).  Keys must reference nodes present in
        *demands* and *capacities*.
    od_routes:
        ``{(origin, shelter): list_of_node_ids}`` — parallel to *od_costs*.

    Returns
    -------
    AssignmentResult
        ``runtime_s`` is set to ``0.0``; use ``run_flood_assignment`` to
        obtain a timed result for the full pipeline.

    Raises
    ------
    ValueError
        If any node appears in both *demands* and *capacities*.
    nx.NetworkXUnfeasible
        If the flow network is infeasible (should not occur with the DUMMY
        fallback path, but raised if the network is otherwise malformed).
    """
    shared = set(demands) & set(capacities)
    if shared:
        raise ValueError(
            f"Nodes appear in both demands and capacities: {sorted(str(n) for n in shared)}"
        )

    total_demand = sum(demands.values())

    if total_demand == 0:
        return AssignmentResult(
            assignments={},
            total_assigned=0,
            total_unassigned=0,
            shelter_loads={s: 0 for s in capacities},
            capacity_violations=[],
            objective_cost=0.0,
            runtime_s=0.0,
            routes={},
        )

    # Scale float OD costs to integers for network_simplex correctness.
    # Rule: int_cost = int(round(cost_m × _INT_COST_SCALE)), 1 mm resolution.
    int_od: dict[tuple, int] = {
        k: int(round(v * _INT_COST_SCALE)) for k, v in od_costs.items()
    }

    # Compute DUMMY penalty in pure integer arithmetic (see module docstring).
    # max_int_cost is the ceiling of the highest real assignment cost in
    # scaled units; DUMMY_PENALTY is strictly greater than any real path cost.
    max_int_cost: int = max(int_od.values()) if int_od else 0
    dummy_penalty: int = total_demand * max_int_cost + 1

    # Build the min-cost flow auxiliary network.
    # Edge insertion order is deterministic (sorted keys) so that identical
    # inputs produce identical DiGraph structures and solver outputs.
    FG: nx.DiGraph = nx.DiGraph()
    FG.add_node(_SOURCE, demand=-total_demand)
    FG.add_node(_SINK, demand=total_demand)
    FG.add_node(_DUMMY, demand=0)

    for o in sorted(demands, key=str):
        d_val = demands[o]
        FG.add_node(o, demand=0)
        FG.add_edge(_SOURCE, o, capacity=d_val, weight=0)
        # Fallback path for demand that cannot reach any shelter.
        FG.add_edge(o, _DUMMY, capacity=d_val, weight=dummy_penalty)

    for (o, s), int_cost in sorted(int_od.items(), key=str):
        if o in demands and s in capacities:
            FG.add_edge(o, s, capacity=demands[o], weight=int_cost)

    for s in sorted(capacities, key=str):
        FG.add_node(s, demand=0)
        FG.add_edge(s, _SINK, capacity=capacities[s], weight=0)

    FG.add_edge(_DUMMY, _SINK, capacity=total_demand, weight=0)

    flow_dict: dict = nx.min_cost_flow(FG)

    # Extract real assignments (ignore DUMMY flows).
    assignments: dict[tuple, int] = {}
    for o in demands:
        for s in capacities:
            f = flow_dict.get(o, {}).get(s, 0)
            if f > 0:
                assignments[(o, s)] = int(round(f))

    # Unassigned = flow that reached DUMMY.
    total_unassigned = int(
        round(sum(flow_dict.get(o, {}).get(_DUMMY, 0) for o in demands))
    )
    total_assigned = total_demand - total_unassigned

    # Objective cost: use original float OD costs (not scaled integers) for
    # sub-millimetre accuracy in the reported result.
    objective_cost = sum(float(f) * od_costs[(o, s)] for (o, s), f in assignments.items())

    # Shelter loads.
    shelter_loads = {
        s: sum(assignments.get((o, s), 0) for o in demands) for s in capacities
    }

    # Capacity violations (should be empty — solver enforces constraints).
    violations: list[str] = []
    for s, load in shelter_loads.items():
        if load > capacities[s]:
            violations.append(
                f"shelter {s}: load={load} exceeds capacity={capacities[s]}"
            )

    # Routes for all assigned pairs (only pairs with positive flow).
    routes: dict[tuple, list] = {
        (o, s): od_routes[(o, s)]
        for (o, s) in assignments
        if (o, s) in od_routes
    }

    return AssignmentResult(
        assignments=assignments,
        total_assigned=total_assigned,
        total_unassigned=total_unassigned,
        shelter_loads=shelter_loads,
        capacity_violations=violations,
        objective_cost=objective_cost,
        runtime_s=0.0,
        routes=routes,
    )


def run_flood_assignment(
    G: nx.MultiDiGraph,
    demands: dict,
    capacities: dict,
    return_period: str = "RP100",
) -> AssignmentResult:
    """Full pipeline: flood-aware OD routing + capacitated min-cost assignment.

    Parameters
    ----------
    G:
        Road graph (``nx.MultiDiGraph``) with JRC flood attributes.
    demands:
        ``{origin_node: demand_units}`` — integer demand at each origin.
    capacities:
        ``{shelter_node: capacity_units}`` — integer capacity at each shelter.
    return_period:
        Flood scenario: ``'RP10'``, ``'RP20'``, or ``'RP100'``.

    Returns
    -------
    AssignmentResult
        ``runtime_s`` is the total wall-clock time for routing and solving.
    """
    t0 = time.perf_counter()

    od_costs, od_routes = compute_od_matrix(
        G,
        list(demands.keys()),
        list(capacities.keys()),
        return_period,
    )

    result = solve_assignment(demands, capacities, od_costs, od_routes)
    result.runtime_s = time.perf_counter() - t0
    return result
