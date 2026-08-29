"""Stage 8 metrics: compute all reported statistics for one experiment run.

Every metric is derived from the ``RunResult`` produced by one of the three
algorithms.  No data is modified; metrics are computed read-only.

Metric definitions
------------------
assigned / unassigned
    Integer demand units successfully assigned to a shelter vs. left
    unassigned (unreachable or capacity-exhausted).

shelter_loads
    Total assigned units per shelter node.

capacity_violations
    Shelters whose load exceeds scenario capacity.  Algorithm C (FloodRoute)
    enforces capacity; A and B do not — violations may appear there.

flood_exposed_length_m
    For each assigned route, sum the ``length_m`` of every edge whose JRC
    flood status equals ``'flooded'`` at the given return period.  Uses the
    minimum-cost edge for each (u, v) pair (consistent with the routing cost
    model).  For algorithm A (ordinary routing) this is computed post-hoc
    using the given return period to quantify the unmitigated exposure.

total_route_cost_m_eq / max_route_cost_m_eq / mean_route_cost_m_eq
    Aggregate, maximum, and mean per-unit route costs using the scenario OD
    costs (flood-aware for B/C, ordinary for A).

detour_ratio
    ``Σ f(o,s) · flood_od_cost(o,s) / Σ f(o,s) · ordinary_od_cost(o,s)``
    for all assigned pairs where both costs are available.  For algorithm A
    this ratio equals 1.0 by definition (both numerator and denominator use
    ordinary costs).

unreachable_origins
    Origin nodes in the demands dict that have no path to any shelter under
    the scenario routing model.  Includes structurally isolated nodes and
    nodes cut off by the conservative flood policy (B/C only).

runtime_s
    Wall-clock seconds as recorded by the algorithm function.
"""

from __future__ import annotations

import networkx as nx

from floodroute.experiments.algorithms import RunResult
from floodroute.optimization.cost import FLOOD_MULTIPLIER


def _best_edge_status(G: nx.MultiDiGraph, u: int, v: int, return_period: str) -> str | None:
    """Status of the minimum-length edge from u to v at the given RP."""
    rp_key = f"jrc_{return_period.lower()}_status"
    best_status: str | None = None
    best_length = float("inf")
    for attrs in G[u][v].values():
        length = float(attrs.get("length_m") or 0.0)
        if length < best_length:
            best_length = length
            best_status = attrs.get(rp_key)
    return best_status


def compute_flood_exposed_length(
    routes: dict[tuple, list],
    assignments: dict[tuple, int],
    G: nx.MultiDiGraph,
    return_period: str,
    algorithm: str = "A",
) -> float:
    """Sum of flooded edge lengths on all assigned routes at the given RP.

    Selects the parallel edge consistent with each algorithm's routing model:

    - Algorithm A (ordinary): minimum-length edge regardless of flood status.
    - Algorithms B/C (flood-aware): minimum-flood-aware-cost edge among
      available edges (``modelled_dry`` or ``flooded`` only).  A dry edge
      always beats a flooded edge of the same length; if all available edges
      are flooded, the shortest flooded edge is selected.

    Unweighted by demand (measures physical route length in metres).
    """
    rp_key = f"jrc_{return_period.lower()}_status"
    total_m = 0.0
    seen_edges: set[tuple] = set()  # avoid double-counting shared edges

    for (o, s), path in routes.items():
        if assignments.get((o, s), 0) <= 0:
            continue
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_key = (u, v)
            if edge_key in seen_edges:
                continue
            if u not in G or v not in G[u]:
                continue
            seen_edges.add(edge_key)

            if algorithm == "A":
                # Ordinary routing: select minimum-length edge (any status).
                best_length = float("inf")
                best_status = None
                for attrs in G[u][v].values():
                    length = float(attrs.get("length_m") or 0.0)
                    if length < best_length:
                        best_length = length
                        best_status = attrs.get(rp_key)
            else:
                # B/C flood-aware routing: select minimum-flood-aware-cost edge
                # among available (modelled_dry / flooded) edges only.
                # A dry edge always beats a flooded edge (cost multiplier > 1),
                # so prefer dry; among ties, prefer shorter.
                best_length = float("inf")
                best_status = None
                best_flood_cost = float("inf")
                for attrs in G[u][v].values():
                    status = attrs.get(rp_key)
                    if status not in ("modelled_dry", "flooded"):
                        continue  # unavailable edge — not usable by B/C
                    length = float(attrs.get("length_m") or 0.0)
                    flood_cost = (
                        length * FLOOD_MULTIPLIER
                        if status == "flooded"
                        else length
                    )
                    if flood_cost < best_flood_cost:
                        best_flood_cost = flood_cost
                        best_length = length
                        best_status = status

            if best_status == "flooded" and best_length < float("inf"):
                total_m += best_length

    return total_m


def compute_metrics(
    result: RunResult,
    G: nx.MultiDiGraph,
) -> dict:
    """Compute all experiment metrics for one run.

    Parameters
    ----------
    result:
        Output from one of the algorithm functions.
    G:
        Road graph with JRC flood attributes (used for flood exposure).

    Returns
    -------
    dict
        Flat dict of metrics suitable for one CSV row.  All keys are strings;
        values are int, float, or str.
    """
    demands = result.demands
    capacities = result.capacities
    assignments = result.assignments
    routes = result.routes
    return_period = result.return_period

    total_demand = sum(demands.values())
    total_assigned = sum(assignments.values())
    total_unassigned = total_demand - total_assigned

    # Shelter loads
    shelter_loads: dict[int, int] = {s: 0 for s in capacities}
    for (_o, s), units in assignments.items():
        shelter_loads[s] = shelter_loads.get(s, 0) + units

    # Capacity violations
    violations: list[int] = [
        s for s, load in shelter_loads.items() if load > capacities[s]
    ]

    # Unreachable origins: in demands but no route to any shelter
    od_keys = set(result.od_costs_scenario)
    unreachable: list[int] = sorted(
        o for o in demands if not any(o == key[0] for key in od_keys)
    )

    # Flood-exposed route length (post-hoc for A; scenario RP for B/C).
    # Edge selection is routing-model-consistent: min-length for A, min-flood-
    # aware-cost among available edges for B/C.
    flood_exposed_m = compute_flood_exposed_length(
        routes, assignments, G, return_period, algorithm=result.algorithm
    )

    # Route cost metrics (use scenario OD costs — ordinary for A, flood-aware for B/C)
    pair_costs = [
        result.od_costs_scenario[(o, s)]
        for (o, s) in assignments
        if (o, s) in result.od_costs_scenario
    ]
    total_cost = sum(assignments.get(k, 0) * c for k, c in result.od_costs_scenario.items() if k in assignments)
    max_cost = max(pair_costs) if pair_costs else 0.0
    mean_cost = (total_cost / total_assigned) if total_assigned > 0 else 0.0

    # Detour ratio: scenario cost / ordinary cost for assigned pairs
    numerator = sum(
        assignments[k] * result.od_costs_scenario[k]
        for k in assignments
        if k in result.od_costs_scenario and k in result.od_costs_ordinary
    )
    denominator = sum(
        assignments[k] * result.od_costs_ordinary[k]
        for k in assignments
        if k in result.od_costs_scenario and k in result.od_costs_ordinary
    )
    detour_ratio = (numerator / denominator) if denominator > 0 else 1.0

    run_id = f"{result.algorithm}_{return_period}_{result.demand_fraction:.2f}"
    return {
        "run_id": run_id,
        "algorithm": result.algorithm,
        "return_period": return_period,
        "demand_fraction": result.demand_fraction,
        "total_population_2020": sum(demands.values()) if result.demand_fraction == 0 else None,
        "total_demand": total_demand,
        "total_assigned": total_assigned,
        "total_unassigned": total_unassigned,
        "assignment_rate": round(total_assigned / total_demand, 4) if total_demand > 0 else 0.0,
        **{f"shelter_{s}_load": shelter_loads.get(s, 0) for s in sorted(capacities)},
        **{f"shelter_{s}_capacity": capacities[s] for s in sorted(capacities)},
        "num_capacity_violations": len(violations),
        "capacity_violation_shelters": ",".join(str(s) for s in sorted(violations)),
        "flood_exposed_length_m": round(flood_exposed_m, 2),
        "total_route_cost_m_eq": round(total_cost, 2),
        "max_route_cost_m_eq": round(max_cost, 2),
        "mean_route_cost_m_eq": round(mean_cost, 2),
        "detour_ratio": round(detour_ratio, 4),
        "num_origins": len(demands),
        "num_unreachable_origins": len(unreachable),
        "unreachable_origin_nodes": ",".join(str(n) for n in unreachable),
        "runtime_s": round(result.runtime_s, 3),
    }
