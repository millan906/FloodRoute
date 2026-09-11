"""Stage 8 / Stage 11 metrics: compute all reported statistics for one experiment run.

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
    Shelters whose load exceeds scenario capacity.  Algorithms A and B do not
    enforce shelter capacity — violations may appear.  Algorithm C (exact MCF)
    and B+ (greedy) enforce capacity as hard constraints; neither can produce
    violations.  Assignment rate must always be read alongside overflow:
    100% assignment with violations for A/B means shelter load exceeded
    capacity.

flood_exposed_length_m
    For each assigned route, sum the ``length_m`` of every edge whose JRC
    flood status equals ``'flooded'`` at the given return period.  Uses the
    minimum-cost edge for each (u, v) pair (consistent with the routing cost
    model).  For algorithm A (ordinary routing) this is computed post-hoc
    using the given return period to quantify the unmitigated exposure.

total_route_cost_m_eq / max_route_cost_m_eq / mean_route_cost_m_eq
    Aggregate, maximum, and mean per-unit route costs using the scenario OD
    costs (flood-aware for B/B+/C, ordinary for A).

detour_ratio
    ``Σ f(o,s) · flood_od_cost(o,s) / Σ f(o,s) · ordinary_od_cost(o,s)``
    for all assigned pairs where both costs are available.  For algorithm A
    this ratio equals 1.0 by definition (both numerator and denominator use
    ordinary costs).  When the ordinary-cost denominator is zero (undefined),
    the ratio falls back to ``1.0``.

physical_route_distance_m / dry_route_distance_m / flooded_route_distance_m
    Population-weighted sums in person·metres:
    - physical = total route length (all edges)
    - flooded  = length of edges classified ``'flooded'`` at the given RP
    - dry      = physical − flooded (these three are additive)

flooded_person_distance_m
    Synonym for ``flooded_route_distance_m`` (person·metres on flooded
    segments), recorded separately for clarity.

penalized_cost_m_eq
    Population-weighted sum of scenario OD costs (metres-equivalent):
    ordinary distances for A, flood-penalised distances for B/B+/C.

unreachable_origins
    Origin nodes in the demands dict that have no path to any shelter under
    the scenario routing model.  Includes structurally isolated nodes and
    nodes cut off by the conservative flood policy (B/B+/C only).

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


def physical_path_distances(
    path: list[int],
    G: nx.MultiDiGraph,
    return_period: str,
) -> tuple[float, float]:
    """Return ``(physical_m, flooded_m)`` for a single node-list path.

    ``physical_m`` is the sum of minimum edge lengths along the path.
    ``flooded_m`` is the subset of that length on edges whose JRC status is
    ``'flooded'`` at the given return period.
    """
    rp_key = f"jrc_{return_period.lower()}_status"
    phys_m = 0.0
    flood_m = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u not in G or v not in G[u]:
            continue
        best_length = float("inf")
        best_status: str | None = None
        for attrs in G[u][v].values():
            length = float(attrs.get("length_m") or 0.0)
            if length < best_length:
                best_length = length
                best_status = attrs.get(rp_key)
        if best_length < float("inf"):
            phys_m += best_length
            if best_status == "flooded":
                flood_m += best_length
    return phys_m, flood_m


def compute_metrics(
    result: RunResult,
    G: nx.MultiDiGraph | None = None,
    total_population: int | None = None,
) -> dict:
    """Compute all reported metrics for one experiment run.

    Parameters
    ----------
    result:
        ``RunResult`` from one of the algorithm functions.
    G:
        Road graph.  Required for flood-exposure and physical-distance metrics;
        when ``None`` those metrics default to 0.0.
    total_population:
        Optional total population for context (not used in rate calculations).
    """
    # ------------------------------------------------------------------ demand
    total_demand = sum(result.demands.values())

    shelter_loads: dict[int, int] = {}
    for (_o, s), flow in result.assignments.items():
        if flow > 0:
            shelter_loads[s] = shelter_loads.get(s, 0) + flow

    total_assigned = sum(flow for flow in result.assignments.values() if flow > 0)
    total_unassigned = total_demand - total_assigned
    assignment_rate = total_assigned / total_demand if total_demand > 0 else 0.0

    # ---------------------------------------------------------------- overflow
    capacity_violations: list[int] = []
    total_overflow = 0
    for s, cap in result.capacities.items():
        load = shelter_loads.get(s, 0)
        overflow = max(0, load - cap)
        if overflow > 0:
            capacity_violations.append(s)
            total_overflow += overflow

    # -------------------------------------------------------- unreachable / reasons
    od_origins = {k[0] for k in result.od_costs_scenario}
    assigned_origins = {k[0] for k, v in result.assignments.items() if v > 0}
    unreachable_origins: list[int] = []
    unassigned_by_reason: dict[str, int] = {}
    for o, d in result.demands.items():
        if d <= 0 or o in assigned_origins:
            continue
        if o not in od_origins:
            unreachable_origins.append(o)
            unassigned_by_reason["unreachable"] = (
                unassigned_by_reason.get("unreachable", 0) + d
            )
        else:
            unassigned_by_reason["capacity_exhausted"] = (
                unassigned_by_reason.get("capacity_exhausted", 0) + d
            )

    # -------------------------------------------------------------- route costs
    total_cost = 0.0
    max_cost = 0.0
    total_weighted_units = 0
    penalized_cost = 0.0

    for (o, s), flow in result.assignments.items():
        if flow <= 0:
            continue
        cost = result.od_costs_scenario.get((o, s), 0.0)
        total_cost += cost * flow
        max_cost = max(max_cost, cost)
        total_weighted_units += flow
        penalized_cost += cost * flow

    mean_cost = total_cost / total_weighted_units if total_weighted_units > 0 else 0.0

    # ---------------------------------------------------------- detour ratio
    num_detour = 0.0
    denom_detour = 0.0
    for (o, s), flow in result.assignments.items():
        if flow <= 0:
            continue
        flood_c = result.od_costs_scenario.get((o, s))
        ord_c = result.od_costs_ordinary.get((o, s))
        if flood_c is not None and ord_c is not None and ord_c > 0:
            num_detour += flood_c * flow
            denom_detour += ord_c * flow
    detour_ratio = num_detour / denom_detour if denom_detour > 0 else 1.0

    # ------------------------------------------ flood-exposed + physical distances
    flood_exposed_length_m = 0.0
    physical_route_distance_m = 0.0
    flooded_route_distance_m = 0.0

    if G is not None:
        flood_exposed_length_m = compute_flood_exposed_length(
            result.routes, result.assignments, G,
            result.return_period, result.algorithm,
        )
        for (o, s), path in result.routes.items():
            flow = result.assignments.get((o, s), 0)
            if flow <= 0:
                continue
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if u not in G or v not in G[u]:
                    continue
                best_length = min(
                    (float(attrs.get("length_m") or 0.0) for attrs in G[u][v].values()),
                    default=0.0,
                )
                best_status = _best_edge_status(G, u, v, result.return_period)
                physical_route_distance_m += best_length * flow
                if best_status == "flooded":
                    flooded_route_distance_m += best_length * flow

    dry_route_distance_m = physical_route_distance_m - flooded_route_distance_m

    # Per-shelter load keys for stage8 backward compatibility
    per_shelter_load = {f"shelter_{s}_load": load for s, load in shelter_loads.items()}
    per_shelter_capacity = {
        f"shelter_{s}_capacity": result.capacities.get(s, 0)
        for s in shelter_loads
    }

    run_id = (
        f"{result.algorithm}_{result.return_period}_{result.demand_fraction:.2f}"
        if result.algorithm and result.return_period
        else ""
    )

    return {
        "total_demand": total_demand,
        "total_assigned": total_assigned,
        "total_unassigned": total_unassigned,
        "assigned_population": total_assigned,
        "unassigned_population": total_unassigned,
        "assignment_rate": assignment_rate,
        "shelter_loads": shelter_loads,
        "capacity_violations": capacity_violations,
        "num_capacity_violations": len(capacity_violations),
        "total_overflow_units": total_overflow,
        "unreachable_origins": unreachable_origins,
        "num_unreachable_origins": len(unreachable_origins),
        "unreachable_origin_nodes": ", ".join(str(n) for n in sorted(unreachable_origins)),
        "unassigned_by_reason": unassigned_by_reason,
        "total_route_cost_m_eq": total_cost,
        "max_route_cost_m_eq": max_cost,
        "mean_route_cost_m_eq": mean_cost,
        "detour_ratio": detour_ratio,
        "penalized_cost_m_eq": penalized_cost,
        "flood_exposed_length_m": flood_exposed_length_m,
        "physical_route_distance_m": physical_route_distance_m,
        "flooded_route_distance_m": flooded_route_distance_m,
        "flooded_person_distance_m": flooded_route_distance_m,
        "dry_route_distance_m": dry_route_distance_m,
        "runtime_s": result.runtime_s,
        "algorithm": result.algorithm,
        "return_period": result.return_period,
        "demand_fraction": result.demand_fraction,
        "run_id": run_id,
        **per_shelter_load,
        **per_shelter_capacity,
    }
