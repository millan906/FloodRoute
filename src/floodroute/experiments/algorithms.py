"""Stage 8 algorithm implementations.

Three algorithms are compared on the same San Jose de Buenavista RP scenario:

A — Ordinary nearest-shelter
    Routes via plain edge length (no flood penalty, no capacity constraint).
    Each origin's full demand is sent to the nearest reachable shelter.
    Capacity violations are reported but not enforced.

B — Static flood-aware nearest-shelter
    Routes via flood-aware edge costs (conservative hazard policy from
    cost.py: only ``modelled_dry`` and ``flooded`` edges are traversable).
    Each origin's full demand is sent to the nearest reachable shelter by
    flood-aware cost.  No capacity constraint.

C — FloodRoute flood-aware min-cost-flow
    Full capacitated assignment via the existing ``run_flood_assignment``
    pipeline.  Lexicographic objective: maximise assigned demand first,
    then minimise total routing cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx

from floodroute.optimization.assignment import solve_assignment
from floodroute.optimization.routing import compute_od_matrix

ALGORITHMS: tuple[str, ...] = ("A", "B", "C")
RETURN_PERIODS: tuple[str, ...] = ("RP10", "RP20", "RP100")
DEMAND_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50)


@dataclass
class RunResult:
    """Result of one experiment run.

    Attributes are populated by the algorithm functions and consumed by
    ``metrics.compute_metrics``.
    """

    algorithm: str
    """``'A'``, ``'B'``, or ``'C'``."""

    return_period: str
    """Return period for flood scenario (``'RP10'``, ``'RP20'``, ``'RP100'``)."""

    demand_fraction: float
    """Demand fraction applied to PSA 2020 population (0.10 / 0.25 / 0.50)."""

    demands: dict[int, int]
    """``{origin_node: demand_units}`` — input to this run."""

    capacities: dict[int, int]
    """``{shelter_node: capacity_units}`` — scenario capacities (not real data)."""

    assignments: dict[tuple, int]
    """``{(origin_node, shelter_node): assigned_units}``."""

    routes: dict[tuple, list]
    """``{(origin_node, shelter_node): [node_ids]}`` — one route per assigned pair."""

    od_costs_scenario: dict[tuple, float]
    """OD costs used by this algorithm (ordinary or flood-aware metres-equivalent)."""

    od_costs_ordinary: dict[tuple, float]
    """Plain-distance OD costs (always length-only) for detour ratio computation."""

    runtime_s: float
    """Wall-clock seconds for routing + assignment."""


def _ordinary_weight_fn():
    """Return a Dijkstra weight function using plain edge length only.

    All edges are traversable regardless of flood status.  The weight is the
    minimum ``length_m`` across parallel edges (or 0.0 if absent).
    """

    def _weight(u: object, v: object, d: dict) -> float:
        best = float("inf")
        for attrs in d.values():
            length = float(attrs.get("length_m") or 0.0)
            if length < best:
                best = length
        return best if best < float("inf") else 0.0

    return _weight


def _nearest_shelter_assignment(
    demands: dict[int, int],
    capacities: dict[int, int],
    od_costs: dict[tuple, float],
    od_routes: dict[tuple, list],
) -> tuple[dict[tuple, int], dict[tuple, list]]:
    """Greedy nearest-shelter assignment — no capacity constraint.

    Each origin's full demand is assigned to the shelter with the lowest OD
    cost.  Ties broken by shelter node ID (ascending) for determinism.

    Returns
    -------
    assignments : dict[tuple, int]
    routes : dict[tuple, list]
    """
    assignments: dict[tuple, int] = {}
    routes: dict[tuple, list] = {}
    shelters = sorted(capacities)
    for o, demand in sorted(demands.items(), key=lambda x: x[0]):
        if demand <= 0:
            continue
        candidates = [
            (od_costs[(o, s)], s)
            for s in shelters
            if (o, s) in od_costs
        ]
        if not candidates:
            continue  # origin unreachable from all shelters
        _, nearest_s = min(candidates)  # ties broken by shelter ID via sort
        assignments[(o, nearest_s)] = demand
        routes[(o, nearest_s)] = od_routes[(o, nearest_s)]
    return assignments, routes


def run_ordinary_nearest(
    G: nx.MultiDiGraph,
    demands: dict[int, int],
    capacities: dict[int, int],
    return_period: str = "RP100",
) -> RunResult:
    """Algorithm A: ordinary nearest-shelter (length only, no capacity).

    Routes via plain edge length regardless of flood status.  The
    ``return_period`` parameter is accepted for interface uniformity; it
    does not affect routing but is recorded in the result for use when
    computing flood-exposure metrics post-hoc.
    """
    t0 = time.perf_counter()
    ordinary_wfn = _ordinary_weight_fn()

    od_costs_ord: dict[tuple, float] = {}
    od_routes_ord: dict[tuple, list] = {}
    for o in sorted(demands):
        if o not in G:
            continue
        try:
            lengths, paths = nx.single_source_dijkstra(G, o, weight=ordinary_wfn)
        except nx.NetworkXError:
            continue
        for s in capacities:
            if s in lengths:
                od_costs_ord[(o, s)] = lengths[s]
                od_routes_ord[(o, s)] = paths[s]

    assignments, routes = _nearest_shelter_assignment(
        demands, capacities, od_costs_ord, od_routes_ord
    )
    runtime_s = time.perf_counter() - t0

    return RunResult(
        algorithm="A",
        return_period=return_period,
        demand_fraction=0.0,  # overwritten by caller
        demands=demands,
        capacities=capacities,
        assignments=assignments,
        routes=routes,
        od_costs_scenario=od_costs_ord,
        od_costs_ordinary=od_costs_ord,
        runtime_s=runtime_s,
    )


def run_flood_aware_nearest(
    G: nx.MultiDiGraph,
    demands: dict[int, int],
    capacities: dict[int, int],
    return_period: str = "RP100",
) -> RunResult:
    """Algorithm B: flood-aware nearest-shelter (no capacity).

    Routes via flood-aware edge costs (conservative policy: only
    ``modelled_dry`` and ``flooded`` edges are traversable).  Each origin's
    full demand is sent to the nearest reachable shelter by flood-aware cost.
    Capacity violations are reported but not enforced.
    """
    t0 = time.perf_counter()
    ordinary_wfn = _ordinary_weight_fn()

    # Flood-aware OD matrix
    od_costs_flood, od_routes_flood = compute_od_matrix(
        G, list(demands), list(capacities), return_period
    )
    # Ordinary OD matrix (for detour ratio)
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

    assignments, routes = _nearest_shelter_assignment(
        demands, capacities, od_costs_flood, od_routes_flood
    )
    runtime_s = time.perf_counter() - t0

    return RunResult(
        algorithm="B",
        return_period=return_period,
        demand_fraction=0.0,
        demands=demands,
        capacities=capacities,
        assignments=assignments,
        routes=routes,
        od_costs_scenario=od_costs_flood,
        od_costs_ordinary=od_costs_ord,
        runtime_s=runtime_s,
    )


def run_floodroute_assignment(
    G: nx.MultiDiGraph,
    demands: dict[int, int],
    capacities: dict[int, int],
    return_period: str = "RP100",
) -> RunResult:
    """Algorithm C: FloodRoute flood-aware min-cost-flow assignment.

    Uses the proven optimizer from Stage 7 (``solve_assignment``).
    Lexicographic objective: first maximise assigned demand (via DUMMY
    penalty), then minimise total flood-aware routing cost.
    """
    t0 = time.perf_counter()
    ordinary_wfn = _ordinary_weight_fn()

    # Flood-aware OD matrix
    od_costs_flood, od_routes_flood = compute_od_matrix(
        G, list(demands), list(capacities), return_period
    )
    # Ordinary OD matrix (for detour ratio)
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

    mcf_result = solve_assignment(demands, capacities, od_costs_flood, od_routes_flood)
    runtime_s = time.perf_counter() - t0

    return RunResult(
        algorithm="C",
        return_period=return_period,
        demand_fraction=0.0,
        demands=demands,
        capacities=capacities,
        assignments=mcf_result.assignments,
        routes=mcf_result.routes,
        od_costs_scenario=od_costs_flood,
        od_costs_ordinary=od_costs_ord,
        runtime_s=runtime_s,
    )
