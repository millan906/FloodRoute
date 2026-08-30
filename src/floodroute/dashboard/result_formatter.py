"""Pure-Python result formatting for the FloodRoute dashboard.

All functions are side-effect-free and testable without Streamlit or
geospatial dependencies.
"""

from __future__ import annotations

ALGORITHM_LABELS: dict[str, str] = {
    "A": "A — Ordinary nearest",
    "B": "B — Flood-aware nearest",
    "C": "C — FloodRoute MCF",
}

RP_LABELS: dict[str, str] = {
    "RP10": "10-year return period",
    "RP20": "20-year return period",
    "RP100": "100-year return period",
}

_VALID_ALGORITHMS = frozenset(ALGORITHM_LABELS)
_VALID_RPS = frozenset(RP_LABELS)
_VALID_FRACTIONS = frozenset({0.10, 0.25, 0.50})

#: Canonical display labels for scenario shelters (used across dashboard panels).
SHELTER_DISPLAY_LABELS: dict[int, str] = {
    33: "Scenario Shelter A (node 33)",
    58: "Scenario Shelter B (node 58)",
}


def validate_inputs(algorithm: str, return_period: str, demand_fraction: float) -> list[str]:
    """Return a list of validation error strings (empty list = valid).

    Parameters
    ----------
    algorithm:
        Must be ``'A'``, ``'B'``, or ``'C'``.
    return_period:
        Must be ``'RP10'``, ``'RP20'``, or ``'RP100'``.
    demand_fraction:
        Must be one of ``0.10``, ``0.25``, ``0.50``.
    """
    errors: list[str] = []
    if algorithm not in _VALID_ALGORITHMS:
        errors.append(f"Algorithm must be A, B, or C; got {algorithm!r}")
    if return_period not in _VALID_RPS:
        errors.append(f"Return period must be RP10, RP20, or RP100; got {return_period!r}")
    # Use rounded comparison to avoid float equality issues
    if round(demand_fraction, 10) not in {round(f, 10) for f in _VALID_FRACTIONS}:
        errors.append(
            f"Demand fraction must be 0.10, 0.25, or 0.50; got {demand_fraction}"
        )
    return errors


def format_feasibility_status(metrics: dict) -> dict:
    """Classify the scenario outcome and collect warning strings.

    Returns
    -------
    dict
        ``status``: ``'ok'`` | ``'warning'`` | ``'error'``
        ``label``: human-readable headline
        ``warnings``: list of warning strings (may be empty)
    """
    warnings: list[str] = []

    assignment_rate = float(metrics.get("assignment_rate", 1.0))
    unassigned = int(metrics.get("total_unassigned", 0))
    violations = int(metrics.get("num_capacity_violations", 0))
    unreachable = int(metrics.get("num_unreachable_origins", 0))

    if unassigned > 0:
        warnings.append(
            f"{unassigned:,} demand units unassigned ({assignment_rate:.1%} coverage)"
        )
    if violations > 0:
        shelters = metrics.get("capacity_violation_shelters", "")
        warnings.append(f"Capacity exceeded at shelter(s): {shelters}")
    if unreachable > 0:
        warnings.append(
            f"{unreachable} representative origin node(s) are disconnected from the "
            "main routable component in the derived OSM road graph. "
            "The physical cause has not been field-verified."
        )

    if assignment_rate == 0.0:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    label_map = {
        "ok": "Feasible",
        "warning": "Feasible with warnings",
        "error": "No assignment produced",
    }
    return {"status": status, "label": label_map[status], "warnings": warnings}


def format_shelter_loads(metrics: dict, capacities: dict[int, int]) -> list[dict]:
    """Return per-shelter load rows suitable for tabular display.

    Each row dict has keys:
    ``shelter`` (str), ``load`` (int), ``capacity`` (int),
    ``utilization`` (str, e.g. ``'84.2%'``), ``over_capacity`` (bool).
    """
    rows: list[dict] = []
    for node in sorted(capacities):
        load = int(metrics.get(f"shelter_{node}_load", 0))
        cap = int(metrics.get(f"shelter_{node}_capacity", capacities[node]))
        pct = load / cap if cap > 0 else 0.0
        label = SHELTER_DISPLAY_LABELS.get(node, f"Node {node}")
        rows.append(
            {
                "shelter": label,
                "load": load,
                "capacity": cap,
                "utilization": f"{pct:.1%}",
                "over_capacity": load > cap,
            }
        )
    return rows


def format_recommendation(algorithm: str, metrics: dict) -> str:
    """Return a one-sentence scenario recommendation for display.

    Based on assignment rate, detour ratio, and flood exposure.
    """
    rate = float(metrics.get("assignment_rate", 1.0))
    detour = float(metrics.get("detour_ratio", 1.0))
    exposed_m = float(metrics.get("flood_exposed_length_m", 0.0))

    if algorithm == "A":
        if exposed_m > 0:
            return (
                f"Algorithm A ignores flood hazard; {exposed_m:,.0f} m of route crosses "
                "flooded road — consider Algorithm B or C."
            )
        return (
            "Algorithm A routes are flood-free at this return period "
            "(post-hoc flood exposure = 0 m)."
        )

    if algorithm == "B":
        if detour > 1.005:
            return (
                f"Algorithm B avoids flood risk with a {detour:.3f}× detour factor; "
                "capacity is not enforced — violations may appear."
            )
        return (
            "Algorithm B routes match ordinary routes at this return period "
            "(no active flood penalty applied)."
        )

    # Algorithm C
    if rate < 1.0:
        return (
            f"Algorithm C assigned {rate:.1%} of demand within capacity constraints; "
            "total shelter capacity is insufficient for this demand level."
        )
    if detour > 1.005:
        return (
            f"Algorithm C fully assigned all demand with a {detour:.3f}× "
            "flood-avoidance detour factor."
        )
    return (
        "Algorithm C fully assigned all demand within capacity constraints "
        "with no flood-avoidance detour at this return period."
    )


def format_run_label(algorithm: str, return_period: str, demand_fraction: float) -> str:
    """Return a compact run label, e.g. ``'C / RP20 / 25%'``."""
    return f"{algorithm} / {return_period} / {demand_fraction:.0%}"


def format_origin_assignment_status(origin_node: int, result) -> dict:
    """Determine how the algorithm handled a specific origin node.

    Intended for Algorithm C (MCF) where capacity constraints can leave
    reachable origins unassigned.  For Algorithms A and B every reachable
    origin is assigned (no capacity enforcement), so the result will always
    be ``'assigned'`` or ``'unreachable'``.

    Parameters
    ----------
    origin_node:
        Graph node ID of the origin to inspect.
    result:
        ``RunResult`` from one of the algorithm functions.

    Returns
    -------
    dict
        ``status``  : ``'assigned'`` | ``'unreachable'`` | ``'unassigned'``
        ``shelter`` : int | None — shelter node the algorithm assigned to
        ``units``   : int — units assigned (0 when not assigned)
        ``reason``  : str — human-readable explanation
    """
    # Check for any positive assignment from this origin
    assigned_pairs: dict[tuple, int] = {
        (o, s): u
        for (o, s), u in result.assignments.items()
        if o == origin_node and u > 0
    }
    if assigned_pairs:
        best_pair = max(assigned_pairs, key=assigned_pairs.__getitem__)
        return {
            "status": "assigned",
            "shelter": best_pair[1],
            "units": assigned_pairs[best_pair],
            "reason": (
                f"Assigned {assigned_pairs[best_pair]:,} units "
                f"to shelter node {best_pair[1]}"
            ),
        }

    # Determine why not assigned: unreachable or capacity-exhausted
    reachable = any(o == origin_node for (o, _s) in result.od_costs_scenario)
    if not reachable:
        return {
            "status": "unreachable",
            "shelter": None,
            "units": 0,
            "reason": "No path to any shelter under the routing model",
        }
    return {
        "status": "unassigned",
        "shelter": None,
        "units": 0,
        "reason": "Shelter capacity exhausted — demand not assigned by Algorithm C",
    }


def format_barangay_card(
    *,
    bgy_name: str,
    population_2020: int,
    demand: int,
    assigned: int,
    shelter_node: int | None,
    ordinary_dist_m: float | None,
    alg_dist_m: float | None,
    flood_exposed_m: float | None,
    assignment_status: dict,
) -> dict:
    """Format the per-barangay result card for display.

    All values are pre-computed by the caller; this function only formats
    them into display-ready strings.  Pure-Python — no graph or Streamlit
    dependencies.

    Parameters
    ----------
    bgy_name:
        Human-readable barangay name.
    population_2020:
        PSA 2020 census population.
    demand:
        Scenario demand units (population × demand fraction).
    assigned:
        Demand units assigned by the algorithm (0 when unassigned or not run).
    shelter_node:
        Graph node ID of the assigned shelter, or ``None``.
    ordinary_dist_m:
        Total route length in metres for the ordinary shortest path, or
        ``None`` when the path could not be computed.
    alg_dist_m:
        Total route length in metres for the selected-algorithm path, or
        ``None``.
    flood_exposed_m:
        Length in metres of the algorithm path that crosses flooded road
        under the selected return period, or ``None``.
    assignment_status:
        Dict from ``format_origin_assignment_status`` (or a synthetic dict
        for Algorithms A/B).  Must contain ``'status'`` and ``'reason'``.

    Returns
    -------
    dict
        Display-ready strings keyed by field name:
        ``bgy_name``, ``population_2020``, ``demand``, ``assigned``,
        ``unassigned``, ``shelter``, ``ordinary_dist``, ``alg_dist``,
        ``flood_exposed``, ``status``, ``reason``, ``show_reason``.
    """
    unassigned = max(0, demand - assigned)
    shelter_str = (
        SHELTER_DISPLAY_LABELS.get(shelter_node, f"Node {shelter_node}")
        if shelter_node is not None
        else "—"
    )
    status = assignment_status.get("status", "unknown")
    return {
        "bgy_name": bgy_name,
        "population_2020": f"{population_2020:,}",
        "demand": f"{demand:,} scenario demand units",
        "assigned": f"{assigned:,}",
        "unassigned": f"{unassigned:,}",
        "shelter": shelter_str,
        "ordinary_dist": f"{ordinary_dist_m:,.0f} m" if ordinary_dist_m is not None else "—",
        "alg_dist": f"{alg_dist_m:,.0f} m" if alg_dist_m is not None else "—",
        "flood_exposed": f"{flood_exposed_m:,.0f} m" if flood_exposed_m is not None else "—",
        "status": status.capitalize(),
        "reason": assignment_status.get("reason", ""),
        "show_reason": status not in ("assigned", "not_run"),
    }
