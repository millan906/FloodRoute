"""Pure-Python result formatting for the FloodRoute dashboard.

All functions are side-effect-free and testable without Streamlit or
geospatial dependencies.
"""

from __future__ import annotations

ALGORITHM_LABELS: dict[str, str] = {
    "A": "A — Ordinary nearest",
    "B": "B — Flood-aware nearest",
    "C": "C — Global shelter allocation (exact MCF)",
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


def get_shelter_display_labels(
    capacities: dict[int, int],
    custom_labels: dict[int, str] | None = None,
) -> dict[int, str]:
    """Build display label strings for the given shelter nodes.

    Merges ``SHELTER_DISPLAY_LABELS`` (nodes 33/58) with caller-supplied
    ``custom_labels``.  For any node not in either source, returns
    ``'Node {node_id}'``.

    Parameters
    ----------
    capacities:
        Shelter capacities dict ``{node_id: capacity}`` — used to determine
        which nodes need labels.
    custom_labels:
        Optional caller-supplied labels that take precedence over the default
        ``SHELTER_DISPLAY_LABELS``.

    Returns
    -------
    dict[int, str]
        Label string for every node in *capacities*.
    """
    labels: dict[int, str] = {}
    for node in capacities:
        if custom_labels and node in custom_labels:
            labels[node] = custom_labels[node]
        else:
            labels[node] = SHELTER_DISPLAY_LABELS.get(node, f"Node {node}")
    return labels


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
