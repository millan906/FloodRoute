"""Flood-aware edge cost functions for evacuation routing.

The cost of traversing an edge equals its physical length multiplied by a
flood-exposure penalty determined by the JRC flood status for the chosen
return period.

Conservative hazard policy
--------------------------
Only two JRC flood statuses produce a finite traversal cost.  Every other
status — including statuses that merely lack hydraulic evidence — makes an
edge *unavailable* to the router (weight function returns ``None``).

  modelled_dry   → ordinary cost (length_m)
                   Within the hydraulic model domain; actively classified as
                   not inundated.  Full positive evidence of safety.

  flooded        → penalised cost (length_m × flood_penalty)
                   Within the domain; actively classified as inundated.

  no_overlap     → unavailable (None)
                   No JRC pixel has interior overlap ≥ MIN_INTERIOR_LEN_M
                   with this edge (edge too short or at a coverage gap).
                   Absence of pixel data is NOT evidence of dryness; the
                   edge carries no hydraulic confirmation in either direction
                   and must not be treated as a safe route.

  outside_domain → unavailable (None)
                   Pixels overlap the edge but carry NODATA (-9999): the
                   hydraulic model was not run for those pixels.  No
                   hydraulic evidence is available.

  None / unknown → unavailable (None)
                   Status attribute missing entirely (key absent or value is
                   None after a GraphML round-trip), or carries a value not
                   in the recognised vocabulary (e.g. from a future raster
                   version).  No hydraulic confirmation → unavailable.

Design notes
------------
- ``_PENALISED_STATUSES`` and ``_ORDINARY_STATUSES`` enumerate every status
  that produces a finite cost.  Any value absent from both sets is
  unavailable by explicit fall-through.
- ``edge_cost`` returns ``None`` for unavailable edges (including zero-length
  edges whose status is unavailable).  Exception: a zero-length edge with a
  *known* status (modelled_dry or flooded) returns ``0.0``.
- ``make_weight_fn`` returns a callable compatible with NetworkX Dijkstra on
  a MultiDiGraph: it receives ``(u, v, d)`` where *d* is a dict of
  ``{edge_key: attr_dict}`` and returns the minimum finite cost across all
  available parallel edges, or ``None`` when every parallel edge is
  unavailable.  Returning ``None`` causes NetworkX to skip the edge entirely.

Configurable flood penalty
--------------------------
``make_weight_fn`` accepts a ``flood_penalty`` parameter of type
``FloodPolicy``.  When ``flood_penalty`` is a float, flooded edges receive
cost ``length_m × flood_penalty``.  When ``flood_penalty == "prohibited"``,
flooded edges are treated as unavailable (same as ``no_overlap``).

The production default ``flood_penalty=10.0`` reproduces the original
behaviour exactly (``FLOOD_MULTIPLIER = 10.0``).
"""

from __future__ import annotations

#: Type alias for flood penalty specification.
#: float  → penalty multiplier (e.g. 1.0, 5.0, 10.0, 20.0)
#: "prohibited" → flooded edges blocked entirely (unavailable)
FloodPolicy = float | str

FLOOD_MULTIPLIER: float = 10.0
"""Production default cost multiplier applied to edges classified as flooded."""

VALID_FLOOD_PENALTIES: tuple = (1.0, 5.0, 10.0, 20.0, "prohibited")
"""Recognised flood-penalty values for experiments."""

_PENALISED_STATUSES: frozenset[str] = frozenset({"flooded"})
"""Status values that attract the flood penalty."""

_ORDINARY_STATUSES: frozenset[str] = frozenset({"modelled_dry"})
"""Status values that receive ordinary cost (length_m)."""

_UNAVAILABLE_STATUSES: frozenset[str] = frozenset({"no_overlap", "outside_domain"})
"""Named status values that are unavailable under the conservative policy.
These are documented here for testing; the cost logic treats *any* value
outside ``_PENALISED_STATUSES | _ORDINARY_STATUSES`` as unavailable."""


def edge_cost(attrs: dict, return_period: str = "RP100") -> float | None:
    """Flood-aware travel cost for a single edge (metres-equivalent).

    Parameters
    ----------
    attrs:
        Edge attribute dict containing at minimum ``length_m`` and optionally
        ``jrc_rp<N>_status`` (e.g. ``jrc_rp100_status``).
    return_period:
        One of ``'RP10'``, ``'RP20'``, ``'RP100'``.

    Returns
    -------
    float or None
        ``length_m * FLOOD_MULTIPLIER`` when status is ``'flooded'``.
        ``length_m`` when status is ``'modelled_dry'``.
        ``0.0`` when ``length_m`` is zero and status is known (modelled_dry
        or flooded).
        ``None`` for all unavailable edges: ``no_overlap``, ``outside_domain``,
        missing key, ``None`` value, or any unrecognised status string.

    Notes
    -----
    A ``None`` return signals to ``make_weight_fn`` (and via it to NetworkX
    Dijkstra) that this edge should be skipped entirely.

    This function always uses ``FLOOD_MULTIPLIER`` (10.0) for the flood
    penalty.  For configurable penalties, use ``make_weight_fn`` with the
    ``flood_penalty`` parameter.
    """
    rp_key = f"jrc_{return_period.lower()}_status"
    status = attrs.get(rp_key)  # None when key absent or value is None

    if status not in _PENALISED_STATUSES and status not in _ORDINARY_STATUSES:
        # no_overlap, outside_domain, None, unknown → unavailable
        return None

    length = float(attrs.get("length_m") or 0.0)
    if status in _PENALISED_STATUSES:
        return length * FLOOD_MULTIPLIER
    # modelled_dry
    return length


def make_weight_fn(
    return_period: str = "RP100",
    flood_penalty: FloodPolicy = 10.0,
):
    """Return a Dijkstra weight function for a NetworkX MultiDiGraph.

    The returned callable is compatible with ``nx.single_source_dijkstra``
    on a ``MultiDiGraph``: it receives ``(u, v, d)`` where *d* is a dict of
    ``{edge_key: attr_dict}`` and returns the minimum cost across all
    *available* parallel edges between *u* and *v*, or ``None`` when every
    parallel edge is unavailable.

    Returning ``None`` causes NetworkX to treat the edge as absent, so
    unavailable edges are never selected by the router.

    Parameters
    ----------
    return_period:
        Flood scenario to use for the penalty lookup.
    flood_penalty:
        Flood-penalty policy.  Pass a float (e.g. ``10.0``) to multiply
        flooded edge lengths by that factor.  Pass ``"prohibited"`` to treat
        flooded edges as unavailable (blocked entirely).  The production
        default ``10.0`` reproduces the original FLOOD_MULTIPLIER behaviour.
    """
    rp_key = f"jrc_{return_period.lower()}_status"
    prohibited = flood_penalty == "prohibited"
    penalty: float = 0.0 if prohibited else float(flood_penalty)

    def _weight(u: object, v: object, d: dict) -> float | None:
        best: float | None = None
        for attrs in d.values():
            status = attrs.get(rp_key)
            if status not in _PENALISED_STATUSES and status not in _ORDINARY_STATUSES:
                continue  # unavailable edge
            if status in _PENALISED_STATUSES and prohibited:
                continue  # flooded + prohibited → unavailable
            length = float(attrs.get("length_m") or 0.0)
            c = length * penalty if status in _PENALISED_STATUSES else length
            if best is None or c < best:
                best = c
        return best

    return _weight
