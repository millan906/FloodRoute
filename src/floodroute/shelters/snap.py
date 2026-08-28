"""Deterministic road-graph entrance snapping for FloodRoute Stage 6B.

Snaps a shelter entrance coordinate (WGS-84) to the nearest node in a
projected road graph (EPSG:32651).  Tie-breaking is deterministic: among
nodes equidistant to millimetre precision, the one with the smallest
node_id (string sort) is chosen.

Thresholds
----------
WARN_THRESHOLD_M  — snap distance that triggers a warning but is accepted.
REJECT_THRESHOLD_M — snap distance at which the candidate is rejected.

Both thresholds are configurable at call-site via keyword arguments.

Only eligible, routable candidates are snapped; all others are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from pyproj import Transformer
from shapely.geometry import Point
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Default thresholds (metres)
# ---------------------------------------------------------------------------

DEFAULT_WARN_M: float = 100.0
DEFAULT_REJECT_M: float = 500.0

# CRS constants
_WGS84 = "EPSG:4326"
_UTM51N = "EPSG:32651"
_COORD_PRECISION = 3  # millimetre precision for distance comparison

# Transformer is thread-safe after construction; build once at module level.
_TO_UTM: Transformer = Transformer.from_crs(_WGS84, _UTM51N, always_xy=True)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SnapResult:
    """Outcome of snapping one shelter entrance to the road graph.

    Attributes
    ----------
    shelter_id:
        Identifier of the shelter candidate.
    snapped_node_id:
        Graph node id of the nearest node.  None if snapping failed.
    snap_distance_m:
        Projected distance in metres to the snapped node.  None if failed.
    entrance_utm_x:
        Projected X coordinate (UTM) of the entrance.  None if failed.
    entrance_utm_y:
        Projected Y coordinate (UTM) of the entrance.  None if failed.
    snapped_node_x:
        X coordinate of the snapped node (UTM metres).
    snapped_node_y:
        Y coordinate of the snapped node (UTM metres).
    status:
        One of: "snapped", "warned", "rejected", "skipped", "no_entrance".
    message:
        Human-readable detail (populated on warn/reject/skip/no_entrance).
    """

    shelter_id: str
    snapped_node_id: str | None
    snap_distance_m: float | None
    entrance_utm_x: float | None
    entrance_utm_y: float | None
    snapped_node_x: float | None
    snapped_node_y: float | None
    status: str  # "snapped" | "warned" | "rejected" | "skipped" | "no_entrance"
    message: str = ""

    @property
    def accepted(self) -> bool:
        """True if the snap result is usable (snapped or warned)."""
        return self.status in {"snapped", "warned"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _project_point(lat: float, lon: float) -> tuple[float, float]:
    """Project WGS-84 lat/lon to UTM Zone 51N (EPSG:32651) x, y in metres."""
    x, y = _TO_UTM.transform(lon, lat)
    return float(x), float(y)


def _build_strtree(graph: nx.Graph) -> tuple[STRtree, list[str]]:
    """Build a shapely STRtree from graph node UTM coordinates.

    Returns (tree, node_ids) where node_ids[i] corresponds to tree geometry i.
    """
    node_ids: list[str] = []
    points: list[Point] = []
    for node_id, data in graph.nodes(data=True):
        node_ids.append(str(node_id))
        points.append(Point(data["x"], data["y"]))
    tree = STRtree(points)
    return tree, node_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def snap_entrance_to_graph(
    shelter_id: str,
    entrance_lat: float,
    entrance_lon: float,
    graph: nx.Graph,
    *,
    warn_m: float = DEFAULT_WARN_M,
    reject_m: float = DEFAULT_REJECT_M,
    _strtree_cache: tuple[STRtree, list[str]] | None = None,
) -> SnapResult:
    """Snap a shelter entrance to the nearest road-graph node.

    Parameters
    ----------
    shelter_id:
        Identifier used in the result (for reporting).
    entrance_lat, entrance_lon:
        WGS-84 coordinates of the shelter entrance.
    graph:
        NetworkX graph with node attributes ``x`` and ``y`` (UTM metres).
    warn_m:
        Distance threshold in metres above which a warning is issued.
    reject_m:
        Distance threshold in metres above which the snap is rejected.
    _strtree_cache:
        Pre-built (STRtree, node_ids) tuple.  Pass this when snapping many
        shelters against the same graph to avoid rebuilding the tree each call.

    Returns
    -------
    SnapResult with status "snapped", "warned", or "rejected".
    """
    if graph.number_of_nodes() == 0:
        return SnapResult(
            shelter_id=shelter_id,
            snapped_node_id=None,
            snap_distance_m=None,
            entrance_utm_x=None,
            entrance_utm_y=None,
            snapped_node_x=None,
            snapped_node_y=None,
            status="rejected",
            message="Graph has no nodes.",
        )

    ent_x, ent_y = _project_point(entrance_lat, entrance_lon)
    entrance_pt = Point(ent_x, ent_y)

    tree, node_ids = _strtree_cache if _strtree_cache is not None else _build_strtree(graph)

    # Query nearest — STRtree.nearest returns the index into the geometry list.
    nearest_idx = tree.nearest(entrance_pt)

    # Candidate nearest node
    best_node_id = node_ids[nearest_idx]
    best_data = graph.nodes[best_node_id]
    best_dist = round(entrance_pt.distance(Point(best_data["x"], best_data["y"])), _COORD_PRECISION)

    # Tie-breaking: find all nodes within millimetre precision of best_dist,
    # then choose the smallest node_id string.
    candidate_indices = tree.query(entrance_pt.buffer(best_dist + 10 ** -_COORD_PRECISION))
    tied_node_ids = []
    for idx in candidate_indices:
        nid = node_ids[int(idx)]
        ndata = graph.nodes[nid]
        d = round(entrance_pt.distance(Point(ndata["x"], ndata["y"])), _COORD_PRECISION)
        if d == best_dist:
            tied_node_ids.append(nid)

    # Deterministic winner: lexicographic minimum node_id
    winner_id = min(tied_node_ids) if tied_node_ids else best_node_id
    winner_data = graph.nodes[winner_id]
    snap_dist = best_dist

    # Apply thresholds
    if snap_dist >= reject_m:
        status = "rejected"
        message = (
            f"Snap distance {snap_dist:.1f} m ≥ reject threshold {reject_m:.0f} m. "
            "Entrance coordinate may be outside the road network extent."
        )
    elif snap_dist >= warn_m:
        status = "warned"
        message = (
            f"Snap distance {snap_dist:.1f} m ≥ warn threshold {warn_m:.0f} m. "
            "Review entrance coordinate accuracy."
        )
    else:
        status = "snapped"
        message = ""

    return SnapResult(
        shelter_id=shelter_id,
        snapped_node_id=winner_id,
        snap_distance_m=snap_dist,
        entrance_utm_x=round(ent_x, _COORD_PRECISION),
        entrance_utm_y=round(ent_y, _COORD_PRECISION),
        snapped_node_x=round(float(winner_data["x"]), _COORD_PRECISION),
        snapped_node_y=round(float(winner_data["y"]), _COORD_PRECISION),
        status=status,
        message=message,
    )


def snap_all_shelters(
    records: list,  # list[ShelterRecord] — avoid circular import typing
    graph: nx.Graph,
    *,
    warn_m: float = DEFAULT_WARN_M,
    reject_m: float = DEFAULT_REJECT_M,
) -> list[SnapResult]:
    """Snap all eligible, routable shelter candidates to the road graph.

    Ineligible records (not ``r.is_eligible``) receive status "skipped".
    Eligible records without entrance coordinates receive "no_entrance".
    """
    strtree_cache = _build_strtree(graph) if graph.number_of_nodes() > 0 else None
    results: list[SnapResult] = []

    for r in records:
        if not r.is_eligible:
            results.append(
                SnapResult(
                    shelter_id=r.shelter_id,
                    snapped_node_id=None,
                    snap_distance_m=None,
                    entrance_utm_x=None,
                    entrance_utm_y=None,
                    snapped_node_x=None,
                    snapped_node_y=None,
                    status="skipped",
                    message="Record is ineligible (excluded tier or ineligible operational status).",
                )
            )
            continue

        if not r.is_routable:
            results.append(
                SnapResult(
                    shelter_id=r.shelter_id,
                    snapped_node_id=None,
                    snap_distance_m=None,
                    entrance_utm_x=None,
                    entrance_utm_y=None,
                    snapped_node_x=None,
                    snapped_node_y=None,
                    status="no_entrance",
                    message="Entrance coordinates are missing; cannot snap to graph.",
                )
            )
            continue

        result = snap_entrance_to_graph(
            shelter_id=r.shelter_id,
            entrance_lat=r.entrance_latitude,
            entrance_lon=r.entrance_longitude,
            graph=graph,
            warn_m=warn_m,
            reject_m=reject_m,
            _strtree_cache=strtree_cache,
        )
        results.append(result)

    return results
