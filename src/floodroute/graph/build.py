"""Stage 4 road-graph construction.

Algorithm
---------
1.  Sort input ways by osm_id (ascending) for deterministic processing.
2.  For each way, the first and last coordinate of its LineString geometry
    become graph nodes.  All way endpoints always become nodes regardless of
    bridge/tunnel tags — the endpoint is the physical location where the
    road starts or ends on the surface.
3.  Interior geometric crossings between pairs of ways are tested via
    shapely's STRtree.  A crossing produces a shared node ONLY when neither
    way is grade-separated at that location (i.e., neither has a recognised
    bridge or tunnel tag).  Grade separation is detected from the way-level
    ``bridge`` and ``tunnel`` attributes.
4.  All split-points (endpoints + at-grade interior crossings) for each way
    are sorted by distance along the way geometry.  The way is split into
    segments between consecutive split-points using
    ``shapely.ops.substring``.
5.  Each segment becomes one or two directed edges in a ``MultiDiGraph``
    (one for oneway=yes/-1, two for bidirectional).
6.  Node IDs are assigned sequentially in the order coordinates are first
    encountered when processing ways in osm_id order.  This guarantees
    deterministic IDs given the same input.

Oneway interpretation
---------------------
- ``yes``, ``1``, ``true``   → single directed edge, start → end
- ``-1``, ``reverse``        → single directed edge, end → start
- ``no``, ``0``, ``false``   → two directed edges (explicit bidirectional)
- ``reversible``, ``alternating`` → two directed edges (conservative)
- absent / null / unrecognised → two directed edges (conservative default)

Grade-separation rules
----------------------
- A way is *elevated* if its ``bridge`` tag value is in
  BRIDGE_GRADE_SEP = {yes, cantilever, viaduct, aqueduct, movable, trestle}.
- A way is *underground* if its ``tunnel`` tag value is in
  TUNNEL_GRADE_SEP = {yes, building_passage, culvert}.
- Two ways that geometrically cross share an interior node ONLY when neither
  is elevated or underground.  Endpoints are always shared.
- ``layer`` tags are not present in the Stage 3 extraction schema and are not
  used.

Node boundary classification
-----------------------------
If a municipality polygon is provided, each node is marked
``on_boundary=True`` when its coordinates fall OUTSIDE the municipality
polygon (i.e., in the 500 m buffer zone appended during Stage 3 extraction).

Coordinate snapping and T-junction design
------------------------------------------
Nodes are identified by coordinates rounded to 1 mm (COORD_PRECISION=3 decimal
places in UTM metres).  A diagnostic audit of the Stage 3 outputs found:

- Zero endpoint pairs in the >0–0.5 m distance band.
- All coincident endpoints are exactly 0 m apart (same reprojected coordinate).
- 0.5 m snapping and 1 mm snapping produce identical graph topology.

On this evidence, the 1 mm key is retained and 0.5 m snapping is NOT added.

T-junctions (where one way's endpoint meets another way's interior) are handled
by adding the endpoint coordinate to the interior split set of the way that
continues through.  The endpoint node is then shared via `_get_or_create`, which
returns the existing node ID because the endpoint was registered earlier in the
endpoint loop.  No second coincident node is created.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import networkx as nx
import shapely
import shapely.ops
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from floodroute.graph.config import (
    BRIDGE_GRADE_SEP,
    COORD_PRECISION,
    ONEWAY_FORWARD,
    ONEWAY_REVERSE,
    TUNNEL_GRADE_SEP,
)

logger = logging.getLogger("floodroute.graph.build")

# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class GraphBuildError(RuntimeError):
    """Raised when graph construction fails for a non-recoverable reason."""


class OutputExistsError(FileExistsError):
    """Raised when an output file already exists and force=False."""


# ---------------------------------------------------------------------------
# Internal helpers (module-level for testability)
# ---------------------------------------------------------------------------


def _snap(x: float, y: float, precision: int = COORD_PRECISION) -> tuple[float, float]:
    """Round coordinates to *precision* decimal places for node deduplication."""
    return (round(x, precision), round(y, precision))


def _is_grade_separated(
    bridge_val: Any,
    tunnel_val: Any,
    bridge_set: frozenset[str] = BRIDGE_GRADE_SEP,
    tunnel_set: frozenset[str] = TUNNEL_GRADE_SEP,
) -> bool:
    """Return True if a way is elevated (bridge) or underground (tunnel)."""
    if bridge_val is not None:
        import pandas as pd

        if not pd.isna(bridge_val) and str(bridge_val).lower() in bridge_set:
            return True
    if tunnel_val is not None:
        import pandas as pd

        if not pd.isna(tunnel_val) and str(tunnel_val).lower() in tunnel_set:
            return True
    return False


def _oneway_direction(
    oneway_val: Any,
    forward_set: frozenset[str] = ONEWAY_FORWARD,
    reverse_set: frozenset[str] = ONEWAY_REVERSE,
) -> str:
    """Return 'forward', 'reverse', or 'both' from an OSM oneway tag value.

    Conservative: any unrecognised or absent value returns 'both'.
    """
    if oneway_val is None:
        return "both"
    import pandas as pd

    if pd.isna(oneway_val):
        return "both"
    v = str(oneway_val).strip().lower()
    if v in forward_set:
        return "forward"
    if v in reverse_set:
        return "reverse"
    return "both"


def _extract_points(inter: BaseGeometry) -> list[Point]:
    """Flatten a geometry intersection result to a list of Points."""
    if inter.is_empty:
        return []
    if inter.geom_type == "Point":
        return [inter]
    if inter.geom_type == "MultiPoint":
        return list(inter.geoms)
    if inter.geom_type == "GeometryCollection":
        pts = []
        for g in inter.geoms:
            pts.extend(_extract_points(g))
        return pts
    # Lines / polygons (collinear overlap): extract endpoints only
    if hasattr(inter, "boundary"):
        bnd = inter.boundary
        if bnd is not None and not bnd.is_empty:
            return _extract_points(bnd)
    return []


def _is_endpoint(pt: Point, geom: LineString, precision: int = COORD_PRECISION) -> bool:
    """Return True if *pt* (snapped) coincides with either endpoint of *geom*."""
    coords = list(geom.coords)
    start = _snap(*coords[0], precision=precision)
    end = _snap(*coords[-1], precision=precision)
    snapped = _snap(pt.x, pt.y, precision=precision)
    return snapped in (start, end)


def _build_edge_attrs(
    row: Any,
    seg_geom: LineString,
    seg_idx: int,
) -> dict[str, Any]:
    """Build the attribute dict for a single graph edge segment.

    Attributes that are NaN/None in the source are stored as None in the graph.
    """
    import pandas as pd

    def _clean(v: Any) -> Any:
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    return {
        "osm_id": int(row["osm_id"]),
        "edge_seq": seg_idx,
        "highway": _clean(row.get("highway")),
        "name": _clean(row.get("name")),
        "ref": _clean(row.get("ref")),
        "oneway": _clean(row.get("oneway")),
        "lanes": _clean(row.get("lanes")),
        "maxspeed": _clean(row.get("maxspeed")),
        "surface": _clean(row.get("surface")),
        "access": _clean(row.get("access")),
        "service": _clean(row.get("service")),
        "bridge": _clean(row.get("bridge")),
        "tunnel": _clean(row.get("tunnel")),
        "length_m": float(seg_geom.length),
        "geometry": seg_geom,
    }


# ---------------------------------------------------------------------------
# Main graph construction
# ---------------------------------------------------------------------------


class GraphBuildStats:
    """Counters accumulated during graph construction."""

    def __init__(self) -> None:
        self.input_road_count: int = 0
        self.self_loops_skipped: int = 0
        self.degenerate_segments_skipped: int = 0
        self.interior_nodes_added: int = 0
        self.grade_sep_crossings_skipped: int = 0
        self.t_junction_splits_added: int = 0
        self.total_directed_edge_length_m: float = 0.0
        self.total_physical_segment_length_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_road_count": self.input_road_count,
            "self_loops_skipped": self.self_loops_skipped,
            "degenerate_segments_skipped": self.degenerate_segments_skipped,
            "interior_nodes_added": self.interior_nodes_added,
            "grade_sep_crossings_skipped": self.grade_sep_crossings_skipped,
            "t_junction_splits_added": self.t_junction_splits_added,
            "total_directed_edge_length_m": self.total_directed_edge_length_m,
            "total_physical_segment_length_m": self.total_physical_segment_length_m,
        }


def build_road_graph(
    roads_gdf: gpd.GeoDataFrame,
    *,
    municipality_polygon: BaseGeometry | None = None,
    coord_precision: int = COORD_PRECISION,
    bridge_grade_sep: frozenset[str] = BRIDGE_GRADE_SEP,
    tunnel_grade_sep: frozenset[str] = TUNNEL_GRADE_SEP,
    oneway_forward: frozenset[str] = ONEWAY_FORWARD,
    oneway_reverse: frozenset[str] = ONEWAY_REVERSE,
) -> tuple[nx.MultiDiGraph, GraphBuildStats]:
    """Build a directed road graph from a GeoDataFrame of road LineStrings.

    Parameters
    ----------
    roads_gdf:
        GeoDataFrame of road LineStrings in a projected CRS (EPSG:32651).
        Must have columns: osm_id, highway, name, ref, oneway, lanes,
        maxspeed, surface, access, service, bridge, tunnel, geometry.
    municipality_polygon:
        Shapely polygon of the municipality boundary in the same CRS.
        Used to classify boundary nodes (nodes in the 500 m buffer zone).
        If None, on_boundary is set to False for all nodes.
    coord_precision:
        Decimal places for coordinate rounding when deduplicating nodes.
    bridge_grade_sep, tunnel_grade_sep:
        Sets of tag values indicating grade separation.
    oneway_forward, oneway_reverse:
        Sets of oneway tag values for forward-only and reverse-only edges.

    Returns
    -------
    (G, stats):
        G is a networkx MultiDiGraph with node attributes x, y, on_boundary
        and edge attributes as documented in _build_edge_attrs.
        stats is a GraphBuildStats instance.
    """
    stats = GraphBuildStats()

    if len(roads_gdf) == 0:
        G: nx.MultiDiGraph = nx.MultiDiGraph()
        G.graph["crs"] = str(roads_gdf.crs) if roads_gdf.crs is not None else "EPSG:32651"
        return G, stats

    # ------------------------------------------------------------------
    # Step 1: Sort by osm_id for deterministic processing
    # ------------------------------------------------------------------
    roads = roads_gdf.sort_values("osm_id", ignore_index=True)
    stats.input_road_count = len(roads)
    crs_str = str(roads.crs) if roads.crs is not None else "EPSG:32651"

    # ------------------------------------------------------------------
    # Step 2: Find at-grade interior intersections using STRtree
    # ------------------------------------------------------------------
    geoms = roads.geometry.values
    tree = shapely.STRtree(geoms)

    # interior_splits[i] = set of snapped (x,y) interior nodes for way i
    interior_splits: list[set[tuple[float, float]]] = [set() for _ in range(len(roads))]

    for i in range(len(roads)):
        geom_i = geoms[i]
        row_i = roads.iloc[i]
        grade_i = _is_grade_separated(
            row_i.get("bridge"), row_i.get("tunnel"), bridge_grade_sep, tunnel_grade_sep
        )

        candidate_js = tree.query(geom_i, predicate="intersects")
        for j in candidate_js:
            if j <= i:
                continue  # process each pair once
            geom_j = geoms[j]
            row_j = roads.iloc[j]
            grade_j = _is_grade_separated(
                row_j.get("bridge"), row_j.get("tunnel"), bridge_grade_sep, tunnel_grade_sep
            )

            inter = geom_i.intersection(geom_j)
            if inter.is_empty:
                continue

            pts = _extract_points(inter)
            for pt in pts:
                snapped = _snap(pt.x, pt.y, coord_precision)
                ep_i = _is_endpoint(pt, geom_i, coord_precision)
                ep_j = _is_endpoint(pt, geom_j, coord_precision)

                if ep_i and ep_j:
                    # Shared endpoint: both ways register this coord in the
                    # endpoint loop below; _get_or_create unifies them to one node.
                    pass
                elif ep_i and not ep_j:
                    # Way i's endpoint touches way j's interior → split way j.
                    # Grade separation is not applied here: a bridge endpoint
                    # always rests at grade (the abutment), so it physically
                    # connects to the road below at that exact point.
                    interior_splits[j].add(snapped)
                    stats.t_junction_splits_added += 1
                elif ep_j and not ep_i:
                    # Way j's endpoint touches way i's interior → split way i.
                    interior_splits[i].add(snapped)
                    stats.t_junction_splits_added += 1
                else:
                    # Interior-to-interior crossing: enforce grade separation.
                    if grade_i or grade_j:
                        stats.grade_sep_crossings_skipped += 1
                        continue
                    interior_splits[i].add(snapped)
                    interior_splits[j].add(snapped)

    stats.interior_nodes_added = sum(len(s) for s in interior_splits)

    # ------------------------------------------------------------------
    # Step 3: Assign node IDs deterministically
    # ------------------------------------------------------------------
    coord_to_nid: dict[tuple[float, float], int] = {}
    next_nid = 0

    def _get_or_create(coord: tuple[float, float]) -> int:
        nonlocal next_nid
        if coord not in coord_to_nid:
            coord_to_nid[coord] = next_nid
            next_nid += 1
        return coord_to_nid[coord]

    # Register endpoints first (in osm_id order), then interior nodes
    for i in range(len(roads)):
        coords = list(geoms[i].coords)
        _get_or_create(_snap(*coords[0], coord_precision))
        _get_or_create(_snap(*coords[-1], coord_precision))

    for i in range(len(roads)):
        for coord in sorted(interior_splits[i]):  # sorted for determinism
            _get_or_create(coord)

    # ------------------------------------------------------------------
    # Step 4: Build graph
    # ------------------------------------------------------------------
    G = nx.MultiDiGraph()
    G.graph["crs"] = crs_str
    G.graph["coord_precision"] = coord_precision

    # Add all nodes
    for coord, nid in coord_to_nid.items():
        on_boundary = False
        if municipality_polygon is not None:
            pt = Point(coord)
            on_boundary = not municipality_polygon.contains(pt)
        G.add_node(nid, x=coord[0], y=coord[1], on_boundary=on_boundary)

    # ------------------------------------------------------------------
    # Step 5: Build edges (one way at a time)
    # ------------------------------------------------------------------
    # Track physical segments for length accounting.
    # physical_seen maps (osm_id, edge_seq) to avoid double-counting
    # bidirectional edges.
    physical_seen: set[tuple[int, int]] = set()

    for i in range(len(roads)):
        row = roads.iloc[i]
        geom = geoms[i]
        coords = list(geom.coords)

        # Collect all split points for this way
        split_set: set[tuple[float, float]] = set()
        split_set.add(_snap(*coords[0], coord_precision))
        split_set.add(_snap(*coords[-1], coord_precision))
        split_set.update(interior_splits[i])

        # Sort by distance along geometry
        split_list: list[tuple[float, tuple[float, float]]] = []
        for coord in split_set:
            pt = Point(coord)
            dist = geom.project(pt)
            split_list.append((dist, coord))
        split_list.sort(key=lambda x: x[0])

        direction = _oneway_direction(row.get("oneway"), oneway_forward, oneway_reverse)

        # Create one edge per consecutive pair of split points
        for seg_idx, ((d1, c1), (d2, c2)) in enumerate(
            zip(split_list, split_list[1:], strict=False)
        ):
            if d1 == d2:
                # Zero-length segment (duplicate split points) — skip
                stats.degenerate_segments_skipped += 1
                continue

            u_nid = coord_to_nid[c1]
            v_nid = coord_to_nid[c2]

            if u_nid == v_nid:
                stats.self_loops_skipped += 1
                continue

            try:
                seg_geom = shapely.ops.substring(geom, d1, d2)
            except Exception:
                stats.degenerate_segments_skipped += 1
                continue

            if seg_geom is None or seg_geom.is_empty or seg_geom.length == 0:
                stats.degenerate_segments_skipped += 1
                continue

            edge_attrs = _build_edge_attrs(row, seg_geom, seg_idx)

            phys_key = (edge_attrs["osm_id"], edge_attrs["edge_seq"])
            phys_len = edge_attrs["length_m"]

            if direction == "forward":
                G.add_edge(u_nid, v_nid, **edge_attrs)
                stats.total_directed_edge_length_m += phys_len
                if phys_key not in physical_seen:
                    physical_seen.add(phys_key)
                    stats.total_physical_segment_length_m += phys_len
            elif direction == "reverse":
                G.add_edge(v_nid, u_nid, **edge_attrs)
                stats.total_directed_edge_length_m += phys_len
                if phys_key not in physical_seen:
                    physical_seen.add(phys_key)
                    stats.total_physical_segment_length_m += phys_len
            else:  # "both"
                G.add_edge(u_nid, v_nid, **edge_attrs)
                G.add_edge(v_nid, u_nid, **{**edge_attrs})
                stats.total_directed_edge_length_m += phys_len * 2
                if phys_key not in physical_seen:
                    physical_seen.add(phys_key)
                    stats.total_physical_segment_length_m += phys_len

    logger.info(
        "Graph built: %d nodes, %d edges (%d input roads, %d interior nodes, "
        "%d grade-sep crossings skipped, %d self-loops skipped)",
        G.number_of_nodes(),
        G.number_of_edges(),
        stats.input_road_count,
        stats.interior_nodes_added,
        stats.grade_sep_crossings_skipped,
        stats.self_loops_skipped,
    )

    return G, stats
