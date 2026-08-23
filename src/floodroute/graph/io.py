"""Stage 4 graph serialization and deserialization.

Formats
-------
GraphML
    networkx built-in; XML-based open standard; includes all scalar node/edge
    attributes.  Geometry objects are excluded (not GraphML-serializable).
    Load with ``nx.read_graphml(path)``.

GeoPackage (GPKG)
    Two layers per municipality:
    - ``nodes``: Point geometries with node attributes (x, y, on_boundary, degree).
    - ``edges``: LineString geometries with all edge attributes.
    Requires geopandas.

The GraphML file is the canonical topology record.  The GeoPackage files
add spatial context for visualization and spatial queries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

logger = logging.getLogger("floodroute.graph.io")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_graphml(G: nx.MultiDiGraph, path: Path) -> None:
    """Serialise *G* to GraphML at *path*.

    Geometry attributes on edges are excluded (not GraphML-serializable).
    All other scalar attributes (str, int, float, bool) are written as-is;
    None values become the empty string to satisfy GraphML schema.
    """
    # Build a copy without geometry to avoid serialization errors
    H = nx.MultiDiGraph()
    H.graph.update(G.graph)

    for n, d in G.nodes(data=True):
        clean = {k: ("" if v is None else v) for k, v in d.items() if k != "geometry"}
        H.add_node(n, **clean)

    for u, v, k, d in G.edges(keys=True, data=True):
        clean = {k2: ("" if v2 is None else v2) for k2, v2 in d.items() if k2 != "geometry"}
        H.add_edge(u, v, key=k, **clean)

    nx.write_graphml(H, str(path))
    logger.info(
        "Wrote GraphML: %s (%d nodes, %d edges)", path, H.number_of_nodes(), H.number_of_edges()
    )


def write_nodes_gpkg(G: nx.MultiDiGraph, path: Path, *, crs: str = "EPSG:32651") -> None:
    """Write graph nodes as a GeoPackage (Point layer)."""
    records = []
    for nid, d in G.nodes(data=True):
        x = d.get("x")
        y = d.get("y")
        if x is None or y is None:
            continue
        records.append(
            {
                "node_id": nid,
                "x": float(x),
                "y": float(y),
                "on_boundary": bool(d.get("on_boundary", False)),
                "degree": G.degree(nid),
                "in_degree": G.in_degree(nid),
                "out_degree": G.out_degree(nid),
                "geometry": Point(x, y),
            }
        )
    if records:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf = gdf.sort_values("node_id", ignore_index=True)
    else:
        gdf = gpd.GeoDataFrame(
            columns=[
                "node_id",
                "x",
                "y",
                "on_boundary",
                "degree",
                "in_degree",
                "out_degree",
                "geometry",
            ],
            geometry="geometry",
            crs=crs,
        )
    gdf.to_file(path, driver="GPKG", layer="nodes")
    logger.info("Wrote nodes GeoPackage: %s (%d nodes)", path, len(gdf))


def write_edges_gpkg(G: nx.MultiDiGraph, path: Path, *, crs: str = "EPSG:32651") -> None:
    """Write graph edges as a GeoPackage (LineString layer)."""
    records = []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry")
        if geom is None:
            # Reconstruct geometry from node coordinates
            u_x, u_y = G.nodes[u].get("x"), G.nodes[u].get("y")
            v_x, v_y = G.nodes[v].get("x"), G.nodes[v].get("y")
            if None not in (u_x, u_y, v_x, v_y):
                geom = LineString([(u_x, u_y), (v_x, v_y)])
        rec: dict[str, Any] = {
            "u_node": u,
            "v_node": v,
            "edge_key": k,
            "osm_id": d.get("osm_id"),
            "edge_seq": d.get("edge_seq"),
            "highway": d.get("highway"),
            "name": d.get("name"),
            "ref": d.get("ref"),
            "oneway": d.get("oneway"),
            "lanes": d.get("lanes"),
            "maxspeed": d.get("maxspeed"),
            "surface": d.get("surface"),
            "access": d.get("access"),
            "service": d.get("service"),
            "bridge": d.get("bridge"),
            "tunnel": d.get("tunnel"),
            "length_m": d.get("length_m"),
            "geometry": geom,
        }
        records.append(rec)

    if records:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf = gdf.sort_values(["osm_id", "edge_seq"], ignore_index=True)
    else:
        cols = [
            "u_node",
            "v_node",
            "edge_key",
            "osm_id",
            "edge_seq",
            "highway",
            "name",
            "ref",
            "oneway",
            "lanes",
            "maxspeed",
            "surface",
            "access",
            "service",
            "bridge",
            "tunnel",
            "length_m",
            "geometry",
        ]
        gdf = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)
    gdf.to_file(path, driver="GPKG", layer="edges")
    logger.info("Wrote edges GeoPackage: %s (%d edges)", path, len(gdf))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_graphml(path: Path) -> nx.MultiDiGraph:
    """Load a graph from GraphML. Returns a MultiDiGraph."""
    G = nx.read_graphml(str(path), node_type=int)
    # Ensure it's a MultiDiGraph
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    logger.info(
        "Loaded GraphML: %s (%d nodes, %d edges)", path, G.number_of_nodes(), G.number_of_edges()
    )
    return G
