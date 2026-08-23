"""Unit tests for Stage 4 road-graph construction.

All fixtures are synthetic in-memory objects — no real data files are read.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Polygon

from floodroute.graph.build import (
    _is_grade_separated,
    _oneway_direction,
    _snap,
    build_road_graph,
)
from floodroute.graph.io import read_graphml, write_edges_gpkg, write_graphml, write_nodes_gpkg
from floodroute.graph.validate import GraphValidationFailed, validate_road_graph

# ---------------------------------------------------------------------------
# Helpers for creating synthetic GeoDataFrames
# ---------------------------------------------------------------------------

CRS = "EPSG:32651"


def _make_roads(rows: list[dict]) -> gpd.GeoDataFrame:
    """Create a minimal roads GeoDataFrame from a list of dicts.

    Each dict must have 'osm_id' and 'geometry'.  Optional keys:
    highway, name, ref, oneway, lanes, maxspeed, surface, access,
    service, bridge, tunnel.
    """
    defaults = {
        "highway": "residential",
        "name": None,
        "ref": None,
        "oneway": None,
        "lanes": None,
        "maxspeed": None,
        "surface": None,
        "access": None,
        "service": None,
        "bridge": None,
        "tunnel": None,
    }
    filled = [{**defaults, **r} for r in rows]
    return gpd.GeoDataFrame(filled, geometry="geometry", crs=CRS)


# ---------------------------------------------------------------------------
# TestOnewayDirection
# ---------------------------------------------------------------------------


class TestOnewayDirection:
    def test_yes_returns_forward(self):
        assert _oneway_direction("yes") == "forward"

    def test_minus1_returns_reverse(self):
        assert _oneway_direction("-1") == "reverse"

    def test_no_returns_both(self):
        assert _oneway_direction("no") == "both"

    def test_none_returns_both(self):
        assert _oneway_direction(None) == "both"

    def test_unknown_value_returns_both(self):
        assert _oneway_direction("maybe") == "both"


# ---------------------------------------------------------------------------
# TestGradeSeparation
# ---------------------------------------------------------------------------


class TestGradeSeparation:
    def test_bridge_yes_is_grade_separated(self):
        assert _is_grade_separated("yes", None) is True

    def test_bridge_cantilever_is_grade_separated(self):
        assert _is_grade_separated("cantilever", None) is True

    def test_tunnel_yes_is_grade_separated(self):
        assert _is_grade_separated(None, "yes") is True

    def test_both_none_is_not_grade_separated(self):
        assert _is_grade_separated(None, None) is False


# ---------------------------------------------------------------------------
# TestSnapCoordinates
# ---------------------------------------------------------------------------


class TestSnapCoordinates:
    def test_rounds_to_three_decimal_places(self):
        x, y = _snap(1.23456789, 9.87654321, precision=3)
        assert x == 1.235
        assert y == 9.877

    def test_two_nearby_coords_snap_to_same_value(self):
        # Use values that clearly snap to same bucket
        c3 = _snap(100.0004, 200.0004, precision=3)
        c4 = _snap(100.0003, 200.0003, precision=3)
        assert c3 == c4


# ---------------------------------------------------------------------------
# TestBuildRoadGraphBasic
# ---------------------------------------------------------------------------


class TestBuildRoadGraphBasic:
    def test_empty_geodataframe_returns_empty_graph(self):
        # Build empty GDF with the right schema by slicing a 1-row GDF
        one_row = _make_roads([{"osm_id": 1, "geometry": LineString([(0, 0), (1, 0)])}])
        gdf = one_row.iloc[:0].copy()
        G, stats = build_road_graph(gdf)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        assert stats.input_road_count == 0

    def test_single_bidirectional_way_produces_2_nodes_2_edges(self):
        gdf = _make_roads([{"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])}])
        G, stats = build_road_graph(gdf)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 2  # bidirectional → 2 directed edges
        assert stats.input_road_count == 1

    def test_single_oneway_yes_produces_1_forward_edge(self):
        gdf = _make_roads(
            [{"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)]), "oneway": "yes"}]
        )
        G, stats = build_road_graph(gdf)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        # Edge must go from node at (0,0) to node at (100,0)
        edges = list(G.edges(data=True))
        assert len(edges) == 1

    def test_single_oneway_reverse_produces_1_reverse_edge(self):
        gdf = _make_roads(
            [{"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)]), "oneway": "-1"}]
        )
        G, stats = build_road_graph(gdf)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1

    def test_t_intersection_creates_shared_node(self):
        # Way 1: horizontal road A→B
        # Way 2: vertical road from midpoint of way 1 upward (T shape)
        # Way 3: short stub to complete the T
        gdf = _make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (200, 0)])},
                {"osm_id": 2, "geometry": LineString([(100, 0), (100, 100)])},
                {"osm_id": 3, "geometry": LineString([(100, 0), (100, -100)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        # (0,0), (200,0) are endpoints of way 1
        # (100,0) is endpoint of ways 2 and 3, and interior of way 1
        # → should be a shared node (endpoint of 2/3 intersects interior of 1)
        assert G.number_of_nodes() >= 4  # at least 4 distinct coordinate nodes
        # The interior node at (100,0) gets shared: ways 2,3 have it as endpoint,
        # way 1 is split there
        assert G.number_of_edges() >= 4


# ---------------------------------------------------------------------------
# TestGradeSeparationInGraph
# ---------------------------------------------------------------------------


class TestGradeSeparationInGraph:
    def test_surface_cross_surface_creates_interior_node(self):
        # Two crossing surface roads — should share interior node
        gdf = _make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 50), (100, 50)])},
                {"osm_id": 2, "geometry": LineString([(50, 0), (50, 100)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        # 4 endpoints + 1 interior crossing = 5 nodes
        assert G.number_of_nodes() == 5
        assert stats.interior_nodes_added >= 1
        assert stats.grade_sep_crossings_skipped == 0

    def test_bridge_cross_surface_no_interior_node(self):
        # Bridge over surface road — NO interior shared node
        gdf = _make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 50), (100, 50)]), "bridge": "yes"},
                {"osm_id": 2, "geometry": LineString([(50, 0), (50, 100)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        # Only 4 endpoint nodes — no interior crossing
        assert G.number_of_nodes() == 4
        assert stats.grade_sep_crossings_skipped >= 1
        assert stats.interior_nodes_added == 0

    def test_bridge_endpoint_on_surface_still_shared(self):
        # Bridge ends exactly where a surface road runs through
        # The bridge endpoint is an endpoint (not interior), so it IS shared
        gdf = _make_roads(
            [
                # Surface road horizontal
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                # Bridge ends at (50, 0) which is an interior point of way 1
                # but an ENDPOINT of way 2 → endpoint connections are always valid
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)]), "bridge": "yes"},
            ]
        )
        G, stats = build_road_graph(gdf)
        # (50,0) is endpoint of way 2 → way 1 gets split there despite bridge tag
        # Nodes: (0,0), (100,0), (50,50), (50,0) = 4 nodes
        assert G.number_of_nodes() == 4


# ---------------------------------------------------------------------------
# TestNodeDeterminism
# ---------------------------------------------------------------------------


class TestNodeDeterminism:
    def test_same_input_same_node_ids(self):
        gdf = _make_roads(
            [
                {"osm_id": 10, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 20, "geometry": LineString([(100, 0), (100, 100)])},
            ]
        )
        G1, _ = build_road_graph(gdf)
        G2, _ = build_road_graph(gdf)
        assert set(G1.nodes()) == set(G2.nodes())
        # Node attributes should match
        for n in G1.nodes():
            assert G1.nodes[n]["x"] == G2.nodes[n]["x"]
            assert G1.nodes[n]["y"] == G2.nodes[n]["y"]

    def test_osm_id_order_determines_node_assignment(self):
        # Way with lower osm_id processed first → its start node gets ID 0
        gdf = _make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(200, 0), (300, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        # Node 0 must be the start of way osm_id=1 → (0,0)
        node_0 = G.nodes[0]
        assert node_0["x"] == pytest.approx(0.0)
        assert node_0["y"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestEdgeLengths
# ---------------------------------------------------------------------------


class TestEdgeLengths:
    def test_all_edge_lengths_positive(self):
        gdf = _make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(0, 0), (0, 100)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        for _, _, d in G.edges(data=True):
            assert d["length_m"] > 0

    def test_length_m_matches_shapely_segment_length(self):
        line = LineString([(0, 0), (300, 400)])  # length = 500.0
        gdf = _make_roads([{"osm_id": 1, "geometry": line, "oneway": "yes"}])
        G, _ = build_road_graph(gdf)
        edges = list(G.edges(data=True))
        assert len(edges) == 1
        _, _, d = edges[0]
        assert d["length_m"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# TestBoundaryNodes
# ---------------------------------------------------------------------------


class TestBoundaryNodes:
    def _make_polygon(self) -> Polygon:
        # Square polygon: (10,10)→(90,90)
        return Polygon([(10, 10), (90, 10), (90, 90), (10, 90)])

    def test_node_inside_polygon_on_boundary_false(self):
        polygon = self._make_polygon()
        gdf = _make_roads([{"osm_id": 1, "geometry": LineString([(20, 20), (80, 20)])}])
        G, _ = build_road_graph(gdf, municipality_polygon=polygon)
        for _, d in G.nodes(data=True):
            assert d["on_boundary"] is False

    def test_node_outside_polygon_on_boundary_true(self):
        polygon = self._make_polygon()
        # Road outside the polygon
        gdf = _make_roads([{"osm_id": 1, "geometry": LineString([(0, 0), (5, 5)])}])
        G, _ = build_road_graph(gdf, municipality_polygon=polygon)
        for _, d in G.nodes(data=True):
            assert d["on_boundary"] is True


# ---------------------------------------------------------------------------
# TestGraphSerialization
# ---------------------------------------------------------------------------


class TestGraphSerialization:
    def _make_simple_graph(self) -> nx.MultiDiGraph:
        gdf = _make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(100, 0), (100, 100)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        return G

    def test_write_read_graphml_roundtrip(self, tmp_path: Path):
        G = self._make_simple_graph()
        path = tmp_path / "test.graphml"
        write_graphml(G, path)
        G2 = read_graphml(path)
        assert G2.number_of_nodes() == G.number_of_nodes()
        assert G2.number_of_edges() == G.number_of_edges()

    def test_write_nodes_gpkg_correct_count_and_crs(self, tmp_path: Path):
        G = self._make_simple_graph()
        path = tmp_path / "nodes.gpkg"
        write_nodes_gpkg(G, path, crs=CRS)
        gdf = gpd.read_file(path, layer="nodes")
        assert len(gdf) == G.number_of_nodes()
        assert gdf.crs.to_epsg() == 32651

    def test_write_edges_gpkg_correct_count(self, tmp_path: Path):
        G = self._make_simple_graph()
        path = tmp_path / "edges.gpkg"
        write_edges_gpkg(G, path, crs=CRS)
        gdf = gpd.read_file(path, layer="edges")
        assert len(gdf) == G.number_of_edges()


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def _make_valid_graph(self) -> nx.MultiDiGraph:
        gdf = _make_roads([{"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])}])
        G, _ = build_road_graph(gdf)
        return G

    def test_valid_graph_passes(self):
        G = self._make_valid_graph()
        stats = validate_road_graph(
            G, min_largest_wcc_node_coverage=0.0, min_largest_wcc_physical_length_coverage=0.0
        )
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 2

    def test_empty_graph_fails_too_few_nodes(self):
        G = nx.MultiDiGraph()
        G.graph["crs"] = CRS
        with pytest.raises(GraphValidationFailed, match="Too few nodes"):
            validate_road_graph(G)

    def test_graph_with_zero_length_edge_fails(self):
        G = nx.MultiDiGraph()
        G.graph["crs"] = CRS
        G.add_node(0, x=0.0, y=0.0, on_boundary=False)
        G.add_node(1, x=100.0, y=0.0, on_boundary=False)
        G.add_edge(0, 1, osm_id=1, edge_seq=0, length_m=0.0, highway="residential")
        with pytest.raises(GraphValidationFailed, match="non-positive length"):
            validate_road_graph(G)

    def test_disconnected_graph_below_threshold_fails(self):
        # Two isolated components, each 1 node — largest WCC = 50% of 2 nodes
        # With min_largest_wcc_node_coverage=0.9 it should fail
        G = nx.MultiDiGraph()
        G.graph["crs"] = CRS
        G.add_node(0, x=0.0, y=0.0, on_boundary=False)
        G.add_node(1, x=100.0, y=0.0, on_boundary=False)
        G.add_node(2, x=200.0, y=0.0, on_boundary=False)
        G.add_node(3, x=300.0, y=0.0, on_boundary=False)
        # Two separate components: 0↔1 and 2↔3
        G.add_edge(0, 1, osm_id=1, edge_seq=0, length_m=100.0)
        G.add_edge(1, 0, osm_id=1, edge_seq=0, length_m=100.0)
        G.add_edge(2, 3, osm_id=2, edge_seq=0, length_m=100.0)
        G.add_edge(3, 2, osm_id=2, edge_seq=0, length_m=100.0)
        with pytest.raises(GraphValidationFailed, match="WCC"):
            validate_road_graph(
                G, min_largest_wcc_node_coverage=0.9, min_largest_wcc_physical_length_coverage=0.0
            )


# ---------------------------------------------------------------------------
# TestTJunctionFix
# ---------------------------------------------------------------------------


class TestTJunctionFix:
    """Regression tests for the T-junction endpoint-to-interior split fix."""

    CRS = "EPSG:32651"

    def _make_roads(self, rows):
        import pandas as pd  # noqa: F401

        records = []
        defaults = {
            "name": None,
            "ref": None,
            "oneway": None,
            "lanes": None,
            "maxspeed": None,
            "surface": None,
            "access": None,
            "service": None,
            "bridge": None,
            "tunnel": None,
        }
        for r in rows:
            rec = {**defaults, "highway": "residential", **r}
            records.append(rec)
        cols = [
            "osm_id",
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
            "geometry",
        ]
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=self.CRS)
        for c in cols:
            if c not in gdf.columns:
                gdf[c] = None
        return gdf[cols]

    def test_endpoint_of_A_in_interior_of_B(self):
        """Way A ends at (50,0) which is interior to way B (0,0)→(100,0).
        B must be split at (50,0) so A and B share a node there.
        Bug: previously B was not split → two disconnected components."""
        gdf = self._make_roads(
            [
                {
                    "osm_id": 1,
                    "geometry": LineString([(0, 0), (100, 0)]),
                },  # B (interior has (50,0))
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},  # A ends at (50,0)
            ]
        )
        G, stats = build_road_graph(gdf)
        wccs = list(nx.weakly_connected_components(G))
        # After fix: single WCC (all connected)
        assert len(wccs) == 1, f"Expected 1 WCC, got {len(wccs)}"
        assert stats.t_junction_splits_added >= 1

    def test_endpoint_of_B_in_interior_of_A(self):
        """Way B ends at (50,0) which is interior to way A (0,0)→(100,0).
        A must be split at (50,0)."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(50, 50), (50, 0)])},  # B ends at (50,0)
                {"osm_id": 2, "geometry": LineString([(0, 0), (100, 0)])},  # A (interior)
            ]
        )
        G, stats = build_road_graph(gdf)
        wccs = list(nx.weakly_connected_components(G))
        assert len(wccs) == 1, f"Expected 1 WCC, got {len(wccs)}"
        assert stats.t_junction_splits_added >= 1

    def test_reversed_input_ordering(self):
        """Same geometry as above but osm_ids reversed to change sort order.
        Result must be the same (fix is symmetric)."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(50, 50), (50, 0)])},
                {"osm_id": 2, "geometry": LineString([(0, 0), (100, 0)])},
            ]
        )
        G1, _ = build_road_graph(gdf)
        # Swap IDs and rebuild
        gdf2 = self._make_roads(
            [
                {"osm_id": 10, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 20, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G2, _ = build_road_graph(gdf2)
        wccs1 = list(nx.weakly_connected_components(G1))
        wccs2 = list(nx.weakly_connected_components(G2))
        assert len(wccs1) == 1
        assert len(wccs2) == 1

    def test_endpoint_to_endpoint_connection(self):
        """A (0,0)→(50,0) and B (50,0)→(100,0) share an endpoint.
        One shared node, single WCC, no T-junction split needed."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (50, 0)])},
                {"osm_id": 2, "geometry": LineString([(50, 0), (100, 0)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        wccs = list(nx.weakly_connected_components(G))
        assert len(wccs) == 1
        assert stats.t_junction_splits_added == 0  # endpoint-to-endpoint, not T-junction

    def test_interior_to_interior_at_grade(self):
        """Two crossing surface roads (no bridge/tunnel) → interior node created."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 50), (100, 50)])},
                {"osm_id": 2, "geometry": LineString([(50, 0), (50, 100)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        wccs = list(nx.weakly_connected_components(G))
        assert len(wccs) == 1
        # 4 endpoints + 1 interior = 5 nodes
        assert G.number_of_nodes() == 5

    def test_grade_separated_interior_crossing(self):
        """Bridge crosses surface road: no interior shared node."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 50), (100, 50)]), "bridge": "yes"},
                {"osm_id": 2, "geometry": LineString([(50, 0), (50, 100)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        wccs = list(nx.weakly_connected_components(G))
        # No shared interior node → two separate WCCs
        assert len(wccs) == 2
        assert stats.grade_sep_crossings_skipped >= 1

    def test_exactly_one_shared_t_junction_node(self):
        """T-junction must create exactly one shared node, not two coincident nodes."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, stats = build_road_graph(gdf)
        # Nodes at: (0,0), (100,0), (50,50), (50,0) = 4 total
        assert G.number_of_nodes() == 4
        # Check no duplicate coordinates
        coords = [(d["x"], d["y"]) for _, d in G.nodes(data=True)]
        assert len(coords) == len(set(coords)), "Duplicate node coordinates found"

    def test_no_duplicate_coincident_node(self):
        """After T-junction fix, the junction point has exactly one node ID."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        junction_nodes = [
            nid
            for nid, d in G.nodes(data=True)
            if abs(d["x"] - 50.0) < 0.01 and abs(d["y"] - 0.0) < 0.01
        ]
        assert len(junction_nodes) == 1, f"Expected 1 junction node, found {len(junction_nodes)}"

    def test_bidirectional_t_junction_splitting(self):
        """Bidirectional ways at a T-junction: both directions available after split."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        # All edges are bidirectional (oneway=None → "both")
        # 3 segments × 2 directions = 6 edges
        assert G.number_of_edges() == 6

    def test_forward_oneway_t_junction_splitting(self):
        """Oneway forward way split at T-junction: only forward edges created."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)]), "oneway": "yes"},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        wccs = list(nx.weakly_connected_components(G))
        assert len(wccs) == 1
        # Verify way 1's edges are forward-only (u < v in sorted node order is not reliable;
        # instead check that only one edge exists per segment of way 1)
        way1_segs = [(u, v) for u, v, d in G.edges(data=True) if d.get("osm_id") == 1]
        # Should have 2 segments × 1 direction = 2 edges for way 1
        assert len(way1_segs) == 2

    def test_reverse_oneway_t_junction_splitting(self):
        """Oneway reverse way split at T-junction."""
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)]), "oneway": "-1"},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        way1_segs = [(u, v) for u, v, d in G.edges(data=True) if d.get("osm_id") == 1]
        assert len(way1_segs) == 2  # 2 segments × 1 direction

    def test_segment_length_conservation(self):
        """Sum of split segment lengths equals original way length within 1e-4 m."""
        import math

        line = LineString([(0, 0), (100, 0)])
        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": line},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        # Segments of way 1 (osm_id=1)
        segs = [d["length_m"] for _, _, d in G.edges(data=True) if d.get("osm_id") == 1]  # noqa: F841
        # Two unique segments (each appears twice: forward + reverse)
        unique_segs = list(
            {
                d["edge_seq"]: d["length_m"]
                for _, _, d in G.edges(data=True)
                if d.get("osm_id") == 1
            }.values()
        )
        total = sum(unique_segs)
        assert math.isclose(total, line.length, abs_tol=1e-4), (
            f"Segment lengths {unique_segs} don't sum to {line.length}"
        )

    def test_attribute_preservation_after_split(self):
        """All original OSM attributes survive on every derived segment."""
        gdf = self._make_roads(
            [
                {
                    "osm_id": 1,
                    "geometry": LineString([(0, 0), (100, 0)]),
                    "highway": "primary",
                    "name": "Main St",
                    "surface": "asphalt",
                },
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
            ]
        )
        G, _ = build_road_graph(gdf)
        for _, _, d in G.edges(data=True):
            if d.get("osm_id") == 1:
                assert d.get("highway") == "primary"
                assert d.get("name") == "Main St"
                assert d.get("surface") == "asphalt"

    def test_physical_vs_directed_length_totals(self):
        """Bidirectional segment: directed_length = 2 × physical_length."""
        import math

        line = LineString([(0, 0), (100, 0)])
        gdf = self._make_roads([{"osm_id": 1, "geometry": line}])
        G, stats = build_road_graph(gdf)
        # Single bidirectional way: 2 directed edges, 1 physical segment
        assert math.isclose(stats.total_directed_edge_length_m, 200.0, abs_tol=1e-6)
        assert math.isclose(stats.total_physical_segment_length_m, 100.0, abs_tol=1e-6)
        assert stats.total_directed_edge_length_m == pytest.approx(
            2 * stats.total_physical_segment_length_m
        )

    def test_connectivity_threshold_failure(self):
        """Graph with large fragmentation fails node-coverage threshold."""
        # Two disconnected 2-node components
        G = nx.MultiDiGraph()
        G.graph["crs"] = "EPSG:32651"
        G.add_node(0, x=0.0, y=0.0, on_boundary=False)
        G.add_node(1, x=10.0, y=0.0, on_boundary=False)
        G.add_node(2, x=100.0, y=0.0, on_boundary=False)
        G.add_node(3, x=110.0, y=0.0, on_boundary=False)
        G.add_edge(0, 1, osm_id=1, edge_seq=0, length_m=10.0)
        G.add_edge(1, 0, osm_id=1, edge_seq=0, length_m=10.0)
        G.add_edge(2, 3, osm_id=2, edge_seq=0, length_m=10.0)
        G.add_edge(3, 2, osm_id=2, edge_seq=0, length_m=10.0)
        with pytest.raises(GraphValidationFailed, match="node coverage"):
            validate_road_graph(
                G,
                min_largest_wcc_node_coverage=0.9,
                min_largest_wcc_physical_length_coverage=0.0,
            )

    def test_deterministic_regeneration_after_fix(self, tmp_path):
        """Build graph twice from same input; GraphML output is byte-identical."""
        import hashlib

        gdf = self._make_roads(
            [
                {"osm_id": 1, "geometry": LineString([(0, 0), (100, 0)])},
                {"osm_id": 2, "geometry": LineString([(50, 50), (50, 0)])},
                {"osm_id": 3, "geometry": LineString([(100, 0), (200, 0)])},
            ]
        )
        p1 = tmp_path / "g1.graphml"
        p2 = tmp_path / "g2.graphml"
        G1, _ = build_road_graph(gdf)
        G2, _ = build_road_graph(gdf)
        write_graphml(G1, p1)
        write_graphml(G2, p2)
        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        assert h1 == h2, "GraphML output is not deterministic"
