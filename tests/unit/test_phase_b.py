"""Unit tests for Stage 5B hazard-attribution module (phase_b.py).

Tests cover:
- _pixel_hits: interior-overlap pixel sampling with synthetic rasters
- _jrc_rp_attr: status determination and depth statistics from PixelHit lists
- _terrain_attr: DEM sampling and elevation statistics
- _waterway_attrs: nearest-distance and crossing detection
- run_phase_b: end-to-end with synthetic graph and mocked rasters
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString

from floodroute.hazard.jrc import MIN_INTERIOR_LEN_M
from floodroute.hazard.phase_b import (
    PhaseBResult,
    PixelHit,
    _jrc_rp_attr,
    _pixel_hits,
    _terrain_attr,
    _waterway_attrs,
    run_phase_b,
)

# ── Synthetic raster writers ────────────────────────────────────────────────

# A small 10×10 pixel raster in WGS84 centred near San Jose
_W, _S = 121.95, 10.70
_PIX = 0.000808484  # ~90 m


def _write_depth_raster(
    path: Path,
    *,
    depth_rp10: float = 0.0,
    depth_rp20: float = 0.0,
    depth_rp100: float = 0.0,
    perm_water: float = -1.0,
    spurious: float = -1.0,
    h: int = 10,
    w: int = 10,
) -> tuple[Path, rasterio.transform.Affine]:
    east = _W + w * _PIX
    north = _S + h * _PIX
    transform = from_bounds(_W, _S, east, north, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)

    bands = [
        np.full((h, w), depth_rp10, dtype=np.float32),
        np.full((h, w), depth_rp20, dtype=np.float32),
        np.full((h, w), depth_rp100, dtype=np.float32),
        np.full((h, w), perm_water, dtype=np.float32),
        np.full((h, w), spurious, dtype=np.float32),
    ]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=5,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i, b in enumerate(bands, 1):
            dst.write(b, i)
    return path, transform


def _write_cat_raster(
    path: Path,
    *,
    rp10_cat: float = -1.0,
    rp20_cat: float = -1.0,
    rp100_cat: float = -1.0,
    h: int = 10,
    w: int = 10,
) -> tuple[Path, rasterio.transform.Affine]:
    """Write a 5-band category raster matching the re-acquired EE format.

    Band order: RP10_cat(0), RP20_cat(1), RP100_cat(2), perm_water(3), spurious(4).
    Not-flooded sentinel: -1.0 (not -9999; EE export uses -1 for both outside-domain
    and modelled-dry cells).
    """
    east = _W + w * _PIX
    north = _S + h * _PIX
    transform = from_bounds(_W, _S, east, north, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = [
        np.full((h, w), rp10_cat, dtype=np.float32),
        np.full((h, w), rp20_cat, dtype=np.float32),
        np.full((h, w), rp100_cat, dtype=np.float32),
        np.full((h, w), -1.0, dtype=np.float32),  # perm_water (-1 = not permanent)
        np.full((h, w), -1.0, dtype=np.float32),  # spurious (-1 = not flagged)
    ]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=5,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i, b in enumerate(bands, 1):
            dst.write(b, i)
    return path, transform


def _make_edge_wgs84(
    x0: float = _W + 2 * _PIX,
    y0: float = _S + 2 * _PIX,
    x1: float = _W + 4 * _PIX,
    y1: float = _S + 4 * _PIX,
) -> LineString:
    """Return a short LineString that crosses ~3 pixels diagonally."""
    return LineString([(x0, y0), (x1, y1)])


# ── TestPixelHits ───────────────────────────────────────────────────────────


class TestPixelHits:
    def _load_arrays(self, depth_path: Path, cat_path: Path):
        with rasterio.open(depth_path) as src:
            d_arr = src.read().astype(np.float32)
            d_tr = src.transform
        with rasterio.open(cat_path) as src:
            c_arr = src.read().astype(np.float32)
            c_tr = src.transform
        return d_arr, d_tr, c_arr, c_tr

    def test_no_hits_when_edge_outside_raster(self, tmp_path: Path) -> None:
        dp, d_tr = _write_depth_raster(tmp_path / "d.tif", depth_rp10=1.5)
        cp, c_tr = _write_cat_raster(tmp_path / "c.tif", rp10_cat=2.0)
        d_arr, _, c_arr, _ = self._load_arrays(dp, cp)

        # Edge completely outside raster extent
        edge = LineString([(100.0, 0.0), (100.1, 0.1)])
        hits = _pixel_hits(edge, 1000.0, d_arr, c_arr, d_tr, c_tr)
        assert hits == []

    def test_hits_found_for_crossing_edge(self, tmp_path: Path) -> None:
        dp, d_tr = _write_depth_raster(tmp_path / "d.tif", depth_rp10=1.5)
        cp, c_tr = _write_cat_raster(tmp_path / "c.tif", rp10_cat=2.0)
        d_arr, _, c_arr, _ = self._load_arrays(dp, cp)

        edge = _make_edge_wgs84()
        # Edge is ~300 m long; each hit should be >= MIN_INTERIOR_LEN_M (5 m)
        length_m = 300.0
        hits = _pixel_hits(edge, length_m, d_arr, c_arr, d_tr, c_tr)
        assert len(hits) > 0
        for h in hits:
            assert h.overlap_m >= MIN_INTERIOR_LEN_M

    def test_hits_carry_depth_values(self, tmp_path: Path) -> None:
        dp, d_tr = _write_depth_raster(
            tmp_path / "d.tif", depth_rp10=0.8, depth_rp20=1.2, depth_rp100=2.0
        )
        cp, c_tr = _write_cat_raster(tmp_path / "c.tif", rp10_cat=2.0, rp20_cat=2.0, rp100_cat=2.0)
        d_arr, _, c_arr, _ = self._load_arrays(dp, cp)

        hits = _pixel_hits(_make_edge_wgs84(), 300.0, d_arr, c_arr, d_tr, c_tr)
        assert len(hits) > 0
        for h in hits:
            assert abs(h.depth_rp10 - 0.8) < 0.01
            assert abs(h.depth_rp20 - 1.2) < 0.01
            assert abs(h.depth_rp100 - 2.0) < 0.01
            # Category bands are read from 5-band cat raster
            assert h.cat_rp10 == pytest.approx(2.0)
            assert h.cat_rp20 == pytest.approx(2.0)
            assert h.cat_rp100 == pytest.approx(2.0)

    def test_not_flooded_pixels_recorded_correctly(self, tmp_path: Path) -> None:
        """Not-flooded pixels have cat = -1 (EE export sentinel for not-flooded)."""
        dp, d_tr = _write_depth_raster(tmp_path / "d.tif", depth_rp10=0.0)
        cp, c_tr = _write_cat_raster(tmp_path / "c.tif", rp10_cat=-1.0)
        d_arr, _, c_arr, _ = self._load_arrays(dp, cp)

        hits = _pixel_hits(_make_edge_wgs84(), 300.0, d_arr, c_arr, d_tr, c_tr)
        assert len(hits) > 0
        for h in hits:
            assert h.depth_rp10 == pytest.approx(0.0)
            assert h.cat_rp10 == pytest.approx(-1.0)

    def test_no_cat_raster_returns_not_flooded_cat(self, tmp_path: Path) -> None:
        """When no category raster is provided, default cat = _NOT_FLOODED_CAT (-1)."""
        dp, d_tr = _write_depth_raster(tmp_path / "d.tif", depth_rp10=1.5)
        with rasterio.open(dp) as src:
            d_arr = src.read().astype(np.float32)
            d_tr = src.transform

        hits = _pixel_hits(_make_edge_wgs84(), 300.0, d_arr, None, d_tr, None)
        assert len(hits) > 0
        for h in hits:
            assert h.cat_rp10 == pytest.approx(-1.0)
            assert h.cat_rp20 == pytest.approx(-1.0)
            assert h.cat_rp100 == pytest.approx(-1.0)


# ── TestJrcRpAttr ───────────────────────────────────────────────────────────


_NOT_FLOODED_CAT = -1.0  # EE export cat sentinel: pixel not flooded (cat=-1)
_DOMAIN_DEPTH = 0.0  # depth == 0 → within model domain but not inundated (modelled_dry)
_OUTSIDE_DOMAIN_DEPTH = -9999.0  # NODATA_SENTINEL → outside GloFAS model domain
_SPURIOUS_NOT_FLAGGED = 0.0  # depth raster: 0 = not spurious
_PERM_WATER_NO = 0.0  # depth raster: 0 = not permanent water

# Legacy alias used by several tests (cat value for not-flooded pixels)
_NOT_FLOODED = _NOT_FLOODED_CAT


def _hit(
    overlap_m: float,
    depth_rp10: float = 0.0,
    depth_rp20: float = 0.0,
    depth_rp100: float = 0.0,
    cat_rp10: float = _NOT_FLOODED,
    cat_rp20: float = _NOT_FLOODED,
    cat_rp100: float = _NOT_FLOODED,
    perm_water: float = _PERM_WATER_NO,
    spurious: float = _SPURIOUS_NOT_FLAGGED,
) -> PixelHit:
    return PixelHit(
        overlap_m=overlap_m,
        depth_rp10=depth_rp10,
        depth_rp20=depth_rp20,
        depth_rp100=depth_rp100,
        cat_rp10=cat_rp10,
        cat_rp20=cat_rp20,
        cat_rp100=cat_rp100,
        perm_water=perm_water,
        spurious=spurious,
    )


class TestJrcRpAttr:
    def test_no_hits_returns_no_overlap(self) -> None:
        attr = _jrc_rp_attr([], 500.0, rp="RP10")
        assert attr.status == "no_overlap"
        assert attr.sample_n == 0
        assert attr.exposed_m == 0.0
        assert math.isnan(attr.depth_max_m)

    def test_modelled_dry_status_when_depth_zero_and_cat_minus_one(self) -> None:
        """depth == 0.0 (in domain) + cat == -1 (not flooded) → modelled_dry."""
        hits = [_hit(50.0, depth_rp10=_DOMAIN_DEPTH, cat_rp10=_NOT_FLOODED_CAT)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.status == "modelled_dry"
        assert attr.exposed_m == 0.0

    def test_modelled_dry_when_all_pixels_in_domain_not_flooded(self) -> None:
        """Multiple in-domain not-flooded pixels → modelled_dry status."""
        hits = [
            _hit(30.0, depth_rp10=_DOMAIN_DEPTH, cat_rp10=_NOT_FLOODED_CAT),
            _hit(20.0, depth_rp10=_DOMAIN_DEPTH, cat_rp10=_NOT_FLOODED_CAT),
        ]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.status == "modelled_dry"

    def test_outside_domain_when_all_hits_have_nodata_depth(self) -> None:
        """All hits with depth == -9999 (NODATA_SENTINEL) → outside_domain."""
        hits = [_hit(50.0, depth_rp10=_OUTSIDE_DOMAIN_DEPTH, cat_rp10=_NOT_FLOODED_CAT)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.status == "outside_domain"
        assert attr.exposed_m == 0.0

    def test_flooded_status_rp10(self) -> None:
        hits = [_hit(50.0, depth_rp10=1.5, cat_rp10=2.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.status == "flooded"
        assert attr.exposed_m == pytest.approx(50.0)
        assert attr.depth_max_m == pytest.approx(1.5)
        assert attr.exposed_pct == pytest.approx(10.0)

    def test_flooded_status_rp100(self) -> None:
        hits = [_hit(80.0, depth_rp100=2.5, cat_rp100=3.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP100")
        assert attr.status == "flooded"
        assert attr.exposed_m == pytest.approx(80.0)
        assert attr.depth_max_m == pytest.approx(2.5)

    def test_rp20_uses_actual_cat_rp20_band(self) -> None:
        """RP20 reads cat_rp20 directly — not inferred from RP10 or depth."""
        hits = [_hit(60.0, depth_rp20=0.9, cat_rp20=2.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP20")
        assert attr.status == "flooded"
        assert attr.exposed_m == pytest.approx(60.0)

    def test_rp20_modelled_dry_when_cat_rp20_is_not_flooded(self) -> None:
        """RP20 modelled_dry when cat_rp20 == -1 and depth is in-domain."""
        hits = [_hit(30.0, depth_rp20=0.9, cat_rp20=-1.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP20")
        assert attr.status == "modelled_dry"
        assert attr.exposed_m == pytest.approx(0.0)

    def test_rp20_does_not_use_cat_rp10(self) -> None:
        """RP20 status comes from cat_rp20, not cat_rp10."""
        # cat_rp10 = 2 (would give flooded if RP10 logic were applied),
        # but cat_rp20 = -1 → RP20 should be modelled_dry.
        hits = [_hit(30.0, depth_rp20=0.9, cat_rp10=2.0, cat_rp20=-1.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP20")
        assert attr.status == "modelled_dry"

    def test_permanent_water_not_counted_as_exposed(self) -> None:
        hits = [_hit(50.0, depth_rp10=1.0, cat_rp10=2.0, perm_water=1.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.status == "flooded"  # flood pixel present
        assert attr.exposed_m == pytest.approx(0.0)  # but excluded (permanent water)
        assert attr.perm_water_m == pytest.approx(50.0)

    def test_spurious_flag_propagates(self) -> None:
        hits = [_hit(50.0, cat_rp10=-1.0, spurious=1.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.spurious is True

    def test_no_spurious_when_not_flagged(self) -> None:
        hits = [_hit(50.0, depth_rp10=1.5, cat_rp10=2.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.spurious is False

    def test_depth_wt_mean_is_overlap_weighted(self) -> None:
        # Two flooded pixels: 50 m at depth 1.0 and 100 m at depth 2.0
        hits = [
            _hit(50.0, depth_rp10=1.0, cat_rp10=2.0),
            _hit(100.0, depth_rp10=2.0, cat_rp10=2.0),
        ]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        expected_mean = (50 * 1.0 + 100 * 2.0) / (50 + 100)
        assert attr.depth_wt_mean_m == pytest.approx(expected_mean, abs=0.001)
        assert attr.depth_max_m == pytest.approx(2.0)

    def test_sample_n_counts_all_hits(self) -> None:
        hits = [_hit(10.0), _hit(20.0), _hit(30.0)]
        attr = _jrc_rp_attr(hits, 500.0, rp="RP10")
        assert attr.sample_n == 3


# ── TestTerrainAttr ─────────────────────────────────────────────────────────


class TestTerrainAttr:
    def _make_dem(self, tmp_path: Path, elev: float = 50.0) -> rasterio.DatasetReader:
        """Write a flat DEM raster in UTM (EPSG:32651)."""
        path = tmp_path / "dem.tif"
        h, w = 200, 200
        transform = from_bounds(300000, 1180000, 310000, 1190000, w, h)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=np.float32,
            crs="EPSG:32651",
            transform=transform,
        ) as dst:
            dst.write(np.full((h, w), elev, dtype=np.float32), 1)
        return rasterio.open(path)

    def test_flat_terrain_has_zero_change(self, tmp_path: Path) -> None:
        dem = self._make_dem(tmp_path, elev=100.0)
        edge = LineString([(300100, 1180100), (300900, 1180100)])
        t = _terrain_attr(edge, dem, edge.length)
        dem.close()
        assert t is not None
        assert t.elev_min_m == pytest.approx(100.0, abs=0.1)
        assert t.elev_max_m == pytest.approx(100.0, abs=0.1)
        assert t.elev_change_m == pytest.approx(0.0, abs=0.1)
        assert t.slope_pct == pytest.approx(0.0, abs=0.01)

    def test_returns_none_when_all_nodata(self, tmp_path: Path) -> None:
        path = tmp_path / "dem_nd.tif"
        h, w = 10, 10
        transform = from_bounds(300000, 1180000, 310000, 1190000, w, h)
        nodata_val = -9999.0
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=np.float32,
            crs="EPSG:32651",
            transform=transform,
            nodata=nodata_val,
        ) as dst:
            dst.write(np.full((h, w), nodata_val, dtype=np.float32), 1)

        dem = rasterio.open(path)
        edge = LineString([(300100, 1180100), (300900, 1180100)])
        t = _terrain_attr(edge, dem, edge.length)
        dem.close()
        assert t is None


# ── TestWaterwayAttrs ───────────────────────────────────────────────────────


class TestWaterwayAttrs:
    def _make_edges_gdf(self) -> gpd.GeoDataFrame:
        """Two edges: one crossing the waterway, one not."""
        return gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(300000, 1180000), (300000, 1181000)]),  # N-S
                    LineString([(302000, 1180000), (303000, 1180000)]),  # far away
                ]
            },
            crs="EPSG:32651",
        )

    def _make_waterway_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "waterway": ["river"],
                "name": ["Sibalom River"],
                "geometry": [LineString([(299000, 1180500), (301000, 1180500)])],  # E-W
            },
            crs="EPSG:32651",
        )

    def test_nearest_distance_computed(self) -> None:
        edges = self._make_edges_gdf()
        ww = self._make_waterway_gdf()
        result = _waterway_attrs(edges, ww)
        assert "waterway_nearest_dist_m" in result.columns
        # Edge 0 crosses the waterway at (300000, 1180500) — distance = 0
        assert result.loc[0, "waterway_nearest_dist_m"] == pytest.approx(0.0, abs=1.0)
        # Edge 1 is far from the waterway
        assert result.loc[1, "waterway_nearest_dist_m"] > 1000

    def test_crossing_detected_for_intersecting_edge(self) -> None:
        edges = self._make_edges_gdf()
        ww = self._make_waterway_gdf()
        result = _waterway_attrs(edges, ww)
        assert result.loc[0, "waterway_crossing"] is True or result.loc[0, "waterway_crossing"] == 1
        assert not result.loc[1, "waterway_crossing"]

    def test_waterway_name_returned(self) -> None:
        edges = self._make_edges_gdf()
        ww = self._make_waterway_gdf()
        result = _waterway_attrs(edges, ww)
        assert result.loc[0, "waterway_nearest_name"] == "Sibalom River"

    def test_empty_waterway_returns_nan_distance(self) -> None:
        edges = self._make_edges_gdf()
        ww = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32651")
        result = _waterway_attrs(edges, ww)
        assert math.isnan(result.loc[0, "waterway_nearest_dist_m"])


# ── TestRunPhaseB integration ────────────────────────────────────────────────


class TestRunPhaseBIntegration:
    """End-to-end test with a tiny synthetic graph and rasters."""

    def _build_graph(self) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
        """Two-node, bidirectional graph in EPSG:32651."""
        G = nx.MultiDiGraph()
        G.graph["crs"] = "EPSG:32651"
        x0, y0 = 305000, 1182000
        x1, y1 = 305200, 1182000
        G.add_node(0, x=x0, y=y0, on_boundary=False)
        G.add_node(1, x=x1, y=y1, on_boundary=False)
        G.add_edge(
            0,
            1,
            key=0,
            osm_id=1,
            edge_seq=0,
            highway="secondary",
            length_m=200.0,
            bridge=None,
            tunnel=None,
            name=None,
            ref=None,
            oneway="yes",
            lanes=None,
            maxspeed=None,
            surface=None,
            access=None,
            service=None,
            geometry=LineString([(x0, y0), (x1, y1)]),
        )
        G.add_edge(
            1,
            0,
            key=0,
            osm_id=1,
            edge_seq=0,
            highway="secondary",
            length_m=200.0,
            bridge=None,
            tunnel=None,
            name=None,
            ref=None,
            oneway="yes",
            lanes=None,
            maxspeed=None,
            surface=None,
            access=None,
            service=None,
            geometry=LineString([(x1, y1), (x0, y0)]),
        )
        edges_gdf = gpd.GeoDataFrame(
            [
                {
                    "u_node": 0,
                    "v_node": 1,
                    "edge_key": 0,
                    "osm_id": 1,
                    "edge_seq": 0,
                    "length_m": 200.0,
                    "highway": "secondary",
                    "bridge": None,
                    "tunnel": None,
                    "name": None,
                    "ref": None,
                    "oneway": "yes",
                    "lanes": None,
                    "maxspeed": None,
                    "surface": None,
                    "access": None,
                    "service": None,
                    "geometry": LineString([(x0, y0), (x1, y1)]),
                },
                {
                    "u_node": 1,
                    "v_node": 0,
                    "edge_key": 0,
                    "osm_id": 1,
                    "edge_seq": 0,
                    "length_m": 200.0,
                    "highway": "secondary",
                    "bridge": None,
                    "tunnel": None,
                    "name": None,
                    "ref": None,
                    "oneway": "yes",
                    "lanes": None,
                    "maxspeed": None,
                    "surface": None,
                    "access": None,
                    "service": None,
                    "geometry": LineString([(x1, y1), (x0, y0)]),
                },
            ],
            geometry="geometry",
            crs="EPSG:32651",
        )
        return G, edges_gdf

    def test_phase_b_only_allowed_for_ph0600613(self, tmp_path: Path) -> None:
        from floodroute.hazard.phase_b import run_phase_b

        with pytest.raises(ValueError, match="PH0600613"):
            run_phase_b(
                tmp_path / "g.graphml",
                tmp_path / "e.gpkg",
                tmp_path / "d.tif",
                tmp_path / "c.tif",
                municipality_pcode="PH0600616",
                output_dir=tmp_path / "out",
            )

    def test_raises_file_exists_error_without_force(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "PH0600613_phase_b_enriched.graphml").touch()

        with pytest.raises(FileExistsError):
            run_phase_b(
                tmp_path / "g.graphml",
                tmp_path / "e.gpkg",
                tmp_path / "d.tif",
                tmp_path / "c.tif",
                output_dir=out_dir,
                force=False,
            )

    def test_full_run_produces_outputs(self, tmp_path: Path) -> None:
        """Full run with synthetic rasters and graph."""
        G, edges_gdf = self._build_graph()

        # Write graph and edges
        from floodroute.graph.io import write_graphml

        graph_path = tmp_path / "graph.graphml"
        edges_path = tmp_path / "edges.gpkg"
        write_graphml(G, graph_path)
        edges_gdf.to_file(edges_path, driver="GPKG", layer="edges")

        # Write JRC rasters in WGS84 covering the edge area
        # Edge runs from ~305000,1182000 to 305200,1182000 in EPSG:32651
        # ~121.99°E, 10.69°N in WGS84 (approximate for San Jose area)
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        lon0, lat0 = tr.transform(304800, 1181800)
        lon1, lat1 = tr.transform(305400, 1182200)

        depth_path, _ = _write_depth_raster(
            tmp_path / "depth.tif",
            depth_rp10=1.2,
            depth_rp20=1.5,
            depth_rp100=2.0,
            h=20,
            w=20,
        )
        # Override to cover the right area
        depth_path2 = tmp_path / "depth2.tif"
        east2 = lon0 + 20 * _PIX
        north2 = lat0 + 20 * _PIX
        tform = from_bounds(lon0, lat0, east2, north2, 20, 20)
        with rasterio.open(
            depth_path2,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=5,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=tform,
        ) as dst:
            for i, val in enumerate([1.2, 1.5, 2.0, -1.0, -1.0], 1):
                dst.write(np.full((20, 20), val, dtype=np.float32), i)

        cat_path2 = tmp_path / "cat2.tif"
        with rasterio.open(
            cat_path2,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=5,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=tform,
        ) as dst:
            # 5 bands: RP10_cat, RP20_cat, RP100_cat, perm_water, spurious
            for i, val in enumerate([2.0, 2.0, 2.0, -1.0, -1.0], 1):
                dst.write(np.full((20, 20), val, dtype=np.float32), i)

        out_dir = tmp_path / "out"
        result = run_phase_b(
            graph_path,
            edges_path,
            depth_path2,
            cat_path2,
            output_dir=out_dir,
            force=True,
        )

        assert isinstance(result, PhaseBResult)
        assert result.municipality_code == "PH0600613"
        assert result.output_graphml.exists()
        assert result.output_gpkg.exists()
        assert len(result.sha256_graphml) == 64
        assert len(result.sha256_gpkg) == 64
        assert result.n_directed_edges == 2
        # Bidirectional pair → 1 physical segment
        assert result.n_physical_segments == 1
        # Physical km ≤ directed km (equal for identical bidirectional exposure)
        assert result.total_exposed_phys_m_rp10 <= result.total_exposed_dir_m_rp10

    def test_determinism(self, tmp_path: Path) -> None:
        """Two runs with same inputs produce identical SHA-256 outputs."""
        G, edges_gdf = self._build_graph()
        from pyproj import Transformer

        from floodroute.graph.io import write_graphml

        graph_path = tmp_path / "graph.graphml"
        edges_path = tmp_path / "edges.gpkg"
        write_graphml(G, graph_path)
        edges_gdf.to_file(edges_path, driver="GPKG", layer="edges")

        tr = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        lon0, lat0 = tr.transform(304800, 1181800)
        tform = from_bounds(lon0, lat0, lon0 + 20 * _PIX, lat0 + 20 * _PIX, 20, 20)

        depth_path = tmp_path / "depth.tif"
        cat_path = tmp_path / "cat.tif"
        for p, vals in [
            (depth_path, [1.2, 1.5, 2.0, 0.0, 0.0]),  # 5-band depth
            (cat_path, [2.0, 2.0, 2.0, -1.0, -1.0]),  # 5-band category
        ]:
            n_bands = len(vals)
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                height=20,
                width=20,
                count=n_bands,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=tform,
            ) as dst:
                for i, v in enumerate(vals, 1):
                    dst.write(np.full((20, 20), v, dtype=np.float32), i)

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        r1 = run_phase_b(graph_path, edges_path, depth_path, cat_path, output_dir=out1, force=True)
        r2 = run_phase_b(graph_path, edges_path, depth_path, cat_path, output_dir=out2, force=True)

        assert r1.sha256_graphml == r2.sha256_graphml

        # GeoPackage embeds SQLite timestamps so byte-identical SHA-256 is not
        # achievable. Verify logical determinism by comparing attribute values.
        gdf1 = gpd.read_file(r1.output_gpkg, layer="edges").sort_values(
            ["osm_id", "edge_seq"], ignore_index=True
        )
        gdf2 = gpd.read_file(r2.output_gpkg, layer="edges").sort_values(
            ["osm_id", "edge_seq"], ignore_index=True
        )
        import pandas as pd

        assert list(gdf1.columns) == list(gdf2.columns)
        for col in gdf1.columns:
            if col == "geometry":
                continue
            # Use pandas assert_series_equal for NaN-safe comparison
            pd.testing.assert_series_equal(
                gdf1[col].reset_index(drop=True),
                gdf2[col].reset_index(drop=True),
                check_names=False,
                obj=f"Column {col!r}",
            )
