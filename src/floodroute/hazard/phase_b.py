"""Stage 5B: JRC flood hazard attribution for the PH0600613 road graph.

Attributes each directed edge with four evidence layers, kept under separate
prefixes to prevent conflation:

  jrc_rp10_*  / jrc_rp20_*  / jrc_rp100_*   — flood depth evidence per RP
  terrain_*                                   — DEM elevation / slope
  waterway_*                                  — OSM waterway proximity / crossing
  network_*                                   — graph topology (WCC id)

CATEGORY RASTER (5-band, acquired 2026-08-23 from EE tile ID230_N20_E120):
  Band 0  RP10_depth_category   — -1 = not flooded; 2 | 3 = flooded
  Band 1  RP20_depth_category   — -1 = not flooded; 2 | 3 = flooded
  Band 2  RP100_depth_category  — -1 = not flooded; 2 | 3 = flooded
  Band 3  permanent_water_class — -1 = not permanent; 1 = permanent water
  Band 4  spurious_depth_category — -1 = not flagged; 1 = flagged

STATUS VOCABULARY (per RP):
  no_overlap      — no JRC pixel has interior overlap ≥ MIN_INTERIOR_LEN_M
  outside_domain  — pixels overlap but all have depth == NODATA_SENTINEL (-9999):
                    these pixels carry no valid hydraulic model output
  modelled_dry    — pixels overlap, at least one is within the model domain
                    (depth ≠ -9999), but none have category 2 or 3
  flooded         — at least one overlapping pixel has cat ∈ {2, 3} and is not
                    permanent water

NOTE: The re-acquired EE category raster uses -1 for both outside-domain and
modelled-dry cells.  These states are distinguished by the depth raster:
outside_domain pixels have depth == NODATA_SENTINEL (-9999, no valid model
output); modelled-dry pixels have depth == 0.0 (within domain, not inundated).
For PH0600613 (San Jose), the exported raster extent lies entirely within the
GloFAS model domain — no outside_domain pixels exist here; all non-flooded
pixels are modelled_dry.

Only non-permanent flooded pixels (permanent_water_class band of DEPTH raster
≠ 1) contribute to ``jrc_rp*_exposed_m``. Permanent-water intersections are
recorded separately as ``jrc_rp*_perm_water_m``.

Interior overlap threshold: MIN_INTERIOR_LEN_M (5.0 m). Zero-length boundary
touches are excluded.

PASSABILITY NOTE
----------------
This module records evidence only. It does NOT infer passability, closure,
travel speed, or routing decisions. Phase B outputs are inputs to Stage 6
routing experiments.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
import rasterio.transform
from pyproj import Transformer
from shapely.geometry import LineString

from floodroute.graph.io import read_graphml, write_graphml
from floodroute.hazard.jrc import MIN_INTERIOR_LEN_M, NODATA_SENTINEL

logger = logging.getLogger("floodroute.hazard.phase_b")

# ── San Jose de Buenavista (PH0600613) constants ───────────────────────────

MUNICIPALITY_PCODE: str = "PH0600613"
_NOT_FLOODED_CAT: int = -1  # category value for "not flooded" in EE export (-9999 not used)
_FLOOD_VALS: frozenset[int] = frozenset({2, 3})  # depth_category values indicating flooding

# Band indices in depth raster (0-based, 5 bands)
_D_RP10 = 0
_D_RP20 = 1
_D_RP100 = 2
_D_PW = 3  # permanent_water_class: 0 = not permanent, 1 = permanent water
_D_SD = 4  # spurious_depth_category: 0 = not flagged

# Band indices in category raster (0-based, 5 bands — acquired 2026-08-23)
_C_RP10 = 0  # RP10_depth_category
_C_RP20 = 1  # RP20_depth_category (acquired directly; not inferred)
_C_RP100 = 2  # RP100_depth_category
_C_PW = 3  # permanent_water_class (-1 = not permanent, 1 = permanent)
_C_SD = 4  # spurious_depth_category (-1 = not flagged, 1 = flagged)

# DEM sampling density (points along edge)
_DEM_SAMPLE_PTS: int = 20

# Return periods to attribute
_RETURN_PERIODS: list[str] = ["RP10", "RP20", "RP100"]


# ── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class PixelHit:
    """Interior-overlap record for one JRC pixel touching a directed edge."""

    overlap_m: float  # interior overlap in metres
    depth_rp10: float  # depth (m); 0.0 for not-flooded pixels
    depth_rp20: float
    depth_rp100: float
    cat_rp10: float  # RP10_depth_category: -1 = not_flooded, 2|3 = flooded
    cat_rp20: float  # RP20_depth_category: -1 = not_flooded, 2|3 = flooded
    cat_rp100: float  # RP100_depth_category: -1 = not_flooded, 2|3 = flooded
    perm_water: float  # from DEPTH raster band 3: 0.0 = not permanent, 1.0 = permanent
    spurious: float  # from DEPTH raster band 4: 0.0 = not flagged, >0 = flagged


@dataclass
class JrcRpAttr:
    """JRC attribution for one directed edge at one return period."""

    status: str  # no_overlap | outside_domain | modelled_dry | flooded
    depth_max_m: float  # nan if no flood exposure
    depth_wt_mean_m: float  # overlap-length-weighted mean depth; nan if no exposure
    exposed_m: float  # interior overlap with non-permanent flooded pixels (m)
    exposed_pct: float  # exposed_m / edge_length_m * 100
    perm_water_m: float  # interior overlap with permanent-water pixels (m)
    spurious: bool  # any spurious pixel in the interior overlap set
    sample_n: int  # pixel hits with interior overlap >= MIN_INTERIOR_LEN_M


@dataclass
class TerrainAttr:
    """DEM-derived terrain evidence for one directed edge."""

    elev_min_m: float
    elev_mean_m: float
    elev_max_m: float
    elev_change_m: float  # max - min elevation along edge
    slope_pct: float  # elev_change_m / length_m * 100


@dataclass
class WaterwayAttr:
    """OSM waterway proximity evidence for one directed edge."""

    nearest_dist_m: float
    crossing: bool
    nearest_name: str | None  # OSM name attribute of nearest waterway
    nearest_type: str | None  # OSM waterway tag (river, stream, canal, …)


@dataclass
class PhaseBResult:
    """Summary of the completed Phase B attribution run.

    Exposure figures are reported at two levels:
    - Directed edges: all directed edges; bidirectional pairs counted twice.
    - Physical segments: unique (osm_id, edge_seq) pairs; each physical road
      segment counted once regardless of travel direction.
    """

    municipality_code: str
    n_directed_edges: int
    n_physical_segments: int

    # ── Directed-edge exposure (both directions counted) ───────────────────
    n_exposed_dir_rp10: int  # directed edges with jrc_rp10_exposed_m > 0
    n_exposed_dir_rp20: int
    n_exposed_dir_rp100: int
    total_exposed_dir_m_rp10: float  # sum of exposed_m over directed edges
    total_exposed_dir_m_rp20: float
    total_exposed_dir_m_rp100: float

    # ── Physical-segment exposure (deduplicated) ───────────────────────────
    n_exposed_phys_rp10: int  # unique physical segments with exposure > 0
    n_exposed_phys_rp20: int
    n_exposed_phys_rp100: int
    total_exposed_phys_m_rp10: float  # sum of per-segment exposed_m (one per segment)
    total_exposed_phys_m_rp20: float
    total_exposed_phys_m_rp100: float

    n_waterway_crossings: int
    output_graphml: Path
    output_gpkg: Path
    sha256_graphml: str
    sha256_gpkg: str
    warnings: list[str] = field(default_factory=list)


# ── JRC pixel sampling ──────────────────────────────────────────────────────


def _pixel_hits(
    edge_geom_4326: Any,
    edge_length_m: float,
    depth_arr: np.ndarray,
    cat_arr: np.ndarray | None,
    depth_transform: Any,
    cat_transform: Any | None,
) -> list[PixelHit]:
    """Return PixelHit records for JRC pixels with interior overlap >= MIN_INTERIOR_LEN_M.

    Parameters
    ----------
    edge_geom_4326:
        Edge LineString reprojected to EPSG:4326.
    edge_length_m:
        Authoritative length from Stage 4 UTM graph (metres).
    depth_arr:
        Full depth raster array, shape (5, H, W).
    cat_arr:
        Full category raster array, shape (5, H, W), or None.
        Bands: RP10_cat(0), RP20_cat(1), RP100_cat(2), perm_water(3), spurious(4).
    depth_transform:
        Affine transform for the depth raster.
    cat_transform:
        Affine transform for the category raster (may differ from depth).
    """
    from shapely.geometry import box as shapely_box  # noqa: PLC0415

    if edge_geom_4326.is_empty or edge_length_m == 0:
        return []

    H, W = depth_arr.shape[1], depth_arr.shape[2]
    min_x, min_y, max_x, max_y = edge_geom_4326.bounds

    # Convert bbox to pixel row/col indices (with 1-pixel padding)
    row_nw, col_nw = rasterio.transform.rowcol(depth_transform, min_x, max_y)
    row_se, col_se = rasterio.transform.rowcol(depth_transform, max_x, min_y)
    r0 = max(0, min(row_nw, row_se) - 1)
    r1 = min(H - 1, max(row_nw, row_se) + 1)
    c0 = max(0, min(col_nw, col_se) - 1)
    c1 = min(W - 1, max(col_nw, col_se) + 1)

    edge_len_4326 = edge_geom_4326.length
    if edge_len_4326 == 0:
        return []

    hits: list[PixelHit] = []
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            px_x, px_y = rasterio.transform.xy(depth_transform, row, col)
            half_w = abs(depth_transform.a) / 2
            half_h = abs(depth_transform.e) / 2
            pbox = shapely_box(px_x - half_w, px_y - half_h, px_x + half_w, px_y + half_h)

            if not pbox.intersects(edge_geom_4326):
                continue
            inter = pbox.intersection(edge_geom_4326)
            if inter.is_empty:
                continue

            frac = inter.length / edge_len_4326
            overlap_m = frac * edge_length_m
            if overlap_m < MIN_INTERIOR_LEN_M:
                continue

            d10 = float(depth_arr[_D_RP10, row, col])
            d20 = float(depth_arr[_D_RP20, row, col])
            d100 = float(depth_arr[_D_RP100, row, col])
            pw = float(depth_arr[_D_PW, row, col])
            sd = float(depth_arr[_D_SD, row, col])

            # Category raster may use a different transform / size.
            # Default to _NOT_FLOODED_CAT so missing coverage → not_flooded.
            c10 = float(_NOT_FLOODED_CAT)
            c20 = float(_NOT_FLOODED_CAT)
            c100 = float(_NOT_FLOODED_CAT)
            if cat_arr is not None and cat_transform is not None:
                cat_H, cat_W = cat_arr.shape[1], cat_arr.shape[2]
                cr, cc = rasterio.transform.rowcol(cat_transform, px_x, px_y)
                if 0 <= cr < cat_H and 0 <= cc < cat_W:
                    c10 = float(cat_arr[_C_RP10, cr, cc])
                    c20 = float(cat_arr[_C_RP20, cr, cc])
                    c100 = float(cat_arr[_C_RP100, cr, cc])

            hits.append(
                PixelHit(
                    overlap_m=overlap_m,
                    depth_rp10=d10,
                    depth_rp20=d20,
                    depth_rp100=d100,
                    cat_rp10=c10,
                    cat_rp20=c20,
                    cat_rp100=c100,
                    perm_water=pw,
                    spurious=sd,
                )
            )

    return hits


# ── JRC attribution from pixel hits ────────────────────────────────────────


def _jrc_rp_attr(
    hits: list[PixelHit],
    edge_length_m: float,
    *,
    rp: str,
) -> JrcRpAttr:
    """Compute JrcRpAttr for one return period from pixel hits.

    Three-state status determination:
      outside_domain — all hits have depth == NODATA_SENTINEL (-9999):
                       no valid hydraulic model output for this pixel
      modelled_dry   — at least one hit is within the model domain
                       (depth ≠ -9999) but none have cat ∈ {2, 3}
      flooded        — at least one hit has cat ∈ {2, 3}

    RP20 uses ``cat_rp20`` from the PixelHit, read directly from the acquired
    RP20_depth_category band — no inference from RP10 is performed.
    """
    nan = float("nan")

    if not hits:
        return JrcRpAttr(
            status="no_overlap",
            depth_max_m=nan,
            depth_wt_mean_m=nan,
            exposed_m=0.0,
            exposed_pct=0.0,
            perm_water_m=0.0,
            spurious=False,
            sample_n=0,
        )

    exposed_m = 0.0
    perm_water_m = 0.0
    spurious = False
    weighted_depth = 0.0
    depth_max = nan
    any_flooded = False
    any_in_domain = False  # depth ≠ NODATA_SENTINEL → within model domain

    for h in hits:
        if rp == "RP10":
            depth = h.depth_rp10
            cat = h.cat_rp10
        elif rp == "RP20":
            depth = h.depth_rp20
            cat = h.cat_rp20
        else:  # RP100
            depth = h.depth_rp100
            cat = h.cat_rp100

        if depth != NODATA_SENTINEL:
            any_in_domain = True

        is_flooded = int(cat) in _FLOOD_VALS

        if is_flooded:
            any_flooded = True
            is_perm = h.perm_water == 1.0
            if is_perm:
                perm_water_m += h.overlap_m
            else:
                exposed_m += h.overlap_m
                weighted_depth += depth * h.overlap_m
                if np.isnan(depth_max) or depth > depth_max:
                    depth_max = depth

        if h.spurious > 0:
            spurious = True

    if any_flooded:
        status = "flooded"
    elif any_in_domain:
        status = "modelled_dry"
    else:
        status = "outside_domain"
    depth_wt_mean = (weighted_depth / exposed_m) if exposed_m > 0 else nan
    exposed_pct = (exposed_m / edge_length_m * 100) if edge_length_m > 0 else 0.0

    return JrcRpAttr(
        status=status,
        depth_max_m=depth_max,
        depth_wt_mean_m=depth_wt_mean,
        exposed_m=round(exposed_m, 4),
        exposed_pct=round(exposed_pct, 4),
        perm_water_m=round(perm_water_m, 4),
        spurious=spurious,
        sample_n=len(hits),
    )


# ── DEM terrain attribution ────────────────────────────────────────────────


def _terrain_attr(
    edge_geom_utm: Any,
    dem_src: rasterio.DatasetReader,
    edge_length_m: float,
) -> TerrainAttr | None:
    """Sample DEM at evenly-spaced points along the edge and compute stats."""
    n_pts = max(3, min(_DEM_SAMPLE_PTS, int(edge_length_m / 5) + 2))
    pts = [edge_geom_utm.interpolate(t, normalized=True) for t in np.linspace(0, 1, n_pts)]
    xy = [(pt.x, pt.y) for pt in pts]
    nodata = dem_src.nodata

    vals: list[float] = []
    for sample in dem_src.sample(xy):
        v = float(sample[0])
        if nodata is not None and v == nodata:
            continue
        if np.isnan(v):
            continue
        vals.append(v)

    if not vals:
        return None

    elev_min = float(np.min(vals))
    elev_max = float(np.max(vals))
    elev_mean = float(np.mean(vals))
    elev_change = elev_max - elev_min
    slope_pct = (elev_change / edge_length_m * 100) if edge_length_m > 0 else 0.0

    return TerrainAttr(
        elev_min_m=round(elev_min, 2),
        elev_mean_m=round(elev_mean, 2),
        elev_max_m=round(elev_max, 2),
        elev_change_m=round(elev_change, 2),
        slope_pct=round(slope_pct, 4),
    )


# ── Waterway attribution (bulk, using spatial joins) ───────────────────────


def _waterway_attrs(
    edges_utm: gpd.GeoDataFrame,
    waterway_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Bulk-compute waterway distance and crossing for all edges.

    Returns a DataFrame indexed to match edges_utm with columns:
    waterway_nearest_dist_m, waterway_crossing, waterway_nearest_name,
    waterway_nearest_type.
    """
    if waterway_gdf.empty:
        result = edges_utm[[]].copy()
        result["waterway_nearest_dist_m"] = float("nan")
        result["waterway_crossing"] = False
        result["waterway_nearest_name"] = None
        result["waterway_nearest_type"] = None
        return result

    # Nearest waterway (distance + attributes)
    name_col = "name" if "name" in waterway_gdf.columns else None
    type_col = "waterway" if "waterway" in waterway_gdf.columns else None

    keep_cols = ["geometry"]
    if name_col:
        keep_cols.append(name_col)
    if type_col:
        keep_cols.append(type_col)

    nearest = gpd.sjoin_nearest(
        edges_utm[["geometry"]].copy(),
        waterway_gdf[keep_cols].copy(),
        how="left",
        distance_col="waterway_nearest_dist_m",
    )
    # Keep only first match per edge (duplicates arise from ties)
    nearest = nearest[~nearest.index.duplicated(keep="first")]

    # Crossing detection (edge geometry intersects waterway geometry)
    crossing_join = gpd.sjoin(
        edges_utm[["geometry"]].copy(),
        waterway_gdf[["geometry"]].copy(),
        how="left",
        predicate="intersects",
    )
    crossing_set = set(crossing_join.dropna(subset=["index_right"]).index)

    result = edges_utm[[]].copy()
    result["waterway_nearest_dist_m"] = nearest["waterway_nearest_dist_m"].round(2)
    result["waterway_crossing"] = result.index.map(lambda i: i in crossing_set)
    if name_col:
        result["waterway_nearest_name"] = nearest[name_col].where(nearest[name_col].notna(), None)
    else:
        result["waterway_nearest_name"] = None
    if type_col:
        result["waterway_nearest_type"] = nearest[type_col].where(nearest[type_col].notna(), None)
    else:
        result["waterway_nearest_type"] = None

    return result


# ── SHA-256 helper ─────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ── Attribute attachment helpers ───────────────────────────────────────────


def _jrc_attrs_to_dict(rp10: JrcRpAttr, rp20: JrcRpAttr, rp100: JrcRpAttr) -> dict[str, Any]:
    """Flatten three JrcRpAttr into a flat dict with prefixed keys."""
    nan = float("nan")
    out: dict[str, Any] = {}
    for prefix, attr in [("jrc_rp10", rp10), ("jrc_rp20", rp20), ("jrc_rp100", rp100)]:
        out[f"{prefix}_status"] = attr.status
        out[f"{prefix}_depth_max_m"] = attr.depth_max_m if not np.isnan(attr.depth_max_m) else nan
        out[f"{prefix}_depth_wt_mean_m"] = (
            attr.depth_wt_mean_m if not np.isnan(attr.depth_wt_mean_m) else nan
        )
        out[f"{prefix}_exposed_m"] = attr.exposed_m
        out[f"{prefix}_exposed_pct"] = attr.exposed_pct
        out[f"{prefix}_perm_water_m"] = attr.perm_water_m
        out[f"{prefix}_spurious"] = int(attr.spurious)
        out[f"{prefix}_sample_n"] = attr.sample_n
    return out


def _terrain_to_dict(t: TerrainAttr | None) -> dict[str, Any]:
    nan = float("nan")
    if t is None:
        return {
            "terrain_elev_min_m": nan,
            "terrain_elev_mean_m": nan,
            "terrain_elev_max_m": nan,
            "terrain_elev_change_m": nan,
            "terrain_slope_pct": nan,
        }
    return {
        "terrain_elev_min_m": t.elev_min_m,
        "terrain_elev_mean_m": t.elev_mean_m,
        "terrain_elev_max_m": t.elev_max_m,
        "terrain_elev_change_m": t.elev_change_m,
        "terrain_slope_pct": t.slope_pct,
    }


# ── Main Phase B entry point ────────────────────────────────────────────────


def run_phase_b(
    graph_path: Path,
    edges_gpkg: Path,
    depth_raster: Path,
    cat_raster: Path,
    *,
    municipality_pcode: str = MUNICIPALITY_PCODE,
    dem_path: Path | None = None,
    waterway_path: Path | None = None,
    output_dir: Path,
    force: bool = False,
) -> PhaseBResult:
    """Run Stage 5B: attribute each directed edge with flood, terrain and waterway evidence.

    Parameters
    ----------
    graph_path:
        Stage 4 GraphML (EPSG:32651 MultiDiGraph).
    edges_gpkg:
        Stage 4 edges GeoPackage (same graph, with geometry).
    depth_raster:
        5-band JRC depth GeoTIFF (RP10/RP20/RP100/perm_water/spurious).
    cat_raster:
        5-band JRC category GeoTIFF
        (RP10_cat/RP20_cat/RP100_cat/perm_water/spurious).
    municipality_pcode:
        PSA adm3_pcode (only PH0600613 is READY for Phase B).
    dem_path:
        Copernicus GLO-30 DEM clipped to municipality (EPSG:32651).  Optional.
    waterway_path:
        OSM waterway GeoPackage (EPSG:32651).  Optional.
    output_dir:
        Directory for enriched GraphML and GeoPackage outputs.
    force:
        Overwrite existing outputs.
    """
    if municipality_pcode != MUNICIPALITY_PCODE:
        raise ValueError(
            f"Phase B is only READY for {MUNICIPALITY_PCODE} (San Jose de Buenavista). "
            f"Received: {municipality_pcode}. "
            f"PH0600616 (Sibalom) is BLOCKED; PH0600608 (Hamtic) is PARTIAL."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_graphml = output_dir / f"{municipality_pcode}_phase_b_enriched.graphml"
    out_gpkg = output_dir / f"{municipality_pcode}_phase_b_enriched.gpkg"

    if out_graphml.exists() and not force:
        raise FileExistsError(
            f"Phase B outputs already exist in {output_dir}. Use force=True to overwrite."
        )

    # ── 1. Load graph and edges ─────────────────────────────────────────────
    logger.info("Loading graph from %s", graph_path)
    G = read_graphml(graph_path)

    logger.info("Loading edges GeoPackage from %s", edges_gpkg)
    edges_utm = gpd.read_file(edges_gpkg, layer="edges")

    # ── 2. Compute WCC ids ──────────────────────────────────────────────────
    logger.info("Computing weakly connected components ...")
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    node_wcc: dict[int, int] = {}
    for wcc_id, component in enumerate(wccs):
        for n in component:
            node_wcc[n] = wcc_id

    # ── 3. Load JRC rasters fully into memory ──────────────────────────────
    logger.info("Loading JRC depth raster: %s", depth_raster)
    with rasterio.open(depth_raster) as src:
        depth_arr = src.read().astype(np.float32)  # (5, H, W)
        depth_transform = src.transform

    logger.info("Loading JRC category raster: %s", cat_raster)
    with rasterio.open(cat_raster) as src:
        cat_arr = src.read().astype(np.float32)  # (5, H, W)
        cat_transform = src.transform

    # ── 4. Prepare reprojector UTM→WGS84 ──────────────────────────────────
    # depth_crs should be EPSG:4326; verify or reproject anyway
    edge_crs = edges_utm.crs
    to_4326 = Transformer.from_crs(edge_crs, "EPSG:4326", always_xy=True)

    def _reproject_line(geom: LineString) -> LineString:
        xs, ys = geom.xy
        lons, lats = to_4326.transform(xs, ys)
        return LineString(zip(lons, lats, strict=True))

    # ── 5. Load optional layers ────────────────────────────────────────────
    dem_src: rasterio.DatasetReader | None = None
    if dem_path is not None and dem_path.exists():
        logger.info("Opening DEM: %s", dem_path)
        dem_src = rasterio.open(dem_path)
    else:
        logger.warning("DEM not available — terrain evidence will be NaN.")

    waterway_gdf: gpd.GeoDataFrame | None = None
    if waterway_path is not None and waterway_path.exists():
        logger.info("Loading waterways: %s", waterway_path)
        waterway_gdf = gpd.read_file(waterway_path)
    else:
        logger.warning("Waterway GeoPackage not available — waterway evidence will be NaN.")

    # ── 6. Bulk waterway attribution ────────────────────────────────────────
    if waterway_gdf is not None:
        logger.info("Computing waterway distances and crossings ...")
        ww_df = _waterway_attrs(edges_utm, waterway_gdf)
    else:
        ww_df = None

    # ── 7. Per-edge attribution ─────────────────────────────────────────────
    logger.info("Attributing %d directed edges ...", len(edges_utm))

    # Build lookup from (u_node, v_node, edge_key) → edge row
    edge_lookup: dict[tuple[int, int, int], int] = {}
    for idx, row in edges_utm.iterrows():
        key = (int(row["u_node"]), int(row["v_node"]), int(row["edge_key"]))
        edge_lookup[key] = idx

    phase_b_attrs: dict[tuple[int, int, int], dict[str, Any]] = {}

    for idx, erow in edges_utm.iterrows():
        geom_utm = erow.geometry
        if geom_utm is None or geom_utm.is_empty:
            continue

        length_m = float(erow.get("length_m") or 0.0)
        u = int(erow["u_node"])
        v = int(erow["v_node"])
        k = int(erow["edge_key"])
        key = (u, v, k)

        # JRC: reproject geometry and sample pixels
        geom_4326 = _reproject_line(geom_utm)
        hits = _pixel_hits(
            geom_4326,
            length_m,
            depth_arr,
            cat_arr,
            depth_transform,
            cat_transform,
        )

        rp10_attr = _jrc_rp_attr(hits, length_m, rp="RP10")
        rp20_attr = _jrc_rp_attr(hits, length_m, rp="RP20")
        rp100_attr = _jrc_rp_attr(hits, length_m, rp="RP100")

        # Terrain
        t_attr = _terrain_attr(geom_utm, dem_src, length_m) if dem_src is not None else None

        # Network
        wcc_id = node_wcc.get(u, -1)

        attrs: dict[str, Any] = {}
        attrs.update(_jrc_attrs_to_dict(rp10_attr, rp20_attr, rp100_attr))
        attrs.update(_terrain_to_dict(t_attr))

        # Waterway (from bulk DataFrame)
        if ww_df is not None and idx in ww_df.index:
            wr = ww_df.loc[idx]
            attrs["waterway_nearest_dist_m"] = wr["waterway_nearest_dist_m"]
            attrs["waterway_crossing"] = int(bool(wr["waterway_crossing"]))
            attrs["waterway_nearest_name"] = wr["waterway_nearest_name"]
            attrs["waterway_nearest_type"] = wr["waterway_nearest_type"]
        else:
            attrs["waterway_nearest_dist_m"] = float("nan")
            attrs["waterway_crossing"] = 0
            attrs["waterway_nearest_name"] = None
            attrs["waterway_nearest_type"] = None

        attrs["network_wcc_id"] = wcc_id

        phase_b_attrs[key] = attrs

    if dem_src is not None:
        dem_src.close()

    # ── 8. Attach to graph ─────────────────────────────────────────────────
    logger.info("Attaching Phase B attributes to graph ...")
    unmatched = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        key = (int(u), int(v), int(k))
        if key in phase_b_attrs:
            data.update(phase_b_attrs[key])
        else:
            unmatched += 1

    if unmatched:
        logger.warning("%d graph edges had no matching GeoPackage row.", unmatched)

    # ── 9. Write enriched outputs ──────────────────────────────────────────
    logger.info("Writing enriched GraphML: %s", out_graphml)
    write_graphml(G, out_graphml)

    logger.info("Writing enriched GeoPackage: %s", out_gpkg)
    # Build enriched GeoDataFrame from edges_utm + phase_b_attrs so that all
    # Phase B columns (jrc_*, terrain_*, waterway_*, network_*) are included.
    enriched_rows = []
    for _idx, erow in edges_utm.iterrows():
        key = (int(erow["u_node"]), int(erow["v_node"]), int(erow["edge_key"]))
        rec = dict(erow)
        rec.update(phase_b_attrs.get(key, {}))
        enriched_rows.append(rec)
    enriched_gdf = gpd.GeoDataFrame(enriched_rows, geometry="geometry", crs=edges_utm.crs)
    enriched_gdf = enriched_gdf.sort_values(["osm_id", "edge_seq"], ignore_index=True)
    enriched_gdf.to_file(out_gpkg, driver="GPKG", layer="edges")
    logger.info("Wrote enriched GeoPackage: %s (%d edges)", out_gpkg, len(enriched_gdf))

    sha_graphml = _sha256(out_graphml)
    sha_gpkg = _sha256(out_gpkg)

    # ── 10. Compute summary statistics ─────────────────────────────────────

    # Directed-edge exposure (both directions counted independently)
    n_exp_dir10 = sum(1 for a in phase_b_attrs.values() if a.get("jrc_rp10_exposed_m", 0) > 0)
    n_exp_dir20 = sum(1 for a in phase_b_attrs.values() if a.get("jrc_rp20_exposed_m", 0) > 0)
    n_exp_dir100 = sum(1 for a in phase_b_attrs.values() if a.get("jrc_rp100_exposed_m", 0) > 0)
    tot_dir10 = sum(a.get("jrc_rp10_exposed_m", 0) for a in phase_b_attrs.values())
    tot_dir20 = sum(a.get("jrc_rp20_exposed_m", 0) for a in phase_b_attrs.values())
    tot_dir100 = sum(a.get("jrc_rp100_exposed_m", 0) for a in phase_b_attrs.values())
    n_crossings = sum(1 for a in phase_b_attrs.values() if a.get("waterway_crossing", 0))

    # Build edge-key → (osm_id, edge_seq) mapping for deduplication
    edge_to_phys: dict[tuple[int, int, int], tuple[Any, Any]] = {}
    for _erow_idx, erow in edges_utm.iterrows():
        key = (int(erow["u_node"]), int(erow["v_node"]), int(erow["edge_key"]))
        edge_to_phys[key] = (erow.get("osm_id"), erow.get("edge_seq"))

    # Physical-segment exposure: per (osm_id, edge_seq) keep the max exposed_m
    # across directed edges of the same physical segment.  For geometrically
    # identical bidirectional pairs the values will be equal; max() is used as
    # a safe aggregation in case of floating-point rounding asymmetry.
    phys_exp10: dict[tuple, float] = {}
    phys_exp20: dict[tuple, float] = {}
    phys_exp100: dict[tuple, float] = {}
    phys_all: set[tuple] = set()

    for edge_key, attrs in phase_b_attrs.items():
        pkey = edge_to_phys.get(edge_key, (None, None))
        phys_all.add(pkey)
        phys_exp10[pkey] = max(phys_exp10.get(pkey, 0.0), attrs.get("jrc_rp10_exposed_m", 0.0))
        phys_exp20[pkey] = max(phys_exp20.get(pkey, 0.0), attrs.get("jrc_rp20_exposed_m", 0.0))
        phys_exp100[pkey] = max(phys_exp100.get(pkey, 0.0), attrs.get("jrc_rp100_exposed_m", 0.0))

    n_physical = len(phys_all)
    n_exp_phys10 = sum(1 for v in phys_exp10.values() if v > 0)
    n_exp_phys20 = sum(1 for v in phys_exp20.values() if v > 0)
    n_exp_phys100 = sum(1 for v in phys_exp100.values() if v > 0)
    tot_phys10 = sum(phys_exp10.values())
    tot_phys20 = sum(phys_exp20.values())
    tot_phys100 = sum(phys_exp100.values())

    result = PhaseBResult(
        municipality_code=municipality_pcode,
        n_directed_edges=len(phase_b_attrs),
        n_physical_segments=n_physical,
        n_exposed_dir_rp10=n_exp_dir10,
        n_exposed_dir_rp20=n_exp_dir20,
        n_exposed_dir_rp100=n_exp_dir100,
        total_exposed_dir_m_rp10=round(tot_dir10, 2),
        total_exposed_dir_m_rp20=round(tot_dir20, 2),
        total_exposed_dir_m_rp100=round(tot_dir100, 2),
        n_exposed_phys_rp10=n_exp_phys10,
        n_exposed_phys_rp20=n_exp_phys20,
        n_exposed_phys_rp100=n_exp_phys100,
        total_exposed_phys_m_rp10=round(tot_phys10, 2),
        total_exposed_phys_m_rp20=round(tot_phys20, 2),
        total_exposed_phys_m_rp100=round(tot_phys100, 2),
        n_waterway_crossings=n_crossings,
        output_graphml=out_graphml,
        output_gpkg=out_gpkg,
        sha256_graphml=sha_graphml,
        sha256_gpkg=sha_gpkg,
    )

    logger.info(
        "Phase B complete: %d edges attributed, "
        "RP10 exposed dir=%d (%.1f m) phys=%d (%.1f m), "
        "RP100 exposed dir=%d (%.1f m) phys=%d (%.1f m), crossings=%d",
        result.n_directed_edges,
        n_exp_dir10,
        tot_dir10,
        n_exp_phys10,
        tot_phys10,
        n_exp_dir100,
        tot_dir100,
        n_exp_phys100,
        tot_phys100,
        n_crossings,
    )
    return result
