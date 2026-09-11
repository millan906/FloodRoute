"""Stage 5B: JRC flood hazard attribution for the PH0600613 road graph.

Attributes each directed edge with four evidence layers, kept under separate
prefixes to prevent conflation:

  jrc_rp10_*  / jrc_rp20_*  / jrc_rp100_*   — flood depth evidence per RP
  terrain_*                                   — DEM elevation / slope
  waterway_*                                  — OSM waterway proximity / crossing
  network_*                                   — graph topology (WCC id)

Three-state pixel semantics (verified from depth_category band in Phase A):
  outside_domain  cat == NODATA_SENTINEL           (GloFAS has no output)
  modelled_dry    cat == dry_val (0 for PH0600613) (within domain, not flooded)
  flooded         cat ∈ flood_vals {2, 3}          (positive depth in band)

Only non-permanent flooded pixels (permanent_water_class ≠ 1) contribute to
``jrc_rp*_exposed_m``. Permanent-water intersections are recorded separately
as ``jrc_rp*_perm_water_m``.

Interior overlap threshold: MIN_INTERIOR_LEN_M (5.0 m).  Zero-length boundary
touches are excluded.

RP20 domain indicator: derived from RP10 depth_category (domain is constant
across return periods within GloFAS v2.1; RP20-specific category not acquired).

PASSABILITY NOTE
----------------
This module records evidence only.  It does NOT infer passability, closure,
travel speed, or routing decisions.  Phase B outputs are inputs to Stage 6
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

from floodroute.graph.io import read_graphml, write_edges_gpkg, write_graphml
from floodroute.hazard.jrc import MIN_INTERIOR_LEN_M, NODATA_SENTINEL

logger = logging.getLogger("floodroute.hazard.phase_b")

# ── San Jose de Buenavista (PH0600613) constants ───────────────────────────

MUNICIPALITY_PCODE: str = "PH0600613"
_DRY_VAL: int = 0          # depth_category value for modelled-dry in this tile sub-area
_FLOOD_VALS: frozenset[int] = frozenset({2, 3})

# Band indices in depth raster (0-based)
_D_RP10 = 0
_D_RP20 = 1
_D_RP100 = 2
_D_PW = 3   # permanent_water_class
_D_SD = 4   # spurious_depth_category

# Band indices in category raster (0-based)
# 5-band format: RP10_cat, RP20_cat, RP100_cat, perm_water, spurious
_C_RP10 = 0
_C_RP20 = 1
_C_RP100 = 2
_C_PW = 3
_C_SD = 4

# DEM sampling density (points along edge)
_DEM_SAMPLE_PTS: int = 20

# Return periods to attribute
_RETURN_PERIODS: list[str] = ["RP10", "RP20", "RP100"]


# ── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class PixelHit:
    """Interior-overlap record for one JRC pixel touching a directed edge."""

    overlap_m: float          # interior overlap in metres
    depth_rp10: float         # depth (m) or NODATA_SENTINEL
    depth_rp20: float
    depth_rp100: float
    cat_rp10: float           # depth_category value: NODATA_SENTINEL | _DRY_VAL | flood_val
    cat_rp20: float
    cat_rp100: float
    perm_water: float         # 1.0 = permanent water; -1.0 = not permanent
    spurious: float           # spurious_depth_category value; -1 = not flagged


@dataclass
class JrcRpAttr:
    """JRC attribution for one directed edge at one return period."""

    status: str               # no_overlap | outside_domain | modelled_dry | flooded
    depth_max_m: float        # nan if no flood exposure
    depth_wt_mean_m: float    # overlap-length-weighted mean depth; nan if no exposure
    exposed_m: float          # interior overlap with non-permanent flooded pixels (m)
    exposed_pct: float        # exposed_m / edge_length_m * 100
    perm_water_m: float       # interior overlap with permanent-water pixels (m)
    spurious: bool            # any spurious pixel in the interior overlap set
    sample_n: int             # pixel hits with interior overlap >= MIN_INTERIOR_LEN_M


@dataclass
class TerrainAttr:
    """DEM-derived terrain evidence for one directed edge."""

    elev_min_m: float
    elev_mean_m: float
    elev_max_m: float
    elev_change_m: float      # max - min elevation along edge
    slope_pct: float          # elev_change_m / length_m * 100


@dataclass
class WaterwayAttr:
    """OSM waterway proximity evidence for one directed edge."""

    nearest_dist_m: float
    crossing: bool
    nearest_name: str | None  # OSM name attribute of nearest waterway
    nearest_type: str | None  # OSM waterway tag (river, stream, canal, …)


@dataclass
class PhaseBResult:
    """Summary of the completed Phase B attribution run."""

    municipality_code: str
    n_directed_edges: int
    n_physical_segments: int
    n_exposed_rp10: int       # edges with jrc_rp10_exposed_m > 0
    n_exposed_rp20: int
    n_exposed_rp100: int
    total_exposed_m_rp10: float
    total_exposed_m_rp20: float
    total_exposed_m_rp100: float
    n_waterway_crossings: int
    output_graphml: Path
    output_gpkg: Path
    sha256_graphml: str
    sha256_gpkg: str
    total_exposed_dir_m_rp10: float = 0.0   # sum over directed edges
    total_exposed_phys_m_rp10: float = 0.0  # sum over physical segments (deduped)
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
        Full category raster array, shape (4, H, W), or None.
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

            # Category raster may use a different transform / size
            _NOT_FLOODED_CAT = -1.0
            c10 = _NOT_FLOODED_CAT
            c20 = _NOT_FLOODED_CAT
            c100 = _NOT_FLOODED_CAT
            if cat_arr is not None and cat_transform is not None:
                cat_H, cat_W = cat_arr.shape[1], cat_arr.shape[2]
                cr, cc = rasterio.transform.rowcol(cat_transform, px_x, px_y)
                if 0 <= cr < cat_H and 0 <= cc < cat_W:
                    c10 = float(cat_arr[_C_RP10, cr, cc])
                    # RP20 band only present in 5-band format; fall back gracefully
                    if cat_arr.shape[0] > _C_RP20:
                        c20 = float(cat_arr[_C_RP20, cr, cc])
                    if cat_arr.shape[0] > _C_RP100:
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

    Uses three-state semantics.  RP20 borrows domain indicator from RP10
    category band (model domain is constant across return periods).
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

    statuses: list[str] = []

    for h in hits:
        # Determine pixel state for this RP
        if rp == "RP10":
            depth = h.depth_rp10
            cat = h.cat_rp10
            is_flooded = float(cat) in _FLOOD_VALS
            is_outside = float(depth) == NODATA_SENTINEL or float(cat) == NODATA_SENTINEL
        elif rp == "RP20":
            depth = h.depth_rp20
            cat = h.cat_rp20  # use actual RP20 category band
            is_flooded = float(cat) in _FLOOD_VALS
            is_outside = float(depth) == NODATA_SENTINEL or float(cat) == NODATA_SENTINEL
        else:  # RP100
            depth = h.depth_rp100
            cat = h.cat_rp100
            is_flooded = float(cat) in _FLOOD_VALS
            is_outside = float(depth) == NODATA_SENTINEL or float(cat) == NODATA_SENTINEL

        is_dry = (not is_flooded) and (not is_outside)

        if is_outside:
            statuses.append("outside_domain")
        elif is_dry:
            statuses.append("modelled_dry")
        else:
            statuses.append("flooded")

        if is_flooded:
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

    # Determine edge-level status
    unique = set(statuses)
    if unique == {"outside_domain"}:
        status = "outside_domain"
    elif "flooded" in unique:
        status = "flooded"
    elif "modelled_dry" in unique:
        status = "modelled_dry"
    else:
        status = "outside_domain"

    total_overlap = sum(h.overlap_m for h in hits)
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
    pts = [
        edge_geom_utm.interpolate(t, normalized=True)
        for t in np.linspace(0, 1, n_pts)
    ]
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
        4-band JRC category GeoTIFF (RP10_cat/RP100_cat/perm_water/spurious).
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
        depth_arr = src.read().astype(np.float32)   # (5, H, W)
        depth_transform = src.transform
        depth_crs = src.crs

    logger.info("Loading JRC category raster: %s", cat_raster)
    with rasterio.open(cat_raster) as src:
        cat_arr = src.read().astype(np.float32)     # (4, H, W)
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
    write_edges_gpkg(G, out_gpkg)

    sha_graphml = _sha256(out_graphml)
    sha_gpkg = _sha256(out_gpkg)

    # ── 10. Compute summary statistics ─────────────────────────────────────
    n_exposed_rp10 = sum(
        1 for a in phase_b_attrs.values() if a.get("jrc_rp10_exposed_m", 0) > 0
    )
    n_exposed_rp20 = sum(
        1 for a in phase_b_attrs.values() if a.get("jrc_rp20_exposed_m", 0) > 0
    )
    n_exposed_rp100 = sum(
        1 for a in phase_b_attrs.values() if a.get("jrc_rp100_exposed_m", 0) > 0
    )
    total_m_rp10 = sum(a.get("jrc_rp10_exposed_m", 0) for a in phase_b_attrs.values())
    total_m_rp20 = sum(a.get("jrc_rp20_exposed_m", 0) for a in phase_b_attrs.values())
    total_m_rp100 = sum(a.get("jrc_rp100_exposed_m", 0) for a in phase_b_attrs.values())
    n_crossings = sum(
        1 for a in phase_b_attrs.values() if a.get("waterway_crossing", 0)
    )

    # Physical segment count and physical exposure (deduplicate bidirectional pairs)
    phys_keys: set[tuple[Any, Any]] = set()
    phys_rp10: dict[tuple[Any, Any], float] = {}
    for _erow_idx, erow in edges_utm.iterrows():
        pk = (erow.get("osm_id"), erow.get("edge_seq"))
        phys_keys.add(pk)
        u2 = int(erow["u_node"])
        v2 = int(erow["v_node"])
        k2 = int(erow["edge_key"])
        exp = phase_b_attrs.get((u2, v2, k2), {}).get("jrc_rp10_exposed_m", 0.0)
        if exp > phys_rp10.get(pk, 0.0):
            phys_rp10[pk] = exp
    n_physical = len(phys_keys)
    total_phys_m_rp10 = sum(phys_rp10.values())

    result = PhaseBResult(
        municipality_code=municipality_pcode,
        n_directed_edges=len(phase_b_attrs),
        n_physical_segments=n_physical,
        n_exposed_rp10=n_exposed_rp10,
        n_exposed_rp20=n_exposed_rp20,
        n_exposed_rp100=n_exposed_rp100,
        total_exposed_m_rp10=round(total_m_rp10, 2),
        total_exposed_m_rp20=round(total_m_rp20, 2),
        total_exposed_m_rp100=round(total_m_rp100, 2),
        total_exposed_dir_m_rp10=round(total_m_rp10, 2),
        total_exposed_phys_m_rp10=round(total_phys_m_rp10, 2),
        n_waterway_crossings=n_crossings,
        output_graphml=out_graphml,
        output_gpkg=out_gpkg,
        sha256_graphml=sha_graphml,
        sha256_gpkg=sha_gpkg,
    )

    logger.info(
        "Phase B complete: %d edges attributed, RP10 exposed=%d (%.1f m), "
        "RP100 exposed=%d (%.1f m), crossings=%d",
        result.n_directed_edges,
        n_exposed_rp10,
        total_m_rp10,
        n_exposed_rp100,
        total_m_rp100,
        n_crossings,
    )
    return result
