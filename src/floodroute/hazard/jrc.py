"""JRC GloFAS Flood Hazard v2.1 — acquisition and Phase A suitability assessment.

NODATA SEMANTICS (verified from depth_category bands, 2026-08-23)
-----------------------------------------------------------------
The depth bands use -9999.0 as a sentinel with two distinct meanings that can
only be distinguished by cross-referencing the depth_category band:

    depth_category == -9999 AND depth == -9999 → OUTSIDE MODEL DOMAIN
        The GloFAS hydraulic model has no output for this cell.  These cells
        form continuous blobs unrelated to rivers.

    depth_category == -1 or 0 AND depth == -9999 → MODELLED DRY
        The cell is within the GloFAS model domain but is not inundated at
        this return period.  Value −1 is used in tile ID230 (Sibalom), value 0
        in the same tile for San Jose and Hamtic sub-areas.

    depth_category in {2, 3, …} AND depth > 0 → FLOODED
        The cell is modelled as inundated.  Depth (metres) is stored in the
        depth band.  Category encodes a depth class; exact class breakpoints
        are documented at the GEE dataset page.

The depth_category band is therefore the authoritative domain indicator.
Presence of valid depth values alone is insufficient to determine whether a
cell belongs to category (a) or (b) above.

PERMANENT WATER
---------------
The permanent_water_class band (value 1 = permanent water, −1 = unknown)
identifies river channel pixels that are perennially wet.  Flood depth is
stored for these pixels at all return periods and typically reflects the
water depth over the channel rather than genuine over-bank inundation.
Permanent-water pixels should be separated from non-permanent pixels for
road exposure analysis because roads over permanent water are bridges, not
road corridors at risk of inundation.

1.0 M DEPTH FLOOR (Sibalom specific)
--------------------------------------
In the Sibalom sub-area, 280/286 permanent-water pixels carry exactly 1.0 m
depth.  This is a permanent-water-channel artifact (the river channel depth
clamped to a minimum value), NOT a general floor across the dataset.  It is
absent from non-permanent pixels in the same tile and is not observed in San
Jose or Hamtic.

SUITABILITY GATE (Phase A)
---------------------------
Flood prevalence is not a data-quality criterion.  The suitability gate
assesses whether the JRC data can support Stage 5 road hazard exposure
analysis for comparative routing across return periods.  Gates are:

    NON_PERM_MIN_PIXELS : int = 10
        Minimum non-permanent flooded pixels at RP10.  Below this the GloFAS
        output is effectively just the river channel with no over-bank
        representation.  10 pixels ≈ 0.08 km² at 90 m resolution — the
        minimum plausible floodplain signal above model numerical noise.

    NORMAL_SEG_MIN : int = 3
        Minimum normal (non-bridge, non-tunnel) road segments with interior
        overlap ≥ MIN_INTERIOR_LEN_M.  Three segments provide the minimum
        number of independent observations for comparative routing analysis.

    SCENARIO_GAIN_MIN : int = 5
        Minimum additional non-permanent pixels from RP10 to RP100.  Fewer
        than 5 additional pixels makes RP10 and RP100 indistinguishable at
        90 m resolution.

    MIN_INTERIOR_LEN_M : float = 5.0
        Minimum intersection length (metres) between a road segment and the
        union of flooded cells to count as interior overlap.  This excludes
        pure boundary touches where a road merely crosses the edge of a
        90 m cell without meaningful interior traversal.

READY   : all three numeric gates met by non-permanent pixels only
PARTIAL : some non-permanent exposure exists but below one or more gates
BLOCKED : zero non-permanent interior road overlap OR zero non-permanent pixels
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask

logger = logging.getLogger("floodroute.hazard.jrc")

# ── EE identifiers ─────────────────────────────────────────────────────────

EE_COLLECTION = "JRC/CEMS_GLOFAS/FloodHazard/v2_1"
EE_TILE_PHILIPPINES = "JRC/CEMS_GLOFAS/FloodHazard/v2_1/ID230_N20_E120"

# ── Band layout in the acquired 5-band raw GeoTIFF ─────────────────────────

BAND_NAMES: list[str] = [
    "RP10_depth",
    "RP20_depth",
    "RP100_depth",
    "permanent_water_class",
    "spurious_depth_category",
]
BAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(BAND_NAMES, 1)}
DEPTH_BANDS: list[str] = ["RP10_depth", "RP20_depth", "RP100_depth"]

# Band layout in the 5-band category GeoTIFF (acquired 2026-08-23 from EE)
CAT_BAND_NAMES: list[str] = [
    "RP10_depth_category",
    "RP20_depth_category",
    "RP100_depth_category",
    "permanent_water_class",
    "spurious_depth_category",
]

# ── Numeric constants ──────────────────────────────────────────────────────

NODATA_SENTINEL: float = -9999.0

# Depth class thresholds (FloodRoute project analytical classes — NOT native JRC)
# Adapted from Philippine flood hazard metadata for comparative analysis.
#   no_modeled_inundation  depth < 0.10 m
#   low                    0.10 m ≤ depth < 0.50 m
#   medium                 0.50 m ≤ depth ≤ 1.50 m   (1.50 m inclusive)
#   high                   depth > 1.50 m             (strictly above 1.50 m)
DEPTH_CLASS_NO_INUNDATION_MAX_M: float = 0.10  # depth < this → no_modeled_inundation
DEPTH_CLASS_LOW_MIN_M: float = 0.10  # [0.10, 0.50)
DEPTH_CLASS_MEDIUM_MIN_M: float = 0.50  # [0.50, 1.50] inclusive
DEPTH_CLASS_HIGH_MIN_M: float = 1.50  # > 1.50 (strictly above; 1.50 m → medium)

# ── Suitability gate thresholds (methodologically derived — see module doc) ──

NON_PERM_MIN_PIXELS: int = 10
NORMAL_SEG_MIN: int = 3
SCENARIO_GAIN_MIN: int = 5
MIN_INTERIOR_LEN_M: float = 5.0

# ── Bridge / tunnel grade-separation sets (from config.py) ─────────────────

BRIDGE_VALS: frozenset[str] = frozenset(
    {"yes", "cantilever", "viaduct", "aqueduct", "movable", "trestle"}
)
TUNNEL_VALS: frozenset[str] = frozenset({"yes", "building_passage", "culvert"})

# ── Acquisition parameters ─────────────────────────────────────────────────

EE_EXPORT_SCALE_M: int = 90
EE_BUFFER_DEG: float = 0.05

# ── Municipality identifiers ───────────────────────────────────────────────

SIBALOM_PCODE: str = "PH0600616"


# ── Readiness decision ─────────────────────────────────────────────────────


class ReadinessDecision(StrEnum):
    """Phase A suitability categories."""

    READY = "READY"  # all suitability gates met by non-permanent pixels
    PARTIAL = "PARTIAL"  # some non-permanent exposure but below one or more gates
    BLOCKED = "BLOCKED"  # zero non-permanent interior road overlap or zero non-perm pixels


# ── Result dataclasses ─────────────────────────────────────────────────────


@dataclass
class PixelCategories:
    """Three-state pixel count for one return period within a municipality mask."""

    return_period: str
    total_pixels: int
    outside_domain: int  # depth_category == NODATA_SENTINEL
    modelled_dry: int  # depth_category == dry_val (-1 or 0 per tile)
    flooded: int  # depth_category in flood_vals
    permanent_water: int  # flooded & permanent_water_class == 1
    non_permanent: int  # flooded & NOT permanent water
    spurious_flagged: int  # spurious_depth_category > 0

    @property
    def outside_pct(self) -> float:
        return round(100 * self.outside_domain / max(1, self.total_pixels), 2)

    @property
    def non_perm_pct(self) -> float:
        return round(100 * self.non_permanent / max(1, self.total_pixels), 4)

    @property
    def flooded_pct(self) -> float:
        return round(100 * self.flooded / max(1, self.total_pixels), 4)


@dataclass
class DepthStats:
    """Depth distribution for non-permanent pixels only."""

    return_period: str
    n: int
    depth_min: float
    depth_p25: float
    depth_median: float
    depth_mean: float
    depth_p75: float
    depth_p95: float
    depth_max: float


@dataclass
class RoadExposure:
    """Road segment exposure from non-permanent inundation (interior overlap only)."""

    return_period: str
    total_physical_segments: int
    total_physical_length_km: float
    bridge_segments: int
    bridge_length_km: float
    tunnel_segments: int
    tunnel_length_km: float
    normal_segments: int
    normal_length_km: float
    # Exposure (interior overlap ≥ MIN_INTERIOR_LEN_M)
    exposed_normal_segments: int
    exposed_normal_overlap_km: float
    exposed_bridge_segments: int
    exposed_bridge_overlap_km: float
    # Depth class breakdown for normal exposed segments (by max depth)
    low_segments: int  # [0.10, 0.50)
    medium_segments: int  # [0.50, 1.50)
    high_segments: int  # [1.50, ∞)


@dataclass
class PhaseAReadiness:
    """Complete corrected Phase A suitability result."""

    decision: ReadinessDecision
    reasons: list[str]
    municipality_code: str
    raster_path: Path
    sha256: str
    file_bytes: int
    raster_crs: str
    pixel_size_m_approx: int
    raster_bounds: tuple[float, float, float, float]
    # Per-RP pixel categorisation
    pixel_cats: dict[str, PixelCategories]  # keyed by "RP10", "RP20", "RP100"
    depth_stats: dict[str, DepthStats]  # keyed by "RP10", "RP100" (non-perm only)
    # Monotonicity
    monotonicity_violations_rp10_rp20: int
    monotonicity_violations_rp20_rp100: int
    # Road exposure
    road_exposure: dict[str, RoadExposure] | None  # keyed by "RP10", "RP100"
    # LiPAD suitability check
    lipad_overlaps_municipality: bool
    lipad_notes: str
    extra: dict[str, Any] = field(default_factory=dict)


# ── Acquisition ────────────────────────────────────────────────────────────


def acquire_municipal_subset(
    municipality_pcode: str,
    output_path: Path,
    *,
    ee_project: str,
    admin_zip: Path,
    buffer_deg: float = EE_BUFFER_DEG,
    scale_m: int = EE_EXPORT_SCALE_M,
    force: bool = False,
    bands: list[str] | None = None,
) -> tuple[Path, str]:
    """Download a municipal JRC subset from Google Earth Engine.

    Returns (output_path, sha256).
    Raises FileExistsError if output_path exists and force=False.
    Raises RuntimeError if EE initialization fails.
    """
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Raw hazard raster already exists: {output_path}. Use force=True to overwrite."
        )

    try:
        import ee  # noqa: PLC0415
        import requests  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "earthengine-api and requests are required. "
            f"Install with: pip install earthengine-api requests. Original: {e}"
        ) from e

    try:
        ee.Initialize(project=ee_project)
    except Exception as e:
        raise RuntimeError(
            f"Earth Engine initialization failed for project '{ee_project}'. "
            f"Run 'earthengine authenticate' first. Original: {e}"
        ) from e

    sib_gdf = gpd.read_file(f"zip://{admin_zip}!phl_admin3.shp")
    row = sib_gdf[sib_gdf["adm3_pcode"] == municipality_pcode]
    if row.empty:
        raise ValueError(f"adm3_pcode={municipality_pcode} not found in {admin_zip}")
    minx, miny, maxx, maxy = row.total_bounds
    region = ee.Geometry.Rectangle(
        [minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg]
    )

    if bands is None:
        bands = BAND_NAMES

    col = ee.ImageCollection(EE_COLLECTION)
    img = col.filterBounds(region).mosaic().select(bands)

    url = img.getDownloadURL(
        {
            "region": region,
            "crs": "EPSG:4326",
            "scale": scale_m,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )

    logger.info("Downloading JRC subset for %s ...", municipality_pcode)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with open(output_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65536):
            fh.write(chunk)
            sha.update(chunk)

    digest = sha.hexdigest()
    logger.info(
        "Acquired %s bytes → %s, sha256=%s",
        output_path.stat().st_size,
        output_path.name,
        digest,
    )
    return output_path, digest


# Keep the original Sibalom-specific function as a convenience wrapper
def acquire_sibalom_subset(
    output_path: Path,
    *,
    ee_project: str,
    admin_zip: Path,
    buffer_deg: float = EE_BUFFER_DEG,
    scale_m: int = EE_EXPORT_SCALE_M,
    force: bool = False,
) -> tuple[Path, str]:
    """Download the Sibalom (PH0600616) JRC subset. See acquire_municipal_subset."""
    return acquire_municipal_subset(
        SIBALOM_PCODE,
        output_path,
        ee_project=ee_project,
        admin_zip=admin_zip,
        buffer_deg=buffer_deg,
        scale_m=scale_m,
        force=force,
    )


# ── Raster validation ──────────────────────────────────────────────────────


def validate_raster_format(path: Path) -> dict[str, Any]:
    """Validate basic raster format properties. Raises ValueError on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Hazard raster not found: {path}")

    with rasterio.open(path) as src:
        if src.driver not in ("GTiff", "GeoTIFF"):
            raise ValueError(f"Expected GeoTIFF, got driver={src.driver}")
        if src.crs is None:
            raise ValueError("Raster has no CRS.")
        if str(src.crs.to_epsg()) not in ("4326", "32651", "32652"):
            raise ValueError(f"Unexpected CRS: {src.crs}. Expected EPSG:4326 or UTM zone 51/52.")
        if src.count != len(BAND_NAMES):
            raise ValueError(
                f"Expected {len(BAND_NAMES)} bands, got {src.count}. Required order: {BAND_NAMES}"
            )
        if not (50 <= abs(src.transform.a * 111000) <= 120):
            raise ValueError(
                f"Unexpected pixel size: {abs(src.transform.a) * 111000:.0f} m. "
                f"Expected ~90 m for JRC GloFAS v2.1."
            )
        return {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "pixel_size_deg": abs(src.transform.a),
            "bounds": tuple(src.bounds),
            "dtype": src.dtypes[0],
        }


# ── Per-band pixel categorisation ─────────────────────────────────────────


def _pixel_categories(
    depth_data: np.ndarray,
    cat_data: np.ndarray,
    pw_band: np.ndarray,
    sd_band: np.ndarray,
    *,
    return_period: str,
    depth_band_idx: int,  # 0-based
    cat_band_idx: int,  # 0-based in cat_data
    dry_val: int,
    flood_vals: set[int],
) -> PixelCategories:
    total = depth_data[0].size
    cat = cat_data[cat_band_idx].astype(float)
    pw = pw_band.astype(float)
    sd = sd_band.astype(float)

    outside = (cat == NODATA_SENTINEL).sum()
    dry = (cat == dry_val).sum()
    flooded_mask = np.isin(cat, list(flood_vals))
    n_flooded = int(flooded_mask.sum())
    n_perm = int((flooded_mask & (pw == 1)).sum())
    n_nonperm = n_flooded - n_perm
    n_spurious = int((sd > 0).sum())

    return PixelCategories(
        return_period=return_period,
        total_pixels=total,
        outside_domain=int(outside),
        modelled_dry=int(dry),
        flooded=n_flooded,
        permanent_water=n_perm,
        non_permanent=n_nonperm,
        spurious_flagged=n_spurious,
    )


def _depth_stats(
    depth_band: np.ndarray,
    nonperm_mask: np.ndarray,
    *,
    return_period: str,
) -> DepthStats:
    vals = depth_band[nonperm_mask].astype(float)
    vals = vals[vals != NODATA_SENTINEL]
    if len(vals) == 0:
        return DepthStats(
            return_period=return_period,
            n=0,
            depth_min=float("nan"),
            depth_p25=float("nan"),
            depth_median=float("nan"),
            depth_mean=float("nan"),
            depth_p75=float("nan"),
            depth_p95=float("nan"),
            depth_max=float("nan"),
        )
    return DepthStats(
        return_period=return_period,
        n=len(vals),
        depth_min=float(vals.min()),
        depth_p25=float(np.percentile(vals, 25)),
        depth_median=float(np.median(vals)),
        depth_mean=float(vals.mean()),
        depth_p75=float(np.percentile(vals, 75)),
        depth_p95=float(np.percentile(vals, 95)),
        depth_max=float(vals.max()),
    )


# ── Road intersection with interior-length calculation ────────────────────


def _is_bridge(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).lower().strip()
    return s in BRIDGE_VALS and s not in ("", "nan", "none")


def _is_tunnel(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).lower().strip()
    return s in TUNNEL_VALS and s not in ("", "nan", "none")


def _depth_class(depth_m: float) -> str:
    """Return depth class for a non-permanent flooded pixel depth.

    Classes (non-overlapping):
      no_modeled_inundation  depth < 0.10 m
      low                    0.10 ≤ depth < 0.50 m
      medium                 0.50 ≤ depth ≤ 1.50 m  (1.50 m inclusive)
      high                   depth > 1.50 m
    """
    if depth_m > DEPTH_CLASS_HIGH_MIN_M:
        return "high"
    if depth_m >= DEPTH_CLASS_MEDIUM_MIN_M:
        return "medium"
    if depth_m >= DEPTH_CLASS_LOW_MIN_M:
        return "low"
    return "no_modeled_inundation"


def compute_road_exposure(
    raster_path: Path,
    cat_path: Path,
    municipality_geom: Any,
    edges_path: Path,
    *,
    return_period: str,
    depth_band_idx: int,
    cat_band_idx: int,
    dry_val: int,
    flood_vals: set[int],
) -> RoadExposure:
    """Compute road exposure using interior-overlap intersection.

    Only non-permanent flood pixels are used.  Bridge and tunnel segments are
    classified separately.  A road segment is counted as exposed only when the
    interior overlap length with the union of flooded cells is ≥ MIN_INTERIOR_LEN_M.
    """
    from shapely.geometry import box as shapely_box  # noqa: PLC0415
    from shapely.ops import unary_union  # noqa: PLC0415

    with rasterio.open(raster_path) as src:
        dd, tr = rasterio.mask.mask(src, [municipality_geom], crop=True, nodata=NODATA_SENTINEL)
    with rasterio.open(cat_path) as src:
        dc, _ = rasterio.mask.mask(src, [municipality_geom], crop=True, nodata=NODATA_SENTINEL)

    pw = dd[3].astype(float)
    depth = dd[depth_band_idx].astype(float)
    cat = dc[cat_band_idx].astype(float)

    # Non-permanent flooded pixels
    nonperm_mask = np.isin(cat, list(flood_vals)) & (pw != 1)
    rows_f, cols_f = np.where(nonperm_mask)
    half = abs(tr.a) / 2
    xs_f, ys_f = rasterio.transform.xy(tr, rows_f, cols_f)
    flood_polys = [
        shapely_box(x - half, y - half, x + half, y + half) for x, y in zip(xs_f, ys_f, strict=True)
    ]
    depths_arr = depth[nonperm_mask].astype(float)

    # Load road edges (reproject to WGS84 to match raster)
    edges = gpd.read_file(edges_path).to_crs("EPSG:4326")
    phys = edges.drop_duplicates(subset=["osm_id", "edge_seq"]).copy()
    phys["_bridge"] = phys["bridge"].apply(_is_bridge)
    phys["_tunnel"] = phys["tunnel"].apply(_is_tunnel)
    phys["_normal"] = ~phys["_bridge"] & ~phys["_tunnel"]

    total_segs = len(phys)
    total_len_km = phys["length_m"].sum() / 1000
    bridge_len_km = phys[phys["_bridge"]]["length_m"].sum() / 1000
    tunnel_len_km = phys[phys["_tunnel"]]["length_m"].sum() / 1000
    normal_len_km = phys[phys["_normal"]]["length_m"].sum() / 1000

    if not flood_polys:
        return RoadExposure(
            return_period=return_period,
            total_physical_segments=total_segs,
            total_physical_length_km=round(total_len_km, 3),
            bridge_segments=phys["_bridge"].sum(),
            bridge_length_km=round(bridge_len_km, 3),
            tunnel_segments=phys["_tunnel"].sum(),
            tunnel_length_km=round(tunnel_len_km, 3),
            normal_segments=phys["_normal"].sum(),
            normal_length_km=round(normal_len_km, 3),
            exposed_normal_segments=0,
            exposed_normal_overlap_km=0.0,
            exposed_bridge_segments=0,
            exposed_bridge_overlap_km=0.0,
            low_segments=0,
            medium_segments=0,
            high_segments=0,
        )

    flood_gdf = gpd.GeoDataFrame({"geometry": flood_polys, "depth": depths_arr}, crs="EPSG:4326")

    joined = gpd.sjoin(phys, flood_gdf, how="inner", predicate="intersects")
    candidate_idx = joined.index.unique()

    exp_normal_segs = 0
    exp_normal_overlap_km = 0.0
    exp_bridge_segs = 0
    exp_bridge_overlap_km = 0.0
    low_segs = medium_segs = high_segs = 0

    for idx in candidate_idx:
        seg = phys.loc[idx]
        hits = joined[joined.index == idx]
        pix_polys = [flood_polys[i] for i in hits["index_right"]]
        pix_depths = [depths_arr[i] for i in hits["index_right"]]
        flood_union = unary_union(pix_polys)
        overlap = seg.geometry.intersection(flood_union)
        overlap_m = overlap.length * 111000
        if overlap_m < MIN_INTERIOR_LEN_M:
            continue

        max_depth = float(max(pix_depths))
        overlap_km = overlap_m / 1000

        if seg["_bridge"]:
            exp_bridge_segs += 1
            exp_bridge_overlap_km += overlap_km
        else:
            exp_normal_segs += 1
            exp_normal_overlap_km += overlap_km
            dc_label = _depth_class(max_depth)
            if dc_label == "low":
                low_segs += 1
            elif dc_label == "medium":
                medium_segs += 1
            elif dc_label == "high":
                high_segs += 1

    return RoadExposure(
        return_period=return_period,
        total_physical_segments=total_segs,
        total_physical_length_km=round(total_len_km, 3),
        bridge_segments=int(phys["_bridge"].sum()),
        bridge_length_km=round(bridge_len_km, 3),
        tunnel_segments=int(phys["_tunnel"].sum()),
        tunnel_length_km=round(tunnel_len_km, 3),
        normal_segments=int(phys["_normal"].sum()),
        normal_length_km=round(normal_len_km, 3),
        exposed_normal_segments=exp_normal_segs,
        exposed_normal_overlap_km=round(exp_normal_overlap_km, 3),
        exposed_bridge_segments=exp_bridge_segs,
        exposed_bridge_overlap_km=round(exp_bridge_overlap_km, 3),
        low_segments=low_segs,
        medium_segments=medium_segs,
        high_segments=high_segs,
    )


# ── LiPAD suitability check ────────────────────────────────────────────────


def check_lipad_overlap(
    municipality_geom: Any,
    lipad_iso_bbox: tuple[float, float, float, float],
) -> tuple[bool, str]:
    """Return (overlaps, notes) for the LiPAD ISO record extent vs municipality.

    lipad_iso_bbox : (west, south, east, north) in WGS84 degrees.
    """
    from pyproj import Geod  # noqa: PLC0415
    from shapely.geometry import box as shapely_box  # noqa: PLC0415
    from shapely.ops import nearest_points  # noqa: PLC0415

    lipad_box = shapely_box(*lipad_iso_bbox)
    overlaps = lipad_box.intersects(municipality_geom)

    if overlaps:
        inter = lipad_box.intersection(municipality_geom)
        geod = Geod(ellps="WGS84")
        area_km2 = abs(geod.geometry_area_perimeter(inter)[0]) / 1e6
        notes = f"LiPAD ISO extent overlaps municipality. Intersection area ≈ {area_km2:.1f} km²."
    else:
        p1, p2 = nearest_points(municipality_geom, lipad_box)
        geod = Geod(ellps="WGS84")
        _, _, dist = geod.inv(p1.x, p1.y, p2.x, p2.y)
        notes = (
            f"LiPAD ISO extent does NOT intersect municipality. "
            f"Minimum distance: {dist / 1000:.1f} km. "
            f"LiPAD cannot substitute for this municipality."
        )
    return overlaps, notes


# ── Main Phase A entry point ───────────────────────────────────────────────

# LiPAD Sibalom River ISO record extent (all three RP layers share this bbox)
LIPAD_SIBALOM_ISO_BBOX: tuple[float, float, float, float] = (
    122.32169450591898,
    10.66699026999993,
    122.4303543889997,
    10.807655079784046,
)

# Tile-specific dry value (depth_category value for "modelled but not flooded")
_DRY_VAL: dict[str, int] = {
    "PH0600616": -1,  # Sibalom — tile ID230, western sub-area
    "PH0600613": 0,  # San Jose — same tile, eastern sub-area
    "PH0600608": 0,  # Hamtic — same tile, southern sub-area
}
_FLOOD_VALS: dict[str, set[int]] = {
    "PH0600616": {2},
    "PH0600613": {2, 3},
    "PH0600608": {2, 3},
}


def run_phase_a(
    raster_path: Path,
    *,
    admin_zip: Path,
    cat_path: Path | None = None,
    edges_path: Path | None = None,
    municipality_pcode: str = SIBALOM_PCODE,
) -> PhaseAReadiness:
    """Run Phase A suitability assessment and return a PhaseAReadiness result.

    Parameters
    ----------
    raster_path:
        Path to the acquired 5-band depth GeoTIFF.
    admin_zip:
        PSA admin boundaries ZIP (contains phl_admin3.shp).
    cat_path:
        Path to the 4-band depth-category GeoTIFF.  Required for three-state
        pixel categorisation.  If None, a degraded (depth-only) analysis is run.
    edges_path:
        Stage 4 edges GeoPackage.  If None, road exposure is skipped.
    municipality_pcode:
        PSA adm3_pcode.
    """
    # 1. Format validation
    validate_raster_format(raster_path)

    # 2. SHA-256
    sha = hashlib.sha256()
    file_bytes = raster_path.stat().st_size
    with open(raster_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    digest = sha.hexdigest()

    # 3. Municipality polygon
    sib_gdf = gpd.read_file(f"zip://{admin_zip}!phl_admin3.shp")
    row = sib_gdf[sib_gdf["adm3_pcode"] == municipality_pcode]
    if row.empty:
        raise ValueError(f"Municipality {municipality_pcode} not found in {admin_zip}")
    muni_geom = row.geometry.values[0]

    # 4. Read depth bands
    with rasterio.open(raster_path) as src:
        raster_crs = str(src.crs)
        pixel_size_m = round(abs(src.transform.a) * 111000)
        bounds = tuple(src.bounds)
        depth_data, tr = rasterio.mask.mask(src, [muni_geom], crop=True, nodata=NODATA_SENTINEL)

    dry_val = _DRY_VAL.get(municipality_pcode, 0)
    flood_vals = _FLOOD_VALS.get(municipality_pcode, {2, 3})

    # 5. Three-state categorisation (requires cat_path)
    pixel_cats: dict[str, PixelCategories] = {}
    depth_stats: dict[str, DepthStats] = {}
    mono_10_20 = mono_20_100 = 0

    if cat_path is not None and cat_path.exists():
        with rasterio.open(cat_path) as src:
            cat_data, _ = rasterio.mask.mask(src, [muni_geom], crop=True, nodata=NODATA_SENTINEL)
        # cat_data bands: RP10_cat(0), RP100_cat(1), perm_water(2), spurious(3)
        pw_cat = cat_data[2].astype(float)
        sd_cat = cat_data[3].astype(float)

        for rp, cat_idx, depth_idx in [("RP10", 0, 0), ("RP100", 1, 2)]:
            pc = _pixel_categories(
                depth_data,
                cat_data,
                pw_cat,
                sd_cat,
                return_period=rp,
                depth_band_idx=depth_idx,
                cat_band_idx=cat_idx,
                dry_val=dry_val,
                flood_vals=flood_vals,
            )
            pixel_cats[rp] = pc
            # Non-permanent mask
            cat_band = cat_data[cat_idx].astype(float)
            pw_band = depth_data[3].astype(float)
            nonperm = np.isin(cat_band, list(flood_vals)) & (pw_band != 1)
            depth_stats[rp] = _depth_stats(
                depth_data[depth_idx].astype(float), nonperm, return_period=rp
            )

        # Monotonicity (over all valid pixels in depth bands)
        b10 = depth_data[0].astype(float)
        b20 = depth_data[1].astype(float)
        b100 = depth_data[2].astype(float)
        both_10_20 = (b10 != NODATA_SENTINEL) & (b20 != NODATA_SENTINEL)
        both_20_100 = (b20 != NODATA_SENTINEL) & (b100 != NODATA_SENTINEL)
        mono_10_20 = int((both_10_20 & (b10 > b20 + 0.001)).sum())
        mono_20_100 = int((both_20_100 & (b20 > b100 + 0.001)).sum())

    # 6. Road exposure
    road_exp: dict[str, RoadExposure] | None = None
    if (
        edges_path is not None
        and edges_path.exists()
        and cat_path is not None
        and cat_path.exists()
    ):
        road_exp = {}
        for rp, cat_idx, depth_idx in [("RP10", 0, 0), ("RP100", 1, 2)]:
            road_exp[rp] = compute_road_exposure(
                raster_path,
                cat_path,
                muni_geom,
                edges_path,
                return_period=rp,
                depth_band_idx=depth_idx,
                cat_band_idx=cat_idx,
                dry_val=dry_val,
                flood_vals=flood_vals,
            )

    # 7. LiPAD check
    lipad_overlaps, lipad_notes = check_lipad_overlap(muni_geom, LIPAD_SIBALOM_ISO_BBOX)

    # 8. Suitability decision
    reasons: list[str] = []

    nonperm10 = pixel_cats.get("RP10")
    nonperm100 = pixel_cats.get("RP100")
    exp10 = road_exp.get("RP10") if road_exp else None

    n_nonperm_10 = nonperm10.non_permanent if nonperm10 else 0
    n_nonperm_100 = nonperm100.non_permanent if nonperm100 else 0
    scenario_gain = n_nonperm_100 - n_nonperm_10
    normal_exp_segs = exp10.exposed_normal_segments if exp10 else 0
    has_cat = cat_path is not None and cat_path.exists()

    if n_nonperm_10 == 0:
        decision = ReadinessDecision.BLOCKED
        reasons.append(
            "Zero non-permanent flood pixels at RP10 within municipality boundary. "
            "JRC GloFAS produces no over-bank inundation signal for this area."
        )
    elif not has_cat:
        decision = ReadinessDecision.PARTIAL
        reasons.append("depth_category raster not available; three-state analysis incomplete.")
    else:
        gate_pixels = n_nonperm_10 >= NON_PERM_MIN_PIXELS
        gate_segs = exp10 is None or normal_exp_segs >= NORMAL_SEG_MIN
        gate_scenario = scenario_gain >= SCENARIO_GAIN_MIN

        if gate_pixels and gate_segs and gate_scenario:
            decision = ReadinessDecision.READY
        else:
            decision = ReadinessDecision.PARTIAL
            if not gate_pixels:
                reasons.append(
                    f"Non-permanent pixels at RP10: {n_nonperm_10} < "
                    f"threshold {NON_PERM_MIN_PIXELS} (minimum for over-bank representation)."
                )
            if exp10 is not None and not gate_segs:
                reasons.append(
                    f"Normal road segments with interior overlap ≥ {MIN_INTERIOR_LEN_M}m: "
                    f"{normal_exp_segs} < threshold {NORMAL_SEG_MIN} "
                    f"(minimum for comparative routing observations)."
                )
            if not gate_scenario:
                reasons.append(
                    f"Non-permanent pixel gain RP10→RP100: {scenario_gain:+d} < "
                    f"threshold {SCENARIO_GAIN_MIN} "
                    f"(scenarios indistinguishable at 90 m resolution)."
                )

    logger.info(
        "Phase A decision: %s — %d non-perm pixels (RP10), "
        "%d normal segs exposed, scenario gain %+d",
        decision.value,
        n_nonperm_10,
        normal_exp_segs,
        scenario_gain,
    )

    return PhaseAReadiness(
        decision=decision,
        reasons=reasons,
        municipality_code=municipality_pcode,
        raster_path=raster_path,
        sha256=digest,
        file_bytes=file_bytes,
        raster_crs=raster_crs,
        pixel_size_m_approx=pixel_size_m,
        raster_bounds=bounds,
        pixel_cats=pixel_cats,
        depth_stats=depth_stats,
        monotonicity_violations_rp10_rp20=mono_10_20,
        monotonicity_violations_rp20_rp100=mono_20_100,
        road_exposure=road_exp,
        lipad_overlaps_municipality=lipad_overlaps,
        lipad_notes=lipad_notes,
    )
