"""OSM feature extraction (Stage 3).

Uses pyosmium (PyPI package ``osmium``) to extract road (``highway=*``) and
waterway features from the Philippines PBF in a single file pass.

Spatial policy
--------------
Each study municipality is buffered by ``OSM_BUFFER_METRES`` (default 500 m)
in EPSG:32651 (UTM Zone 51N) and the buffered polygon is reprojected back to
EPSG:4326 for intersection testing against OSM way geometries, which are also
stored in WGS84 by pyosmium's WKBFactory.

Ways are retained if they intersect the buffered boundary of at least one
study municipality.  Output geometries are **not** clipped to the buffer — the
full way geometry is written so that cross-boundary road continuity is
preserved for later graph construction.  If a way crosses into multiple
municipality buffers it appears in each relevant output (counted as a
legitimate cross-municipal duplicate and reported in stats).

Memory considerations
---------------------
``apply_file(locations=True, idx="flex_mem")`` resolves node coordinates for
every way in the PBF.  For the Philippines national extract (~604 MB
compressed) this requires approximately 1–2 GB RAM for the node location
index.  If memory is constrained, pre-clip the PBF to the study region with
``osmium extract`` before running this step.

Single-pass guarantee
---------------------
The PBF is passed to ``apply_file()`` exactly once (from the caller's
perspective).  osmium internally makes two sub-passes — one to index all node
locations and one to process ways — but this is the minimum required to
reconstruct way geometries from a PBF and is unavoidable without a
pre-extracted clip.  Roads and waterways are collected in that single call.

Stage 3 scope boundary
----------------------
This module converts raw OSM data to analysis-ready vector layers.  It does
NOT:
  - build a routing graph or split ways at intersections
  - intersect roads with flood hazard data
  - infer speed, capacity, passability, direction, or road quality
  - assign shelters or generate any research result
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger("floodroute.preprocessing.osm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default buffer distance (metres) applied to each municipality polygon
#: before PBF intersection testing.  Ways intersecting the buffered boundary
#: are retained; full (unclipped) geometries are written to output.
OSM_BUFFER_METRES: float = 500.0

#: Waterway tag values included by default.  The list is configurable via
#: ``PreprocessingConfig.waterway_classes``.
DEFAULT_WATERWAY_CLASSES: frozenset[str] = frozenset({"river", "stream", "canal", "drain"})

#: Road attribute columns written to output GeoPackages.
#: Tags absent from a feature are written as null — values are never inferred.
ROAD_COLUMNS: list[str] = [
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
]

#: Waterway attribute columns written to output GeoPackages.
WATERWAY_COLUMNS: list[str] = [
    "osm_id",
    "waterway",
    "name",
    "intermittent",
    "tunnel",
    "bridge",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OsmExtractionError(RuntimeError):
    """Raised when OSM extraction fails for a non-recoverable reason."""


class OsmBackendUnavailable(RuntimeError):
    """Raised when the ``osmium`` package is not importable.

    Since osmium 4.3.1 is declared as a project dependency, this exception
    should only arise in environments where the package has been removed or in
    tests that mock the import.  The package is installed from PyPI as
    ``osmium`` (not ``pyosmium``).
    """

    _DEFAULT_DETAIL = (
        "The 'osmium' package (pyosmium) is not importable.\n"
        "\n"
        "Install it with:\n"
        "  pip install osmium\n"
        "\n"
        "Do NOT download a substitute OSM extract or invoke an undocumented"
        " web service as a workaround."
    )

    def __init__(self, detail: str | None = None) -> None:
        msg = detail if detail is not None else self._DEFAULT_DETAIL
        super().__init__(msg)
        self.detail = msg


class ChecksumMismatch(ValueError):
    """Raised when the PBF file's SHA-256 does not match the acquisition manifest."""


class OutputExistsError(FileExistsError):
    """Raised when an output file already exists and force=False."""


# ---------------------------------------------------------------------------
# Backend availability check (used by CLI and tests)
# ---------------------------------------------------------------------------


def check_osm_backend(pbf_path: Path) -> None:
    """Verify that pyosmium is importable and the PBF file exists.

    Raises ``OsmBackendUnavailable`` with a clear message if either condition
    is not met.  Called as a pre-flight check in the CLI and in tests.
    """
    if not pbf_path.exists():
        raise OsmBackendUnavailable(
            f"PBF file not found: {pbf_path}\n\n" + OsmBackendUnavailable._DEFAULT_DETAIL
        )
    try:
        import osmium  # noqa: F401
    except ImportError as exc:
        raise OsmBackendUnavailable(
            f"Import of 'osmium' failed: {exc}\n\n" + OsmBackendUnavailable._DEFAULT_DETAIL
        ) from exc


# ---------------------------------------------------------------------------
# Version retrieval
# ---------------------------------------------------------------------------


def get_osmium_version() -> str:
    """Return the installed osmium package version via ``importlib.metadata``.

    Uses ``importlib.metadata`` rather than ``osmium.__version__`` because the
    attribute-based version is not reliably set in all osmium builds.
    """
    return importlib.metadata.version("osmium")


# ---------------------------------------------------------------------------
# PBF source resolution and verification
# ---------------------------------------------------------------------------


def resolve_pbf_from_manifest(
    dataset_id: str,
    manifests_dir: Path,
    data_dir: Path,
) -> tuple[Path, str]:
    """Resolve the PBF path and expected SHA-256 from the acquisition manifest.

    Returns ``(pbf_path, expected_sha256)``.  The path is constructed from
    ``data_dir / manifest.local_path`` so no machine-specific absolute path is
    hard-coded.

    Raises ``OsmExtractionError`` if the manifest is missing or does not
    declare ``local_path`` and ``sha256``.
    """
    import yaml

    manifest_path = manifests_dir / f"{dataset_id}.yaml"
    if not manifest_path.exists():
        raise OsmExtractionError(
            f"Acquisition manifest not found: {manifest_path}\n"
            f"Expected dataset_id '{dataset_id}' in {manifests_dir}."
        )

    with manifest_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    local_rel = data.get("local_path")
    if not local_rel:
        raise OsmExtractionError(f"Manifest '{manifest_path.name}' does not declare 'local_path'.")

    expected_sha256 = data.get("sha256")
    if not expected_sha256:
        raise OsmExtractionError(f"Manifest '{manifest_path.name}' does not declare 'sha256'.")

    pbf_path = data_dir / local_rel
    return pbf_path, str(expected_sha256).strip().lower()


def verify_pbf_checksum(pbf_path: Path, expected_sha256: str) -> None:
    """Verify SHA-256 of the PBF file against the expected digest.

    Raises ``ChecksumMismatch`` on mismatch.  This is a blocking pre-flight
    check before extraction so data-integrity failures are caught immediately.

    The SHA-256 of the Philippines PBF (604 MB) takes 5–15 s on typical
    hardware.
    """
    from floodroute.preprocessing.prep_manifest import compute_file_sha256

    logger.info("Verifying PBF checksum: %s", pbf_path.name)
    actual = compute_file_sha256(pbf_path)
    expected = expected_sha256.strip().lower()
    if actual != expected:
        raise ChecksumMismatch(
            f"PBF checksum mismatch for {pbf_path.name}:\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}"
        )
    logger.info("PBF checksum OK: %s", pbf_path.name)


# ---------------------------------------------------------------------------
# Boundary preparation
# ---------------------------------------------------------------------------


def build_municipality_buffers(
    municipality_gdf: gpd.GeoDataFrame,
    *,
    pcode_field: str = "adm3_pcode",
    buffer_metres: float = OSM_BUFFER_METRES,
    projected_crs: str = "EPSG:32651",
    geographic_crs: str = "EPSG:4326",
) -> dict[str, BaseGeometry]:
    """Return a dict mapping pcode -> buffered municipality polygon in WGS84.

    Procedure:
      1. Reproject municipality polygons to *projected_crs* (a metric CRS).
      2. Buffer by *buffer_metres*.
      3. Reproject the buffered polygons back to *geographic_crs* (WGS84).

    The returned shapely polygons are in WGS84 and can be compared directly
    with OSM way geometries produced by pyosmium's ``WKBFactory`` (also
    WGS84).

    Raises ``ValueError`` if *municipality_gdf* has no CRS.
    """
    if municipality_gdf.crs is None:
        raise ValueError("municipality_gdf has no CRS — cannot buffer.")

    gdf = municipality_gdf[[pcode_field, "geometry"]].to_crs(geographic_crs)
    gdf_proj = gdf.to_crs(projected_crs).copy()
    gdf_proj["geometry"] = gdf_proj.geometry.buffer(buffer_metres)
    gdf_geo = gdf_proj.to_crs(geographic_crs)

    return {str(row[pcode_field]): row["geometry"] for _, row in gdf_geo.iterrows()}


# ---------------------------------------------------------------------------
# Extraction statistics
# ---------------------------------------------------------------------------


@dataclass
class ExtractionStats:
    """Counters accumulated during a single PBF extraction pass."""

    ways_examined: int = 0
    road_candidates: int = 0  # ways with non-empty highway tag
    waterway_candidates: int = 0  # ways with included waterway tag
    incomplete_location: int = 0  # geometry failed due to missing node location
    invalid_geom: int = 0  # WKB parse error or shapely validation failure
    empty_geom: int = 0  # geometry object is empty after parsing
    outside_all: int = 0  # way passed highway/waterway filter but intersects no buffer

    # osm_ids that appear in >1 municipality output (legitimate cross-boundary features)
    road_cross_municipal_ids: set[int] = field(default_factory=set)
    waterway_cross_municipal_ids: set[int] = field(default_factory=set)

    # Retained feature counts per municipality (populated after spatial filter)
    retained_roads: dict[str, int] = field(default_factory=dict)
    retained_waterways: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core per-way processing logic (module-level for testability)
# ---------------------------------------------------------------------------


def _copy_road_tags(osm_id: int, highway: str, w_tags: Any) -> dict[str, Any]:
    """Copy road-relevant tags from a pyosmium TagList into a plain dict.

    Must be called while the osmium way object is still in scope.
    """
    return {
        "osm_id": osm_id,
        "highway": highway,
        "name": w_tags.get("name"),
        "ref": w_tags.get("ref"),
        "oneway": w_tags.get("oneway"),
        "lanes": w_tags.get("lanes"),
        "maxspeed": w_tags.get("maxspeed"),
        "surface": w_tags.get("surface"),
        "access": w_tags.get("access"),
        "service": w_tags.get("service"),
        "bridge": w_tags.get("bridge"),
        "tunnel": w_tags.get("tunnel"),
    }


def _copy_waterway_tags(osm_id: int, waterway: str, w_tags: Any) -> dict[str, Any]:
    """Copy waterway-relevant tags from a pyosmium TagList into a plain dict."""
    return {
        "osm_id": osm_id,
        "waterway": waterway,
        "name": w_tags.get("name"),
        "intermittent": w_tags.get("intermittent"),
        "tunnel": w_tags.get("tunnel"),
        "bridge": w_tags.get("bridge"),
    }


def _process_way_data(
    *,
    osm_id: int,
    road_tags: dict[str, Any] | None,
    waterway_tags: dict[str, Any] | None,
    wkb_bytes: bytes | str | None,
    geom_error: str | None,
    muni_buffers: dict[str, BaseGeometry],
    union_bounds: tuple[float, float, float, float],
    roads: dict[str, list[dict]],
    waterways: dict[str, list[dict]],
    stats: ExtractionStats,
) -> None:
    """Apply spatial filtering to one pre-copied way and update roads/waterways/stats.

    This function contains the core spatial filtering and record assignment
    logic.  It is separated from the osmium handler so it can be exercised in
    unit tests with synthetic data, without needing real osmium.osm.Way
    objects.

    Parameters
    ----------
    road_tags:
        Pre-copied road attribute dict (from ``_copy_road_tags``), or None if
        the way is not a road candidate.
    waterway_tags:
        Pre-copied waterway attribute dict, or None if not a waterway.
    wkb_bytes:
        WKB-encoded LineString bytes from pyosmium, or None if geometry
        creation failed.
    geom_error:
        Error string from geometry creation failure, or None on success.
    muni_buffers:
        Dict mapping pcode -> buffered municipality polygon (WGS84).
    union_bounds:
        Bounding box ``(minx, miny, maxx, maxy)`` of the union of all
        municipality buffers.  Used for fast pre-filter before full
        intersection testing.
    roads, waterways:
        Mutable dicts ``{pcode: [record, ...]}`` updated in place.
    stats:
        Mutable stats object updated in place.
    """
    if wkb_bytes is None:
        # Already counted by the caller (incomplete_location or invalid_geom).
        return

    # Parse WKB with shapely.
    # pyosmium WKBFactory.create_linestring() returns a hex-encoded string in
    # pyosmium 4.x.  shapely.from_wkb() (shapely 2.x) auto-detects the
    # format: str → hex-WKB, bytes → binary WKB.  Both paths are exercised:
    # live osmium passes hex strings; unit tests pass binary bytes via
    # shapely.to_wkb().
    try:
        geom: BaseGeometry = shapely.from_wkb(wkb_bytes)
    except Exception:
        stats.invalid_geom += 1
        return

    if geom is None or geom.is_empty:
        stats.empty_geom += 1
        return

    # Quick bounding-box pre-filter against union of all municipality buffers.
    # Ways whose bbox does not overlap the union bbox cannot intersect any buffer.
    # They are counted in outside_all because they represent excluded candidates.
    gb = geom.bounds  # (minx, miny, maxx, maxy)
    ub_minx, ub_miny, ub_maxx, ub_maxy = union_bounds
    if gb[2] < ub_minx or gb[0] > ub_maxx or gb[3] < ub_miny or gb[1] > ub_maxy:
        stats.outside_all += 1
        return

    # Full intersection test per municipality
    matched: list[str] = [pcode for pcode, buf in muni_buffers.items() if geom.intersects(buf)]

    if not matched:
        stats.outside_all += 1
        return

    # Track cross-municipal duplicates (legitimate)
    if road_tags is not None and len(matched) > 1:
        stats.road_cross_municipal_ids.add(osm_id)
    if waterway_tags is not None and len(matched) > 1:
        stats.waterway_cross_municipal_ids.add(osm_id)

    # Assign to each matching municipality
    for pcode in matched:
        if road_tags is not None:
            rec = {**road_tags, "geometry": geom}
            roads[pcode].append(rec)
        if waterway_tags is not None:
            rec = {**waterway_tags, "geometry": geom}
            waterways[pcode].append(rec)


# ---------------------------------------------------------------------------
# GeoDataFrame builder
# ---------------------------------------------------------------------------


def _build_gdf(
    records: list[dict[str, Any]],
    columns: list[str],
    crs: str,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from a list of record dicts, sorted by osm_id.

    Columns not present in a record are filled with None (null).  The result
    is sorted ascending by ``osm_id`` for deterministic output.
    """
    if not records:
        # Return a properly typed empty GeoDataFrame
        import pandas as pd

        empty_df = pd.DataFrame(columns=columns + ["geometry"])
        return gpd.GeoDataFrame(empty_df, geometry="geometry", crs=crs)

    gdf = gpd.GeoDataFrame(records, columns=columns + ["geometry"], geometry="geometry", crs=crs)
    gdf = gdf.sort_values("osm_id", ignore_index=True)
    return gdf


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_osm_features(
    pbf_path: Path,
    municipality_gdf: gpd.GeoDataFrame,
    output_osm_dir: Path,
    *,
    pcode_field: str = "adm3_pcode",
    buffer_metres: float = OSM_BUFFER_METRES,
    waterway_classes: frozenset[str] = DEFAULT_WATERWAY_CLASSES,
    target_crs: str = "EPSG:32651",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract OSM roads and waterways for all study municipalities.

    Single-pass extraction: the PBF is read once (via osmium's two-sub-pass
    mechanism) and roads and waterways are collected together.

    Outputs written (per municipality):
      - ``{pcode}_roads_wgs84.gpkg``  (layer: roads, EPSG:4326)
      - ``{pcode}_roads_utm51n.gpkg``  (layer: roads, EPSG:32651)
      - ``{pcode}_waterways_wgs84.gpkg``  (layer: waterways, EPSG:4326)
      - ``{pcode}_waterways_utm51n.gpkg``  (layer: waterways, EPSG:32651)

    Parameters
    ----------
    pbf_path:
        Path to the verified Philippines OSM PBF file.
    municipality_gdf:
        GeoDataFrame of study municipality polygons in WGS84 (EPSG:4326).
    output_osm_dir:
        Directory under which output GeoPackages are written.
    pcode_field:
        Column name for municipality codes in *municipality_gdf*.
    buffer_metres:
        Buffer distance in metres applied to each municipality polygon.
    waterway_classes:
        Waterway tag values to include.
    target_crs:
        Projected CRS for UTM output files.
    force:
        Overwrite existing output files.
    dry_run:
        Validate inputs and report planned outputs without writing files.

    Returns
    -------
    dict with keys:
      - ``stats``: ExtractionStats instance
      - ``output_paths``: dict mapping output_id -> Path
      - ``municipality_codes``: list of pcodes processed
      - ``buffer_metres``: buffer distance used
      - ``waterway_classes``: waterway classes used
      - ``osmium_version``: osmium package version
    """
    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------
    try:
        import osmium
        import osmium.geom
    except ImportError as exc:
        raise OsmBackendUnavailable(
            f"Cannot import osmium: {exc}\nInstall with: pip install osmium"
        ) from exc

    # ------------------------------------------------------------------
    # Build buffered municipality boundaries
    # ------------------------------------------------------------------
    logger.info("Building municipality buffers (%.0f m in EPSG:32651).", buffer_metres)
    muni_buffers = build_municipality_buffers(
        municipality_gdf,
        pcode_field=pcode_field,
        buffer_metres=buffer_metres,
        projected_crs=target_crs,
        geographic_crs="EPSG:4326",
    )
    pcodes = sorted(muni_buffers.keys())

    # Pre-compute union bounds for fast per-way bbox pre-filter
    union_geom = shapely.unary_union(list(muni_buffers.values()))
    union_bounds: tuple[float, float, float, float] = union_geom.bounds

    logger.info(
        "Study union buffer: %.4f,%.4f,%.4f,%.4f (WGS84)",
        *union_bounds,
    )

    # ------------------------------------------------------------------
    # Pre-flight output existence check
    # ------------------------------------------------------------------
    output_osm_dir.mkdir(parents=True, exist_ok=True)

    planned_outputs: dict[str, Path] = {}
    for pcode in pcodes:
        for suffix in (
            "_roads_wgs84.gpkg",
            "_roads_utm51n.gpkg",
            "_waterways_wgs84.gpkg",
            "_waterways_utm51n.gpkg",
        ):
            out_id = pcode + suffix.removesuffix(".gpkg")
            out_path = output_osm_dir / f"{pcode}{suffix}"
            if out_path.exists() and not force and not dry_run:
                raise OutputExistsError(f"Output exists (use --force to overwrite): {out_path}")
            planned_outputs[out_id] = out_path

    if dry_run:
        logger.info("DRY RUN: would write %d OSM output files.", len(planned_outputs))
        return {
            "dry_run": True,
            "planned_outputs": {k: str(v) for k, v in planned_outputs.items()},
            "municipality_codes": pcodes,
            "buffer_metres": buffer_metres,
            "waterway_classes": sorted(waterway_classes),
            "osmium_version": get_osmium_version(),
        }

    # ------------------------------------------------------------------
    # Single-pass PBF extraction
    # ------------------------------------------------------------------
    stats = ExtractionStats()
    roads: dict[str, list[dict]] = {p: [] for p in pcodes}
    waterways: dict[str, list[dict]] = {p: [] for p in pcodes}

    wkbfab = osmium.geom.WKBFactory()

    class _Handler(osmium.SimpleHandler):
        """osmium SimpleHandler for a single-pass road + waterway collection."""

        def way(self, w: Any) -> None:  # noqa: ANN401
            stats.ways_examined += 1

            highway: str | None = w.tags.get("highway")
            waterway_val: str | None = w.tags.get("waterway")

            is_road = highway is not None
            is_waterway = waterway_val is not None and waterway_val in waterway_classes

            if not (is_road or is_waterway):
                return

            if is_road:
                stats.road_candidates += 1
            if is_waterway:
                stats.waterway_candidates += 1

            # ---- Copy all data from the osmium object immediately ----
            # pyosmium objects are callback-scoped; tag values and geometry
            # must be copied before this method returns.

            osm_id: int = w.id

            road_tags: dict[str, Any] | None = None
            if is_road:
                road_tags = _copy_road_tags(osm_id, highway, w.tags)  # type: ignore[arg-type]

            ww_tags: dict[str, Any] | None = None
            if is_waterway:
                ww_tags = _copy_waterway_tags(osm_id, waterway_val, w.tags)  # type: ignore[arg-type]

            wkb_bytes: str | None = None
            geom_error: str | None = None
            try:
                # WKBFactory.create_linestring() returns a hex-encoded WKB
                # string in pyosmium 4.x.  Store it directly; _process_way_data
                # detects the type and passes hex=True to shapely.from_wkb.
                wkb_bytes = wkbfab.create_linestring(w)
            except Exception as exc:
                geom_error = str(exc)
                err_lower = geom_error.lower()
                if any(kw in err_lower for kw in ("location", "node", "incomplete")):
                    stats.incomplete_location += 1
                else:
                    stats.invalid_geom += 1
                return
            # ----------------------------------------------------------

            _process_way_data(
                osm_id=osm_id,
                road_tags=road_tags,
                waterway_tags=ww_tags,
                wkb_bytes=wkb_bytes,
                geom_error=geom_error,
                muni_buffers=muni_buffers,
                union_bounds=union_bounds,
                roads=roads,
                waterways=waterways,
                stats=stats,
            )

    logger.info("Starting PBF extraction: %s (locations=True, idx=flex_mem)", pbf_path.name)
    handler = _Handler()
    handler.apply_file(str(pbf_path), locations=True, idx="flex_mem")

    logger.info(
        "PBF pass complete: %d ways examined, %d road candidates, "
        "%d waterway candidates, %d incomplete-location, %d invalid, "
        "%d empty, %d outside all buffers",
        stats.ways_examined,
        stats.road_candidates,
        stats.waterway_candidates,
        stats.incomplete_location,
        stats.invalid_geom,
        stats.empty_geom,
        stats.outside_all,
    )

    # ------------------------------------------------------------------
    # Build GeoDataFrames and write outputs
    # ------------------------------------------------------------------
    src_crs = "EPSG:4326"
    output_paths: dict[str, Path] = {}

    for pcode in pcodes:
        # Roads
        road_gdf_wgs84 = _build_gdf(roads[pcode], ROAD_COLUMNS, src_crs)
        stats.retained_roads[pcode] = len(road_gdf_wgs84)

        # Waterways
        ww_gdf_wgs84 = _build_gdf(waterways[pcode], WATERWAY_COLUMNS, src_crs)
        stats.retained_waterways[pcode] = len(ww_gdf_wgs84)

        for feat_type, gdf_wgs84, columns in (
            ("roads", road_gdf_wgs84, ROAD_COLUMNS),
            ("waterways", ww_gdf_wgs84, WATERWAY_COLUMNS),
        ):
            layer = feat_type

            # WGS84 output
            wgs84_path = output_osm_dir / f"{pcode}_{feat_type}_wgs84.gpkg"
            gdf_wgs84.to_file(wgs84_path, driver="GPKG", layer=layer)
            out_id_wgs84 = f"{pcode}_{feat_type}_wgs84"
            output_paths[out_id_wgs84] = wgs84_path
            logger.info(
                "Wrote %s %s (%d features) → %s",
                pcode,
                feat_type,
                len(gdf_wgs84),
                wgs84_path,
            )

            # UTM51N output — reproject from WGS84 preserving all features
            if len(gdf_wgs84) > 0:
                gdf_utm = gdf_wgs84.to_crs(target_crs)
            else:
                gdf_utm = _build_gdf([], columns, target_crs)

            utm_path = output_osm_dir / f"{pcode}_{feat_type}_utm51n.gpkg"
            gdf_utm.to_file(utm_path, driver="GPKG", layer=layer)
            out_id_utm = f"{pcode}_{feat_type}_utm51n"
            output_paths[out_id_utm] = utm_path
            logger.info(
                "Wrote %s %s UTM51N (%d features) → %s",
                pcode,
                feat_type,
                len(gdf_utm),
                utm_path,
            )

    return {
        "stats": stats,
        "output_paths": output_paths,
        "municipality_codes": pcodes,
        "buffer_metres": buffer_metres,
        "waterway_classes": sorted(waterway_classes),
        "osmium_version": get_osmium_version(),
    }
