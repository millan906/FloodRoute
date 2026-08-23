"""Geospatial preprocessing configuration (Stage 3).

Defines municipality codes, source paths, target CRS, output paths,
and deterministic-processing options.  All path constants are relative
to the project data/ directory so they can be resolved by the caller
with ``data_dir / CONSTANT``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Municipality codes (PSGC pcode format: "PH" + 7-digit PSGC code)
# ---------------------------------------------------------------------------

MUNICIPALITY_CODES: list[str] = ["PH0600616", "PH0600613", "PH0600608"]

MUNICIPALITY_NAMES: dict[str, str] = {
    "PH0600616": "Sibalom",
    "PH0600613": "San Jose (Capital)",
    "PH0600608": "Hamtic",
}

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------

# Source CRS: WGS84 geographic (matching all raw inputs)
SOURCE_CRS: str = "EPSG:4326"

# Analysis CRS: UTM Zone 51N (WGS84), covers 120°E–126°E
# Confirmed: all three municipalities lie between 121.8°E and 122.3°E
TARGET_CRS: str = "EPSG:32651"

# ---------------------------------------------------------------------------
# Source paths (relative to data_dir)
# ---------------------------------------------------------------------------

ADMIN_ARCHIVE: str = "raw/psa_administrative_boundaries_antique.zip"
OSM_PBF: str = "raw/philippines-latest.osm.pbf"
DEM_DIR: str = "raw/dem_antique"
DEM_TILE_NAMES: list[str] = [
    "cop_dem30_N10_E121.tif",
    "cop_dem30_N10_E122.tif",
]

# Dataset IDs (must match data/manifests/*.yaml)
ADMIN_DATASET_ID: str = "phl_admin_boundaries"
DEM_DATASET_ID: str = "dem_antique_municipalities"
OSM_DATASET_ID: str = "osm_philippines"

# ---------------------------------------------------------------------------
# Output paths (relative to data_dir)
# ---------------------------------------------------------------------------

OUTPUT_ADMIN_DIR: str = "processed/admin"
OUTPUT_DEM_DIR: str = "processed/dem"
OUTPUT_OSM_DIR: str = "processed/osm"
OUTPUT_MANIFESTS_DIR: str = "processed/preprocessing_manifests"

# Preprocessing manifest schema version (separate from dataset manifest schema)
PREPROCESSING_MANIFEST_SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# DEM processing parameters
# ---------------------------------------------------------------------------

# Bilinear resampling for continuous elevation data
DEM_RESAMPLING: str = "bilinear"

# Nodata value written to clipped/reprojected output rasters.
# The raw COG tiles report nodata=None; we assign -9999 on output.
DEM_OUTPUT_NODATA: float = -9999.0

# ---------------------------------------------------------------------------
# Study-area bounds for output validation
# ---------------------------------------------------------------------------

# Envelope covering all three study municipalities, WGS84 (west, south, east, north).
# Derived from documented municipality extents in the acquisition manifest.
# Used for post-write spatial-containment checks on EPSG:4326 outputs.
STUDY_AREA_BOUNDS_WGS84: tuple[float, float, float, float] = (121.80, 10.55, 122.25, 10.90)

# Per-municipality WGS84 envelopes (west, south, east, north).
# Source: dem_antique_municipalities.yaml tile-coverage analysis (2026-08-22).
MUNICIPALITY_BOUNDS_WGS84: dict[str, tuple[float, float, float, float]] = {
    "PH0600616": (121.964, 10.683, 122.207, 10.845),  # Sibalom
    "PH0600613": (121.850, 10.700, 122.000, 10.810),  # San Jose (Capital)
    "PH0600608": (121.950, 10.581, 122.066, 10.754),  # Hamtic
}

# ---------------------------------------------------------------------------
# OSM extraction parameters
# ---------------------------------------------------------------------------

#: Buffer distance in metres applied to each municipality before PBF filtering.
#: Ways are retained if they intersect the buffered boundary. Full way geometries
#: are written (not clipped) to preserve cross-boundary road continuity.
OSM_BUFFER_METRES: float = 500.0

#: Waterway tag values included in OSM extraction by default.
#: Configurable via PreprocessingConfig.waterway_classes.
OSM_WATERWAY_CLASSES: frozenset[str] = frozenset({"river", "stream", "canal", "drain"})

#: OSM output sub-directory (relative to data_dir).
OUTPUT_OSM_DIR: str = "processed/osm"

# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------

# Sort municipalities by pcode before writing to ensure row-order determinism
DETERMINISTIC_SORT_FIELD: str = "adm3_pcode"
BARANGAY_SORT_FIELDS: list[str] = ["adm3_pcode", "adm4_pcode"]

# ---------------------------------------------------------------------------
# PreprocessingConfig dataclass (overridable at runtime)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Mutable preprocessing configuration; defaults match module constants."""

    municipality_codes: list[str] = field(default_factory=lambda: list(MUNICIPALITY_CODES))
    source_crs: str = SOURCE_CRS
    target_crs: str = TARGET_CRS
    admin_archive: str = ADMIN_ARCHIVE
    osm_pbf: str = OSM_PBF
    dem_dir: str = DEM_DIR
    dem_tile_names: list[str] = field(default_factory=lambda: list(DEM_TILE_NAMES))
    output_admin_dir: str = OUTPUT_ADMIN_DIR
    output_dem_dir: str = OUTPUT_DEM_DIR
    output_osm_dir: str = OUTPUT_OSM_DIR
    output_manifests_dir: str = OUTPUT_MANIFESTS_DIR
    osm_buffer_metres: float = OSM_BUFFER_METRES
    waterway_classes: frozenset[str] = field(
        default_factory=lambda: frozenset(OSM_WATERWAY_CLASSES)
    )
    dem_resampling: str = DEM_RESAMPLING
    dem_output_nodata: float = DEM_OUTPUT_NODATA

    def resolve(self, data_dir: Path) -> ResolvedConfig:
        """Return a copy with all paths resolved against *data_dir*."""
        return ResolvedConfig(
            base=self,
            data_dir=data_dir,
        )


@dataclass
class ResolvedConfig:
    """Preprocessing configuration with all paths resolved to absolute Paths."""

    base: PreprocessingConfig
    data_dir: Path

    @property
    def municipality_codes(self) -> list[str]:
        return self.base.municipality_codes

    @property
    def source_crs(self) -> str:
        return self.base.source_crs

    @property
    def target_crs(self) -> str:
        return self.base.target_crs

    @property
    def admin_archive(self) -> Path:
        return self.data_dir / self.base.admin_archive

    @property
    def osm_pbf(self) -> Path:
        return self.data_dir / self.base.osm_pbf

    @property
    def dem_tiles(self) -> list[Path]:
        return [self.data_dir / self.base.dem_dir / t for t in self.base.dem_tile_names]

    @property
    def output_admin_dir(self) -> Path:
        return self.data_dir / self.base.output_admin_dir

    @property
    def output_dem_dir(self) -> Path:
        return self.data_dir / self.base.output_dem_dir

    @property
    def output_osm_dir(self) -> Path:
        return self.data_dir / self.base.output_osm_dir

    @property
    def osm_buffer_metres(self) -> float:
        return self.base.osm_buffer_metres

    @property
    def waterway_classes(self) -> frozenset[str]:
        return self.base.waterway_classes

    @property
    def output_manifests_dir(self) -> Path:
        return self.data_dir / self.base.output_manifests_dir

    @property
    def dem_resampling(self) -> str:
        return self.base.dem_resampling

    @property
    def dem_output_nodata(self) -> float:
        return self.base.dem_output_nodata
