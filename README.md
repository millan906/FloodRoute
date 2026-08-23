# FloodRoute

**FloodRoute** is an MSCS thesis project that researches flood-resilient evacuation routing and shelter allocation for a municipal case study. The goal is to produce reproducible, evidence-grounded analyses rather than a deployed product.

---

## Project purpose

Floods displace residents and saturate shelter networks simultaneously. This project models the interaction between road-network degradation (from inundation), population demand, and shelter capacity under multiple hazard scenarios to answer:

- Which routes remain passable under a given flood extent?
- Which shelter sites absorb demand equitably and without exceeding capacity?
- How sensitive are outcomes to scenario assumptions (return period, demand model, capacity estimate)?

---

## Repository structure

```
floodroute/
├── configs/           # Validated YAML configuration files (schema-versioned)
├── data/              # Data directory — large files are git-ignored
│   ├── raw/           # Unmodified source data (git-ignored, tracked by manifests)
│   ├── interim/       # Intermediate processing artefacts (git-ignored)
│   ├── processed/     # Analysis-ready datasets (git-ignored)
│   └── manifests/     # Checksums and provenance records (tracked)
├── src/floodroute/    # Python package (src layout)
├── tests/             # Pytest suite
│   ├── unit/          # Fast, in-memory unit tests with toy fixtures
│   └── integration/   # Tests that require the processed data layer
├── results/           # Generated outputs (git-ignored except READMEs/.gitkeeps)
│   ├── runs/          # Per-experiment result artefacts
│   ├── tables/        # Summary tables
│   └── figures/       # Plots and maps
├── dashboard/         # (Future) interactive exploration tool
└── scripts/           # Utility shell/Python scripts
```

---

## Evidence categories

All data values carry one of three provenance labels:

| Label | Meaning |
|---|---|
| `observed` | Derived directly from a primary source (sensor, census, survey). |
| `derived` | Computed from one or more `observed` values via a documented method. |
| `scenario_based` | An assumption made to represent a hypothetical or planning condition. |

Unknown values are recorded as `null` — never replaced by invented defaults.

---

## Setup

```bash
# Requires Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## CLI commands

| Command | Stage | Description |
|---|---|---|
| `floodroute validate-config` | 0 | Validate all YAML configs against Pydantic models. |
| `floodroute inspect-data` | 1 | Report on data availability and manifest status. |
| `floodroute list-datasets` | 1 | List registered datasets with acquisition/validation status. |
| `floodroute validate-manifests` | 1 | Validate all acquisition manifest YAML files. |
| `floodroute acquire-dataset <ID>` | 1 | Download a single dataset by manifest ID. |
| `floodroute verify-dataset <ID>` | 1 | Verify SHA-256 of a locally acquired dataset. |
| `floodroute preprocess-geospatial` | 3 | Convert verified raw inputs to analysis-ready municipal layers. |
| `floodroute build-graph` | 4+ | Build the road-network graph from processed data (placeholder). |
| `floodroute run-analysis` | 4+ | Run the core routing/allocation analysis (placeholder). |
| `floodroute run-experiment` | 4+ | Execute a named experiment (placeholder). |

Commands that require data or processing steps not yet complete exit with a clear error.

---

## Stage 3: geospatial preprocessing

### Command

```bash
# Validate inputs and report what would be written (no files created):
floodroute preprocess-geospatial --dry-run --skip-osm

# Run preprocessing (writes to data/processed/):
floodroute preprocess-geospatial --skip-osm

# Re-run and overwrite existing outputs:
floodroute preprocess-geospatial --skip-osm --force
```

### Required inputs

| File | Dataset ID | Notes |
|---|---|---|
| `data/raw/psa_administrative_boundaries_antique.zip` | `phl_admin_boundaries` | Philippines COD-AB ZIP with shapefiles admin0–admin4 |
| `data/raw/dem_antique/cop_dem30_N10_E121.tif` | `dem_antique_municipalities` | Copernicus GLO-30 DEM tile |
| `data/raw/dem_antique/cop_dem30_N10_E122.tif` | `dem_antique_municipalities` | Copernicus GLO-30 DEM tile |
| `data/raw/philippines-latest.osm.pbf` | `osm_philippines` | Full Philippines OSM extract (OSM blocker; see below) |

### Output structure

```
data/processed/
├── admin/
│   ├── municipalities_wgs84.gpkg    # 3 municipality polygons, EPSG:4326
│   ├── municipalities_utm51n.gpkg   # 3 municipality polygons, EPSG:32651
│   ├── barangays_wgs84.gpkg         # 151 barangay polygons, EPSG:4326
│   └── barangays_utm51n.gpkg        # 151 barangay polygons, EPSG:32651
├── dem/
│   ├── mosaic.vrt                   # GDAL VRT mosaicking both GLO-30 tiles
│   ├── PH0600608_dem_utm51n.tif     # Hamtic DEM, EPSG:32651, ~30 m
│   ├── PH0600613_dem_utm51n.tif     # San Jose DEM, EPSG:32651, ~30 m
│   └── PH0600616_dem_utm51n.tif     # Sibalom DEM, EPSG:32651, ~30 m
├── osm/                             # Roads and waterways (pending OSM blocker)
└── preprocessing_manifests/         # JSON provenance records for every output
    ├── municipalities_wgs84.json
    ├── municipalities_utm51n.json
    ├── barangays_wgs84.json
    ├── barangays_utm51n.json
    ├── dem_PH0600608_utm51n.json
    ├── dem_PH0600613_utm51n.json
    └── dem_PH0600616_utm51n.json
```

### CRS choice

All outputs are produced in two CRS:
- **EPSG:4326** (WGS84 geographic) — matches raw source CRS; used for source-aligned outputs.
- **EPSG:32651** (UTM Zone 51N) — analysis CRS. All three study municipalities
  (Sibalom, San Jose, Hamtic) fall within 121.8°E–122.3°E, which is inside
  UTM Zone 51N (120°E–126°E). Bilinear resampling is used for DEM reprojection.

### Study municipalities

| Name | adm3_pcode | Barangays |
|---|---|---|
| Sibalom | PH0600616 | 76 |
| San Jose (Capital) | PH0600613 | 28 |
| Hamtic | PH0600608 | 47 |

### OSM blocker

`fiona 1.10.1` (installed) raises `DriverError: unsupported driver: 'OSM'` when
opening the Philippines PBF — this GDAL build lacks the OSM driver.  `pyosmium`
is not installed.  Use `--skip-osm` to run admin/DEM preprocessing without OSM.

Remediation (choose one):
- `pip install pyosmium`
- `pip install 'fiona[gdal-full]'` (GDAL with OSM driver)

### Stage 3 scope boundary

Stage 3 **only** converts raw inputs to analysis-ready layers.  It does **not**:
- Perform routing or graph construction
- Intersect roads with flood hazard data
- Calculate slope, aspect, or flood depth
- Assign shelters or generate research results

---

## Data categories

| Directory | Contents | Git-tracked? |
|---|---|---|
| `data/raw/` | Unmodified source files | No (tracked by manifests) |
| `data/interim/` | Intermediate processing artefacts | No |
| `data/processed/` | Analysis-ready outputs from Stage 3 | No |
| `data/manifests/` | Dataset acquisition manifests (YAML) | Yes |
| `data/processed/preprocessing_manifests/` | Processing provenance (JSON) | No |
| `tests/fixtures/` | Synthetic software-test fixtures | Yes |

**Synthetic test fixtures** (`tests/fixtures/`, `tests/unit/`) are minimal data
structures used exclusively for software testing.  They are not research data.

---

## Stage 0 limitations (historical)

Stage 0 established the repository skeleton, configuration schema, CLI entrypoints,
and test harness without any geospatial libraries or real data.  Stage 3 introduces
geospatial preprocessing; routing and analysis remain in later stages.

---

## Linting and testing

```bash
ruff check src tests        # lint
ruff format --check src tests  # format check
pytest                      # run test suite
```
