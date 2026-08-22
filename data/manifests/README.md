# data/manifests/

Each `*.yaml` file in this directory is a **dataset manifest** describing one
external dataset used in the FloodRoute case study. Manifests are tracked in
git so the data lineage is auditable even though the large data files
themselves are git-ignored.

## Registered datasets (Stage 1)

| Dataset ID | Category | Access method | Status |
|---|---|---|---|
| `psa_administrative_boundaries_antique` | administrative_boundaries | manual_download | not_requested |
| `psa_barangay_population_2020_r06` | barangay_population | manual_download | not_requested |
| `osm_san_jose_de_buenavista` | road_network | direct_download | not_requested |
| `osm_waterways_sjdb` | waterways | derived_later | not_requested |
| `noah_flood_hazard_antique` | flood_hazard | manual_download | not_requested |
| `dem_san_jose_de_buenavista` | terrain_dem | manual_download | not_requested |
| `sjdb_evacuation_shelters` | evacuation_shelters | local_authority | not_requested |
| `sjdb_historical_flood_incidents` | historical_incidents | formal_request | not_requested |

## Manifest schema

```yaml
schema_version: "1.0"      # always "1.0"
dataset_id: <stable_id>    # matches this table
title: "..."
category: <one of 8 controlled values>

# Source
source_agency: "..."
source_url: null | "https://..."
landing_page_url: null | "https://..."
license_name: null | "..."
license_url: null | "https://..."

# Acquisition
access_method: direct_download | api | manual_download | formal_request
              | local_authority | derived_later
acquisition_status: not_requested | pending | acquired | unavailable
acquisition_date: null | "YYYY-MM-DD"

# Spatial & temporal
temporal_coverage: null | "YYYY" | "YYYY-YYYY"
geographic_coverage: "..."
expected_format: null | csv | geojson | shapefile | geotiff | pbf | ...
crs: null | "EPSG:4326"

# Integrity
local_path: null | "raw/..."
file_size_bytes: null | <integer>
sha256: null | "<64-char hex>"

# Evidence
provenance: observed | derived | scenario_based

# Validation
validation_status: not_validated | partially_validated | validated | rejected
validation_notes: null | "..."

required_for_stage: <integer>
citation: null | "..."
```

## Lifecycle

1. Add manifest with `acquisition_status: not_requested`.
2. Run `floodroute list-datasets` to see pending datasets.
3. Acquire the file (download or LGU transfer).
4. Update `acquisition_status: acquired` and record `sha256`.
5. Run `floodroute verify-dataset <id>` to confirm integrity.
6. Update `validation_status` after format and content checks.

## Checksums

SHA-256 digests prove file identity — a matching checksum confirms you have
the same bytes as the original acquisition. It does not certify that the
data are scientifically valid or appropriate for the analysis.
