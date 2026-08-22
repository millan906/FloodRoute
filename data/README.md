# data/

This directory holds all case-study data for the FloodRoute project.

| Subdirectory | Contents | Git-tracked? |
|---|---|---|
| `raw/` | Unmodified source files (shapefiles, rasters, CSVs) | No — git-ignored |
| `interim/` | Intermediate processing artefacts | No — git-ignored |
| `processed/` | Analysis-ready datasets | No — git-ignored |
| `manifests/` | YAML manifests with checksums and provenance | **Yes** |

## Data provenance

Every file entering `raw/` must have a corresponding entry in `data_sources.yaml`
and a manifest file in `manifests/` recording:

- Source URL or contact
- Download date
- SHA-256 checksum
- Provenance label (`observed`, `derived`, or `scenario_based`)

## Stage 0 status

No case-study data has been ingested. Subdirectories contain only `.gitkeep`
files to preserve directory structure. Data ingestion begins in Stage 1.
