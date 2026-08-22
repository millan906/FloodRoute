# data/

This directory holds all case-study data for the FloodRoute project.

## Directory layout

| Subdirectory | Contents | Git-tracked? |
|---|---|---|
| `raw/` | Unmodified source files as downloaded or received | No — git-ignored |
| `interim/` | Intermediate artefacts (clipped, reprojected, merged) | No — git-ignored |
| `processed/` | Analysis-ready datasets (graph edges, demand rasters) | No — git-ignored |
| `manifests/` | YAML provenance manifests — one per dataset | **Yes** |
| `templates/` | CSV templates for manual LGU data entry | **Yes** |

## Immutability of raw data

Files placed in `raw/` must never be modified after acquisition. All
transformations produce new files in `interim/` or `processed/`. This
preserves a clear audit trail from source to result.

## Manifest-driven provenance

Every file entering `raw/` must have a corresponding manifest in `manifests/`.
Manifests record:

| Field | Purpose |
|---|---|
| `dataset_id` | Stable identifier used in CLI commands |
| `source_agency` | Who produced the data |
| `source_url` / `landing_page_url` | Where to re-acquire it |
| `license_name` | Usage conditions |
| `access_method` | How the file was (or will be) obtained |
| `acquisition_status` | `not_requested` → `pending` → `acquired` / `unavailable` |
| `sha256` | Hex digest proving file identity (not scientific validity) |
| `provenance` | `observed`, `derived`, or `scenario_based` |
| `validation_status` | `not_validated` → `partially_validated` → `validated` / `rejected` |

Checksums prove file identity — they do not certify scientific correctness.

## Evidence categories

| Label | Meaning |
|---|---|
| `observed` | Derived directly from a primary source (sensor, census, survey) |
| `derived` | Computed from one or more `observed` values via a documented method |
| `scenario_based` | An assumption representing a hypothetical or planning condition |

These categories must remain distinguishable throughout the pipeline.

## Manual LGU records

Shelter and incident data supplied by local government units (LGUs) require
explicit validation before use:

1. Enter records in the appropriate template (`templates/`).
2. Cross-reference each entry against official documentation.
3. Update the manifest `validation_status` to `validated` only after
   independent confirmation.
4. Document the validation source and date in `validation_notes`.

## Stage 1 status

No case-study data has been ingested yet. Eight dataset manifests are
registered. Run:

```bash
floodroute list-datasets
floodroute validate-manifests
```
