# data/manifests/

Each YAML file in this directory records the provenance of one ingested dataset.

## Manifest schema (example)

```yaml
schema_version: "1.0"
source_id: road_network          # matches key in configs/data_sources.yaml
filename: roads_raw.geojson
raw_path: data/raw/roads_raw.geojson
download_date: "2026-01-15"
checksum_sha256: "abc123..."
provenance: observed
source_url: "https://..."
notes: null
```

Manifests are tracked in git so the data lineage is auditable even though the
data files themselves are git-ignored.
