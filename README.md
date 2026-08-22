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

| Command | Description |
|---|---|
| `floodroute validate-config` | Validate all YAML configs against Pydantic models. |
| `floodroute inspect-data` | Report on data availability and manifest status. |
| `floodroute build-graph` | Build the road-network graph from processed data. |
| `floodroute run-analysis` | Run the core routing/allocation analysis. |
| `floodroute run-experiment` | Execute a named experiment defined in experiments.yaml. |

Commands that require data that has not yet been ingested exit with a clear error rather than generating synthetic outputs.

---

## Stage 0 limitations

Stage 0 establishes the repository skeleton, configuration schema, CLI entrypoints, and test harness. The following are **intentionally absent** in this stage:

- No geospatial or routing libraries (OSMnx, NetworkX, Shapely, etc.)
- No real or synthetic case-study data
- No dashboards or visualisation outputs
- No analysis results or scenario outputs
- `build-graph`, `run-analysis`, and `run-experiment` exit immediately because the data layer does not yet exist

These will be introduced in later thesis stages.

---

## Linting and testing

```bash
ruff check src tests        # lint
ruff format --check src tests  # format check
pytest                      # run test suite
```
