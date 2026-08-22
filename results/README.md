# results/

Generated outputs from FloodRoute analyses and experiments.

All subdirectories except `README.md` and `manifests/` are git-ignored.
Result artefacts are large, reproducible from code + data, and must not be
committed to the repository.

| Subdirectory | Contents |
|---|---|
| `runs/` | Per-experiment result files (metrics JSON, route GeoJSON, etc.) |
| `tables/` | Summary tables (CSV, LaTeX) |
| `figures/` | Maps and plots (PNG, PDF) |
| `manifests/` | *(reserved)* Result manifests for archival |

## Reproducing results

```bash
floodroute run-experiment <experiment-id>
```

Stage 0: no results exist because the data layer has not been ingested.
