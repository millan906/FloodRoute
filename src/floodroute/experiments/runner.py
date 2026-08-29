"""Stage 8 experiment runner: 27-run matrix over (algorithm, RP, demand fraction).

Scenario parameters
-------------------
Municipality : San Jose de Buenavista (PH0600613)
Graph        : PH0600613_phase_b_enriched.graphml (RP10/20/100 flood attributes)
Population   : PSA 2020 Census (psa_barangay_population_sjdb_2020.csv)
Shelter nodes: {33, 58} — scenario-based, NOT from verified shelter records
               (All four registered SJDB shelters are BLOCKED: no verified
               coordinates, all evidence-tier B/C, all operational_status unknown.)
Shelter capacity provenance: scenario_based — set to span the 25 % demand
               scenario (capacity sum = 22 000 ≈ PSA_total × 0.338).

Matrix
------
3 algorithms × 3 return periods × 3 demand fractions = 27 runs

Outputs (written to output_dir/)
--------------------------------
manifest.json         — experiment provenance and parameter record
results.csv           — 27 rows × all metrics (flat, one row per run)
results_detailed.json — same 27 rows with full assignments and route lists
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx

from floodroute.experiments.algorithms import (
    ALGORITHMS,
    DEMAND_FRACTIONS,
    RETURN_PERIODS,
    RunResult,
    run_flood_aware_nearest,
    run_floodroute_assignment,
    run_ordinary_nearest,
)
from floodroute.experiments.demand import (
    BarangayOrigin,
    build_demands,
    load_psa_population,
    snap_barangay_origins,
)
from floodroute.experiments.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Scenario constants (Stage 8)
# ---------------------------------------------------------------------------

MUNICIPALITY_PSGC: str = "PH0600613"
MUNICIPALITY_NAME: str = "San Jose de Buenavista"

# Scenario shelter nodes — controlled parameters, NOT from real shelter data.
# Real SJDB shelter records (sjdb_evacuation_shelters.csv) are all BLOCKED
# (MDRRMO verification pending; no verified coordinates; tiers B/C).
# These nodes are used as controlled scenario supply points for Stage 8.
SCENARIO_SHELTER_CAPACITIES: dict[int, int] = {
    33: 12_000,   # scenario_based
    58: 10_000,   # scenario_based
}
"""Scenario shelter capacities — NOT derived from verified shelter records.
Total capacity = 22 000 ≈ 34 % of PSA 2020 total, spanning the 25 % demand level."""

# Default file paths (relative to project root)
_DEFAULT_GRAPHML = Path("data/processed/hazard/PH0600613_phase_b_enriched.graphml")
_DEFAULT_POP_CSV = Path("data/raw/psa_barangay_population_sjdb_2020.csv")
_DEFAULT_BARANGAY_GPKG = Path("data/processed/admin/barangays_utm51n.gpkg")
_DEFAULT_NODES_GPKG = Path("data/processed/graph/PH0600613_nodes.gpkg")
_DEFAULT_OUTPUT_DIR = Path("results/stage8")


def _git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_graph(graphml_path: Path) -> nx.MultiDiGraph:
    return nx.read_graphml(str(graphml_path), node_type=int)


def _run_one(
    G: nx.MultiDiGraph,
    algorithm: str,
    return_period: str,
    demand_fraction: float,
    demands: dict[int, int],
    capacities: dict[int, int],
) -> RunResult:
    """Dispatch to the appropriate algorithm function."""
    if algorithm == "A":
        result = run_ordinary_nearest(G, demands, capacities, return_period)
    elif algorithm == "B":
        result = run_flood_aware_nearest(G, demands, capacities, return_period)
    elif algorithm == "C":
        result = run_floodroute_assignment(G, demands, capacities, return_period)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    result.demand_fraction = demand_fraction
    return result


def build_experiment_manifest(
    origins: list[BarangayOrigin],
    scenario_shelters: dict[int, int],
    graphml_path: Path,
    population_csv: Path,
    output_dir: Path,
) -> dict:
    """Build the experiment provenance manifest."""
    total_population = sum(o.population_2020 for o in origins)
    return {
        "experiment": "stage8",
        "municipality_psgc": MUNICIPALITY_PSGC,
        "municipality_name": MUNICIPALITY_NAME,
        "graph": str(graphml_path),
        "population_csv": str(population_csv),
        "total_population_2020": total_population,
        "num_barangays": len(origins),
        "scenario_shelter_capacities": {str(k): v for k, v in scenario_shelters.items()},
        "shelter_capacity_provenance": "scenario_based",
        "shelter_note": (
            "All four registered SJDB shelter records are BLOCKED "
            "(sjdb_evacuation_shelters.csv): no verified coordinates, "
            "tiers B/C, operational_status unknown.  Nodes 33 and 58 "
            "are controlled scenario supply points, not real shelters."
        ),
        "algorithms": {
            "A": "ordinary_nearest_shelter (length only, no capacity)",
            "B": "flood_aware_nearest_shelter (conservative hazard policy, no capacity)",
            "C": "floodroute_mcf (flood-aware min-cost-flow, capacitated)",
        },
        "demand_fractions": list(DEMAND_FRACTIONS),
        "return_periods": list(RETURN_PERIODS),
        "num_runs": len(ALGORITHMS) * len(RETURN_PERIODS) * len(DEMAND_FRACTIONS),
        "output_dir": str(output_dir),
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "origin_nodes": {o.psgc: o.origin_node for o in origins},
        "snap_distances_m": {o.psgc: round(o.snap_distance_m, 1) for o in origins},
    }


def run_stage8_matrix(
    graphml_path: Path = _DEFAULT_GRAPHML,
    population_csv: Path = _DEFAULT_POP_CSV,
    barangay_gpkg: Path = _DEFAULT_BARANGAY_GPKG,
    nodes_gpkg: Path = _DEFAULT_NODES_GPKG,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    scenario_shelters: dict[int, int] | None = None,
) -> list[dict]:
    """Run the 27-experiment matrix and write outputs.

    Parameters
    ----------
    graphml_path:
        Phase-B enriched GraphML for PH0600613.
    population_csv:
        PSA 2020 barangay population CSV.
    barangay_gpkg:
        Barangay admin boundaries in EPSG:32651.
    nodes_gpkg:
        Graph node positions in EPSG:32651.
    output_dir:
        Directory to write manifest.json, results.csv, results_detailed.json.
    scenario_shelters:
        ``{node_id: scenario_capacity}`` — defaults to
        ``SCENARIO_SHELTER_CAPACITIES``.

    Returns
    -------
    list[dict]
        One metrics dict per run (27 total).
    """
    if scenario_shelters is None:
        scenario_shelters = SCENARIO_SHELTER_CAPACITIES

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    records = load_psa_population(population_csv, adm3_filter=MUNICIPALITY_PSGC)
    origins = snap_barangay_origins(
        records,
        barangay_gpkg=barangay_gpkg,
        nodes_gpkg=nodes_gpkg,
        municipality_psgc=MUNICIPALITY_PSGC,
        exclude_nodes=set(scenario_shelters),
    )
    G = _load_graph(graphml_path)

    # --- Manifest ---
    manifest = build_experiment_manifest(
        origins, scenario_shelters, graphml_path, population_csv, output_dir
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 27-run matrix ---
    # Pre-build demands per fraction so routing is not repeated unnecessarily.
    demands_by_fraction: dict[float, dict[int, int]] = {}
    for frac in DEMAND_FRACTIONS:
        d, _ = build_demands(origins, frac)
        demands_by_fraction[frac] = d

    all_metrics: list[dict] = []
    all_detailed: list[dict] = []

    for alg in ALGORITHMS:
        for rp in RETURN_PERIODS:
            for frac in DEMAND_FRACTIONS:
                demands = demands_by_fraction[frac]
                result = _run_one(G, alg, rp, frac, demands, scenario_shelters)
                metrics = compute_metrics(result, G)
                all_metrics.append(metrics)

                # Detailed record — adds serialisable assignments/routes
                detailed = dict(metrics)
                detailed["assignments"] = {
                    f"{o},{s}": units for (o, s), units in result.assignments.items()
                }
                detailed["routes"] = {
                    f"{o},{s}": route for (o, s), route in result.routes.items()
                }
                all_detailed.append(detailed)

    # --- Write results.csv ---
    csv_path = output_dir / "results.csv"
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

    # --- Write results_detailed.json ---
    (output_dir / "results_detailed.json").write_text(
        json.dumps(all_detailed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return all_metrics
