"""Stage 8 / Stage 11 experiment runner.

Stage 8 runs a fixed 27-run matrix (3 algorithms × 3 RPs × 3 fractions).
Stage 11 adds configurable ExperimentConfig with Cartesian product scenarios.

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
import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx

from floodroute.dashboard.road_overrides import RoadOverrideStore, apply_overrides
from floodroute.experiments.algorithms import (
    ALGORITHMS,
    DEMAND_FRACTIONS,
    RETURN_PERIODS,
    RunResult,
    make_ordinary_weight_fn,
    run_flood_aware_nearest,
    run_floodroute_assignment,
    run_ordinary_nearest,
)
from floodroute.experiments.baselines import run_flood_aware_greedy_capacitated
from floodroute.experiments.demand import (
    BarangayOrigin,
    build_demands,
    load_psa_population,
    snap_barangay_origins,
)
from floodroute.experiments.metrics import compute_metrics
from floodroute.optimization.cost import make_weight_fn

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
            "C": "Global shelter allocation — exact minimum-cost flow",
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
    total_population = sum(o.population_2020 for o in origins)
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
                metrics = compute_metrics(result, G, total_population=total_population)
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
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_metrics)

    # --- Write results_detailed.json ---
    (output_dir / "results_detailed.json").write_text(
        json.dumps(all_detailed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return all_metrics


# ---------------------------------------------------------------------------
# Stage 11: configurable experiment infrastructure
# ---------------------------------------------------------------------------


def make_configuration_hash(config_dict: dict) -> str:
    """Compute a deterministic 12-hex-char SHA-256 of normalised scientific parameters.

    Parameters ``experiment_id``, ``configuration_hash``, and ``created_utc``
    are excluded so that re-running the same scientific configuration always
    yields the same hash.
    """
    exclude = {"experiment_id", "configuration_hash", "created_utc"}
    filtered = {k: v for k, v in config_dict.items() if k not in exclude}
    canonical = json.dumps(filtered, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def make_experiment_id(config_hash: str, timestamp: str) -> str:
    """Compose a unique experiment ID from a UTC timestamp and configuration hash.

    Format: ``<YYYYMMDDTHHMMSSz>_<config_hash>``

    The timestamp component guarantees uniqueness across reruns of the same
    configuration.  The config_hash suffix identifies the scientific
    configuration without obscuring the experiment ID.

    Parameters
    ----------
    config_hash:
        12-hex-char configuration hash from :func:`make_configuration_hash`.
    timestamp:
        UTC timestamp string (e.g. ``'20260903T154201Z'``).
    """
    return f"{timestamp}_{config_hash}"


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully-specified Stage 11 experiment configuration."""

    experiment_id: str
    """Unique run identifier: ``<UTC-timestamp>_<configuration_hash>``."""

    configuration_hash: str
    """Deterministic 12-hex-char SHA-256 of normalised scientific parameters,
    including canonicalized road overrides.  Identical configurations always
    yield the same hash regardless of when the experiment is run."""

    municipality: str
    """Municipality PSGC code (e.g. ``'PH0600613'``)."""

    algorithms: tuple
    """Algorithm labels to run (e.g. ``('A', 'B', 'B+', 'C')``)."""

    return_periods: tuple
    """Return period labels (e.g. ``('RP10', 'RP20', 'RP100')``)."""

    demand_fractions: tuple
    """Demand fractions (e.g. ``(0.10, 0.25, 0.50)``)."""

    capacity_multipliers: tuple
    """Capacity scale factors (e.g. ``(0.5, 1.0, 2.0)``)."""

    flood_penalties: tuple
    """Flood penalty values (e.g. ``(1.0, 10.0, 'prohibited')``)."""

    nominal_capacities: dict
    """``{shelter_node: nominal_capacity}`` — scenario baseline capacities."""

    pilot_mode: bool
    """When True, only one scenario per algorithm is generated."""

    created_utc: str
    """ISO 8601 UTC timestamp when the config was created."""

    road_overrides: dict = dataclasses.field(default_factory=dict)
    """Canonicalized directed edge overrides frozen at run time.
    Keys are ``"u,v"`` strings; values are override attribute dicts.
    Included in ``configuration_hash`` so two runs with different overrides
    always produce different hashes."""


def make_experiment_config(
    municipality: str,
    algorithms: tuple = ("A", "B", "B+", "C"),
    return_periods: tuple = ("RP10", "RP20", "RP100"),
    demand_fractions: tuple = (0.10, 0.25, 0.50),
    capacity_multipliers: tuple = (0.50, 0.75, 1.00, 1.25),
    flood_penalties: tuple = (1.0, 10.0, "prohibited"),
    nominal_capacities: dict | None = None,
    pilot_mode: bool = False,
    road_overrides: dict | None = None,
) -> ExperimentConfig:
    """Construct an ExperimentConfig with configuration_hash and unique experiment_id.

    The ``configuration_hash`` is a deterministic 12-hex SHA-256 of the
    scientific parameters; it is identical for identical configurations.
    Road overrides are canonicalized (sorted by edge key, sub-keys sorted) and
    included in the hash so two runs that differ only in overrides are distinct.
    The ``experiment_id`` prepends a UTC timestamp so that each run is unique
    and a legitimate rerun of the same configuration is always allowed.

    Parameters
    ----------
    road_overrides:
        Optional dict in ``road_override_store_dict`` format
        (``{"u,v": {u, v, status, ...}, ...}``).  When ``None`` the override
        set is treated as empty (equivalent to no overrides).
    """
    if nominal_capacities is None:
        nominal_capacities = dict(SCENARIO_SHELTER_CAPACITIES)
    # Canonicalize road overrides: sort by edge key, then sort each sub-dict.
    _ro = road_overrides or {}
    canonical_overrides: dict = {
        k: dict(sorted(_ro[k].items())) for k in sorted(_ro)
    }
    now_utc = datetime.now(UTC)
    created_utc = now_utc.isoformat()
    timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    config_dict = {
        "municipality": municipality,
        "algorithms": list(algorithms),
        "return_periods": list(return_periods),
        "demand_fractions": list(demand_fractions),
        "capacity_multipliers": list(capacity_multipliers),
        "flood_penalties": list(flood_penalties),
        "nominal_capacities": {str(k): v for k, v in nominal_capacities.items()},
        "pilot_mode": pilot_mode,
        "road_overrides": canonical_overrides,
    }
    config_hash = make_configuration_hash(config_dict)
    exp_id = make_experiment_id(config_hash, timestamp)
    return ExperimentConfig(
        experiment_id=exp_id,
        configuration_hash=config_hash,
        municipality=municipality,
        algorithms=tuple(algorithms),
        return_periods=tuple(return_periods),
        demand_fractions=tuple(demand_fractions),
        capacity_multipliers=tuple(capacity_multipliers),
        flood_penalties=tuple(flood_penalties),
        nominal_capacities=dict(nominal_capacities),
        pilot_mode=pilot_mode,
        created_utc=created_utc,
        road_overrides=canonical_overrides,
    )


@dataclass(frozen=True)
class ScenarioKey:
    """Unique key identifying one scenario in the experiment matrix."""

    algorithm: str
    return_period: str
    demand_fraction: float
    capacity_multiplier: float
    flood_penalty: object  # float | str


@dataclass
class ScenarioResult:
    """Result of one experiment scenario."""

    key: ScenarioKey
    nominal_capacities: dict
    adjusted_capacities: dict
    metrics: dict
    assignments: dict
    route_metrics: dict
    facility_metrics: dict
    unassigned_reasons: dict  # {origin_node: "unreachable"|"capacity_exhausted"}
    run_status: str
    error_message: str | None
    barangay_demand: dict = None  # {adm4_pcode: demand_units} — Hamilton-apportioned audit trail
    demands: dict = None  # {origin_node: demand_units} — per-origin demand for unassigned export

    def __post_init__(self):
        if self.barangay_demand is None:
            self.barangay_demand = {}
        if self.demands is None:
            self.demands = {}


def generate_scenarios(config: ExperimentConfig) -> list[ScenarioKey]:
    """Generate the Cartesian product of all scenario parameters.

    When ``pilot_mode=True``, return one scenario per algorithm (the first
    matching RP10 + lowest demand_fraction + multiplier=1.0 + first penalty,
    falling back to first available values if defaults are not in the config).
    """
    if config.pilot_mode:
        keys: list[ScenarioKey] = []
        rp = "RP10" if "RP10" in config.return_periods else config.return_periods[0]
        frac = min(config.demand_fractions)
        mult = 1.0 if 1.0 in config.capacity_multipliers else config.capacity_multipliers[0]
        penalty = config.flood_penalties[0]
        for alg in config.algorithms:
            keys.append(ScenarioKey(
                algorithm=alg,
                return_period=rp,
                demand_fraction=frac,
                capacity_multiplier=mult,
                flood_penalty=penalty,
            ))
        return keys

    keys = []
    for alg in config.algorithms:
        for rp in config.return_periods:
            for frac in config.demand_fractions:
                for mult in config.capacity_multipliers:
                    for penalty in config.flood_penalties:
                        keys.append(ScenarioKey(
                            algorithm=alg,
                            return_period=rp,
                            demand_fraction=frac,
                            capacity_multiplier=mult,
                            flood_penalty=penalty,
                        ))
    return keys


def _adjusted_capacities(nominal: dict, multiplier: float) -> dict:
    """Scale nominal capacities by multiplier, rounding to nearest int."""
    return {n: round(nom * multiplier) for n, nom in nominal.items()}


def run_scenario(
    G: nx.MultiDiGraph,
    origins: list,
    config: ExperimentConfig,
    key: ScenarioKey,
    road_override_store: RoadOverrideStore | None = None,
) -> ScenarioResult:
    """Run one scenario and return a ScenarioResult.

    Parameters
    ----------
    G:
        Road graph.
    origins:
        List of ``BarangayOrigin`` objects.
    config:
        Experiment configuration (provides nominal_capacities).
    key:
        Scenario parameters.
    road_override_store:
        Optional road condition overrides.  When provided and non-empty, a
        composite weight function is built and passed to flood-aware algorithms
        (B, B+, C).  Algorithm A ignores flood overrides by design but still
        respects hard closures via its own ordinary weight function.
    """
    adj_caps = _adjusted_capacities(config.nominal_capacities, key.capacity_multiplier)

    try:
        demands, barangay_demand_dict = build_demands(origins, key.demand_fraction)

        # Build origin → psgc mapping for B+ deterministic tie-breaking
        origin_psgc = {o.origin_node: o.psgc for o in origins}

        # Build override-aware weight functions.
        # Two variants are constructed when overrides are present:
        #   _flood_override_wfn   — full overrides (closures + flood penalties) for B, B+, C
        #   _closures_only_wfn    — hard-closure overrides only (no flood penalty) for A
        #
        # Algorithm A must honour operationally closed / impassable roads but must
        # NOT apply flood-deterrence cost multipliers (those belong to flood-aware
        # routing only).  Passing flood_penalty_fn=None to apply_overrides achieves
        # this: _IMPASSABLE edges become None (blocked), all others fall through to
        # the ordinary length-based weight function unchanged.
        _flood_override_wfn = None
        _closures_only_wfn = None
        if road_override_store and len(road_override_store) > 0:
            _base_flood_wfn = make_weight_fn(key.return_period)
            _flood_override_wfn = apply_overrides(
                _base_flood_wfn,
                road_override_store,
                flood_penalty_fn=_base_flood_wfn,
            )
            _closures_only_wfn = apply_overrides(
                make_ordinary_weight_fn(),
                road_override_store,
                flood_penalty_fn=None,  # no flood cost for Algorithm A
            )

        if key.algorithm == "A":
            result = run_ordinary_nearest(
                G, demands, adj_caps, key.return_period,
                weight_fn=_closures_only_wfn,  # None when no overrides → uses default
            )
        elif key.algorithm == "B":
            result = run_flood_aware_nearest(
                G, demands, adj_caps, key.return_period,
                flood_penalty=key.flood_penalty,
                weight_fn=_flood_override_wfn,
            )
        elif key.algorithm == "B+":
            result = run_flood_aware_greedy_capacitated(
                G, demands, adj_caps, key.return_period,
                flood_penalty=key.flood_penalty,
                origin_psgc=origin_psgc,
                weight_fn=_flood_override_wfn,
            )
        elif key.algorithm == "C":
            result = run_floodroute_assignment(
                G, demands, adj_caps, key.return_period,
                flood_penalty=key.flood_penalty,
                weight_fn=_flood_override_wfn,
            )
        else:
            raise ValueError(f"Unknown algorithm: {key.algorithm!r}")

        result.demand_fraction = key.demand_fraction
        metrics = compute_metrics(result, G)

        # Route metrics per (o, s)
        route_metrics: dict = {}
        for (o, s), flow in result.assignments.items():
            if flow <= 0:
                continue
            route = result.routes.get((o, s))
            if route is not None:
                from floodroute.experiments.metrics import physical_path_distances
                phys_m, flood_m = physical_path_distances(route, G, key.return_period)
            else:
                phys_m, flood_m = 0.0, 0.0
            penalized = result.od_costs_scenario.get((o, s), 0.0) * flow
            route_metrics[(o, s)] = {
                "physical_m": phys_m,
                "flooded_m": flood_m,
                "penalized_m_eq": penalized,
            }

        # Facility metrics
        shelter_loads: dict[int, int] = {s: 0 for s in adj_caps}
        for (_o, s), flow in result.assignments.items():
            if flow > 0:
                shelter_loads[s] = shelter_loads.get(s, 0) + flow
        facility_metrics: dict = {}
        for s, cap in adj_caps.items():
            load = shelter_loads.get(s, 0)
            facility_metrics[s] = {
                "load": load,
                "capacity": cap,
                "utilization": load / cap if cap > 0 else 0.0,
                "overflow": max(0, load - cap),
            }

        # Unassigned reasons
        od_origins = {k[0] for k in result.od_costs_scenario}
        assigned_origins = {k[0] for k, v in result.assignments.items() if v > 0}
        unassigned_reasons: dict = {}
        for o, d in demands.items():
            if d <= 0 or o in assigned_origins:
                continue
            if o not in od_origins:
                unassigned_reasons[o] = "unreachable"
            elif key.algorithm in ("C", "B+"):
                unassigned_reasons[o] = "capacity_exhausted"

        return ScenarioResult(
            key=key,
            nominal_capacities=dict(config.nominal_capacities),
            adjusted_capacities=adj_caps,
            metrics=metrics,
            assignments=dict(result.assignments),
            route_metrics=route_metrics,
            facility_metrics=facility_metrics,
            unassigned_reasons=unassigned_reasons,
            run_status="completed",
            error_message=None,
            barangay_demand=dict(barangay_demand_dict),
            demands=dict(demands),
        )

    except Exception as exc:  # noqa: BLE001
        return ScenarioResult(
            key=key,
            nominal_capacities=dict(config.nominal_capacities),
            adjusted_capacities=_adjusted_capacities(config.nominal_capacities, key.capacity_multiplier),
            metrics={},
            assignments={},
            route_metrics={},
            facility_metrics={},
            unassigned_reasons={},
            run_status="failed",
            error_message=str(exc),
        )


def run_experiment(
    G: nx.MultiDiGraph,
    origins: list,
    config: ExperimentConfig,
    progress_callback: Callable | None = None,
    road_override_store: RoadOverrideStore | None = None,
) -> list[ScenarioResult]:
    """Run all scenarios in the experiment and return results.

    Parameters
    ----------
    G:
        Road graph.
    origins:
        List of ``BarangayOrigin`` objects for demand building.
    config:
        Experiment configuration.
    progress_callback:
        Optional callable ``(i, total, key)`` called after each scenario.
    road_override_store:
        Optional road condition overrides forwarded to every ``run_scenario``
        call.  When None (default) all scenarios run without manual overrides.

    Returns
    -------
    list[ScenarioResult]
        One result per scenario in the same order as ``generate_scenarios``.
    """
    scenarios = generate_scenarios(config)
    total = len(scenarios)
    results: list[ScenarioResult] = []

    for i, key in enumerate(scenarios):
        sr = run_scenario(G, origins, config, key, road_override_store=road_override_store)
        results.append(sr)
        if progress_callback is not None:
            progress_callback(i + 1, total, key)

    return results
