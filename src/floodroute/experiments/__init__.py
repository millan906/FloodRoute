"""floodroute.experiments — Stage 8 evacuation experiment framework."""

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
    BarangayRecord,
    build_demands,
    load_psa_population,
    snap_barangay_origins,
)
from floodroute.experiments.metrics import compute_flood_exposed_length, compute_metrics
from floodroute.experiments.runner import (
    SCENARIO_SHELTER_CAPACITIES,
    build_experiment_manifest,
    run_stage8_matrix,
)

__all__ = [
    "ALGORITHMS",
    "DEMAND_FRACTIONS",
    "RETURN_PERIODS",
    "RunResult",
    "run_flood_aware_nearest",
    "run_floodroute_assignment",
    "run_ordinary_nearest",
    "BarangayOrigin",
    "BarangayRecord",
    "build_demands",
    "load_psa_population",
    "snap_barangay_origins",
    "compute_flood_exposed_length",
    "compute_metrics",
    "SCENARIO_SHELTER_CAPACITIES",
    "build_experiment_manifest",
    "run_stage8_matrix",
]
