"""One focused test: exact demand is conserved across all barangays (Hamilton method)."""
from __future__ import annotations

import pytest

from floodroute.experiments.demand import BarangayOrigin, build_demands_exact


def _origins() -> list[BarangayOrigin]:
    return [
        BarangayOrigin(psgc="PH001", name="A", population_2020=3000,
                       origin_node=1, snap_distance_m=0.0),
        BarangayOrigin(psgc="PH002", name="B", population_2020=7000,
                       origin_node=2, snap_distance_m=0.0),
        BarangayOrigin(psgc="PH003", name="C", population_2020=1500,
                       origin_node=3, snap_distance_m=0.0),
    ]


@pytest.mark.parametrize("exact_total", [0, 1, 7, 100, 999, 1000, 11500])
def test_exact_demand_conservation(exact_total):
    """sum(barangay demands) == exact_total for every valid input."""
    origins = _origins()
    demands, bgy_demand = build_demands_exact(origins, exact_total)

    assert sum(bgy_demand.values()) == exact_total, (
        f"Per-barangay sum {sum(bgy_demand.values())} != exact_total {exact_total}"
    )
    assert sum(demands.values()) == exact_total, (
        f"Node-demand sum {sum(demands.values())} != exact_total {exact_total}"
    )
    # Each barangay demand is non-negative
    for psgc, d in bgy_demand.items():
        assert d >= 0, f"Negative demand for {psgc}: {d}"
