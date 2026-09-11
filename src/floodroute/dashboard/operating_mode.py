"""Three operating modes for FloodRoute Stage 10."""
from __future__ import annotations

from enum import StrEnum


class OperatingMode(StrEnum):
    OPERATIONAL = "operational"
    CONTROLLED_RESEARCH = "controlled_research"
    LEGACY_BENCHMARK = "legacy_benchmark"


MODE_LABELS: dict[OperatingMode, str] = {
    OperatingMode.OPERATIONAL: "Operational",
    OperatingMode.CONTROLLED_RESEARCH: "Controlled Research",
    OperatingMode.LEGACY_BENCHMARK: "Legacy Benchmark",
}

MODE_DESCRIPTIONS: dict[OperatingMode, str] = {
    OperatingMode.OPERATIONAL: (
        "Live planning mode using current field data and verified shelter readiness."
    ),
    OperatingMode.CONTROLLED_RESEARCH: (
        "Scenario-based research mode using nominal capacities and modelled flood extents. "
        "Not a real-time prediction."
    ),
    OperatingMode.LEGACY_BENCHMARK: (
        "Stage 8 benchmark mode reproducing the original scenario shelter configuration "
        "({33: 12 000, 58: 10 000}) for reproducibility comparison."
    ),
}
