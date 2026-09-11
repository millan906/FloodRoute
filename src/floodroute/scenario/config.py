"""ScenarioConfig — single authority for all scenario parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from floodroute.scenario.catalog import FacilityCatalog


@dataclass
class ScenarioConfig:
    """The active scenario configuration. Must be the sole source of shelter inputs."""

    municipality: str = "PH0600613"
    return_period: str = "RP100"

    # Demand
    demand_mode: str = "fraction"       # "fraction" | "exact"
    demand_fraction: float = 0.25
    demand_exact: int = 0

    # Facilities: facility_id → scenario capacity (positive int)
    # facility_id is stable: osm_id for OSM candidates, facility_id for FACILITY_REGISTRY items
    selected_facility_ids: list[str] = field(default_factory=list)
    facility_capacities: dict[str, int] = field(default_factory=dict)

    # Experiment params
    algorithms: tuple = ("A", "B", "B+", "C")
    capacity_multipliers: tuple = (0.50, 0.75, 1.00, 1.25)
    flood_penalties: tuple = (10.0,)

    # Catalog fingerprint (SHA-256 of sorted facility IDs+types)
    catalog_fingerprint: str = ""

    # Road condition overrides (list of dicts, each with road_name, condition, segment_count)
    road_conditions: list = field(default_factory=list)

    def selected_with_capacity(self) -> dict[str, int]:
        """Return {facility_id: capacity} for selected facilities with positive capacity."""
        return {
            fid: self.facility_capacities[fid]
            for fid in self.selected_facility_ids
            if fid in self.facility_capacities and self.facility_capacities[fid] > 0
        }

    def validation_errors(self, catalog: FacilityCatalog) -> list[str]:
        """Return list of validation error messages, empty if valid."""
        errors = []
        for fid in self.selected_facility_ids:
            if not catalog.get(fid):
                errors.append(f"Facility {fid!r} not found in current catalog")
            elif self.facility_capacities.get(fid, 0) <= 0:
                errors.append(f"Facility {catalog.get(fid).name!r} has no scenario capacity")
        if not self.selected_facility_ids:
            errors.append("No facilities selected")
        return errors

        if not self.selected_facility_ids:
            errors.append("No facilities selected")
        return errors
