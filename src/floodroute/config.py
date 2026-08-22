"""Pydantic configuration models for FloodRoute YAML files.

Every model enforces:
- schema_version presence
- Provenance literals restricted to the three recognised categories
- null (None) for unknown values — no invented defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

Provenance = Literal["observed", "derived", "scenario_based"]

SCHEMA_VERSION = "1.0"


class ProvenancedValue(BaseModel):
    """A scalar measurement paired with its provenance label and source."""

    value: float | int | str | None = None
    provenance: Provenance | None = None
    source: str | None = None


# ---------------------------------------------------------------------------
# Municipality
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    min_lon: float | None = None
    max_lon: float | None = None
    min_lat: float | None = None
    max_lat: float | None = None
    provenance: Provenance | None = None
    source: str | None = None


class PopulationEntry(BaseModel):
    value: int | None = None
    provenance: Provenance | None = None
    source: str | None = None
    reference_year: int | None = None


class AreaEntry(BaseModel):
    value: float | None = None
    provenance: Provenance | None = None
    source: str | None = None


class MunicipalityEntry(BaseModel):
    name: str | None = None
    fips_code: str | None = None
    state: str | None = None
    crs: str | None = None
    area_km2: AreaEntry = Field(default_factory=AreaEntry)
    population: PopulationEntry = Field(default_factory=PopulationEntry)
    bounding_box: BoundingBox = Field(default_factory=BoundingBox)


class MunicipalityConfig(BaseModel):
    schema_version: str
    municipality: MunicipalityEntry

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version '{v}'; expected '{SCHEMA_VERSION}'")
        return v


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


class DataSourceEntry(BaseModel):
    description: str | None = None
    provenance: Provenance | None = None
    provider: str | None = None
    url: str | None = None
    format: str | None = None
    raw_path: str | None = None
    crs: str | None = None
    reference_date: str | None = None
    reference_year: int | None = None
    checksum_sha256: str | None = None


class DataSourcesConfig(BaseModel):
    schema_version: str
    sources: dict[str, DataSourceEntry]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version '{v}'; expected '{SCHEMA_VERSION}'")
        return v


# ---------------------------------------------------------------------------
# Shelters
# ---------------------------------------------------------------------------


class ShelterCapacity(BaseModel):
    value: int | None = None
    provenance: Provenance | None = None
    source: str | None = None


class ShelterEntry(BaseModel):
    id: str
    name: str | None = None
    address: str | None = None
    lon: float | None = None
    lat: float | None = None
    crs: str | None = None
    capacity: ShelterCapacity = Field(default_factory=ShelterCapacity)
    accessible: bool | None = None
    operator: str | None = None
    activation_lead_time_hours: ProvenancedValue = Field(default_factory=ProvenancedValue)
    notes: str | None = None


class SheltersConfig(BaseModel):
    schema_version: str
    shelters: list[ShelterEntry]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version '{v}'; expected '{SCHEMA_VERSION}'")
        return v


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


class EligibilityRule(BaseModel):
    description: str | None = None
    provenance: Provenance | None = None
    source: str | None = None


class CapacityEnforcementRule(EligibilityRule):
    treat_null_capacity_as_uncapacitated: bool = True


class GeographicConstraintRule(EligibilityRule):
    max_travel_time_minutes: ProvenancedValue = Field(default_factory=ProvenancedValue)


class PetFriendlyRule(EligibilityRule):
    shelters_confirmed_pet_friendly: list[str] = Field(default_factory=list)


class EligibilityConfig(BaseModel):
    schema_version: str
    eligibility: dict[str, Any]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version '{v}'; expected '{SCHEMA_VERSION}'")
        return v


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class ScenarioEntry(BaseModel):
    id: str
    label: str
    description: str | None = None
    return_period_years: int | None = None
    inundation_depth_threshold_m: ProvenancedValue = Field(default_factory=ProvenancedValue)
    flood_extent_source: str | None = None
    provenance: Provenance
    notes: str | None = None


class ScenariosConfig(BaseModel):
    schema_version: str
    scenarios: list[ScenarioEntry]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version '{v}'; expected '{SCHEMA_VERSION}'")
        return v


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class ExperimentEntry(BaseModel):
    id: str
    label: str
    description: str | None = None
    scenario_id: str | None = None
    algorithm: str | None = None
    demand_model: str | None = None
    seed: int
    crs: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    notes: str | None = None


class ExperimentsConfig(BaseModel):
    schema_version: str
    experiments: list[ExperimentEntry]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version '{v}'; expected '{SCHEMA_VERSION}'")
        return v


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_municipality_config(path: Path) -> MunicipalityConfig:
    return MunicipalityConfig.model_validate(_load_yaml(path))


def load_data_sources_config(path: Path) -> DataSourcesConfig:
    return DataSourcesConfig.model_validate(_load_yaml(path))


def load_shelters_config(path: Path) -> SheltersConfig:
    return SheltersConfig.model_validate(_load_yaml(path))


def load_eligibility_config(path: Path) -> EligibilityConfig:
    return EligibilityConfig.model_validate(_load_yaml(path))


def load_scenarios_config(path: Path) -> ScenariosConfig:
    return ScenariosConfig.model_validate(_load_yaml(path))


def load_experiments_config(path: Path) -> ExperimentsConfig:
    return ExperimentsConfig.model_validate(_load_yaml(path))


# ---------------------------------------------------------------------------
# Validate all configs at once
# ---------------------------------------------------------------------------


class AllConfigs(BaseModel):
    """Container returned by validate_all_configs."""

    municipality: MunicipalityConfig
    data_sources: DataSourcesConfig
    shelters: SheltersConfig
    eligibility: EligibilityConfig
    scenarios: ScenariosConfig
    experiments: ExperimentsConfig


def validate_all_configs(configs_dir: Path) -> AllConfigs:
    """Load and validate every YAML file in configs_dir.

    Raises ValueError if any file fails validation.
    """
    return AllConfigs(
        municipality=load_municipality_config(configs_dir / "municipality.yaml"),
        data_sources=load_data_sources_config(configs_dir / "data_sources.yaml"),
        shelters=load_shelters_config(configs_dir / "shelters.yaml"),
        eligibility=load_eligibility_config(configs_dir / "eligibility.yaml"),
        scenarios=load_scenarios_config(configs_dir / "scenarios.yaml"),
        experiments=load_experiments_config(configs_dir / "experiments.yaml"),
    )
