"""Dataset manifest model and loaders for FloodRoute Stage 1.

Each manifest YAML file records the provenance, acquisition status,
and integrity metadata for one external dataset. Manifests are
version-controlled; large data files are git-ignored.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from floodroute.config import Provenance

MANIFEST_SCHEMA_VERSION = "1.0"


class AcquisitionStatus(StrEnum):
    not_requested = "not_requested"
    pending = "pending"
    acquired = "acquired"
    unavailable = "unavailable"


class ValidationStatus(StrEnum):
    not_validated = "not_validated"
    partially_validated = "partially_validated"
    validated = "validated"
    rejected = "rejected"


class AccessMethod(StrEnum):
    direct_download = "direct_download"
    api = "api"
    manual_download = "manual_download"
    formal_request = "formal_request"
    local_authority = "local_authority"
    derived_later = "derived_later"
    researcher_assembled = "researcher_assembled"  # curated from public sources; no LGU data-sharing


class DatasetCategory(StrEnum):
    administrative_boundaries = "administrative_boundaries"
    barangay_population = "barangay_population"
    road_network = "road_network"
    waterways = "waterways"
    flood_hazard = "flood_hazard"
    terrain_dem = "terrain_dem"
    evacuation_shelters = "evacuation_shelters"
    historical_incidents = "historical_incidents"


class DatasetManifest(BaseModel):
    schema_version: str
    dataset_id: str
    title: str
    category: DatasetCategory
    description: str | None = None
    source_agency: str | None = None
    source_url: str | None = None
    landing_page_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    access_method: AccessMethod
    acquisition_status: AcquisitionStatus
    acquisition_date: str | None = None
    temporal_coverage: str | None = None
    geographic_coverage: str | None = None
    expected_format: str | None = None
    local_path: str | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    crs: str | None = None
    provenance: Provenance
    validation_status: ValidationStatus
    validation_notes: str | None = None
    required_for_stage: int | None = None
    citation: str | None = None

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema_version '{v}'; expected '{MANIFEST_SCHEMA_VERSION}'"
            )
        return v


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_manifest(path: Path) -> DatasetManifest:
    """Load and validate a single dataset manifest YAML file."""
    return DatasetManifest.model_validate(_load_yaml(path))


def load_all_manifests(manifests_dir: Path) -> dict[str, DatasetManifest]:
    """Load all *.yaml manifest files from manifests_dir.

    Returns a mapping of dataset_id → DatasetManifest.
    Raises ValueError on duplicate dataset IDs.
    """
    if not manifests_dir.is_dir():
        return {}

    result: dict[str, DatasetManifest] = {}
    for path in sorted(manifests_dir.glob("*.yaml")):
        manifest = load_manifest(path)
        if manifest.dataset_id in result:
            raise ValueError(
                f"Duplicate dataset_id '{manifest.dataset_id}' "
                f"found in '{path.name}' and a previously loaded manifest."
            )
        result[manifest.dataset_id] = manifest

    return result
