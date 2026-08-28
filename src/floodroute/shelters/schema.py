"""Shelter import schema for FloodRoute Stage 6A.

Defines the evidence-tiered ShelterRecord model and collection-level validation.
Unknown values must remain null — never inferred or invented.
"""

from __future__ import annotations

import csv
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Philippine geographic bounds (WGS-84 decimal degrees)
# ---------------------------------------------------------------------------

_PHL_LAT_MIN: float = 4.5
_PHL_LAT_MAX: float = 21.1
_PHL_LON_MIN: float = 116.0
_PHL_LON_MAX: float = 127.0


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class EvidenceTier(StrEnum):
    """Evidence quality tier for a shelter record.

    A — authoritative primary source (official government document).
    B — documented secondary source (news, NGO report with verifiable reference).
    C — anecdotal / community-reported, unverified.
    excluded — record is known but explicitly excluded from analysis.
    """

    A = "A"
    B = "B"
    C = "C"
    excluded = "excluded"


class OperationalStatus(StrEnum):
    """Whether the facility is currently available for evacuation use."""

    operational = "operational"
    project_only = "project_only"  # Proposed / under study; not currently usable
    standby = "standby"  # Conditionally activatable
    decommissioned = "decommissioned"
    unknown = "unknown"


class VerificationStatus(StrEnum):
    """Whether the record has been validated against an authoritative source."""

    verified = "verified"
    unverified = "unverified"
    rejected = "rejected"


class CapacityProvenance(StrEnum):
    """Origin of the capacity figure.

    authoritative — capacity stated in an official government document.
    reported       — capacity cited in a secondary source without primary reference.
    estimated      — capacity derived or estimated (e.g., floor area / standard).
    not_documented — no capacity source identified.
    """

    authoritative = "authoritative"
    reported = "reported"
    estimated = "estimated"
    not_documented = "not_documented"


# ---------------------------------------------------------------------------
# ShelterRecord — single shelter import record
# ---------------------------------------------------------------------------


class ShelterRecord(BaseModel):
    """One shelter candidate record in the evidence-tiered import schema.

    Rules enforced:
    - official_capacity requires capacity_provenance='authoritative'.
    - official_capacity and scenario_capacity are semantically distinct fields:
      official_capacity is the LGU-stamped figure; scenario_capacity is an
      analytical adjustment for a specific planning scenario.  Numerical
      equality is permitted — the two fields preserve distinct meaning through
      their field semantics and provenance, not through a value difference.
    - Coordinates, when provided, must lie within Philippine bounds.
    - Capacity values must be non-negative integers.
    - project_only facilities cannot hold evidence_tier='A' (supplementary
      schema-level guard; eligibility is enforced independently via is_eligible).
    """

    shelter_id: str
    name: str | None = None
    barangay_name: str | None = None
    barangay_psgc: str | None = None
    facility_type: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    entrance_latitude: float | None = None
    entrance_longitude: float | None = None
    official_capacity: int | None = None
    scenario_capacity: int | None = None
    capacity_provenance: CapacityProvenance | None = None
    operational_status: OperationalStatus | None = None
    hazard_eligibility: str | None = None
    evidence_tier: EvidenceTier
    verification_status: VerificationStatus
    source_title: str | None = None
    source_url: str | None = None
    source_date: str | None = None
    issuing_office: str | None = None
    notes: str | None = None

    # ------------------------------------------------------------------ #
    # Field-level validators                                               #
    # ------------------------------------------------------------------ #

    @field_validator("shelter_id")
    @classmethod
    def shelter_id_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("shelter_id must be a non-empty string")
        return v.strip()

    @field_validator("latitude", "entrance_latitude")
    @classmethod
    def validate_latitude(cls, v: float | None) -> float | None:
        if v is not None and not (_PHL_LAT_MIN <= v <= _PHL_LAT_MAX):
            raise ValueError(
                f"Latitude {v} is outside Philippine geographic bounds "
                f"[{_PHL_LAT_MIN}, {_PHL_LAT_MAX}]"
            )
        return v

    @field_validator("longitude", "entrance_longitude")
    @classmethod
    def validate_longitude(cls, v: float | None) -> float | None:
        if v is not None and not (_PHL_LON_MIN <= v <= _PHL_LON_MAX):
            raise ValueError(
                f"Longitude {v} is outside Philippine geographic bounds "
                f"[{_PHL_LON_MIN}, {_PHL_LON_MAX}]"
            )
        return v

    @field_validator("official_capacity", "scenario_capacity")
    @classmethod
    def capacity_nonnegative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("Capacity values must be non-negative integers")
        return v

    # ------------------------------------------------------------------ #
    # Cross-field validators                                               #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def require_authoritative_provenance_for_official_capacity(self) -> ShelterRecord:
        """official_capacity requires capacity_provenance='authoritative'."""
        if (
            self.official_capacity is not None
            and self.capacity_provenance != CapacityProvenance.authoritative
        ):
            raise ValueError(
                "official_capacity requires capacity_provenance='authoritative'; "
                f"got capacity_provenance={self.capacity_provenance!r}. "
                "Use scenario_capacity for non-authoritative figures."
            )
        return self

    @model_validator(mode="after")
    def protect_project_only_from_tier_a(self) -> ShelterRecord:
        """project_only facilities cannot hold evidence_tier='A'.

        Tier A implies authoritative operational-status documentation.
        A project_only facility is not currently operational and must not
        silently become eligible as an operational shelter.
        """
        if (
            self.operational_status == OperationalStatus.project_only
            and self.evidence_tier == EvidenceTier.A
        ):
            raise ValueError(
                "operational_status='project_only' is incompatible with "
                "evidence_tier='A'. Tier A implies authoritative operational "
                "status; project_only facilities must use tier 'B', 'C', or "
                "'excluded'."
            )
        return self

    # ------------------------------------------------------------------ #
    # Helper properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def is_routable(self) -> bool:
        """True if entrance coordinates are present (required for graph snapping)."""
        return self.entrance_latitude is not None and self.entrance_longitude is not None

    #: Operational statuses that render a facility ineligible for routing analysis.
    #: project_only   — proposed/under study; not currently available.
    #: decommissioned — no longer in service.
    #: unknown        — status unconfirmed; excluded conservatively.
    _INELIGIBLE_STATUSES: ClassVar[frozenset[OperationalStatus]] = frozenset(
        {
            OperationalStatus.project_only,
            OperationalStatus.decommissioned,
            OperationalStatus.unknown,
        }
    )

    @property
    def is_eligible(self) -> bool:
        """True if this record is eligible for routing analysis.

        A record is ineligible when:
        - evidence_tier == 'excluded' (explicitly removed from analysis), OR
        - operational_status is project_only, decommissioned, or unknown.

        This check is independent of evidence_tier; a project_only facility
        with tier B or C still fails here and never contributes to PARTIAL or
        READY counts.
        """
        return (
            self.evidence_tier != EvidenceTier.excluded
            and self.operational_status not in self._INELIGIBLE_STATUSES
        )

    @property
    def has_authoritative_capacity(self) -> bool:
        """True if official_capacity is present with authoritative provenance."""
        return (
            self.official_capacity is not None
            and self.capacity_provenance == CapacityProvenance.authoritative
        )


# ---------------------------------------------------------------------------
# Collection-level validation
# ---------------------------------------------------------------------------


class ShelterValidationError(Exception):
    """Raised when a shelter record collection fails validation."""


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert a raw CSV row (all strings) to typed values for model_validate.

    Empty strings are normalised to None. Numeric fields are parsed to int/float.
    """
    # Fields that must remain strings (or None)
    string_fields = {
        "shelter_id",
        "name",
        "barangay_name",
        "barangay_psgc",
        "facility_type",
        "capacity_provenance",
        "operational_status",
        "hazard_eligibility",
        "evidence_tier",
        "verification_status",
        "source_title",
        "source_url",
        "source_date",
        "issuing_office",
        "notes",
    }
    float_fields = {
        "latitude",
        "longitude",
        "entrance_latitude",
        "entrance_longitude",
    }
    int_fields = {"official_capacity", "scenario_capacity"}

    out: dict[str, Any] = {}
    for k, v in row.items():
        k = k.strip()
        v_stripped = v.strip() if isinstance(v, str) else v
        if v_stripped == "":
            out[k] = None
        elif k in float_fields:
            try:
                out[k] = float(v_stripped)
            except (ValueError, TypeError):
                out[k] = v_stripped  # let Pydantic produce a typed error
        elif k in int_fields:
            try:
                out[k] = int(v_stripped)
            except (ValueError, TypeError):
                out[k] = v_stripped
        elif k in string_fields:
            out[k] = v_stripped
        else:
            out[k] = v_stripped
    return out


def validate_shelter_records(
    rows: list[dict[str, Any]],
) -> tuple[list[ShelterRecord], list[str]]:
    """Validate a list of raw shelter rows.

    Returns (valid_records, error_messages).  Errors are non-fatal; the caller
    decides whether to abort on any error.  Duplicate shelter_id detection is
    performed after per-record validation.
    """
    from pydantic import ValidationError

    valid: list[ShelterRecord] = []
    errors: list[str] = []

    for i, row in enumerate(rows):
        coerced = _coerce_row(row) if all(isinstance(v, str) for v in row.values()) else row
        shelter_id_hint = coerced.get("shelter_id") or f"<row {i + 1}>"
        try:
            record = ShelterRecord.model_validate(coerced)
            valid.append(record)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else "record"
                errors.append(f"[{shelter_id_hint}] {loc}: {err['msg']}")

    # Duplicate ID detection
    seen: dict[str, int] = {}
    for record in valid:
        if record.shelter_id in seen:
            errors.append(
                f"Duplicate shelter_id '{record.shelter_id}' "
                f"(first occurrence at position {seen[record.shelter_id]})"
            )
        else:
            seen[record.shelter_id] = list(r.shelter_id for r in valid).index(
                record.shelter_id
            )

    return valid, errors


def load_shelter_csv(path: Path) -> list[dict[str, str]]:
    """Read a shelter import CSV and return a list of raw row dicts.

    Returns an empty list if the file has only headers (template state).
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]
