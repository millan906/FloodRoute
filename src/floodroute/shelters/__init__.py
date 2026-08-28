"""floodroute.shelters — evidence-tiered shelter data foundation (Stage 6A/6B)."""

from floodroute.shelters.containment import (
    ContainmentResult,
    check_all_containment,
    check_containment,
    load_municipality_polygon,
)
from floodroute.shelters.phase_b import ShelterSnapReport, run_phase_b
from floodroute.shelters.readiness import (
    ReadinessDecision,
    ShelterReadinessResult,
    evaluate_shelter_readiness,
)
from floodroute.shelters.schema import (
    CapacityProvenance,
    EvidenceTier,
    OperationalStatus,
    ShelterRecord,
    ShelterValidationError,
    VerificationStatus,
    load_shelter_csv,
    validate_shelter_records,
)
from floodroute.shelters.snap import (
    SnapResult,
    snap_all_shelters,
    snap_entrance_to_graph,
)

__all__ = [
    "CapacityProvenance",
    "ContainmentResult",
    "EvidenceTier",
    "OperationalStatus",
    "ReadinessDecision",
    "ShelterReadinessResult",
    "ShelterRecord",
    "ShelterSnapReport",
    "ShelterValidationError",
    "SnapResult",
    "VerificationStatus",
    "check_all_containment",
    "check_containment",
    "evaluate_shelter_readiness",
    "load_municipality_polygon",
    "load_shelter_csv",
    "run_phase_b",
    "snap_all_shelters",
    "snap_entrance_to_graph",
    "validate_shelter_records",
]
