"""Shelter readiness evaluator for FloodRoute Stage 6A.

Returns a READY / PARTIAL / BLOCKED decision based on the evidence quality
and completeness of the current shelter candidate pool.

Decision rules
--------------
READY   — ≥2 eligible, verified, routable shelters with authoritative (LGU)
          official capacity.  Authoritative LGU data are required here because
          READY gates the transition to operational shelter allocation.

PARTIAL — ≥2 eligible, routable candidates documented from any source (tier A,
          B, or C) but missing verified status or authoritative official capacity.
          Publicly documented candidate facilities are sufficient for PARTIAL.
          PARTIAL permits reproducible algorithmic research to proceed; it does
          NOT license operational deployment.

BLOCKED — fewer than 2 eligible routable candidates.  The pipeline cannot
          proceed with routing or allocation at any tier.

Operational-deployment note
---------------------------
Official LGU data (authoritative provenance, verified status) are required
before the results can be used in any operational evacuation plan.  PARTIAL
supports algorithm development, sensitivity analysis, and thesis research
without requiring finalised LGU sign-off.

Definitions
-----------
eligible  : evidence_tier != 'excluded'
            AND operational_status not in {project_only, decommissioned, unknown}
routable  : entrance_latitude and entrance_longitude are both present
verified  : verification_status == 'verified'
auth_cap  : official_capacity is present and capacity_provenance == 'authoritative'

Excluded from all counts
------------------------
- evidence_tier = 'excluded'
- operational_status = 'project_only'  (proposed/not yet operational)
- operational_status = 'decommissioned' (no longer in service)
- operational_status = 'unknown'        (status unconfirmed; conservative exclusion)
- records lacking complete entrance coordinates (not routable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from floodroute.shelters.schema import ShelterRecord


class ReadinessDecision(StrEnum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


@dataclass
class ShelterReadinessResult:
    """Full readiness report for the current shelter candidate pool."""

    decision: ReadinessDecision
    reasons: list[str]
    n_total: int
    n_eligible: int
    n_routable: int        # eligible + routable
    n_verified: int        # eligible + routable + verified
    n_auth_capacity: int   # eligible + routable + verified + authoritative capacity
    n_partial_candidates: int  # eligible + routable (used for PARTIAL gate)
    n_ready_candidates: int    # all four criteria met (used for READY gate)
    blocking_ids: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        sym = {"READY": "[✓]", "PARTIAL": "[~]", "BLOCKED": "[✗]"}.get(
            self.decision.value, "[?]"
        )
        return f"{sym} {self.decision.value}"


# ---------------------------------------------------------------------------
# Readiness gate constants
# ---------------------------------------------------------------------------

_MIN_READY_CANDIDATES: int = 2   # minimum eligible+routable+verified+auth_cap
_MIN_PARTIAL_CANDIDATES: int = 2  # minimum eligible+routable (any documentation)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate_shelter_readiness(records: list[ShelterRecord]) -> ShelterReadinessResult:
    """Evaluate shelter pool readiness and return a structured result.

    Parameters
    ----------
    records:
        Validated ShelterRecord instances.  Pass an empty list when no data
        has been entered yet (yields BLOCKED).
    """
    reasons: list[str] = []
    blocking_ids: list[str] = []

    n_total = len(records)

    eligible = [r for r in records if r.is_eligible]
    routable = [r for r in eligible if r.is_routable]
    verified = [r for r in routable if r.verification_status.value == "verified"]
    auth_cap = [r for r in verified if r.has_authoritative_capacity]

    n_eligible = len(eligible)
    n_routable = len(routable)
    n_verified = len(verified)
    n_auth_capacity = len(auth_cap)
    n_partial_candidates = n_routable       # eligible + routable
    n_ready_candidates = n_auth_capacity    # all four criteria

    # Collect IDs of eligible shelters missing entrance coordinates
    for r in eligible:
        if not r.is_routable:
            blocking_ids.append(r.shelter_id)

    # ---- BLOCKED gate -------------------------------------------------------
    if n_partial_candidates < _MIN_PARTIAL_CANDIDATES:
        if n_total == 0:
            reasons.append("No shelter records loaded.")
        elif n_eligible == 0:
            reasons.append(
                "No eligible records (all are excluded-tier or project_only)."
            )
        elif n_routable < _MIN_PARTIAL_CANDIDATES:
            reasons.append(
                f"Only {n_routable} eligible routable candidate(s); "
                f"need ≥{_MIN_PARTIAL_CANDIDATES}. "
                "Add entrance coordinates to enable graph snapping."
            )
        return ShelterReadinessResult(
            decision=ReadinessDecision.BLOCKED,
            reasons=reasons,
            n_total=n_total,
            n_eligible=n_eligible,
            n_routable=n_routable,
            n_verified=n_verified,
            n_auth_capacity=n_auth_capacity,
            n_partial_candidates=n_partial_candidates,
            n_ready_candidates=n_ready_candidates,
            blocking_ids=blocking_ids,
        )

    # ---- READY gate ---------------------------------------------------------
    if n_ready_candidates >= _MIN_READY_CANDIDATES:
        return ShelterReadinessResult(
            decision=ReadinessDecision.READY,
            reasons=reasons,
            n_total=n_total,
            n_eligible=n_eligible,
            n_routable=n_routable,
            n_verified=n_verified,
            n_auth_capacity=n_auth_capacity,
            n_partial_candidates=n_partial_candidates,
            n_ready_candidates=n_ready_candidates,
            blocking_ids=blocking_ids,
        )

    # ---- PARTIAL: ≥2 routable but READY criteria not met -------------------
    if n_verified < _MIN_READY_CANDIDATES:
        reasons.append(
            f"Only {n_verified} verified routable candidate(s); "
            f"need ≥{_MIN_READY_CANDIDATES} verified records."
        )
    if n_auth_capacity < _MIN_READY_CANDIDATES:
        reasons.append(
            f"Only {n_auth_capacity} candidate(s) with authoritative official capacity; "
            f"need ≥{_MIN_READY_CANDIDATES}. "
            "Obtain official capacity figures from MDRRMO/LGU documentation."
        )

    return ShelterReadinessResult(
        decision=ReadinessDecision.PARTIAL,
        reasons=reasons,
        n_total=n_total,
        n_eligible=n_eligible,
        n_routable=n_routable,
        n_verified=n_verified,
        n_auth_capacity=n_auth_capacity,
        n_partial_candidates=n_partial_candidates,
        n_ready_candidates=n_ready_candidates,
        blocking_ids=blocking_ids,
    )
