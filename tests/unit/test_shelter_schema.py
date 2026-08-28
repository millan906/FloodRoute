"""Unit tests for Stage 6A shelter import schema and readiness evaluator.

Covers:
- Duplicate shelter_id detection
- Invalid Philippine coordinates
- Negative capacity values
- Official capacity without authoritative provenance
- scenario_capacity / official_capacity semantics (equal values permitted)
- project_only, decommissioned, unknown operational-status protection
- Missing entrance coordinates (is_routable property)
- READY / PARTIAL / BLOCKED readiness decisions
- Explicit exclusion of all ineligible statuses from readiness counts
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from floodroute.shelters.readiness import ReadinessDecision, evaluate_shelter_readiness
from floodroute.shelters.schema import (
    CapacityProvenance,
    EvidenceTier,
    OperationalStatus,
    ShelterRecord,
    VerificationStatus,
    validate_shelter_records,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    shelter_id="SH001",
    evidence_tier=EvidenceTier.A,
    verification_status=VerificationStatus.verified,
)

_VALID_LAT = 10.74  # San Jose de Buenavista approximate latitude
_VALID_LON = 121.94  # San Jose de Buenavista approximate longitude


def _make_record(**overrides: object) -> ShelterRecord:
    """Return a minimal valid ShelterRecord with optional field overrides."""
    data = dict(_BASE)
    data.update(overrides)
    return ShelterRecord.model_validate(data)


def _make_routable_ready(**overrides: object) -> ShelterRecord:
    """Return a minimal record that satisfies all READY criteria."""
    data = dict(
        shelter_id="SH_READY",
        evidence_tier=EvidenceTier.A,
        verification_status=VerificationStatus.verified,
        official_capacity=200,
        capacity_provenance=CapacityProvenance.authoritative,
        operational_status=OperationalStatus.operational,
        entrance_latitude=_VALID_LAT,
        entrance_longitude=_VALID_LON,
    )
    data.update(overrides)
    return ShelterRecord.model_validate(data)


# ---------------------------------------------------------------------------
# Duplicate shelter_id detection
# ---------------------------------------------------------------------------


class TestDuplicateShelterId:
    def test_duplicate_id_reported_as_error(self) -> None:
        rows = [
            {
                "shelter_id": "SH001",
                "evidence_tier": "A",
                "verification_status": "verified",
            },
            {
                "shelter_id": "SH001",  # duplicate
                "evidence_tier": "B",
                "verification_status": "unverified",
            },
        ]
        valid, errors = validate_shelter_records(rows)
        assert any("Duplicate shelter_id" in e and "SH001" in e for e in errors)

    def test_unique_ids_produce_no_error(self) -> None:
        rows = [
            {"shelter_id": "SH001", "evidence_tier": "A", "verification_status": "verified"},
            {"shelter_id": "SH002", "evidence_tier": "B", "verification_status": "unverified"},
        ]
        valid, errors = validate_shelter_records(rows)
        dup_errors = [e for e in errors if "Duplicate" in e]
        assert dup_errors == []
        assert len(valid) == 2


# ---------------------------------------------------------------------------
# Invalid coordinates
# ---------------------------------------------------------------------------


class TestInvalidCoordinates:
    def test_latitude_below_philippine_bounds_raises(self) -> None:
        with pytest.raises(ValidationError, match="Philippine geographic bounds"):
            _make_record(latitude=3.0)  # below 4.5 N

    def test_latitude_above_philippine_bounds_raises(self) -> None:
        with pytest.raises(ValidationError, match="Philippine geographic bounds"):
            _make_record(latitude=22.0)  # above 21.1 N

    def test_longitude_below_philippine_bounds_raises(self) -> None:
        with pytest.raises(ValidationError, match="Philippine geographic bounds"):
            _make_record(longitude=115.0)  # below 116.0 E

    def test_longitude_above_philippine_bounds_raises(self) -> None:
        with pytest.raises(ValidationError, match="Philippine geographic bounds"):
            _make_record(longitude=128.0)  # above 127.0 E

    def test_entrance_coordinates_also_validated(self) -> None:
        with pytest.raises(ValidationError, match="Philippine geographic bounds"):
            _make_record(entrance_latitude=2.0, entrance_longitude=_VALID_LON)

    def test_valid_philippine_coordinates_accepted(self) -> None:
        record = _make_record(latitude=_VALID_LAT, longitude=_VALID_LON)
        assert record.latitude == _VALID_LAT
        assert record.longitude == _VALID_LON

    def test_null_coordinates_accepted(self) -> None:
        record = _make_record(latitude=None, longitude=None)
        assert record.latitude is None


# ---------------------------------------------------------------------------
# Negative capacity values
# ---------------------------------------------------------------------------


class TestNegativeCapacity:
    def test_negative_official_capacity_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            _make_record(
                official_capacity=-1,
                capacity_provenance=CapacityProvenance.authoritative,
            )

    def test_negative_scenario_capacity_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            _make_record(scenario_capacity=-50)

    def test_zero_capacity_is_accepted(self) -> None:
        # Zero is a valid documented figure (e.g., temporarily closed)
        record = _make_record(
            official_capacity=0,
            capacity_provenance=CapacityProvenance.authoritative,
        )
        assert record.official_capacity == 0

    def test_positive_capacity_accepted(self) -> None:
        record = _make_record(
            official_capacity=300,
            capacity_provenance=CapacityProvenance.authoritative,
        )
        assert record.official_capacity == 300


# ---------------------------------------------------------------------------
# Official capacity requires authoritative provenance
# ---------------------------------------------------------------------------


class TestOfficialCapacityProvenance:
    def test_official_capacity_without_authoritative_provenance_raises(self) -> None:
        with pytest.raises(ValidationError, match="authoritative"):
            _make_record(
                official_capacity=200,
                capacity_provenance=CapacityProvenance.reported,
            )

    def test_official_capacity_with_estimated_provenance_raises(self) -> None:
        with pytest.raises(ValidationError, match="authoritative"):
            _make_record(
                official_capacity=150,
                capacity_provenance=CapacityProvenance.estimated,
            )

    def test_official_capacity_with_null_provenance_raises(self) -> None:
        with pytest.raises(ValidationError, match="authoritative"):
            _make_record(official_capacity=100, capacity_provenance=None)

    def test_official_capacity_with_authoritative_provenance_accepted(self) -> None:
        record = _make_record(
            official_capacity=200,
            capacity_provenance=CapacityProvenance.authoritative,
        )
        assert record.official_capacity == 200

    def test_null_official_capacity_needs_no_provenance(self) -> None:
        # When no official capacity is provided, provenance rule does not apply
        record = _make_record(official_capacity=None, capacity_provenance=None)
        assert record.official_capacity is None


# ---------------------------------------------------------------------------
# scenario_capacity and official_capacity are semantically distinct fields;
# numerical equality is permitted.
# ---------------------------------------------------------------------------


class TestScenarioCapacitySemantics:
    def test_scenario_equal_to_official_is_permitted(self) -> None:
        # Equal values are valid: the fields carry distinct meaning through their
        # roles (LGU-stamped figure vs. scenario adjustment) and provenance
        # enforcement, not through a required value difference.
        record = _make_record(
            official_capacity=200,
            scenario_capacity=200,
            capacity_provenance=CapacityProvenance.authoritative,
        )
        assert record.official_capacity == 200
        assert record.scenario_capacity == 200

    def test_scenario_different_from_official_accepted(self) -> None:
        record = _make_record(
            official_capacity=200,
            scenario_capacity=150,
            capacity_provenance=CapacityProvenance.authoritative,
        )
        assert record.scenario_capacity == 150

    def test_scenario_without_official_accepted(self) -> None:
        record = _make_record(official_capacity=None, scenario_capacity=180)
        assert record.scenario_capacity == 180

    def test_both_null_accepted(self) -> None:
        record = _make_record(official_capacity=None, scenario_capacity=None)
        assert record.scenario_capacity is None

    def test_scenario_capacity_does_not_require_authoritative_provenance(self) -> None:
        # scenario_capacity carries no provenance requirement of its own;
        # provenance enforcement applies only to official_capacity.
        record = _make_record(
            official_capacity=None,
            scenario_capacity=160,
            capacity_provenance=None,
        )
        assert record.scenario_capacity == 160


# ---------------------------------------------------------------------------
# project_only operational-status protection
# ---------------------------------------------------------------------------


class TestProjectOnlyProtection:
    def test_project_only_with_tier_a_raises(self) -> None:
        # Supplementary schema-level guard: tier A implies authoritative
        # operational status; project_only is incompatible.
        with pytest.raises(ValidationError, match="project_only"):
            _make_record(
                operational_status=OperationalStatus.project_only,
                evidence_tier=EvidenceTier.A,
            )

    def test_project_only_with_tier_b_accepted_at_schema_level(self) -> None:
        # project_only + tier B passes schema validation but is still ineligible.
        record = _make_record(
            operational_status=OperationalStatus.project_only,
            evidence_tier=EvidenceTier.B,
        )
        assert record.operational_status == OperationalStatus.project_only

    def test_project_only_with_excluded_tier_accepted(self) -> None:
        record = _make_record(
            operational_status=OperationalStatus.project_only,
            evidence_tier=EvidenceTier.excluded,
        )
        assert record.evidence_tier == EvidenceTier.excluded

    def test_operational_status_with_tier_a_accepted(self) -> None:
        record = _make_record(
            operational_status=OperationalStatus.operational,
            evidence_tier=EvidenceTier.A,
        )
        assert record.operational_status == OperationalStatus.operational

    def test_project_only_is_not_eligible_regardless_of_tier(self) -> None:
        """is_eligible checks operational_status independently of evidence_tier."""
        for tier in (EvidenceTier.B, EvidenceTier.C, EvidenceTier.excluded):
            record = _make_record(
                shelter_id="SH_PROJ",
                operational_status=OperationalStatus.project_only,
                evidence_tier=tier,
            )
            assert not record.is_eligible, f"Expected ineligible for tier={tier}"

    def test_decommissioned_is_not_eligible(self) -> None:
        record = _make_record(
            shelter_id="SH_DECOMM",
            operational_status=OperationalStatus.decommissioned,
            evidence_tier=EvidenceTier.A,
        )
        assert not record.is_eligible

    def test_unknown_status_is_not_eligible(self) -> None:
        record = _make_record(
            shelter_id="SH_UNKN",
            operational_status=OperationalStatus.unknown,
            evidence_tier=EvidenceTier.B,
        )
        assert not record.is_eligible

    def test_operational_and_standby_are_eligible(self) -> None:
        for status in (OperationalStatus.operational, OperationalStatus.standby):
            record = _make_record(
                shelter_id="SH_ACTIVE",
                operational_status=status,
                evidence_tier=EvidenceTier.A,
            )
            assert record.is_eligible, f"Expected eligible for status={status}"


# ---------------------------------------------------------------------------
# Missing entrance coordinates (is_routable)
# ---------------------------------------------------------------------------


class TestRoutability:
    def test_no_entrance_coords_is_not_routable(self) -> None:
        record = _make_record(entrance_latitude=None, entrance_longitude=None)
        assert not record.is_routable

    def test_partial_entrance_coords_is_not_routable(self) -> None:
        record = _make_record(entrance_latitude=_VALID_LAT, entrance_longitude=None)
        assert not record.is_routable

    def test_full_entrance_coords_is_routable(self) -> None:
        record = _make_record(
            entrance_latitude=_VALID_LAT,
            entrance_longitude=_VALID_LON,
        )
        assert record.is_routable

    def test_excluded_tier_is_not_eligible(self) -> None:
        record = _make_record(evidence_tier=EvidenceTier.excluded)
        assert not record.is_eligible

    def test_tier_a_no_project_only_is_eligible(self) -> None:
        record = _make_record(
            operational_status=OperationalStatus.operational,
            evidence_tier=EvidenceTier.A,
        )
        assert record.is_eligible


# ---------------------------------------------------------------------------
# READY / PARTIAL / BLOCKED readiness decisions
# ---------------------------------------------------------------------------


class TestReadinessDecisions:
    def _ready_record(self, shelter_id: str) -> ShelterRecord:
        return _make_routable_ready(shelter_id=shelter_id)

    # ---- BLOCKED -----------------------------------------------------------

    def test_blocked_when_no_records(self) -> None:
        result = evaluate_shelter_readiness([])
        assert result.decision == ReadinessDecision.BLOCKED
        assert "No shelter records" in result.reasons[0]

    def test_blocked_when_all_excluded_tier(self) -> None:
        records = [
            _make_record(
                shelter_id="SH001",
                evidence_tier=EvidenceTier.excluded,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
            _make_record(
                shelter_id="SH002",
                evidence_tier=EvidenceTier.excluded,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.BLOCKED

    def test_blocked_when_all_project_only(self) -> None:
        records = [
            _make_record(
                shelter_id="SH001",
                operational_status=OperationalStatus.project_only,
                evidence_tier=EvidenceTier.B,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.BLOCKED
        assert result.n_eligible == 0

    def test_blocked_when_fewer_than_two_routable(self) -> None:
        records = [
            _make_record(
                shelter_id="SH001",
                evidence_tier=EvidenceTier.A,
                verification_status=VerificationStatus.verified,
                operational_status=OperationalStatus.operational,
                official_capacity=200,
                capacity_provenance=CapacityProvenance.authoritative,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
            _make_record(
                shelter_id="SH002",
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.unverified,
                # No entrance coordinates → not routable
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.BLOCKED
        assert result.n_routable == 1

    # ---- PARTIAL -----------------------------------------------------------

    def test_partial_when_routable_but_no_verified_records(self) -> None:
        records = [
            _make_record(
                shelter_id=f"SH00{i}",
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.unverified,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            )
            for i in range(1, 3)
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.PARTIAL
        assert result.n_routable == 2
        assert result.n_verified == 0

    def test_partial_when_verified_but_no_authoritative_capacity(self) -> None:
        records = [
            _make_record(
                shelter_id=f"SH00{i}",
                evidence_tier=EvidenceTier.A,
                verification_status=VerificationStatus.verified,
                official_capacity=None,  # no capacity documented
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            )
            for i in range(1, 3)
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.PARTIAL
        assert result.n_auth_capacity == 0
        assert any("authoritative" in r for r in result.reasons)

    def test_partial_when_only_one_ready_candidate(self) -> None:
        records = [
            _make_routable_ready(shelter_id="SH001"),
            _make_record(
                shelter_id="SH002",
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.unverified,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.PARTIAL
        assert result.n_partial_candidates == 2
        assert result.n_ready_candidates == 1

    # ---- READY -------------------------------------------------------------

    def test_ready_with_two_fully_qualifying_records(self) -> None:
        records = [
            self._ready_record("SH001"),
            self._ready_record("SH002"),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.READY
        assert result.reasons == []
        assert result.n_ready_candidates == 2

    def test_ready_with_mix_of_quality_when_two_meet_criteria(self) -> None:
        records = [
            self._ready_record("SH001"),
            self._ready_record("SH002"),
            _make_record(
                shelter_id="SH003",
                evidence_tier=EvidenceTier.C,
                verification_status=VerificationStatus.unverified,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.READY
        assert result.n_total == 3

    def test_ready_excludes_project_only_from_count(self) -> None:
        """project_only must not count toward READY regardless of tier or verification."""
        records = [
            self._ready_record("SH001"),
            _make_record(
                shelter_id="SH002",
                operational_status=OperationalStatus.project_only,
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.verified,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision != ReadinessDecision.READY
        assert result.n_eligible == 1  # project_only independently excluded

    def test_decommissioned_excluded_from_all_counts(self) -> None:
        """Decommissioned facilities must not contribute to PARTIAL or READY."""
        records = [
            self._ready_record("SH001"),
            _make_record(
                shelter_id="SH002",
                operational_status=OperationalStatus.decommissioned,
                evidence_tier=EvidenceTier.A,
                verification_status=VerificationStatus.verified,
                official_capacity=300,
                capacity_provenance=CapacityProvenance.authoritative,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision != ReadinessDecision.READY
        assert result.n_eligible == 1  # decommissioned excluded

    def test_unknown_status_excluded_from_all_counts(self) -> None:
        """Unknown operational status is excluded conservatively."""
        records = [
            self._ready_record("SH001"),
            _make_record(
                shelter_id="SH002",
                operational_status=OperationalStatus.unknown,
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.verified,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.n_eligible == 1  # unknown excluded

    def test_partial_reached_with_tier_b_unverified_public_sources(self) -> None:
        """Publicly documented (tier B, unverified) candidates satisfy PARTIAL."""
        records = [
            _make_record(
                shelter_id=f"SH00{i}",
                operational_status=OperationalStatus.operational,
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.unverified,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
                source_title="Municipal Website",
            )
            for i in range(1, 3)
        ]
        result = evaluate_shelter_readiness(records)
        assert result.decision == ReadinessDecision.PARTIAL
        assert result.n_routable == 2

    # ---- Counts ------------------------------------------------------------

    def test_result_counts_are_consistent(self) -> None:
        records = [
            self._ready_record("SH001"),
            self._ready_record("SH002"),
            _make_record(shelter_id="SH003", evidence_tier=EvidenceTier.excluded),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.n_total == 3
        assert result.n_eligible == 2
        assert result.n_routable == 2
        assert result.n_verified == 2
        assert result.n_auth_capacity == 2
        assert result.n_ready_candidates == 2

    def test_ineligible_statuses_do_not_inflate_counts(self) -> None:
        """All three ineligible statuses must be excluded from n_eligible."""
        records = [
            _make_record(
                shelter_id="SH_PROJ",
                operational_status=OperationalStatus.project_only,
                evidence_tier=EvidenceTier.B,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
            _make_record(
                shelter_id="SH_DECOMM",
                operational_status=OperationalStatus.decommissioned,
                evidence_tier=EvidenceTier.A,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
            _make_record(
                shelter_id="SH_UNKN",
                operational_status=OperationalStatus.unknown,
                evidence_tier=EvidenceTier.B,
                entrance_latitude=_VALID_LAT,
                entrance_longitude=_VALID_LON,
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert result.n_eligible == 0
        assert result.n_routable == 0
        assert result.decision == ReadinessDecision.BLOCKED

    def test_blocking_ids_lists_eligible_without_entrance_coords(self) -> None:
        records = [
            _make_record(
                shelter_id="SH001",
                operational_status=OperationalStatus.operational,
                evidence_tier=EvidenceTier.B,
                verification_status=VerificationStatus.unverified,
                # No entrance coords
            ),
        ]
        result = evaluate_shelter_readiness(records)
        assert "SH001" in result.blocking_ids
