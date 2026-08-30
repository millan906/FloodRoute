"""Documented and candidate facility registry — San Jose de Buenavista, Antique.

Single tracked source of truth for dashboard display
-----------------------------------------------------
``data/raw/sjdb_evacuation_shelters.csv`` is the authoritative researcher-curated
data file but is gitignored (``data/raw/**`` rule) and unavailable in a fresh clone.
This Python module is the **tracked researcher-curated dashboard input** — it mirrors
the CSV exactly and is version-controlled so the dashboard remains consistent across
environments.  It must be kept in sync with any CSV updates manually.

No facility is invented, inferred, or added without a citable source record.
``FACILITY_REGISTRY`` contains exactly the four rows from the CSV.

Source file  : data/raw/sjdb_evacuation_shelters.csv  (gitignored; use this file as reference)
Manifest     : data/manifests/sjdb_evacuation_shelters.yaml
Registry PSGC: PH0600613 (San Jose de Buenavista, Antique)
Readiness    : BLOCKED — all four candidates ineligible.

Reference layer only
--------------------
These facilities are not used by the current Stage 8/9 optimization experiment.
The optimization uses two separate controlled scenario shelter nodes (33, 58).

Design constraints enforced by this module
------------------------------------------
• candidate_only facilities must never become eligible automatically.
• Historical activation alone does not establish current designation.
• Unknown capacities remain null.
• Scenario capacity, when set, is labelled researcher-defined.
• Only verified eligible facilities or explicitly scenario-activated
  controlled facilities may enter optimization.
• FacilityRecord is frozen — scenario state is managed separately in
  st.session_state so that activation never modifies evidence fields.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Display label maps (for UI rendering; do not use for logic)
# ---------------------------------------------------------------------------

FACILITY_TYPE_LABELS: dict[str, str] = {
    "evacuation_center": "Evacuation center (purpose-built)",
    "covered_court": "Covered court",
    "multi_purpose_hall": "Multi-purpose hall",
    "school": "School",
    "barangay_hall": "Barangay hall",
}

DESIGNATION_TYPE_LABELS: dict[str, str] = {
    "permanent": "Permanent — formally designated by government authority",
    "contingency": "Contingency — designated when primary shelters are insufficient",
    "historically_activated": "Historically activated — evidence of past evacuation use",
    "candidate_only": "Candidate only — proposed; not yet designated",
}

EVIDENCE_TIER_LABELS: dict[str, str] = {
    "A": "A — authoritative primary (official government document)",
    "B": "B — documented secondary (news article, NGO record, draft plan)",
    "C": "C — anecdotal / community-reported",
}

OPERATIONAL_STATUS_NOTES: dict[str, str] = {
    "unknown": "status unconfirmed; conservative default",
    "project_only": "proposed but not yet operational or formally designated",
    "operational": "confirmed operationally active",
    "standby": "on standby for activation",
}

#: Folium Font-Awesome icon name and marker colors by facility type.
#: color_unknown     — used when operational_status=unknown
#: color_project_only — used when operational_status=project_only or designation_type=candidate_only
FACILITY_MARKER_CONFIG: dict[str, dict] = {
    "evacuation_center": {
        "icon": "hospital-o",
        "color_unknown": "orange",
        "color_project_only": "lightgray",
    },
    "covered_court": {
        "icon": "th-large",
        "color_unknown": "blue",
        "color_project_only": "lightgray",
    },
    "multi_purpose_hall": {
        "icon": "users",
        "color_unknown": "purple",
        "color_project_only": "lightgray",
    },
    "school": {
        "icon": "graduation-cap",
        "color_unknown": "orange",
        "color_project_only": "lightgray",
    },
    "barangay_hall": {
        "icon": "flag",
        "color_unknown": "green",
        "color_project_only": "lightgray",
    },
}

#: Marker color when a facility is scenario-activated (researcher-defined).
SCENARIO_ACTIVE_COLOR: str = "green"


# ---------------------------------------------------------------------------
# FacilityRecord — immutable evidence record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FacilityRecord:
    """Immutable evidence record for one sourced potential facility.

    Scenario state (activation status, scenario capacity) is intentionally
    excluded so that activating a facility as a scenario destination never
    modifies the evidence record.  Scenario state is managed separately in
    ``st.session_state``.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    facility_id: str
    facility_name: str
    barangay_name: str
    facility_type: str          # see FACILITY_TYPE_LABELS for valid values

    # ── Coordinates (WGS-84; null when undocumented) ─────────────────────
    latitude: float | None      # facility centroid; null = no coordinate documented
    longitude: float | None
    coordinate_source: str | None
    coordinate_method: str | None
    coordinate_uncertainty_m: float | None

    # ── Designation ─────────────────────────────────────────────────────────
    designation_type: str       # permanent | contingency | historically_activated | candidate_only
    designation_source: str     # citable source for the designation claim

    # ── Status ──────────────────────────────────────────────────────────────
    operational_status: str     # unknown | project_only | operational | standby
    verification_status: str    # unverified | verified
    evidence_tier: str          # A | B | C

    # ── Entrance / access proxy ──────────────────────────────────────────
    entrance_lat: float | None      # GPS-surveyed entrance or provisional proxy
    entrance_lon: float | None
    entrance_status: str            # present | missing

    # ── Graph snapping ───────────────────────────────────────────────────
    snapped_node_id: int | None         # road-graph node near entrance
    snap_distance_m: float | None       # distance from entrance to node
    snap_is_pipeline_result: bool       # True only when Stage 6B produced the snap

    # ── Capacity ────────────────────────────────────────────────────────
    official_capacity: int | None       # authoritative figure; null when undocumented
    historical_capacity_note: str | None  # non-computational reference; NOT usable as capacity

    # ── Provenance ──────────────────────────────────────────────────────
    source_title: str
    source_url: str | None
    source_date: str
    issuing_office: str | None

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def has_coordinates(self) -> bool:
        """True when facility centroid coordinates are documented."""
        return self.latitude is not None and self.longitude is not None

    @property
    def has_entrance(self) -> bool:
        """True when an entrance or access proxy coordinate is documented."""
        return self.entrance_lat is not None and self.entrance_lon is not None

    @property
    def can_be_scenario_activated(self) -> bool:
        """True when the facility has enough reference data to be used as a
        scenario destination.

        Requires an entrance coordinate reference and a graph node reference.
        The node may be a research reference rather than a pipeline snap result.
        Activation does not change the evidence classification.
        """
        return self.has_entrance and self.snapped_node_id is not None

    @property
    def eligible_for_optimization(self) -> bool:
        """True only for verified, currently operational facilities.

        All facilities in the current registry return False.

        Invariants enforced:
        - candidate_only designation never becomes eligible automatically.
        - Historical activation alone does not establish current designation.
        - unknown and project_only operational status are always ineligible.
        """
        if self.operational_status not in ("operational", "standby"):
            return False
        if self.verification_status != "verified":
            return False
        return self.designation_type != "candidate_only"


# ---------------------------------------------------------------------------
# Registry — exact records from data/raw/sjdb_evacuation_shelters.csv
# ---------------------------------------------------------------------------
#
# Columns used: shelter_id, name, barangay_name, facility_type, latitude,
# longitude, entrance_latitude, entrance_longitude, official_capacity,
# operational_status, evidence_tier, verification_status, source_title,
# source_url, source_date, issuing_office, coordinate_source,
# coordinate_method, coordinate_uncertainty_m, notes (selected fields).
#
# Manifest readiness: BLOCKED — all four candidates ineligible.
#   SJDB-001 operational_status=unknown (2019 turnover = historical evidence only).
#   SJDB-002 operational_status=unknown (community report, no official docs).
#   SJDB-003 operational_status=project_only (draft contingency plan reference).
#   SJDB-004 operational_status=project_only (informal barangay plan mention).
# ---------------------------------------------------------------------------

FACILITY_REGISTRY: list[FacilityRecord] = [
    # ------------------------------------------------------------------
    # SJDB-001 — Antique Regional Evacuation Center
    # Source: PNA article "OCD turns over P36-M regional evacuation center
    #         to Antique" (pna.gov.ph/articles/1065266), 2019-03-21
    #         + OSM way:1394870512 (tags: social_facility=shelter)
    # Coordinates: OSM polygon centroid + west-face midpoint entrance proxy
    # operational_status=unknown: 2019 OCD/DPWH turnover is historical
    #   evidence only; no post-2019 source confirms 2026 availability.
    # ------------------------------------------------------------------
    FacilityRecord(
        facility_id="SJDB-001",
        facility_name="Antique Regional Evacuation Center",
        barangay_name="San Pedro",
        facility_type="evacuation_center",
        latitude=10.800394,
        longitude=121.948801,
        coordinate_source="OpenStreetMap way:1394870512 (Real Street, San Pedro)",
        coordinate_method=(
            "osm_polygon_centroid_facility; "
            "building_west_face_midpoint_entrance_proxy"
        ),
        coordinate_uncertainty_m=25.0,
        designation_type="permanent",
        designation_source=(
            "OCD-6 / DPWH Region VI — official P36M facility turnover to the "
            "Municipality of San Jose de Buenavista, 2019-03-21 "
            "(PNA article pna.gov.ph/articles/1065266)"
        ),
        operational_status="unknown",
        verification_status="unverified",
        evidence_tier="B",
        entrance_lat=10.800406,
        entrance_lon=121.948580,
        entrance_status="present",
        # Research reference — Stage 6B pipeline skipped (ineligible)
        snapped_node_id=1345,
        snap_distance_m=6.3,
        snap_is_pipeline_result=False,
        official_capacity=None,
        historical_capacity_note=(
            "PNA article (2019-03-21) exact quotation: "
            '"can accommodate at least 33 families or 130 individuals". '
            "Not verified as current usable capacity. "
            "Must not be used as official_capacity or scenario_capacity "
            "without MDRRMO/LGU confirmation."
        ),
        source_title="OCD turns over P36-M regional evacuation center to Antique",
        source_url="https://www.pna.gov.ph/articles/1065266",
        source_date="2019-03-21",
        issuing_office="OCD-6 / DPWH Region VI",
    ),
    # ------------------------------------------------------------------
    # SJDB-002 — Bariri Covered Court
    # Source: Community report via MDRRMO verbal communication (2023)
    # No coordinates documented.
    # historically_activated: community-reported evacuation use during
    #   2022 flooding; evidence tier C (anecdotal). Historical activation
    #   alone does not establish current designation.
    # ------------------------------------------------------------------
    FacilityRecord(
        facility_id="SJDB-002",
        facility_name="Bariri Covered Court",
        barangay_name="Bariri",
        facility_type="covered_court",
        latitude=None,
        longitude=None,
        coordinate_source=None,
        coordinate_method=None,
        coordinate_uncertainty_m=None,
        designation_type="historically_activated",
        designation_source=(
            "Community report via MDRRMO verbal communication (2023) — "
            "evacuation use during 2022 flooding (anecdotal; evidence tier C)"
        ),
        operational_status="unknown",
        verification_status="unverified",
        evidence_tier="C",
        entrance_lat=None,
        entrance_lon=None,
        entrance_status="missing",
        snapped_node_id=None,
        snap_distance_m=None,
        snap_is_pipeline_result=False,
        official_capacity=None,
        historical_capacity_note=None,
        source_title="Community report via MDRRMO verbal communication",
        source_url=None,
        source_date="2023",
        issuing_office=None,
    ),
    # ------------------------------------------------------------------
    # SJDB-003 — Badiang Multi-Purpose Hall (Proposed Evacuation Center)
    # Source: SJDB MDRRMO Contingency Plan 2023–2024 (draft)
    # No coordinates documented.
    # candidate_only: referenced in draft plan; not yet formally designated.
    # ------------------------------------------------------------------
    FacilityRecord(
        facility_id="SJDB-003",
        facility_name="Badiang Multi-Purpose Hall (Proposed Evacuation Center)",
        barangay_name="Badiang",
        facility_type="multi_purpose_hall",
        latitude=None,
        longitude=None,
        coordinate_source=None,
        coordinate_method=None,
        coordinate_uncertainty_m=None,
        designation_type="candidate_only",
        designation_source=(
            "SJDB MDRRMO Contingency Plan 2023–2024 (draft; not finalised LGU record)"
        ),
        operational_status="project_only",
        verification_status="unverified",
        evidence_tier="B",
        entrance_lat=None,
        entrance_lon=None,
        entrance_status="missing",
        snapped_node_id=None,
        snap_distance_m=None,
        snap_is_pipeline_result=False,
        official_capacity=None,
        historical_capacity_note=None,
        source_title="SJDB MDRRMO Contingency Plan 2023-2024 (draft)",
        source_url=None,
        source_date="2023",
        issuing_office="SJDB MDRRMO",
    ),
    # ------------------------------------------------------------------
    # SJDB-004 — Barangay 8 Covered Court
    # Source: Barangay 8 Local Disaster Risk Reduction Plan 2022 (informal)
    # No coordinates documented.
    # candidate_only: mentioned in informal barangay plan; no LGU endorsement.
    # ------------------------------------------------------------------
    FacilityRecord(
        facility_id="SJDB-004",
        facility_name="Barangay 8 Covered Court",
        barangay_name="Barangay 8 (Pob.)",
        facility_type="covered_court",
        latitude=None,
        longitude=None,
        coordinate_source=None,
        coordinate_method=None,
        coordinate_uncertainty_m=None,
        designation_type="candidate_only",
        designation_source=(
            "Barangay 8 Local Disaster Risk Reduction Plan 2022 "
            "(informal; no LGU endorsement)"
        ),
        operational_status="project_only",
        verification_status="unverified",
        evidence_tier="C",
        entrance_lat=None,
        entrance_lon=None,
        entrance_status="missing",
        snapped_node_id=None,
        snap_distance_m=None,
        snap_is_pipeline_result=False,
        official_capacity=None,
        historical_capacity_note=None,
        source_title="Barangay 8 Local Disaster Risk Reduction Plan 2022 (informal)",
        source_url=None,
        source_date="2022",
        issuing_office=None,
    ),
]


def get_facility_by_id(facility_id: str) -> FacilityRecord | None:
    """Return the FacilityRecord for *facility_id*, or ``None`` if not found."""
    return next((f for f in FACILITY_REGISTRY if f.facility_id == facility_id), None)
