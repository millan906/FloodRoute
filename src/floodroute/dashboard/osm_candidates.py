"""OSM-derived candidate evacuation facility registry for Stage 10.

Source
------
``data/processed/osm/sjdb_osm_facility_candidates.json`` — pre-computed from
``data/raw/philippines-latest.osm.pbf`` during Stage 10 data preparation.

All 37 facilities are named OSM features within the San Jose de Buenavista
municipal boundary (PSGC PH0600613), filtered to named schools, covered
courts, barangay halls, multi-purpose halls, and evacuation centers.

Evidence constraints
--------------------
Every record:
- Has a real name from OSM data — no invented or inferred names.
- Has OSM-sourced centroid coordinates — no invented coordinates.
- Has a nearest road-graph node (research reference; not a pipeline snap).
- Has facility_type drawn from the same controlled vocabulary as
  ``facilities.FACILITY_TYPE_LABELS``.
- Does NOT have a verified LGU designation.
- Does NOT have a verified operational status.
- MUST NOT be described as currently operational or officially designated.

A candidate may enter an optimization scenario ONLY when:
- its ``snapped_node_id`` is not None (it always is for these records);
- the planner explicitly activates it in the dashboard;
- the planner supplies a scenario capacity (since ``official_capacity`` is
  always ``None`` for OSM-only records in this registry).

Activating a candidate does NOT change its evidence classification.

Exclusions
----------
These records are intentionally excluded:
- ``Antique Regional Evacuation Center`` — in FACILITY_REGISTRY as SJDB-001.
- ``Brgy. Funda-Dalipe Evacuation Center`` — in FACILITY_REGISTRY as SJDB-005.
- ``Compman`` — OSM tagging error (not a school).
- ``ANS Farm`` — not an evacuation facility.
- ``Proposed Funda-Dalipe Elementary School`` — proposed, not built.
- ``Barbaza Multi-Purpose Cooperative`` — cooperative, not a public facility.
- ``San Jose Multi-Purpose Cooperative`` — cooperative, not a public facility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# JSON file is in data/processed/osm/ relative to project root.
# .resolve() is required: when loaded via the site-packages symlink
# (macOS editable-install workaround), Path(__file__) returns the symlink
# path, and the parent chain points into .venv/ instead of the project root.
_JSON_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed" / "osm" / "sjdb_osm_facility_candidates.json"

FACILITY_TYPE_VOCAB = frozenset({
    "school", "covered_court", "barangay_hall", "multi_purpose_hall", "evacuation_center"
})

CANDIDATE_TYPE_LABELS: dict[str, str] = {
    "school": "Candidate school",
    "covered_court": "Candidate covered court",
    "barangay_hall": "Candidate barangay hall",
    "multi_purpose_hall": "Candidate multi-purpose hall",
    "evacuation_center": "Candidate evacuation center",
}


@dataclass(frozen=True)
class OsmCandidate:
    """Immutable record for one OSM-sourced candidate facility.

    Evidence is OSM-only. No LGU designation or operational status is claimed.
    Scenario state (activation, scenario capacity) is managed separately in
    ``st.session_state`` so that activation never modifies this record.
    """

    osm_id: str                     # e.g. "way:1394870512" or "node:12884911346"
    name: str                       # exact OSM name tag
    facility_type: str              # one of FACILITY_TYPE_VOCAB
    barangay_name: str
    barangay_psgc: str
    latitude: float
    longitude: float
    coordinate_source: str          # e.g. "way:1394870512"
    coordinate_method: str          # osm_polygon_centroid | osm_node_location
    coordinate_uncertainty_m: int   # 30 for ways, 50 for nodes
    snapped_node_id: int            # nearest road-graph node (research reference)
    snap_distance_m: float          # centroid-to-node distance (m)
    snap_is_pipeline_result: bool   # always False for OSM candidates
    official_capacity: int | None   # always None for OSM candidates
    source_date: str
    notes: str

    @property
    def has_coordinates(self) -> bool:
        return True  # all OSM candidates have coordinates

    @property
    def can_be_scenario_activated(self) -> bool:
        """True for all OSM candidates — they all have a snapped_node_id."""
        return self.snapped_node_id is not None

    @property
    def display_type_label(self) -> str:
        return CANDIDATE_TYPE_LABELS.get(self.facility_type, self.facility_type)


def _load_candidates() -> list[OsmCandidate]:
    if not _JSON_PATH.exists():
        return []
    with _JSON_PATH.open(encoding="utf-8") as fh:
        records = json.load(fh)
    out: list[OsmCandidate] = []
    for r in records:
        ft = r.get("facility_type", "school")
        if ft not in FACILITY_TYPE_VOCAB:
            continue
        out.append(
            OsmCandidate(
                osm_id=r["osm_id"],
                name=r["name"],
                facility_type=ft,
                barangay_name=r.get("barangay_name", ""),
                barangay_psgc=r.get("barangay_psgc", ""),
                latitude=float(r["latitude"]),
                longitude=float(r["longitude"]),
                coordinate_source=r.get("coordinate_source", r["osm_id"]),
                coordinate_method=r.get("coordinate_method", "osm_polygon_centroid"),
                coordinate_uncertainty_m=int(r.get("coordinate_uncertainty_m", 30)),
                snapped_node_id=int(r["snapped_node_id"]),
                snap_distance_m=float(r["snap_distance_m"]),
                snap_is_pipeline_result=bool(r.get("snap_is_pipeline_result", False)),
                official_capacity=r.get("official_capacity"),
                source_date=r.get("source_date", "unknown"),
                notes=r.get("notes", ""),
            )
        )
    return out


#: All 37 OSM-derived candidate facilities within the SJDB municipal boundary.
OSM_CANDIDATES: list[OsmCandidate] = _load_candidates()


def get_candidate_by_osm_id(osm_id: str) -> OsmCandidate | None:
    """Return the OsmCandidate with *osm_id*, or ``None`` if not found."""
    return next((c for c in OSM_CANDIDATES if c.osm_id == osm_id), None)


def candidates_by_type(facility_type: str) -> list[OsmCandidate]:
    """Return all candidates of the given facility_type."""
    return [c for c in OSM_CANDIDATES if c.facility_type == facility_type]
