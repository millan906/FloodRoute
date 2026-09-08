"""Unified facility catalog — combines FACILITY_REGISTRY and OSM candidates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from floodroute.scenario.config import ScenarioConfig


@dataclass(frozen=True)
class CatalogEntry:
    """One facility in the unified catalog."""

    facility_id: str           # stable: osm_id or FACILITY_REGISTRY facility_id
    name: str
    facility_type: str         # school|covered_court|barangay_hall|multi_purpose_hall|evacuation_center  # noqa: E501
    latitude: float | None
    longitude: float | None
    osm_element_type: str | None   # "way" | "node" | "relation" | None
    osm_id_raw: str | None         # e.g. "1394870512"
    osm_url: str | None            # e.g. "https://www.openstreetmap.org/way/1394870512"
    designation_status: str        # "government_confirmed_from_published_sources" |
    #                                "lgu_verified" | "historically_activated" |
    #                                "candidate_only"
    designation_source: str
    official_capacity: int | None
    capacity_status: str           # "official" | "unverified" | "unknown"
    snapped_node: int | None
    snapping_distance_m: float | None
    source: str                    # "facility_registry" | "osm_candidate"
    barangay_name: str
    inside_study_boundary: bool | None = None
    notes: str = ""

    @property
    def is_government_confirmed(self) -> bool:
        return self.designation_status in (
            "government_confirmed_from_published_sources",
            "lgu_verified",
        )

    @property
    def can_be_selected(self) -> bool:
        return self.snapped_node is not None


def build_catalog() -> FacilityCatalog:
    """Build the unified catalog from FACILITY_REGISTRY and OSM candidates."""
    from floodroute.dashboard.facilities import FACILITY_REGISTRY
    from floodroute.dashboard.osm_candidates import OSM_CANDIDATES

    entries: list[CatalogEntry] = []

    # Add FACILITY_REGISTRY entries
    for fr in FACILITY_REGISTRY:
        # Extract OSM element type and raw ID from coordinate_source when present.
        # e.g. "OpenStreetMap way:1394870512 (Real Street, San Pedro)"
        _osm_elem_type: str | None = None
        _osm_id_raw: str | None = None
        _osm_url: str | None = None
        _coord_src = fr.coordinate_source or ""
        _m = re.search(r'\b(way|node|relation):(\d+)', _coord_src)
        if _m:
            _osm_elem_type = _m.group(1)
            _osm_id_raw = _m.group(2)
            _osm_url = f"https://www.openstreetmap.org/{_osm_elem_type}/{_osm_id_raw}"

        entries.append(CatalogEntry(
            facility_id=fr.facility_id,
            name=fr.facility_name,
            facility_type=fr.facility_type,
            latitude=fr.latitude,
            longitude=fr.longitude,
            osm_element_type=_osm_elem_type,
            osm_id_raw=_osm_id_raw,
            osm_url=_osm_url,
            designation_status=_map_designation(fr.designation_type, fr.verification_status),
            designation_source=fr.designation_source,
            official_capacity=fr.official_capacity,
            capacity_status="unverified" if fr.official_capacity is None else "official",
            snapped_node=fr.snapped_node_id,
            snapping_distance_m=fr.snap_distance_m,
            source="facility_registry",
            barangay_name=fr.barangay_name,
            inside_study_boundary=True,  # All FACILITY_REGISTRY entries are within PH0600613
        ))

    # Add OSM candidates
    for oc in OSM_CANDIDATES:
        _cand_elem_type = oc.osm_id.split(":")[0] if ":" in oc.osm_id else None
        _cand_id_raw = oc.osm_id.split(":")[-1] if ":" in oc.osm_id else None
        _cand_osm_url: str | None = None
        if _cand_elem_type and _cand_id_raw:
            _cand_osm_url = (
                f"https://www.openstreetmap.org/{_cand_elem_type}/{_cand_id_raw}"
            )
        entries.append(CatalogEntry(
            facility_id=oc.osm_id,
            name=oc.name,
            facility_type=oc.facility_type,
            latitude=oc.latitude,
            longitude=oc.longitude,
            osm_element_type=_cand_elem_type,
            osm_id_raw=_cand_id_raw,
            osm_url=_cand_osm_url,
            designation_status="candidate_only",
            designation_source="OpenStreetMap",
            official_capacity=None,
            capacity_status="unknown",
            snapped_node=oc.snapped_node_id,
            snapping_distance_m=oc.snap_distance_m,
            source="osm_candidate",
            barangay_name=oc.barangay_name,
            inside_study_boundary=True,  # All OSM candidates are within PH0600613
            notes=oc.notes,
        ))

    return FacilityCatalog(entries)


def _map_designation(designation_type: str, verification_status: str) -> str:
    """Map FacilityRecord designation fields to CatalogEntry designation_status.

    Status values:
    - lgu_verified: direct LGU document obtained (reserved for future use)
    - government_confirmed_from_published_sources: government action documented
      via news or secondary sources; no direct LGU document held by researcher
    - historically_activated: evidence of past evacuation use
    - candidate_only: proposed or OSM-tagged; not formally designated
    """
    if designation_type == "permanent" and verification_status == "verified":
        return "lgu_verified"  # reserved for direct LGU documentation
    if designation_type == "permanent":
        return "government_confirmed_from_published_sources"
    if designation_type in ("historically_activated", "contingency"):
        return "historically_activated"
    return "candidate_only"


class FacilityCatalog:
    """Queryable unified facility catalog."""

    def __init__(self, entries: list[CatalogEntry]) -> None:
        self._entries = entries
        self._by_id: dict[str, CatalogEntry] = {e.facility_id: e for e in entries}

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[CatalogEntry]:
        return list(self._entries)

    def get(self, facility_id: str) -> CatalogEntry | None:
        return self._by_id.get(facility_id)

    def by_type(self, facility_type: str) -> list[CatalogEntry]:
        return [e for e in self._entries if e.facility_type == facility_type]

    def fingerprint(self) -> str:
        """Deterministic SHA-256 of sorted facility_id|facility_type pairs."""
        payload = json.dumps(
            sorted(f"{e.facility_id}|{e.facility_type}" for e in self._entries),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def resolve_node_shelters(
        self,
        scenario: ScenarioConfig,
        exclude_nodes: set[int] | None = None,
    ) -> dict[int, int]:
        """Convert ScenarioConfig selections to {node_id: capacity} for optimizer.

        Aggregates capacity when multiple facilities share a node.
        Excludes facilities whose node is in exclude_nodes (origin conflict).
        """
        exclude_nodes = exclude_nodes or set()
        node_caps: dict[int, int] = {}
        for fid, cap in scenario.selected_with_capacity().items():
            entry = self.get(fid)
            if entry is None or entry.snapped_node is None:
                continue
            if entry.snapped_node in exclude_nodes:
                continue
            node_caps[entry.snapped_node] = node_caps.get(entry.snapped_node, 0) + cap
        return node_caps
