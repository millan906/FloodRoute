"""Stage 6B pipeline: shelter candidate boundary validation and graph snapping.

Pipeline steps
--------------
1. Load and validate shelter CSV (schema.py).
2. Check each record against the municipal boundary (containment.py).
3. Snap eligible, routable entrances to the road graph (snap.py).
4. Evaluate readiness on the full validated record set (readiness.py).
5. Assemble and return a machine-readable ShelterSnapReport.

Usage
-----
This module is called by the ``snap-shelters`` CLI command.
All file paths are resolved by the caller; this module is path-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from floodroute.shelters.containment import (
    ContainmentResult,
    check_all_containment,
    load_municipality_polygon,
)
from floodroute.shelters.readiness import (
    ShelterReadinessResult,
    evaluate_shelter_readiness,
)
from floodroute.shelters.schema import ShelterRecord, load_shelter_csv, validate_shelter_records
from floodroute.shelters.snap import SnapResult, snap_all_shelters

logger = logging.getLogger("floodroute.shelters.phase_b")

# ---------------------------------------------------------------------------
# Report type
# ---------------------------------------------------------------------------


@dataclass
class ShelterSnapReport:
    """Machine-readable output of the Stage 6B pipeline.

    Attributes
    ----------
    records:
        All validated shelter records (eligible and ineligible).
    validation_errors:
        Per-row validation error strings from schema validation.
    containment_results:
        One ContainmentResult per record.
    snap_results:
        One SnapResult per record (status "skipped"/"no_entrance" for
        ineligible or coordinate-missing records).
    readiness:
        READY / PARTIAL / BLOCKED decision based on the full record set.
    outside_boundary:
        shelter_ids whose facility coordinate lies outside the municipal boundary.
    """

    records: list[ShelterRecord]
    validation_errors: list[str]
    containment_results: list[ContainmentResult]
    snap_results: list[SnapResult]
    readiness: ShelterReadinessResult
    outside_boundary: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_validated(self) -> int:
        return len(self.records)

    @property
    def n_accepted_snaps(self) -> int:
        return sum(1 for r in self.snap_results if r.accepted)

    @property
    def n_rejected_snaps(self) -> int:
        return sum(1 for r in self.snap_results if r.status == "rejected")

    @property
    def n_no_entrance(self) -> int:
        return sum(1 for r in self.snap_results if r.status == "no_entrance")

    def snap_by_id(self, shelter_id: str) -> SnapResult | None:
        for r in self.snap_results:
            if r.shelter_id == shelter_id:
                return r
        return None

    def containment_by_id(self, shelter_id: str) -> ContainmentResult | None:
        for r in self.containment_results:
            if r.shelter_id == shelter_id:
                return r
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_phase_b(
    shelter_csv: Path,
    municipalities_gpkg: Path,
    graph_graphml: Path,
    target_psgc: str,
    *,
    warn_m: float = 100.0,
    reject_m: float = 500.0,
    abort_on_validation_errors: bool = False,
) -> ShelterSnapReport:
    """Run the full Stage 6B pipeline and return a ShelterSnapReport.

    Parameters
    ----------
    shelter_csv:
        Path to the shelter import CSV (Stage 6A / 6B format).
    municipalities_gpkg:
        Path to the municipalities WGS-84 GeoPackage.
    graph_graphml:
        Path to the road-graph GraphML file for the target municipality.
    target_psgc:
        PSGC code of the target municipality (e.g. "PH0600613").
    warn_m:
        Snap distance warning threshold in metres.
    reject_m:
        Snap distance rejection threshold in metres.
    abort_on_validation_errors:
        If True, raise ValueError when any schema validation error is found.
        Default False: errors are collected and the pipeline continues.

    Returns
    -------
    ShelterSnapReport with all intermediate and final results.

    Raises
    ------
    ValueError:
        When abort_on_validation_errors=True and schema errors are found.
    FileNotFoundError:
        When any required file is missing.
    """
    for p in (shelter_csv, municipalities_gpkg, graph_graphml):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Step 1 — Load and validate
    logger.info("Loading shelter CSV: %s", shelter_csv)
    raw_rows = load_shelter_csv(shelter_csv)
    records, validation_errors = validate_shelter_records(raw_rows)

    if validation_errors:
        for msg in validation_errors:
            logger.warning("Validation error: %s", msg)
        if abort_on_validation_errors:
            raise ValueError(
                f"{len(validation_errors)} shelter record validation error(s). "
                "Fix errors before continuing."
            )

    logger.info(
        "Validated %d record(s); %d error(s).", len(records), len(validation_errors)
    )

    # Step 2 — Containment
    logger.info("Loading municipal boundary: %s (PSGC=%s)", municipalities_gpkg, target_psgc)
    muni_geom = load_municipality_polygon(municipalities_gpkg, target_psgc)
    containment_results = check_all_containment(records, muni_geom)
    outside_boundary = [r.shelter_id for r in containment_results if not r.inside and r.distance_to_boundary_m is not None]
    if outside_boundary:
        logger.warning(
            "%d candidate(s) outside municipal boundary: %s",
            len(outside_boundary),
            outside_boundary,
        )

    # Step 3 — Graph snapping
    logger.info("Loading road graph: %s", graph_graphml)
    graph: nx.MultiDiGraph = nx.read_graphml(graph_graphml)
    logger.info("Graph loaded: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges())
    snap_results = snap_all_shelters(records, graph, warn_m=warn_m, reject_m=reject_m)

    n_accepted = sum(1 for r in snap_results if r.accepted)
    n_no_ent = sum(1 for r in snap_results if r.status == "no_entrance")
    n_rejected = sum(1 for r in snap_results if r.status == "rejected")
    logger.info(
        "Snap results: %d accepted, %d no_entrance, %d rejected.",
        n_accepted, n_no_ent, n_rejected,
    )

    # Step 4 — Readiness
    readiness = evaluate_shelter_readiness(records)
    logger.info("Readiness decision: %s", readiness.decision.value)

    return ShelterSnapReport(
        records=records,
        validation_errors=validation_errors,
        containment_results=containment_results,
        snap_results=snap_results,
        readiness=readiness,
        outside_boundary=outside_boundary,
    )
