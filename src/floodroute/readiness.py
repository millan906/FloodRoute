"""Data-readiness report for FloodRoute Stage 1.

Summarises the acquisition and validation state of every registered
dataset manifest, including local-file existence and checksum status.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from floodroute.acquisition import compute_sha256, resolve_local_path
from floodroute.manifest import DatasetManifest


@dataclass
class ReadinessEntry:
    dataset_id: str
    category: str
    acquisition_status: str
    validation_status: str
    file_exists: bool
    checksum_matches: bool | None  # None = not checked (no sha256 or file absent)
    blocking_issue: str | None


def build_readiness_report(
    manifests: dict[str, DatasetManifest],
    data_dir: Path,
) -> list[ReadinessEntry]:
    """Return one ReadinessEntry per manifest, in dataset_id order."""
    entries: list[ReadinessEntry] = []

    for manifest in manifests.values():
        local = resolve_local_path(manifest, data_dir)
        file_exists = local is not None and local.exists()

        checksum_matches: bool | None = None
        if file_exists and manifest.sha256 is not None and local is not None:
            checksum_matches = compute_sha256(local) == manifest.sha256

        entries.append(
            ReadinessEntry(
                dataset_id=manifest.dataset_id,
                category=manifest.category.value,
                acquisition_status=manifest.acquisition_status.value,
                validation_status=manifest.validation_status.value,
                file_exists=file_exists,
                checksum_matches=checksum_matches,
                blocking_issue=_blocking(manifest, file_exists, checksum_matches),
            )
        )

    return sorted(entries, key=lambda e: e.dataset_id)


def _blocking(
    manifest: DatasetManifest,
    file_exists: bool,
    checksum_matches: bool | None,
) -> str | None:
    if manifest.acquisition_status.value == "unavailable":
        return "Dataset marked unavailable"
    if manifest.acquisition_status.value == "acquired" and not file_exists:
        return "Manifest says acquired but local file is missing"
    if checksum_matches is False:
        return "SHA-256 checksum mismatch"
    if manifest.validation_status.value == "rejected":
        return "Validation rejected"
    return None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_report_text(entries: list[ReadinessEntry]) -> str:
    """Format as a human-readable aligned table."""
    if not entries:
        return "No dataset manifests found."

    col_id = max((len(e.dataset_id) for e in entries), default=10)
    col_cat = max((len(e.category) for e in entries), default=8)
    col_acq = max((len(e.acquisition_status) for e in entries), default=10)
    col_val = max((len(e.validation_status) for e in entries), default=13)

    # Apply minimums matching column headers
    col_id = max(col_id, 10)
    col_cat = max(col_cat, 8)
    col_acq = max(col_acq, 10)
    col_val = max(col_val, 13)

    hdr = (
        f"{'DATASET_ID':<{col_id}}  {'CATEGORY':<{col_cat}}  "
        f"{'ACQ_STATUS':<{col_acq}}  {'VALIDATION_STATUS':<{col_val}}  "
        f"{'FILE':<4}  {'CHK':<4}  BLOCKING"
    )
    rows = [hdr, "-" * (len(hdr) + 20)]

    for e in entries:
        file_s = "yes" if e.file_exists else "no"
        chk_s = {True: "ok", False: "FAIL", None: "-"}[e.checksum_matches]
        rows.append(
            f"{e.dataset_id:<{col_id}}  {e.category:<{col_cat}}  "
            f"{e.acquisition_status:<{col_acq}}  {e.validation_status:<{col_val}}  "
            f"{file_s:<4}  {chk_s:<4}  {e.blocking_issue or ''}"
        )

    return "\n".join(rows)


def format_report_json(entries: list[ReadinessEntry]) -> str:
    """Format as a JSON array."""
    return json.dumps([asdict(e) for e in entries], indent=2)
