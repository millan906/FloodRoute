"""Stage 11 evidence bundle: atomic experiment directory with manifest and CSVs.

Saves a completed experiment to ``experiments/<experiment_id>/`` with:
- ``manifest.json``        — provenance, config, dataset inventory, file hashes
- ``dataset_inventory.json`` — input dataset paths and SHA-256 checksums
- ``scenario_results.csv`` — one row per scenario (metrics flattened)
- ``assignments.csv``      — per-origin-shelter flow records
- ``route_metrics.csv``    — physical/flooded/penalized distances per pair
- ``facility_metrics.csv`` — per-shelter load/capacity/utilization/overflow
- ``unassigned_reasons.csv`` — origin-level unassigned reason decomposition
- ``checksums.sha256``     — shasum-format checksums for all files

Write is atomic: files are written to a temp directory
``experiments/<id>_tmp_<pid>`` and then renamed to ``experiments/<id>``.

The ``experiment_id`` is unique (timestamp + config hash), so a legitimate
rerun of the same configuration always creates a new directory.  A warning
is emitted (not an error) if another experiment with the same configuration
hash already exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path


def _project_root() -> Path:
    """Return the project root directory (four parents up from this file)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _git_info() -> dict:
    """Return git provenance dict; graceful fallback if git unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "dirty": bool(dirty_out)}
    except Exception:  # noqa: BLE001
        return {"commit": "unknown", "dirty": False}


def _package_versions() -> dict:
    """Return a dict of relevant package versions."""
    versions: dict[str, str] = {}
    versions["python"] = sys.version.split()[0]
    for pkg in ("networkx", "floodroute"):
        try:
            import importlib.metadata as _meta

            versions[pkg] = _meta.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = "unknown"
    return versions


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_inventory(dataset_paths: dict[str, Path]) -> list[dict]:
    """Build a dataset inventory list with path and SHA-256 for each input.

    Parameters
    ----------
    dataset_paths:
        Mapping of dataset role → absolute Path.  Standard keys:
        ``population_csv``, ``road_graph``, ``hazard_rp10``, ``hazard_rp20``,
        ``hazard_rp100``, ``facility_candidates``, ``barangay_boundaries``,
        ``nodes_gpkg``, ``capacity_config`` (optional, omit if not file-backed).

    Returns
    -------
    list[dict]
        One record per dataset with keys: ``role``, ``path``, ``sha256``,
        ``exists`` (bool), and ``note`` (only on missing files).
    """
    inventory: list[dict] = []
    for role, path in dataset_paths.items():
        entry: dict = {"role": role, "path": str(path)}
        if Path(path).exists():
            entry["sha256"] = _sha256_file(Path(path))
            entry["exists"] = True
        else:
            entry["sha256"] = None
            entry["exists"] = False
            entry["note"] = "file not found at save time"
        inventory.append(entry)
    return inventory


def _write_scenario_results_csv(path: Path, results: list) -> None:
    """Write one row per ScenarioResult with metrics flattened."""
    if not results:
        path.write_text("", encoding="utf-8")
        return

    rows = []
    for sr in results:
        k = sr.key
        row: dict = {
            "algorithm": k.algorithm,
            "return_period": k.return_period,
            "demand_fraction": k.demand_fraction,
            "capacity_multiplier": k.capacity_multiplier,
            "flood_penalty": str(k.flood_penalty),
            "run_status": sr.run_status,
            "error_message": sr.error_message or "",
        }
        # Flatten metrics — skip nested dicts (shelter_overflow etc.)
        for key_m, val_m in sr.metrics.items():
            if isinstance(val_m, dict):
                continue
            row[key_m] = val_m
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_assignments_csv(path: Path, results: list) -> None:
    fieldnames = [
        "algorithm", "return_period", "demand_fraction", "capacity_multiplier",
        "flood_penalty", "origin_node", "shelter_node", "units",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sr in results:
            k = sr.key
            for (o, s), units in sorted(sr.assignments.items(), key=str):
                writer.writerow({
                    "algorithm": k.algorithm,
                    "return_period": k.return_period,
                    "demand_fraction": k.demand_fraction,
                    "capacity_multiplier": k.capacity_multiplier,
                    "flood_penalty": str(k.flood_penalty),
                    "origin_node": o,
                    "shelter_node": s,
                    "units": units,
                })


def _write_route_metrics_csv(path: Path, results: list) -> None:
    fieldnames = [
        "algorithm", "rp", "frac", "mult", "penalty",
        "origin_node", "shelter_node", "physical_m", "flooded_m", "penalized_m_eq",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sr in results:
            k = sr.key
            for (o, s), rm in sorted(sr.route_metrics.items(), key=str):
                writer.writerow({
                    "algorithm": k.algorithm,
                    "rp": k.return_period,
                    "frac": k.demand_fraction,
                    "mult": k.capacity_multiplier,
                    "penalty": str(k.flood_penalty),
                    "origin_node": o,
                    "shelter_node": s,
                    "physical_m": rm.get("physical_m", 0.0),
                    "flooded_m": rm.get("flooded_m", 0.0),
                    "penalized_m_eq": rm.get("penalized_m_eq", 0.0),
                })


def _write_facility_metrics_csv(path: Path, results: list) -> None:
    fieldnames = [
        "algorithm", "rp", "frac", "mult", "penalty",
        "shelter_node", "nominal_capacity", "adjusted_capacity",
        "load", "utilization", "overflow",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sr in results:
            k = sr.key
            for s, fm in sorted(sr.facility_metrics.items(), key=str):
                writer.writerow({
                    "algorithm": k.algorithm,
                    "rp": k.return_period,
                    "frac": k.demand_fraction,
                    "mult": k.capacity_multiplier,
                    "penalty": str(k.flood_penalty),
                    "shelter_node": s,
                    "nominal_capacity": sr.nominal_capacities.get(s, ""),
                    "adjusted_capacity": sr.adjusted_capacities.get(s, fm.get("capacity", "")),
                    "load": fm.get("load", 0),
                    "utilization": fm.get("utilization", 0.0),
                    "overflow": fm.get("overflow", 0),
                })


def _write_unassigned_reasons_csv(path: Path, results: list) -> None:
    fieldnames = [
        "algorithm", "rp", "frac", "mult", "penalty",
        "origin_node", "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sr in results:
            k = sr.key
            for o, reason in sorted(sr.unassigned_reasons.items(), key=str):
                writer.writerow({
                    "algorithm": k.algorithm,
                    "rp": k.return_period,
                    "frac": k.demand_fraction,
                    "mult": k.capacity_multiplier,
                    "penalty": str(k.flood_penalty),
                    "origin_node": o,
                    "reason": reason,
                })


def save_experiment(
    config: object,  # ExperimentConfig
    results: list,
    *,
    experiments_root: Path | None = None,
    dataset_paths: dict[str, Path] | None = None,
) -> Path:
    """Atomically save a completed experiment to ``experiments/<experiment_id>/``.

    Each call creates a fresh directory because ``experiment_id`` embeds a UTC
    timestamp.  A warning (not an error) is issued if another experiment with
    the same ``configuration_hash`` already exists — this records that the
    configuration is being re-run intentionally.

    Parameters
    ----------
    config:
        ``ExperimentConfig`` instance describing the experiment.
    results:
        List of ``ScenarioResult`` objects from ``run_experiment``.
    experiments_root:
        Override the default experiments directory (project root /
        ``experiments/``).  Useful for testing.
    dataset_paths:
        Optional mapping of dataset role → Path for input provenance
        (see :func:`build_dataset_inventory`).  Pass ``None`` to omit
        the inventory (acceptable for unit tests).

    Returns
    -------
    Path
        Path to the saved experiment directory.
    """
    if experiments_root is None:
        experiments_root = _project_root() / "experiments"

    experiments_root.mkdir(parents=True, exist_ok=True)

    # Warn (do not raise) if another experiment with the same configuration
    # hash already exists — a new experiment_id is always unique.
    config_hash = getattr(config, "configuration_hash", None)
    if config_hash is not None:
        prior = [
            d for d in experiments_root.iterdir()
            if d.is_dir() and d.name.endswith(f"_{config_hash}")
            and d.name != config.experiment_id
        ]
        if prior:
            warnings.warn(
                f"Re-running configuration {config_hash!r}: "
                f"{len(prior)} prior experiment(s) exist with the same hash "
                f"({', '.join(d.name for d in prior[:3])}).",
                stacklevel=2,
            )

    dest_dir = experiments_root / config.experiment_id
    # experiment_id includes a timestamp, so collision is extremely unlikely;
    # guard against it anyway.
    if dest_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {dest_dir}. "
            "This should not happen — experiment_id includes a UTC timestamp."
        )

    tmp_dir = experiments_root / f"{config.experiment_id}_tmp_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        git = _git_info()
        pkg_versions = _package_versions()
        saved_utc = datetime.now(UTC).isoformat()

        # Dataset inventory
        inventory: list[dict] = []
        if dataset_paths is not None:
            inventory = build_dataset_inventory(dataset_paths)
        (tmp_dir / "dataset_inventory.json").write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write CSV files
        _write_scenario_results_csv(tmp_dir / "scenario_results.csv", results)
        _write_assignments_csv(tmp_dir / "assignments.csv", results)
        _write_route_metrics_csv(tmp_dir / "route_metrics.csv", results)
        _write_facility_metrics_csv(tmp_dir / "facility_metrics.csv", results)
        _write_unassigned_reasons_csv(tmp_dir / "unassigned_reasons.csv", results)

        # Compute file hashes (all outputs except checksums file)
        output_files = [
            "dataset_inventory.json",
            "scenario_results.csv",
            "assignments.csv",
            "route_metrics.csv",
            "facility_metrics.csv",
            "unassigned_reasons.csv",
        ]
        file_hashes = {fname: _sha256_file(tmp_dir / fname) for fname in output_files}

        # Build and write manifest
        manifest = {
            "experiment_id": config.experiment_id,
            "configuration_hash": getattr(config, "configuration_hash", None),
            "created_utc": config.created_utc,
            "saved_utc": saved_utc,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "municipality": config.municipality,
            "algorithms": list(config.algorithms),
            "return_periods": list(config.return_periods),
            "demand_fractions": list(config.demand_fractions),
            "capacity_multipliers": list(config.capacity_multipliers),
            "flood_penalties": [str(p) for p in config.flood_penalties],
            "nominal_capacities": {str(k): v for k, v in config.nominal_capacities.items()},
            "pilot_mode": config.pilot_mode,
            "python_version": pkg_versions["python"],
            "networkx_version": pkg_versions.get("networkx", "unknown"),
            "package_versions": pkg_versions,
            "dataset_inventory": inventory,
            "file_hashes": file_hashes,
        }
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Add manifest hash
        file_hashes["manifest.json"] = _sha256_file(manifest_path)

        # Write checksums.sha256
        checksums_path = tmp_dir / "checksums.sha256"
        with checksums_path.open("w", encoding="utf-8") as fh:
            for fname in output_files + ["manifest.json"]:
                fh.write(f"{file_hashes[fname]}  {fname}\n")

        # Atomic rename
        os.rename(tmp_dir, dest_dir)
        return dest_dir

    except Exception:
        # Clean up temp dir on failure
        import shutil

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
