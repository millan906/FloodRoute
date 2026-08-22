"""FloodRoute command-line interface.

Stage 0: validate-config is fully operational.
Stage 1: list-datasets, validate-manifests, acquire-dataset, verify-dataset.
Later commands exit with a clear error until their data dependencies exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from floodroute.logging_config import configure_logging, get_logger

app = typer.Typer(
    name="floodroute",
    help="Flood-resilient evacuation routing and shelter allocation (MSCS thesis).",
    no_args_is_help=True,
)

logger = get_logger("cli")

# Default paths resolved relative to the installed package root.
# resolve() is required when the package is imported through a symlink
# (e.g., .venv/site-packages/floodroute -> src/floodroute) so that
# __file__ is de-symlinked before computing the project root.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIGS = _PKG_ROOT / "configs"
_DEFAULT_DATA = _PKG_ROOT / "data"
_DEFAULT_MANIFESTS = _DEFAULT_DATA / "manifests"


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------


@app.command("validate-config")
def validate_config(
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", "-c", help="Path to the configs/ directory."),
    ] = _DEFAULT_CONFIGS,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate all YAML configuration files against their Pydantic schemas."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("Validating configs in %s", configs_dir)

    if not configs_dir.is_dir():
        typer.echo(f"ERROR: configs directory not found: {configs_dir}", err=True)
        raise typer.Exit(code=1)

    from pydantic import ValidationError

    from floodroute.config import validate_all_configs

    try:
        cfg = validate_all_configs(configs_dir)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: missing config file — {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValidationError as exc:
        typer.echo(f"ERROR: config validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("All configuration files are valid.")
    n_scenarios = len(cfg.scenarios.scenarios)
    n_experiments = len(cfg.experiments.experiments)
    n_shelters = len(cfg.shelters.shelters)
    typer.echo(f"  scenarios={n_scenarios}  experiments={n_experiments}  shelters={n_shelters}")
    logger.info("Config validation complete.")


# ---------------------------------------------------------------------------
# inspect-data  (Stage 1: manifest-aware)
# ---------------------------------------------------------------------------


@app.command("inspect-data")
def inspect_data(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Path to the data/ directory."),
    ] = _DEFAULT_DATA,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Report on data availability and manifest status."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("Inspecting data directory: %s", data_dir)

    if not data_dir.is_dir():
        typer.echo(f"ERROR: data directory not found: {data_dir}", err=True)
        raise typer.Exit(code=1)

    for sub in ("raw", "interim", "processed"):
        sub_path = data_dir / sub
        if not sub_path.is_dir():
            typer.echo(f"  {sub}/  — directory missing")
        else:
            files = [f for f in sub_path.iterdir() if f.name != ".gitkeep"]
            typer.echo(f"  {sub}/  — {len(files)} file(s)")

    # Manifest-aware readiness (never crashes on missing manifests)
    manifests_dir = data_dir / "manifests"
    try:
        from floodroute.manifest import load_all_manifests
        from floodroute.readiness import build_readiness_report, format_report_text

        manifests = load_all_manifests(manifests_dir)
        if manifests:
            entries = build_readiness_report(manifests, data_dir)
            typer.echo(f"\n  manifests/  — {len(manifests)} manifest(s)\n")
            typer.echo(format_report_text(entries))
        else:
            typer.echo("  manifests/  — 0 manifest(s)")
            typer.echo("\nNo manifests loaded. Add dataset manifests to data/manifests/.")
    except Exception as exc:
        logger.warning("Could not load manifests: %s", exc)
        typer.echo(f"  manifests/  — could not load ({exc})")


# ---------------------------------------------------------------------------
# list-datasets
# ---------------------------------------------------------------------------


@app.command("list-datasets")
def list_datasets(
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """List all registered datasets with their acquisition and validation status."""
    configure_logging(log_level)  # type: ignore[arg-type]

    from pydantic import ValidationError

    from floodroute.manifest import load_all_manifests
    from floodroute.readiness import build_readiness_report, format_report_json, format_report_text

    try:
        manifests = load_all_manifests(manifests_dir)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (ValidationError, ValueError) as exc:
        typer.echo(f"ERROR: manifest error — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not manifests:
        typer.echo("No manifest files found in: " + str(manifests_dir))
        return

    data_dir = manifests_dir.parent
    entries = build_readiness_report(manifests, data_dir)

    if output_format == "json":
        typer.echo(format_report_json(entries))
    else:
        typer.echo(format_report_text(entries))


# ---------------------------------------------------------------------------
# validate-manifests
# ---------------------------------------------------------------------------


@app.command("validate-manifests")
def validate_manifests(
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate all manifest YAML files against the DatasetManifest schema."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("Validating manifests in %s", manifests_dir)

    if not manifests_dir.is_dir():
        typer.echo(f"ERROR: manifests directory not found: {manifests_dir}", err=True)
        raise typer.Exit(code=1)

    from pydantic import ValidationError

    from floodroute.manifest import load_manifest

    yaml_files = sorted(manifests_dir.glob("*.yaml"))
    if not yaml_files:
        typer.echo("No *.yaml files found in: " + str(manifests_dir))
        return

    errors: list[str] = []
    seen_ids: dict[str, str] = {}

    for path in yaml_files:
        try:
            m = load_manifest(path)
            if m.dataset_id in seen_ids:
                errors.append(
                    f"  DUPLICATE  {path.name}: dataset_id '{m.dataset_id}' "
                    f"already used in '{seen_ids[m.dataset_id]}'"
                )
            else:
                seen_ids[m.dataset_id] = path.name
                typer.echo(f"  OK         {path.name}  ({m.dataset_id})")
        except (ValidationError, Exception) as exc:
            errors.append(f"  FAIL       {path.name}: {exc}")

    if errors:
        typer.echo("\nValidation errors:", err=True)
        for e in errors:
            typer.echo(e, err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"\nAll {len(seen_ids)} manifest(s) valid.")


# ---------------------------------------------------------------------------
# acquire-dataset
# ---------------------------------------------------------------------------


@app.command("acquire-dataset")
def acquire_dataset(
    dataset_id: Annotated[
        str,
        typer.Argument(help="Dataset ID as declared in a manifest YAML."),
    ],
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Root data directory."),
    ] = _DEFAULT_DATA,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing local file."),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="HTTP request timeout in seconds."),
    ] = 30,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Download a single dataset by its manifest ID.

    Manual-acquisition datasets print instructions instead of downloading.
    There is no bulk acquire-all command; every download requires an explicit ID.
    """
    configure_logging(log_level)  # type: ignore[arg-type]

    from pydantic import ValidationError

    from floodroute.acquisition import (
        ChecksumMismatch,
        FileAlreadyExists,
        ManualAcquisitionRequired,
        download_dataset,
    )
    from floodroute.manifest import load_all_manifests

    try:
        manifests = load_all_manifests(manifests_dir)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"ERROR: manifest error — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if dataset_id not in manifests:
        typer.echo(f"ERROR: dataset_id '{dataset_id}' not found.", err=True)
        typer.echo(f"Known IDs: {', '.join(sorted(manifests))}", err=True)
        raise typer.Exit(code=1)

    manifest = manifests[dataset_id]

    try:
        dest = download_dataset(manifest, data_dir, force=force, timeout=timeout)
        typer.echo(f"Acquired: {dest}")
    except ManualAcquisitionRequired as exc:
        typer.echo("\n" + exc.instructions)
        raise typer.Exit(code=0) from None
    except FileAlreadyExists as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        typer.echo("Use --force to overwrite.", err=True)
        raise typer.Exit(code=1) from exc
    except ChecksumMismatch as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"ERROR: download failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# verify-dataset
# ---------------------------------------------------------------------------


@app.command("verify-dataset")
def verify_dataset(
    dataset_id: Annotated[
        str,
        typer.Argument(help="Dataset ID to verify."),
    ],
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Root data directory."),
    ] = _DEFAULT_DATA,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Verify the SHA-256 checksum of a locally acquired dataset."""
    configure_logging(log_level)  # type: ignore[arg-type]

    from pydantic import ValidationError

    from floodroute.acquisition import compute_sha256, resolve_local_path
    from floodroute.manifest import load_all_manifests

    try:
        manifests = load_all_manifests(manifests_dir)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"ERROR: manifest error — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if dataset_id not in manifests:
        typer.echo(f"ERROR: dataset_id '{dataset_id}' not found.", err=True)
        raise typer.Exit(code=1)

    manifest = manifests[dataset_id]
    path = resolve_local_path(manifest, data_dir)

    if path is None:
        typer.echo("ERROR: No local_path defined in manifest; cannot verify.", err=True)
        raise typer.Exit(code=1)

    if not path.exists():
        typer.echo(f"ERROR: Local file not found: {path}", err=True)
        raise typer.Exit(code=1)

    if manifest.sha256 is None:
        actual = compute_sha256(path)
        typer.echo("WARNING: No expected SHA-256 in manifest — recording actual value only.")
        typer.echo(f"  Actual SHA-256 : {actual}")
        typer.echo("  Record this value in the manifest once the file is trusted.")
        raise typer.Exit(code=0)

    actual = compute_sha256(path)
    if actual == manifest.sha256:
        typer.echo(f"OK  {dataset_id}  sha256={actual}")
    else:
        typer.echo(
            f"FAIL  {dataset_id}\n  expected : {manifest.sha256}\n  actual   : {actual}",
            err=True,
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# build-graph  (Stage 2+)
# ---------------------------------------------------------------------------


@app.command("build-graph")
def build_graph(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Build the road-network graph from processed data (Stage 2+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("build-graph called but no processed data is available.")
    typer.echo(
        "ERROR: build-graph requires processed road-network data. "
        "Complete Stage 1 data ingestion first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-analysis  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-analysis")
def run_analysis(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Run the core routing and shelter-allocation analysis (Stage 3+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("run-analysis called but data layer is absent.")
    typer.echo(
        "ERROR: run-analysis requires the processed data layer and a built graph. "
        "Complete Stages 1–2 first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-experiment  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-experiment")
def run_experiment(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help="Experiment ID from experiments.yaml."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Execute a named experiment (Stage 3+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("run-experiment called but prerequisites are absent.")
    typer.echo(
        "ERROR: run-experiment requires the full analysis pipeline (Stages 1–3). "
        "No results will be generated yet.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
