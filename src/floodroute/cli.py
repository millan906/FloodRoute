"""FloodRoute command-line interface.

Stage 0: validate-config is fully operational.
Other commands exit with an informative error until their data
dependencies are satisfied in later thesis stages.
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

# Default paths resolved relative to the package install root
_DEFAULT_CONFIGS = Path(__file__).parent.parent.parent / "configs"


def _configs_dir_option(default: Path = _DEFAULT_CONFIGS) -> Path:
    return default


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
# inspect-data
# ---------------------------------------------------------------------------


@app.command("inspect-data")
def inspect_data(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Path to the data/ directory."),
    ] = Path("data"),
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

    subdirs = ["raw", "interim", "processed"]
    for sub in subdirs:
        sub_path = data_dir / sub
        if not sub_path.is_dir():
            typer.echo(f"  {sub}/  — directory missing")
            continue
        files = [f for f in sub_path.iterdir() if f.name != ".gitkeep"]
        typer.echo(f"  {sub}/  — {len(files)} file(s)")

    manifests = (
        list((data_dir / "manifests").glob("*.yaml")) if (data_dir / "manifests").is_dir() else []
    )
    typer.echo(f"  manifests/  — {len(manifests)} manifest(s)")

    if all(
        len([f for f in (data_dir / sub).iterdir() if f.name != ".gitkeep"]) == 0
        for sub in subdirs
        if (data_dir / sub).is_dir()
    ):
        typer.echo(
            "\nNo case-study data has been ingested yet (Stage 0 — expected).",
            err=False,
        )


# ---------------------------------------------------------------------------
# build-graph
# ---------------------------------------------------------------------------


@app.command("build-graph")
def build_graph(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Build the road-network graph from processed data.

    Requires: processed road-network data (available in Stage 1+).
    """
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("build-graph called but no processed data is available (Stage 0).")
    typer.echo(
        "ERROR: build-graph requires processed road-network data which has not yet been "
        "ingested. Run data ingestion (Stage 1+) first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-analysis
# ---------------------------------------------------------------------------


@app.command("run-analysis")
def run_analysis(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Run the core routing and shelter-allocation analysis.

    Requires: processed data and a built graph (available in Stage 2+).
    """
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("run-analysis called but data layer is absent (Stage 0).")
    typer.echo(
        "ERROR: run-analysis requires the processed data layer and a built graph. "
        "Complete Stage 1 data ingestion and Stage 2 graph construction first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-experiment
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
    """Execute a named experiment defined in experiments.yaml.

    Requires: all prior stages complete (available in Stage 3+).
    """
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("run-experiment called but prerequisites are absent (Stage 0).")
    typer.echo(
        "ERROR: run-experiment requires the full analysis pipeline (Stages 1–3). "
        "No results will be generated in Stage 0.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
