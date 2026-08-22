"""Unit tests for the Typer CLI entrypoints.

Tests use the Typer test runner (which wraps Click's CliRunner) and
do not invoke real data processing.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from floodroute.cli import app

runner = CliRunner()

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


# ---------------------------------------------------------------------------
# CLI startup
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero():
    """Running with --help exits 0 and prints usage."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "floodroute" in result.output.lower() or "Usage" in result.output


def test_validate_config_with_real_configs():
    """validate-config succeeds against the shipped YAML files."""
    if not CONFIGS_DIR.is_dir():
        import pytest

        pytest.skip("configs/ directory not found relative to test location")

    result = runner.invoke(app, ["validate-config", "--configs-dir", str(CONFIGS_DIR)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_config_missing_dir_exits_nonzero():
    """validate-config exits 1 when the configs directory does not exist."""
    result = runner.invoke(app, ["validate-config", "--configs-dir", "/nonexistent/path"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Commands that require absent data must exit non-zero
# ---------------------------------------------------------------------------


def test_build_graph_exits_nonzero():
    """build-graph exits 1 because no processed data exists in Stage 0."""
    result = runner.invoke(app, ["build-graph"])
    assert result.exit_code == 1
    assert "ERROR" in result.output or "error" in result.output.lower()


def test_run_analysis_exits_nonzero():
    """run-analysis exits 1 because no data layer exists in Stage 0."""
    result = runner.invoke(app, ["run-analysis"])
    assert result.exit_code == 1


def test_run_experiment_exits_nonzero():
    """run-experiment exits 1 because no pipeline exists in Stage 0."""
    result = runner.invoke(app, ["run-experiment"])
    assert result.exit_code == 1
