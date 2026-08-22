"""Unit tests for the Typer CLI entrypoints.

Tests use the Typer test runner (which wraps Click's CliRunner) and
do not invoke real data processing or network calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from floodroute.cli import app

runner = CliRunner()

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
MANIFESTS_DIR = Path(__file__).parent.parent.parent / "data" / "manifests"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# CLI startup
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "floodroute" in result.output.lower() or "Usage" in result.output


def test_cli_help_lists_stage1_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list-datasets" in result.output
    assert "validate-manifests" in result.output
    assert "acquire-dataset" in result.output
    assert "verify-dataset" in result.output


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------


def test_validate_config_with_real_configs():
    if not CONFIGS_DIR.is_dir():
        pytest.skip("configs/ not found")
    result = runner.invoke(app, ["validate-config", "--configs-dir", str(CONFIGS_DIR)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_config_missing_dir_exits_nonzero():
    result = runner.invoke(app, ["validate-config", "--configs-dir", "/nonexistent/path"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# validate-manifests (Stage 1)
# ---------------------------------------------------------------------------


def test_validate_manifests_with_real_manifests():
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    result = runner.invoke(app, ["validate-manifests", "--manifests-dir", str(MANIFESTS_DIR)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_manifests_missing_dir_exits_nonzero():
    result = runner.invoke(app, ["validate-manifests", "--manifests-dir", "/nonexistent/manifests"])
    assert result.exit_code == 1


def test_validate_manifests_with_bad_file_exits_nonzero(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: '1.0'\ndataset_id: x\ntitle: T\n"
        "category: invalid_cat\naccess_method: direct_download\n"
        "acquisition_status: not_requested\nprovenance: observed\n"
        "validation_status: not_validated\n"
    )
    result = runner.invoke(app, ["validate-manifests", "--manifests-dir", str(tmp_path)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# list-datasets (Stage 1)
# ---------------------------------------------------------------------------


def test_list_datasets_with_real_manifests():
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    result = runner.invoke(app, ["list-datasets", "--manifests-dir", str(MANIFESTS_DIR)])
    assert result.exit_code == 0, result.output
    # Output should contain at least one dataset_id
    assert "osm_philippines" in result.output or len(result.output) > 10


def test_list_datasets_json_output():
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    import json

    result = runner.invoke(
        app,
        ["list-datasets", "--manifests-dir", str(MANIFESTS_DIR), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert all("dataset_id" in entry for entry in parsed)


def test_list_datasets_empty_dir_exits_zero(tmp_path):
    result = runner.invoke(app, ["list-datasets", "--manifests-dir", str(tmp_path)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# acquire-dataset (Stage 1)
# ---------------------------------------------------------------------------


def test_acquire_dataset_unknown_id_exits_nonzero():
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    result = runner.invoke(
        app,
        [
            "acquire-dataset",
            "nonexistent_id_xyz",
            "--manifests-dir",
            str(MANIFESTS_DIR),
        ],
    )
    assert result.exit_code == 1


def test_acquire_dataset_manual_method_prints_instructions():
    """shelter dataset requires local_authority — CLI prints instructions, exits 0."""
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    result = runner.invoke(
        app,
        [
            "acquire-dataset",
            "sjdb_evacuation_shelters",
            "--manifests-dir",
            str(MANIFESTS_DIR),
            "--data-dir",
            str(DATA_DIR),
        ],
    )
    # Manual acquisition prints instructions and exits 0
    assert result.exit_code == 0
    assert "sjdb_evacuation_shelters" in result.output or "Manual" in result.output


def test_acquire_dataset_direct_download_mocked(tmp_path):
    """direct_download dataset with mocked network succeeds.

    Uses a synthetic manifest written to tmp_path so the test is not coupled
    to sha256 values in the real dataset manifests (which now carry checksums
    of the actual downloaded files).
    """
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "test_direct.yaml").write_text(
        "schema_version: '1.0'\ndataset_id: test_direct\ntitle: T\n"
        "category: road_network\naccess_method: direct_download\n"
        "source_url: 'http://example.test/data.csv'\nlocal_path: 'raw/test.csv'\n"
        "acquisition_status: not_requested\nprovenance: observed\n"
        "validation_status: not_validated\n"
    )
    data = b"mocked road network data"

    def fake_stream(url, part, *, timeout):
        part.write_bytes(data)

    with patch("floodroute.acquisition._stream_download", side_effect=fake_stream):
        result = runner.invoke(
            app,
            [
                "acquire-dataset",
                "test_direct",
                "--manifests-dir",
                str(manifests_dir),
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Acquired" in result.output


# ---------------------------------------------------------------------------
# verify-dataset (Stage 1)
# ---------------------------------------------------------------------------


def test_verify_dataset_missing_file_exits_nonzero(tmp_path):
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    result = runner.invoke(
        app,
        [
            "verify-dataset",
            "osm_philippines",
            "--manifests-dir",
            str(MANIFESTS_DIR),
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1


def test_verify_dataset_unknown_id_exits_nonzero(tmp_path):
    if not MANIFESTS_DIR.is_dir():
        pytest.skip("data/manifests/ not found")
    result = runner.invoke(
        app,
        [
            "verify-dataset",
            "totally_unknown_id",
            "--manifests-dir",
            str(MANIFESTS_DIR),
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# inspect-data (Stage 1 — manifest-aware)
# ---------------------------------------------------------------------------


def test_inspect_data_with_real_data_dir():
    if not DATA_DIR.is_dir():
        pytest.skip("data/ directory not found")
    result = runner.invoke(app, ["inspect-data", "--data-dir", str(DATA_DIR)])
    assert result.exit_code == 0, result.output


def test_inspect_data_missing_dir_exits_nonzero():
    result = runner.invoke(app, ["inspect-data", "--data-dir", "/nonexistent/data"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Commands that still require absent data
# ---------------------------------------------------------------------------


def test_build_graph_exits_nonzero():
    result = runner.invoke(app, ["build-graph"])
    assert result.exit_code == 1


def test_run_analysis_exits_nonzero():
    result = runner.invoke(app, ["run-analysis"])
    assert result.exit_code == 1


def test_run_experiment_exits_nonzero():
    result = runner.invoke(app, ["run-experiment"])
    assert result.exit_code == 1
