"""Unit tests for configuration loading and validation.

All fixtures are synthetic toy data in tests/fixtures/.
No real case-study data is used or required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Valid configuration loading
# ---------------------------------------------------------------------------


def test_valid_municipality_loads():
    """A correctly formed municipality YAML loads without error."""
    from floodroute.config import load_municipality_config

    cfg = load_municipality_config(FIXTURES / "municipality_valid.yaml")
    assert cfg.schema_version == "1.0"
    assert cfg.municipality.name == "Testville"
    assert cfg.municipality.population.value == 10000
    assert cfg.municipality.population.provenance == "observed"
    assert cfg.municipality.crs == "EPSG:3310"


def test_valid_shelters_null_capacity_loads():
    """A shelter with null capacity loads without error and preserves null."""
    from floodroute.config import load_shelters_config

    cfg = load_shelters_config(FIXTURES / "shelters_null_capacity.yaml")
    assert cfg.schema_version == "1.0"
    assert len(cfg.shelters) == 1
    shelter = cfg.shelters[0]
    assert shelter.id == "TEST_SHELTER_A"
    # Null capacity must not be replaced by any default
    assert shelter.capacity.value is None


def test_valid_experiment_seed_loaded():
    """Seed field is loaded exactly as declared in the fixture."""
    from floodroute.config import load_experiments_config

    cfg = load_experiments_config(FIXTURES / "experiments_seeded.yaml")
    assert cfg.experiments[0].seed == 42


# ---------------------------------------------------------------------------
# Invalid provenance values
# ---------------------------------------------------------------------------


def test_invalid_provenance_rejected():
    """A YAML with an unrecognised provenance value must raise ValidationError."""
    from floodroute.config import load_municipality_config

    with pytest.raises(ValidationError) as exc_info:
        load_municipality_config(FIXTURES / "municipality_bad_provenance.yaml")

    # Confirm the error mentions the bad value
    errors = exc_info.value.errors()
    provenance_errors = [e for e in errors if "provenance" in str(e.get("loc", ""))]
    assert provenance_errors, f"Expected a provenance error, got: {errors}"


# ---------------------------------------------------------------------------
# Deterministic seed handling
# ---------------------------------------------------------------------------


def test_seed_is_integer():
    """Seed loaded from fixture is an integer, not a string or float."""
    from floodroute.config import load_experiments_config

    cfg = load_experiments_config(FIXTURES / "experiments_seeded.yaml")
    seed = cfg.experiments[0].seed
    assert isinstance(seed, int), f"Expected int, got {type(seed)}"
    assert seed == 42


def test_seed_determines_random_output():
    """Python random module produces the same output given the same seed."""
    import random

    from floodroute.config import load_experiments_config

    cfg = load_experiments_config(FIXTURES / "experiments_seeded.yaml")
    seed = cfg.experiments[0].seed

    random.seed(seed)
    first_run = [random.random() for _ in range(5)]

    random.seed(seed)
    second_run = [random.random() for _ in range(5)]

    assert first_run == second_run


# ---------------------------------------------------------------------------
# Missing dataset errors
# ---------------------------------------------------------------------------


def test_missing_config_file_raises():
    """Loading a non-existent config file raises FileNotFoundError."""
    from floodroute.config import load_municipality_config

    with pytest.raises(FileNotFoundError):
        load_municipality_config(FIXTURES / "does_not_exist.yaml")


def test_validate_all_configs_missing_dir_raises():
    """validate_all_configs raises when the configs directory does not exist."""
    from floodroute.config import validate_all_configs

    with pytest.raises(FileNotFoundError):
        validate_all_configs(Path("/nonexistent/configs"))
