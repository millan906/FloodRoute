"""Smoke and variable-recording tests for the Experimental Evaluation entry-point.

These tests verify that:
- The module imports cleanly and render_experiment_page is a callable.
- The formal experiment variables (algorithm, return_period, demand_fraction,
  capacity_multiplier, flood_penalty, nominal_capacities) reach ExperimentConfig.
- save_experiment records all formal variables in manifest.json.

Pure-Python — no Streamlit, no geospatial execution.
"""
from __future__ import annotations

import inspect
import json


def test_render_experiment_page_is_importable():
    """render_experiment_page must be importable from experiment_page."""
    from floodroute.dashboard.experiment_page import render_experiment_page  # noqa: F401


def test_render_experiment_page_is_callable():
    """render_experiment_page must be a callable."""
    from floodroute.dashboard.experiment_page import render_experiment_page
    assert callable(render_experiment_page)


def test_render_experiment_page_signature():
    """render_experiment_page must accept G and origins positional parameters."""
    from floodroute.dashboard.experiment_page import render_experiment_page
    sig = inspect.signature(render_experiment_page)
    params = list(sig.parameters)
    assert "G" in params, "render_experiment_page must have a 'G' parameter"
    assert "origins" in params, "render_experiment_page must have an 'origins' parameter"


def test_experiment_page_module_exposes_save_experiment():
    """experiment_page must re-export save_experiment (used internally)."""
    import floodroute.dashboard.experiment_page as ep
    assert hasattr(ep, "save_experiment")


def test_experiment_page_runner_constants_importable():
    """Dataset path constants from runner must be accessible via experiment_page import chain."""
    from pathlib import Path

    from floodroute.experiments.runner import (
        _DEFAULT_BARANGAY_GPKG,
        _DEFAULT_GRAPHML,
        _DEFAULT_NODES_GPKG,
        _DEFAULT_POP_CSV,
    )
    assert isinstance(_DEFAULT_GRAPHML, Path)
    assert isinstance(_DEFAULT_POP_CSV, Path)
    assert isinstance(_DEFAULT_BARANGAY_GPKG, Path)
    assert isinstance(_DEFAULT_NODES_GPKG, Path)


# ---------------------------------------------------------------------------
# Formal variable capture — ExperimentConfig
# ---------------------------------------------------------------------------


def test_experiment_config_records_algorithm():
    """make_experiment_config must store the selected algorithms tuple."""
    from floodroute.experiments.runner import make_experiment_config
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A", "B+"),
        return_periods=("RP20",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 5000},
    )
    assert set(cfg.algorithms) == {"A", "B+"}


def test_experiment_config_records_return_period():
    """make_experiment_config must store the selected return periods."""
    from floodroute.experiments.runner import make_experiment_config
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP10", "RP100"),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.00,),
        flood_penalties=(5.0,),
        nominal_capacities={33: 5000},
    )
    assert set(cfg.return_periods) == {"RP10", "RP100"}


def test_experiment_config_records_demand_fraction():
    """make_experiment_config must store the selected demand fractions."""
    from floodroute.experiments.runner import make_experiment_config
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP20",),
        demand_fractions=(0.10, 0.50),
        capacity_multipliers=(1.00,),
        flood_penalties=(1.0,),
        nominal_capacities={33: 5000},
    )
    assert set(cfg.demand_fractions) == {0.10, 0.50}


def test_experiment_config_records_capacity_multiplier():
    """make_experiment_config must store the selected capacity multipliers."""
    from floodroute.experiments.runner import make_experiment_config
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("B",),
        return_periods=("RP100",),
        demand_fractions=(0.25,),
        capacity_multipliers=(0.75, 1.25),
        flood_penalties=(10.0,),
        nominal_capacities={33: 5000},
    )
    assert set(cfg.capacity_multipliers) == {0.75, 1.25}


def test_experiment_config_records_flood_penalty():
    """make_experiment_config must store the flood-deterrence penalty values."""
    from floodroute.experiments.runner import make_experiment_config
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("B+",),
        return_periods=("RP10",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(20.0, "prohibited"),
        nominal_capacities={33: 5000},
    )
    assert set(cfg.flood_penalties) == {20.0, "prohibited"}


def test_experiment_config_records_nominal_capacities():
    """make_experiment_config must store the facility node→capacity mapping."""
    from floodroute.experiments.runner import make_experiment_config
    caps = {7: 3000, 42: 8000}
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP20",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities=caps,
    )
    assert cfg.nominal_capacities == caps


# ---------------------------------------------------------------------------
# Manifest records all formal experiment variables
# ---------------------------------------------------------------------------


def _make_minimal_scenario_result():
    """Return a minimal ScenarioResult-like object for manifest tests."""
    from dataclasses import dataclass

    @dataclass
    class _MinResult:
        key: object
        metrics: dict
        assignments: dict
        route_metrics: dict
        facility_metrics: dict
        unassigned_reasons: dict
        barangay_demand: dict
        nominal_capacities: dict
        adjusted_capacities: dict
        run_status: str = "ok"
        error_message: str = ""

    @dataclass
    class _Key:
        algorithm: str = "C"
        return_period: str = "RP20"
        demand_fraction: float = 0.25
        capacity_multiplier: float = 1.00
        flood_penalty: object = 10.0

    return _MinResult(
        key=_Key(),
        metrics={"total_assigned": 0, "total_demand": 0},
        assignments={},
        route_metrics={},
        facility_metrics={},
        unassigned_reasons={},
        barangay_demand={},
        nominal_capacities={33: 5000},
        adjusted_capacities={33: 5000},
    )


def test_manifest_records_algorithms(tmp_path):
    """manifest.json must include the algorithms list from the experiment config."""
    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A", "B", "B+", "C"),
        return_periods=("RP100",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 5000},
    )
    result = _make_minimal_scenario_result()
    exp_dir = save_experiment(cfg, [result], experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert set(manifest["algorithms"]) == {"A", "B", "B+", "C"}


def test_manifest_records_return_periods(tmp_path):
    """manifest.json must include the return_periods list."""
    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP10", "RP100"),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 5000},
    )
    result = _make_minimal_scenario_result()
    exp_dir = save_experiment(cfg, [result], experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert set(manifest["return_periods"]) == {"RP10", "RP100"}


def test_manifest_records_flood_penalties(tmp_path):
    """manifest.json must include the flood_penalties (flood-deterrence multipliers)."""
    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("B+",),
        return_periods=("RP20",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(5.0, 20.0),
        nominal_capacities={33: 5000},
    )
    result = _make_minimal_scenario_result()
    exp_dir = save_experiment(cfg, [result], experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    # manifest stores as strings
    assert set(manifest["flood_penalties"]) == {"5.0", "20.0"}


def test_manifest_records_nominal_capacities(tmp_path):
    """manifest.json must include the facility node→capacity mapping."""
    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    caps = {7: 3000, 42: 8000}
    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP100",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities=caps,
    )
    result = _make_minimal_scenario_result()
    exp_dir = save_experiment(cfg, [result], experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    # nominal_capacities is stored as {str(node): capacity}
    assert manifest["nominal_capacities"] == {"7": 3000, "42": 8000}


def test_manifest_records_demand_fractions_and_capacity_multipliers(tmp_path):
    """manifest.json must record demand_fractions and capacity_multipliers."""
    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP10",),
        demand_fractions=(0.10, 0.50),
        capacity_multipliers=(0.75, 1.25),
        flood_penalties=(1.0,),
        nominal_capacities={33: 5000},
    )
    result = _make_minimal_scenario_result()
    exp_dir = save_experiment(cfg, [result], experiments_root=tmp_path)
    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert set(manifest["demand_fractions"]) == {0.10, 0.50}
    assert set(manifest["capacity_multipliers"]) == {0.75, 1.25}


def test_experiment_tab_wired_in_app():
    """app.py must contain 'with experiment_tab:' and 'render_experiment_page'."""
    from pathlib import Path
    app_path = Path(__file__).parent.parent.parent / "src" / "floodroute" / "dashboard" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "with experiment_tab:" in source, "experiment_tab block must be present in app.py"
    assert "render_experiment_page" in source, "render_experiment_page must be called in app.py"


# ---------------------------------------------------------------------------
# unassigned_reasons.csv — full/partial coverage and reconciliation
# ---------------------------------------------------------------------------


def _make_fake_origin(origin_node, psgc, name):
    """Return a lightweight BarangayOrigin-like object."""
    from dataclasses import dataclass

    @dataclass
    class _FakeOrigin:
        origin_node: int
        psgc: str
        name: str

    return _FakeOrigin(origin_node=origin_node, psgc=psgc, name=name)


def _make_scenario_result_with_demands(
    *,
    algorithm="C",
    demands: dict,
    assignments: dict,
    unassigned_reasons: dict,
):
    """Build a real ScenarioResult for unassigned_reasons.csv tests."""
    from floodroute.experiments.runner import ScenarioKey, ScenarioResult

    total_demand = sum(demands.values())
    total_assigned = sum(v for v in assignments.values() if v > 0)
    total_unassigned = total_demand - total_assigned

    key = ScenarioKey(
        algorithm=algorithm,
        return_period="RP20",
        demand_fraction=0.25,
        capacity_multiplier=1.00,
        flood_penalty=10.0,
    )
    return ScenarioResult(
        key=key,
        nominal_capacities={33: 5000},
        adjusted_capacities={33: 5000},
        metrics={
            "total_demand": total_demand,
            "assigned_population": total_assigned,
            "unassigned_population": total_unassigned,
        },
        assignments=assignments,
        route_metrics={},
        facility_metrics={},
        unassigned_reasons=unassigned_reasons,
        run_status="completed",
        error_message=None,
        barangay_demand={},
        demands=demands,
    )


def test_unassigned_reasons_csv_fully_unreachable_origin(tmp_path):
    """A fully unreachable origin must produce one row with reason='unreachable' and OSM note."""
    import csv as _csv

    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP20",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 5000},
    )
    # Origin 10 has demand 50 but is fully unreachable (no assignments, in unassigned_reasons)
    sr = _make_scenario_result_with_demands(
        demands={10: 50},
        assignments={},
        unassigned_reasons={10: "unreachable"},
    )
    origins = [_make_fake_origin(10, "PH0600613001", "Barangay Poblacion")]
    exp_dir = save_experiment(cfg, [sr], experiments_root=tmp_path, origins=origins)

    rows = list(_csv.DictReader((exp_dir / "unassigned_reasons.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["origin_node"]) == 10
    assert row["adm4_pcode"] == "PH0600613001"
    assert row["barangay_name"] == "Barangay Poblacion"
    assert int(row["scenario_demand"]) == 50
    assert int(row["assigned_count"]) == 0
    assert int(row["unassigned_count"]) == 50
    assert row["reason"] == "unreachable"
    assert row["osm_network_note"] != ""  # OSM note must be present for unreachable


def test_unassigned_reasons_csv_partially_assigned_origin(tmp_path):
    """A partially assigned origin must appear with reason='capacity_exhausted' and no OSM note."""
    import csv as _csv

    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP20",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 30},
    )
    # Origin 7 has demand 50; only 30 assigned (capacity exhausted)
    # Not in unassigned_reasons because it IS partially assigned
    sr = _make_scenario_result_with_demands(
        demands={7: 50},
        assignments={(7, 33): 30},
        unassigned_reasons={},
    )
    origins = [_make_fake_origin(7, "PH0600613002", "Barangay San Jose")]
    exp_dir = save_experiment(cfg, [sr], experiments_root=tmp_path, origins=origins)

    rows = list(_csv.DictReader((exp_dir / "unassigned_reasons.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["origin_node"]) == 7
    assert row["adm4_pcode"] == "PH0600613002"
    assert row["barangay_name"] == "Barangay San Jose"
    assert int(row["scenario_demand"]) == 50
    assert int(row["assigned_count"]) == 30
    assert int(row["unassigned_count"]) == 20
    assert row["reason"] == "capacity_exhausted"
    assert row["osm_network_note"] == ""  # No OSM note for capacity_exhausted


def test_unassigned_reasons_csv_reconciliation(tmp_path):
    """sum(unassigned_count) must equal total_unassigned for every ScenarioResult."""
    import csv as _csv

    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("C",),
        return_periods=("RP20",),
        demand_fractions=(0.25,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 40},
    )
    # Three origins:
    #   node 1: demand=20, assigned=20 (fully assigned → skip)
    #   node 2: demand=30, assigned=0  (fully unreachable)
    #   node 3: demand=40, assigned=15 (partially assigned)
    # total_unassigned = (30 - 0) + (40 - 15) = 55
    sr = _make_scenario_result_with_demands(
        demands={1: 20, 2: 30, 3: 40},
        assignments={(1, 33): 20, (3, 33): 15},
        unassigned_reasons={2: "unreachable"},
    )
    exp_dir = save_experiment(cfg, [sr], experiments_root=tmp_path)

    rows = list(_csv.DictReader((exp_dir / "unassigned_reasons.csv").open(encoding="utf-8")))
    # Only nodes 2 and 3 must appear
    assert len(rows) == 2
    total_csv_unassigned = sum(int(r["unassigned_count"]) for r in rows)
    assert total_csv_unassigned == sr.metrics["unassigned_population"]
    assert total_csv_unassigned == 55


def test_unassigned_reasons_csv_fully_assigned_origin_omitted(tmp_path):
    """Origins with zero unassigned demand must not appear in the CSV."""
    import csv as _csv

    from floodroute.experiments.manifest import save_experiment
    from floodroute.experiments.runner import make_experiment_config

    cfg = make_experiment_config(
        municipality="PH0600613",
        algorithms=("A",),
        return_periods=("RP10",),
        demand_fractions=(0.10,),
        capacity_multipliers=(1.00,),
        flood_penalties=(10.0,),
        nominal_capacities={33: 100},
    )
    # All demand assigned
    sr = _make_scenario_result_with_demands(
        demands={5: 25, 6: 15},
        assignments={(5, 33): 25, (6, 33): 15},
        unassigned_reasons={},
    )
    exp_dir = save_experiment(cfg, [sr], experiments_root=tmp_path)

    rows = list(_csv.DictReader((exp_dir / "unassigned_reasons.csv").open(encoding="utf-8")))
    assert rows == [], "No rows expected when all origins are fully assigned"
