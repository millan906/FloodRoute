"""Focused tests for saved-scenario deletion in the Evacuation Planner.

Tests cover:
  1.  Deleting the selected scenario removes only that file.
  2.  Unrelated (non-selected) presets are not touched.
  3.  Session-state sc_id is cleared when it matches the deleted scenario.
  4.  Session-state sc_id is preserved when a different scenario is deleted.
  5.  delete_scenario is a no-op if the file is already absent (no exception).
  6.  list_scenarios no longer lists the deleted scenario after deletion.
  7.  delete_scenario does not remove files outside configs/scenario_presets/.
  8.  Multiple unrelated scenarios all survive deletion of one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from floodroute.scenario.persistence import (
    delete_scenario,
    list_scenarios,
    new_scenario_id,
    save_scenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_preset(tmp_dir: Path, sid: str, name: str) -> Path:
    """Write a minimal valid scenario JSON and return the file path."""
    data: dict = {
        "schema_version": "1.0",
        "scenario_id": sid,
        "scenario_name": name,
        "selected_facility_ids": [],
        "facility_capacities": {},
    }
    dest = tmp_dir / f"scenario_{sid}.json"
    dest.write_text(json.dumps(data), encoding="utf-8")
    return dest


def _simulate_session_clear(session: dict, deleted_id: str) -> None:
    """Replicate the dashboard's session-clearing logic for unit testing.

    Mirrors the branch in app.py:

        if st.session_state.get("sc_id") == _del_id:
            st.session_state["sc_id"] = None
            st.session_state["sc_dirty"] = False
        st.session_state["sc_confirm_delete"] = None
    """
    if session.get("sc_id") == deleted_id:
        session["sc_id"] = None
        session["sc_dirty"] = False
    session["sc_confirm_delete"] = None


# ---------------------------------------------------------------------------
# Test 1: Selected scenario file is deleted
# ---------------------------------------------------------------------------


def test_delete_removes_selected_scenario_file(tmp_path):
    """delete_scenario must unlink the target scenario's JSON file."""
    sid = new_scenario_id()
    path = _write_preset(tmp_path, sid, "Target")
    assert path.exists(), "Precondition: file must exist before deletion"

    delete_scenario(sid, scenarios_dir=tmp_path)

    assert not path.exists(), "Target scenario file must be removed"


# ---------------------------------------------------------------------------
# Test 2: Unrelated presets are not touched
# ---------------------------------------------------------------------------


def test_delete_does_not_touch_other_presets(tmp_path):
    """Deleting one scenario must leave all other scenario files intact."""
    sid_target = new_scenario_id()
    sid_other_a = new_scenario_id()
    sid_other_b = new_scenario_id()

    _write_preset(tmp_path, sid_target, "Target")
    path_a = _write_preset(tmp_path, sid_other_a, "Other A")
    path_b = _write_preset(tmp_path, sid_other_b, "Other B")

    delete_scenario(sid_target, scenarios_dir=tmp_path)

    assert path_a.exists(), "Other A must survive deletion of Target"
    assert path_b.exists(), "Other B must survive deletion of Target"


# ---------------------------------------------------------------------------
# Test 3: sc_id is cleared when it matches the deleted scenario
# ---------------------------------------------------------------------------


def test_session_sc_id_cleared_when_matches_deleted(tmp_path):
    """Session sc_id must become None when the active scenario is deleted."""
    sid = new_scenario_id()
    _write_preset(tmp_path, sid, "Active")

    session = {"sc_id": sid, "sc_dirty": True, "sc_confirm_delete": sid}

    delete_scenario(sid, scenarios_dir=tmp_path)
    _simulate_session_clear(session, sid)

    assert session["sc_id"] is None, "sc_id must be cleared after deleting the loaded scenario"
    assert session["sc_dirty"] is False, "sc_dirty must be cleared alongside sc_id"
    assert session["sc_confirm_delete"] is None, "sc_confirm_delete must be reset"


# ---------------------------------------------------------------------------
# Test 4: sc_id is preserved when a different scenario is deleted
# ---------------------------------------------------------------------------


def test_session_sc_id_preserved_when_different_scenario_deleted(tmp_path):
    """Session sc_id must remain unchanged when a non-active scenario is deleted."""
    sid_loaded = new_scenario_id()
    sid_other = new_scenario_id()

    _write_preset(tmp_path, sid_loaded, "Loaded")
    _write_preset(tmp_path, sid_other, "Other")

    session = {"sc_id": sid_loaded, "sc_dirty": False, "sc_confirm_delete": sid_other}

    delete_scenario(sid_other, scenarios_dir=tmp_path)
    _simulate_session_clear(session, sid_other)

    assert session["sc_id"] == sid_loaded, (
        "sc_id must remain the loaded scenario ID when a different preset is deleted"
    )
    assert session["sc_dirty"] is False, "sc_dirty must not change"
    assert session["sc_confirm_delete"] is None, "sc_confirm_delete must be reset"


# ---------------------------------------------------------------------------
# Test 5: delete_scenario is a no-op for absent files
# ---------------------------------------------------------------------------


def test_delete_scenario_no_op_when_file_absent(tmp_path):
    """delete_scenario must not raise if the target file does not exist."""
    sid = new_scenario_id()
    absent = tmp_path / f"scenario_{sid}.json"
    assert not absent.exists(), "Precondition: file must not exist"

    # Must not raise
    delete_scenario(sid, scenarios_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test 6: list_scenarios no longer lists the deleted scenario
# ---------------------------------------------------------------------------


def test_list_scenarios_excludes_deleted(tmp_path):
    """list_scenarios must not return the deleted scenario after deletion."""
    sid_keep = new_scenario_id()
    sid_delete = new_scenario_id()

    data_keep = {
        "schema_version": "1.0",
        "scenario_id": sid_keep,
        "scenario_name": "Keep",
        "selected_facility_ids": [],
        "facility_capacities": {},
    }
    data_delete = {
        "schema_version": "1.0",
        "scenario_id": sid_delete,
        "scenario_name": "Delete",
        "selected_facility_ids": [],
        "facility_capacities": {},
    }
    save_scenario(data_keep, scenarios_dir=tmp_path)
    save_scenario(data_delete, scenarios_dir=tmp_path)

    before = {s["scenario_id"] for s in list_scenarios(scenarios_dir=tmp_path)}
    assert sid_keep in before and sid_delete in before, "Both must exist before deletion"

    delete_scenario(sid_delete, scenarios_dir=tmp_path)

    after = {s["scenario_id"] for s in list_scenarios(scenarios_dir=tmp_path)}
    assert sid_delete not in after, "Deleted scenario must not appear in list_scenarios"
    assert sid_keep in after, "Surviving scenario must still appear in list_scenarios"


# ---------------------------------------------------------------------------
# Test 7: delete_scenario does not touch files outside the presets directory
# ---------------------------------------------------------------------------


def test_delete_scenario_confined_to_presets_dir(tmp_path):
    """delete_scenario must not remove files in sibling directories."""
    presets_dir = tmp_path / "scenario_presets"
    presets_dir.mkdir()
    other_dir = tmp_path / "experiments"
    other_dir.mkdir()

    sid = new_scenario_id()
    _write_preset(presets_dir, sid, "Preset")

    # A file in the sibling directory with the same naming pattern
    sibling_file = other_dir / f"scenario_{sid}.json"
    sibling_file.write_text("{}", encoding="utf-8")

    delete_scenario(sid, scenarios_dir=presets_dir)

    assert not (presets_dir / f"scenario_{sid}.json").exists(), (
        "Target file in presets_dir must be deleted"
    )
    assert sibling_file.exists(), (
        "File in sibling directory must not be touched"
    )


# ---------------------------------------------------------------------------
# Test 8: multiple unrelated scenarios all survive deletion of one
# ---------------------------------------------------------------------------


def test_multiple_unrelated_scenarios_survive(tmp_path):
    """Deleting one scenario from a set of five must leave the other four."""
    sids = [new_scenario_id() for _ in range(5)]
    for i, sid in enumerate(sids):
        _write_preset(tmp_path, sid, f"Scenario {i}")

    target = sids[2]
    delete_scenario(target, scenarios_dir=tmp_path)

    surviving_files = list(tmp_path.glob("scenario_*.json"))
    surviving_ids = {p.stem.removeprefix("scenario_") for p in surviving_files}

    assert target not in surviving_ids, "Deleted scenario must be gone"
    for sid in sids:
        if sid != target:
            assert sid in surviving_ids, f"Scenario {sid} must survive"
