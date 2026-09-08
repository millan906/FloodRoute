"""Streamlit AppTest regression tests for the All facilities checkbox sync.

These tests verify that fac_all_toggle and every individual fsel_{fid} widget
key stay consistent with the canonical sc_selected_fids list across all
interaction sequences.

Note: at.session_state is SafeSessionState — use [] and ``in`` for access,
not .get() (SafeSessionState does not implement the dict .get() protocol).
"""
from __future__ import annotations

import textwrap

import pytest

_MINI_APP = textwrap.dedent("""\
    import streamlit as st
    from floodroute.scenario.catalog import build_catalog
    from floodroute.scenario.config import ScenarioConfig
    from floodroute.dashboard.scenario_ui import render_facility_selection

    catalog = build_catalog()
    sc = ScenarioConfig(municipality="PH0600613", return_period="RP100")
    render_facility_selection(catalog, sc, None)
""")


def _all_selectable():
    """Return selectable CatalogEntry objects in deterministic order."""
    from floodroute.scenario.catalog import build_catalog
    return [e for e in build_catalog().all() if e.can_be_selected]


def _make_at():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(_MINI_APP, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial render raised: {at.exception}"
    return at


def _ss(at, key, default=None):
    """Safe read from at.session_state (SafeSessionState has no .get())."""
    return at.session_state[key] if key in at.session_state else default  # noqa: SIM401


class TestAllFacilitiesCheckboxSync:
    """Interaction regression: All facilities toggle syncs every individual widget."""

    def test_initial_state_nothing_selected(self):
        at = _make_at()
        entries = _all_selectable()
        assert len(entries) > 0

        # Nothing selected initially
        assert _ss(at, "sc_selected_fids", []) == []
        assert _ss(at, "fac_all_toggle", False) is False

        # Pre-render sync writes fsel_ keys to False on first render
        for e in entries:
            val = _ss(at, f"fsel_{e.facility_id}", False)
            assert val is False, (
                f"fsel_{e.facility_id} should start False, got {val}"
            )

    def test_click_all_facilities_checks_every_individual_widget(self):
        """Clicking All facilities must set every fsel_{fid} to True."""
        at = _make_at()
        entries = _all_selectable()
        n = len(entries)
        assert n > 0, "Catalog must have selectable facilities"

        at.checkbox(key="fac_all_toggle").set_value(True).run()
        assert not at.exception, f"Raised after All facilities click: {at.exception}"

        for e in entries:
            key = f"fsel_{e.facility_id}"
            val = at.session_state[key]
            assert val is True, f"Expected {key}=True after All facilities, got {val}"

        assert len(at.session_state["sc_selected_fids"]) == n

    def test_summary_count_correct_after_select_all(self):
        """sc_selected_fids length must equal number of selectable facilities."""
        at = _make_at()
        entries = _all_selectable()

        at.checkbox(key="fac_all_toggle").set_value(True).run()
        assert not at.exception

        assert len(at.session_state["sc_selected_fids"]) == len(entries)

    def test_uncheck_one_after_select_all_updates_individual_and_toggle(self):
        """After Select all, unchecking one facility makes that facility False
        and fac_all_toggle False."""
        at = _make_at()
        entries = _all_selectable()
        n = len(entries)

        at.checkbox(key="fac_all_toggle").set_value(True).run()
        assert not at.exception

        first = entries[0]
        fkey = f"fsel_{first.facility_id}"
        at.checkbox(key=fkey).set_value(False).run()
        assert not at.exception

        assert at.session_state[fkey] is False, (
            f"{fkey} must be False after explicit uncheck"
        )
        assert at.session_state["fac_all_toggle"] is False, (
            "fac_all_toggle must be False when not all facilities are selected"
        )
        assert first.facility_id not in at.session_state["sc_selected_fids"]
        assert len(at.session_state["sc_selected_fids"]) == n - 1

    def test_uncheck_all_facilities_clears_every_individual_widget(self):
        """Unchecking All facilities must set every fsel_{fid} to False."""
        at = _make_at()
        entries = _all_selectable()

        at.checkbox(key="fac_all_toggle").set_value(True).run()
        assert not at.exception
        at.checkbox(key="fac_all_toggle").set_value(False).run()
        assert not at.exception

        for e in entries:
            key = f"fsel_{e.facility_id}"
            val = at.session_state[key]
            assert val is False, f"Expected {key}=False after deselect all, got {val}"

        assert at.session_state["sc_selected_fids"] == []

    def test_capacity_values_preserved_after_deselect_all(self):
        """Capacity values stored in sc_facility_caps must survive deselection
        so reselection restores the previous capacity without user re-entry."""
        at = _make_at()

        # Select all (populates sc_facility_caps with defaults)
        at.checkbox(key="fac_all_toggle").set_value(True).run()
        assert not at.exception

        saved_caps = dict(at.session_state["sc_facility_caps"])
        assert saved_caps, "sc_facility_caps should be non-empty after Select all"

        # Deselect all
        at.checkbox(key="fac_all_toggle").set_value(False).run()
        assert not at.exception

        caps_after = at.session_state["sc_facility_caps"]
        for fid, cap in saved_caps.items():
            actual = caps_after.get(fid)
            assert actual == cap, (
                f"Capacity for {fid} should be preserved: expected {cap}, "
                f"got {actual}"
            )

    def test_manual_select_all_makes_toggle_checked(self):
        """Manually checking every individual facility must make fac_all_toggle True."""
        at = _make_at()
        entries = _all_selectable()

        if len(entries) < 2:
            pytest.skip("Need at least 2 selectable facilities for this test")

        for e in entries:
            at.checkbox(key=f"fsel_{e.facility_id}").set_value(True).run()
            assert not at.exception, (
                f"Raised after checking {e.facility_id}: {at.exception}"
            )

        assert at.session_state["fac_all_toggle"] is True, (
            "fac_all_toggle must be True when all individual facilities are selected"
        )
        assert len(at.session_state["sc_selected_fids"]) == len(entries)
