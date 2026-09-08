"""Default session-state factory for the FloodRoute Evacuation Planner.

``make_default_session_state()`` is the single source of truth for every
Planner session-state key and its canonical startup value.  It is called in
two places inside ``app.py``:

1. **Initial app load** — populates only keys that do not yet exist in
   ``st.session_state`` so that a fresh browser session starts cleanly.
2. **"Reset entire scenario"** — overwrites all transient working state with
   fresh defaults, returning the Planner to exactly the same condition as a
   first visit.

Because every call returns a *new* dict with *new* container objects ({}, [],
set()), there is no aliasing between the defaults list and live session state.
"""
from __future__ import annotations


def make_default_session_state() -> dict:
    """Return a dict mapping every Planner session-state key to its default.

    All mutable values (``{}``, ``[]``, ``set()``) are freshly constructed on
    each call so callers can assign directly to ``st.session_state`` without
    risk of sharing state between resets or between sessions.
    """
    return {
        # ── Run result ────────────────────────────────────────────────────
        "result": None,
        "metrics": None,
        "run_params": None,
        "run_error": None,
        "run_scenario_key": None,
        "ordinary_path": None,
        "alg_c_path": None,
        "alg_c_assigned_shelter": None,
        "alg_c_origin_status": None,
        # ── Road conditions ───────────────────────────────────────────────
        "road_override_store_dict": {},
        "road_name_conditions": {},
        "step4_last_applied": None,
        # ── Scenario identity ─────────────────────────────────────────────
        "sc_name": "Untitled scenario",
        "sc_id": None,
        "sc_dirty": False,
        "sc_unresolved_fids": [],
        "sc_confirm_delete": None,
        # ── Facility selection ────────────────────────────────────────────
        "sc_selected_fids": [],
        "sc_facility_caps": {},
        "activated_osm_ids": set(),
        "scenario_capacities": {},
        # ── Map / origin highlight ────────────────────────────────────────
        "selected_origin_node": None,
        "_fit_bounds": False,
        # ── Road editor ───────────────────────────────────────────────────
        "road_editor_road": None,
        "road_editor_cond": None,
        "highlighted_override_road": None,
        "_pending_clear_road_search": False,
        "_confirm_remove_road": None,
        "_confirm_clear_roads": False,
        # ── Dialog / confirmation flags ───────────────────────────────────
        "_confirm_scenario_reset": False,
        # ── Streamlit widget keys ─────────────────────────────────────────
        # Setting these before the widget renders causes the widget to show
        # the intended default value.  The guard in the initial-load loop
        # (``if _k not in st.session_state``) prevents overwriting an
        # already-rendered widget's live value.
        "all_barangays_check": True,
        "barangay_multiselect": [],
        "demand_type_radio": "25%",
        "demand_exact_value": 0,
    }
