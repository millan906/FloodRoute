"""Step 4 facility selection panel for the Evacuation Planner.

Renders the unified facility selection UI backed by ScenarioConfig and FacilityCatalog.
All state is managed via st.session_state keys sc_selected_fids and sc_facility_caps.

Widget synchronisation
----------------------
Two checkbox hierarchies must stay in sync:

  fac_all_toggle          — the "All facilities" master toggle
  fsel_{facility_id}      — one per selectable facility

The canonical selection lives in ``sc_selected_fids`` (session_state list).
``on_change`` callbacks update the canonical list *and* the peer widget keys
before the script body renders any widget, so every checkbox reflects the
true selection state on every run.

For external state changes (scenario load, st.rerun() from app.py), a
pre-render sync block corrects any stale widget keys before the first
``st.checkbox`` call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from floodroute.scenario.catalog import CatalogEntry, FacilityCatalog
    from floodroute.scenario.config import ScenarioConfig

# Group display order and plain labels
_FACILITY_TYPE_ORDER = [
    "evacuation_center",
    "barangay_hall",
    "school",
    "covered_court",
    "multi_purpose_hall",
]

_FACILITY_TYPE_PLAIN: dict[str, str] = {
    "evacuation_center": "Evacuation-labelled facilities",
    "barangay_hall": "Barangay halls",
    "school": "Schools",
    "covered_court": "Covered courts",
    "multi_purpose_hall": "Multi-purpose halls",
}

_SJDB001_ID = "SJDB-001"


# ── on_change callbacks ───────────────────────────────────────────────────────
# Must be module-level so Streamlit can pickle them for the callback registry.
# They read _sc_all_selectable_fids from session_state (written in the render
# function before any widget is instantiated).

def _on_all_fac_toggle() -> None:
    """on_change for the 'All facilities' master checkbox."""
    new_val: bool = st.session_state["fac_all_toggle"]
    all_selectable: list[str] = list(
        st.session_state.get("_sc_all_selectable_fids") or []
    )
    selected: list[str] = list(st.session_state.get("sc_selected_fids") or [])

    if new_val:
        for fid in all_selectable:
            if fid not in selected:
                selected.append(fid)
            st.session_state[f"fsel_{fid}"] = True
    else:
        selected = [f for f in selected if f not in all_selectable]
        for fid in all_selectable:
            st.session_state[f"fsel_{fid}"] = False
    # Capacity values are intentionally NOT cleared so they survive deselection.
    st.session_state["sc_selected_fids"] = selected
    st.session_state["sc_dirty"] = True


def _on_fac_toggle(fid: str) -> None:
    """on_change for an individual facility checkbox."""
    checked: bool = st.session_state[f"fsel_{fid}"]
    selected: list[str] = list(st.session_state.get("sc_selected_fids") or [])

    if checked and fid not in selected:
        selected.append(fid)
    elif not checked and fid in selected:
        selected.remove(fid)
        # Capacity is intentionally NOT cleared so it survives deselection.

    st.session_state["sc_selected_fids"] = selected
    st.session_state["sc_dirty"] = True

    # Keep fac_all_toggle in sync — can write to it here because it was
    # rendered earlier this run and we are inside a callback (callbacks run
    # before the next render, so this value is used on that next render).
    all_selectable: list[str] = list(
        st.session_state.get("_sc_all_selectable_fids") or []
    )
    all_now = bool(all_selectable) and all(f in selected for f in all_selectable)
    st.session_state["fac_all_toggle"] = all_now


# ─────────────────────────────────────────────────────────────────────────────

def _designation_label(entry: CatalogEntry) -> str:
    """Return a plain-language designation label for one catalog entry."""
    if entry.facility_id == _SJDB001_ID:
        return (
            "Government-confirmed evacuation center · "
            "Official capacity not yet verified · "
            "Experimental scenario capacity"
        )
    if entry.source == "facility_registry":
        if entry.designation_status == "government_confirmed":
            return "Government-confirmed facility"
        if entry.designation_status == "historically_activated":
            return "Historically activated · not currently confirmed"
        return "Registry facility · unverified designation"
    # OSM candidate
    return "Candidate — not officially designated"


def _capacity_label(entry: CatalogEntry) -> str:  # noqa: ARG001
    """Capacity input label — always 'Experimental scenario capacity'."""
    return "Experimental scenario capacity"


def render_facility_selection(
    catalog: FacilityCatalog,
    scenario: ScenarioConfig,
    origin_nodes: set[int] | None,
    last_reachable_nodes: set[int] | None = None,
) -> tuple[ScenarioConfig, bool, str]:
    """Render facility selection UI.

    Reads/writes sc_selected_fids and sc_facility_caps in session_state.
    Returns (updated_scenario, is_blocked, block_reason).
    """
    from floodroute.scenario.config import ScenarioConfig

    # ── Canonical state ───────────────────────────────────────────────────────
    all_entries = catalog.all()
    all_selectable_fids: list[str] = [
        e.facility_id for e in all_entries if e.can_be_selected
    ]

    if "sc_selected_fids" not in st.session_state:
        st.session_state["sc_selected_fids"] = []
    if "sc_facility_caps" not in st.session_state:
        st.session_state["sc_facility_caps"] = {}

    selected_fids: list[str] = list(st.session_state["sc_selected_fids"])
    facility_caps: dict[str, int] = dict(st.session_state["sc_facility_caps"])

    # ── Store selectable fids for callbacks ───────────────────────────────────
    # Written before any widget is instantiated so that callbacks can read it.
    st.session_state["_sc_all_selectable_fids"] = all_selectable_fids

    # ── PRE-RENDER SYNC ───────────────────────────────────────────────────────
    # Correct any widget keys that are stale relative to canonical sc_selected_fids.
    # Happens when sc_selected_fids was changed externally (scenario load, app.py
    # st.rerun()) without updating the individual fsel_ or fac_all_toggle keys.
    # Safe: no widget has been rendered yet in this run.
    _all_checked_now = bool(all_selectable_fids) and all(
        f in selected_fids for f in all_selectable_fids
    )
    if st.session_state.get("fac_all_toggle") != _all_checked_now:
        st.session_state["fac_all_toggle"] = _all_checked_now
    for _fid in all_selectable_fids:
        _expected = _fid in selected_fids
        if st.session_state.get(f"fsel_{_fid}") != _expected:
            st.session_state[f"fsel_{_fid}"] = _expected

    # ── Warning banner ────────────────────────────────────────────────────────
    st.caption(
        "Candidate facilities from OpenStreetMap are not automatically official "
        "LGU-designated evacuation centers. Capacities marked experimental are "
        "scenario assumptions."
    )

    # ── "All facilities" master toggle ────────────────────────────────────────
    # on_change callback updates all fsel_ keys and sc_selected_fids atomically
    # before the individual checkboxes are rendered.
    st.checkbox(
        "All facilities",
        key="fac_all_toggle",
        on_change=_on_all_fac_toggle,
    )

    # Re-read after potential callback mutation
    selected_fids = list(st.session_state["sc_selected_fids"])
    facility_caps = dict(st.session_state["sc_facility_caps"])

    # ── Overall summary counts ────────────────────────────────────────────────
    n_selected = len(selected_fids)
    n_total = len(all_selectable_fids)
    total_cap = sum(facility_caps.get(f, 0) for f in selected_fids)
    if n_selected > 0:
        st.caption(
            f"{n_selected} of {n_total} facilities selected · "
            f"Total scenario capacity: {total_cap:,}"
        )
    else:
        st.caption(f"0 of {n_total} facilities selected")

    # ── Search box ────────────────────────────────────────────────────────────
    search_term = st.text_input(
        "Search facilities",
        value="",
        key="fac_search",
        label_visibility="collapsed",
        placeholder="Search facilities…",
    )
    search_lower = search_term.strip().lower()

    # ── Build grouped view ────────────────────────────────────────────────────
    by_type: dict[str, list[CatalogEntry]] = {}
    for entry in all_entries:
        by_type.setdefault(entry.facility_type, []).append(entry)

    def _matches(e: CatalogEntry) -> bool:
        if not search_lower:
            return True
        return (
            search_lower in e.name.lower()
            or search_lower in e.barangay_name.lower()
            or search_lower in e.facility_type.lower()
        )

    # ── Per-type counts ───────────────────────────────────────────────────────
    type_sel_count: dict[str, int] = {}
    type_total_count: dict[str, int] = {}
    for ftype in _FACILITY_TYPE_ORDER:
        entries_t = [e for e in by_type.get(ftype, []) if e.can_be_selected]
        type_total_count[ftype] = len(entries_t)
        type_sel_count[ftype] = sum(
            1 for e in entries_t if e.facility_id in selected_fids
        )

    # ── Facility groups ───────────────────────────────────────────────────────
    ordered_types = _FACILITY_TYPE_ORDER + [
        t for t in by_type if t not in _FACILITY_TYPE_ORDER
    ]

    missing_cap_fids: list[str] = []

    for ftype in ordered_types:
        entries_in_type = by_type.get(ftype, [])
        if not entries_in_type:
            continue

        # Hide groups with no selectable entries (e.g. covered courts 0/0)
        if type_total_count.get(ftype, 0) == 0:
            continue

        # Filter to visible entries for display
        visible_in_type = [e for e in entries_in_type if _matches(e)]
        if not visible_in_type:
            continue

        plain = _FACILITY_TYPE_PLAIN.get(ftype, ftype)
        n_sel_t = type_sel_count.get(ftype, 0)
        n_tot_t = type_total_count.get(ftype, 0)
        header = f"{plain} ({n_sel_t}/{n_tot_t})"

        auto_exp = ftype == "evacuation_center"

        with st.expander(header, expanded=auto_exp):
            for entry in visible_in_type:
                if not entry.can_be_selected:
                    st.caption(f"{entry.name} — no road node available")
                    continue

                fid = entry.facility_id

                # Individual checkbox — on_change callback updates sc_selected_fids
                # and fac_all_toggle atomically before the next render.
                st.checkbox(
                    entry.name,
                    key=f"fsel_{fid}",
                    on_change=_on_fac_toggle,
                    kwargs={"fid": fid},
                )

                # Derive selected state from canonical key (synced by pre-render sync
                # or callbacks).
                is_selected: bool = bool(st.session_state.get(f"fsel_{fid}"))

                # Designation label
                st.caption(_designation_label(entry))

                if is_selected:
                    existing_cap = facility_caps.get(fid, 200)
                    cap_val = st.number_input(
                        _capacity_label(entry),
                        min_value=1,
                        max_value=50_000,
                        value=existing_cap,
                        step=50,
                        key=f"fcap_{fid}",
                    )
                    facility_caps[fid] = int(cap_val)
                    st.session_state["sc_facility_caps"] = facility_caps
                    if int(cap_val) != existing_cap:
                        st.session_state["sc_dirty"] = True

                    # Reachability / origin-collision notice
                    if (
                        entry.snapped_node is not None
                        and origin_nodes is not None
                        and entry.snapped_node in origin_nodes
                    ):
                        st.caption(
                            "Excluded — nearest road node is also a barangay origin "
                            "(cannot be destination)"
                        )
                    elif last_reachable_nodes is not None:
                        if entry.snapped_node in last_reachable_nodes:
                            st.caption("Reachable")
                        else:
                            st.caption(
                                "Not reachable — access road excluded "
                                "under the current routing model"
                            )

                    if facility_caps.get(fid, 0) <= 0:
                        missing_cap_fids.append(entry.name)

    # ── Consolidated missing-capacity warning ─────────────────────────────────
    if missing_cap_fids:
        names_str = ", ".join(f"**{n}**" for n in missing_cap_fids[:5])
        if len(missing_cap_fids) > 5:
            names_str += f" + {len(missing_cap_fids) - 5} more"
        st.warning(
            f"The following selected facilit"
            f"{'y has' if len(missing_cap_fids) == 1 else 'ies have'} "
            f"no scenario capacity set: {names_str}. "
            "They will be excluded from routing.",
            icon="⚠️",
        )

    # ── Derive updated ScenarioConfig from current state ──────────────────────
    # Re-read canonical selection one final time (callbacks may have run).
    final_selected = list(st.session_state.get("sc_selected_fids") or [])
    final_caps = dict(st.session_state.get("sc_facility_caps") or {})

    updated_scenario = ScenarioConfig(
        municipality=scenario.municipality,
        return_period=scenario.return_period,
        demand_mode=scenario.demand_mode,
        demand_fraction=scenario.demand_fraction,
        demand_exact=scenario.demand_exact,
        selected_facility_ids=final_selected,
        facility_capacities=final_caps,
        algorithms=scenario.algorithms,
        capacity_multipliers=scenario.capacity_multipliers,
        flood_penalties=scenario.flood_penalties,
        catalog_fingerprint=scenario.catalog_fingerprint,
        road_conditions=list(getattr(scenario, "road_conditions", [])),
    )

    # ── Blocked state ─────────────────────────────────────────────────────────
    if not final_selected:
        return updated_scenario, True, "Select at least one facility."

    effective = catalog.resolve_node_shelters(updated_scenario, exclude_nodes=origin_nodes)
    if not effective:
        return updated_scenario, True, "Enter available spaces for at least one facility."

    if last_reachable_nodes is not None:
        reachable_selected = {
            e.snapped_node
            for e in catalog.all()
            if e.facility_id in final_selected
            and e.snapped_node is not None
            and e.snapped_node in last_reachable_nodes
        }
        if not reachable_selected:
            return (
                updated_scenario,
                True,
                "No selected facility is reachable under the current routing model.",
            )

    return updated_scenario, False, ""
