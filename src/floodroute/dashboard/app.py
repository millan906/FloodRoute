"""FloodRoute Dashboard — Evacuation Planner.

Three tabs:
  Evacuation Planner      — five-step end-user workflow
  Technical Details       — research and algorithm inspection
  Experimental Evaluation — Stage 11 configurable batch experiments

Launch with:
    streamlit run src/floodroute/dashboard/app.py
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import streamlit as st
from streamlit_folium import st_folium

from floodroute.dashboard.experiment_page import render_experiment_page
from floodroute.dashboard.facilities import (
    DESIGNATION_TYPE_LABELS,
    FACILITY_REGISTRY,
    FACILITY_TYPE_LABELS,
)
from floodroute.dashboard.map_builder import build_analytical_map
from floodroute.dashboard.operating_mode import OperatingMode
from floodroute.dashboard.osm_candidates import OSM_CANDIDATES
from floodroute.dashboard.past_experiments import discover_experiments, verify_checksums
from floodroute.dashboard.result_formatter import (
    format_origin_assignment_status,
)
from floodroute.dashboard.road_overrides import (
    RoadOverride,
    RoadOverrideStore,
    apply_overrides,
)
from floodroute.dashboard.scenario_ui import render_facility_selection
from floodroute.experiments.algorithms import (
    RETURN_PERIODS,
    run_flood_aware_nearest,
    run_floodroute_assignment,
    run_ordinary_nearest,
)
from floodroute.experiments.demand import (
    build_demands,
    build_demands_exact,
    load_psa_population,
    snap_barangay_origins,
)
from floodroute.experiments.metrics import compute_metrics
from floodroute.experiments.runner import (
    _DEFAULT_BARANGAY_GPKG,
    _DEFAULT_GRAPHML,
    _DEFAULT_NODES_GPKG,
    _DEFAULT_POP_CSV,
    MUNICIPALITY_PSGC,
    SCENARIO_SHELTER_CAPACITIES,
)
from floodroute.optimization.cost import make_weight_fn
from floodroute.scenario.catalog import FacilityCatalog, build_catalog
from floodroute.scenario.config import ScenarioConfig
from floodroute.scenario.persistence import (
    delete_scenario,
    list_scenarios,
    load_scenario,
    new_scenario_id,
    save_scenario,
    scenario_from_dict,
    scenario_to_dict,
)

# ───────────────────────────────────────────────────────────────────────────
# Page configuration
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodRoute — Evacuation Planner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────────────────────────────────────────────────────
# Plain-language constants
# ───────────────────────────────────────────────────────────────────────────
_RP_LABELS: dict[str, str] = {
    "RP10":  "Moderate flood scenario (RP10 — ~10% annual chance)",
    "RP20":  "Severe flood scenario (RP20 — ~5% annual chance)",
    "RP100": "Extreme flood scenario (RP100 — ~1% annual chance)",
}
_RP_SHORT: dict[str, str] = {
    "RP10": "Moderate (RP10)", "RP20": "Severe (RP20)", "RP100": "Extreme (RP100)",
}

# Planner-facing condition labels (no internal status values exposed)
_CONDITION_LABELS: dict[str, str] = {
    "use_model":              "Use baseline flood information",
    "dry_confirmed_passable": "Dry and passable",
    "flooded_passable":       "Flooded but passable",
    "flooded_impassable":     "Flooded and cannot be used",
    "road_closed":            "Closed",
    "unknown":                "Condition unknown",
}
_CONDITION_ROUTING_EFFECT: dict[str, str] = {
    "use_model":              "Baseline flood model applies",
    "dry_confirmed_passable": "Available to all routes at ordinary cost",
    "flooded_passable":       "Available with flood-adjusted cost",
    "flooded_impassable":     "Excluded from all routes",
    "road_closed":            "Excluded from all routes",
    "unknown":                "Excluded from routes (conservative policy)",
}
_CONDITION_CONFIRMATION: dict[str, str] = {
    "use_model":              "{name} uses baseline flood information.",
    "dry_confirmed_passable": "{name} is dry and passable in this scenario.",
    "flooded_passable":       "{name} is flooded but passable in this scenario.",
    "flooded_impassable":     "{name} is flooded and cannot be used in this scenario.",
    "road_closed":            "{name} is closed in this scenario.",
    "unknown":                "{name} has an unknown condition and is excluded from routing.",
}

_FACILITY_TYPE_PLAIN: dict[str, str] = {
    "evacuation_center": "Evacuation-labelled facilities",
    "covered_court":     "Covered courts",
    "multi_purpose_hall": "Multi-purpose halls",
    "barangay_hall":     "Barangay halls",
    "school":            "Schools",
}
_FACILITY_TYPE_ORDER = list(_FACILITY_TYPE_PLAIN)

# ───────────────────────────────────────────────────────────────────────────
# Routing helpers
# ───────────────────────────────────────────────────────────────────────────

def _ordinary_weight(u: object, v: object, d: dict) -> float:
    best: float | None = None
    for attrs in d.values():
        length = float(attrs.get("length_m") or 0.0)
        if best is None or length < best:
            best = length
    return best if best is not None else 0.0


def _route_detail(
    G: nx.MultiDiGraph,
    path: list[int],
    store: RoadOverrideStore | None,
    return_period: str,
) -> tuple[float, list[str], list[str]]:
    """Return (distance_m, ordered_road_names, conditions_encountered).

    Road names are taken from the OSM ``name`` attribute of the cheapest
    parallel edge at each hop. Unnamed hops appear as "Unnamed road segment".
    Each name is listed once in the order first encountered.

    Conditions lists any edge that carries a planner override (excluding
    ``use_model``) or a flooded JRC status under the active return period.
    """
    rp_lower = return_period.lower()
    status_col = f"jrc_{rp_lower}_status"
    total_m = 0.0
    road_names: list[str] = []
    conditions: list[str] = []
    seen_names: set[str] = set()
    seen_conds: set[str] = set()
    overrides_map: dict = store.overrides if store else {}

    for u, v in zip(path[:-1], path[1:], strict=False):
        if not G.has_edge(u, v):
            continue
        best = min(G[u][v].values(), key=lambda a: float(a.get("length_m") or 0.0))
        total_m += float(best.get("length_m") or 0.0)

        raw = best.get("name", "")
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        road_name = (raw or "").strip() or "Unnamed road segment"
        if road_name not in seen_names:
            seen_names.add(road_name)
            road_names.append(road_name)

        ov = overrides_map.get((int(u), int(v)))
        if ov and ov.status != "use_model":
            label = _CONDITION_LABELS.get(ov.status, ov.status)
            cond = f"{road_name} — {label}"
            if cond not in seen_conds:
                seen_conds.add(cond)
                conditions.append(cond)
        elif best.get(status_col) == "flooded":
            cond = f"{road_name} — flooded (JRC model)"
            if cond not in seen_conds:
                seen_conds.add(cond)
                conditions.append(cond)

    return total_m, road_names, conditions


def _path_length_m(
    G: nx.MultiDiGraph,
    path: list[int],
    return_period: str | None = None,
) -> tuple[float, float]:
    rp_lower = return_period.lower() if return_period else None
    status_col = f"jrc_{rp_lower}_status" if rp_lower else None
    total_m = 0.0
    flood_m = 0.0
    for u, v in zip(path[:-1], path[1:], strict=False):
        if not G.has_edge(u, v):
            continue
        best = min(G[u][v].values(), key=lambda a: float(a.get("length_m") or 0.0))
        length = float(best.get("length_m") or 0.0)
        total_m += length
        if status_col and best.get(status_col) == "flooded":
            flood_m += length
    return total_m, flood_m


def _shortest_path_to_shelters(
    G: nx.MultiDiGraph,
    origin: int,
    shelters: dict[int, int],
    weight_fn,
) -> list[int] | None:
    if origin not in G:
        return None
    try:
        lengths, paths = nx.single_source_dijkstra(G, origin, weight=weight_fn)
    except nx.NetworkXError:
        return None
    best_path: list[int] | None = None
    best_cost = float("inf")
    for s in shelters:
        if s in lengths and lengths[s] < best_cost:
            best_cost = lengths[s]
            best_path = paths[s]
    return best_path


# ───────────────────────────────────────────────────────────────────────────
# Cached loaders
# ───────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading road graph…")
def _load_graph() -> nx.MultiDiGraph:
    return nx.read_graphml(str(_DEFAULT_GRAPHML), node_type=int)


@st.cache_resource(show_spinner="Loading population data…")
def _load_origins():
    records = load_psa_population(_DEFAULT_POP_CSV, adm3_filter=MUNICIPALITY_PSGC)
    # No static exclude_nodes — origin/destination conflicts are resolved
    # dynamically at resolve_node_shelters() time using the active ScenarioConfig.
    origins_ = snap_barangay_origins(
        records,
        barangay_gpkg=_DEFAULT_BARANGAY_GPKG,
        nodes_gpkg=_DEFAULT_NODES_GPKG,
        municipality_psgc=MUNICIPALITY_PSGC,
    )
    total_pop_ = sum(o.population_2020 for o in origins_)
    return origins_, total_pop_


@st.cache_resource(show_spinner="Loading facility catalog…")
def _load_catalog() -> FacilityCatalog:
    return build_catalog()


origins, total_pop = _load_origins()

_BARANGAY_NAMES: list[str] = sorted(o.name for o in origins)
_NAME_TO_ORIGIN: dict[str, object] = {o.name: o for o in origins}
_NODE_INFO: dict[int, dict] = {
    o.origin_node: {"name": o.name, "population_2020": o.population_2020}
    for o in origins
}
_OSM_BY_ID = {c.osm_id: c for c in OSM_CANDIDATES}


@st.cache_resource(show_spinner=False)
def _build_road_groups() -> dict[str, list[tuple[int, int]]]:
    """Road display-name → [(u, v), ...] for the condition editor.

    Named roads are grouped by their OSM ``name`` attribute.
    Unnamed roads are labeled "Unnamed road near <nearest barangay>".
    """
    G = _load_graph()
    # Spatial index: (barangay_name, x, y) for nearest-barangay labeling
    _orig_xy = [
        (o.name,
         float(G.nodes[o.origin_node].get("x", 0.0)) if o.origin_node in G else 0.0,
         float(G.nodes[o.origin_node].get("y", 0.0)) if o.origin_node in G else 0.0)
        for o in origins
    ]
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for u, v, attrs in G.edges(data=True):
        if (u, v) in seen:
            continue
        seen.add((u, v))
        raw = attrs.get("name", "")
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        name = (raw or "").strip()
        if name:
            groups[name].append((u, v))
        else:
            ux = float(G.nodes[u].get("x", 0.0)) if u in G.nodes else 0.0
            uy = float(G.nodes[u].get("y", 0.0)) if u in G.nodes else 0.0
            best_bgy = "unknown area"
            best_d = float("inf")
            for bgy_name, ox, oy in _orig_xy:
                d = (ux - ox) ** 2 + (uy - oy) ** 2
                if d < best_d:
                    best_d = d
                    best_bgy = bgy_name
            groups[f"Unnamed road near {best_bgy}"].append((u, v))
    return dict(groups)


# ───────────────────────────────────────────────────────────────────────────
# Override helpers
# ───────────────────────────────────────────────────────────────────────────

def _get_override_store() -> RoadOverrideStore:
    d = st.session_state.get("road_override_store_dict") or {}
    return RoadOverrideStore.from_dict(d) if d else RoadOverrideStore()


def _apply_name_condition(road_name: str, condition_key: str) -> int:
    """Apply *condition_key* to all segments named *road_name*. Returns count."""
    groups = _build_road_groups()
    segments = groups.get(road_name, [])
    store = _get_override_store()
    if condition_key == "use_model":
        for u, v in segments:
            store.remove(u, v)
    else:
        for u, v in segments:
            store.add(
                RoadOverride(
                    u=u, v=v, status=condition_key,
                    evidence_type="controlled_assumption",
                )
            )
    st.session_state["road_override_store_dict"] = store.to_dict()
    # Track name-level condition for the "Active road conditions" list
    nc: dict = dict(st.session_state.get("road_name_conditions") or {})
    if condition_key == "use_model":
        nc.pop(road_name, None)
    else:
        nc[road_name] = {"condition": condition_key, "count": len(segments)}
    st.session_state["road_name_conditions"] = nc
    return len(segments)


def _make_override_wfn(store: RoadOverrideStore, base_flood_wfn):
    """Full override weight function for Algorithm C (flood-aware + conditions)."""
    return apply_overrides(base_flood_wfn, store, flood_penalty_fn=base_flood_wfn)


# ───────────────────────────────────────────────────────────────────────────
# Scenario fingerprint for stale detection
# ───────────────────────────────────────────────────────────────────────────

def _scenario_key(
    return_period: str,
    selected_bgy_names: list[str],
    demand_type: str,
    demand_value: object,
    effective_shelters: dict,
    override_dict: dict,
) -> str:
    return "|".join([
        return_period,
        str(sorted(selected_bgy_names)),
        demand_type,
        str(demand_value),
        str(tuple(sorted(effective_shelters.items()))),
        str(sorted(override_dict.items())),
    ])


# ───────────────────────────────────────────────────────────────────────────
# Session state defaults
# ───────────────────────────────────────────────────────────────────────────
_STATE_DEFAULTS: list[tuple] = [
    ("result", None),
    ("metrics", None),
    ("run_params", None),
    ("run_error", None),
    ("run_scenario_key", None),
    ("ordinary_path", None),
    ("alg_c_path", None),
    ("alg_c_assigned_shelter", None),
    ("alg_c_origin_status", None),
    ("road_override_store_dict", {}),
    ("road_name_conditions", {}),
    ("step4_last_applied", None),
    ("activated_osm_ids", set()),
    ("scenario_capacities", {}),
    ("_confirm_scenario_reset", False),
    ("selected_origin_node", None),
    ("road_editor_road", None),
    ("road_editor_cond", None),
    ("highlighted_override_road", None),
    ("_pending_clear_road_search", False),
    ("_fit_bounds", False),
    ("sc_selected_fids", []),
    ("sc_facility_caps", {}),
    ("sc_name", "Untitled scenario"),
    ("sc_id", None),
    ("sc_dirty", False),
    ("sc_confirm_delete", None),
    ("sc_unresolved_fids", []),
]
for _k, _v in _STATE_DEFAULTS:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ───────────────────────────────────────────────────────────────────────────
# Responsive CSS
# ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@media (min-width: 768px) {
    section[data-testid="stHorizontalBlock"] > div:nth-child(2) iframe {
        min-height: 700px !important;
    }
}
@media (max-width: 767px) {
    section[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    section[data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important;
    }
    section[data-testid="stHorizontalBlock"] > div:nth-child(2) iframe {
        min-height: 500px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────
# Header and tabs
# ───────────────────────────────────────────────────────────────────────────
st.title("FloodRoute — Evacuation Planner")

planner_tab, technical_tab, experiment_tab = st.tabs(
    ["Evacuation Planner", "Technical Details", "Experimental Evaluation"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: EVACUATION PLANNER
# ═══════════════════════════════════════════════════════════════════════════
with planner_tab:

    st.caption(
        "Interactive planning scenario based on the project's documented population, "
        "road, flood-hazard and facility datasets. "
        "Not live conditions or an official evacuation directive."
    )

    col_left, col_map, col_right = st.columns([1, 3.2, 1])

    # ─────────────────────────────────────────────────────────────────────
    # LEFT COLUMN — five-step inputs
    # ─────────────────────────────────────────────────────────────────────
    with col_left:

        # ── Reset entire scenario ─────────────────────────────────────────
        if st.session_state.get("_confirm_scenario_reset"):
            st.warning("This will clear all inputs and results.", icon="⚠️")
            _ry, _rn = st.columns(2)
            if _ry.button("Yes, reset", type="primary", use_container_width=True):
                for _ck, _cd in _STATE_DEFAULTS:
                    st.session_state[_ck] = _cd
                st.session_state["all_barangays_check"] = True
                st.session_state["barangay_multiselect"] = []
                st.session_state["demand_type_radio"] = "25%"
                st.session_state["demand_exact_value"] = 0
                st.session_state["_confirm_scenario_reset"] = False
                st.session_state["selected_origin_node"] = None
                st.session_state["road_editor_road"] = None
                st.session_state["road_editor_cond"] = None
                st.session_state["highlighted_override_road"] = None
                st.session_state["sc_selected_fids"] = []
                st.session_state["sc_facility_caps"] = {}
                st.session_state["sc_dirty"] = False
                st.rerun()
            if _rn.button("Cancel", use_container_width=True):
                st.session_state["_confirm_scenario_reset"] = False
                st.rerun()
        else:
            if st.button("Reset entire scenario", use_container_width=True):
                st.session_state["_confirm_scenario_reset"] = True
                st.rerun()

        # ── Scenario management ───────────────────────────────────────────
        with st.expander("Scenario", expanded=False):
            _sc_name_val = st.session_state.get("sc_name", "Untitled scenario")
            _new_name = st.text_input("Scenario name", value=_sc_name_val, key="sc_name_input")
            if _new_name != _sc_name_val:
                st.session_state["sc_name"] = _new_name
                st.session_state["sc_dirty"] = True

            if st.session_state.get("sc_dirty"):
                st.caption("Unsaved changes")

            _c1, _c2 = st.columns(2)
            if _c1.button("Save", use_container_width=True, key="sc_save_btn"):
                _sc_obj = ScenarioConfig(
                    selected_facility_ids=st.session_state.get("sc_selected_fids", []),
                    facility_capacities=st.session_state.get("sc_facility_caps", {}),
                )
                _cat_fp = _load_catalog().fingerprint()
                _d = scenario_to_dict(
                    _sc_obj,
                    _cat_fp,
                    st.session_state.get("sc_name", "Untitled"),
                    scenario_id=st.session_state.get("sc_id") or new_scenario_id(),
                )
                save_scenario(_d)
                st.session_state["sc_id"] = _d["scenario_id"]
                st.session_state["sc_dirty"] = False
                st.success("Saved.")

            if _c2.button("Save as new", use_container_width=True, key="sc_save_new_btn"):
                _sc_obj = ScenarioConfig(
                    selected_facility_ids=st.session_state.get("sc_selected_fids", []),
                    facility_capacities=st.session_state.get("sc_facility_caps", {}),
                )
                _cat_fp = _load_catalog().fingerprint()
                _d = scenario_to_dict(
                    _sc_obj, _cat_fp, st.session_state.get("sc_name", "Untitled")
                )
                save_scenario(_d)
                st.session_state["sc_id"] = _d["scenario_id"]
                st.session_state["sc_dirty"] = False
                st.success("Saved as new.")

            # Load scenario
            _saved = list_scenarios()
            if _saved:
                _names = {
                    s["scenario_id"]: s.get("scenario_name", s["scenario_id"])
                    for s in _saved
                }
                _sel_id = st.selectbox(
                    "Load saved scenario",
                    options=[""] + list(_names.keys()),
                    format_func=lambda x: _names.get(x, x) if x else "— select —",
                    key="sc_load_select",
                )
                if _sel_id and st.button("Load", use_container_width=True, key="sc_load_btn"):
                    if st.session_state.get("sc_dirty"):
                        st.warning("Unsaved changes will be lost.")
                    try:
                        _loaded = load_scenario(_sel_id)
                        _load_cat = _load_catalog()
                        _sc_loaded, _unresolved = scenario_from_dict(
                            _loaded, catalog=_load_cat
                        )
                        # Catalog fingerprint mismatch warning
                        _saved_fp = _loaded.get("catalog_fingerprint", "")
                        _current_fp = _load_cat.fingerprint()
                        if _saved_fp and _saved_fp != _current_fp:
                            st.warning(
                                "Catalog fingerprint mismatch — the facility catalog "
                                f"has changed since this scenario was saved "
                                f"(saved: `{_saved_fp}`, current: `{_current_fp}`). "
                                "Verify all selected facilities still exist.",
                                icon="⚠️",
                            )
                        # Unresolved facility warning — preserve and block
                        if _unresolved:
                            st.error(
                                "The following selected facilit"
                                + ("y is" if len(_unresolved) == 1 else "ies are")
                                + " no longer in the current catalog: "
                                + ", ".join(f"`{f}`" for f in _unresolved)
                                + ". The scenario was loaded but execution is blocked "
                                "until unresolved facilities are removed or the catalog "
                                "is updated.",
                                icon="🚫",
                            )
                            st.session_state["sc_unresolved_fids"] = _unresolved
                        else:
                            st.session_state["sc_unresolved_fids"] = []
                        st.session_state["sc_selected_fids"] = list(
                            _sc_loaded.selected_facility_ids
                        )
                        st.session_state["sc_facility_caps"] = dict(
                            _sc_loaded.facility_capacities
                        )
                        st.session_state["sc_name"] = _loaded.get("scenario_name", "Loaded")
                        st.session_state["sc_id"] = _sel_id
                        st.session_state["sc_dirty"] = False
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Failed to load: {_e}")

            # Delete with confirmation
            _del_id = st.session_state.get("sc_confirm_delete")
            if _del_id and _saved:
                _names_del = {
                    s["scenario_id"]: s.get("scenario_name", s["scenario_id"])
                    for s in _saved
                }
                st.warning(f"Delete '{_names_del.get(_del_id, _del_id)}'?")
                _dy, _dn = st.columns(2)
                if _dy.button(  # noqa: E501
                    "Delete", type="primary", use_container_width=True, key="sc_del_confirm"
                ):
                    delete_scenario(_del_id)
                    if st.session_state.get("sc_id") == _del_id:
                        st.session_state["sc_id"] = None
                        st.session_state["sc_dirty"] = False
                    st.session_state["sc_confirm_delete"] = None
                    st.rerun()
                if _dn.button("Cancel", use_container_width=True, key="sc_del_cancel"):
                    st.session_state["sc_confirm_delete"] = None
                    st.rerun()
            elif _saved and st.button(
                "Delete current saved scenario",
                use_container_width=True,
                key="sc_del_btn",
            ):
                st.session_state["sc_confirm_delete"] = st.session_state.get("sc_id")

        st.divider()

        # ── Step 1: Flood scenario ────────────────────────────────────────
        st.markdown("### 1. Flood scenario")
        return_period: str = st.radio(
            "Flood scenario",
            options=list(RETURN_PERIODS),
            format_func=lambda rp: _RP_LABELS[rp],
            index=1,
            key="return_period_radio",
            label_visibility="collapsed",
        )

        st.divider()

        # ── Step 2: Affected barangays ────────────────────────────────────
        st.markdown("### 2. Affected barangays")
        _use_all = st.checkbox(
            "All barangays",
            value=st.session_state.get("all_barangays_check", True),
            key="all_barangays_check",
        )
        if _use_all:
            selected_bgy_names: list[str] = list(_BARANGAY_NAMES)
            st.caption(f"All {len(origins)} barangays selected.")
        else:
            selected_bgy_names = st.multiselect(
                "Select barangays",
                options=_BARANGAY_NAMES,
                default=st.session_state.get("barangay_multiselect") or [],
                key="barangay_multiselect",
                label_visibility="collapsed",
                placeholder="Choose one or more barangays…",
            )
            if not selected_bgy_names:
                st.caption("No barangays selected.")

        selected_origins = [
            _NAME_TO_ORIGIN[n] for n in selected_bgy_names if n in _NAME_TO_ORIGIN
        ]
        _sel_pop = sum(o.population_2020 for o in selected_origins)

        # Highlight node: explicit result-panel selection takes priority;
        # fall back to the single-barangay default from Step 2.
        _explicit_sel = st.session_state.get("selected_origin_node")
        highlight_node: int | None = (
            _explicit_sel
            if _explicit_sel is not None
            else (selected_origins[0].origin_node if len(selected_origins) == 1 else None)
        )

        st.divider()

        # ── Step 3: People to evacuate ────────────────────────────────────
        st.markdown("### 3. People to evacuate")
        _demand_opts = ["10%", "25%", "50%", "Custom %", "Exact number"]
        demand_type: str = st.radio(
            "Demand",
            options=_demand_opts,
            index=1,
            key="demand_type_radio",
            label_visibility="collapsed",
        )

        demand_fraction: float | None = None
        exact_demand_value: int | None = None
        _demand_valid = bool(selected_bgy_names)
        _demand_block_reason = "" if selected_bgy_names else "Select at least one barangay."

        if demand_type in ("10%", "25%", "50%"):
            demand_fraction = float(demand_type.rstrip("%")) / 100.0
            _calc = round(_sel_pop * demand_fraction) if _sel_pop > 0 else 0
            st.caption(f"People to evacuate: **{_calc:,}**")
        elif demand_type == "Custom %":
            _pct = st.number_input(
                "Percentage", min_value=1, max_value=100, value=25,
                key="custom_pct", label_visibility="collapsed",
            )
            demand_fraction = _pct / 100.0
            _calc = round(_sel_pop * demand_fraction) if _sel_pop > 0 else 0
            st.caption(f"People to evacuate: **{_calc:,}**")
        else:  # Exact number
            _max_ex = max(_sel_pop, 1)
            _raw_ex = st.number_input(
                "Number of people",
                min_value=0, max_value=_max_ex,
                value=min(500, _max_ex),
                step=50,
                key="demand_exact_value",
                label_visibility="collapsed",
            )
            exact_demand_value = int(_raw_ex)
            if exact_demand_value == 0:
                st.caption("Zero people — plan generation disabled.")
                if _demand_valid:
                    _demand_valid = False
                    _demand_block_reason = "Enter at least 1 person."
            elif exact_demand_value > _sel_pop and _sel_pop > 0:
                st.error(f"Exceeds selected population ({_sel_pop:,}).")
                _demand_valid = False
                _demand_block_reason = "Exact count exceeds population."
            else:
                st.caption(f"People to evacuate: **{exact_demand_value:,}**")

        st.divider()

        # ── Step 4: Evacuation facilities ─────────────────────────────────
        st.markdown("### 4. Evacuation facilities")

        # Reachable shelter nodes from the last completed run (empty when no run yet).
        _last_result = st.session_state.get("result")
        _reachable_nodes_from_run: set[int] | None = (
            {s for (_, s) in _last_result.od_costs_scenario}
            if _last_result is not None else None
        )

        # Origin nodes for the current barangay selection.
        _origin_nodes: set[int] = {o.origin_node for o in selected_origins}

        # Build current ScenarioConfig from session state for passing into UI
        _catalog = _load_catalog()
        _sc_current = ScenarioConfig(
            selected_facility_ids=st.session_state.get("sc_selected_fids", []),
            facility_capacities=st.session_state.get("sc_facility_caps", {}),
            municipality=MUNICIPALITY_PSGC,
            return_period=return_period,
        )

        _sc_updated, _fac_blocked, _fac_block_reason = render_facility_selection(
            catalog=_catalog,
            scenario=_sc_current,
            origin_nodes=_origin_nodes,
            last_reachable_nodes=_reachable_nodes_from_run,
        )

        # Build effective shelters from unified ScenarioConfig
        effective_shelters: dict[int, int] = _catalog.resolve_node_shelters(
            _sc_updated, exclude_nodes=_origin_nodes
        )

        # Keep activated_osm_ids in sync for backward-compat map call
        activated_ids: set[str] = set(st.session_state.get("sc_selected_fids", []))
        st.session_state["activated_osm_ids"] = activated_ids

        # Build shelter display labels using catalog entries
        _node_to_names: dict[int, list[str]] = {}
        _node_to_types: dict[int, list[str]] = {}
        for _fid in st.session_state.get("sc_selected_fids", []):
            _entry = _catalog.get(_fid)
            if _entry and _entry.snapped_node is not None:
                _node_to_names.setdefault(_entry.snapped_node, []).append(_entry.name)
                _node_to_types.setdefault(_entry.snapped_node, []).append(_entry.facility_type)
        shelter_labels: dict[int, str] = {
            n: " / ".join(names) for n, names in _node_to_names.items()
        }
        # facility_type for the compact shelter popup (first type when co-located)
        shelter_types: dict[int, str] = {
            n: types[0] for n, types in _node_to_types.items() if types
        }

        st.divider()

        # ── Step 5: Road conditions ───────────────────────────────────────
        _s5h, _s5r = st.columns([3, 1])
        _s5h.markdown("### 5. Road conditions")

        _road_name_conds: dict = dict(st.session_state.get("road_name_conditions") or {})
        _n_road_conds = len(_road_name_conds)

        if _n_road_conds > 0 and _s5r.button(
            "Reset all road conditions", key="s5_reset_all", use_container_width=True
        ):
            st.session_state["road_override_store_dict"] = {}
            st.session_state["road_name_conditions"] = {}
            st.session_state["step4_last_applied"] = None
            st.session_state["run_scenario_key"] = None
            st.session_state["road_editor_road"] = None
            st.session_state["road_editor_cond"] = None
            st.session_state["highlighted_override_road"] = None
            st.rerun()

        _road_groups = _build_road_groups()
        _road_names_sorted = sorted(_road_groups.keys())

        # ── Condition editor: two modes ───────────────────────────────────
        # EDIT mode — triggered by "Change" button in the active list
        _edit_road = st.session_state.get("road_editor_road")
        _edit_cond = st.session_state.get("road_editor_cond")
        _cond_options = [k for k in _CONDITION_LABELS if k != "use_model"]

        if _edit_road:
            st.caption(f"Editing: **{_edit_road}**")
            _edit_cond_idx = (
                _cond_options.index(_edit_cond)
                if _edit_cond and _edit_cond in _cond_options else 0
            )
            _new_cond = st.radio(
                "Updated condition",
                options=_cond_options,
                format_func=lambda k: _CONDITION_LABELS[k],
                index=_edit_cond_idx,
                key="road_cond_radio",
                label_visibility="collapsed",
            )
            _eu, _ec = st.columns(2)
            if _eu.button(
                "Update road condition", type="primary",
                key="apply_road_cond", use_container_width=True,
            ):
                _cnt = _apply_name_condition(_edit_road, _new_cond)
                st.session_state["step4_last_applied"] = (_edit_road, _new_cond, _cnt)
                st.session_state["run_scenario_key"] = None
                st.session_state["road_editor_road"] = None
                st.session_state["road_editor_cond"] = None
                st.rerun()
            if _ec.button("Cancel", key="cancel_edit_road", use_container_width=True):
                st.session_state["road_editor_road"] = None
                st.session_state["road_editor_cond"] = None
                st.rerun()

        else:
            # ADD mode — search, select road, pick condition, click Add
            # Clear search from the previous "Add road condition" click.
            # Must happen before the widget is instantiated.
            if st.session_state.pop("_pending_clear_road_search", False):
                st.session_state["road_search"] = ""

            _road_search = st.text_input(
                "Search road name",
                value="",
                placeholder="e.g. Rizal, National, Airport",
                key="road_search",
                label_visibility="collapsed",
            )
            _road_filtered = (
                [n for n in _road_names_sorted if _road_search.lower() in n.lower()]
                if _road_search.strip()
                else _road_names_sorted[:60]
            )

            if _road_filtered:
                _sel_road = st.selectbox(
                    "Road",
                    options=_road_filtered,
                    key="road_select",
                    label_visibility="collapsed",
                )
                _seg_count = len(_road_groups.get(_sel_road, []))
                _exist_cond = _road_name_conds.get(_sel_road, {}).get(
                    "condition", "use_model"
                )
                _exist_label = _CONDITION_LABELS.get(_exist_cond, "")
                if _exist_cond != "use_model":
                    st.caption(
                        f"{_seg_count} segment{'s' if _seg_count != 1 else ''} — "
                        f"currently **{_exist_label}**"
                    )
                else:
                    st.caption(
                        f"{_seg_count} segment{'s' if _seg_count != 1 else ''} — "
                        f"no condition set"
                    )

                _default_idx = (
                    _cond_options.index(_exist_cond)
                    if _exist_cond in _cond_options else 0
                )
                _new_cond = st.radio(
                    "Condition",
                    options=_cond_options,
                    format_func=lambda k: _CONDITION_LABELS[k],
                    index=_default_idx,
                    key="road_cond_radio",
                    label_visibility="collapsed",
                )

                _btn_label = (
                    "Add road condition"
                    if _exist_cond == "use_model"
                    else "Update road condition"
                )
                if st.button(
                    _btn_label, type="primary",
                    key="apply_road_cond", use_container_width=True,
                ):
                    _cnt = _apply_name_condition(_sel_road, _new_cond)
                    st.session_state["step4_last_applied"] = (_sel_road, _new_cond, _cnt)
                    st.session_state["run_scenario_key"] = None
                    # Flag to clear search on the next rerun (before the widget
                    # is instantiated — Streamlit forbids writing a widget key
                    # after it has already been rendered this run).
                    st.session_state["_pending_clear_road_search"] = True
                    st.rerun()

                # Confirmation
                _last = st.session_state.get("step4_last_applied")
                if _last and _last[0] == _sel_road:
                    _tmpl = _CONDITION_CONFIRMATION.get(_last[1])
                    if _tmpl:
                        _cnt_conf = _last[2] if len(_last) > 2 else 1
                        st.success(
                            _tmpl.format(name=_sel_road)
                            + f" ({_cnt_conf} segment{'s' if _cnt_conf != 1 else ''})"
                        )
            else:
                st.caption("No matching roads.")

        # ── Active road conditions list ───────────────────────────────────
        _road_name_conds = dict(st.session_state.get("road_name_conditions") or {})
        _n_road_conds = len(_road_name_conds)
        _hl_road = st.session_state.get("highlighted_override_road")
        if _n_road_conds > 0:
            st.markdown("**Active road conditions**")
            for _rn, _rc in list(_road_name_conds.items()):
                _rn_cond = _rc.get("condition", "use_model")
                _is_hl_road = _rn == _hl_road
                _cond_label = _CONDITION_LABELS.get(_rn_cond, _rn_cond)
                st.caption(
                    ("★ " if _is_hl_road else "")
                    + f"**{_rn}** — {_cond_label}"
                )
                _rb1, _rb2, _rb3 = st.columns(3)
                if _rb1.button(
                    "Hide" if _is_hl_road else "Show on map",
                    key=f"show_{_rn}", use_container_width=True,
                ):
                    st.session_state["highlighted_override_road"] = (
                        None if _is_hl_road else _rn
                    )
                    st.rerun()
                if _rb2.button(
                    "Change", key=f"chg_{_rn}", use_container_width=True
                ):
                    st.session_state["road_editor_road"] = _rn
                    st.session_state["road_editor_cond"] = _rn_cond
                    st.rerun()
                if _rb3.button(
                    "Remove", key=f"rst_{_rn}", use_container_width=True
                ):
                    _apply_name_condition(_rn, "use_model")
                    if _hl_road == _rn:
                        st.session_state["highlighted_override_road"] = None
                    if st.session_state.get("road_editor_road") == _rn:
                        st.session_state["road_editor_road"] = None
                        st.session_state["road_editor_cond"] = None
                    st.session_state["run_scenario_key"] = None
                    st.rerun()

        # ── Pre-run condition summary ─────────────────────────────────────
        _road_name_conds_pre = dict(st.session_state.get("road_name_conditions") or {})
        _n_pre = len(_road_name_conds_pre)
        if _n_pre > 0:
            st.info(
                f"{_n_pre} road condition{'s' if _n_pre != 1 else ''} will be applied "
                f"to this evacuation scenario."
            )

        st.divider()

        # ── Generate plan button ──────────────────────────────────────────
        st.markdown("### Generate evacuation plan")

        _unresolved_fids = st.session_state.get("sc_unresolved_fids", [])
        _unresolved_blocked = bool(_unresolved_fids)
        _run_blocked = not _demand_valid or _fac_blocked or _unresolved_blocked
        _block_reason = _demand_block_reason or _fac_block_reason
        if _unresolved_blocked and not _block_reason:
            _block_reason = (
                "Loaded scenario contains facilities no longer in the catalog: "
                + ", ".join(_unresolved_fids[:3])
                + (f" + {len(_unresolved_fids)-3} more" if len(_unresolved_fids) > 3 else "")
            )

        # Scenario to calculate (pre-run summary)
        if not _run_blocked:
            pass
















































        st.divider()

        # ── Generate plan button ──────────────────────────────────────────
        st.markdown("### Generate evacuation plan")

        _unresolved_fids = st.session_state.get("sc_unresolved_fids", [])
        _unresolved_blocked = bool(_unresolved_fids)
        _run_blocked = not _demand_valid or _fac_blocked or _unresolved_blocked
        _block_reason = _demand_block_reason or _fac_block_reason
        if _unresolved_blocked and not _block_reason:
            _block_reason = (
                "Loaded scenario contains facilities no longer in the catalog: "
                + ", ".join(_unresolved_fids[:3])
                + (f" + {len(_unresolved_fids)-3} more" if len(_unresolved_fids) > 3 else "")
            )

        # Scenario to calculate (pre-run summary)
        if not _run_blocked:
            with st.expander("Scenario to calculate", expanded=False):
                st.caption(f"**Flood scenario:** {_RP_SHORT.get(return_period, return_period)}")
                if len(selected_bgy_names) == len(origins):
                    st.caption(f"**Barangays:** All {len(origins)}")
                else:
                    _blist = ", ".join(selected_bgy_names[:3])
                    if len(selected_bgy_names) > 3:
                        _blist += f" … +{len(selected_bgy_names) - 3} more"
                    st.caption(f"**Barangays:** {_blist}")

            # Always use Algorithm C (FloodRoute MCF) in the Evacuation Planner
            result = run_floodroute_assignment(
                G, demands, effective_shelters, return_period,
                weight_fn=_override_wfn,
            )
            result.demand_fraction = demand_fraction or 0.0
            metrics = compute_metrics(result, G, total_population=total_pop)

            # Route overlays for single highlighted barangay
            ordinary_path: list[int] | None = None
            alg_c_path: list[int] | None = None
            alg_c_assigned_shelter: int | None = None
            alg_c_origin_status: dict | None = None

            if highlight_node is not None:
                ordinary_path = _shortest_path_to_shelters(
                    G, highlight_node, effective_shelters, _ordinary_weight
                )
                _ocs = format_origin_assignment_status(highlight_node, result)
                alg_c_origin_status = _ocs
                if _ocs["status"] == "assigned":
                    _best_pair = max(
                        ((o, s) for (o, s) in result.routes if o == highlight_node),
                        key=lambda k: result.assignments.get(k, 0),
                        default=None,
                    )
                    if _best_pair is not None:
                        alg_c_path = result.routes[_best_pair]
                        alg_c_assigned_shelter = _best_pair[1]

            st.session_state.update({
                "result": result,
                "metrics": metrics,
                "run_params": (
                    return_period,
                    sorted(selected_bgy_names),
                    demand_type,
                    _demand_val_for_key,
                    dict(effective_shelters),
                    dict(shelter_labels),
                ),
                "run_scenario_key": _current_key,
                "ordinary_path": ordinary_path,
                "alg_c_path": alg_c_path,
                "alg_c_assigned_shelter": alg_c_assigned_shelter,
                "alg_c_origin_status": alg_c_origin_status,
                "run_error": None,
            })
            st.rerun()

    # ── CENTER COLUMN: Map ────────────────────────────────────────────────
    with col_map:
        result = st.session_state["result"]
        metrics = st.session_state["metrics"]
        run_params = st.session_state["run_params"]

        if result is not None and run_params is not None:
            _rp_h = run_params[0]
            _dt_h = run_params[2]
            _dv_h = run_params[3]
            if _dt_h == "Exact number" and _dv_h is not None:
                _plan_label = f"{_dv_h:,} people (exact)"
            elif _dv_h is not None:
                _plan_label = f"{_dv_h:.0%} of population"
            else:
                _plan_label = ""
            st.subheader(
                f"Evacuation Plan — {_RP_SHORT.get(_rp_h, _rp_h)} · {_plan_label}"
            )
            if _result_is_stale:
                st.error(
                    "Results below are from a previous run. "
                    "Click **Generate evacuation plan** to apply the current scenario.",
                    icon="🔄",
                )
        else:
            st.subheader("Evacuation map")

        G = _load_graph()
        _disp_shelters = (
            run_params[4]
            if result is not None and run_params and len(run_params) > 4
            else effective_shelters
        )
        _disp_labels = (
            run_params[5]
            if result is not None and run_params and len(run_params) > 5
            else shelter_labels
        )

        # ── Compact scenario summary (shown after a plan has been generated) ─
        if result is not None and metrics is not None and run_params is not None:
            _sum_rp = _RP_SHORT.get(run_params[0], run_params[0])
            _sum_bgys = run_params[1]
            _sum_td = metrics.get("total_demand", 0)
            _sum_ta = metrics.get("total_assigned", 0)
            _sum_tu = _sum_td - _sum_ta
            _sum_facs = len(run_params[4]) if run_params and len(run_params) > 4 else 0
            _road_nc = dict(st.session_state.get("road_name_conditions") or {})
            _n_unavail = sum(
                1 for _v in _road_nc.values()
                if _v.get("condition") in ("road_closed", "flooded_impassable", "unknown")
            )
            _summary_parts = [
                f"**Flood scenario:** {_sum_rp}",
                f"**Barangays:** {len(_sum_bgys)}",
                f"**People to evacuate:** {_sum_td:,}",
                f"**Assigned:** {_sum_ta:,}",
            ]
            if _sum_tu > 0:
                _summary_parts.append(f"**Without assignment:** {_sum_tu:,}")
            _summary_parts.append(f"**Facilities:** {_sum_facs}")
            _flood_exp_m = metrics.get("flood_exposed_length_m", 0.0) or 0.0
            if _flood_exp_m > 0:
                _summary_parts.append(f"**Flood-exposed route:** {_flood_exp_m:,.0f} m")
            if _n_unavail > 0:
                _summary_parts.append(f"**Unavailable roads:** {_n_unavail}")
            st.caption("  ·  ".join(_summary_parts))

        # ── Map control buttons ───────────────────────────────────────────
        if result is not None:
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            if _mc1.button("All routes", use_container_width=True, key="mc_all_routes"):
                st.session_state["selected_origin_node"] = None
                st.rerun()
            _sel_for_mc = st.session_state.get("selected_origin_node")
            _mc2.button(
                "Focus selected" if _sel_for_mc is not None else "Focus selected",
                use_container_width=True,
                key="mc_focus_sel",
                disabled=_sel_for_mc is None,
            )
            _road_nc_mc = dict(st.session_state.get("road_name_conditions") or {})
            if _mc3.button(
                "Road conditions",
                use_container_width=True,
                key="mc_road_conds",
                disabled=not _road_nc_mc,
            ):
                # Highlight all override polylines (clear per-road highlight)
                st.session_state["highlighted_override_road"] = None
                st.rerun()
            if _mc4.button("Fit plan", use_container_width=True, key="mc_fit_plan"):
                st.session_state["_fit_bounds"] = True
                st.rerun()

        # Highlighted override pairs for "Show on map" per-entry action
        _hl_override_road = st.session_state.get("highlighted_override_road")
        _hl_override_pairs: set[tuple[int, int]] | None = None
        if _hl_override_road:
            _rg = _build_road_groups()
            _hl_segs = _rg.get(_hl_override_road, [])
            if _hl_segs:
                _hl_override_pairs = set(_hl_segs)

        # Consume fit_bounds flag before rendering
        _fit_bounds_now = st.session_state.pop("_fit_bounds", False)

        try:
            fmap = build_analytical_map(
                G=G,
                return_period=return_period,
                capacities=_disp_shelters,
                result=result,
                metrics=metrics,
                selected_origin_node=highlight_node,
                ordinary_path=None,  # not shown in planner — see Technical Details
                floodroute_path=None,  # replaced by all_routes
                reference_path=None,
                alg_c_assigned_shelter=st.session_state["alg_c_assigned_shelter"],
                all_routes=result.routes if result is not None else None,
                algorithm="C",
                facility_registry=FACILITY_REGISTRY,
                node_info=_NODE_INFO,
                demand_fraction=demand_fraction,
                shelter_labels=_disp_labels,
                osm_candidates=OSM_CANDIDATES,
                active_osm_ids=activated_ids,
                operating_mode=OperatingMode.CONTROLLED_RESEARCH,
                road_overrides=_get_override_store(),
                highlighted_override_pairs=_hl_override_pairs,
                shelter_types=shelter_types,
                fit_bounds_to_routes=_fit_bounds_now,
            )
            st_folium(fmap, use_container_width=True, height=700)
        except FileNotFoundError as exc:
            st.error(
                f"Map data missing — a required file is not found:\n\n```\n{exc}\n```"
            )

        # Coverage banner — plain language states only
        if metrics is not None:
            _td = metrics.get("total_demand", 0)
            _ta = metrics.get("total_assigned", 0)
            _tu = _td - _ta
            if _ta == 0:
                st.error(
                    "**NO PLAN FOUND** — no one could be assigned to a facility. "
                    "Check that facilities are reachable and road conditions are set correctly."
                )
            elif _tu > 0:
                st.warning(
                    f"**PARTIAL COVERAGE** — {_ta:,} of {_td:,} people have an assignment. "
                    f"**{_tu:,} remain without one** (insufficient reachable facility capacity)."
                )
            else:
                st.success(
                    f"**FULL COVERAGE** — all {_ta:,} people have an assignment."
                )
        else:
            st.info(
                "Complete steps 1–5 and click **Generate evacuation plan** to see results."
            )

    # ── RIGHT COLUMN: Results ─────────────────────────────────────────────
    with col_right:
        if _result_is_stale and st.session_state["metrics"] is not None:
            st.error(
                "Results below are from a previous run. "
                "Click **Generate evacuation plan** to apply the current scenario.",
                icon="🔄",
            )

        if st.session_state["result"] is not None and st.session_state["metrics"] is not None:
            result = st.session_state["result"]
            metrics = st.session_state["metrics"]
            run_params = st.session_state["run_params"]

            _shelters_r: dict[int, int] = (
                run_params[4] if run_params and len(run_params) > 4 else effective_shelters
            )
            _labels_r: dict[int, str] = (
                run_params[5] if run_params and len(run_params) > 5 else shelter_labels
            )
            _rp_r: str = run_params[0] if run_params else return_period

            _td = metrics.get("total_demand", 0)
            _ta = metrics.get("total_assigned", 0)
            _tu = _td - _ta

            st.subheader("Evacuation plan")
            _m1, _m2 = st.columns(2)
            _m1.metric("People to evacuate", f"{_td:,}")
            _m2.metric("With an assignment", f"{_ta:,}")
            if _tu > 0:
                st.error(
                    f"**{_tu:,} without an assignment** ({_tu / max(_td, 1):.0%})"
                )

            # ── Per-barangay breakdown ────────────────────────────────────
            st.markdown("**By barangay pickup point**")
            _reachable_set = {o for (o, _s) in result.od_costs_scenario}
            _bgy_rows: list[dict] = []
            for _o, _od in sorted(result.demands.items(), key=lambda x: -x[1]):
                if _od <= 0:
                    continue
                _bname = _NODE_INFO.get(_o, {}).get("name") or f"Pickup point {_o}"
                _pairs = {
                    (o, s): u
                    for (o, s), u in result.assignments.items()
                    if o == _o and u > 0
                }
                _a_o = sum(_pairs.values())
                _u_o = _od - _a_o
                _bgy_rows.append({
                    "node": _o,
                    "name": _bname, "demand": _od,
                    "assigned": _a_o, "unassigned": _u_o,
                    "pairs": _pairs,
                    "in_graph": _o in G.nodes,
                    "reachable": _o in _reachable_set,
                })

            _run_store_r = _get_override_store()
            for _row in _bgy_rows[:12]:
                _o = _row["node"]
                _is_hl = _o == st.session_state.get("selected_origin_node")
                _expander_label = (
                    ("★ " if _is_hl else "")
                    + f"{_row['name']} — {_row['demand']:,} people"
                )
                with st.expander(
                    _expander_label,
                    expanded=(_row["unassigned"] > 0 or _is_hl),
                ):
                    # Highlight / de-highlight button
                    if st.button(
                        "★ Highlighted on map" if _is_hl else "Highlight on map",
                        key=f"hl_{_o}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_origin_node"] = (
                            None if _is_hl else _o
                        )
                        st.rerun()

                    if not _row["in_graph"]:
                        st.caption(
                            "No route found in the current road map "
                            "(barangay pickup point not in graph)."
                        )
                    elif not _row["reachable"]:
                        st.caption("No reachable facility in this scenario.")
                    else:
                        # Per-facility assignments with route details
                        _fac_loads: dict[int, int] = {}
                        for (_, s), u in _row["pairs"].items():
                            _fac_loads[s] = _fac_loads.get(s, 0) + u
                        _fac_sorted = sorted(_fac_loads.items(), key=lambda x: -x[1])
                        for _s, _u in _fac_sorted:
                            _fname = _labels_r.get(_s, f"Facility {_s}")
                            _fload = int(metrics.get(f"shelter_{_s}_load", 0))
                            _fcap = _shelters_r.get(_s, 0)
                            _frem = max(0, _fcap - _fload)
                            st.caption(f"**{_u:,} →** {_fname}")
                            # Route detail
                            _path_r = result.routes.get((_o, _s)) or []
                            if len(_path_r) >= 2:
                                _dm, _rnames, _rconds = _route_detail(
                                    G, _path_r, _run_store_r, _rp_r
                                )
                                st.caption(f"  Distance: {_dm / 1000:.1f} km")
                                if _rnames:
                                    st.caption(
                                        "  Roads: " + " → ".join(_rnames)
                                    )
                                if _rconds:
                                    for _rc in _rconds:
                                        st.caption(f"  ⚠ {_rc}")
                                else:
                                    st.caption("  Conditions: all passable")
                            # Remaining capacity after complete plan
                            if _frem == 0:
                                st.caption(f"  {_fname}: full ({_fcap:,} spaces, all assigned)")
                            else:
                                st.caption(f"  {_fname}: {_frem:,} spaces remaining")

                        if _row["unassigned"] > 0:
                            st.error(
                                f"**{_row['unassigned']:,} people without an assignment** — "
                                "insufficient reachable facility capacity"
                            )
                        else:
                            st.caption(f"✓ All {_row['demand']:,} people assigned")

            if len(_bgy_rows) > 12:
                st.caption(f"… and {len(_bgy_rows) - 12} more barangays (see Technical Details)")

            st.divider()

            # ── Facility utilisation ──────────────────────────────────────
            st.markdown("**Facility utilisation**")
            _reachable_nodes = {s for (_, s) in result.od_costs_scenario}
            for _sn, _sc in sorted(_shelters_r.items()):
                _slbl = _labels_r.get(_sn, f"Facility {_sn}")
                _load = int(metrics.get(f"shelter_{_sn}_load", 0))
                _rem = max(0, _sc - _load)
                _pct = _load / _sc if _sc > 0 else 0.0
                if _sn not in _reachable_nodes:
                    _fstatus = "No route found"
                elif _load >= _sc:
                    _fstatus = "Full"
                elif _load == 0:
                    _fstatus = f"Empty ({_sc:,} spaces available)"
                else:
                    _fstatus = f"{_rem:,} spaces remaining"
                st.caption(
                    f"**{_slbl}**: {_load:,} / {_sc:,} ({_pct:.0%}) — {_fstatus}"
                )

        else:
            st.info("Results will appear here after generating a plan.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: TECHNICAL DETAILS
# ═══════════════════════════════════════════════════════════════════════════
with technical_tab:
    st.header("Technical Details")
    st.caption("Research and algorithm documentation for thesis inspection.")

    # ── Algorithms ─────────────────────────────────────────────────────────
    with st.expander("Algorithms A, B, C", expanded=False):
        st.markdown("""
**Algorithm A — Ordinary nearest-shelter**
Routes via plain edge length (no flood penalty, no capacity constraint).
Each origin's full demand is sent to the nearest reachable shelter.
Capacity violations are reported but not enforced.

**Algorithm B — Static flood-aware nearest-shelter**
Routes via flood-aware edge costs (conservative hazard policy).
Only `modelled_dry` and `flooded` edges are traversable under RP scenarios.
Each origin's full demand goes to the nearest reachable shelter by flood-aware cost.
No capacity constraint.

**Algorithm C — FloodRoute flood-aware min-cost-flow** *(used by Evacuation Planner)*
Full capacitated assignment via min-cost-flow (network simplex solver).
Lexicographic objective: maximise assigned demand, then minimise total routing cost.
Capacity is strictly enforced. Demand is split across facilities as needed.
""")

    # ── Hamilton allocation ─────────────────────────────────────────────────
    with st.expander("Hamilton allocation and demand rounding", expanded=False):
        st.markdown("""
Demand is distributed proportionally using the **largest-remainder method** (Hamilton method):

                            )
                        else:
                            st.caption(f"✓ All {_row['demand']:,} people assigned")

            if len(_bgy_rows) > 12:
                st.caption(f"… and {len(_bgy_rows) - 12} more barangays (see Technical Details)")

            st.divider()

            # ── Facility utilisation ──────────────────────────────────────
            st.markdown("**Facility utilisation**")
            _reachable_nodes = {s for (_, s) in result.od_costs_scenario}
            for _sn, _sc in sorted(_shelters_r.items()):
                _slbl = _labels_r.get(_sn, f"Facility {_sn}")
                _load = int(metrics.get(f"shelter_{_sn}_load", 0))
                _rem = max(0, _sc - _load)






























""")
        _run_alg_label = st.radio(
            "Algorithm",
            ["C — FloodRoute MCF", "A — Ordinary nearest", "B — Flood-aware nearest"],
            key="tech_alg_radio",
        )
    # ── Provenance and limitations ──────────────────────────────────────────










        _run_frac = st.selectbox(
            "Demand fraction",
            options=[0.10, 0.25, 0.50],
            value=0.25,
            format_func=lambda f: f"{f:.0%}",
            key="tech_frac",
        )
        _run_rp = st.selectbox("Return period", list(RETURN_PERIODS), index=1, key="tech_rp")
        st.caption(
            "**Legacy preset** — uses named scenario nodes 33 and 58 "
            "(``legacy_unknown_shelter_node_33`` and Atabay Elementary School). "
            "These are the original Stage 8 scenario-based supply points, "
            "retained for reproducibility. Capacities 12,000 and 10,000 are "
            "experimental assumptions, not official figures."
        )
        if st.button("Run Stage 8 benchmark (legacy preset)", key="tech_run_stage8"):
            with st.spinner("Running…"):
                _G8 = _load_graph()
                _demands8, _ = build_demands(origins, _run_frac)
                _caps8 = dict(SCENARIO_SHELTER_CAPACITIES)  # explicit legacy preset
                _alg8 = _run_alg_label[0]
                if _alg8 == "A":
                    _r8 = run_ordinary_nearest(_G8, _demands8, _caps8, _run_rp)
                elif _alg8 == "B":
                    _r8 = run_flood_aware_nearest(_G8, _demands8, _caps8, _run_rp)
                else:
                    _r8 = run_floodroute_assignment(_G8, _demands8, _caps8, _run_rp)
                _m8 = compute_metrics(_r8, _G8, total_population=total_pop)
                st.metric("Assignment rate", f"{_m8['assignment_rate']:.1%}")
                st.metric("Total assigned", f"{_m8['total_assigned']:,}")
                st.metric("Total unassigned", f"{_m8['total_unassigned']:,}")
