"""FloodRoute Dashboard — Stage 9 Streamlit MVP.

Launch with::

    streamlit run src/floodroute/dashboard/app.py

Three-column layout:

  Left column   : scenario inputs (algorithm, RP, demand fraction, barangay)
  Centre column : analytical folium map — updates road hazard colours on RP
                  change; adds ordinary + FloodRoute routes on Run
  Right column  : aggregate metrics and shelter utilisation
"""

from __future__ import annotations

import networkx as nx
import streamlit as st
from streamlit_folium import st_folium

from floodroute.dashboard.facilities import (
    DESIGNATION_TYPE_LABELS,
    FACILITY_REGISTRY,
    FACILITY_TYPE_LABELS,
)
from floodroute.dashboard.map_builder import build_analytical_map
from floodroute.dashboard.result_formatter import (
    format_barangay_card,
    format_feasibility_status,
    format_origin_assignment_status,
    format_recommendation,
    format_run_label,
    format_shelter_loads,
    validate_inputs,
)
from floodroute.experiments.algorithms import (
    DEMAND_FRACTIONS,
    RETURN_PERIODS,
    run_flood_aware_nearest,
    run_floodroute_assignment,
    run_ordinary_nearest,
)
from floodroute.experiments.demand import (
    build_demands,
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

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodRoute Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Ordinary-routing weight function (MultiDiGraph format: d = {key: attrs})
# ---------------------------------------------------------------------------


def _ordinary_weight(u: object, v: object, d: dict) -> float | None:
    """Min-length weight across parallel edges (ordinary routing)."""
    best: float | None = None
    for attrs in d.values():
        length = float(attrs.get("length_m") or 0.0)
        if best is None or length < best:
            best = length
    return best if best is not None else 0.0


def _path_length_m(
    G: nx.MultiDiGraph,
    path: list[int],
    return_period: str | None = None,
) -> tuple[float, float]:
    """Return ``(total_m, flood_exposed_m)`` for *path* through *G*.

    Sums the minimum-length parallel edge at each consecutive node pair.
    When *return_period* is given, also accumulates edges whose
    ``jrc_{rp}_status`` equals ``'flooded'``.
    """
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
    """Return the path from *origin* to the nearest reachable shelter.

    Uses a single Dijkstra sweep from *origin*; picks the shelter with the
    lowest path cost.  Returns ``None`` if no shelter is reachable.
    """
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


# ---------------------------------------------------------------------------
# Cached heavy loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading road graph…")
def _load_graph() -> nx.MultiDiGraph:
    return nx.read_graphml(str(_DEFAULT_GRAPHML), node_type=int)


@st.cache_resource(show_spinner="Loading population data…")
def _load_origins():
    records = load_psa_population(_DEFAULT_POP_CSV, adm3_filter=MUNICIPALITY_PSGC)
    origins = snap_barangay_origins(
        records,
        barangay_gpkg=_DEFAULT_BARANGAY_GPKG,
        nodes_gpkg=_DEFAULT_NODES_GPKG,
        municipality_psgc=MUNICIPALITY_PSGC,
        exclude_nodes=set(SCENARIO_SHELTER_CAPACITIES),
    )
    total_pop = sum(o.population_2020 for o in origins)
    return origins, total_pop


# ---------------------------------------------------------------------------
# Eager data load (origins needed for the barangay selector)
# ---------------------------------------------------------------------------
origins, total_pop = _load_origins()

_BARANGAY_OPTIONS: list[str] = ["— (show all origins)"] + [
    f"{o.name} ({o.psgc})" for o in origins
]
_PSGC_TO_ORIGIN: dict[str, int] = {o.psgc: o.origin_node for o in origins}
_PSGC_TO_NAME: dict[str, str] = {o.psgc: o.name for o in origins}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("FloodRoute — Evacuation Routing Dashboard")
st.caption(
    "San Jose de Buenavista · Antique · Philippines · "
    "PSA 2020 Census · Scenario shelters (nodes 33, 58)"
)

# ---------------------------------------------------------------------------
# Three-column layout
# ---------------------------------------------------------------------------
col_left, col_map, col_right = st.columns([1, 2.5, 1.5])

# ── Left column: inputs ──────────────────────────────────────────────────
with col_left:
    st.subheader("Scenario inputs")

    algorithm = st.selectbox(
        "Algorithm",
        options=["A", "B", "C"],
        format_func=lambda a: {
            "A": "A — Ordinary nearest",
            "B": "B — Flood-aware nearest",
            "C": "C — FloodRoute MCF",
        }[a],
        index=2,
        help=(
            "**A**: shortest path ignoring flood hazard; no capacity enforcement.\n\n"
            "**B**: flood-aware shortest path; no capacity enforcement.\n\n"
            "**C**: flood-aware min-cost-flow with capacity constraints."
        ),
    )

    return_period = st.selectbox(
        "Return period",
        options=list(RETURN_PERIODS),
        index=1,  # RP20 default
        help="JRC Global Flood Map return period used for edge hazard classification.",
    )

    demand_fraction = st.select_slider(
        "Demand fraction",
        options=list(DEMAND_FRACTIONS),
        value=0.25,
        format_func=lambda f: f"{f:.0%} of population",
        help="Fraction of PSA 2020 barangay population requiring evacuation.",
    )

    # Barangay selector — highlights one origin on the map and shows its
    # individual ordinary / FloodRoute routes.
    barangay_label = st.selectbox(
        "Highlight barangay",
        options=_BARANGAY_OPTIONS,
        index=0,
        help=(
            "Select a barangay to highlight its origin node and show its "
            "individual ordinary and FloodRoute routes on the map."
        ),
    )

    run_clicked = st.button("Run scenario", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Shelters** *(scenario-based — not real shelter data)*")
    st.caption("• Scenario Shelter A (node 33) — capacity 12,000")
    st.caption("• Scenario Shelter B (node 58) — capacity 10,000")
    st.divider()
    st.caption("**Road graph:** OpenStreetMap-derived road graph via Overpass")
    st.caption(
        "**Flood hazard:** JRC GloFAS modeled hazard scenarios — not live conditions"
    )
    st.caption(
        "**Population:** PhilAtlas secondary table attributed to PSA 2020 CPH"
    )
    st.caption(
        "**Shelters:** Controlled scenario shelters and capacities — not official LGU data"
    )

# ---------------------------------------------------------------------------
# Resolve selected barangay
# ---------------------------------------------------------------------------
selected_origin_node: int | None = None
selected_bgy_name: str | None = None
if barangay_label != _BARANGAY_OPTIONS[0]:
    # Extract PSGC from the label "Name (PSGCCODE)"
    psgc = barangay_label.rsplit("(", 1)[-1].rstrip(")")
    selected_origin_node = _PSGC_TO_ORIGIN.get(psgc)
    selected_bgy_name = _PSGC_TO_NAME.get(psgc)

# Resolve the full OriginRecord for the selected barangay (for population etc.)
selected_origin = (
    next((o for o in origins if o.origin_node == selected_origin_node), None)
    if selected_origin_node is not None
    else None
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in [
    ("result", None),
    ("metrics", None),
    ("run_params", None),
    ("ordinary_path", None),
    ("alg_c_path", None),           # Algorithm C MCF route only
    ("reference_path", None),       # Flood-aware reference; shown with explicit label
    ("alg_c_assigned_shelter", None),
    ("alg_c_origin_status", None),
    ("bgy_ordinary_dist_m", None),  # per-barangay ordinary route distance (m)
    ("bgy_alg_dist_m", None),       # per-barangay selected-algorithm route distance (m)
    ("bgy_flood_exposed_m", None),  # per-barangay flood-exposed distance (m)
    ("run_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------------
if run_clicked:
    errors = validate_inputs(algorithm, return_period, demand_fraction)
    if errors:
        st.session_state["run_error"] = errors
    else:
        st.session_state["run_error"] = None
        with st.spinner("Running scenario…"):
            G = _load_graph()
            demands, _ = build_demands(origins, demand_fraction)

            # Full-municipality run for the selected algorithm.
            if algorithm == "A":
                result = run_ordinary_nearest(
                    G, demands, SCENARIO_SHELTER_CAPACITIES, return_period
                )
            elif algorithm == "B":
                result = run_flood_aware_nearest(
                    G, demands, SCENARIO_SHELTER_CAPACITIES, return_period
                )
            else:
                result = run_floodroute_assignment(
                    G, demands, SCENARIO_SHELTER_CAPACITIES, return_period
                )
            result.demand_fraction = demand_fraction
            metrics = compute_metrics(result, G, total_population=total_pop)

            # ── Route overlays for the selected barangay ──────────────────
            origin = selected_origin_node
            ordinary_path: list[int] | None = None
            alg_c_path: list[int] | None = None       # MCF result only
            reference_path: list[int] | None = None   # Flood-aware reference
            alg_c_assigned_shelter: int | None = None
            alg_c_origin_status: dict | None = None

            if origin is not None:
                # Ordinary shortest route: minimises road length, traverses
                # ALL edges regardless of flood status (no conservative
                # hazard exclusion).
                ordinary_path = _shortest_path_to_shelters(
                    G, origin, SCENARIO_SHELTER_CAPACITIES, _ordinary_weight
                )

                if algorithm == "C":
                    # Algorithm C only: show the actual MCF-assigned route.
                    # If the origin was unassigned, show nothing and record why.
                    alg_c_origin_status = format_origin_assignment_status(
                        origin, result
                    )
                    if alg_c_origin_status["status"] == "assigned":
                        best_pair = max(
                            (
                                (o, s)
                                for (o, s) in result.routes
                                if o == origin
                            ),
                            key=lambda k: result.assignments.get(k, 0),
                            default=None,
                        )
                        if best_pair is not None:
                            alg_c_path = result.routes[best_pair]
                            alg_c_assigned_shelter = best_pair[1]
                    # When unreachable or capacity-exhausted: alg_c_path
                    # remains None — do NOT compute or display a fallback.

                else:
                    # Algorithm A or B: compute a flood-aware nearest path as
                    # a reference only.  Explicitly labelled so it is never
                    # mistaken for an Algorithm C result.
                    flood_weight = make_weight_fn(return_period)
                    reference_path = _shortest_path_to_shelters(
                        G, origin, SCENARIO_SHELTER_CAPACITIES, flood_weight
                    )

            # ── Per-barangay route distances ───────────────────────────────
            bgy_ordinary_dist_m: float | None = None
            bgy_alg_dist_m: float | None = None
            bgy_flood_exposed_m: float | None = None

            if origin is not None:
                if ordinary_path:
                    bgy_ordinary_dist_m, _ = _path_length_m(G, ordinary_path)
                active_path = alg_c_path if algorithm == "C" else reference_path
                if active_path:
                    bgy_alg_dist_m, bgy_flood_exposed_m = _path_length_m(
                        G, active_path, return_period
                    )

            st.session_state.update(
                {
                    "result": result,
                    "metrics": metrics,
                    "run_params": (algorithm, return_period, demand_fraction),
                    "ordinary_path": ordinary_path,
                    "alg_c_path": alg_c_path,
                    "reference_path": reference_path,
                    "alg_c_assigned_shelter": alg_c_assigned_shelter,
                    "alg_c_origin_status": alg_c_origin_status,
                    "bgy_ordinary_dist_m": bgy_ordinary_dist_m,
                    "bgy_alg_dist_m": bgy_alg_dist_m,
                    "bgy_flood_exposed_m": bgy_flood_exposed_m,
                }
            )

# ---------------------------------------------------------------------------
# Centre column: map (always rendered; road colours update with RP selection)
# ---------------------------------------------------------------------------
with col_map:
    if st.session_state.get("run_error"):
        for e in st.session_state["run_error"]:
            st.error(e)

    result = st.session_state["result"]
    metrics = st.session_state["metrics"]
    run_params = st.session_state["run_params"]

    if result is not None and run_params is not None:
        alg, _rp, _frac = run_params
        st.subheader(f"Routes — {format_run_label(alg, _rp, _frac)}")
    else:
        st.subheader("Analytical map")

    G = _load_graph()

    try:
        fmap = build_analytical_map(
            G=G,
            return_period=return_period,
            capacities=SCENARIO_SHELTER_CAPACITIES,
            result=result,
            metrics=metrics,
            selected_origin_node=selected_origin_node,
            ordinary_path=st.session_state["ordinary_path"] if result is not None else None,
            floodroute_path=st.session_state["alg_c_path"] if result is not None else None,
            reference_path=st.session_state["reference_path"] if result is not None else None,
            alg_c_assigned_shelter=st.session_state["alg_c_assigned_shelter"],
            algorithm=algorithm,
            facility_registry=FACILITY_REGISTRY,
        )
        st_folium(fmap, use_container_width=True, height=530)
    except FileNotFoundError as exc:
        st.error(
            f"**Map data error** — a required source file is missing:\n\n```\n{exc}\n```"
        )

    # Feasibility banner
    if metrics is not None:
        fstatus = format_feasibility_status(metrics)
        msg = fstatus["label"]
        if fstatus["warnings"]:
            msg += " — " + "; ".join(fstatus["warnings"])
        if fstatus["status"] == "ok":
            st.success(msg)
        elif fstatus["status"] == "warning":
            st.warning(msg)
        else:
            st.error(msg)
    else:
        st.info(
            "Road hazard colours update when you change the return period. "
            "Select a barangay and click **Run scenario** to draw routes."
        )

# ---------------------------------------------------------------------------
# Right column: selected-barangay card + municipality-wide results
# ---------------------------------------------------------------------------
with col_right:
    if st.session_state["metrics"] is not None:
        metrics = st.session_state["metrics"]
        alg, rp, frac = st.session_state["run_params"]
        result = st.session_state["result"]

        # ── Selected barangay result card ─────────────────────────────────
        if selected_bgy_name and selected_origin is not None and result is not None:
            st.subheader("Selected barangay result")

            origin_node = selected_origin.origin_node
            demand = result.demands.get(origin_node, 0)

            # Sum all positive assignments from this origin across shelters
            assigned_pairs = {
                (o, s): u
                for (o, s), u in result.assignments.items()
                if o == origin_node and u > 0
            }
            assigned = sum(assigned_pairs.values())
            best_shelter = (
                max(assigned_pairs, key=assigned_pairs.__getitem__)[1]
                if assigned_pairs
                else None
            )

            # Derive assignment status for display
            if alg == "C":
                status_dict = st.session_state.get("alg_c_origin_status") or {
                    "status": "unknown",
                    "reason": "",
                }
            else:
                # Algorithms A/B assign all reachable origins
                if assigned > 0:
                    status_dict = {
                        "status": "assigned",
                        "reason": "Assigned (capacity constraints not enforced by Algorithm A/B)",
                    }
                elif st.session_state.get("ordinary_path") is None:
                    status_dict = {
                        "status": "unreachable",
                        "reason": "No path to any shelter in the routing model",
                    }
                else:
                    status_dict = {"status": "unassigned", "reason": ""}

            card = format_barangay_card(
                bgy_name=selected_bgy_name,
                population_2020=selected_origin.population_2020,
                demand=demand,
                assigned=assigned,
                shelter_node=best_shelter,
                ordinary_dist_m=st.session_state.get("bgy_ordinary_dist_m"),
                alg_dist_m=st.session_state.get("bgy_alg_dist_m"),
                flood_exposed_m=st.session_state.get("bgy_flood_exposed_m"),
                assignment_status=status_dict,
            )

            st.markdown(f"**{card['bgy_name']}**")
            cb1, cb2 = st.columns(2)
            cb1.metric("Population (2020)", card["population_2020"])
            cb2.metric("Scenario demand", f"{demand:,}")
            cb1.metric("Assigned", card["assigned"])
            cb2.metric("Unassigned", card["unassigned"])
            st.caption(f"**Assigned shelter:** {card['shelter']}")
            st.caption(f"**Ordinary-route distance:** {card['ordinary_dist']}")
            st.caption(
                f"**Algorithm {alg} route distance:** {card['alg_dist']}"
            )
            st.caption(
                f"**Flood-exposed on route ({rp}):** {card['flood_exposed']}"
            )
            if card["show_reason"]:
                st.caption(
                    f"**Status:** {card['status']} — {card['reason']}"
                )
            st.divider()

        # ── Municipality-wide scenario results ────────────────────────────
        st.subheader("Municipality-wide scenario results")

        st.metric("Assignment rate", f"{metrics['assignment_rate']:.1%}")

        c1, c2 = st.columns(2)
        c1.metric("Assigned", f"{metrics['total_assigned']:,}")
        c2.metric("Unassigned", f"{metrics['total_unassigned']:,}")

        st.metric(
            "Detour ratio",
            f"{metrics['detour_ratio']:.3f}",
            help=(
                "Flood-aware cost / ordinary cost for assigned pairs. "
                "1.000 means no flood avoidance detour."
            ),
        )
        st.metric(
            "Flood-exposed route",
            f"{metrics['flood_exposed_length_m']:,.0f} m",
            help="Total length of assigned route segments crossing flooded road.",
        )
        st.metric(
            "Disconnected origins",
            metrics["num_unreachable_origins"],
            help=(
                "Representative origin nodes disconnected from the main routable "
                "component in the derived OSM road graph."
            ),
        )
        st.metric("Runtime", f"{metrics['runtime_s']:.2f} s")

        st.divider()
        st.subheader("Shelter utilisation")
        for row in format_shelter_loads(metrics, SCENARIO_SHELTER_CAPACITIES):
            prefix = "⚠ " if row["over_capacity"] else ""
            st.write(
                f"{prefix}**{row['shelter']}**: "
                f"{row['load']:,} / {row['capacity']:,} "
                f"({row['utilization']})"
            )

        st.divider()
        st.caption("**Recommendation**")
        st.caption(format_recommendation(alg, metrics))

        st.divider()
        st.caption(
            f"Total population (PSA 2020): "
            f"{metrics.get('total_population_2020', 'n/a'):,}"
        )
        st.caption(
            f"Total scenario demand units ({frac:.0%}): {metrics['total_demand']:,}"
        )
        st.caption(f"Barangay origins: {metrics['num_origins']}")
    else:
        st.info("Results will appear here after running a scenario.")

    # ── Documented and candidate facilities registry panel (always shown) ──
    st.divider()
    with st.expander(
        "Documented and candidate facilities registry (sourced data)", expanded=False
    ):
        st.caption(
            "**Reference layer only — these facilities are not used by the current "
            "optimization experiment.**"
        )
        st.caption(
            "Registry counts: **4** sourced records · "
            "**1** with mappable coordinates (displayed on map) · "
            "**3** without coordinates (not displayed on map) · "
            "**0** registry facilities used by the current optimization · "
            "**2** separate controlled scenario shelter nodes (nodes 33, 58) used by Stage 8."
        )
        for fac in FACILITY_REGISTRY:
            st.markdown(f"**{fac.facility_id}** — {fac.facility_name}")

            # Conservative context label (items 4 and 5)
            if fac.designation_type == "permanent" and fac.operational_status == "unknown":
                st.caption(
                    "Historically documented purpose-built evacuation center. "
                    "Current operational status: Unknown. "
                    "Current LGU verification: Not obtained. "
                    "'Permanent' reflects design intent and 2019 OCD/DPWH formal "
                    "designation — not verified current availability."
                )
            elif fac.designation_type in ("candidate_only", "historically_activated"):
                st.caption(
                    "⚠ Unverified research lead — source document not available "
                    "in the repository. Names, designation, and historical use are "
                    "not independently verified."
                )

            st.caption(
                f"Type: {FACILITY_TYPE_LABELS.get(fac.facility_type, fac.facility_type)} · "
                f"Barangay: {fac.barangay_name}"
            )
            st.caption(
                f"Designation: "
                f"{DESIGNATION_TYPE_LABELS.get(fac.designation_type, fac.designation_type)}"
            )
            st.caption(
                f"Operational: {fac.operational_status} · "
                f"Verification: {fac.verification_status} · "
                f"Evidence tier: {fac.evidence_tier}"
            )
            if fac.has_coordinates:
                st.caption(
                    f"Coordinates: {fac.latitude:.6f}, {fac.longitude:.6f} "
                    f"(±{fac.coordinate_uncertainty_m:.0f} m)"
                )
            else:
                st.caption("Coordinates: not documented — not displayed on map")
            if fac.official_capacity is not None:
                st.caption(f"Official capacity: {fac.official_capacity:,}")
            else:
                st.caption("Official capacity: not documented")
            st.caption(f"Source: {fac.source_title} ({fac.source_date})")
            st.caption(
                f"Eligible for optimization: "
                f"{'Yes' if fac.eligible_for_optimization else 'No'}"
            )
            st.divider()
