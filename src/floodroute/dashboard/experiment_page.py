"""Stage 11 Experimental Evaluation dashboard page.

Renders the Streamlit "Experimental Evaluation" tab via
``render_experiment_page(G, origins)``.

This page provides:
- Experiment configuration (algorithms, return periods, demand fractions,
  capacity multipliers, flood penalties)
- Pilot run ("one scenario per algorithm") and full-run buttons
- Results tables and comparison charts
- Manifest preview and evidence bundle download

All labels use plain language: "metres-equivalent", "nominal experimental
capacities", "planning scenario, not real-time prediction".
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import streamlit as st

from floodroute.dashboard.road_overrides import RoadOverrideStore
from floodroute.experiments.manifest import save_experiment
from floodroute.experiments.runner import (
    _DEFAULT_BARANGAY_GPKG,
    _DEFAULT_GRAPHML,
    _DEFAULT_NODES_GPKG,
    _DEFAULT_POP_CSV,
    SCENARIO_SHELTER_CAPACITIES,
    ExperimentConfig,
    generate_scenarios,
    make_experiment_config,
    run_experiment,
)


def render_experiment_page(G: object, origins: list) -> None:  # type: ignore[type-arg]
    """Render the Experimental Evaluation Streamlit page.

    Parameters
    ----------
    G:
        Road graph (``nx.MultiDiGraph``).
    origins:
        List of ``BarangayOrigin`` objects.
    """
    st.header("Experimental Evaluation")
    st.caption(
        "Planning scenario — not a real-time prediction.  "
        "All capacities are nominal experimental values."
    )

    # -----------------------------------------------------------------------
    # Experiment name
    # -----------------------------------------------------------------------
    st.text_input(
        "Experiment name",
        value="Stage 11 evaluation",
        help="Human-readable label for this experiment run (not used in hashing).",
    )

    # -----------------------------------------------------------------------
    # Factor selectors
    # -----------------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        sel_algorithms = st.multiselect(
            "Algorithms",
            options=["A", "B", "B+", "C"],
            default=["A", "B", "B+", "C"],
            help=(
                "A: ordinary nearest (no flood, no capacity). "
                "B: flood-aware nearest (no capacity). "
                "B+: CASPER-inspired flood-aware greedy (capacitated, demand splitting). "
                "C: global shelter allocation — exact minimum-cost flow "
                "(optimal for the formulated MCF with scalar OD costs, capacitated)."
            ),
        )
        sel_return_periods = st.multiselect(
            "Return periods",
            options=["RP10", "RP20", "RP100"],
            default=["RP100"],
            help="RP10 = ~10% annual chance; RP20 = ~5%; RP100 = ~1%.",
        )
        sel_demand_fractions = st.multiselect(
            "Demand fractions (%)",
            options=[10, 25, 50],
            default=[25],
            format_func=lambda x: f"{x}%",
        )
    with col2:
        sel_capacity_multipliers = st.multiselect(
            "Capacity multipliers",
            options=[0.50, 0.75, 1.00, 1.25],
            default=[1.00],
            help=(
                "Scale nominal shelter capacities by this factor. "
                "adjusted_capacity = round(nominal_capacity × multiplier). "
                "Both nominal and adjusted values are recorded in evidence outputs."
            ),
        )
        sel_flood_penalties = st.multiselect(
            "Flood penalties",
            options=["1.0", "5.0", "10.0", "20.0", "prohibited"],
            default=["10.0"],
            help=(
                "Penalty multiplier applied to flooded edges. "
                "'prohibited' blocks flooded edges entirely."
            ),
        )

    # Convert selections to correct types
    demand_fractions_f = tuple(v / 100 for v in sel_demand_fractions)
    capacity_multipliers_f = tuple(float(v) for v in sel_capacity_multipliers)
    flood_penalties_typed = tuple(
        "prohibited" if v == "prohibited" else float(v) for v in sel_flood_penalties
    )

    # Scenario count estimate
    if sel_algorithms and sel_return_periods and sel_demand_fractions \
            and sel_capacity_multipliers and sel_flood_penalties:
        n_scenarios = (
            len(sel_algorithms)
            * len(sel_return_periods)
            * len(sel_demand_fractions)
            * len(sel_capacity_multipliers)
            * len(sel_flood_penalties)
        )
        st.info(f"Estimated scenario count: **{n_scenarios}**")
    else:
        st.warning("Select at least one option in each factor.")
        return

    # -----------------------------------------------------------------------
    # Nominal capacities — prefer Evacuation Planner's current facility selection
    # -----------------------------------------------------------------------
    # run_params[4] is the {node_id: capacity} dict built by the Evacuation
    # Planner from the current facility selection.  Using it here ensures that
    # the experiment reflects the same facilities as the planner, not the
    # hard-coded Stage 8 legacy nodes (33, 58).  Falls back to the Stage 11
    # scenario defaults only when the planner has not yet been run.
    _ep_run_params = st.session_state.get("run_params")
    _planner_node_caps: dict[int, int] | None = (
        _ep_run_params[4]
        if isinstance(_ep_run_params, (list, tuple))
        and len(_ep_run_params) > 4
        and isinstance(_ep_run_params[4], dict)
        and _ep_run_params[4]
        else None
    )
    nominal_caps: dict[int, int] = (
        _planner_node_caps
        or st.session_state.get("nominal_capacities")
        or dict(SCENARIO_SHELTER_CAPACITIES)
    )
    with st.expander("Nominal experimental capacities"):
        if _planner_node_caps:
            st.caption(
                "Using the current Evacuation Planner facility configuration. "
                "Update facilities in the Evacuation Planner tab."
            )
        else:
            st.caption(
                "Stage 11 scenario defaults (nodes 33, 58 — scenario-based values, "
                "not verified shelter records). "
                "Configure facilities in the Evacuation Planner tab to override."
            )
        for node, cap in nominal_caps.items():
            st.write(f"Shelter node {node}: {cap:,} units")

    # -----------------------------------------------------------------------
    # Active road-condition overrides
    # -----------------------------------------------------------------------
    _override_dict = st.session_state.get("road_override_store_dict") or {}
    _active_store = (
        RoadOverrideStore.from_dict(_override_dict) if _override_dict else RoadOverrideStore()
    )
    _n_overrides = len(_active_store)
    if _n_overrides > 0:
        _override_names = ", ".join(
            str(k) for k in list(_active_store.overrides)[:3]
        )
        if _n_overrides > 3:
            _override_names += f" … (+{_n_overrides - 3} more)"
        st.info(
            f"Road conditions active ({_n_overrides} segment(s)): {_override_names}",
            icon="🛣️",
        )
    else:
        _active_store = None  # pass None when there are no overrides

    # -----------------------------------------------------------------------
    # Run buttons
    # -----------------------------------------------------------------------
    # _override_dict_for_run captures the exact override state used in both
    # _make_config (for the hash) and run_experiment (for routing).  Using
    # the same object ensures the manifest hash matches what was actually run.
    _override_dict_for_run = st.session_state.get("road_override_store_dict") or {}

    def _make_config(pilot: bool) -> ExperimentConfig:
        return make_experiment_config(
            municipality="PH0600613",
            algorithms=tuple(sel_algorithms),
            return_periods=tuple(sel_return_periods),
            demand_fractions=demand_fractions_f,
            capacity_multipliers=capacity_multipliers_f,
            flood_penalties=flood_penalties_typed,
            nominal_capacities=nominal_caps,
            pilot_mode=pilot,
            road_overrides=_override_dict_for_run,
        )

    # Compute the "would-run" config hash for stale detection (cheap — no routing).
    _current_config_hash = _make_config(pilot=False).configuration_hash

    col_pilot, col_full = st.columns(2)

    with col_pilot:
        if st.button("Run Pilot (one per algorithm)", type="secondary"):
            config = _make_config(pilot=True)
            pilot_scenarios = generate_scenarios(config)
            st.write(f"Running {len(pilot_scenarios)} pilot scenario(s)…")
            prog = st.progress(0)
            results = []

            def _cb(i: int, total: int, key: object) -> None:
                prog.progress(i / total)

            with st.spinner("Running pilot…"):
                results = run_experiment(
                    G, origins, config, progress_callback=_cb,
                    road_override_store=_active_store,
                )
            prog.progress(1.0)
            st.session_state["experiment_results"] = results
            st.session_state["experiment_config"] = config
            # Freeze source_scenario_id at run time so export reflects the
            # exact preset that was loaded when this run was triggered.
            st.session_state["experiment_sc_id_at_run"] = st.session_state.get("sc_id")
            st.success(f"Pilot complete: {len(results)} scenario(s).")

    with col_full:
        if st.button("Run Full Experiment", type="primary"):
            config = _make_config(pilot=False)
            all_scenarios = generate_scenarios(config)
            confirmed = st.checkbox(
                f"Confirm: run all {len(all_scenarios)} scenarios?",
                key="confirm_full_run",
            )
            if confirmed:
                prog = st.progress(0)

                def _cb_full(i: int, total: int, key: object) -> None:
                    prog.progress(i / total)

                with st.spinner(f"Running {len(all_scenarios)} scenarios…"):
                    results_full = run_experiment(
                        G, origins, config, progress_callback=_cb_full,
                        road_override_store=_active_store,
                    )
                prog.progress(1.0)
                st.session_state["experiment_results"] = results_full
                st.session_state["experiment_config"] = config
                # Freeze source_scenario_id at run time.
                st.session_state["experiment_sc_id_at_run"] = st.session_state.get("sc_id")
                st.success(f"Full experiment complete: {len(results_full)} scenarios.")

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------
    results = st.session_state.get("experiment_results")
    config_used: ExperimentConfig | None = st.session_state.get("experiment_config")

    if not results:
        st.info("Run a pilot or full experiment above to see results.")
        return

    st.subheader("Results")

    # Summary table
    rows = []
    for sr in results:
        k = sr.key
        m = sr.metrics
        rows.append({
            "Algorithm": k.algorithm,
            "RP": k.return_period,
            "Demand %": f"{k.demand_fraction * 100:.0f}%",
            "Cap ×": k.capacity_multiplier,
            "Penalty": str(k.flood_penalty),
            "Assigned": m.get("assigned_population", "—"),
            "Unassigned": m.get("unassigned_population", "—"),
            "Assignment rate": f"{m.get('assignment_rate', 0):.1%}",
            "Phys. dist. (m-eq)": f"{m.get('physical_route_distance_m', 0):,.0f}",
            "Flooded person-dist. (m)": f"{m.get('flooded_person_distance_m', 0):,.0f}",
            "Feasible": (
                "Yes"
                if (m.get("total_overflow_units", 0) or 0) == 0
                else f"No ({(m.get('total_overflow_units', 0) or 0):,} overflow)"
            ),
            "Status": sr.run_status,
        })

    import pandas as pd  # local import — only needed when results exist

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Direct facility-utilisation CSV download (in-memory, no save required)
    _fac_util_rows: list[dict] = []
    for sr in results:
        k = sr.key
        for s, fm in sorted(sr.facility_metrics.items(), key=str):
            _fac_util_rows.append({
                "algorithm": k.algorithm,
                "return_period": k.return_period,
                "demand_fraction": k.demand_fraction,
                "capacity_multiplier": k.capacity_multiplier,
                "flood_penalty": str(k.flood_penalty),
                "shelter_node": s,
                "nominal_capacity": sr.nominal_capacities.get(s, ""),
                "adjusted_capacity": sr.adjusted_capacities.get(s, fm.get("capacity", "")),
                "load": fm.get("load", 0),
                "utilization": fm.get("utilization", 0.0),
                "overflow": fm.get("overflow", 0),
            })
    if _fac_util_rows:
        _fac_util_buf = io.StringIO()
        _fac_util_fieldnames = list(_fac_util_rows[0].keys())
        import csv as _csv
        _fac_util_writer = _csv.DictWriter(_fac_util_buf, fieldnames=_fac_util_fieldnames,
                                           lineterminator="\n")
        _fac_util_writer.writeheader()
        _fac_util_writer.writerows(_fac_util_rows)
        st.download_button(
            label="Download facility utilisation CSV",
            data=_fac_util_buf.getvalue().encode("utf-8"),
            file_name="facility_metrics.csv",
            mime="text/csv",
            key="download_facility_util_csv",
        )

    # Assignment-rate comparison
    st.subheader("Assignment rate by algorithm")
    rate_data: dict[str, list] = {}
    for sr in results:
        alg = sr.key.algorithm
        rate = sr.metrics.get("assignment_rate", 0)
        rate_data.setdefault(alg, []).append(rate)
    rate_means = {alg: sum(vals) / len(vals) for alg, vals in rate_data.items()}
    st.bar_chart(rate_means)

    # Shelter overflow comparison
    st.subheader("Total overflow units by algorithm")
    overflow_data: dict[str, list] = {}
    for sr in results:
        alg = sr.key.algorithm
        ov = sr.metrics.get("total_overflow_units", 0)
        overflow_data.setdefault(alg, []).append(ov)
    overflow_means = {alg: sum(vals) / len(vals) for alg, vals in overflow_data.items()}
    st.bar_chart(overflow_means)

    # Flooded person-distance comparison
    st.subheader("Flooded person-distance (metres) by algorithm")
    fpd_data: dict[str, list] = {}
    for sr in results:
        alg = sr.key.algorithm
        fpd = sr.metrics.get("flooded_person_distance_m", 0)
        fpd_data.setdefault(alg, []).append(fpd)
    fpd_means = {alg: sum(vals) / len(vals) for alg, vals in fpd_data.items()}
    st.bar_chart(fpd_means)

    # Unassigned reason decomposition
    st.subheader("Unassigned reason decomposition")
    reason_rows = []
    for sr in results:
        k = sr.key
        for reason, count in sr.metrics.get("unassigned_by_reason", {}).items():
            reason_rows.append({
                "Algorithm": k.algorithm,
                "RP": k.return_period,
                "Reason": reason,
                "Units": count,
            })
    if reason_rows:
        st.dataframe(pd.DataFrame(reason_rows), use_container_width=True)
    else:
        st.write("No unassigned demand in any scenario.")

    # -----------------------------------------------------------------------
    # Manifest preview
    # -----------------------------------------------------------------------
    if config_used is not None:
        with st.expander("Manifest preview (JSON)"):
            st.json({
                "experiment_id": config_used.experiment_id,
                "configuration_hash": config_used.configuration_hash,
                "created_utc": config_used.created_utc,
                "municipality": config_used.municipality,
                "algorithms": list(config_used.algorithms),
                "return_periods": list(config_used.return_periods),
                "demand_fractions": list(config_used.demand_fractions),
                "capacity_multipliers": list(config_used.capacity_multipliers),
                "flood_penalties": [str(p) for p in config_used.flood_penalties],
                "nominal_capacities": {
                    str(k): v for k, v in config_used.nominal_capacities.items()
                },
                "pilot_mode": config_used.pilot_mode,
                "num_scenarios": len(results),
            })

    # -----------------------------------------------------------------------
    # Save evidence bundle — explicit action, not automatic
    # -----------------------------------------------------------------------
    st.subheader("Evidence bundle")
    st.caption(
        "Save an immutable evidence folder to disk, then download a ZIP of the "
        "same files.  Saving is an explicit action — runs are never saved automatically."
    )

    _saved_dir: Path | None = st.session_state.get("saved_experiment_dir")

    # Stale-result guard: warn when any relevant input has changed since the
    # last run.  config_used is the frozen ExperimentConfig from the run;
    # _current_config_hash is the hash of what WOULD run with current UI state
    # (computed above from _make_config).  Because road_overrides are included
    # in the hash, any override change also triggers staleness.
    _exp_config_at_run: ExperimentConfig | None = st.session_state.get("experiment_config")
    _results_stale = (
        results is not None
        and config_used is not None
        and config_used.configuration_hash != _current_config_hash
    )
    if _results_stale:
        st.warning(
            "The displayed results are from a previous run configuration. "
            "Re-run the experiment before saving to ensure the bundle matches "
            "the current settings.",
            icon="⚠️",
        )

    _save_disabled = _results_stale or config_used is None

    if st.button(
        "Save evidence bundle to disk",
        key="save_evidence_btn",
        type="primary",
        disabled=_save_disabled,
        help=(
            "Writes experiments/<experiment_id>/ with manifest, CSVs and checksums. "
            "Each save creates a new immutable directory. "
            "Disabled when results are stale (config changed since last run)."
        ),
    ):
        # Reconciliation check: assignments sum must equal total_assigned metric.
        _reconcile_errors: list[str] = []
        for _sr in results:
            _metric_assigned = _sr.metrics.get("total_assigned", None)
            if _metric_assigned is not None:
                _asgn_sum = sum(_sr.assignments.values())
                if _asgn_sum != int(_metric_assigned):
                    _reconcile_errors.append(
                        f"Scenario {_sr.key}: assignments sum {_asgn_sum} "
                        f"≠ total_assigned metric {int(_metric_assigned)}"
                    )
        if _reconcile_errors:
            st.error(
                "Reconciliation failed — assignments CSV would not match metrics. "
                "Do not use this bundle for analysis.\n\n"
                + "\n".join(_reconcile_errors)
            )
        else:
            # Collect facility registry rows from catalog + nominal capacities.
            _fac_rows: list[dict] = []
            try:
                from floodroute.scenario.catalog import build_catalog
                _cat = build_catalog()
                for fid, configured_cap in st.session_state.get("sc_facility_caps", {}).items():
                    _entry = _cat.get(fid)
                    if _entry is not None:
                        _fac_rows.append({
                            "facility_id": fid,
                            "name": _entry.name,
                            "facility_type": _entry.facility_type,
                            "designation_status": _entry.designation_status,
                            "snapped_node": _entry.snapped_node,
                            "snapping_distance_m": _entry.snapping_distance_m,
                            "configured_capacity": configured_cap,
                        })
            except Exception:  # noqa: BLE001
                pass  # catalog unavailable — facility_registry.csv will be empty

            # Dataset inventory paths (relative to project root).
            _project_root = Path(__file__).resolve().parent.parent.parent.parent
            _hazard_gpkg = Path("data/processed/hazard/PH0600613_phase_b_enriched.gpkg")
            _dataset_paths: dict[str, Path] = {
                "road_graph": _project_root / _DEFAULT_GRAPHML,
                "population_csv": _project_root / _DEFAULT_POP_CSV,
                "barangay_boundaries": _project_root / _DEFAULT_BARANGAY_GPKG,
                "nodes_gpkg": _project_root / _DEFAULT_NODES_GPKG,
                "hazard_gpkg": _project_root / _hazard_gpkg,
            }

            with st.spinner("Saving evidence bundle…"):
                try:
                    _saved_path = save_experiment(
                        config_used,
                        results,
                        # Use the sc_id that was active when the run was triggered,
                        # not the current sc_id which may have changed since then.
                        source_scenario_id=st.session_state.get("experiment_sc_id_at_run"),
                        # road_overrides come from config_used.road_overrides (frozen
                        # at run time); no need to re-read mutable session state.
                        facility_registry_rows=_fac_rows or None,
                        origins=origins,
                        dataset_paths=_dataset_paths,
                    )
                    st.session_state["saved_experiment_dir"] = _saved_path
                    _saved_dir = _saved_path
                    st.success(f"Saved: `{_saved_path.name}`")
                except FileExistsError:
                    st.error(
                        "This experiment directory already exists on disk. "
                        "Run the experiment again to generate a new timestamp-based ID."
                    )
                except Exception as _exc:  # noqa: BLE001
                    st.error(f"Save failed: {_exc}")

    # -----------------------------------------------------------------------
    # Download ZIP — built from the saved directory (identical to disk files)
    # -----------------------------------------------------------------------
    if _saved_dir is not None and _saved_dir.exists() and config_used is not None:
        _zip_files = sorted(_saved_dir.iterdir())
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _fp in _zip_files:
                if _fp.is_file():
                    zf.write(_fp, arcname=_fp.name)
        zip_buf.seek(0)
        st.download_button(
            label="Download evidence bundle (zip)",
            data=zip_buf,
            file_name=f"floodroute_experiment_{config_used.experiment_id}.zip",
            mime="application/zip",
            key="download_evidence_zip",
        )
    elif config_used is not None and _saved_dir is None:
        st.info("Save the evidence bundle above to enable download.")
