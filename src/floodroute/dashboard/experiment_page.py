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
import json
import zipfile

import streamlit as st

from floodroute.dashboard.road_overrides import RoadOverrideStore
from floodroute.experiments.runner import (
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
    # Nominal capacities display
    # -----------------------------------------------------------------------
    nominal_caps = st.session_state.get("nominal_capacities", SCENARIO_SHELTER_CAPACITIES)
    with st.expander("Nominal experimental capacities"):
        st.caption("These are scenario-based values, not verified shelter records.")
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
        )

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
            manifest_preview = {
                "experiment_id": config_used.experiment_id,
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
            }
            st.json(manifest_preview)

    # -----------------------------------------------------------------------
    # Download evidence bundle (zip)
    # -----------------------------------------------------------------------
    st.subheader("Download evidence bundle")

    if config_used is not None:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Manifest
            manifest_json = json.dumps(
                {
                    "experiment_id": config_used.experiment_id,
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
                },
                indent=2,
            )
            zf.writestr("manifest.json", manifest_json)

            # Scenario results CSV
            import csv as _csv

            csv_buf = io.StringIO()
            if results:
                first_sr = results[0]
                flat_keys = [
                    k for k, v in first_sr.metrics.items()
                    if not isinstance(v, dict)
                ]
                fieldnames = [
                    "algorithm", "return_period", "demand_fraction",
                    "capacity_multiplier", "flood_penalty", "run_status",
                ] + flat_keys
                writer = _csv.DictWriter(
                    csv_buf, fieldnames=fieldnames, extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                for sr in results:
                    row = {
                        "algorithm": sr.key.algorithm,
                        "return_period": sr.key.return_period,
                        "demand_fraction": sr.key.demand_fraction,
                        "capacity_multiplier": sr.key.capacity_multiplier,
                        "flood_penalty": str(sr.key.flood_penalty),
                        "run_status": sr.run_status,
                    }
                    for k in flat_keys:
                        row[k] = sr.metrics.get(k, "")
                    writer.writerow(row)
            zf.writestr("scenario_results.csv", csv_buf.getvalue())

        zip_buf.seek(0)
        st.download_button(
            label="Download evidence bundle (zip)",
            data=zip_buf,
            file_name=f"floodroute_experiment_{config_used.experiment_id}.zip",
            mime="application/zip",
        )

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Manifest
            manifest_json = json.dumps(
                {
                    "experiment_id": config_used.experiment_id,
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
                },
                indent=2,
            )
            zf.writestr("manifest.json", manifest_json)

            # Scenario results CSV
            import csv as _csv

            csv_buf = io.StringIO()
            if results:
                first_sr = results[0]
                flat_keys = [
                    k for k, v in first_sr.metrics.items()
                    if not isinstance(v, dict)
                ]
                fieldnames = [
                    "algorithm", "return_period", "demand_fraction",
                    "capacity_multiplier", "flood_penalty", "run_status",
                ] + flat_keys
                writer = _csv.DictWriter(
                    csv_buf, fieldnames=fieldnames, extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                for sr in results:
                    row = {
                        "algorithm": sr.key.algorithm,
                        "return_period": sr.key.return_period,
                        "demand_fraction": sr.key.demand_fraction,
                        "capacity_multiplier": sr.key.capacity_multiplier,
                        "flood_penalty": str(sr.key.flood_penalty),
                        "run_status": sr.run_status,
                    }
                    for k in flat_keys:
                        row[k] = sr.metrics.get(k, "")
                    writer.writerow(row)
            zf.writestr("scenario_results.csv", csv_buf.getvalue())

        zip_buf.seek(0)
        st.download_button(
            label="Download evidence bundle (zip)",
            data=zip_buf,
            file_name=f"floodroute_experiment_{config_used.experiment_id}.zip",
            mime="application/zip",
        )
