"""Stage 11 integrity validation for experiment results.

Validates a list of ScenarioResult objects against an ExperimentConfig and
expected demands.  Returns a list of violation strings (empty = valid).
"""

from __future__ import annotations


def validate_experiment(
    results: list,
    config: object,  # ExperimentConfig
    demands_by_scenario: dict,  # {ScenarioKey: dict[int, int]}
) -> list[str]:
    """Validate a completed experiment for internal consistency.

    Parameters
    ----------
    results:
        List of ``ScenarioResult`` from ``run_experiment``.
    config:
        ``ExperimentConfig`` used to generate the results.
    demands_by_scenario:
        ``{ScenarioKey: {origin_node: demand_units}}`` — the demand dicts
        used for each scenario.  Used for cross-checking assignment totals.

    Returns
    -------
    list[str]
        Violation descriptions.  An empty list means the experiment is valid.

    Checks
    ------
    1. All requested ScenarioKeys are represented in results.
    2. For every origin in demands: assigned + unassigned = demand.
    3. No shelter exceeds adjusted capacity (for algorithms C and B+).
    4. No unreachable OD pair has positive flow.
    5. Recalculated objective = Σ f[o,s] * od_costs_scenario[(o,s)] is
       consistent with reported penalized_cost_m_eq.
    6. Assignment rate = assigned / total_demand is consistent.
    """
    from floodroute.experiments.runner import generate_scenarios

    violations: list[str] = []

    # Build a lookup from key to result
    result_map = {sr.key: sr for sr in results}

    # Check 1: all requested scenario keys are represented
    expected_keys = generate_scenarios(config)
    for key in expected_keys:
        if key not in result_map:
            violations.append(
                f"Missing scenario: alg={key.algorithm} rp={key.return_period} "
                f"frac={key.demand_fraction} mult={key.capacity_multiplier} "
                f"penalty={key.flood_penalty}"
            )

    # Checks 2-5: per-scenario validation
    for sr in results:
        pass  # NOTE: rest of file not captured
