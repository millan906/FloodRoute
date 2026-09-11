"""Coverage state classification for FloodRoute dashboard.

Do NOT use 'Feasible with warnings' when demand remains unassigned.
Use CoverageState.PARTIAL_COVERAGE for any scenario where assignment is
incomplete or capacity constraints are violated.
"""
from __future__ import annotations

from enum import StrEnum


class CoverageState(StrEnum):
    FULL_COVERAGE = "full_coverage"
    PARTIAL_COVERAGE = "partial_coverage"
    NO_FEASIBLE_PLAN = "no_feasible_plan"


COVERAGE_DESCRIPTIONS: dict[CoverageState, str] = {
    CoverageState.FULL_COVERAGE: (
        "All demand units assigned within shelter capacity. "
        "No capacity violations detected."
    ),
    CoverageState.PARTIAL_COVERAGE: (
        "Some demand units could not be assigned or shelter capacity was exceeded. "
        "Review shelter loads and consider increasing capacity or adjusting demand."
    ),
    CoverageState.NO_FEASIBLE_PLAN: (
        "No demand units were assigned. "
        "The routing model produced no feasible assignments for this scenario."
    ),
}


def classify_coverage(metrics: dict) -> CoverageState:
    """Classify scenario coverage based on assignment rate and violations.

    Parameters
    ----------
    metrics:
        Dict with keys ``assignment_rate`` (float) and
        ``num_capacity_violations`` (int).

    Returns
    -------
    CoverageState
    """
    rate = float(metrics.get("assignment_rate", 1.0))
    violations = int(metrics.get("num_capacity_violations", 0))

    if rate == 0.0:
        return CoverageState.NO_FEASIBLE_PLAN
    if rate < 1.0 or violations > 0:
        return CoverageState.PARTIAL_COVERAGE
    return CoverageState.FULL_COVERAGE


def format_coverage_banner(state: CoverageState, metrics: dict) -> dict:
    """Format a coverage state banner for Streamlit display.

    Parameters
    ----------
    state:
        Classified coverage state.
    metrics:
        Full metrics dict for detail construction.

    Returns
    -------
    dict
        Keys: ``status`` (str), ``label`` (str), ``detail`` (str),
        ``streamlit_type`` (``'success'`` | ``'warning'`` | ``'error'``).
    """
    rate = float(metrics.get("assignment_rate", 1.0))
    unassigned = int(metrics.get("total_unassigned", 0))
    violations = int(metrics.get("num_capacity_violations", 0))
    unreachable = int(metrics.get("num_unreachable_origins", 0))

    if state == CoverageState.FULL_COVERAGE:
        return {
            "status": "full_coverage",
            "label": "Full Coverage",
            "detail": "All demand units assigned within capacity constraints.",
            "streamlit_type": "success",
        }

    if state == CoverageState.PARTIAL_COVERAGE:
        parts = []
        if unassigned > 0:
            parts.append(f"{unassigned:,} units unassigned ({rate:.1%} coverage)")
        if violations > 0:
            parts.append(f"{violations} shelter(s) over capacity")
        if unreachable > 0:
            parts.append(f"{unreachable} origin(s) unreachable")
        detail = "; ".join(parts) if parts else "Partial assignment."
        return {
            "status": "partial_coverage",
            "label": "Partial Coverage",
            "detail": detail,
            "streamlit_type": "warning",
        }

    # NO_FEASIBLE_PLAN
    return {
        "status": "no_feasible_plan",
        "label": "No Feasible Plan",
        "detail": "No demand units were assigned. Check routing model and shelter connectivity.",
        "streamlit_type": "error",
    }
