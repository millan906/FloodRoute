"""Folium layer builder for the 'Documented and candidate facilities' FeatureGroup.

Reference layer only — these facilities are not used by the current optimization
experiment.  Renders exactly the facilities documented in
``facilities.FACILITY_REGISTRY`` using marker styles that encode designation type
and operational status.

Design constraints
------------------
• Only facilities with ``has_coordinates=True`` can appear as map markers.
• This layer is display-only.  No facility here participates in optimization.
• Marker color encodes designation/status — no "active" or "green" state is shown
  because no facility is used by the algorithm.
• Tooltips expose all evidence fields at the level of confidence the source supports.
• SJDB-001 is labelled as historically documented, not currently verified.
• SJDB-002/003/004 are labelled as unverified research leads.
"""

from __future__ import annotations

import html as _html

import folium

from .facilities import (
    DESIGNATION_TYPE_LABELS,
    EVIDENCE_TIER_LABELS,
    FACILITY_MARKER_CONFIG,
    FACILITY_TYPE_LABELS,
    FacilityRecord,
)

# ---------------------------------------------------------------------------
# Context-note helpers (field-derived, not hardcoded per facility_id)
# ---------------------------------------------------------------------------


def _context_note(fac: FacilityRecord) -> str | None:
    """Return a short display-level caution string derived from evidence fields.

    Returns ``None`` when no special caution applies.
    """
    if fac.designation_type == "permanent" and fac.operational_status == "unknown":
        return (
            "Historically documented purpose-built evacuation center. "
            "Current operational status: Unknown. "
            "Current LGU verification: Not obtained. "
            "Note: \u2018permanent\u2019 reflects the facility\u2019s design intent "
            "and 2019 OCD/DPWH formal designation — it does not establish verified "
            "current availability."
        )
    if fac.designation_type in ("candidate_only", "historically_activated"):
        return (
            "Unverified research lead \u2014 source document not available in the "
            "repository. Names, designation, and historical use are not independently "
            "verified."
        )
    return None


# ---------------------------------------------------------------------------
# Tooltip builder
# ---------------------------------------------------------------------------

_SECTION = (
    "<div style='margin-top:6px;margin-bottom:2px;"
    "font-size:11px;font-weight:bold;color:#374151;'>{title}</div>"
)
_ROW = (
    "<div style='display:flex;justify-content:space-between;"
    "gap:12px;font-size:11px;'>"
    "<span style='color:#6B7280;flex-shrink:0'>{label}</span>"
    "<span style='text-align:right;color:#111827'>{value}</span></div>"
)
_BADGE_TMPL = (
    "<span style='display:inline-block;padding:1px 6px;"
    "border-radius:9px;font-size:10px;font-weight:600;"
    "background:{bg};color:{fg};'>{text}</span>"
)

_DESIGNATION_BADGE_COLORS: dict[str, tuple[str, str]] = {
    "permanent": ("#FEF3C7", "#92400E"),          # amber — historically documented, not verified
    "contingency": ("#FEF3C7", "#92400E"),
    "historically_activated": ("#F3F4F6", "#374151"),
    "candidate_only": ("#F3F4F6", "#374151"),
}

_STATUS_BADGE_COLORS: dict[str, tuple[str, str]] = {
    "operational": ("#D1FAE5", "#065F46"),
    "standby": ("#E0F2FE", "#0C4A6E"),
    "unknown": ("#FEF3C7", "#92400E"),
    "project_only": ("#F3F4F6", "#374151"),
}


def _badge(text: str, colors: tuple[str, str]) -> str:
    return _BADGE_TMPL.format(bg=colors[0], fg=colors[1], text=_html.escape(text))


def _row(label: str, value: str) -> str:
    return _ROW.format(label=_html.escape(label), value=value)


def _section(title: str) -> str:
    return _SECTION.format(title=_html.escape(title))


def facility_tooltip_html(fac: FacilityRecord) -> str:
    """Return an HTML string for a facility tooltip (folium Tooltip).

    Parameters
    ----------
    fac:
        The immutable evidence record.
    """
    e = _html.escape
    parts: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    parts.append(
        f"<div style='font-size:13px;font-weight:bold;color:#1D4ED8;"
        f"border-bottom:1px solid #E5E7EB;padding-bottom:4px;margin-bottom:4px;'>"
        f"{e(fac.facility_name)}</div>"
    )

    # ── Reference-layer notice ────────────────────────────────────────────
    parts.append(
        "<div style='font-size:10px;color:#6B7280;margin-bottom:4px;'>"
        "Reference layer only — not used by the current optimization experiment"
        "</div>"
    )

    # ── Context caution (field-derived) ───────────────────────────────────
    note = _context_note(fac)
    if note:
        parts.append(
            f"<div style='font-size:10px;color:#92400E;background:#FFFBEB;"
            f"border:1px solid #FDE68A;border-radius:4px;padding:4px 6px;margin-bottom:4px;'>"
            f"{e(note)}</div>"
        )

    # ── Identity ─────────────────────────────────────────────────────────
    parts.append(_section("Identity"))
    parts.append(_row("Facility ID", e(fac.facility_id)))
    parts.append(_row("Type", e(FACILITY_TYPE_LABELS.get(fac.facility_type, fac.facility_type))))
    parts.append(_row("Barangay", e(fac.barangay_name)))

    # ── Designation ──────────────────────────────────────────────────────
    parts.append(_section("Designation"))
    des_colors = _DESIGNATION_BADGE_COLORS.get(fac.designation_type, ("#F3F4F6", "#374151"))
    des_label = DESIGNATION_TYPE_LABELS.get(fac.designation_type, fac.designation_type)
    parts.append(_row("Designation", _badge(des_label.split("—")[0].strip(), des_colors)))
    parts.append(
        f"<div style='font-size:10px;color:#6B7280;margin-top:2px;'>"
        f"Source: {e(fac.designation_source)}</div>"
    )

    # ── Status ────────────────────────────────────────────────────────────
    parts.append(_section("Status"))
    op_colors = _STATUS_BADGE_COLORS.get(fac.operational_status, ("#F3F4F6", "#374151"))
    op_label = fac.operational_status.replace("_", " ").title()
    parts.append(_row("Operational", _badge(op_label, op_colors)))
    parts.append(_row("Verification", e(fac.verification_status.title())))
    tier_label = EVIDENCE_TIER_LABELS.get(fac.evidence_tier, fac.evidence_tier)
    parts.append(_row("Evidence tier", e(tier_label)))

    # ── Coordinates ───────────────────────────────────────────────────────
    parts.append(_section("Coordinates"))
    if fac.has_coordinates:
        parts.append(_row("Centroid", f"{fac.latitude:.6f}, {fac.longitude:.6f}"))
        if fac.coordinate_source:
            parts.append(
                f"<div style='font-size:10px;color:#6B7280;'>"
                f"Source: {e(fac.coordinate_source)}</div>"
            )
        if fac.coordinate_uncertainty_m is not None:
            parts.append(_row("Uncertainty", f"±{fac.coordinate_uncertainty_m:.0f} m"))
    else:
        parts.append(
            "<div style='font-size:11px;color:#9CA3AF;'>No coordinates documented</div>"
        )

    # ── Entrance / access proxy ───────────────────────────────────────────
    parts.append(_section("Entrance / access point"))
    if fac.has_entrance:
        parts.append(_row("Entrance", f"{fac.entrance_lat:.6f}, {fac.entrance_lon:.6f}"))
        if fac.snapped_node_id is not None:
            pipeline_note = (
                "pipeline" if fac.snap_is_pipeline_result else "research reference"
            )
            parts.append(
                _row(
                    "Snapped node",
                    f"{fac.snapped_node_id} "
                    f"({fac.snap_distance_m:.1f} m; {pipeline_note})",
                )
            )
    else:
        parts.append(
            "<div style='font-size:11px;color:#9CA3AF;'>Entrance not documented</div>"
        )

    # ── Capacity ──────────────────────────────────────────────────────────
    parts.append(_section("Capacity"))
    if fac.official_capacity is not None:
        parts.append(_row("Official capacity", f"{fac.official_capacity:,}"))
    else:
        parts.append(_row("Official capacity", "Not documented"))
    if fac.historical_capacity_note:
        parts.append(
            f"<div style='font-size:10px;color:#6B7280;margin-top:2px;'>"
            f"Historical note: {e(fac.historical_capacity_note)}</div>"
        )

    # ── Source ────────────────────────────────────────────────────────────
    parts.append(_section("Source"))
    parts.append(_row("Title", e(fac.source_title)))
    parts.append(_row("Date", e(fac.source_date)))
    if fac.issuing_office:
        parts.append(_row("Office", e(fac.issuing_office)))
    if fac.source_url:
        parts.append(
            f"<div style='font-size:10px;margin-top:2px;'>"
            f"<a href='{e(fac.source_url)}' target='_blank' "
            f"style='color:#2563EB;'>Source link</a></div>"
        )

    wrapper = (
        "<div style='font-family:sans-serif;max-width:360px;"
        "padding:8px 10px;line-height:1.4;'>"
        + "".join(parts)
        + "</div>"
    )
    return wrapper


# ---------------------------------------------------------------------------
# Marker builder
# ---------------------------------------------------------------------------


def _marker_color(fac: FacilityRecord) -> str:
    """Determine Folium marker color for a facility.

    No green/active color is used — this is a reference layer only.
    """
    cfg = FACILITY_MARKER_CONFIG.get(fac.facility_type, {})
    if fac.designation_type == "candidate_only" or fac.operational_status == "project_only":
        return cfg.get("color_project_only", "lightgray")
    return cfg.get("color_unknown", "blue")


def facility_marker(fac: FacilityRecord) -> folium.Marker:
    """Return a styled ``folium.Marker`` for *fac*.

    Only call this for facilities where ``fac.has_coordinates`` is ``True``.
    No activation state is shown — this layer is reference-only.
    """
    cfg = FACILITY_MARKER_CONFIG.get(fac.facility_type, {"icon": "building", "color_unknown": "blue"})
    color = _marker_color(fac)
    tooltip_html = facility_tooltip_html(fac)

    return folium.Marker(
        location=[fac.latitude, fac.longitude],
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
        icon=folium.Icon(
            color=color,
            icon=cfg.get("icon", "building"),
            prefix="fa",
        ),
    )


# ---------------------------------------------------------------------------
# Layer builder
# ---------------------------------------------------------------------------


def add_facility_layer(
    m: folium.Map,
    registry: list[FacilityRecord],
) -> int:
    """Add a 'Documented and candidate facilities' FeatureGroup to *m*.

    This is a reference-only display layer.  No facility here participates in
    optimization or routing.  Only facilities with ``has_coordinates=True``
    are rendered as markers.

    Parameters
    ----------
    m:
        The folium Map to add the layer to.
    registry:
        List of ``FacilityRecord`` objects (typically ``FACILITY_REGISTRY``).

    Returns
    -------
    int
        Number of facilities rendered as map markers (those with coordinates).
    """
    fg = folium.FeatureGroup(name="Documented and candidate facilities", show=True)
    mapped = 0

    for fac in registry:
        if not fac.has_coordinates:
            continue  # cannot place marker without centroid
        marker = facility_marker(fac)
        marker.add_to(fg)
        mapped += 1

    fg.add_to(m)
    return mapped
