"""Analytical Folium map builder for the FloodRoute dashboard — Stage 9.

Loads all geodata once per process and caches it in WGS84.  The map
includes:

  - San Jose de Buenavista municipal boundary
  - OSM waterways (rivers, canals, streams)
  - Hazard-attributed road network coloured by the selected return period
  - Barangay origin nodes (selected barangay highlighted)
  - Scenario shelter markers with capacity / load tooltips
  - Ordinary route overlay (blue dashed — Algorithm A)
  - FloodRoute route overlay (green solid — Algorithm C)
  - Layer controls for all togglable layers

Coordinate conversion
---------------------
EPSG:32651 (UTM Zone 51N) → EPSG:4326 is applied once via GeoPandas
``.to_crs()`` for bulk GeoDataFrame data.  A ``pyproj.Transformer`` is
used for individual node lookups (graph node x/y attributes).

Geometry simplification
-----------------------
Edge geometries are simplified with a 2 m tolerance in EPSG:32651 before
reprojection.  This reduces vertex count for dashboard rendering without
modifying the scientific source data stored in the GeoPackage.

Road status colour scheme (selected return period)
--------------------------------------------------
  modelled_dry           : #9CA3AF  light gray — routable, dry
  flooded (depth < 0.5 m): #FCD34D  yellow     — shallow inundation
  flooded (0.5–1.0 m)    : #F97316  orange     — moderate inundation
  flooded (> 1.0 m)      : #EF4444  red        — deep inundation
  no_overlap / outside_domain / missing / unknown:
                           #E5E7EB  very pale gray — unavailable under
                                    conservative hazard policy
"""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
from branca.element import Element as _BrancaElement
from pyproj import Transformer

from floodroute.experiments.algorithms import RunResult

from .facilities import FacilityRecord
from .facility_layer import add_facility_layer

# ---------------------------------------------------------------------------
# File paths (relative to project root)
# ---------------------------------------------------------------------------

_ENRICHED_GPKG = Path("data/processed/hazard/PH0600613_phase_b_enriched.gpkg")
_WATERWAYS_WGS84_GPKG = Path("data/processed/osm/PH0600613_waterways_wgs84.gpkg")
_BOUNDARY_WGS84_GPKG = Path("data/processed/admin/municipalities_wgs84.gpkg")

_MUNICIPALITY_PSGC = "PH0600613"
_SJDB_CENTER: tuple[float, float] = (10.7606, 121.9463)

# ---------------------------------------------------------------------------
# Coordinate transformer — loaded once at import time
# ---------------------------------------------------------------------------

_UTM51N_TO_WGS84 = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)

# ---------------------------------------------------------------------------
# Module-level geodata cache (populated on first access)
# ---------------------------------------------------------------------------

_EDGES_GDF: gpd.GeoDataFrame | None = None
_WATERWAYS_GDF: gpd.GeoDataFrame | None = None
_BOUNDARY_GDF: gpd.GeoDataFrame | None = None

# Columns retained from the enriched GeoPackage for dashboard rendering
_EDGE_KEEP_COLS: list[str] = [
    "name",
    "highway",
    "length_m",
    "jrc_rp10_status",
    "jrc_rp10_depth_max_m",
    "jrc_rp20_status",
    "jrc_rp20_depth_max_m",
    "jrc_rp100_status",
    "jrc_rp100_depth_max_m",
]

# ---------------------------------------------------------------------------
# Road hazard colour / weight helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Route and waterway rendering constants (referenced in legend and tests)
# ---------------------------------------------------------------------------

#: Ordinary shortest route — dark navy distinguishes it from lighter waterways.
ORDINARY_ROUTE_COLOR: str = "#1E3A8A"

#: FloodRoute MCF route (Algorithm C) — green, solid, thicker.
FLOODROUTE_COLOR: str = "#16A34A"

#: Flood-aware feasibility reference path — purple dashed, thin.
REFERENCE_ROUTE_COLOR: str = "#7C3AED"

#: OSM waterways — light blue, thinner than routes.
WATERWAY_COLOR: str = "#60A5FA"
WATERWAY_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Road hazard status helpers
# ---------------------------------------------------------------------------

#: Status → routing availability description
STATUS_AVAILABILITY: dict[str, str] = {
    "modelled_dry": "Available (dry)",
    "flooded": "Available (flood-penalised)",
    "no_overlap": "Unavailable — no JRC pixel overlap",
    "outside_domain": "Unavailable — outside hydraulic domain",
    "missing": "Unavailable — missing data",
}


def flooded_color(depth_m: float | None) -> str:
    """Return the fill colour for a flooded edge based on maximum depth.

    Parameters
    ----------
    depth_m:
        JRC maximum inundation depth in metres, or ``None`` when absent.
    """
    if depth_m is None or depth_m != depth_m:  # also catches NaN
        return "#F97316"  # orange — depth unknown but flooded
    if depth_m < 0.5:
        return "#FCD34D"  # yellow — shallow
    if depth_m < 1.0:
        return "#F97316"  # orange — moderate
    return "#EF4444"  # red — deep


def edge_color(status: str | None, depth_m: float | None) -> str:
    """Return the stroke colour for a road edge by flood status and depth.

    Parameters
    ----------
    status:
        JRC status string for the selected return period, or ``None``.
    depth_m:
        JRC maximum inundation depth in metres (used only when flooded).
    """
    if status == "flooded":
        return flooded_color(depth_m)
    if status == "modelled_dry":
        return "#9CA3AF"  # light gray
    return "#E5E7EB"  # very pale gray — unavailable


def edge_weight(status: str | None) -> float:
    """Return the stroke weight for a road edge by flood status."""
    if status == "flooded":
        return 2.5
    if status == "modelled_dry":
        return 1.5
    return 0.8


def edge_opacity(status: str | None) -> float:
    """Return the stroke opacity for a road edge by flood status."""
    if status in ("flooded", "modelled_dry"):
        return 0.85
    return 0.35


# ---------------------------------------------------------------------------
# Geodata loaders (module-level cache)
# ---------------------------------------------------------------------------


def _get_edges_wgs84() -> gpd.GeoDataFrame:
    """Load, simplify, reproject, and cache the enriched edge GeoDataFrame.

    Returns a GeoDataFrame with selected columns in EPSG:4326.  Geometry is
    simplified with a 2 m tolerance in EPSG:32651 before reprojection;
    the source GeoPackage is not modified.

    Raises
    ------
    FileNotFoundError
        With a descriptive message if the source GeoPackage is absent.
    """
    global _EDGES_GDF  # noqa: PLW0603
    if _EDGES_GDF is None:
        if not _ENRICHED_GPKG.exists():
            raise FileNotFoundError(
                f"Hazard-attributed edge file not found: {_ENRICHED_GPKG}\n"
                "Run the Stage 7 hazard-enrichment pipeline to generate it."
            )
        gdf = gpd.read_file(str(_ENRICHED_GPKG), columns=_EDGE_KEEP_COLS + ["geometry"])
        gdf["geometry"] = gdf["geometry"].simplify(2.0, preserve_topology=True)
        _EDGES_GDF = gdf.to_crs("EPSG:4326")
    return _EDGES_GDF


def _get_waterways_wgs84() -> gpd.GeoDataFrame:
    """Load and cache the SJDB waterways GeoDataFrame (already in WGS84).

    Raises
    ------
    FileNotFoundError
        With a descriptive message if the source GeoPackage is absent.
    """
    global _WATERWAYS_GDF  # noqa: PLW0603
    if _WATERWAYS_GDF is None:
        if not _WATERWAYS_WGS84_GPKG.exists():
            raise FileNotFoundError(
                f"Waterway file not found: {_WATERWAYS_WGS84_GPKG}\n"
                "Run the Stage 3 OSM waterway extraction pipeline to generate it."
            )
        _WATERWAYS_GDF = gpd.read_file(str(_WATERWAYS_WGS84_GPKG))
    return _WATERWAYS_GDF


def _get_boundary_wgs84() -> gpd.GeoDataFrame:
    """Load and cache the SJDB municipal boundary (already in WGS84).

    Only retains columns needed for rendering and tooltip to avoid
    Timestamp / non-JSON-serializable fields present in the source file.

    Raises
    ------
    FileNotFoundError
        With a descriptive message if the source GeoPackage is absent.
    """
    global _BOUNDARY_GDF  # noqa: PLW0603
    if _BOUNDARY_GDF is None:
        if not _BOUNDARY_WGS84_GPKG.exists():
            raise FileNotFoundError(
                f"Municipal boundary file not found: {_BOUNDARY_WGS84_GPKG}\n"
                "Run the Stage 2 admin boundary pipeline to generate it."
            )
        muni = gpd.read_file(str(_BOUNDARY_WGS84_GPKG))
        sjdb = muni[muni["adm3_pcode"] == _MUNICIPALITY_PSGC].copy()
        _BOUNDARY_GDF = sjdb[["adm3_name", "adm3_pcode", "area_sqkm", "geometry"]].copy()
    return _BOUNDARY_GDF


# ---------------------------------------------------------------------------
# Node coordinate helper
# ---------------------------------------------------------------------------


def node_to_latlon(G: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    """Return ``(lat, lon)`` in WGS84 for a graph node.

    Graph nodes carry ``x`` and ``y`` attributes in EPSG:32651.

    Parameters
    ----------
    G:
        Road graph with ``x``/``y`` node attributes in EPSG:32651.
    node:
        Integer node ID.

    Raises
    ------
    KeyError
        If *node* is absent from *G*.
    """
    attrs = G.nodes[node]
    lon, lat = _UTM51N_TO_WGS84.transform(float(attrs["x"]), float(attrs["y"]))
    return lat, lon


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------


#: Scenario shelter labels — controlled scenario supply points, not real facilities.
_SHELTER_LABELS: dict[int, str] = {
    33: "Scenario Shelter A (node 33)",
    58: "Scenario Shelter B (node 58)",
}


def _build_legend_html(algorithm: str = "C") -> str:
    """Return inline HTML for the compact permanent map legend.

    Route entries are muted (opacity reduced) when not applicable to the
    currently selected algorithm.
    """

    def _line(color: str, dash: bool = False, weight: int = 3) -> str:
        style = (
            f"display:inline-block;width:26px;height:0;"
            f"border-top:{weight}px {'dashed' if dash else 'solid'} {color};"
            "vertical-align:middle;flex-shrink:0;"
        )
        return f'<span style="{style}"></span>'

    def _block(color: str, height: int = 4) -> str:
        return (
            f'<span style="display:inline-block;width:20px;height:{height}px;'
            f'background:{color};vertical-align:middle;flex-shrink:0;"></span>'
        )

    def _circle(fill: str, border: str, size: int = 8) -> str:
        return (
            f'<span style="display:inline-block;width:{size}px;height:{size}px;'
            f'border-radius:50%;background:{fill};border:2px solid {border};'
            f"vertical-align:middle;flex-shrink:0;\"></span>"
        )

    def _row(swatch: str, label: str, muted: bool = False) -> str:
        opacity = "opacity:0.35;" if muted else ""
        return (
            f'<div style="display:flex;align-items:center;gap:5px;'
            f'margin:1px 0;{opacity}">'
            f"{swatch}"
            f'<span style="flex:1">{label}</span></div>'
        )

    def _section(title: str) -> str:
        return (
            f'<div style="font-weight:700;color:#374151;'
            f"margin:6px 0 2px;font-size:10px;letter-spacing:0.04em;\">"
            f"{title}</div>"
        )

    mcf_active = algorithm == "C"
    reference_applicable = algorithm in ("A", "B")

    rows: list[str] = []

    rows.append(_section("ROAD / HAZARD STATUS"))
    rows.append(_row(_block("#9CA3AF"), "Modelled dry (selected RP)"))
    rows.append(_row(_block("#FCD34D"), "Flooded &lt;0.5 m"))
    rows.append(_row(_block("#F97316"), "Flooded 0.5–1.0 m"))
    rows.append(_row(_block("#EF4444"), "Flooded &gt;1.0 m"))
    rows.append(_row(
        '<span style="display:inline-block;width:20px;height:0;'
        'border-top:2px dashed #CBD5E1;vertical-align:middle;flex-shrink:0;"></span>',
        "Unavailable / unknown",
    ))
    rows.append(_row(_line(WATERWAY_COLOR, weight=2), "OSM waterway"))
    rows.append(_row(_block("#1E40AF", height=2), "Municipal boundary"))

    rows.append(_section("ROUTES"))
    rows.append(_row(
        _line(ORDINARY_ROUTE_COLOR, dash=True, weight=3),
        "Alg A — ordinary (ignores hazard &amp; capacity)",
    ))
    rows.append(_row(
        _line(FLOODROUTE_COLOR, weight=5),
        "Alg C — FloodRoute MCF (flood-aware, capacity-constrained)",
        muted=not mcf_active,
    ))
    rows.append(_row(
        _line(REFERENCE_ROUTE_COLOR, dash=True, weight=2),
        "Flood-aware reference (not an Alg C result)",
        muted=not reference_applicable,
    ))

    rows.append(_section("DOCUMENTED &amp; CANDIDATE FACILITIES"))
    rows.append(_row(
        '<span style="font-size:11px;line-height:1;flex-shrink:0;">&#x1F3E5;</span>',
        "Evacuation center (reference only — not used by optimization)",
    ))
    rows.append(_row(
        '<span style="font-size:10px;line-height:1;flex-shrink:0;color:#6B7280;">◯</span>',
        "Candidate only / project only — not designated",
    ))

    rows.append(_section("ORIGINS &amp; SHELTERS"))
    rows.append(_row(_circle("#F59E0B", "#92400E", 10), "Selected barangay origin"))
    rows.append(_row(_circle("#3B82F6", "#1E40AF", 7), "Other assigned origin"))
    rows.append(_row(_circle("#6B7280", "#374151", 7), "Disconnected origin"))
    rows.append(_row(
        '<span style="font-size:12px;line-height:1;flex-shrink:0;">🏠</span>',
        "Scenario Shelter A / B",
    ))

    body = "\n".join(rows)
    return f"""<div id="fr-legend" style="
    position:fixed;bottom:24px;left:10px;z-index:1000;
    background:rgba(255,255,255,0.93);border:1px solid #CBD5E1;
    border-radius:6px;padding:7px 10px 8px;
    font-size:10.5px;font-family:-apple-system,sans-serif;
    max-width:230px;box-shadow:0 2px 6px rgba(0,0,0,0.14);
    color:#1F2937;line-height:1.45;">
  <details open>
    <summary style="font-weight:700;font-size:11px;cursor:pointer;
        user-select:none;display:flex;justify-content:space-between;">
      Map Legend
      <span style="font-size:9px;opacity:0.55;margin-left:4px;">&#9660;</span>
    </summary>
    <div style="margin-top:4px;">
{body}
    </div>
  </details>
</div>"""


def build_analytical_map(
    G: nx.MultiDiGraph,
    return_period: str,
    capacities: dict[int, int],
    *,
    result: RunResult | None = None,
    metrics: dict | None = None,
    selected_origin_node: int | None = None,
    ordinary_path: list[int] | None = None,
    floodroute_path: list[int] | None = None,
    reference_path: list[int] | None = None,
    alg_c_assigned_shelter: int | None = None,
    algorithm: str = "C",
    facility_registry: list[FacilityRecord] | None = None,
    zoom_start: int = 13,
) -> folium.Map:
    """Build the full analytical folium Map for the FloodRoute dashboard.

    Parameters
    ----------
    G:
        Road graph with ``x``/``y`` node attributes in EPSG:32651.
    return_period:
        Selected JRC return period (``'RP10'``, ``'RP20'``, ``'RP100'``).
        Determines road hazard coloring and tooltip fields.
    capacities:
        Scenario shelter capacities ``{node_id: capacity}``.
    result:
        Full-municipality algorithm run result.  When present, all origin
        nodes are plotted with assignment status.
    metrics:
        Flat metrics dict from ``compute_metrics``.  Used for shelter load
        tooltips.
    selected_origin_node:
        Graph node ID of the selected barangay origin.  Highlighted with a
        larger, amber marker.
    ordinary_path:
        Ordered list of graph node IDs for the ordinary shortest route.
        Minimises total road length; traverses all edges regardless of
        flood status.  Drawn as a blue dashed polyline.
    floodroute_path:
        Ordered list of graph node IDs for the **actual** Algorithm C MCF
        route.  Only set when Algorithm C ran and assigned the origin.
        Drawn as a green solid polyline.
    reference_path:
        Flood-aware nearest path computed independently of the MCF result.
        Only displayed when Algorithm A or B is selected, and explicitly
        labelled "Feasible path reference — not selected by Algorithm C".
        Drawn as a purple dashed polyline.
    alg_c_assigned_shelter:
        Shelter node that Algorithm C actually assigned the selected origin
        to.  When set, that shelter marker is highlighted to confirm the
        route endpoint matches the MCF assignment.
    algorithm:
        Currently selected algorithm (``'A'``, ``'B'``, or ``'C'``).
        Controls which route legend entries are highlighted vs muted and
        the default visibility of the flood-aware reference path.
    facility_registry:
        List of ``FacilityRecord`` objects to render as the 'Documented and
        candidate facilities' reference layer.  Only facilities with
        ``has_coordinates=True`` are rendered as markers.  This layer is
        display-only — no facility here participates in optimization.
        Pass ``None`` to omit the layer.
    zoom_start:
        Initial folium zoom level (default 13).

    Returns
    -------
    folium.Map
        Configured map — render with ``streamlit_folium.st_folium``.

    Raises
    ------
    FileNotFoundError
        If a required source GeoPackage is absent (boundary, waterways,
        or hazard-attributed edges).
    """
    rp_lower = return_period.lower()
    status_col = f"jrc_{rp_lower}_status"
    depth_col = f"jrc_{rp_lower}_depth_max_m"

    # ── Base map (tiles=None; vector layers work without any tile layer) ────
    # Both tile layers require network access — neither is guaranteed to work
    # offline.  The boundary, roads, waterways, origins, shelters and routes
    # are vector GeoJSON/FeatureGroup layers and remain usable when tiles are
    # disabled or unavailable.
    m = folium.Map(location=_SJDB_CENTER, zoom_start=zoom_start, tiles=None)

    folium.TileLayer(
        tiles="CartoDB Positron",
        name="Basemap — CartoDB Positron (requires network access)",
        attr=(
            "Map tiles by <a href='https://carto.com/'>CartoDB</a>, "
            "under CC BY 3.0. Data by "
            "<a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> "
            "contributors."
        ),
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Basemap — OpenStreetMap (requires network access)",
        attr=(
            "Map data © <a href='https://www.openstreetmap.org/copyright'>"
            "OpenStreetMap</a> contributors"
        ),
        show=False,
    ).add_to(m)

    # ── Municipal boundary ────────────────────────────────────────────────
    boundary = _get_boundary_wgs84()
    folium.GeoJson(
        data=boundary.to_json(),
        name="Municipal boundary",
        style_function=lambda _: {
            "color": "#1E40AF",
            "weight": 2.5,
            "fillOpacity": 0.03,
            "fillColor": "#BFDBFE",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["adm3_name"],
            aliases=["Municipality"],
            localize=False,
            sticky=False,
        ),
        show=True,
    ).add_to(m)

    # ── Waterways ─────────────────────────────────────────────────────────
    waterways = _get_waterways_wgs84()
    folium.GeoJson(
        data=waterways[["waterway", "name", "geometry"]].to_json(),
        name="Waterways (OSM)",
        style_function=lambda _: {
            "color": WATERWAY_COLOR,
            "weight": WATERWAY_WEIGHT,
            "opacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["waterway", "name"],
            aliases=["Type", "Name"],
            localize=False,
            sticky=False,
        ),
        show=True,
    ).add_to(m)

    # ── Road network (hazard-attributed) ──────────────────────────────────
    edges_base = _get_edges_wgs84()

    # Build a display copy with current-RP derived columns for the tooltip.
    # Using pandas where/fillna on a copy avoids modifying the cached GDF.
    road_gdf = edges_base[_EDGE_KEEP_COLS + ["geometry"]].copy()
    road_gdf["_road_name"] = road_gdf["name"].fillna("—")

    # Generic "_status" and "_depth_m" columns so the style function
    # does not need to know the specific RP key.
    road_gdf["_status"] = road_gdf[status_col]
    road_gdf["_depth_m"] = road_gdf[depth_col]

    # Human-readable depth string for the tooltip.
    road_gdf["_depth_display"] = road_gdf["_depth_m"].apply(
        lambda d: f"{d:.2f} m" if d is not None and d == d else "—"
    )

    # Routing availability under the conservative hazard policy.
    road_gdf["_availability"] = road_gdf["_status"].apply(
        lambda s: STATUS_AVAILABILITY.get(s or "missing", "Unavailable — unknown status")
    )

    tooltip_cols = ["_road_name", "highway", "_status", "_depth_display", "_availability"]
    tooltip_aliases = [
        "Road name",
        "Type",
        f"Flood status ({return_period})",
        "Max depth",
        "Routing availability",
    ]

    def _road_style(feature: dict) -> dict:
        props = feature.get("properties", {})
        s = props.get("_status")
        d = props.get("_depth_m")
        return {
            "color": edge_color(s, d),
            "weight": edge_weight(s),
            "opacity": edge_opacity(s),
        }

    folium.GeoJson(
        data=road_gdf[tooltip_cols + ["geometry"]].to_json(),
        name=f"Roads — {return_period} flood status",
        style_function=_road_style,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_cols,
            aliases=tooltip_aliases,
            localize=False,
            sticky=False,
        ),
        show=True,
    ).add_to(m)

    # ── Barangay origin nodes ─────────────────────────────────────────────
    origin_group = folium.FeatureGroup(name="Barangay origins", show=True)

    if result is not None:
        # Plot all assigned origins with status colouring.
        for o, demand in result.demands.items():
            if o not in G.nodes:
                continue
            lat, lon = node_to_latlon(G, o)
            is_selected = o == selected_origin_node
            assigned = sum(
                v for (orig, _s), v in result.assignments.items() if orig == o
            )
            color = "#F59E0B" if is_selected else ("#3B82F6" if assigned > 0 else "#6B7280")
            border = "#92400E" if is_selected else "#1E40AF" if assigned > 0 else "#374151"
            radius = 8 if is_selected else 4
            folium.CircleMarker(
                location=(lat, lon),
                radius=radius,
                color=border,
                weight=2 if is_selected else 1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9 if is_selected else 0.7,
                tooltip=(
                    f"Node {o} — demand {demand:,}, assigned {assigned:,}"
                    + (" ★ selected" if is_selected else "")
                ),
            ).add_to(origin_group)

    elif selected_origin_node is not None and selected_origin_node in G.nodes:
        # No run yet — show only the selected origin.
        lat, lon = node_to_latlon(G, selected_origin_node)
        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color="#92400E",
            weight=2,
            fill=True,
            fill_color="#F59E0B",
            fill_opacity=0.9,
            tooltip=f"Selected origin: node {selected_origin_node}",
        ).add_to(origin_group)

    origin_group.add_to(m)

    # ── Scenario shelters ─────────────────────────────────────────────────
    shelter_group = folium.FeatureGroup(name="Scenario shelters", show=True)
    for s, cap in sorted(capacities.items()):
        if s not in G.nodes:
            continue
        lat, lon = node_to_latlon(G, s)
        load = int(metrics.get(f"shelter_{s}_load", 0)) if metrics else 0
        remaining = cap - load
        label = _SHELTER_LABELS.get(s, f"Scenario Shelter (node {s})")
        is_alg_c_target = s == alg_c_assigned_shelter
        tooltip_lines = [
            label,
            f"Scenario capacity: {cap:,}",
            f"Assigned demand: {load:,}",
            f"Remaining capacity: {remaining:,}",
        ]
        if is_alg_c_target:
            tooltip_lines.append("★ Route destination selected by Algorithm C")
        # Highlight the shelter Algorithm C actually assigned this origin to
        icon_color = "red" if is_alg_c_target else "darkred"
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color=icon_color, icon="home", prefix="fa"),
            tooltip="<br>".join(tooltip_lines),
        ).add_to(shelter_group)
    shelter_group.add_to(m)

    # ── Ordinary shortest route — blue dashed ─────────────────────────────
    # Minimises total road length; traverses ALL edges including those with
    # flooded, no_overlap, outside_domain, or unknown status.  Never dry-only.
    if ordinary_path and len(ordinary_path) >= 2:
        ordinary_group = folium.FeatureGroup(
            name="Ordinary shortest route (A)", show=True
        )
        latlons = [
            node_to_latlon(G, n) for n in ordinary_path if n in G.nodes
        ]
        if len(latlons) >= 2:
            folium.PolyLine(
                latlons,
                color=ORDINARY_ROUTE_COLOR,
                weight=4,
                opacity=0.85,
                dash_array="10 5",
                tooltip=(
                    "Ordinary shortest route (Algorithm A) — "
                    "minimises road length, no flood penalty, "
                    "traverses all edges regardless of hazard status"
                ),
            ).add_to(ordinary_group)
        ordinary_group.add_to(m)

    # ── FloodRoute MCF route (Algorithm C) — green solid ──────────────────
    # Only rendered when Algorithm C actually assigned this origin.
    # Source: result.routes[(origin, shelter)] from the MCF solve.
    if floodroute_path and len(floodroute_path) >= 2:
        flood_group = folium.FeatureGroup(
            name="FloodRoute MCF route (C)", show=True
        )
        latlons = [
            node_to_latlon(G, n) for n in floodroute_path if n in G.nodes
        ]
        if len(latlons) >= 2:
            folium.PolyLine(
                latlons,
                color=FLOODROUTE_COLOR,
                weight=5,
                opacity=0.95,
                tooltip=(
                    "FloodRoute MCF route (Algorithm C) — "
                    "actual route assigned by the flood-aware min-cost-flow solver"
                ),
            ).add_to(flood_group)
        flood_group.add_to(m)

    # ── Feasible path reference — purple dashed ───────────────────────────
    # Shown only when Algorithm A or B is selected.  This is a separately
    # computed flood-aware nearest path and is NOT an Algorithm C result.
    if reference_path and len(reference_path) >= 2:
        ref_group = folium.FeatureGroup(
            name="Feasible path reference (flood-aware — not selected by Algorithm C)",
            show=False,
        )
        latlons = [
            node_to_latlon(G, n) for n in reference_path if n in G.nodes
        ]
        if len(latlons) >= 2:
            folium.PolyLine(
                latlons,
                color=REFERENCE_ROUTE_COLOR,
                weight=2.5,
                opacity=0.75,
                dash_array="4 4",
                tooltip=(
                    "Feasible path reference — flood-aware nearest path "
                    "(independently computed, NOT selected by Algorithm C)"
                ),
            ).add_to(ref_group)
        ref_group.add_to(m)

    # ── Documented and candidate facilities (reference layer only) ───────
    if facility_registry:
        add_facility_layer(m, facility_registry)

    # ── Compact permanent legend (HTML — always visible) ──────────────────
    m.get_root().html.add_child(_BrancaElement(_build_legend_html(algorithm)))

    # ── Layer control (collapsed by default; expand via icon) ─────────────
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    return m
