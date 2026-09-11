"""Municipal boundary containment check for FloodRoute Stage 6B.

Validates that a shelter coordinate lies within (or on the boundary of)
the target municipality polygon.  All geometry operations are performed
in EPSG:32651 (UTM Zone 51N) to use projected, metric coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point

# CRS constants
_WGS84 = "EPSG:4326"
_UTM51N = "EPSG:32651"

_TO_UTM: Transformer = Transformer.from_crs(_WGS84, _UTM51N, always_xy=True)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ContainmentResult:
    """Outcome of a containment check for one shelter candidate.

    Attributes
    ----------
    shelter_id:
        Identifier of the shelter candidate.
    inside:
        True if the candidate coordinate is within or on the boundary.
    distance_to_boundary_m:
        Distance from the point to the polygon boundary in metres.
        Negative values indicate the point is inside (interior distance).
        Zero means exactly on the boundary.
        Positive means outside.
    message:
        Human-readable note (populated when outside or missing coords).
    """

    shelter_id: str
    inside: bool
    distance_to_boundary_m: float | None
    message: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_municipality_polygon(
    gpkg_path: Path,
    psgc: str,
    *,
    psgc_column: str = "adm3_pcode",
) -> gpd.GeoSeries:
    """Load the dissolved polygon for one municipality from a GPKG file.

    Parameters
    ----------
    gpkg_path:
        Path to the municipalities GeoPackage (WGS-84).
    psgc:
        PSGC code of the target municipality (e.g. "PH0600613").
    psgc_column:
        Column name containing PSGC codes.

    Returns
    -------
    GeoSeries with a single row: the dissolved municipal polygon in EPSG:32651.

    Raises
    ------
    KeyError:
        If the PSGC is not found in the GeoPackage.
    """
    gdf = gpd.read_file(gpkg_path)
    mask = gdf[psgc_column] == psgc
    if not mask.any():
        raise KeyError(f"PSGC '{psgc}' not found in {gpkg_path} column '{psgc_column}'.")
    muni = gdf.loc[mask].copy()
    muni_utm: gpd.GeoDataFrame = muni.to_crs(_UTM51N)
    dissolved = muni_utm.dissolve()
    return dissolved.geometry


def check_containment(
    shelter_id: str,
    lat: float | None,
    lon: float | None,
    municipality_geom: gpd.GeoSeries,
) -> ContainmentResult:
    """Check whether a WGS-84 coordinate is within the municipal boundary.

    Parameters
    ----------
    shelter_id:
        Identifier used in the result.
    lat, lon:
        WGS-84 latitude and longitude of the shelter.  If either is None
        the result has ``inside=False`` and a descriptive message.
    municipality_geom:
        GeoSeries (EPSG:32651) of the target municipality, as returned by
        :func:`load_municipality_polygon`.

    Returns
    -------
    ContainmentResult describing the containment outcome.
    """
    if lat is None or lon is None:
        return ContainmentResult(
            shelter_id=shelter_id,
            inside=False,
            distance_to_boundary_m=None,
            message="No coordinates available for containment check.",
        )

    x, y = _TO_UTM.transform(lon, lat)
    pt = Point(x, y)
    poly = municipality_geom.iloc[0]

    inside = poly.contains(pt) or poly.boundary.distance(pt) < 1e-6
    dist = poly.exterior.distance(pt) if not inside else -poly.exterior.distance(pt)

    if inside:
        msg = ""
    else:
        msg = (
            f"Coordinate ({lat:.5f}, {lon:.5f}) is outside the municipal boundary "
            f"by {dist:.1f} m."
        )

    return ContainmentResult(
        shelter_id=shelter_id,
        inside=inside,
        distance_to_boundary_m=round(dist, 3),
        message=msg,
    )


def check_all_containment(
    records: list,  # list[ShelterRecord]
    municipality_geom: gpd.GeoSeries,
) -> list[ContainmentResult]:
    """Check containment for all records against the municipal boundary.

    Uses the facility coordinate (lat/lon), not the entrance coordinate,
    for boundary membership — a shelter may be inside the municipality
    even if its entrance is on a boundary road.
    """
    return [
        check_containment(r.shelter_id, r.latitude, r.longitude, municipality_geom)
        for r in records
    ]
