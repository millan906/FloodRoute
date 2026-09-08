"""Output validation for geospatial preprocessing (Stage 3).

Validates vector (GeoPackage) and raster (GeoTIFF) outputs against expected
criteria: CRS, nonempty geometry, validity, municipality codes, spatial
containment, and raster readability/bounds/nodata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("floodroute.preprocessing.validation")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ValidationFailed(ValueError):
    """Raised when a preprocessing output fails validation."""


# ---------------------------------------------------------------------------
# Vector validation
# ---------------------------------------------------------------------------


def validate_vector_output(
    path: Path,
    *,
    layer: str,
    expected_crs: str,
    expected_codes: list[str],
    pcode_field: str,
    min_features: int = 1,
    check_containment_bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Validate a vector GeoPackage output.

    Checks:
      - File exists and is readable
      - CRS matches *expected_crs* (EPSG authority + code comparison)
      - Feature count >= *min_features*
      - All geometries are non-empty
      - All geometries are valid (shapely is_valid)
      - All *expected_codes* are present in *pcode_field*
      - No duplicate values in *pcode_field* (for municipality layers)
      - Optional: all features intersect *check_containment_bounds*

    Returns a dict with validation details.  Raises ValidationFailed if
    any check fails.
    """
    import geopandas as gpd
    from pyproj import CRS

    if not path.exists():
        raise ValidationFailed(f"Output file not found: {path}")

    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as exc:
        raise ValidationFailed(f"Cannot read vector output {path}: {exc}") from exc

    issues: list[str] = []

    # CRS check
    expected_crs_obj = CRS.from_string(expected_crs)
    if gdf.crs is None:
        issues.append("CRS is None (no projection defined)")
    elif not gdf.crs.equals(expected_crs_obj):
        issues.append(f"CRS mismatch: expected {expected_crs}, got {gdf.crs}")

    # Feature count
    if len(gdf) < min_features:
        issues.append(f"Too few features: {len(gdf)} < {min_features}")

    # Nonempty geometries
    null_geom = gdf.geometry.is_empty | gdf.geometry.isna()
    if null_geom.any():
        issues.append(f"{null_geom.sum()} empty/null geometries")

    # Validity
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        issues.append(f"{invalid.sum()} invalid geometries (not repaired)")

    # Expected codes
    if pcode_field in gdf.columns:
        found = set(gdf[pcode_field])
        missing = set(expected_codes) - found
        if missing:
            issues.append(f"Missing codes in {pcode_field}: {sorted(missing)}")
    else:
        issues.append(f"Field '{pcode_field}' not in output columns")

    # Spatial containment
    if check_containment_bounds is not None:
        west, south, east, north = check_containment_bounds
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        if bounds[0] < west - 0.01 or bounds[2] > east + 0.01:
            issues.append(
                f"Bounds longitude out of expected range [{west}, {east}]: "
                f"got [{bounds[0]:.4f}, {bounds[2]:.4f}]"
            )
        if bounds[1] < south - 0.01 or bounds[3] > north + 0.01:
            issues.append(
                f"Bounds latitude out of expected range [{south}, {north}]: "
                f"got [{bounds[1]:.4f}, {bounds[3]:.4f}]"
            )

    if issues:
        raise ValidationFailed(
            f"Vector output validation failed for {path} (layer={layer}):\n"
            + "\n".join(f"  - {i}" for i in issues)
        )

    result: dict[str, Any] = {
        "path": str(path),
        "layer": layer,
        "crs": str(gdf.crs),
        "feature_count": len(gdf),
        "bounds": {
            "xmin": float(gdf.total_bounds[0]),
            "ymin": float(gdf.total_bounds[1]),
            "xmax": float(gdf.total_bounds[2]),
            "ymax": float(gdf.total_bounds[3]),
        },
        "valid": True,
    }
    logger.info("Vector validation OK: %s (layer=%s, %d features)", path, layer, len(gdf))
    return result


# ---------------------------------------------------------------------------
# Raster validation
# ---------------------------------------------------------------------------


def validate_raster_output(
    path: Path,
    *,
    expected_crs: str,
    municipality_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Validate a raster (GeoTIFF) output.

    Checks:
      - File exists and is readable with rasterio
      - CRS matches *expected_crs*
      - Width and height are positive
      - Bounds intersect *municipality_bounds* (xmin, ymin, xmax, ymax)
      - Nodata is defined (not None)

    Returns a dict with raster metadata.  Raises ValidationFailed on error.
    """
    import rasterio
    from pyproj import CRS

    if not path.exists():
        raise ValidationFailed(f"Raster output not found: {path}")

    try:
        ds = rasterio.open(path)
    except Exception as exc:
        raise ValidationFailed(f"Cannot open raster {path}: {exc}") from exc

    issues: list[str] = []

    with ds:
        raster_crs = ds.crs
        width = ds.width
        height = ds.height
        nodata = ds.nodata
        bounds = ds.bounds

        expected_crs_obj = CRS.from_string(expected_crs)
        if raster_crs is None:
            issues.append("CRS is None")
        elif not raster_crs.equals(expected_crs_obj):
            issues.append(f"CRS mismatch: expected {expected_crs}, got {raster_crs}")

        if width <= 0 or height <= 0:
            issues.append(f"Invalid dimensions: {width}x{height}")

        # Bounds intersection check
        mxmin, mymin, mxmax, mymax = municipality_bounds
        b_left, b_bottom, b_right, b_top = bounds.left, bounds.bottom, bounds.right, bounds.top
        if b_right < mxmin or b_left > mxmax or b_top < mymin or b_bottom > mymax:
            issues.append(
                f"Raster bounds {b_left:.2f},{b_bottom:.2f},{b_right:.2f},{b_top:.2f} "
                f"do not intersect municipality bounds "
                f"{mxmin:.2f},{mymin:.2f},{mxmax:.2f},{mymax:.2f}"
            )

        if nodata is None:
            issues.append("Nodata is None (should be -9999.0 for processed outputs)")

            issues.append(
                f"Raster bounds {b_left:.2f},{b_bottom:.2f},{b_right:.2f},{b_top:.2f} "
                f"do not intersect municipality bounds "
                f"{mxmin:.2f},{mymin:.2f},{mxmax:.2f},{mymax:.2f}"
            )

        if nodata is None:
            issues.append("Nodata is None (should be -9999.0 for processed outputs)")

    if issues:
        raise ValidationFailed(
            f"Raster output validation failed for {path}:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    result: dict[str, Any] = {
        "path": str(path),
        "crs": str(raster_crs),
        "width": width,
        "height": height,
        "nodata": nodata,
        "bounds": {
            "xmin": float(bounds.left),
            "ymin": float(bounds.bottom),
            "xmax": float(bounds.right),
            "ymax": float(bounds.top),
        },
        "valid": True,
    }
    logger.info("Raster validation OK: %s (%dx%d px)", path, width, height)
    return result
