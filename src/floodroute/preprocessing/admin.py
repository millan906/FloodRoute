"""Administrative boundary extraction (Stage 3).

Reads the Philippines HDX COD-AB shapefile archive and extracts the three
study municipality polygons (admin3) and their barangay polygons (admin4)
by adm3_pcode.  Outputs are GeoPackage files under data/processed/admin/.

Source: data/raw/psa_administrative_boundaries_antique.zip
Layers used: phl_admin3 (municipalities), phl_admin4 (barangays)
Key field: adm3_pcode (lowercase, 9-char PH-prefix PSGC code)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("floodroute.preprocessing.admin")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AdminExtractionError(RuntimeError):
    """Raised when administrative boundary extraction fails."""


class OutputExistsError(FileExistsError):
    """Raised when an output file already exists and force=False."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _vsizip_layer(zip_path: Path, layer_name: str) -> str:
    """Return the GDAL /vsizip/ path for a shapefile layer inside a ZIP."""
    return f"/vsizip/{zip_path}/{layer_name}.shp"


def _repair_geometries(gdf: Any) -> tuple[Any, int]:
    """Repair invalid geometries using shapely.make_valid().

    Returns the (possibly repaired) GeoDataFrame and a count of repairs.
    Every repair is logged at WARNING level — no silent fixes.
    """
    import shapely

    repairs = 0
    mask = ~gdf.geometry.is_valid
    if mask.any():
        n = int(mask.sum())
        for idx in gdf.index[mask]:
            code = gdf.at[idx, "adm3_pcode"] if "adm3_pcode" in gdf.columns else str(idx)
            logger.warning("Invalid geometry for %s at index %s — applying make_valid()", code, idx)
        gdf = gdf.copy()
        gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(lambda g: shapely.make_valid(g))
        repairs = n
    return gdf, repairs


def _sort_gdf(gdf: Any, sort_fields: list[str]) -> Any:
    """Sort a GeoDataFrame by *sort_fields* for deterministic output ordering."""
    existing = [f for f in sort_fields if f in gdf.columns]
    if existing:
        gdf = gdf.sort_values(existing).reset_index(drop=True)
    return gdf


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_municipalities(
    zip_path: Path,
    codes: list[str],
    output_path: Path,
    *,
    target_crs: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract municipality polygons (admin3) for *codes* from the ZIP archive.

    Parameters
    ----------
    zip_path:
        Absolute path to the admin boundary ZIP.
    codes:
        List of adm3_pcode values to extract (e.g. ["PH0600616", "PH0600613", "PH0600608"]).
    output_path:
        Destination GeoPackage file.
    target_crs:
        If given, reproject to this CRS before writing (e.g. "EPSG:32651").
        If None, write in the source CRS (EPSG:4326).
    force:
        Overwrite *output_path* if it exists.
    dry_run:
        Validate inputs and compute what *would* be written; do not write.

    Returns
    -------
    dict with keys: feature_count, crs, bounds, geometry_repairs, output_path
    """
    import geopandas as gpd

    if not zip_path.exists():
        raise AdminExtractionError(f"Admin archive not found: {zip_path}")

    if output_path.exists() and not force and not dry_run:
        raise OutputExistsError(
            f"Output already exists: {output_path}. Pass force=True or --force to overwrite."
        )

    layer_path = _vsizip_layer(zip_path, "phl_admin3")
    logger.info("Reading admin3 layer from %s", layer_path)

    try:
        gdf = gpd.read_file(layer_path)
    except Exception as exc:
        raise AdminExtractionError(f"Failed to read phl_admin3: {exc}") from exc

    if "adm3_pcode" not in gdf.columns:
        raise AdminExtractionError(
            f"Expected field 'adm3_pcode' not found. Available: {list(gdf.columns)}"
        )

    selected = gdf[gdf["adm3_pcode"].isin(codes)].copy()
    missing = set(codes) - set(selected["adm3_pcode"])
    if missing:
        raise AdminExtractionError(
            f"Municipality code(s) not found in phl_admin3: {sorted(missing)}"
        )

    selected, repairs = _repair_geometries(selected)
    selected = _sort_gdf(selected, ["adm3_pcode"])

    if target_crs:
        selected = selected.to_crs(target_crs)
        out_crs = target_crs
    else:
        out_crs = str(selected.crs)

    bounds_total = selected.total_bounds  # [minx, miny, maxx, maxy]
    result: dict[str, Any] = {
        "feature_count": len(selected),
        "crs": out_crs,
        "bounds": {
            "west": float(bounds_total[0]),
            "south": float(bounds_total[1]),
            "east": float(bounds_total[2]),
            "north": float(bounds_total[3]),
        },
        "geometry_repairs": repairs,
        "output_path": output_path,
    }

    if dry_run:
        logger.info("DRY RUN: would write %d municipalities to %s", len(selected), output_path)
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_file(output_path, driver="GPKG", layer="municipalities")
    logger.info("Wrote %d municipality polygons → %s", len(selected), output_path)
    return result


def extract_barangays(
    zip_path: Path,
    codes: list[str],
    output_path: Path,
    *,
    target_crs: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract barangay polygons (admin4) for municipalities identified by *codes*.

    Parameters
    ----------
    zip_path:
        Absolute path to the admin boundary ZIP.
    codes:
        List of adm3_pcode values whose barangays to include.
    output_path:
        Destination GeoPackage file.
    target_crs:
        If given, reproject to this CRS before writing.
    force:
        Overwrite *output_path* if it exists.
    dry_run:
        Validate inputs; do not write.

    Returns
    -------
    dict with keys: feature_count, crs, bounds, geometry_repairs, output_path
    """
    import geopandas as gpd

    if not zip_path.exists():
        raise AdminExtractionError(f"Admin archive not found: {zip_path}")

    if output_path.exists() and not force and not dry_run:
        raise OutputExistsError(
            f"Output already exists: {output_path}. Pass force=True or --force to overwrite."
        )

    layer_path = _vsizip_layer(zip_path, "phl_admin4")
    logger.info("Reading admin4 layer from %s", layer_path)

    try:
        gdf = gpd.read_file(layer_path)
    except Exception as exc:
        raise AdminExtractionError(f"Failed to read phl_admin4: {exc}") from exc

    if "adm3_pcode" not in gdf.columns:
        raise AdminExtractionError(
            f"Expected field 'adm3_pcode' not found. Available: {list(gdf.columns)}"
        )

    selected = gdf[gdf["adm3_pcode"].isin(codes)].copy()
    found_codes = set(selected["adm3_pcode"])
    missing = set(codes) - found_codes
    if missing:
        raise AdminExtractionError(
            f"No barangays found for municipality code(s): {sorted(missing)}"
        )

    selected, repairs = _repair_geometries(selected)
    selected = _sort_gdf(selected, ["adm3_pcode", "adm4_pcode"])

    if target_crs:
        selected = selected.to_crs(target_crs)
        out_crs = target_crs
    else:
        out_crs = str(selected.crs)

    bounds_total = selected.total_bounds
    result: dict[str, Any] = {
        "feature_count": len(selected),
        "crs": out_crs,
        "bounds": {
            "west": float(bounds_total[0]),
            "south": float(bounds_total[1]),
            "east": float(bounds_total[2]),
            "north": float(bounds_total[3]),
        },
        "geometry_repairs": repairs,
        "output_path": output_path,
    }

    if dry_run:
        logger.info("DRY RUN: would write %d barangays to %s", len(selected), output_path)
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_file(output_path, driver="GPKG", layer="barangays")
    logger.info("Wrote %d barangay polygons → %s", len(selected), output_path)
    return result
