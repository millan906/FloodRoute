"""DEM preprocessing (Stage 3).

Builds a two-tile mosaic VRT from the Copernicus GLO-30 tiles, clips
the DEM to each municipality boundary, and reprojects to EPSG:32651
(UTM Zone 51N).

Inputs:
  data/raw/dem_antique/cop_dem30_N10_E121.tif
  data/raw/dem_antique/cop_dem30_N10_E122.tif

Outputs (per municipality code):
  data/processed/dem/mosaic.vrt          — reproducible mosaic descriptor
  data/processed/dem/{code}_dem_utm51n.tif  — clipped + reprojected DEM

Resampling method: bilinear (appropriate for continuous elevation data).
Nodata: -9999.0 (the raw COG tiles report nodata=None; we assign -9999 on write).

Stage 3 scope: elevation values only.  No slope, aspect, flood depth, or
hazard calculation is performed here.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

import rasterio
import rasterio.crs
import rasterio.mask
import rasterio.transform
import rasterio.warp
from rasterio.enums import Resampling

logger = logging.getLogger("floodroute.preprocessing.dem")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DemProcessingError(RuntimeError):
    """Raised when DEM preprocessing fails."""


class OutputExistsError(FileExistsError):
    """Raised when an output file already exists and force=False."""


# ---------------------------------------------------------------------------
# Resampling name → rasterio enum
# ---------------------------------------------------------------------------

_RESAMPLING_MAP: dict[str, Resampling] = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
}


# ---------------------------------------------------------------------------
# VRT builder
# ---------------------------------------------------------------------------


def build_dem_vrt(tile_paths: list[Path], vrt_path: Path) -> Path:
    """Write a minimal GDAL VRT that mosaics two adjacent DEM tiles.

    The VRT uses relative paths so it remains valid when the project
    directory is moved (as long as the relative positions of vrt_path
    and tile_paths are preserved).

    Parameters
    ----------
    tile_paths:
        Ordered list of GeoTIFF tile paths; tiles must be in the same CRS,
        the same pixel size, and must be adjacent east-west.
    vrt_path:
        Destination .vrt file.

    Returns
    -------
    vrt_path
    """
    if len(tile_paths) != 2:
        raise DemProcessingError("build_dem_vrt requires exactly two tile paths.")

    # Inspect both tiles
    metas: list[dict[str, Any]] = []
    for tp in tile_paths:
        if not tp.exists():
            raise DemProcessingError(f"DEM tile not found: {tp}")
        with rasterio.open(tp) as ds:
            metas.append(
                {
                    "path": tp,
                    "crs": ds.crs,
                    "transform": ds.transform,
                    "width": ds.width,
                    "height": ds.height,
                    "dtype": ds.dtypes[0],
                    "nodata": ds.nodata,
                }
            )

    # Verify both tiles have the same CRS and row dimensions
    if metas[0]["crs"] != metas[1]["crs"]:
        raise DemProcessingError(
            f"DEM tiles have different CRS: {metas[0]['crs']} vs {metas[1]['crs']}"
        )
    if metas[0]["height"] != metas[1]["height"]:
        raise DemProcessingError("DEM tiles have different row counts; cannot build east-west VRT.")

    # Combined mosaic dimensions
    total_width = metas[0]["width"] + metas[1]["width"]
    total_height = metas[0]["height"]

    # Left tile provides the origin
    left_transform = metas[0]["transform"]
    vrt_crs = metas[0]["crs"]
    dtype = metas[0]["dtype"]

    # Relative paths from VRT file to tile files
    rel_paths = [_relative_path(vrt_path, tp) for tp in tile_paths]

    vrt_xml = textwrap.dedent(f"""\
        <VRTDataset rasterXSize="{total_width}" rasterYSize="{total_height}">
          <SRS>{vrt_crs.to_wkt()}</SRS>
          <GeoTransform>{left_transform.c}, {left_transform.a}, \
{left_transform.b}, {left_transform.f}, {left_transform.d}, {left_transform.e}</GeoTransform>
          <VRTRasterBand dataType="{_gdal_dtype(dtype)}" band="1">
            <ColorInterp>Gray</ColorInterp>
            <SimpleSource>
              <SourceFilename relativeToVRT="1">{rel_paths[0]}</SourceFilename>
              <SourceBand>1</SourceBand>
              <SourceProperties RasterXSize="{metas[0]["width"]}" \
RasterYSize="{metas[0]["height"]}" DataType="{_gdal_dtype(dtype)}" \
BlockXSize="512" BlockYSize="512" />
              <SrcRect xOff="0" yOff="0" xSize="{metas[0]["width"]}" \
ySize="{metas[0]["height"]}" />
              <DstRect xOff="0" yOff="0" xSize="{metas[0]["width"]}" \
ySize="{metas[0]["height"]}" />
            </SimpleSource>
            <SimpleSource>
              <SourceFilename relativeToVRT="1">{rel_paths[1]}</SourceFilename>
              <SourceBand>1</SourceBand>
              <SourceProperties RasterXSize="{metas[1]["width"]}" \
RasterYSize="{metas[1]["height"]}" DataType="{_gdal_dtype(dtype)}" \
BlockXSize="512" BlockYSize="512" />
              <SrcRect xOff="0" yOff="0" xSize="{metas[1]["width"]}" \
ySize="{metas[1]["height"]}" />
              <DstRect xOff="{metas[0]["width"]}" yOff="0" \
xSize="{metas[1]["width"]}" ySize="{metas[1]["height"]}" />
            </SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
    """)

    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    vrt_path.write_text(vrt_xml, encoding="utf-8")
    logger.info("Wrote DEM mosaic VRT → %s", vrt_path)
    return vrt_path


def _relative_path(from_file: Path, to_file: Path) -> str:
    """Return a relative path string from *from_file*'s directory to *to_file*."""
    try:
        return str(to_file.resolve().relative_to(from_file.resolve().parent))
    except ValueError:
        # Fallback: use ../../... style if not under the same root
        import os

        return os.path.relpath(to_file.resolve(), from_file.resolve().parent)


def _gdal_dtype(numpy_dtype: str) -> str:
    """Map a NumPy dtype string to the GDAL VRT DataType name."""
    mapping = {
        "float32": "Float32",
        "float64": "Float64",
        "int16": "Int16",
        "int32": "Int32",
        "uint8": "Byte",
        "uint16": "UInt16",
    }
    return mapping.get(str(numpy_dtype), "Float32")


# ---------------------------------------------------------------------------
# Clip + reproject
# ---------------------------------------------------------------------------


def clip_and_reproject_dem(
    src_path: Path,
    municipality_geom: Any,
    out_path: Path,
    *,
    dst_crs: str,
    resampling: str = "bilinear",
    nodata: float = -9999.0,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clip *src_path* to *municipality_geom* and reproject to *dst_crs*.

    The clip polygon must be in the same CRS as *src_path* (EPSG:4326).
    The output is a single-band GeoTIFF in *dst_crs*.

    Parameters
    ----------
    src_path:
        Source raster (VRT or GeoTIFF), EPSG:4326.
    municipality_geom:
        Shapely geometry (polygon) in EPSG:4326 used as clip mask.
    out_path:
        Destination GeoTIFF (EPSG:32651 or other *dst_crs*).
    dst_crs:
        Target projected CRS string, e.g. "EPSG:32651".
    resampling:
        Resampling method name (bilinear recommended for continuous data).
    nodata:
        Nodata sentinel value written to the output.
    force:
        Overwrite *out_path* if it exists.
    dry_run:
        Validate inputs; do not write.

    Returns
    -------
    dict with keys: crs, bounds (xmin/ymin/xmax/ymax), width, height,
                    nodata, dtype, resolution_m, output_path
    """
    if not src_path.exists():
        raise DemProcessingError(f"Source DEM not found: {src_path}")

    if out_path.exists() and not force and not dry_run:
        raise OutputExistsError(
            f"Output already exists: {out_path}. Pass force=True or --force to overwrite."
        )

    resamp = _RESAMPLING_MAP.get(resampling)
    if resamp is None:
        raise DemProcessingError(
            f"Unknown resampling method '{resampling}'. Choose from: {list(_RESAMPLING_MAP)}"
        )

    with rasterio.open(src_path) as src:
        # Step 1: clip to municipality boundary (still in source CRS)
        clipped_data, clip_transform = rasterio.mask.mask(
            src, [municipality_geom], crop=True, filled=True, nodata=nodata
        )
        clip_crs = src.crs
        clip_profile = src.profile.copy()
        clip_profile.update(
            {
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clip_transform,
                "nodata": nodata,
                "count": 1,
                "dtype": "float32",
            }
        )

    # Step 2: reproject clipped data to dst_crs
    dst_crs_obj = rasterio.crs.CRS.from_string(dst_crs)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        clip_crs,
        dst_crs_obj,
        clip_profile["width"],
        clip_profile["height"],
        *rasterio.transform.array_bounds(
            clip_profile["height"], clip_profile["width"], clip_transform
        ),
    )

    out_bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
    # resolution in metres (approximate, from x pixel size)
    res_m = abs(dst_transform.a)

    result: dict[str, Any] = {
        "crs": dst_crs,
        "bounds": {
            "xmin": float(out_bounds[0]),
            "ymin": float(out_bounds[1]),
            "xmax": float(out_bounds[2]),
            "ymax": float(out_bounds[3]),
        },
        "width": dst_width,
        "height": dst_height,
        "nodata": nodata,
        "dtype": "float32",
        "resolution_m": round(res_m, 2),
        "output_path": out_path,
    }

    if dry_run:
        logger.info(
            "DRY RUN: would write DEM %dx%d px @ %.1f m → %s",
            dst_width,
            dst_height,
            res_m,
            out_path,
        )
        return result

    out_profile = clip_profile.copy()
    out_profile.update(
        {
            "crs": dst_crs_obj,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": nodata,
            "dtype": "float32",
            "driver": "GTiff",
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        rasterio.warp.reproject(
            source=clipped_data,
            destination=rasterio.band(dst, 1),
            src_transform=clip_transform,
            src_crs=clip_crs,
            src_nodata=nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs_obj,
            resampling=resamp,
        )

    logger.info(
        "Wrote DEM %dx%d px → %s",
        dst_width,
        dst_height,
        out_path,
    )
    return result


# ---------------------------------------------------------------------------
# Raster metadata reader (for validation)
# ---------------------------------------------------------------------------


def read_raster_metadata(path: Path) -> dict[str, Any]:
    """Return a metadata dict for a raster file (for validation/reporting)."""
    if not path.exists():
        raise DemProcessingError(f"Raster not found: {path}")
    with rasterio.open(path) as ds:
        bounds = ds.bounds
        return {
            "path": str(path),
            "crs": str(ds.crs),
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": str(ds.dtypes[0]),
            "nodata": ds.nodata,
            "bounds": {
                "xmin": float(bounds.left),
                "ymin": float(bounds.bottom),
                "xmax": float(bounds.right),
                "ymax": float(bounds.top),
            },
            "transform": list(ds.transform)[:6],
        }
