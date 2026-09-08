"""FloodRoute command-line interface.

Stage 0: validate-config is fully operational.
Stage 1: list-datasets, validate-manifests, acquire-dataset, verify-dataset.
Stage 3: preprocess-geospatial — converts verified raw inputs to analysis-ready layers.
Later commands exit with a clear error until their data dependencies exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from floodroute.logging_config import configure_logging, get_logger

app = typer.Typer(
    name="floodroute",
    help="Flood-resilient evacuation routing and shelter allocation (MSCS thesis).",
    no_args_is_help=True,
)

logger = get_logger("cli")

# Default paths resolved relative to the installed package root.
# resolve() is required when the package is imported through a symlink
# (e.g., .venv/site-packages/floodroute -> src/floodroute) so that
# __file__ is de-symlinked before computing the project root.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIGS = _PKG_ROOT / "configs"
_DEFAULT_DATA = _PKG_ROOT / "data"
_DEFAULT_MANIFESTS = _DEFAULT_DATA / "manifests"


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------


@app.command("validate-config")
def validate_config(
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", "-c", help="Path to the configs/ directory."),
    ] = _DEFAULT_CONFIGS,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate all YAML configuration files against their Pydantic schemas."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("Validating configs in %s", configs_dir)

    if not configs_dir.is_dir():
        typer.echo(f"ERROR: configs directory not found: {configs_dir}", err=True)
        raise typer.Exit(code=1)

    from pydantic import ValidationError

    from floodroute.config import validate_all_configs

    try:
        cfg = validate_all_configs(configs_dir)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: missing config file — {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValidationError as exc:
        typer.echo(f"ERROR: config validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("All configuration files are valid.")
    n_scenarios = len(cfg.scenarios.scenarios)
    n_experiments = len(cfg.experiments.experiments)
    n_shelters = len(cfg.shelters.shelters)
    typer.echo(f"  scenarios={n_scenarios}  experiments={n_experiments}  shelters={n_shelters}")
    logger.info("Config validation complete.")


# ---------------------------------------------------------------------------
# inspect-data  (Stage 1: manifest-aware)
# ---------------------------------------------------------------------------


@app.command("inspect-data")
def inspect_data(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Path to the data/ directory."),
    ] = _DEFAULT_DATA,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Report on data availability and manifest status."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("Inspecting data directory: %s", data_dir)

    if not data_dir.is_dir():
        typer.echo(f"ERROR: data directory not found: {data_dir}", err=True)
        raise typer.Exit(code=1)

    for sub in ("raw", "interim", "processed"):
        sub_path = data_dir / sub
        if not sub_path.is_dir():
            typer.echo(f"  {sub}/  — directory missing")
        else:
            files = [f for f in sub_path.iterdir() if f.name != ".gitkeep"]
            typer.echo(f"  {sub}/  — {len(files)} file(s)")

    # Manifest-aware readiness (never crashes on missing manifests)
    manifests_dir = data_dir / "manifests"
    try:
        from floodroute.manifest import load_all_manifests
        from floodroute.readiness import build_readiness_report, format_report_text

        manifests = load_all_manifests(manifests_dir)
        if manifests:
            entries = build_readiness_report(manifests, data_dir)
            typer.echo(f"\n  manifests/  — {len(manifests)} manifest(s)\n")
            typer.echo(format_report_text(entries))
        else:
            typer.echo("  manifests/  — 0 manifest(s)")
            typer.echo("\nNo manifests loaded. Add dataset manifests to data/manifests/.")
    except Exception as exc:
        logger.warning("Could not load manifests: %s", exc)
        typer.echo(f"  manifests/  — could not load ({exc})")


# ---------------------------------------------------------------------------
# list-datasets
# ---------------------------------------------------------------------------


@app.command("list-datasets")
def list_datasets(
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """List all registered datasets with their acquisition and validation status."""
    configure_logging(log_level)  # type: ignore[arg-type]

    from pydantic import ValidationError

    from floodroute.manifest import load_all_manifests
    from floodroute.readiness import build_readiness_report, format_report_json, format_report_text

    try:
        manifests = load_all_manifests(manifests_dir)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (ValidationError, ValueError) as exc:
        typer.echo(f"ERROR: manifest error — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not manifests:
        typer.echo("No manifest files found in: " + str(manifests_dir))
        return

    data_dir = manifests_dir.parent
    entries = build_readiness_report(manifests, data_dir)

    if output_format == "json":
        typer.echo(format_report_json(entries))
    else:
        typer.echo(format_report_text(entries))


# ---------------------------------------------------------------------------
# validate-manifests
# ---------------------------------------------------------------------------


@app.command("validate-manifests")
def validate_manifests(
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate all manifest YAML files against the DatasetManifest schema."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("Validating manifests in %s", manifests_dir)

    if not manifests_dir.is_dir():
        typer.echo(f"ERROR: manifests directory not found: {manifests_dir}", err=True)
        raise typer.Exit(code=1)

    from pydantic import ValidationError

    from floodroute.manifest import load_manifest

    yaml_files = sorted(manifests_dir.glob("*.yaml"))
    if not yaml_files:
        typer.echo("No *.yaml files found in: " + str(manifests_dir))
        return

    errors: list[str] = []
    seen_ids: dict[str, str] = {}

    for path in yaml_files:
        try:
            m = load_manifest(path)
            if m.dataset_id in seen_ids:
                errors.append(
                    f"  DUPLICATE  {path.name}: dataset_id '{m.dataset_id}' "
                    f"already used in '{seen_ids[m.dataset_id]}'"
                )
            else:
                seen_ids[m.dataset_id] = path.name
                typer.echo(f"  OK         {path.name}  ({m.dataset_id})")
        except (ValidationError, Exception) as exc:
            errors.append(f"  FAIL       {path.name}: {exc}")

    if errors:
        typer.echo("\nValidation errors:", err=True)
        for e in errors:
            typer.echo(e, err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"\nAll {len(seen_ids)} manifest(s) valid.")


# ---------------------------------------------------------------------------
# acquire-dataset
# ---------------------------------------------------------------------------


@app.command("acquire-dataset")
def acquire_dataset(
    dataset_id: Annotated[
        str,
        typer.Argument(help="Dataset ID as declared in a manifest YAML."),
    ],
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Root data directory."),
    ] = _DEFAULT_DATA,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing local file."),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="HTTP request timeout in seconds."),
    ] = 30,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Download a single dataset by its manifest ID.

    Manual-acquisition datasets print instructions instead of downloading.
    There is no bulk acquire-all command; every download requires an explicit ID.
    """
    configure_logging(log_level)  # type: ignore[arg-type]

    from pydantic import ValidationError

    from floodroute.acquisition import (
        ChecksumMismatch,
        FileAlreadyExists,
        ManualAcquisitionRequired,
        download_dataset,
    )
    from floodroute.manifest import load_all_manifests

    try:
        manifests = load_all_manifests(manifests_dir)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"ERROR: manifest error — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if dataset_id not in manifests:
        typer.echo(f"ERROR: dataset_id '{dataset_id}' not found.", err=True)
        typer.echo(f"Known IDs: {', '.join(sorted(manifests))}", err=True)
        raise typer.Exit(code=1)

    manifest = manifests[dataset_id]

    try:
        dest = download_dataset(manifest, data_dir, force=force, timeout=timeout)
        typer.echo(f"Acquired: {dest}")
    except ManualAcquisitionRequired as exc:
        typer.echo("\n" + exc.instructions)
        raise typer.Exit(code=0) from None
    except FileAlreadyExists as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        typer.echo("Use --force to overwrite.", err=True)
        raise typer.Exit(code=1) from exc
    except ChecksumMismatch as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"ERROR: download failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# verify-dataset
# ---------------------------------------------------------------------------


@app.command("verify-dataset")
def verify_dataset(
    dataset_id: Annotated[
        str,
        typer.Argument(help="Dataset ID to verify."),
    ],
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to the data/manifests/ directory."),
    ] = _DEFAULT_MANIFESTS,
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Root data directory."),
    ] = _DEFAULT_DATA,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Verify the SHA-256 checksum of a locally acquired dataset."""
    configure_logging(log_level)  # type: ignore[arg-type]

    from pydantic import ValidationError

    from floodroute.acquisition import compute_sha256, resolve_local_path
    from floodroute.manifest import load_all_manifests

    try:
        manifests = load_all_manifests(manifests_dir)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"ERROR: manifest error — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if dataset_id not in manifests:
        typer.echo(f"ERROR: dataset_id '{dataset_id}' not found.", err=True)
        raise typer.Exit(code=1)

    manifest = manifests[dataset_id]
    path = resolve_local_path(manifest, data_dir)

    if path is None:
        typer.echo("ERROR: No local_path defined in manifest; cannot verify.", err=True)
        raise typer.Exit(code=1)

    if not path.exists():
        typer.echo(f"ERROR: Local file not found: {path}", err=True)
        raise typer.Exit(code=1)

    if manifest.sha256 is None:
        actual = compute_sha256(path)
        typer.echo("WARNING: No expected SHA-256 in manifest — recording actual value only.")
        typer.echo(f"  Actual SHA-256 : {actual}")
        typer.echo("  Record this value in the manifest once the file is trusted.")
        raise typer.Exit(code=0)

    actual = compute_sha256(path)
    if actual == manifest.sha256:
        typer.echo(f"OK  {dataset_id}  sha256={actual}")
    else:
        typer.echo(
            f"FAIL  {dataset_id}\n  expected : {manifest.sha256}\n  actual   : {actual}",
            err=True,
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# preprocess-geospatial  (Stage 3)
# ---------------------------------------------------------------------------


@app.command("preprocess-geospatial")
def preprocess_geospatial(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Root data/ directory."),
    ] = _DEFAULT_DATA,
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", "-m", help="Path to data/manifests/."),
    ] = _DEFAULT_MANIFESTS,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing processed outputs."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Validate inputs and report what would be written; do not write."
        ),
    ] = False,
    skip_osm: Annotated[
        bool,
        typer.Option(
            "--skip-osm", help="Skip OSM extraction (useful when PBF backend unavailable)."
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Preprocess verified raw geospatial inputs into analysis-ready municipal layers.

    \b
    Stage 3 operations:
      A. Administrative boundaries — municipalities (admin3) and barangays (admin4)
         extracted from the Philippines COD-AB archive by adm3_pcode.
         Outputs: data/processed/admin/  (EPSG:4326 and EPSG:32651)

      B. DEM — two Copernicus GLO-30 tiles mosaicked, clipped per municipality,
         reprojected to EPSG:32651 (UTM Zone 51N) with bilinear resampling.
         Outputs: data/processed/dem/

      C. OSM — roads and waterways extracted from the Philippines PBF.
         Outputs: data/processed/osm/
         NOTE: reports a blocker if no OSM-capable backend is available.

    Use --dry-run to validate inputs without writing any outputs.
    Use --force to overwrite existing outputs.

    \b
    This command does NOT perform routing, graph construction, hazard
    intersection, shelter assignment, or any research-result generation.
    """
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("preprocess-geospatial starting (dry_run=%s, force=%s)", dry_run, force)

    from floodroute.preprocessing.config import (
        ADMIN_DATASET_ID,
        DEM_DATASET_ID,
        MUNICIPALITY_CODES,
        MUNICIPALITY_NAMES,
        OSM_DATASET_ID,
        PreprocessingConfig,
    )
    from floodroute.preprocessing.dem import (
        DemProcessingError,
        build_dem_vrt,
        clip_and_reproject_dem,
    )
    from floodroute.preprocessing.dem import (
        OutputExistsError as DemOutputExistsError,
    )
    from floodroute.preprocessing.osm import OsmBackendUnavailable, check_osm_backend
    from floodroute.preprocessing.prep_manifest import (
        build_preprocessing_manifest,
        source_checksum_from_manifest_dir,
        write_preprocessing_manifest,
    )

    cfg_base = PreprocessingConfig()
    cfg = cfg_base.resolve(data_dir)

    mode = "DRY RUN" if dry_run else "LIVE"
    typer.echo(f"[{mode}] preprocess-geospatial — data_dir={data_dir}")

    # ------------------------------------------------------------------
    # Preflight: verify required raw inputs exist
    # ------------------------------------------------------------------
    missing_inputs: list[str] = []
    if not cfg.admin_archive.exists():
        missing_inputs.append(f"Admin archive: {cfg.admin_archive}")
    for tile in cfg.dem_tiles:
        if not tile.exists():
            missing_inputs.append(f"DEM tile: {tile}")
    if not skip_osm and not cfg.osm_pbf.exists():
        missing_inputs.append(f"OSM PBF: {cfg.osm_pbf}")

    if missing_inputs:
        typer.echo("ERROR: Required raw inputs are missing:", err=True)
        for m in missing_inputs:
            typer.echo(f"  - {m}", err=True)
        raise typer.Exit(code=1)

    # Source checksums from acquisition manifests (for preprocessing manifests)
    admin_sha = source_checksum_from_manifest_dir(ADMIN_DATASET_ID, manifests_dir)
    dem_sha = source_checksum_from_manifest_dir(DEM_DATASET_ID, manifests_dir)
    _ = source_checksum_from_manifest_dir(
        OSM_DATASET_ID, manifests_dir
    )  # reserved for OSM manifest

    errors: list[str] = []
    outputs_written: list[str] = []

    # ------------------------------------------------------------------
    # A. Administrative boundaries
    # ------------------------------------------------------------------
    typer.echo("\n[A] Administrative boundaries")

    from floodroute.preprocessing.admin import (
        AdminExtractionError,
        extract_barangays,
        extract_municipalities,
    )
    from floodroute.preprocessing.admin import (
        OutputExistsError as AdminOutputExistsError,
    )
    from floodroute.preprocessing.config import (
        STUDY_AREA_BOUNDS_WGS84,
    )
    from floodroute.preprocessing.validation import ValidationFailed, validate_vector_output

    admin_outputs = [
        ("municipalities_wgs84", "municipalities_wgs84.gpkg", None, "municipalities"),
        ("municipalities_utm51n", "municipalities_utm51n.gpkg", cfg.target_crs, "municipalities"),
        ("barangays_wgs84", "barangays_wgs84.gpkg", None, "barangays"),
        ("barangays_utm51n", "barangays_utm51n.gpkg", cfg.target_crs, "barangays"),
    ]

    for out_id, filename, target_crs, layer_type in admin_outputs:
        out_path = cfg.output_admin_dir / filename
        is_muni = layer_type == "municipalities"
        crs_label = target_crs or "EPSG:4326"
        is_geographic = target_crs is None  # EPSG:4326 if no reprojection
        try:
            if is_muni:
                result = extract_municipalities(
                    cfg.admin_archive,
                    MUNICIPALITY_CODES,
                    out_path,
                    target_crs=target_crs,
                    force=force,
                    dry_run=dry_run,
                )
            else:
                result = extract_barangays(
                    cfg.admin_archive,
                    MUNICIPALITY_CODES,
                    out_path,
                    target_crs=target_crs,
                    force=force,
                    dry_run=dry_run,
                )
            typer.echo(
                f"  {'[DRY RUN] ' if dry_run else ''}"
                f"{out_id}: {result['feature_count']} features, {crs_label}"
            )
            if result.get("geometry_repairs"):
                typer.echo(f"  WARNING: {result['geometry_repairs']} geometry repair(s) applied")

            val_status: dict[str, Any] = {}
            if not dry_run:
                # Post-write validation
                try:
                    val_result = validate_vector_output(
                        out_path,
                        layer=layer_type,
                        expected_crs=crs_label,
                        expected_codes=MUNICIPALITY_CODES,
                        pcode_field="adm3_pcode" if is_muni else "adm3_pcode",
                        min_features=3 if is_muni else 151,
                        # Containment check only for WGS84 outputs (bounds in degrees)
                        check_containment_bounds=STUDY_AREA_BOUNDS_WGS84 if is_geographic else None,
                        check_duplicate_ids=is_muni,  # municipality pcodes must be unique
                    )
                    typer.echo(
                        f"    VALID: {val_result['feature_count']} features, all geometries OK"
                    )
                    val_status = {"validation": "passed", "validation_detail": val_result}
                except ValidationFailed as vexc:
                    typer.echo(f"    VALIDATION FAIL: {vexc}", err=True)
                    errors.append(f"Validation failed for {out_id}: {vexc}")
                    val_status = {"validation": "failed", "validation_error": str(vexc)}

                man = build_preprocessing_manifest(
                    output_id=out_id,
                    operation="admin_boundary_extraction",
                    parameters={
                        "municipality_codes": MUNICIPALITY_CODES,
                        "layer": layer_type,
                        "source_layer": "phl_admin3" if is_muni else "phl_admin4",
                        "filter_field": "adm3_pcode",
                        "target_crs": crs_label,
                    },
                    source_dataset_ids=[ADMIN_DATASET_ID],
                    source_checksums={ADMIN_DATASET_ID: admin_sha},
                    output_path=out_path,
                    output_crs=crs_label,
                    output_bounds=result["bounds"],
                    feature_count=result["feature_count"],
                    geometry_repairs=result["geometry_repairs"],
                    extra=val_status if val_status else None,
                )
                write_preprocessing_manifest(man, cfg.output_manifests_dir / f"{out_id}.json")
                outputs_written.append(str(out_path))

        except (AdminOutputExistsError, DemOutputExistsError) as exc:
            typer.echo(f"  SKIP {out_id}: {exc}", err=True)
            errors.append(f"Output exists (use --force): {out_id}")
        except (AdminExtractionError, Exception) as exc:
            typer.echo(f"  FAIL {out_id}: {exc}", err=True)
            errors.append(f"Admin extraction failed ({out_id}): {exc}")

    # ------------------------------------------------------------------
    # B. DEM preprocessing
    # ------------------------------------------------------------------
    typer.echo("\n[B] DEM preprocessing")

    # Build VRT mosaic
    vrt_path = cfg.output_dem_dir / "mosaic.vrt"
    if not dry_run:
        try:
            if vrt_path.exists() and not force:
                typer.echo("  SKIP mosaic.vrt (exists; use --force to regenerate)")
            else:
                build_dem_vrt(cfg.dem_tiles, vrt_path)
                typer.echo(f"  mosaic.vrt written ({len(cfg.dem_tiles)} tiles)")
        except Exception as exc:
            typer.echo(f"  FAIL mosaic.vrt: {exc}", err=True)
            errors.append(f"DEM VRT build failed: {exc}")
    else:
        typer.echo(f"  [DRY RUN] would build mosaic.vrt from {len(cfg.dem_tiles)} tiles")

    # Load municipalities for clip boundaries
    try:
        import geopandas as gpd

        muni_wgs84_path = cfg.output_admin_dir / "municipalities_wgs84.gpkg"
        if muni_wgs84_path.exists() and not dry_run:
            muni_gdf = gpd.read_file(muni_wgs84_path, layer="municipalities")
        elif dry_run and cfg.admin_archive.exists():
            # In dry-run, read directly from source
            src_layer = f"/vsizip/{cfg.admin_archive}/phl_admin3.shp"
            muni_gdf = gpd.read_file(src_layer)
            muni_gdf = muni_gdf[muni_gdf["adm3_pcode"].isin(MUNICIPALITY_CODES)]
        else:
            muni_gdf = None
    except Exception as exc:
        typer.echo(f"  WARNING: cannot load municipality boundaries for DEM clip: {exc}")
        muni_gdf = None

    from floodroute.preprocessing.validation import validate_raster_output

    dem_src = (
        vrt_path
        if (vrt_path.exists() and not dry_run)
        else (cfg.dem_tiles[0] if cfg.dem_tiles else None)
    )
    if dry_run and cfg.dem_tiles:
        dem_src = cfg.dem_tiles[0]  # Use first tile as proxy in dry-run

    # Load municipality bounds in UTM51N for raster validation (from processed admin if available)
    muni_utm_bounds: dict[str, tuple[float, float, float, float]] = {}
    muni_utm51n_path = cfg.output_admin_dir / "municipalities_utm51n.gpkg"
    if muni_utm51n_path.exists() and not dry_run:
        try:
            import geopandas as _gpd2

            _mu = _gpd2.read_file(muni_utm51n_path, layer="municipalities")
            for _code in MUNICIPALITY_CODES:
                _row = _mu[_mu["adm3_pcode"] == _code]
                if len(_row) > 0:
                    _b = _row.total_bounds
                    muni_utm_bounds[_code] = (
                        float(_b[0]),
                        float(_b[1]),
                        float(_b[2]),
                        float(_b[3]),
                    )
        except Exception as _exc:
            logger.warning("Could not load UTM51N municipality bounds for DEM validation: %s", _exc)

    for code in sorted(MUNICIPALITY_CODES):
        name = MUNICIPALITY_NAMES.get(code, code)
        out_path = cfg.output_dem_dir / f"{code}_dem_utm51n.tif"

        if muni_gdf is None or dem_src is None:
            typer.echo(f"  SKIP DEM {code}: dependencies not ready")
            continue

        rows = muni_gdf[muni_gdf["adm3_pcode"] == code] if muni_gdf is not None else None
        if rows is None or len(rows) == 0:
            typer.echo(f"  SKIP DEM {code}: municipality not in boundary layer")
            continue

        geom = rows.geometry.iloc[0]
        try:
            result = clip_and_reproject_dem(
                dem_src,
                geom,
                out_path,
                dst_crs=cfg.target_crs,
                resampling=cfg.dem_resampling,
                nodata=cfg.dem_output_nodata,
                force=force,
                dry_run=dry_run,
            )
            typer.echo(
                f"  {'[DRY RUN] ' if dry_run else ''}"
                f"DEM {code} ({name}): {result['width']}×{result['height']} px, "
                f"{result['resolution_m']:.1f} m, {cfg.target_crs}"
            )

            val_status_dem: dict[str, Any] = {}
            if not dry_run:
                # Post-write raster validation using UTM51N municipality bounds
                muni_bounds_utm = muni_utm_bounds.get(
                    code, tuple(result["bounds"][k] for k in ("xmin", "ymin", "xmax", "ymax"))
                )
                try:
                    val_result = validate_raster_output(
                        out_path,
                        expected_crs=cfg.target_crs,
                        municipality_bounds=muni_bounds_utm,
                    )
                    typer.echo(
                        f"    VALID: {val_result['width']}×{val_result['height']} px, "
                        f"nodata={val_result['nodata']}"
                    )
                    val_status_dem = {"validation": "passed"}
                except ValidationFailed as vexc:
                    typer.echo(f"    VALIDATION FAIL: {vexc}", err=True)
                    errors.append(f"Validation failed for dem_{code}: {vexc}")
                    val_status_dem = {"validation": "failed", "validation_error": str(vexc)}

                man = build_preprocessing_manifest(
                    output_id=f"dem_{code}_utm51n",
                    operation="dem_clip_reproject",
                    parameters={
                        "municipality_code": code,
                        "municipality_name": name,
                        "source_tiles": [t.name for t in cfg.dem_tiles],
                        "resampling": cfg.dem_resampling,
                        "dst_crs": cfg.target_crs,
                        "nodata": cfg.dem_output_nodata,
                    },
                    source_dataset_ids=[DEM_DATASET_ID],
                    source_checksums={DEM_DATASET_ID: dem_sha},
                    output_path=out_path,
                    output_crs=cfg.target_crs,
                    output_bounds=result["bounds"],
                    raster_width=result["width"],
                    raster_height=result["height"],
                    raster_nodata=result["nodata"],
                    raster_dtype=result["dtype"],
                    extra=val_status_dem if val_status_dem else None,
                )
                write_preprocessing_manifest(
                    man, cfg.output_manifests_dir / f"dem_{code}_utm51n.json"
                )
                outputs_written.append(str(out_path))

        except DemOutputExistsError as exc:
            typer.echo(f"  SKIP DEM {code}: {exc}", err=True)
            errors.append(f"Output exists (use --force): dem_{code}")
        except (DemProcessingError, Exception) as exc:
            typer.echo(f"  FAIL DEM {code}: {exc}", err=True)
            errors.append(f"DEM processing failed ({code}): {exc}")

    # ------------------------------------------------------------------
    # C. OSM extraction
    # ------------------------------------------------------------------
    typer.echo("\n[C] OSM extraction")

    if skip_osm:
        typer.echo("  SKIPPED (--skip-osm flag set)")
    else:
        try:
            check_osm_backend(cfg.osm_pbf)
            typer.echo("  OSM backend available — extraction would proceed")
            # Full extraction not yet implemented; backend check passed.
        except OsmBackendUnavailable as exc:
            typer.echo("  BLOCKER: OSM extraction unavailable.", err=True)
            typer.echo(str(exc), err=True)
            errors.append("OSM extraction blocked: no PBF-capable backend")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    typer.echo("")
    if dry_run:
        typer.echo("[DRY RUN] No files written.")
    else:
        typer.echo(f"Outputs written: {len(outputs_written)}")
        for p in outputs_written:
            typer.echo(f"  {p}")

    if errors:
        typer.echo(f"\nWarnings/blockers: {len(errors)}")
        for e in errors:
            # OSM blocker is expected; other errors are failures
            if "OSM" in e or "exists" in e.lower():
                typer.echo(f"  NOTE: {e}")
            else:
                typer.echo(f"  ERROR: {e}", err=True)
        # Only hard-fail on non-OSM, non-overwrite errors
        hard_errors = [e for e in errors if "OSM" not in e and "exists" not in e.lower()]
        if hard_errors:
            raise typer.Exit(code=1)
    else:
        typer.echo("preprocess-geospatial completed successfully.")


# ---------------------------------------------------------------------------
# build-graph  (Stage 2+)
# ---------------------------------------------------------------------------


@app.command("build-graph")
def build_graph(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Build the road-network graph from processed data (Stage 2+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("build-graph called but no processed data is available.")
    typer.echo(
        "ERROR: build-graph requires processed road-network data. "
        "Complete Stage 1 data ingestion first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-analysis  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-analysis")
def run_analysis(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Run the core routing and shelter-allocation analysis (Stage 3+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("run-analysis called but data layer is absent.")
    typer.echo(
        "ERROR: run-analysis requires the processed data layer and a built graph. "
        "Complete Stages 1–2 first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-experiment  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-experiment")
def run_experiment(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help="Experiment ID from experiments.yaml."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Execute a named experiment (Stage 3+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("run-experiment called but prerequisites are absent.")
    typer.echo(
        "ERROR: run-experiment requires the full analysis pipeline (Stages 1–3). "
        "No results will be generated yet.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

























































































































































































# ---------------------------------------------------------------------------
# build-graph  (Stage 2+)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# validate-preprocessing-manifests  (Stage 3)
# ---------------------------------------------------------------------------


@app.command("validate-preprocessing-manifests")
def validate_preprocessing_manifests_cmd(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Root data/ directory."),
    ] = _DEFAULT_DATA,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate all preprocessing manifests under data/processed/preprocessing_manifests/."""
    configure_logging(log_level)  # type: ignore[arg-type]

@app.command("build-graph")
def build_graph(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Build the road-network graph from processed data (Stage 2+)."""
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.warning("build-graph called but no processed data is available.")
    typer.echo(
        "ERROR: build-graph requires processed road-network data. "
        "Complete Stage 1 data ingestion first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-analysis  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-analysis")
def run_analysis(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
    municipality: Annotated[
        str | None,
        typer.Option(
            "--municipality",
            "-m",
            help="Process only this pcode (e.g. PH0600608). Default: all.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate inputs and report planned outputs; do not write files.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing graph outputs."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Build directed road-network graphs from Stage 3 road layers (Stage 4)."""
    configure_logging(log_level)
    logger.info("build-graph starting (dry_run=%s, force=%s)", dry_run, force)
    typer.echo(f"[LIVE] build-graph — data_dir={data_dir}")

    import geopandas as gpd

    from floodroute.graph.build import GraphBuildError, build_road_graph
    from floodroute.graph.config import GraphConfig
    from floodroute.graph.io import write_edges_gpkg, write_graphml, write_nodes_gpkg
    from floodroute.graph.manifest import (
        build_graph_manifest,
        compute_file_sha256,
        write_graph_manifest,
    )
    from floodroute.graph.validate import GraphValidationFailed, validate_road_graph
    from floodroute.preprocessing.config import MUNICIPALITY_CODES, MUNICIPALITY_NAMES

    cfg = GraphConfig().resolve(data_dir)
    osm_dir = data_dir / "processed" / "osm"
    admin_dir = data_dir / "processed" / "admin"
    municipalities_gpkg = admin_dir / "municipalities_utm51n.gpkg"

    # Verify prerequisites
    if not osm_dir.is_dir():
        typer.echo("ERROR: processed/osm/ not found. Run preprocess-geospatial first.", err=True)
        raise typer.Exit(code=1)
    if not municipalities_gpkg.exists():
        typer.echo(
            f"ERROR: {municipalities_gpkg} not found. Run preprocess-geospatial first.",
            err=True,
        )
        raise typer.Exit(code=1)

    pcodes = [municipality] if municipality else list(MUNICIPALITY_CODES)

    # Validate pcode if provided
    if municipality and municipality not in MUNICIPALITY_CODES:
        typer.echo(
            f"ERROR: Unknown municipality code '{municipality}'. Valid: {MUNICIPALITY_CODES}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Check source files exist
    missing = []
    for pcode in pcodes:
        gpkg = osm_dir / f"{pcode}_roads_utm51n.gpkg"
        if not gpkg.exists():
            missing.append(str(gpkg))
    if missing:
        typer.echo(
            "ERROR: Missing Stage 3 road files:\n" + "\n".join(f"  {m}" for m in missing),
            err=True,
        )
        raise typer.Exit(code=1)

    cfg.output_graph_dir.mkdir(parents=True, exist_ok=True)
    cfg.graph_manifests_dir.mkdir(parents=True, exist_ok=True)

    # Check output existence before dry-run
    if not force and not dry_run:
        existing = []
        for pcode in pcodes:
            for suffix in ("_graph.graphml", "_nodes.gpkg", "_edges.gpkg"):
                p = cfg.output_graph_dir / f"{pcode}{suffix}"
                if p.exists():
                    existing.append(str(p))
        if existing:
            typer.echo(
                "ERROR: Output files already exist (use --force to overwrite):\n"
                + "\n".join(f"  {e}" for e in existing),
                err=True,
            )
            raise typer.Exit(code=1)

    if dry_run:
        typer.echo("DRY RUN — inputs verified. Planned outputs:")
        for pcode in pcodes:
            for suffix in ("_graph.graphml", "_nodes.gpkg", "_edges.gpkg"):
                typer.echo(f"  {cfg.output_graph_dir / f'{pcode}{suffix}'}")
        typer.echo("build-graph DRY RUN complete — no files written.")
        return

    # Load municipality polygons
    munis_gdf = gpd.read_file(municipalities_gpkg, layer="municipalities")

    all_ok = True
    output_count = 0

    for pcode in pcodes:
        name = MUNICIPALITY_NAMES.get(pcode, pcode)
        typer.echo(f"\n[{pcode}] {name}")

        roads_gpkg = osm_dir / f"{pcode}_roads_utm51n.gpkg"
        typer.echo(f"  Loading roads: {roads_gpkg.name}")
        roads_gdf = gpd.read_file(roads_gpkg, layer="roads")
        typer.echo(f"  {len(roads_gdf)} road features, CRS={roads_gdf.crs.to_epsg()}")

        # Get municipality polygon
        muni_row = munis_gdf[munis_gdf["adm3_pcode"] == pcode]
        muni_polygon = muni_row.geometry.iloc[0] if len(muni_row) > 0 else None

        # Build graph
        typer.echo("  Building road graph ...")
        try:
            G, build_stats = build_road_graph(
                roads_gdf,
                municipality_polygon=muni_polygon,
                coord_precision=cfg.coord_precision,
                bridge_grade_sep=cfg.bridge_grade_sep,
                tunnel_grade_sep=cfg.tunnel_grade_sep,
                oneway_forward=cfg.oneway_forward,
                oneway_reverse=cfg.oneway_reverse,
            )
        except GraphBuildError as exc:
            typer.echo(f"  ERROR building graph: {exc}", err=True)
            all_ok = False
            continue

        typer.echo(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        typer.echo(f"  Build stats: {build_stats.to_dict()}")

        # Validate
        try:
            graph_stats = validate_road_graph(G)
            val_status = "passed"
            typer.echo(
                f"  Validation PASSED: WCC={graph_stats['weakly_connected_component_count']}, "
                f"largest_wcc={graph_stats['largest_wcc_coverage']:.1%}"
            )
        except GraphValidationFailed as exc:
            typer.echo(f"  Validation FAILED: {exc}", err=True)
            val_status = "failed"
            all_ok = False

        # Write outputs
        graphml_path = cfg.output_graph_dir / f"{pcode}_graph.graphml"
        nodes_path = cfg.output_graph_dir / f"{pcode}_nodes.gpkg"
        edges_path = cfg.output_graph_dir / f"{pcode}_edges.gpkg"

        write_graphml(G, graphml_path)
        write_nodes_gpkg(G, nodes_path, crs=cfg.crs)
        write_edges_gpkg(G, edges_path, crs=cfg.crs)
        typer.echo(f"  Wrote: {graphml_path.name}, {nodes_path.name}, {edges_path.name}")
        output_count += 3

        # Compute source checksum
        roads_sha256 = compute_file_sha256(roads_gpkg)

        # Build and write manifest
        parameters = {
            "crs": cfg.crs,
            "buffer_metres": cfg.buffer_metres,
            "coord_precision_mm": 10 ** (3 - cfg.coord_precision),
            "oneway_forward_values": sorted(cfg.oneway_forward),
            "oneway_reverse_values": sorted(cfg.oneway_reverse),
            "oneway_bidirectional_default": True,
            "bridge_grade_sep_values": sorted(cfg.bridge_grade_sep),
            "tunnel_grade_sep_values": sorted(cfg.tunnel_grade_sep),
            "grade_sep_rule": (
                "Interior geometric crossings between a bridge/tunnel way and any "
                "other way do not produce shared nodes. Endpoint connections are "
                "always preserved regardless of bridge/tunnel tags."
            ),
        }

        manifest = build_graph_manifest(
            municipality_code=pcode,
            municipality_name=name,
            source_roads_path=roads_gpkg,
            source_roads_sha256=roads_sha256,
            source_admin_path=municipalities_gpkg,
            parameters=parameters,
            build_stats=build_stats.to_dict(),
            graph_stats=graph_stats if val_status == "passed" else {},
            output_graphml_path=graphml_path,
            output_nodes_gpkg_path=nodes_path,
            output_edges_gpkg_path=edges_path,
            validation_status=val_status,
        )
        manifest_path = cfg.graph_manifests_dir / f"{pcode}_graph.json"
        write_graph_manifest(manifest, manifest_path)
        typer.echo(f"  Manifest: {manifest_path.name}")

    typer.echo(f"\nOutputs written: {output_count}")
    if all_ok:
        typer.echo("build-graph Stage 4 complete.")
    else:
        typer.echo("build-graph PARTIAL — some municipalities failed.", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# inspect-graph  (Stage 4)
# ---------------------------------------------------------------------------


@app.command("inspect-graph")
def inspect_graph_cmd(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Path to the data/ directory."),
    ] = _DEFAULT_DATA,
    municipality: Annotated[
        str | None,
        typer.Option("--municipality", "-m", help="Inspect only this pcode."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Report statistics for built road-network graphs."""
    configure_logging(log_level)

    from floodroute.graph.config import GraphConfig
    from floodroute.graph.io import read_graphml
    from floodroute.graph.validate import compute_graph_stats
    from floodroute.preprocessing.config import MUNICIPALITY_CODES, MUNICIPALITY_NAMES

    cfg = GraphConfig().resolve(data_dir)
    pcodes = [municipality] if municipality else list(MUNICIPALITY_CODES)

    for pcode in pcodes:
        name = MUNICIPALITY_NAMES.get(pcode, pcode)
        graphml_path = cfg.output_graph_dir / f"{pcode}_graph.graphml"
        if not graphml_path.exists():
            typer.echo(f"{pcode} ({name}): graph not found at {graphml_path}")
            continue
        G = read_graphml(graphml_path)
        stats = compute_graph_stats(G)
        typer.echo(f"\n{pcode} ({name})")
        typer.echo(f"  Nodes:  {stats['node_count']}")
        typer.echo(f"  Edges:  {stats['edge_count']}")
        typer.echo(f"  Self-loops: {stats['self_loop_count']}")
        typer.echo(f"  Parallel edge pairs: {stats['parallel_edge_pairs']}")
        typer.echo(f"  Isolated nodes: {stats['isolated_node_count']}")
        typer.echo(f"  Boundary nodes: {stats['boundary_node_count']}")
        typer.echo(f"  WCC count: {stats['weakly_connected_component_count']}")
        typer.echo(
            f"  Largest WCC: {stats['largest_wcc_node_count']} nodes "
            f"({stats['largest_wcc_coverage']:.1%})"
        )
        typer.echo(f"  SCC count (>1 node): {stats['strongly_connected_component_count']}")
        typer.echo(
            f"  Largest SCC: {stats['largest_scc_node_count']} nodes "
            f"({stats['largest_scc_coverage']:.1%})"
        )


# ---------------------------------------------------------------------------
# validate-graph  (Stage 4)
# ---------------------------------------------------------------------------


@app.command("validate-graph")
def validate_graph_cmd(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Path to the data/ directory."),
    ] = _DEFAULT_DATA,
    municipality: Annotated[
        str | None,
        typer.Option("--municipality", "-m", help="Validate only this pcode."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate built road-network graphs against structural criteria."""
    configure_logging(log_level)

    from floodroute.graph.config import GraphConfig
    from floodroute.graph.io import read_graphml
    from floodroute.graph.validate import GraphValidationFailed, validate_road_graph
    from floodroute.preprocessing.config import MUNICIPALITY_CODES, MUNICIPALITY_NAMES

    cfg = GraphConfig().resolve(data_dir)
    pcodes = [municipality] if municipality else list(MUNICIPALITY_CODES)
    all_ok = True

    for pcode in pcodes:
        name = MUNICIPALITY_NAMES.get(pcode, pcode)
        graphml_path = cfg.output_graph_dir / f"{pcode}_graph.graphml"
        if not graphml_path.exists():
            typer.echo(f"ERROR: {pcode} graph not found: {graphml_path}", err=True)
            all_ok = False
            continue
        G = read_graphml(graphml_path)
        try:
            stats = validate_road_graph(G)
            typer.echo(
                f"{pcode} ({name}): VALID — {stats['node_count']} nodes, "
                f"{stats['edge_count']} edges, WCC={stats['largest_wcc_coverage']:.1%}"
            )
        except GraphValidationFailed as exc:
            typer.echo(f"{pcode} ({name}): FAILED — {exc}", err=True)
            all_ok = False

    if not all_ok:
        raise typer.Exit(code=1)
    typer.echo("validate-graph: all graphs passed.")


# ---------------------------------------------------------------------------
# run-analysis  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-analysis")
def run_analysis(
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),


















































        "ERROR: run-analysis requires the processed data layer and a built graph. "
        "Complete Stages 1–2 first.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run-experiment  (Stage 3+)
# ---------------------------------------------------------------------------


@app.command("run-experiment")
def run_experiment(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help="Experiment ID from experiments.yaml."),
    ] = None,
    log_level: Annotated[
        str,
    logger.warning("run-experiment called but prerequisites are absent.")
    typer.echo(
        "ERROR: run-experiment requires the full analysis pipeline (Stages 1–3). "
        "No results will be generated yet.",
        err=True,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
# Stage 5 — hazard commands
# ---------------------------------------------------------------------------

_DEFAULT_JRC_RAW = _DEFAULT_DATA / "raw" / "jrc_glofas_flood_hazard"
_DEFAULT_JRC_RASTER = _DEFAULT_JRC_RAW / "sibalom_jrc_glofas_v21_raw.tif"
_DEFAULT_HAZARD_OUT = _DEFAULT_DATA / "processed" / "hazard"
_GEE_PROJECT_ENV = "GEE_PROJECT"
_GEE_PROJECT_DEFAULT = "geo-analyzer-web"


def _get_gee_project(gee_project: str | None) -> str:
    """Return GEE project from argument, env var, or documented default."""
    if gee_project:
        return gee_project
    env = os.environ.get(_GEE_PROJECT_ENV)
    if env:
        return env
    return _GEE_PROJECT_DEFAULT


@app.command("acquire-hazard")
def acquire_hazard(
    gee_project: Annotated[
        str | None,
        typer.Option(
            "--gee-project",
            help=f"Google Earth Engine Cloud project ID. "
            f"Can also be set via ${_GEE_PROJECT_ENV} env var. "
            f"Default: {_GEE_PROJECT_DEFAULT}",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for the raw JRC raster."),
    ] = _DEFAULT_JRC_RAW,
    admin_zip: Annotated[
        Path,
        typer.Option("--admin-zip", help="PSA admin boundaries ZIP."),
    ] = _DEFAULT_DATA / "raw" / "psa_administrative_boundaries_antique.zip",
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite existing raster."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Download the JRC GloFAS v2.1 Sibalom subset from Google Earth Engine.

    Acquires the minimum spatial subset (Sibalom municipality + 0.05° buffer) and
    the five required bands: RP10_depth, RP20_depth, RP100_depth,
    permanent_water_class, spurious_depth_category.

    The GEE project must be authenticated before running this command.
    Run: earthengine authenticate
    """
    configure_logging(log_level)  # type: ignore[arg-type]
    project = _get_gee_project(gee_project)
    out_path = output_dir / "sibalom_jrc_glofas_v21_raw.tif"

    logger.info("acquire-hazard: project=%s, output=%s, force=%s", project, out_path, force)
    typer.echo(f"[acquire-hazard] GEE project: {project}")

    if not admin_zip.exists():
        typer.echo(f"ERROR: admin ZIP not found: {admin_zip}", err=True)
        raise typer.Exit(code=1)

    from floodroute.hazard.jrc import acquire_sibalom_subset  # noqa: PLC0415

    try:
        path, sha256 = acquire_sibalom_subset(
            out_path,
            ee_project=project,
            admin_zip=admin_zip,
            force=force,
        )
        typer.echo(f"Saved: {path}")
        typer.echo(f"SHA-256: {sha256}")
        typer.echo(f"Size: {path.stat().st_size:,} bytes")
    except FileExistsError as exc:
        typer.echo(f"ERROR: {exc}. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("inspect-hazard")
def inspect_hazard(
    raster_path: Annotated[
        Path,
        typer.Option("--raster", help="Path to the acquired JRC raw GeoTIFF."),
    ] = _DEFAULT_JRC_RASTER,
    admin_zip: Annotated[
        Path,
        typer.Option("--admin-zip", help="PSA admin boundaries ZIP."),
    ] = _DEFAULT_DATA / "raw" / "psa_administrative_boundaries_antique.zip",
    edges_gpkg: Annotated[
        Path | None,
        typer.Option("--edges-gpkg", help="Stage 4 edges GeoPackage for road intersection stats."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Run Phase A hazard-readiness inspection for Sibalom.

    Validates the JRC GloFAS v2.1 raster and reports:
    - CRS, resolution, bounds
    - Valid and nodata pixel counts per return period
    - Flooded pixel statistics
    - Monotonicity checks
    - Permanent water and spurious depth flags
    - Road graph intersection (if --edges-gpkg supplied)
    - Phase A readiness decision: READY / PARTIAL / BLOCKED
    """
    configure_logging(log_level)  # type: ignore[arg-type]
    logger.info("inspect-hazard: raster=%s", raster_path)

    if not raster_path.exists():
        typer.echo(
            f"ERROR: JRC raster not found: {raster_path}\nRun: floodroute acquire-hazard",
            err=True,
        )
        raise typer.Exit(code=1)

    if not admin_zip.exists():
        typer.echo(f"ERROR: admin ZIP not found: {admin_zip}", err=True)
        raise typer.Exit(code=1)

    edges_path: Path | None = edges_gpkg
    if edges_path is None:
        default_edges = _DEFAULT_DATA / "processed" / "graph" / "PH0600616_edges.gpkg"
        if default_edges.exists():
            edges_path = default_edges

    from floodroute.hazard.jrc import ReadinessDecision, run_phase_a  # noqa: PLC0415

    result = run_phase_a(
        raster_path,
        admin_zip=admin_zip,
        edges_path=edges_path,
    )

    typer.echo("\nJRC GloFAS v2.1 — Phase A Readiness Report")
    typer.echo(f"Municipality: {result.municipality_code}")
    typer.echo(f"Raster: {result.raster_path.name}")
    typer.echo(f"SHA-256: {result.sha256}")
    typer.echo(f"CRS: {result.raster_crs}  Pixel: ~{result.pixel_size_m_approx} m")
    typer.echo(f"Bounds (W/S/E/N): {result.raster_bounds}")
    typer.echo(f"Municipality pixels: {result.municipality_total_pixels:,}")

    typer.echo("\n--- Return-period depth coverage ---")
    for band_name, cov in result.band_coverage.items():
        typer.echo(
            f"  {band_name}: valid={cov.valid_pixels:,} ({cov.valid_pct:.2f}%), "
            f"flooded≥0.10m={cov.flooded_ge_threshold:,} ({cov.flooded_pct:.2f}%)"
        )
        if cov.flooded_ge_threshold > 0:
            typer.echo(
                f"    depth min/med/max={cov.depth_min:.3f}/{cov.depth_median:.3f}/{cov.depth_max:.3f} m"
            )

    typer.echo(
        f"\nMonotonicity: RP10≤RP20 violations={result.monotonicity_violations_rp10_rp20}, "
        f"RP20≤RP100 violations={result.monotonicity_violations_rp20_rp100}"
    )
    typer.echo(
        f"Permanent water: {result.permanent_water_pixels} pixels  "
        f"Spurious flagged: {result.spurious_flagged_pixels} pixels"
    )

    if result.road_intersection:
        ri = result.road_intersection
        typer.echo("\n--- Road graph intersection ---")
        typer.echo(
            f"  Directed edges touching valid pixel: "
            f"{ri.edges_touching_valid}/{ri.total_directed_edges} ({ri.edges_touching_pct:.1f}%)"
        )
        typer.echo(
            f"  Physical segments: "
            f"{ri.segments_touching_valid}/{ri.total_physical_segments} ({ri.segments_touching_pct:.1f}%)"
        )
        typer.echo(
            f"  Physical length: "
            f"{ri.length_touching_km:.2f}/{ri.total_physical_length_km:.2f} km "
            f"({ri.length_touching_pct:.1f}%)"
        )

    typer.echo(f"\n{'=' * 60}")
    decision_sym = {"READY": "✓", "PARTIAL": "~", "BLOCKED": "✗"}.get(result.decision.value, "?")
    typer.echo(f"PHASE A DECISION: [{decision_sym}] {result.decision.value}")
    for r in result.reasons:
        typer.echo(f"  - {r}")

    if result.decision != ReadinessDecision.READY:
        typer.echo("\nPhase B integration will NOT run (decision is not READY).")
        typer.echo(
            "See data/manifests/jrc_glofas_flood_hazard_v21.yaml and "
            "data/processed/hazard/PH0600616_jrc_phase_a_readiness.yaml for details."
        )

    typer.echo("")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()














































































        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("\nStage 5B — Phase B Integration Complete")
    typer.echo(f"Municipality:        {result.municipality_code}")
    typer.echo(f"Directed edges:      {result.n_directed_edges:,}")
    typer.echo(f"Physical segments:   {result.n_physical_segments:,}")
    typer.echo(
        f"Exposed edges RP10:  {result.n_exposed_rp10:,}  ({result.total_exposed_m_rp10:.1f} m)"
    )
    typer.echo(
        f"Exposed edges RP20:  {result.n_exposed_rp20:,}  ({result.total_exposed_m_rp20:.1f} m)"
    )
    typer.echo(
        f"Exposed edges RP100: {result.n_exposed_rp100:,}  ({result.total_exposed_m_rp100:.1f} m)"
    )
    typer.echo(f"Waterway crossings:  {result.n_waterway_crossings:,}")
    typer.echo(f"\nEnriched GraphML: {result.output_graphml}")
    typer.echo(f"  SHA-256: {result.sha256_graphml}")
    typer.echo(f"Enriched GPKG:    {result.output_gpkg}")
    typer.echo(f"  SHA-256: {result.sha256_gpkg}")
    typer.echo("")


@app.command("validate-hazard")
def validate_hazard(
    enriched_gpkg: Annotated[
        Path,
        typer.Option("--gpkg", help="Phase B enriched GeoPackage to validate."),
    ] = _DEFAULT_HAZARD_OUT / "PH0600613_phase_b_enriched.gpkg",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Validate Phase B enriched GeoPackage: check required columns, value ranges and consistency.

    Exit code 0 if all checks pass; 1 if any validation fails.
    """
    configure_logging(log_level)  # type: ignore[arg-type]

    if not enriched_gpkg.exists():
        typer.echo(f"ERROR: enriched GeoPackage not found: {enriched_gpkg}", err=True)
        typer.echo("Run: floodroute integrate-hazard", err=True)
        raise typer.Exit(code=1)

    import geopandas as gpd  # noqa: PLC0415



































    required = jrc_cols + terrain_cols + waterway_cols + network_cols
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        failures.append(f"Missing columns: {missing}")

    if not missing:
        # Valid status values
        valid_statuses = {"no_overlap", "outside_domain", "modelled_dry", "flooded"}
        for rp in ("rp10", "rp20", "rp100"):
            col = f"jrc_{rp}_status"
            bad = gdf[~gdf[col].isin(valid_statuses)][col].unique()
            if len(bad):
                failures.append(f"{col} has invalid values: {bad}")

        # exposed_m must be >= 0
        for rp in ("rp10", "rp20", "rp100"):
            col = f"jrc_{rp}_exposed_m"
            if (gdf[col].dropna() < 0).any():
                failures.append(f"{col} has negative values")

        # exposed_pct must be in [0, 100]
        for rp in ("rp10", "rp20", "rp100"):
            col = f"jrc_{rp}_exposed_pct"
            if (gdf[col].dropna() > 100.01).any():
                failures.append(f"{col} exceeds 100%")

        # Monotonicity: RP10 exposed ≤ RP100 exposed (not always true per edge, skip)

        # Terrain slopes must be ≥ 0
        if (gdf["terrain_slope_pct"].dropna() < 0).any():
            failures.append("terrain_slope_pct has negative values")

        # WCC ids must be >= 0 or -1 (isolated)
        if (gdf["network_wcc_id"].dropna() < -1).any():
            failures.append("network_wcc_id has values below -1")

        # Paired-edge consistency: for each (osm_id, edge_seq), both directions
        # should have the same jrc_rp10_exposed_m
        for rp in ("rp10", "rp100"):
            col = f"jrc_{rp}_exposed_m"
            grp = gdf.groupby(["osm_id", "edge_seq"])[col].agg(["min", "max"])
            inconsistent = grp[(grp["max"] - grp["min"]).abs() > 0.01]
            if len(inconsistent) > 0:
                failures.append(
                    f"{col}: {len(inconsistent)} physical segments have inconsistent "
                    f"values across directed-edge pairs (max diff > 0.01 m)"
                )

    typer.echo(f"\nPhase B Validation — {enriched_gpkg.name}")
    typer.echo(f"Edges: {n:,}  Columns checked: {len(required)}")

    if failures:
        typer.echo(f"\nFAILED ({len(failures)} issues):", err=True)
        for f in failures:
            typer.echo(f"  ✗ {f}", err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"\nPASSED — all {len(required)} column checks satisfied.")
        typer.echo("")


@app.command("hazard-summary")
def hazard_summary(
    enriched_gpkg: Annotated[
        Path,
        typer.Option("--gpkg", help="Phase B enriched GeoPackage."),
    ] = _DEFAULT_HAZARD_OUT / "PH0600613_phase_b_enriched.gpkg",
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level."),
    ] = "INFO",
) -> None:
    """Print an exposure summary for the Phase B enriched graph.

    Reports per-RP counts and lengths of exposed road segments, depth class
    breakdown, waterway crossings, terrain statistics, and WCC coverage.
    Does NOT infer passability or routing decisions.
    """
    configure_logging(log_level)  # type: ignore[arg-type]

    if not enriched_gpkg.exists():
        typer.echo(f"ERROR: enriched GeoPackage not found: {enriched_gpkg}", err=True)
        typer.echo("Run: floodroute integrate-hazard", err=True)
        raise typer.Exit(code=1)

    import geopandas as gpd  # noqa: PLC0415

    gdf = gpd.read_file(enriched_gpkg, layer="edges")
    n_total = len(gdf)

    typer.echo(f"\n{'=' * 65}")
    typer.echo("Stage 5B Exposure Summary — PH0600613 San Jose de Buenavista")
    typer.echo(f"{'=' * 65}")
    typer.echo(f"Total directed edges : {n_total:,}")

    # Deduplicate to physical segments
    if "osm_id" in gdf.columns and "edge_seq" in gdf.columns:
        phys = gdf.drop_duplicates(subset=["osm_id", "edge_seq"])
        total_km = phys["length_m"].sum() / 1000
        typer.echo(f"Physical segments    : {len(phys):,}  ({total_km:.1f} km)")

    typer.echo("")
    typer.echo("JRC Flood Evidence (non-permanent over-bank inundation):")
    typer.echo(
        f"  {'Return Period':<14} {'Exposed edges':>13} {'Exposed km':>10} {'% of total km':>14}"
    )
    typer.echo(f"  {'-' * 55}")
    for rp in ("rp10", "rp20", "rp100"):
        col_m = f"jrc_{rp}_exposed_m"
        if col_m not in gdf.columns:
            continue
        exp_edges = int((gdf[col_m] > 0).sum())
        exp_km = gdf[col_m].sum() / 1000
        pct = (exp_km / total_km * 100) if total_km > 0 else 0
        typer.echo(f"  {rp.upper():<14} {exp_edges:>13,} {exp_km:>10.3f} {pct:>13.2f}%")

    typer.echo("")
    typer.echo("JRC Status breakdown (RP10):")
    if "jrc_rp10_status" in gdf.columns:
        for status, cnt in gdf["jrc_rp10_status"].value_counts().items():
            typer.echo(f"  {status:<20} {cnt:>6,}")

    typer.echo("")
    typer.echo("Depth class breakdown (RP10, non-permanent, by max depth per edge):")
    if "jrc_rp10_depth_max_m" in gdf.columns:
        exposed = gdf[gdf["jrc_rp10_exposed_m"] > 0].copy()
        low = int((exposed["jrc_rp10_depth_max_m"].between(0.10, 0.50, inclusive="left")).sum())
        med = int((exposed["jrc_rp10_depth_max_m"].between(0.50, 1.50, inclusive="left")).sum())
        hi = int((exposed["jrc_rp10_depth_max_m"] >= 1.50).sum())
        typer.echo(f"  Low  [0.10–0.50) m : {low:>6,}")
        typer.echo(f"  Med  [0.50–1.50) m : {med:>6,}")
        typer.echo(f"  High [1.50+)     m : {hi:>6,}")

    typer.echo("")
    typer.echo("Waterway evidence:")
    if "waterway_crossing" in gdf.columns:
        n_cross = int(gdf["waterway_crossing"].sum())
        typer.echo(f"  Edges crossing waterway : {n_cross:,}")
    if "waterway_nearest_dist_m" in gdf.columns:
        med_dist = gdf["waterway_nearest_dist_m"].median()
        typer.echo(f"  Median dist to waterway : {med_dist:.0f} m")

    typer.echo("")
    typer.echo("Terrain evidence:")
    for col, label in [
        ("terrain_elev_mean_m", "Mean elevation  "),
        ("terrain_slope_pct", "Mean slope      "),
    ]:
        if col in gdf.columns:
            v = gdf[col].mean()
            typer.echo(f"  {label}: {v:.2f}")

    typer.echo("")
    typer.echo("Network evidence:")
    if "network_wcc_id" in gdf.columns:
        n_wcc = gdf["network_wcc_id"].nunique()
        pct_main = int((gdf["network_wcc_id"] == 0).sum()) / n_total * 100
        typer.echo(f"  Weakly connected components : {n_wcc}")
        typer.echo(f"  Edges in largest WCC        : {pct_main:.1f}%")

    typer.echo(f"\n{'=' * 65}")
    typer.echo("NOTE: This report records evidence only.")
    typer.echo("Passability and routing decisions are NOT inferred here.")
    typer.echo("")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

    if "waterway_crossing" in gdf.columns:
        n_cross = int(gdf["waterway_crossing"].sum())
        typer.echo(f"  Edges crossing waterway : {n_cross:,}")
    if "waterway_nearest_dist_m" in gdf.columns:
        med_dist = gdf["waterway_nearest_dist_m"].median()
        typer.echo(f"  Median dist to waterway : {med_dist:.0f} m")
