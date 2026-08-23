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
    from floodroute.preprocessing.osm import (
        ChecksumMismatch,
        OsmBackendUnavailable,
        OsmExtractionError,
        extract_osm_features,
        get_osmium_version,
        resolve_pbf_from_manifest,
        verify_pbf_checksum,
    )
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
    osm_pbf_path: Path = Path()  # resolved below when not skip_osm
    osm_expected_sha256: str = ""
    if not cfg.admin_archive.exists():
        missing_inputs.append(f"Admin archive: {cfg.admin_archive}")
    for tile in cfg.dem_tiles:
        if not tile.exists():
            missing_inputs.append(f"DEM tile: {tile}")
    if not skip_osm:
        # Resolve PBF path from acquisition manifest (no hard-coded path)
        try:
            osm_pbf_path, osm_expected_sha256 = resolve_pbf_from_manifest(
                OSM_DATASET_ID, manifests_dir, data_dir
            )
        except OsmExtractionError as exc:
            typer.echo(f"ERROR: Cannot resolve OSM source: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        if not osm_pbf_path.exists():
            missing_inputs.append(f"OSM PBF: {osm_pbf_path}")

    if missing_inputs:
        typer.echo("ERROR: Required raw inputs are missing:", err=True)
        for m in missing_inputs:
            typer.echo(f"  - {m}", err=True)
        raise typer.Exit(code=1)

    # Source checksums from acquisition manifests (for preprocessing manifests)
    admin_sha = source_checksum_from_manifest_dir(ADMIN_DATASET_ID, manifests_dir)
    dem_sha = source_checksum_from_manifest_dir(DEM_DATASET_ID, manifests_dir)
    osm_sha = source_checksum_from_manifest_dir(OSM_DATASET_ID, manifests_dir)

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
                    validation_status=val_status.get("validation"),
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
                    validation_status=val_status_dem.get("validation"),
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

    osm_skipped = False
    if skip_osm:
        typer.echo("  SKIPPED (--skip-osm flag set — Stage 3 will be partial)")
        osm_skipped = True
    else:
        try:
            # Verify PBF checksum against acquisition manifest
            typer.echo(
                f"  Verifying PBF checksum: {osm_pbf_path.name} … "
                "(this may take 5–15 s for a 604 MB file)"
            )
            try:
                verify_pbf_checksum(osm_pbf_path, osm_expected_sha256)
                typer.echo(f"  PBF checksum OK: {osm_pbf_path.name}")
            except ChecksumMismatch as exc:
                typer.echo(f"  ERROR: {exc}", err=True)
                errors.append(f"OSM PBF checksum mismatch: {exc}")
                raise typer.Exit(code=1) from exc

            if dry_run:
                typer.echo(
                    f"  [DRY RUN] osmium {get_osmium_version()} available; "
                    f"PBF checksum verified; "
                    f"would extract roads and waterways for "
                    f"{len(MUNICIPALITY_CODES)} municipalities "
                    f"(buffer={cfg.osm_buffer_metres:.0f} m)."
                )
                typer.echo(f"  [DRY RUN] waterway classes: {sorted(cfg.waterway_classes)}")
            else:
                # Load municipality boundaries for buffer computation
                from floodroute.preprocessing.validation import (
                    ValidationFailed,
                    validate_osm_output,
                )

                muni_wgs84_path = cfg.output_admin_dir / "municipalities_wgs84.gpkg"
                if not muni_wgs84_path.exists():
                    raise OsmExtractionError(
                        f"Municipality boundaries required for OSM extraction "
                        f"but not found: {muni_wgs84_path}\n"
                        "Run admin preprocessing first (without --skip-osm)."
                    )
                import geopandas as gpd

                muni_gdf = gpd.read_file(muni_wgs84_path, layer="municipalities")

                typer.echo(
                    f"  osmium {get_osmium_version()} — extracting roads and "
                    f"waterways for {len(MUNICIPALITY_CODES)} municipalities "
                    f"(buffer={cfg.osm_buffer_metres:.0f} m) …"
                )
                typer.echo("  (single PBF pass; ~1–2 GB RAM for location index)")

                osm_result = extract_osm_features(
                    osm_pbf_path,
                    muni_gdf,
                    cfg.output_osm_dir,
                    pcode_field="adm3_pcode",
                    buffer_metres=cfg.osm_buffer_metres,
                    waterway_classes=cfg.waterway_classes,
                    target_crs=cfg.target_crs,
                    force=force,
                    dry_run=False,
                )

                osm_stats = osm_result["stats"]

                # Report extraction counts
                typer.echo(
                    f"  Ways examined: {osm_stats.ways_examined:,}  "
                    f"road candidates: {osm_stats.road_candidates:,}  "
                    f"waterway candidates: {osm_stats.waterway_candidates:,}"
                )
                typer.echo(
                    f"  Incomplete location: {osm_stats.incomplete_location}  "
                    f"Invalid geometry: {osm_stats.invalid_geom}  "
                    f"Empty geometry: {osm_stats.empty_geom}  "
                    f"Outside all buffers: {osm_stats.outside_all}"
                )
                if osm_stats.road_cross_municipal_ids:
                    typer.echo(
                        f"  Cross-municipal road ways (legitimate): "
                        f"{len(osm_stats.road_cross_municipal_ids)}"
                    )
                if osm_stats.waterway_cross_municipal_ids:
                    typer.echo(
                        f"  Cross-municipal waterway ways (legitimate): "
                        f"{len(osm_stats.waterway_cross_municipal_ids)}"
                    )

                # Per-municipality counts and validation + manifests
                from floodroute.preprocessing.osm import build_municipality_buffers

                muni_buffers = build_municipality_buffers(
                    muni_gdf,
                    buffer_metres=cfg.osm_buffer_metres,
                    projected_crs=cfg.target_crs,
                )

                for pcode in sorted(MUNICIPALITY_CODES):
                    muni_name = MUNICIPALITY_NAMES.get(pcode, pcode)
                    n_roads = osm_stats.retained_roads.get(pcode, 0)
                    n_ww = osm_stats.retained_waterways.get(pcode, 0)
                    typer.echo(
                        f"  {pcode} ({muni_name}): "
                        f"{n_roads} road features, {n_ww} waterway features"
                    )

                    # Validate and manifest each of the 4 outputs per municipality
                    for feat_type, layer in (("roads", "roads"), ("waterways", "waterways")):
                        for crs_label, crs_suffix, _out_crs in (
                            ("EPSG:4326", "wgs84", "EPSG:4326"),
                            (cfg.target_crs, "utm51n", cfg.target_crs),
                        ):
                            out_id = f"{pcode}_{feat_type}_{crs_suffix}"
                            out_path = osm_result["output_paths"].get(out_id)
                            if out_path is None:
                                errors.append(f"Missing expected output: {out_id}")
                                continue

                            # Build municipality buffer in the output CRS for validation
                            if crs_suffix == "wgs84":
                                val_buffer = muni_buffers[pcode]
                            else:
                                import geopandas as _gpd_v

                                _buf_gdf = _gpd_v.GeoDataFrame(
                                    geometry=[muni_buffers[pcode]], crs="EPSG:4326"
                                ).to_crs(cfg.target_crs)
                                val_buffer = _buf_gdf.geometry.iloc[0]

                            # WGS84 ids are authoritative; UTM must match
                            wgs84_id = f"{pcode}_{feat_type}_wgs84"
                            expected_ids: set[int] | None = None
                            if crs_suffix == "utm51n":
                                wgs84_path = osm_result["output_paths"].get(wgs84_id)
                                if wgs84_path is not None and wgs84_path.exists():
                                    try:
                                        _ref_gdf = gpd.read_file(wgs84_path, layer=layer)
                                        expected_ids = set(_ref_gdf["osm_id"].tolist())
                                    except Exception:
                                        pass

                            val_status_osm = "passed"
                            val_error_osm: str | None = None
                            try:
                                validate_osm_output(
                                    out_path,
                                    layer=layer,
                                    expected_crs=crs_label,
                                    municipality_buffer=val_buffer,
                                    feature_type=feat_type,
                                    expected_osm_id_set=expected_ids,
                                )
                            except ValidationFailed as vexc:
                                val_status_osm = "failed"
                                val_error_osm = str(vexc)
                                typer.echo(f"    VALIDATION FAIL {out_id}: {vexc}", err=True)
                                errors.append(f"OSM validation failed ({out_id}): {vexc}")

                            if val_status_osm == "passed":
                                n_feat = n_roads if feat_type == "roads" else n_ww
                                typer.echo(f"    VALID {out_id}: {n_feat} features, {crs_label}")

                            # Compute bounds for manifest
                            try:
                                _gdf_m = gpd.read_file(out_path, layer=layer)
                                if len(_gdf_m) > 0:
                                    _tb = _gdf_m.total_bounds
                                    _bounds = {
                                        "xmin": float(_tb[0]),
                                        "ymin": float(_tb[1]),
                                        "xmax": float(_tb[2]),
                                        "ymax": float(_tb[3]),
                                    }
                                    _n_feat = len(_gdf_m)
                                else:
                                    _bounds = {}
                                    _n_feat = 0
                            except Exception:
                                _bounds = {}
                                _n_feat = n_roads if feat_type == "roads" else n_ww

                            man_extra: dict[str, Any] = {
                                "osmium_version": osm_result["osmium_version"],
                                "tag_policy": {
                                    "road_filter": "highway tag present and non-empty",
                                    "waterway_filter": "waterway tag in included_classes",
                                    "included_waterway_classes": osm_result["waterway_classes"],
                                    "unknown_tags": "null (never inferred)",
                                },
                                "buffer_metres": osm_result["buffer_metres"],
                                "spatial_retention_rule": (
                                    "Ways intersecting the buffered municipality boundary "
                                    "are retained. Full way geometry is written (not clipped) "
                                    "to preserve cross-boundary road continuity."
                                ),
                                "error_accounting": {
                                    "ways_examined": osm_stats.ways_examined,
                                    "road_candidates": osm_stats.road_candidates,
                                    "waterway_candidates": osm_stats.waterway_candidates,
                                    "incomplete_location": osm_stats.incomplete_location,
                                    "invalid_geom": osm_stats.invalid_geom,
                                    "empty_geom": osm_stats.empty_geom,
                                    "outside_all_buffers": osm_stats.outside_all,
                                    "cross_municipal_road_ids": len(
                                        osm_stats.road_cross_municipal_ids
                                    ),
                                    "cross_municipal_waterway_ids": len(
                                        osm_stats.waterway_cross_municipal_ids
                                    ),
                                },
                            }
                            if val_error_osm:
                                man_extra["validation_error"] = val_error_osm

                            man = build_preprocessing_manifest(
                                output_id=out_id,
                                operation="osm_feature_extraction",
                                parameters={
                                    "municipality_code": pcode,
                                    "municipality_name": muni_name,
                                    "feature_type": feat_type,
                                    "output_crs": crs_label,
                                    "buffer_metres": osm_result["buffer_metres"],
                                    "waterway_classes": osm_result["waterway_classes"],
                                    "source_crs": "EPSG:4326",
                                    "projected_buffer_crs": cfg.target_crs,
                                },
                                source_dataset_ids=[OSM_DATASET_ID, ADMIN_DATASET_ID],
                                source_checksums={
                                    OSM_DATASET_ID: osm_sha,
                                    ADMIN_DATASET_ID: admin_sha,
                                },
                                output_path=out_path,
                                output_crs=crs_label,
                                output_bounds=_bounds,
                                feature_count=_n_feat,
                                validation_status=val_status_osm,
                                extra=man_extra,
                            )
                            man_path = cfg.output_manifests_dir / f"{out_id}.json"
                            write_preprocessing_manifest(man, man_path)
                            outputs_written.append(str(out_path))

        except (OsmBackendUnavailable, OsmExtractionError) as exc:
            typer.echo(f"  ERROR: {exc}", err=True)
            errors.append(f"OSM extraction failed: {exc}")
        except typer.Exit:
            raise
        except Exception as exc:
            typer.echo(f"  FAIL OSM extraction: {exc}", err=True)
            errors.append(f"OSM extraction failed (unexpected): {exc}")

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
        typer.echo(f"\nIssues: {len(errors)}")
        for e in errors:
            if "exists" in e.lower():
                typer.echo(f"  NOTE: {e}")
            else:
                typer.echo(f"  ERROR: {e}", err=True)
        hard_errors = [e for e in errors if "exists" not in e.lower()]
        if hard_errors:
            raise typer.Exit(code=1)
    elif osm_skipped:
        typer.echo(
            "preprocess-geospatial PARTIAL — admin and DEM complete; "
            "OSM skipped (--skip-osm supplied). "
            "Re-run without --skip-osm to complete Stage 3."
        )
    else:
        typer.echo("preprocess-geospatial Stage 3 complete.")


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

    from floodroute.preprocessing.prep_manifest import validate_preprocessing_manifests

    manifests_dir = data_dir / "processed" / "preprocessing_manifests"
    if not manifests_dir.is_dir():
        typer.echo(f"ERROR: preprocessing manifests directory not found: {manifests_dir}", err=True)
        raise typer.Exit(code=1)

    try:
        results = validate_preprocessing_manifests(manifests_dir)
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not results:
        typer.echo("No preprocessing manifests found.")
        return

    for m in results:
        vs = m.get("validation_status", "—")
        typer.echo(f"  OK  {m['output_id']}  (op={m['operation']}, validation_status={vs})")
    typer.echo(f"\nAll {len(results)} preprocessing manifest(s) valid.")


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
