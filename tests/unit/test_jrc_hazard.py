"""Phase A unit tests for JRC GloFAS v2.1 hazard module (corrected methodology).

Tests cover:
- validate_raster_format: missing file, format errors
- _pixel_categories: three-state pixel categorisation (outside / dry / flooded, perm / nonperm)
- check_lipad_overlap: geographic extent intersection check
- run_phase_a: BLOCKED / PARTIAL / READY suitability decisions with new gates
- Monotonicity violation detection
- Acquisition force protection and EE error handling
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box as shapely_box

from floodroute.hazard.jrc import (
    NODATA_SENTINEL,
    PhaseAReadiness,
    PixelCategories,
    ReadinessDecision,
    _depth_class,
    _pixel_categories,
    check_lipad_overlap,
    run_phase_a,
    validate_raster_format,
)

# ── Synthetic raster writers ────────────────────────────────────────────────

_W, _S, _PIX = 121.924, 10.639, 0.000808  # default extent for test rasters


def _write_depth_raster(
    path: Path,
    *,
    width: int = 20,
    height: int = 20,
    crs: str = "EPSG:4326",
    band_count: int = 5,
    pixel_size: float = _PIX,
    bands: list[np.ndarray | None] | None = None,
    west: float = _W,
    south: float = _S,
) -> Path:
    """Write a synthetic 5-band depth GeoTIFF for testing."""
    east = west + width * pixel_size
    north = south + height * pixel_size
    transform = from_bounds(west, south, east, north, width, height)

    def _fill(b: np.ndarray | None) -> np.ndarray:
        if b is not None:
            return np.array(b, dtype=np.float32).reshape(height, width)
        return np.full((height, width), NODATA_SENTINEL, dtype=np.float32)

    filled = [_fill(b) for b in (bands or [None] * 5)]
    while len(filled) < band_count:
        filled.append(filled[0].copy())
    filled = filled[:band_count]

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=band_count,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        for i, b in enumerate(filled, 1):
            dst.write(b, i)
    return path


def _write_cat_raster(
    path: Path,
    *,
    width: int = 20,
    height: int = 20,
    pixel_size: float = _PIX,
    west: float = _W,
    south: float = _S,
    rp10_cat: np.ndarray | None = None,
    rp100_cat: np.ndarray | None = None,
    perm_water: np.ndarray | None = None,
    spurious: np.ndarray | None = None,
) -> Path:
    """Write a synthetic 4-band depth-category GeoTIFF for testing."""
    east = west + width * pixel_size
    north = south + height * pixel_size
    transform = from_bounds(west, south, east, north, width, height)

    def _fill(arr: np.ndarray | None, fill_val: float = NODATA_SENTINEL) -> np.ndarray:
        if arr is not None:
            return np.array(arr, dtype=np.float32).reshape(height, width)
        return np.full((height, width), fill_val, dtype=np.float32)

    bands = [
        _fill(rp10_cat),  # band 1: RP10_depth_category
        _fill(rp100_cat),  # band 2: RP100_depth_category
        _fill(perm_water, -1.0),  # band 3: permanent_water_class  (-1 = not permanent water)
        _fill(spurious, -1.0),  # band 4: spurious_depth_category (-1 = not spurious)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i, b in enumerate(bands, 1):
            dst.write(b, i)
    return path


def _muni_geom() -> object:
    """Return a Shapely box matching the default 20×20 test raster extent."""
    east = _W + 20 * _PIX
    north = _S + 20 * _PIX
    return shapely_box(_W, _S, east, north)


def _mock_gpd_with_geom(geom: object) -> MagicMock:
    """Return a mock GeoDataFrame whose read_file returns the given geometry."""
    mock_row = MagicMock()
    mock_row.empty = False
    mock_row.geometry.values = [geom]
    mock_gdf = MagicMock()
    mock_gdf.__getitem__ = MagicMock(return_value=mock_row)
    return mock_gdf


def _run(
    tmp_path: Path,
    *,
    n_nonperm_rp10: int = 0,
    n_nonperm_rp100: int = 0,
    with_cat: bool = True,
    municipality_pcode: str = "PH0600616",
) -> PhaseAReadiness:
    """Build synthetic rasters and run Phase A with a mocked municipality polygon."""
    height = width = 20
    total = height * width

    # Depth raster — values only affect depth_stats (not the gate decisions)
    depth_path = _write_depth_raster(
        tmp_path / "depth.tif",
        width=width,
        height=height,
    )

    cat_path: Path | None = None
    if with_cat:
        rp10_cat = np.full(total, NODATA_SENTINEL, dtype=np.float32)
        rp10_cat[:n_nonperm_rp10] = 2.0
        rp100_cat = np.full(total, NODATA_SENTINEL, dtype=np.float32)
        rp100_cat[:n_nonperm_rp100] = 2.0
        # perm_water stays -1 everywhere → all flooded pixels are non-permanent
        cat_path = _write_cat_raster(
            tmp_path / "cat.tif",
            width=width,
            height=height,
            rp10_cat=rp10_cat,
            rp100_cat=rp100_cat,
        )

    admin_zip = tmp_path / "admin.zip"
    admin_zip.touch()

    with patch("floodroute.hazard.jrc.gpd") as mock_gpd:
        mock_gpd.read_file.return_value = _mock_gpd_with_geom(_muni_geom())
        return run_phase_a(
            depth_path,
            admin_zip=admin_zip,
            cat_path=cat_path,
            edges_path=None,
            municipality_pcode=municipality_pcode,
        )


# ── TestValidateRasterFormat ────────────────────────────────────────────────


class TestValidateRasterFormat:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_raster_format(tmp_path / "nonexistent.tif")

    def test_valid_raster_passes(self, tmp_path: Path) -> None:
        path = _write_depth_raster(tmp_path / "ok.tif")
        meta = validate_raster_format(path)
        assert meta["crs"] == "EPSG:4326"
        assert meta["band_count"] == 5

    def test_wrong_band_count_raises(self, tmp_path: Path) -> None:
        path = _write_depth_raster(tmp_path / "bad_bands.tif", band_count=3)
        with pytest.raises(ValueError, match="Expected 5 bands"):
            validate_raster_format(path)

    def test_wrong_crs_raises(self, tmp_path: Path) -> None:
        path = _write_depth_raster(tmp_path / "bad_crs.tif", crs="EPSG:3857")
        with pytest.raises(ValueError, match="Unexpected CRS"):
            validate_raster_format(path)

    def test_wrong_resolution_raises(self, tmp_path: Path) -> None:
        # pixel_size of 0.01 deg ≈ 1110 m — outside the 50–120 m window
        path = _write_depth_raster(tmp_path / "bad_res.tif", pixel_size=0.01)
        with pytest.raises(ValueError, match="Unexpected pixel size"):
            validate_raster_format(path)

    def test_utm_crs_accepted(self, tmp_path: Path) -> None:
        path = _write_depth_raster(tmp_path / "utm.tif", crs="EPSG:32651")
        meta = validate_raster_format(path)
        assert "32651" in meta["crs"]


# ── TestPixelCategories ─────────────────────────────────────────────────────


class TestPixelCategories:
    """Direct unit tests for _pixel_categories (three-state categorisation)."""

    def _make_arrays(
        self,
        h: int = 4,
        w: int = 5,
        *,
        n_flooded: int = 0,
        n_perm: int = 0,
        n_spurious: int = 0,
        dry_val: int = -1,
        flood_val: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (depth_data, cat_data, pw_band, sd_band) for _pixel_categories."""
        total = h * w
        cat_flat = np.full(total, NODATA_SENTINEL, dtype=np.float32)
        cat_flat[:n_flooded] = float(flood_val)
        # Mark the first n_perm flooded pixels as permanent water
        pw_flat = np.full(total, -1.0, dtype=np.float32)
        pw_flat[:n_perm] = 1.0
        sd_flat = np.full(total, -1.0, dtype=np.float32)
        sd_flat[:n_spurious] = 1.0

        depth_data = np.full((5, h, w), NODATA_SENTINEL, dtype=np.float32)
        cat_data = np.full((4, h, w), NODATA_SENTINEL, dtype=np.float32)
        cat_data[0] = cat_flat.reshape(h, w)

        return depth_data, cat_data, pw_flat.reshape(h, w), sd_flat.reshape(h, w)

    def _call(
        self,
        depth_data: np.ndarray,
        cat_data: np.ndarray,
        pw_band: np.ndarray,
        sd_band: np.ndarray,
        *,
        dry_val: int = -1,
        flood_vals: set[int] | None = None,
    ) -> PixelCategories:
        return _pixel_categories(
            depth_data,
            cat_data,
            pw_band,
            sd_band,
            return_period="RP10",
            depth_band_idx=0,
            cat_band_idx=0,
            dry_val=dry_val,
            flood_vals=flood_vals or {2},
        )

    def test_all_outside_domain(self) -> None:
        depth_data, cat_data, pw, sd = self._make_arrays(n_flooded=0)
        pc = self._call(depth_data, cat_data, pw, sd)
        assert pc.outside_domain == 20
        assert pc.flooded == 0
        assert pc.non_permanent == 0

    def test_flooded_nonperm_pixels_counted(self) -> None:
        depth_data, cat_data, pw, sd = self._make_arrays(n_flooded=8)
        pc = self._call(depth_data, cat_data, pw, sd)
        assert pc.flooded == 8
        assert pc.permanent_water == 0
        assert pc.non_permanent == 8

    def test_permanent_water_separated_from_nonperm(self) -> None:
        # 8 flooded total, 3 permanent water → 5 non-permanent
        depth_data, cat_data, pw, sd = self._make_arrays(n_flooded=8, n_perm=3)
        pc = self._call(depth_data, cat_data, pw, sd)
        assert pc.flooded == 8
        assert pc.permanent_water == 3
        assert pc.non_permanent == 5

    def test_spurious_counted_separately(self) -> None:
        depth_data, cat_data, pw, sd = self._make_arrays(n_flooded=5, n_spurious=2)
        pc = self._call(depth_data, cat_data, pw, sd)
        assert pc.spurious_flagged == 2

    def test_dry_val_zero_for_san_jose(self) -> None:
        """San Jose / Hamtic use dry_val=0 (not -1)."""
        h, w = 4, 5
        cat_data = np.full((4, h, w), NODATA_SENTINEL, dtype=np.float32)
        # Set some pixels to 0 (modelled dry in San Jose sub-area)
        cat_data[0, 0, :] = 0.0
        # Set some pixels to 2 (flooded)
        cat_data[0, 1, :] = 2.0
        depth_data = np.full((5, h, w), NODATA_SENTINEL, dtype=np.float32)
        pw = np.full((h, w), -1.0, dtype=np.float32)
        sd = np.full((h, w), -1.0, dtype=np.float32)

        pc = _pixel_categories(
            depth_data,
            cat_data,
            pw,
            sd,
            return_period="RP10",
            depth_band_idx=0,
            cat_band_idx=0,
            dry_val=0,
            flood_vals={2, 3},
        )
        assert pc.modelled_dry == 5
        assert pc.flooded == 5
        assert pc.non_permanent == 5

    def test_non_perm_pct_property(self) -> None:
        depth_data, cat_data, pw, sd = self._make_arrays(n_flooded=4)
        pc = self._call(depth_data, cat_data, pw, sd)
        # 4 non-perm / 20 total = 20%
        assert pc.non_perm_pct == pytest.approx(20.0)


# ── TestCheckLipadOverlap ───────────────────────────────────────────────────


class TestCheckLipadOverlap:
    """Tests for check_lipad_overlap — LiPAD ISO extent vs municipality geometry."""

    def test_no_overlap_returns_false_and_distance_note(self) -> None:
        # Mock municipality near the real Sibalom location
        muni = shapely_box(121.92, 10.63, 121.94, 10.66)
        # LiPAD bbox is east of Sibalom
        lipad_bbox = (122.32, 10.67, 122.43, 10.81)
        overlaps, notes = check_lipad_overlap(muni, lipad_bbox)
        assert overlaps is False
        assert "does NOT intersect" in notes
        assert "km" in notes

    def test_overlapping_returns_true_and_area_note(self) -> None:
        # Municipality that encloses the LiPAD bbox
        muni = shapely_box(122.30, 10.65, 122.45, 10.82)
        lipad_bbox = (122.32, 10.67, 122.43, 10.81)
        overlaps, notes = check_lipad_overlap(muni, lipad_bbox)
        assert overlaps is True
        assert "overlaps" in notes
        assert "km²" in notes


# ── TestReadinessDecision ───────────────────────────────────────────────────


class TestReadinessDecision:
    """BLOCKED / PARTIAL / READY decisions via run_phase_a with corrected gates."""

    def test_blocked_when_zero_nonperm_rp10(self, tmp_path: Path) -> None:
        # All category pixels are NODATA → zero non-permanent pixels at RP10
        result = _run(tmp_path, n_nonperm_rp10=0, n_nonperm_rp100=0)
        assert result.decision == ReadinessDecision.BLOCKED
        assert any("zero" in r.lower() or "non-permanent" in r.lower() for r in result.reasons)

    def test_blocked_without_cat_path_and_no_depth(self, tmp_path: Path) -> None:
        # No cat_path supplied → n_nonperm_10 = 0 → BLOCKED (no cat → partial degraded,
        # but since n_nonperm_10 == 0 triggers first)
        # When cat_path is absent, pixel_cats is empty → n_nonperm_10 = 0 → BLOCKED
        result = _run(tmp_path, with_cat=False)
        assert result.decision == ReadinessDecision.BLOCKED

    def test_partial_when_nonperm_pixels_below_gate(self, tmp_path: Path) -> None:
        # NON_PERM_MIN_PIXELS = 10; use 5 < 10
        result = _run(tmp_path, n_nonperm_rp10=5, n_nonperm_rp100=15)
        assert result.decision == ReadinessDecision.PARTIAL
        assert any("Non-permanent pixels" in r for r in result.reasons)

    def test_partial_when_scenario_gain_below_gate(self, tmp_path: Path) -> None:
        # SCENARIO_GAIN_MIN = 5; gain = 12 - 10 = 2 < 5
        result = _run(tmp_path, n_nonperm_rp10=10, n_nonperm_rp100=12)
        assert result.decision == ReadinessDecision.PARTIAL
        assert any("gain" in r.lower() or "RP10" in r for r in result.reasons)

    def test_ready_when_all_gates_met_without_edges(self, tmp_path: Path) -> None:
        # NON_PERM_MIN_PIXELS=10, SCENARIO_GAIN_MIN=5, NORMAL_SEG_MIN skipped (no edges)
        # Use 20 RP10 and 30 RP100: gain = 10 ≥ 5 ✓, pixels = 20 ≥ 10 ✓
        result = _run(tmp_path, n_nonperm_rp10=20, n_nonperm_rp100=30)
        assert result.decision == ReadinessDecision.READY
        assert result.reasons == []

    def test_partial_when_cat_path_absent_but_has_nonperm(self, tmp_path: Path) -> None:
        # No category raster → pixel_cats is empty → n_nonperm_10 = 0 → BLOCKED
        # This documents that cat_path is required for anything above BLOCKED
        result = _run(tmp_path, n_nonperm_rp10=50, n_nonperm_rp100=100, with_cat=False)
        assert result.decision == ReadinessDecision.BLOCKED

    def test_lipad_does_not_overlap_sibalom(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        assert result.lipad_overlaps_municipality is False
        assert "does NOT intersect" in result.lipad_notes

    def test_pixel_cats_populated_when_cat_provided(self, tmp_path: Path) -> None:
        result = _run(tmp_path, n_nonperm_rp10=15, n_nonperm_rp100=25)
        assert "RP10" in result.pixel_cats
        assert "RP100" in result.pixel_cats
        assert result.pixel_cats["RP10"].non_permanent == 15
        assert result.pixel_cats["RP100"].non_permanent == 25

    def test_municipality_code_stored(self, tmp_path: Path) -> None:
        result = _run(tmp_path, municipality_pcode="PH0600613")
        assert result.municipality_code == "PH0600613"

    def test_file_metadata_populated(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        assert result.file_bytes > 0
        assert len(result.sha256) == 64
        assert result.raster_crs == "EPSG:4326"
        assert result.pixel_size_m_approx > 0
        assert len(result.raster_bounds) == 4


# ── TestMonotonicity ────────────────────────────────────────────────────────


class TestMonotonicity:
    """Monotonicity violations are computed from depth bands when cat_path is provided."""

    def _run_mono(
        self,
        tmp_path: Path,
        rp10: np.ndarray,
        rp20: np.ndarray,
        rp100: np.ndarray,
    ) -> PhaseAReadiness:
        h, w = 4, 5
        depth_path = _write_depth_raster(
            tmp_path / "mono_depth.tif",
            height=h,
            width=w,
            bands=[rp10.flatten(), rp20.flatten(), rp100.flatten(), None, None],
        )
        # Provide a cat raster (all NODATA → BLOCKED, but monotonicity is still computed)
        cat_path = _write_cat_raster(tmp_path / "mono_cat.tif", height=h, width=w)
        admin_zip = tmp_path / "admin.zip"
        admin_zip.touch()

        with patch("floodroute.hazard.jrc.gpd") as mock_gpd:
            east = _W + w * _PIX
            north = _S + h * _PIX
            mock_gpd.read_file.return_value = _mock_gpd_with_geom(shapely_box(_W, _S, east, north))
            return run_phase_a(
                depth_path,
                admin_zip=admin_zip,
                cat_path=cat_path,
                edges_path=None,
                municipality_pcode="PH0600616",
            )

    def test_no_violations_for_monotonically_increasing_depths(self, tmp_path: Path) -> None:
        h, w = 4, 5
        rp10 = np.full((h, w), 0.5, dtype=np.float32)
        rp20 = np.full((h, w), 1.0, dtype=np.float32)
        rp100 = np.full((h, w), 1.5, dtype=np.float32)
        result = self._run_mono(tmp_path, rp10, rp20, rp100)
        assert result.monotonicity_violations_rp10_rp20 == 0
        assert result.monotonicity_violations_rp20_rp100 == 0

    def test_violations_detected_when_rp10_exceeds_rp20(self, tmp_path: Path) -> None:
        h, w = 4, 5
        # RP10 > RP20 everywhere — all 20 pixels are violations
        rp10 = np.full((h, w), 2.0, dtype=np.float32)
        rp20 = np.full((h, w), 1.0, dtype=np.float32)
        rp100 = np.full((h, w), 1.5, dtype=np.float32)
        result = self._run_mono(tmp_path, rp10, rp20, rp100)
        assert result.monotonicity_violations_rp10_rp20 == h * w

    def test_violations_detected_when_rp20_exceeds_rp100(self, tmp_path: Path) -> None:
        h, w = 4, 5
        rp10 = np.full((h, w), 0.5, dtype=np.float32)
        rp20 = np.full((h, w), 2.0, dtype=np.float32)
        rp100 = np.full((h, w), 1.0, dtype=np.float32)
        result = self._run_mono(tmp_path, rp10, rp20, rp100)
        assert result.monotonicity_violations_rp20_rp100 == h * w

    def test_marginal_difference_below_tolerance_not_violation(self, tmp_path: Path) -> None:
        h, w = 4, 5
        # Difference of 0.0005 m is below the 0.001 tolerance
        rp10 = np.full((h, w), 1.0005, dtype=np.float32)
        rp20 = np.full((h, w), 1.0, dtype=np.float32)
        rp100 = np.full((h, w), 1.5, dtype=np.float32)
        result = self._run_mono(tmp_path, rp10, rp20, rp100)
        assert result.monotonicity_violations_rp10_rp20 == 0


# ── TestAcquisitionForceProtection ─────────────────────────────────────────


class TestAcquisitionForceProtection:
    def test_raises_if_file_exists_and_no_force(self, tmp_path: Path) -> None:
        from floodroute.hazard.jrc import acquire_sibalom_subset

        out = tmp_path / "existing.tif"
        out.touch()
        with pytest.raises(FileExistsError, match="force=True"):
            acquire_sibalom_subset(
                out,
                ee_project="dummy",
                admin_zip=tmp_path / "admin.zip",
                force=False,
            )

    def test_missing_ee_raises_import_error(self, tmp_path: Path) -> None:
        from floodroute.hazard.jrc import acquire_sibalom_subset

        out = tmp_path / "out.tif"
        with (
            patch.dict(sys.modules, {"ee": None, "requests": None}),
            pytest.raises((ImportError, TypeError)),
        ):
            acquire_sibalom_subset(
                out,
                ee_project="dummy",
                admin_zip=tmp_path / "admin.zip",
                force=True,
            )

    def test_bad_ee_project_raises_runtime_error(self, tmp_path: Path) -> None:
        from floodroute.hazard.jrc import acquire_sibalom_subset

        out = tmp_path / "out.tif"
        mock_ee = MagicMock()
        mock_ee.Initialize.side_effect = RuntimeError("Auth failed")

        with (
            patch.dict(sys.modules, {"ee": mock_ee, "requests": MagicMock()}),
            pytest.raises(RuntimeError, match="Earth Engine initialization failed"),
        ):
            acquire_sibalom_subset(
                out,
                ee_project="bad-project",
                admin_zip=tmp_path / "admin.zip",
                force=True,
            )

    def test_acquire_municipal_subset_delegates_to_acquire(self, tmp_path: Path) -> None:
        """acquire_municipal_subset with force=False raises FileExistsError if file exists."""
        from floodroute.hazard.jrc import acquire_municipal_subset

        out = tmp_path / "muni.tif"
        out.touch()
        with pytest.raises(FileExistsError, match="force=True"):
            acquire_municipal_subset(
                "PH0600613",
                out,
                ee_project="dummy",
                admin_zip=tmp_path / "admin.zip",
                force=False,
            )


# ── TestDepthClass ───────────────────────────────────────────────────────────


class TestDepthClass:
    @pytest.mark.parametrize(
        "depth_m, expected",
        [
            (0.000, "no_modeled_inundation"),  # below threshold
            (0.099, "no_modeled_inundation"),  # just below 0.10
            (0.100, "low"),  # exactly 0.10 — low boundary
            (0.499, "low"),  # just below 0.50
            (0.500, "medium"),  # exactly 0.50 — medium boundary
            (1.500, "medium"),  # exactly 1.50 — still medium (inclusive)
            (1.501, "high"),  # strictly above 1.50 — high
        ],
    )
    def test_depth_class_boundaries(self, depth_m: float, expected: str) -> None:
        assert _depth_class(depth_m) == expected
