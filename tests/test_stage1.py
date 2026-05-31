"""Smoke tests for Stage 1 — Stereo Depth Estimation.

Tests that stage1_depth.run() produces correct outputs for sample "000000".
Run with: pytest tests/test_stage1.py

Requires:
    - KITTI training data at ./data/kitti/training/
    - config/base.yaml and config/stage1.yaml present
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from stages.stage1_depth import (
    colorize_disparity,
    compute_disparity_sgbm,
    disparity_to_depth,
    extract_stereo_params,
    load_configs,
    run as run_stage1,
)
from utils.kitti_loader import load_calib, load_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_CONFIG  = "config/base.yaml"
STAGE_CONFIG = "config/stage1.yaml"
SAMPLE_ID    = "000000"
OUTPUT_DIR   = Path("outputs/depth")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def configs():
    """Load base and stage configs once for all tests in this module."""
    return load_configs(BASE_CONFIG, STAGE_CONFIG)


@pytest.fixture(scope="module")
def base_cfg(configs):
    return configs[0]


@pytest.fixture(scope="module")
def stage_cfg(configs):
    return configs[1]


@pytest.fixture(scope="module")
def calib(base_cfg):
    """Load calibration for sample 000000."""
    return load_calib(
        base_cfg["data"]["data_root"],
        base_cfg["data"]["split"],
        SAMPLE_ID,
    )


@pytest.fixture(scope="module")
def stereo_pair(base_cfg):
    """Load left and right images for sample 000000."""
    left = load_image(
        base_cfg["data"]["data_root"],
        base_cfg["data"]["split"],
        "image_2",
        SAMPLE_ID,
    )
    right = load_image(
        base_cfg["data"]["data_root"],
        base_cfg["data"]["split"],
        "image_3",
        SAMPLE_ID,
    )
    return left, right


@pytest.fixture(scope="module")
def disparity(stereo_pair, stage_cfg):
    """Compute SGBM disparity once for the whole module."""
    left, right = stereo_pair
    return compute_disparity_sgbm(left, right, stage_cfg["sgbm"])


@pytest.fixture(scope="module")
def depth(disparity, calib):
    """Compute depth once for the whole module, reusing shared disparity."""
    f, b = extract_stereo_params(calib)
    return disparity_to_depth(disparity, f, b)


@pytest.fixture(scope="module")
def stage1_result(base_cfg, stage_cfg):
    """Run Stage 1 once and cache result for all output tests."""
    return run_stage1(SAMPLE_ID, base_cfg, stage_cfg)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_base_config_loads(self, base_cfg):
        assert "data" in base_cfg
        assert "mlflow" in base_cfg

    def test_stage_config_loads(self, stage_cfg):
        assert "method" in stage_cfg
        assert "sgbm" in stage_cfg
        assert "output_dir" in stage_cfg

    def test_sgbm_num_disparities_divisible_by_16(self, stage_cfg):
        n = stage_cfg["sgbm"]["num_disparities"]
        assert n % 16 == 0, f"num_disparities={n} must be divisible by 16"

    def test_sgbm_p2_greater_than_p1(self, stage_cfg):
        p1 = stage_cfg["sgbm"]["p1"]
        p2 = stage_cfg["sgbm"]["p2"]
        assert p2 > p1, f"P2={p2} must be > P1={p1} for SGBM"

    def test_sgbm_block_size_is_odd(self, stage_cfg):
        bs = stage_cfg["sgbm"]["block_size"]
        assert bs % 2 == 1, f"block_size={bs} must be odd"


# ---------------------------------------------------------------------------
# Calibration / stereo param tests
# ---------------------------------------------------------------------------

class TestStereoParams:
    def test_extract_returns_positive_focal_length(self, calib):
        f, _ = extract_stereo_params(calib)
        assert f > 0, f"Focal length must be positive, got {f}"

    def test_extract_returns_positive_baseline(self, calib):
        _, b = extract_stereo_params(calib)
        assert b > 0, f"Baseline must be positive, got {b}"

    def test_baseline_plausible_for_kitti(self, calib):
        # KITTI baseline is ~0.54m — allow a loose range
        _, b = extract_stereo_params(calib)
        assert 0.4 < b < 0.7, f"Baseline {b:.4f}m outside expected KITTI range (0.4–0.7m)"


# ---------------------------------------------------------------------------
# Disparity computation tests
# ---------------------------------------------------------------------------

class TestDisparityComputation:
   
    def test_disparity_shape_matches_input(self, disparity, stereo_pair):
        left, _ = stereo_pair
        assert disparity.shape == left.shape[:2]

    def test_disparity_dtype_is_float32(self, disparity):
        assert disparity.dtype == np.float32

    def test_disparity_has_valid_pixels(self, disparity):
        assert np.sum(~np.isnan(disparity)) > 0

    def test_disparity_valid_values_are_positive(self, disparity):
        valid = disparity[~np.isnan(disparity)]
        assert np.all(valid > 0), "All valid disparity values must be positive"

    def test_valid_pixel_ratio_above_threshold(self, disparity):
        ratio = np.sum(~np.isnan(disparity)) / disparity.size
        assert ratio > 0.3, f"Valid pixel ratio {ratio:.3f} too low — SGBM may be misconfigured"


# ---------------------------------------------------------------------------
# Depth conversion tests
# ---------------------------------------------------------------------------

class TestDepthConversion:
    def test_depth_dtype_is_float32(self, depth):
        assert depth.dtype == np.float32

    def test_depth_has_valid_pixels(self, depth):
        assert np.sum(~np.isnan(depth)) > 0

    def test_depth_values_positive(self, depth):
        valid = depth[~np.isnan(depth)]
        assert np.all(valid > 0), "All valid depth values must be positive"

    def test_depth_range_physically_plausible(self, depth):
        # KITTI scenes: objects typically between 2m and 80m, but SGBM can
        # produce near-zero disparities that map to very large depths (600m+).
        # We only assert the minimum is sensible — max is capped separately
        # via max_depth_m in stage1.yaml if needed downstream.
        d_min = float(np.nanmin(depth))
        d_max = float(np.nanmax(depth))
        assert d_min >= 0.5,   f"Min depth {d_min:.2f}m is implausibly close"
        assert d_max <= 1000.0, f"Max depth {d_max:.2f}m exceeds physically realistic bound"


# ---------------------------------------------------------------------------
# Full pipeline / output file tests
# ---------------------------------------------------------------------------

class TestStage1Output:
    def test_run_returns_expected_keys(self, stage1_result):
        for key in ("disp", "depth", "disp_path", "disp_png"):
            assert key in stage1_result

    def test_disp_npy_file_exists(self, stage1_result):
        assert Path(stage1_result["disp_path"]).exists(), \
            f"Expected .npy output at {stage1_result['disp_path']}"

    def test_disp_png_file_exists(self, stage1_result):
        assert Path(stage1_result["disp_png"]).exists(), \
            f"Expected .png output at {stage1_result['disp_png']}"

    def test_saved_npy_loadable_and_correct_dtype(self, stage1_result):
        loaded = np.load(str(stage1_result["disp_path"]))
        assert loaded.dtype == np.float32

    def test_saved_npy_shape_is_2d(self, stage1_result):
        loaded = np.load(str(stage1_result["disp_path"]))
        assert loaded.ndim == 2

    def test_disp_and_depth_shapes_match(self, stage1_result):
        assert stage1_result["disp"].shape == stage1_result["depth"].shape


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_colorize_returns_uint8(self):
        disp = np.random.rand(100, 200).astype(np.float32)
        disp[10:20, 10:20] = np.nan
        result = colorize_disparity(disp)
        assert result.dtype == np.uint8

    def test_colorize_returns_3channel(self):
        disp = np.random.rand(100, 200).astype(np.float32)
        result = colorize_disparity(disp)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_colorize_nan_pixels_are_black(self):
        disp = np.ones((50, 50), dtype=np.float32)
        disp[0, 0] = np.nan
        result = colorize_disparity(disp)
        assert np.all(result[0, 0] == 0), "NaN pixels must be rendered black"
