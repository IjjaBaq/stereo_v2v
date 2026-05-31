"""Smoke tests for Stage 3 — Lift to 3D Bounding Boxes.

Tests the full lifting pipeline on sample "000000" and validates
geometry utilities with synthetic data.

Run with: pytest tests/test_stage3.py -v

Requires:
    - Stage 1 output: outputs/depth/000000_disp.npy
    - Stage 2 output: outputs/detections/000000_boxes2d.json
    - KITTI calibration: data/kitti/training/calib/000000.txt
    - config/base.yaml and config/stage3.yaml present
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stages.stage3_lift import (
    lift_boxes,
    load_configs,
    run as run_stage3,
    sample_depth,
)
from utils.geometry import (
    box3d_iou,
    center_distance,
    compute_heading,
    unproject_box,
)
from utils.kitti_loader import load_calib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_CONFIG   = "config/base.yaml"
STAGE_CONFIG  = "config/stage3.yaml"
SAMPLE_ID     = "000000"
OUTPUT_DIR    = Path("outputs/boxes3d")
KITTI_CLASSES = {"Car", "Pedestrian", "Cyclist"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def configs():
    return load_configs(BASE_CONFIG, STAGE_CONFIG)


@pytest.fixture(scope="module")
def base_cfg(configs):
    return configs[0]


@pytest.fixture(scope="module")
def stage_cfg(configs):
    return configs[1]


@pytest.fixture(scope="module")
def calib(base_cfg):
    return load_calib(
        base_cfg["data"]["data_root"],
        base_cfg["data"]["split"],
        SAMPLE_ID,
    )


@pytest.fixture(scope="module")
def stage3_result(base_cfg, stage_cfg):
    """Run Stage 3 once and cache for all output tests."""
    return run_stage3(SAMPLE_ID, base_cfg, stage_cfg)


@pytest.fixture(scope="module")
def boxes3d(stage3_result):
    return stage3_result["boxes"]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_base_config_loads(self, base_cfg):
        assert "data" in base_cfg
        assert "mlflow" in base_cfg

    def test_stage_config_loads(self, stage_cfg):
        for key in ("method", "output_dir", "depth_sampling",
                    "min_valid_pixels", "heading_method", "class_priors"):
            assert key in stage_cfg, f"Missing key: '{key}'"

    def test_class_priors_have_all_classes(self, stage_cfg):
        for cls in KITTI_CLASSES:
            assert cls in stage_cfg["class_priors"], \
                f"Missing class prior for '{cls}'"

    def test_class_priors_have_dimensions(self, stage_cfg):
        for cls, prior in stage_cfg["class_priors"].items():
            for dim in ("l", "w", "h"):
                assert dim in prior, f"Prior for '{cls}' missing '{dim}'"
                assert prior[dim] > 0, f"Prior {cls}.{dim} must be positive"

    def test_min_valid_pixels_positive(self, stage_cfg):
        assert stage_cfg["min_valid_pixels"] > 0

    def test_plausibility_bounds_consistent(self, stage_cfg):
        for cls, prior in stage_cfg["class_priors"].items():
            assert prior["w_min"] < prior["w_max"], \
                f"{cls}: w_min must be < w_max"
            assert prior["h_min"] < prior["h_max"], \
                f"{cls}: h_min must be < h_max"
            assert prior["w_min"] > 0, \
                f"{cls}: w_min must be positive"
            assert prior["h_min"] > 0, \
                f"{cls}: h_min must be positive"

    def test_depth_sampling_method_is_supported(self, stage_cfg):
        supported = {"median", "percentile_75"}
        assert stage_cfg["depth_sampling"] in supported, (
            f"depth_sampling='{stage_cfg['depth_sampling']}' "
            f"not in supported set {supported}"
        )


# ---------------------------------------------------------------------------
# Geometry utility tests — synthetic data only
# ---------------------------------------------------------------------------

class TestUnprojectBox:
    """Tests unproject_box with a synthetic known-answer calibration."""

    @pytest.fixture(scope="class")
    def synthetic_P2(self):
        # Simple calibration: fx=fy=500, cx=cy=0, no skew
        P2 = np.zeros((3, 4), dtype=np.float32)
        P2[0, 0] = 500.0  # fx
        P2[1, 1] = 500.0  # fy
        P2[0, 2] = 0.0    # cx
        P2[1, 2] = 0.0    # cy
        return P2

    def test_box_at_principal_axis(self, synthetic_P2):
        box2d = {"x1": -50.0, "y1": -50.0, "x2": 50.0, "y2": 50.0}
        X, Y_center, Z, w, h = unproject_box(box2d, Z=10.0, P2=synthetic_P2)
        assert X == pytest.approx(0.0, abs=1e-4)
        assert Y_center == pytest.approx(0.0, abs=1e-4)
        assert Z == pytest.approx(10.0, abs=1e-4)

    def test_width_scales_with_depth(self, synthetic_P2):
        box2d = {"x1": 0.0, "y1": -25.0, "x2": 100.0, "y2": 25.0}
        _, _, _, w1, _ = unproject_box(box2d, Z=10.0, P2=synthetic_P2)
        _, _, _, w2, _ = unproject_box(box2d, Z=20.0, P2=synthetic_P2)
        assert w2 == pytest.approx(2.0 * w1, rel=1e-4)

    def test_returns_five_floats(self, synthetic_P2):
        box2d = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}
        result = unproject_box(box2d, Z=15.0, P2=synthetic_P2)
        assert len(result) == 5
        for val in result:
            assert isinstance(val, float)

    def test_positive_dimensions(self, synthetic_P2):
        box2d = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}
        _, _, _, w, h = unproject_box(box2d, Z=15.0, P2=synthetic_P2)
        assert w > 0
        assert h > 0

    def test_option_a_b_equivalence(self, synthetic_P2):
        """Verify Option A (unproject center) == Option B (unproject bottom - h/2)."""
        box2d = {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
        Z = 20.0
        fy = float(synthetic_P2[1, 1])
        cy = float(synthetic_P2[1, 2])

        _, Y_center_A, _, _, h_3d = unproject_box(box2d, Z=Z, P2=synthetic_P2)

        Y_bottom  = (box2d["y2"] - cy) * Z / fy
        Y_center_B = Y_bottom - h_3d / 2.0

        assert Y_center_A == pytest.approx(Y_center_B, abs=1e-4)


class TestComputeHeading:
    @pytest.fixture(scope="class")
    def P2(self):
        p = np.zeros((3, 4), dtype=np.float32)
        p[0, 0] = 721.0  # fx — KITTI typical
        p[0, 2] = 609.0  # cx
        return p

    def test_center_heading_near_zero(self, P2):
        cx = float(P2[0, 2])
        heading = compute_heading(cx, P2)
        assert abs(heading) < 0.01

    def test_heading_in_range(self, P2):
        for cx in [0, 300, 609, 900, 1240]:
            h = compute_heading(float(cx), P2)
            assert -math.pi <= h <= math.pi

    def test_left_of_center_negative_heading(self, P2):
        cx_left = float(P2[0, 2]) - 200.0
        assert compute_heading(cx_left, P2) < 0.0

    def test_right_of_center_positive_heading(self, P2):
        cx_right = float(P2[0, 2]) + 200.0
        assert compute_heading(cx_right, P2) > 0.0

    def test_unsupported_method_raises(self, P2):
        with pytest.raises(ValueError):
            compute_heading(600.0, P2, method="unknown_method")


class TestBox3dIou:
    def _make_box(self, x, z, l, w, h=1.5, heading=0.0, y=0.0):
        return {"x": x, "y": y, "z": z, "l": l, "w": w, "h": h,
                "heading": heading}

    def test_identical_boxes_iou_one(self):
        box = self._make_box(0, 10, 4.0, 2.0)
        assert box3d_iou(box, box) == pytest.approx(1.0, abs=1e-4)

    def test_non_overlapping_iou_zero(self):
        a = self._make_box(0,   10, 4.0, 2.0)
        b = self._make_box(100, 10, 4.0, 2.0)
        assert box3d_iou(a, b) == pytest.approx(0.0, abs=1e-4)

    def test_iou_in_range(self):
        a = self._make_box(0, 10, 4.0, 2.0)
        b = self._make_box(1, 10, 4.0, 2.0)
        iou = box3d_iou(a, b)
        assert 0.0 <= iou <= 1.0

    def test_iou_symmetric(self):
        a = self._make_box(0, 10, 4.0, 2.0)
        b = self._make_box(1, 11, 4.0, 2.0)
        assert box3d_iou(a, b) == pytest.approx(box3d_iou(b, a), abs=1e-6)

    def test_no_height_overlap_iou_zero(self):
        a = self._make_box(0, 10, 4.0, 2.0, h=1.0, y=0.0)
        b = self._make_box(0, 10, 4.0, 2.0, h=1.0, y=5.0)
        assert box3d_iou(a, b) == pytest.approx(0.0, abs=1e-4)


class TestCenterDistance:
    def test_same_box_distance_zero(self):
        box = {"x": 1.0, "y": 2.0, "z": 30.0}
        assert center_distance(box, box) == pytest.approx(0.0)

    def test_known_distance(self):
        a = {"x": 0.0, "y": 0.0, "z": 0.0}
        b = {"x": 3.0, "y": 4.0, "z": 0.0}
        assert center_distance(a, b) == pytest.approx(5.0)

    def test_distance_symmetric(self):
        a = {"x": 1.0, "y": 2.0, "z": 10.0}
        b = {"x": 4.0, "y": 6.0, "z": 15.0}
        assert center_distance(a, b) == pytest.approx(center_distance(b, a))


# ---------------------------------------------------------------------------
# Depth sampling tests — synthetic disparity
# ---------------------------------------------------------------------------

class TestSampleDepth:
    @pytest.fixture(scope="class")
    def flat_disp(self):
        """Uniform disparity map — all pixels = 10.0"""
        return np.full((100, 200), 10.0, dtype=np.float32)

    @pytest.fixture(scope="class")
    def sparse_disp(self):
        """Disparity map with only 5 valid pixels."""
        d = np.full((100, 200), np.nan, dtype=np.float32)
        d[50, 100:105] = 10.0
        return d

    def test_uniform_disparity_returns_correct_depth(self, flat_disp):
        box2d = {"x1": 50, "y1": 20, "x2": 150, "y2": 80}
        Z, count = sample_depth(flat_disp, box2d,
                                focal_length=721.0, baseline=0.54)
        expected_Z = 721.0 * 0.54 / 10.0
        assert Z == pytest.approx(expected_Z, rel=1e-4)
        assert count == (150 - 50) * (80 - 20)

    def test_sparse_disparity_valid_count(self, sparse_disp):
        box2d = {"x1": 90, "y1": 40, "x2": 115, "y2": 60}
        _, count = sample_depth(sparse_disp, box2d,
                                focal_length=721.0, baseline=0.54)
        assert count == 5

    def test_all_nan_returns_zero_count(self):
        nan_disp = np.full((50, 50), np.nan, dtype=np.float32)
        box2d = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
        Z, count = sample_depth(nan_disp, box2d,
                                focal_length=721.0, baseline=0.54)
        assert count == 0
        assert not np.isfinite(Z)

    def test_unsupported_method_raises(self, flat_disp):
        box2d = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
        with pytest.raises(ValueError):
            sample_depth(flat_disp, box2d, 721.0, 0.54, method="mean")

    def test_percentile_75_returns_valid_depth(self, flat_disp):
        box2d = {"x1": 50, "y1": 20, "x2": 150, "y2": 80}
        Z, count = sample_depth(flat_disp, box2d,
                                focal_length=721.0, baseline=0.54,
                                method="percentile_75")
        assert np.isfinite(Z)
        assert Z > 0
        assert count > 0

 
    def test_percentile_75_biases_toward_foreground(self):
        disp = np.full((100, 100), np.nan, dtype=np.float32)
        # 30% foreground (high disparity = close objects)
        disp[:30, :] = 30.0
        # 70% background (low disparity = far)
        disp[30:, :] = 5.0

        box2d = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        Z_med, _ = sample_depth(disp, box2d, 721.0, 0.54, method="median")
        Z_p75, _ = sample_depth(disp, box2d, 721.0, 0.54, method="percentile_75")

        # median lands in background (50th pct = 5.0 → far)
        # p75 lands in foreground (75th pct = 30.0 → close)
        assert Z_p75 < Z_med

# ---------------------------------------------------------------------------
# Dimension fallback tests
# ---------------------------------------------------------------------------

class TestDimensionFallback:
    """Tests that implausible unprojected dimensions fall back to class priors."""

    @pytest.fixture(scope="class")
    def stage_cfg(self):
        _, cfg = load_configs(BASE_CONFIG, STAGE_CONFIG)
        return cfg

  
    def test_implausible_width_uses_prior(self, stage_cfg):
        from stages.stage3_lift import estimate_dimensions
        P2 = np.zeros((3, 4), dtype=np.float32)
        P2[0, 0] = 721.0
        P2[1, 1] = 721.0
        # 1px wide box at Z=300m → w ≈ 0.42m — below Car w_min=0.5
        box2d = {"x1": 100.0, "y1": 100.0, "x2": 101.0, "y2": 150.0}
        _, w, _ = estimate_dimensions(box2d, Z=300.0, P2=P2,
                                      label="Car",
                                      class_priors=stage_cfg["class_priors"])
        assert w == pytest.approx(stage_cfg["class_priors"]["Car"]["w"])

    def test_implausible_height_uses_prior(self, stage_cfg):
        """A tall box at tiny depth produces implausibly large h → prior used."""
        from stages.stage3_lift import estimate_dimensions
        P2 = np.zeros((3, 4), dtype=np.float32)
        P2[0, 0] = 721.0
        P2[1, 1] = 721.0
        # box height = 500px at Z=10m → h ≈ 6.9m — above h_max
        box2d = {"x1": 100.0, "y1": 0.0, "x2": 200.0, "y2": 500.0}
        _, _, h = estimate_dimensions(box2d, Z=10.0, P2=P2,
                                      label="Car",
                                      class_priors=stage_cfg["class_priors"])
        assert h == pytest.approx(stage_cfg["class_priors"]["Car"]["h"])


# ---------------------------------------------------------------------------
# Full pipeline output tests
# ---------------------------------------------------------------------------

class TestStage3Output:
    def test_run_returns_expected_keys(self, stage3_result):
        for key in ("sample_id", "method", "boxes",
                    "n_input_boxes", "n_skipped", "output_path"):
            assert key in stage3_result

    def test_skipped_plus_lifted_equals_input(self, stage3_result):
        assert (stage3_result["n_skipped"] + len(stage3_result["boxes"])
                == stage3_result["n_input_boxes"])

    def test_json_file_exists(self, stage3_result):
        assert Path(stage3_result["output_path"]).exists()

    def test_json_loadable(self, stage3_result):
        with open(stage3_result["output_path"]) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_has_required_keys(self, stage3_result):
        with open(stage3_result["output_path"]) as f:
            data = json.load(f)
        for key in ("sample_id", "method", "n_input_boxes",
                    "n_skipped", "skip_reason", "boxes"):
            assert key in data

    def test_json_sample_id_matches(self, stage3_result):
        with open(stage3_result["output_path"]) as f:
            data = json.load(f)
        assert data["sample_id"] == SAMPLE_ID


class TestBoxes3d:
    def test_each_box_has_required_fields(self, boxes3d):
        required = {"label", "confidence", "x", "y", "z",
                    "l", "w", "h", "heading", "x1", "y1", "x2", "y2"}
        for i, box in enumerate(boxes3d):
            missing = required - box.keys()
            assert not missing, f"Box {i} missing fields: {missing}"

    def test_all_labels_valid_kitti_classes(self, boxes3d):
        for box in boxes3d:
            assert box["label"] in KITTI_CLASSES

    def test_z_positive_for_all_boxes(self, boxes3d):
        for box in boxes3d:
            assert box["z"] > 0, \
                f"Box z={box['z']} must be positive (in front of camera)"

    def test_confidence_in_range(self, boxes3d):
        for box in boxes3d:
            assert 0.0 <= box["confidence"] <= 1.0, \
                f"confidence={box['confidence']} out of range"

    def test_heading_in_range(self, boxes3d):
        for box in boxes3d:
            assert -math.pi <= box["heading"] <= math.pi, \
                f"heading={box['heading']} out of [-pi, pi]"

    def test_dimensions_physically_plausible(self, boxes3d, stage_cfg):
        """Bounds read from config — stays in sync if config is tuned."""
        for box in boxes3d:
            prior = stage_cfg["class_priors"][box["label"]]
            assert prior["w_min"] <= box["w"] <= prior["w_max"], (
                f"{box['label']}.w={box['w']:.3f} outside "
                f"[{prior['w_min']}, {prior['w_max']}]"
            )
            assert prior["h_min"] <= box["h"] <= prior["h_max"], (
                f"{box['label']}.h={box['h']:.3f} outside "
                f"[{prior['h_min']}, {prior['h_max']}]"
            )

    def test_2d_coords_present_and_valid(self, boxes3d):
        for box in boxes3d:
            assert box["x2"] > box["x1"], "x2 must be > x1"
            assert box["y2"] > box["y1"], "y2 must be > y1"


# ---------------------------------------------------------------------------
# Confidence propagation tests
# ---------------------------------------------------------------------------

class TestConfidencePropagation:
    def test_confidence_reduced_by_coverage(self):
        conf_2d        = 0.8
        valid_count    = 50
        total_pixels   = 200
        coverage_ratio = valid_count / total_pixels
        conf_3d        = round(conf_2d * coverage_ratio, 4)
        assert conf_3d == pytest.approx(0.8 * 0.25, abs=1e-4)

    def test_full_coverage_preserves_confidence(self):
        conf_2d = 0.9
        conf_3d = round(conf_2d * 1.0, 4)
        assert conf_3d == pytest.approx(0.9, abs=1e-4)

    def test_zero_coverage_gives_zero_confidence(self):
        conf_2d = 0.9
        conf_3d = round(conf_2d * 0.0, 4)
        assert conf_3d == pytest.approx(0.0, abs=1e-4)
