"""Smoke tests for Stage 2 — 2D Object Detection.

Tests RT-DETR detection pipeline on sample "000000".
Model is loaded once per session and reused across all tests.

Run with: pytest tests/test_stage2.py -v

Requires:
    - KITTI training data at ./data/kitti/training/image_2/000000.png
    - config/base.yaml and config/stage2.yaml present
    - Model weights in ./models/ (downloaded on first run)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from stages.stage2_detect import (
    build_coco_to_kitti_map,
    detect,
    load_configs,
    load_model,
    run as run_stage2,
)
from stages.validate_stage2_detect import compute_ap
from utils.geometry import box_iou
from utils.kitti_loader import load_image, load_labels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_CONFIG  = "config/base.yaml"
STAGE_CONFIG = "config/stage2.yaml"
SAMPLE_ID    = "000000"
OUTPUT_DIR   = Path("outputs/detections")
KITTI_CLASSES = {"Car", "Pedestrian", "Cyclist"}

# ---------------------------------------------------------------------------
# Session-scoped fixtures — model loaded once for entire test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def configs():
    return load_configs(BASE_CONFIG, STAGE_CONFIG)


@pytest.fixture(scope="session")
def base_cfg(configs):
    return configs[0]


@pytest.fixture(scope="session")
def stage_cfg(configs):
    return configs[1]


@pytest.fixture(scope="session")
def model_and_processor(stage_cfg):
    """Load RT-DETR once for the entire test session."""
    processor, model = load_model(stage_cfg["model"])
    return processor, model


@pytest.fixture(scope="session")
def processor(model_and_processor):
    return model_and_processor[0]


@pytest.fixture(scope="session")
def model(model_and_processor):
    return model_and_processor[1]


@pytest.fixture(scope="session")
def coco_to_kitti(model, stage_cfg):
    return build_coco_to_kitti_map(stage_cfg["class_mapping"], model)


@pytest.fixture(scope="session")
def left_image(base_cfg):
    return load_image(
        base_cfg["data"]["data_root"],
        base_cfg["data"]["split"],
        "image_2",
        SAMPLE_ID,
        suffix=".png",
    )


@pytest.fixture(scope="session")
def gt_labels(base_cfg):
    return load_labels(
        base_cfg["data"]["data_root"],
        base_cfg["data"]["split"],
        SAMPLE_ID,
    )


@pytest.fixture(scope="session")
def detections(left_image, processor, model, coco_to_kitti, stage_cfg):
    """Run detection once and reuse across all tests."""
    return detect(
        left_image,
        processor,
        model,
        coco_to_kitti,
        confidence_threshold=stage_cfg["model"]["confidence_threshold"],
    )


@pytest.fixture(scope="session")
def stage2_result(base_cfg, stage_cfg, processor, model):
    """Run full Stage 2 pipeline once and cache result."""
    return run_stage2(SAMPLE_ID, base_cfg, stage_cfg, processor, model)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_base_config_loads(self, base_cfg):
        assert "data" in base_cfg
        assert "mlflow" in base_cfg

    def test_stage_config_loads(self, stage_cfg):
        assert "method" in stage_cfg
        assert "model" in stage_cfg
        assert "class_mapping" in stage_cfg
        assert "output_dir" in stage_cfg

    def test_model_name_present(self, stage_cfg):
        assert "name" in stage_cfg["model"]
        assert len(stage_cfg["model"]["name"]) > 0

    def test_cache_dir_present(self, stage_cfg):
        assert "cache_dir" in stage_cfg["model"]

    def test_confidence_threshold_in_range(self, stage_cfg):
        thresh = stage_cfg["model"]["confidence_threshold"]
        assert 0.0 < thresh < 1.0, f"confidence_threshold={thresh} must be in (0, 1)"

    def test_class_mapping_targets_valid_kitti_classes(self, stage_cfg):
        for coco_cls, kitti_cls in stage_cfg["class_mapping"].items():
            assert kitti_cls in KITTI_CLASSES, (
                f"'{coco_cls}' maps to '{kitti_cls}' which is not a valid KITTI class"
            )


# ---------------------------------------------------------------------------
# Class mapping tests
# ---------------------------------------------------------------------------

class TestClassMapping:
    def test_map_is_not_empty(self, coco_to_kitti):
        assert len(coco_to_kitti) > 0

    def test_all_values_are_kitti_classes(self, coco_to_kitti):
        for idx, kitti_cls in coco_to_kitti.items():
            assert kitti_cls in KITTI_CLASSES, (
                f"COCO idx {idx} mapped to unknown KITTI class '{kitti_cls}'"
            )

    def test_all_keys_are_integers(self, coco_to_kitti):
        for k in coco_to_kitti:
            assert isinstance(k, int)


# ---------------------------------------------------------------------------
# Detection output tests
# ---------------------------------------------------------------------------

class TestDetections:
    def test_returns_a_list(self, detections):
        assert isinstance(detections, list)

    def test_each_box_has_required_fields(self, detections):
        required = {"label", "confidence", "x1", "y1", "x2", "y2"}
        for i, box in enumerate(detections):
            missing = required - box.keys()
            assert not missing, f"Box {i} missing fields: {missing}"

    def test_all_labels_are_valid_kitti_classes(self, detections):
        for box in detections:
            assert box["label"] in KITTI_CLASSES, (
                f"Unexpected label '{box['label']}'"
            )

    def test_confidence_values_in_range(self, detections):
        for box in detections:
            assert 0.0 <= box["confidence"] <= 1.0, (
                f"confidence={box['confidence']} out of range"
            )

    def test_box_coordinates_within_image(self, detections, left_image):
        h, w = left_image.shape[:2]
        for box in detections:
            assert box["x1"] >= -1.0,    f"x1={box['x1']} out of bounds"
            assert box["y1"] >= -1.0,    f"y1={box['y1']} out of bounds"
            assert box["x2"] <= w + 1.0, f"x2={box['x2']} > image width {w}"
            assert box["y2"] <= h + 1.0, f"y2={box['y2']} > image height {h}"

    def test_box_coordinates_are_valid(self, detections):
        for box in detections:
            assert box["x2"] > box["x1"], f"x2 <= x1: {box}"
            assert box["y2"] > box["y1"], f"y2 <= y1: {box}"

    def test_at_least_one_detection_on_kitti_scene(self, detections):
        # Sample 000000 has cars visible — if zero detections something is wrong
        assert len(detections) > 0, (
            "No detections on sample 000000. "
            "Check model loading or confidence threshold."
        )


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------

class TestOutputFile:
    def test_json_file_exists(self, stage2_result):
        assert Path(stage2_result["output_path"]).exists()

    def test_json_is_loadable(self, stage2_result):
        with open(stage2_result["output_path"]) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_has_required_top_level_keys(self, stage2_result):
        with open(stage2_result["output_path"]) as f:
            data = json.load(f)
        for key in ("sample_id", "method", "boxes"):
            assert key in data, f"Missing top-level key: '{key}'"

    def test_json_sample_id_matches(self, stage2_result):
        with open(stage2_result["output_path"]) as f:
            data = json.load(f)
        assert data["sample_id"] == SAMPLE_ID

    def test_json_method_matches_config(self, stage2_result, stage_cfg):
        with open(stage2_result["output_path"]) as f:
            data = json.load(f)
        assert data["method"] == stage_cfg["method"]

    def test_json_boxes_is_list(self, stage2_result):
        with open(stage2_result["output_path"]) as f:
            data = json.load(f)
        assert isinstance(data["boxes"], list)


# ---------------------------------------------------------------------------
# GT label loader tests
# ---------------------------------------------------------------------------

class TestGTLabels:
    def test_returns_a_list(self, gt_labels):
        assert isinstance(gt_labels, list)

    def test_each_label_has_required_fields(self, gt_labels):
        required = {"label", "x1", "y1", "x2", "y2", "x", "y", "z", "h", "w", "l"}
        for i, obj in enumerate(gt_labels):
            missing = required - obj.keys()
            assert not missing, f"GT object {i} missing fields: {missing}"

    def test_gt_boxes_have_positive_dimensions(self, gt_labels):
        for obj in gt_labels:
            assert obj["x2"] > obj["x1"]
            assert obj["y2"] > obj["y1"]


# ---------------------------------------------------------------------------
# IoU utility tests — synthetic data, no KITTI needed
# ---------------------------------------------------------------------------

class TestBoxIou:
    def test_identical_boxes_iou_is_one(self):
        box = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
        assert box_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        a = {"x1": 0,  "y1": 0,  "x2": 10, "y2": 10}
        b = {"x1": 20, "y1": 20, "x2": 30, "y2": 30}
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap_iou(self):
        a = {"x1": 0,  "y1": 0, "x2": 20, "y2": 10}
        b = {"x1": 10, "y1": 0, "x2": 30, "y2": 10}
        # Intersection = 10*10=100, Union = 20*10 + 20*10 - 100 = 300
        assert box_iou(a, b) == pytest.approx(100 / 300)

    def test_iou_is_symmetric(self):
        a = {"x1": 0,  "y1": 0, "x2": 20, "y2": 20}
        b = {"x1": 10, "y1": 0, "x2": 30, "y2": 20}
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))


# ---------------------------------------------------------------------------
# AP utility tests — synthetic data
# ---------------------------------------------------------------------------

class TestComputeAp:
    def test_perfect_predictions_ap_is_one(self):
        gt   = [{"sample_id": "000000", "label": "Car",
                 "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        pred = [{"sample_id": "000000", "label": "Car", "confidence": 0.99,
                 "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        ap = compute_ap(pred, gt, "Car")
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_no_predictions_ap_is_zero(self):
        gt   = [{"sample_id": "000000", "label": "Car",
                 "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        ap = compute_ap([], gt, "Car")
        assert ap == pytest.approx(0.0)

    def test_no_gt_ap_is_zero(self):
        pred = [{"sample_id": "000000", "label": "Car", "confidence": 0.99,
                 "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        ap = compute_ap(pred, [], "Car")
        assert ap == pytest.approx(0.0)

    def test_ap_value_in_range(self):
        gt   = [{"sample_id": "000000", "label": "Car",
                 "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
        pred = [{"sample_id": "000000", "label": "Car", "confidence": 0.5,
                 "x1": 2, "y1": 2, "x2": 12, "y2": 12}]
        ap = compute_ap(pred, gt, "Car")
        assert 0.0 <= ap <= 1.0
