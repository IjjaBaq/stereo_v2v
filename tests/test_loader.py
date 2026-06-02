"""Smoke tests for utils/kitti_loader.py

Tests all three loader functions on sample "000000".
Run with: pytest tests/
"""

import numpy as np
import pytest

from utils.kitti_loader import load_calib, load_disparity_gt, load_image, load_labels, load_sample

DATA_ROOT = "./data/kitti/detection"
SPLIT = "training"
SAMPLE_ID = "000000"


class TestLoadImage:
    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, request):
        # Cache the loaded images in memory for this test class
        request.cls.left = load_image(DATA_ROOT, SPLIT, "image_2", SAMPLE_ID)
        request.cls.right = load_image(DATA_ROOT, SPLIT, "image_3", SAMPLE_ID)

    def test_left_image_shape(self):
        assert self.left.ndim == 3
        assert self.left.shape[2] == 3

    def test_left_image_dtype(self):
        assert self.left.dtype == np.uint8

    def test_right_image_shape(self):
        assert self.right.ndim == 3
        assert self.right.shape[2] == 3

    def test_stereo_pair_same_shape(self):
        assert self.left.shape == self.right.shape

    def test_invalid_camera_raises(self):
        with pytest.raises(ValueError):
            load_image(DATA_ROOT, SPLIT, "image_99", SAMPLE_ID)

    def test_missing_sample_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image(DATA_ROOT, SPLIT, "image_2", "999999")


class TestLoadDisparityGt:
    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, request):
        # Cache disparity map in memory for this test class
        request.cls.disp = load_disparity_gt(DATA_ROOT, SPLIT, SAMPLE_ID)

    def test_shape_is_2d(self):
        assert self.disp.ndim == 2

    def test_dtype_is_float32(self):
        assert self.disp.dtype == np.float32

    def test_has_valid_pixels(self):
        assert np.sum(~np.isnan(self.disp)) > 0

    def test_invalid_pixels_are_nan(self):
        # KITTI always has some invalid pixels
        assert np.any(np.isnan(self.disp))

    def test_missing_sample_raises(self):
        with pytest.raises(FileNotFoundError):
            load_disparity_gt(DATA_ROOT, SPLIT, "999999")


class TestLoadCalib:
    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, request):
        # Cache calibration settings in memory for this test class
        request.cls.calib = load_calib(DATA_ROOT, SPLIT, SAMPLE_ID)

    def test_returns_all_keys(self):
        for key in ("P2", "P3", "R_rect_00", "Tr_velo_to_cam"):
            assert key in self.calib

    def test_P2_shape(self):
        assert self.calib["P2"].shape == (3, 4)

    def test_P3_shape(self):
        assert self.calib["P3"].shape == (3, 4)

    def test_R_rect_shape(self):
        assert self.calib["R_rect_00"].shape == (3, 3)

    def test_Tr_shape(self):
        assert self.calib["Tr_velo_to_cam"].shape == (3, 4)

    def test_all_float32(self):
        for key, val in self.calib.items():
            assert val.dtype == np.float32, f"{key} dtype is {val.dtype}"

    def test_missing_sample_raises(self):
        with pytest.raises(FileNotFoundError):
            load_calib(DATA_ROOT, SPLIT, "999999")

class TestLoadLabels:
    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, request):
        request.cls.labels = load_labels(DATA_ROOT, SPLIT, SAMPLE_ID)

    def test_returns_list(self):
        assert isinstance(self.labels, list)

    def test_list_not_empty(self):
        # Sample 000000 has visible objects — should never be empty
        assert len(self.labels) > 0

    def test_each_item_is_dict(self):
        for obj in self.labels:
            assert isinstance(obj, dict)

    def test_required_keys_present(self):
        required = (
            "label", "truncated", "occluded", "alpha",
            "x1", "y1", "x2", "y2",
            "h", "w", "l",
            "x", "y", "z",
            "rotation_y",
        )
        for obj in self.labels:
            for key in required:
                assert key in obj, f"Missing key '{key}' in object: {obj}"

    def test_no_dontcare_objects(self):
        for obj in self.labels:
            assert obj["label"] != "DontCare"

    def test_valid_kitti_class_labels(self):
        valid_classes = {
            "Car", "Van", "Truck", "Pedestrian",
            "Person_sitting", "Cyclist", "Tram", "Misc",
        }
        for obj in self.labels:
            assert obj["label"] in valid_classes, \
                f"Unexpected class label: '{obj['label']}'"

    def test_truncated_in_range(self):
        for obj in self.labels:
            assert 0.0 <= obj["truncated"] <= 1.0, \
                f"truncated={obj['truncated']} out of [0, 1]"

    def test_occluded_in_range(self):
        for obj in self.labels:
            assert obj["occluded"] in (0, 1, 2, 3), \
                f"occluded={obj['occluded']} not in {{0,1,2,3}}"

    def test_bbox_coordinates_positive(self):
        for obj in self.labels:
            assert obj["x1"] >= 0 and obj["y1"] >= 0
            assert obj["x2"] >= 0 and obj["y2"] >= 0

    def test_bbox_x2_greater_than_x1(self):
        for obj in self.labels:
            assert obj["x2"] > obj["x1"], \
                f"x2={obj['x2']} must be > x1={obj['x1']}"

    def test_bbox_y2_greater_than_y1(self):
        for obj in self.labels:
            assert obj["y2"] > obj["y1"], \
                f"y2={obj['y2']} must be > y1={obj['y1']}"

    def test_dimensions_positive(self):
        for obj in self.labels:
            assert obj["h"] > 0 and obj["w"] > 0 and obj["l"] > 0, \
                f"Non-positive dimension in {obj}"

    def test_depth_z_positive(self):
        # In KITTI camera coords, Z is always positive (forward)
        for obj in self.labels:
            assert obj["z"] > 0, \
                f"z={obj['z']} must be positive (camera forward)"

    def test_missing_sample_raises(self):
        with pytest.raises(FileNotFoundError):
            load_labels(DATA_ROOT, SPLIT, "999999")

class TestLoadSample:
    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, request):
        # Cache the full stereo sample dict in memory for this test class
        request.cls.sample = load_sample(DATA_ROOT, SPLIT, SAMPLE_ID)

    def test_returns_all_keys(self):
        for key in ("left", "right", "disp_gt", "calib"):
            assert key in self.sample

    def test_stereo_pair_match(self):
        assert self.sample["left"].shape == self.sample["right"].shape

    def test_disp_gt_not_none_for_training(self):
        assert self.sample["disp_gt"] is not None

    def test_disp_gt_matches_image_spatial(self):
        """Verify that disparity GT matches left image dimensions exactly.
        
        NOTE: In an uncorrupted KITTI Stereo 2015 benchmark directory, both the
        color images (*_10.png) and disparity maps (*_10.png) originate from
        the same rectification matrix step, meaning their shapes match on disk perfectly.
        """
        h, w = self.sample["left"].shape[:2]
        assert self.sample["disp_gt"].shape == (h, w), (
            f"Spatial mismatch: Image is {(h, w)}, but Disparity is {self.sample['disp_gt'].shape}. "
            f"Double-check that your directory is not cross-contaminated with Object split files."
        )
