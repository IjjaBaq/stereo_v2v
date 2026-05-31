"""KITTI dataset loader utilities for the stereo_v2v pipeline.

All functions are stateless and operate on explicit paths derived from
KITTI's standard directory structure.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(data_root: str, split: str, camera: str, sample_id: str, suffix: str = "_10.png") -> np.ndarray:
    """Load a KITTI image as a BGR numpy array.

    Args:
        data_root: Root path to the KITTI dataset (e.g. './data/kitti').
        split: Dataset split, either 'training' or 'testing'.
        camera: Camera directory, either 'image_2' (left) or 'image_3' (right).
        sample_id: Zero-padded 6-digit sample identifier (e.g. '000042').
        suffix: File suffix including extension (default '_10.png' for Stereo 2015,
                use '.png' for Object Detection split).

    Returns:
        Image array of shape (H, W, 3) and dtype uint8 in BGR format.

    Raises:
        FileNotFoundError: If the image file does not exist at the expected path.
        ValueError: If camera is not 'image_2' or 'image_3'.
    """
    if camera not in ("image_2", "image_3"):
        raise ValueError(f"camera must be 'image_2' or 'image_3', got '{camera}'")

    # Fixed to target the '_10.png' suffix used in the KITTI Stereo Benchmark.
    # This prevents cross-contamination if the Object Detection split is in the same folder.
    path = Path(data_root) / split / camera / f"{sample_id}{suffix}"

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            f"  data_root={data_root}, split={split}, "
            f"camera={camera}, sample_id={sample_id}"
        )

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"OpenCV failed to read image (may be corrupt): {path}")

    logger.debug("Loaded image %s — shape=%s dtype=%s", path.name, image.shape, image.dtype)
    return image


def load_disparity_gt(data_root: str, split: str, sample_id: str) -> np.ndarray:
    """Load a KITTI ground-truth disparity map.

    KITTI stores disparity as uint16 PNG where the true disparity in pixels
    is raw_value / 256.0. Pixels with raw value 0 are invalid (occluded or
    out-of-range) and are returned as np.nan.

    Args:
        data_root: Root path to the KITTI dataset (e.g. './data/kitti').
        split: Dataset split, either 'training' or 'testing'.
        sample_id: Zero-padded 6-digit sample identifier (e.g. '000042').

    Returns:
        Disparity map of shape (H, W) and dtype float32.
        Invalid pixels are set to np.nan.

    Raises:
        FileNotFoundError: If the disparity file does not exist or cannot be read.
    """
    # Updated to handle the standard '_10.png' suffix used in KITTI stereo benchmark
    path = Path(data_root) / split / "disp_noc_0" / f"{sample_id}_10.png"

    if not path.exists():
        raise FileNotFoundError(
            f"Disparity GT not found: {path}\n"
            f"  data_root={data_root}, split={split}, sample_id={sample_id}"
        )

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if raw is None:
        raise FileNotFoundError(f"OpenCV failed to read disparity map (may be corrupt): {path}")

    disp = raw.astype(np.float32) / 256.0
    disp[raw == 0] = np.nan

    logger.debug(
        "Loaded disparity GT %s — shape=%s valid_pixels=%d",
        path.name,
        disp.shape,
        int(np.sum(~np.isnan(disp))),
    )
    return disp


def load_calib(data_root: str, split: str, sample_id: str) -> dict:
    """Load a KITTI calibration file and return camera matrices as numpy arrays.

    Parses the standard KITTI .txt calibration format and extracts the
    four matrices needed for stereo depth and 3D lifting.

    Args:
        data_root: Root path to the KITTI dataset (e.g. './data/kitti').
        split: Dataset split, either 'training' or 'testing'.
        sample_id: Zero-padded 6-digit sample identifier (e.g. '000042').

    Returns:
        Dictionary with keys:
            P2 (np.ndarray): Left camera projection matrix, shape (3, 4).
            P3 (np.ndarray): Right camera projection matrix, shape (3, 4).
            R_rect_00 (np.ndarray): Rectification rotation matrix, shape (3, 3).
            Tr_velo_to_cam (np.ndarray): Velodyne-to-camera transform, shape (3, 4).

    Raises:
        FileNotFoundError: If the calibration file does not exist.
        KeyError: If any expected key is missing from the calibration file.
    """
    path = Path(data_root) / split / "calib" / f"{sample_id}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            f"  data_root={data_root}, split={split}, sample_id={sample_id}"
        )

    raw: dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            raw[key.strip()] = np.array(
                [float(v) for v in values.split()], dtype=np.float32
            )

    # KITTI file uses 'R0_rect', but our pipeline expects 'R_rect_00'
    if "R0_rect" in raw and "R_rect_00" not in raw:
        raw["R_rect_00"] = raw["R0_rect"]

    required_keys = ("P2", "P3", "R_rect_00", "Tr_velo_to_cam")
    for key in required_keys:
        if key not in raw:
            raise KeyError(
                f"Expected key '{key}' not found in calibration file: {path}\n"
                f"  Available keys: {list(raw.keys())}"
            )

    calib = {
        "P2":             raw["P2"].reshape(3, 4),
        "P3":             raw["P3"].reshape(3, 4),
        "R_rect_00":      raw["R_rect_00"].reshape(3, 3),
        "Tr_velo_to_cam": raw["Tr_velo_to_cam"].reshape(3, 4),
    }

    logger.debug("Loaded calib %s — keys=%s", path.name, list(calib.keys()))
    return calib


def load_sample(data_root: str, split: str, sample_id: str, suffix: str = "_10.png") -> dict:
    """Convenience wrapper — loads all inputs for one KITTI sample.

    Calls load_image (both left and right cameras), load_calib, and 
    attempts to load load_disparity_gt. If disparity GT is unavailable 
    (e.g., testing split), disp_gt is cleanly set to None and a 
    WARNING is logged instead of raising an exception.

    Args:
        data_root: Root path to the KITTI dataset (e.g. './data/kitti').
        split: Dataset split, either 'training' or 'testing'.
        sample_id: Zero-padded 6-digit sample identifier (e.g. '000042').
        suffix: File suffix for images (default '_10.png' for Stereo 2015,
                use '.png' for Object Detection split).

    Returns:
        Dictionary with keys:
            left (np.ndarray): Left color image, shape (H, W, 3), uint8.
            right (np.ndarray): Right color image, shape (H, W, 3), uint8.
            disp_gt (np.ndarray | None): Disparity GT, shape (H, W), float32, or None.
            calib (dict): Dictionary of reshaped calibration matrices.
    """
    left  = load_image(data_root, split, "image_2", sample_id, suffix)
    right = load_image(data_root, split, "image_3", sample_id, suffix)
    calib = load_calib(data_root, split, sample_id)

    disp_gt = None
    try:
        disp_gt = load_disparity_gt(data_root, split, sample_id)
    except FileNotFoundError:
        logger.warning(
            "Disparity GT not found for sample %s (split=%s) — disp_gt set to None",
            sample_id,
            split,
        )

    logger.info(
        "Sample %s loaded — left=%s right=%s disp_gt=%s",
        sample_id,
        left.shape,
        right.shape,
        disp_gt.shape if disp_gt is not None else "None",
    )
    return {"left": left, "right": right, "disp_gt": disp_gt, "calib": calib}
    
    
def load_labels(data_root: str, split: str, sample_id: str) -> list[dict]:
    """Load KITTI 2D/3D object labels from label_2/.
 
    Parses the standard KITTI label format (15 fields per line).
    Only returns objects with 3D annotations (excludes 'DontCare').
 
    KITTI label fields (space-separated):
        0  type          — class name string
        1  truncated     — float [0,1], degree of truncation
        2  occluded      — int {0,1,2,3}
        3  alpha         — observation angle [-pi, pi]
        4-7  bbox        — 2D bounding box: left, top, right, bottom (pixels)
        8-10 dimensions  — 3D height, width, length (metres)
        11-13 location   — 3D x, y, z in camera coords (metres)
        14 rotation_y    — rotation around Y-axis [-pi, pi]
 
    Args:
        data_root: Root path to the KITTI dataset (e.g. './data/kitti').
        split: Dataset split, either 'training' or 'testing'.
        sample_id: Zero-padded 6-digit sample identifier (e.g. '000042').
 
    Returns:
        List of dicts, one per object, with keys:
            label (str), truncated (float), occluded (int), alpha (float),
            x1 (float), y1 (float), x2 (float), y2 (float),
            h (float), w (float), l (float),
            x (float), y (float), z (float),
            rotation_y (float)
        DontCare objects are excluded.
 
    Raises:
        FileNotFoundError: If the label file does not exist.
    """
    path = Path(data_root) / split / "label_2" / f"{sample_id}.txt"
 
    if not path.exists():
        raise FileNotFoundError(
            f"Label file not found: {path}\n"
            f"  data_root={data_root}, split={split}, sample_id={sample_id}"
        )
 
    objects = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 15:
                logger.warning("Skipping malformed label line in %s: %s", path.name, line)
                continue
 
            label = fields[0]
            if label == "DontCare":
                continue
 
            objects.append({
                "label":      label,
                "truncated":  float(fields[1]),
                "occluded":   int(fields[2]),
                "alpha":      float(fields[3]),
                "x1":         float(fields[4]),
                "y1":         float(fields[5]),
                "x2":         float(fields[6]),
                "y2":         float(fields[7]),
                "h":          float(fields[8]),
                "w":          float(fields[9]),
                "l":          float(fields[10]),
                "x":          float(fields[11]),
                "y":          float(fields[12]),
                "z":          float(fields[13]),
                "rotation_y": float(fields[14]),
            })
 
    logger.debug("Loaded %d objects from %s", len(objects), path.name)
    return objects
 
