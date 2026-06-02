"""Geometry utilities for the stereo_v2v pipeline.

Pure functions — no imports from stages/. Safe to import from any stage
or test without circular dependencies.

All 3D coordinates follow KITTI camera convention:
    X → right, Y → down, Z → forward (into scene)
"""

import logging
import math
from typing import Literal

import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unprojection
# ---------------------------------------------------------------------------

def unproject_box(
    box2d: dict,
    Z: float,
    P2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Unproject a 2D bounding box to a 3D center and metric dimensions.

    Uses Option A — unprojects the 2D box center directly to get Y_center.
    No Y correction is applied because cy_2d is already the box center.

    Equivalence (Option B):
        Y_bottom = (y2 - P2[1,2]) * Z / P2[1,1]
        Y_center = Y_bottom - h_3d / 2
    Both options give the same result. Option A is used for clarity.

    GT validation note: KITTI label_2 stores y = bottom of object.
    Convert before computing metrics: gt_y_center = gt_y - gt_h / 2
    (Y points down in camera coords — bottom = largest Y, center = bottom - h/2)

    Args:
        box2d: Dict with keys x1, y1, x2, y2 in pixel coordinates.
        Z: Metric depth of the box center in metres.
        P2: Left camera projection matrix, shape (3, 4), from load_calib.

    Returns:
        Tuple of (X, Y_center, Z, w_3d, h_3d) all in metres.
        X, Y_center, Z — 3D center in camera coordinates.
        w_3d, h_3d     — metric width and height of the object.
    """
    x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]

    cx_2d = (x1 + x2) / 2.0
    cy_2d = (y1 + y2) / 2.0

    fx = float(P2[0, 0])
    fy = float(P2[1, 1])
    cx = float(P2[0, 2])
    cy = float(P2[1, 2])

    X        = (cx_2d - cx) * Z / fx
    Y_center = (cy_2d - cy) * Z / fy
    w_3d     = (x2 - x1)   * Z / fx
    h_3d     = (y2 - y1)   * Z / fy

    return float(X), float(Y_center), float(Z), float(w_3d), float(h_3d)


# ---------------------------------------------------------------------------
# Heading
# ---------------------------------------------------------------------------

def compute_heading(
    cx_2d: float,
    P2: np.ndarray,
    method: Literal["ray_angle"] = "ray_angle",
) -> float:
    """Estimate object heading from its 2D image position.

    Supported methods:
        ray_angle: heading = arctan2(cx_2d - cx, fx)
            Measures the angle of the ray from the optical axis to the
            box center. Objects left of center → negative heading,
            right of center → positive heading, center → ~0.

    This is an approximation — it gives the viewing ray angle, not the
    true object orientation. Validated against GT rotation_y via
    mean_heading_error metric in validate_stage3_lift.py.

    New methods can be added here without changing stage3_lift.py —
    controlled via config/stage3.yaml heading_method field.

    Args:
        cx_2d: Horizontal pixel coordinate of the 2D box center.
        P2: Left camera projection matrix, shape (3, 4).
        method: Heading estimation method. Currently only 'ray_angle'.

    Returns:
        Heading angle in radians, in range [-pi, pi].

    Raises:
        ValueError: If method is not supported.
    """
    if method == "ray_angle":
        fx = float(P2[0, 0])
        cx = float(P2[0, 2])
        return float(math.atan2(cx_2d - cx, fx))

    raise ValueError(
        f"Unsupported heading method: '{method}'. "
        f"Supported: ['ray_angle']"
    )


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to the range [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Equivalent angle in [-pi, pi].
    """
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def recover_rotation_y(
    alpha: float,
    cx_2d: float,
    P2: np.ndarray,
) -> float:
    """Recover global heading (rotation_y) from a learned local angle alpha.

    The 'learned' heading method splits the orientation into two parts:
        alpha     — the allocentric / observation angle, predicted by an
                    appearance model from the object's image crop. This is
                    the only part a crop can determine (it carries no
                    information about where in the image the crop came from).
        ray angle — the viewing-ray angle to the box center, atan2(x, z),
                    which is purely geometric and identical to
                    compute_heading(cx_2d, P2, 'ray_angle').

    KITTI relation (verified on tracking seq 0000):
        rotation_y = alpha + atan2(x, z)
    where atan2(x, z) == atan2(cx_2d - cx, fx) since x/z = (cx_2d - cx)/fx.

    This is the geometric counterpart to the learned head — the model
    supplies alpha, this function adds back the ray term geometry already
    knows. Replaces the ray_angle-only heading, which implicitly assumed
    alpha = 0 and produced ~146 deg error for road-aligned vehicles.

    Args:
        alpha: Allocentric observation angle in radians (from the model).
        cx_2d: Horizontal pixel coordinate of the 2D box center.
        P2: Left camera projection matrix, shape (3, 4).

    Returns:
        Global heading rotation_y in radians, wrapped to [-pi, pi].
    """
    ray = compute_heading(cx_2d, P2, method="ray_angle")
    return wrap_to_pi(alpha + ray)


# ---------------------------------------------------------------------------
# 3D IoU
# ---------------------------------------------------------------------------

def _box3d_to_bev_polygon(box: dict) -> Polygon:
    """Convert a 3D box dict to a Shapely BEV polygon (X-Z plane).

    The box is represented as a rotated rectangle in the BEV plane.
    Rotation is around the Y axis (heading).

    Args:
        box: Dict with keys x, y, z, l, w, h, heading.

    Returns:
        Shapely Polygon representing the BEV footprint.
    """
    cx  = box["x"]
    cz  = box["z"]
    l   = box["l"]
    w   = box["w"]
    yaw = box["heading"]

    # Four corners of the box in local frame (l along Z, w along X)
    corners_local = np.array([
        [ w / 2,  l / 2],
        [-w / 2,  l / 2],
        [-w / 2, -l / 2],
        [ w / 2, -l / 2],
    ])

    # Rotate around Y axis (in BEV: rotate in X-Z plane)
    cos_h = math.cos(yaw)
    sin_h = math.sin(yaw)
    R = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h],
    ])

    corners_world = corners_local @ R.T + np.array([cx, cz])
    return Polygon(corners_world)


def box3d_iou(pred: dict, gt: dict) -> float:
    """Compute 3D IoU between two 3D bounding boxes.

    Decomposes into:
        IoU_3d = IoU_bev * height_overlap_ratio

    BEV IoU uses Shapely polygon intersection (handles rotation correctly).
    Height overlap uses the Y axis (camera coords: Y points down).

    Both arguments must be dicts with keys:
        x, y, z   — 3D center in camera coordinates (metres)
        l, w, h   — length, width, height (metres)
        heading   — rotation around Y-axis (radians)

    Args:
        pred: Predicted 3D box dict.
        gt:   Ground-truth 3D box dict.

    Returns:
        3D IoU value in [0, 1].
    """
    # --- BEV IoU (X-Z plane) ---
    poly_pred = _box3d_to_bev_polygon(pred)
    poly_gt   = _box3d_to_bev_polygon(gt)

    if not poly_pred.is_valid or not poly_gt.is_valid:
        logger.warning("Invalid BEV polygon — returning IoU=0.0")
        return 0.0

    inter_bev = poly_pred.intersection(poly_gt).area
    if inter_bev == 0.0:
        return 0.0

    union_bev = poly_pred.union(poly_gt).area
    iou_bev   = inter_bev / union_bev if union_bev > 0 else 0.0

    # --- Height overlap (Y axis, Y points down) ---
    # y + h/2 = bottom (larger Y), y - h/2 = top (smaller Y)
    # Formula is numerically correct — overlap is symmetric
    pred_y_top = pred["y"] - pred["h"] / 2.0
    pred_y_bot = pred["y"] + pred["h"] / 2.0
    gt_y_top   = gt["y"]   - gt["h"]   / 2.0
    gt_y_bot   = gt["y"]   + gt["h"]   / 2.0

    inter_h = max(0.0, min(pred_y_bot, gt_y_bot) - max(pred_y_top, gt_y_top))
    union_h = max(
        pred_y_bot - pred_y_top + gt_y_bot - gt_y_top - inter_h,
        1e-9,
    )
    iou_h = inter_h / union_h

    return float(iou_bev * iou_h)


# ---------------------------------------------------------------------------
# Center distance
# ---------------------------------------------------------------------------

def center_distance(pred: dict, gt: dict) -> float:
    """Compute Euclidean distance between two 3D box centers.

    Both arguments must be dicts with keys x, y, z (metres).

    Args:
        pred: Predicted 3D box dict.
        gt:   Ground-truth 3D box dict.

    Returns:
        Euclidean distance in metres.
    """
    dx = pred["x"] - gt["x"]
    dy = pred["y"] - gt["y"]
    dz = pred["z"] - gt["z"]
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))
    
def project_box3d_to_image(
    box: dict,
    P2: np.ndarray,
) -> np.ndarray | None:
    """Project 3D box corners onto the image plane via P2.

    Computes the 8 corners of the 3D box in camera coordinates,
    then projects each through P2 to get pixel coordinates.

    Corner ordering (local frame, Y points down):
        4 top corners (y - h/2), 4 bottom corners (y + h/2)
        each rotated by heading around Y axis.

    Args:
        box: 3D box dict with keys x, y, z, l, w, h, heading.
        P2: Left camera projection matrix, shape (3, 4).

    Returns:
        Projected corners as int array, shape (8, 2) in pixel coords (u, v).
        Returns None if any corner is behind the camera (z <= 0).
    """
    cx, cy, cz = box["x"], box["y"], box["z"]
    l, w, h    = box["l"], box["w"], box["h"]
    yaw        = box["heading"]

    # 8 corners in local box frame (X-Z rotated, Y=height)
    # format: [dx, dy, dz] where dy is along Y (down)
    corners_local = np.array([
        [ w/2,  -h/2,  l/2],
        [-w/2,  -h/2,  l/2],
        [-w/2,  -h/2, -l/2],
        [ w/2,  -h/2, -l/2],
        [ w/2,   h/2,  l/2],
        [-w/2,   h/2,  l/2],
        [-w/2,   h/2, -l/2],
        [ w/2,   h/2, -l/2],
    ], dtype=np.float32)

    # Rotation around Y axis by heading
    cos_h = math.cos(yaw)
    sin_h = math.sin(yaw)
    R = np.array([
        [ cos_h, 0, sin_h],
        [     0, 1,     0],
        [-sin_h, 0, cos_h],
    ], dtype=np.float32)

    # Rotate and translate to camera frame
    corners_cam = corners_local @ R.T + np.array([cx, cy, cz])

    # Check all corners are in front of camera
    if np.any(corners_cam[:, 2] <= 0):
        return None

    # Project through P2: [u, v, 1] = P2 @ [X, Y, Z, 1]
    ones = np.ones((8, 1), dtype=np.float32)
    corners_h = np.hstack([corners_cam, ones])  # (8, 4)
    projected = (P2 @ corners_h.T).T            # (8, 3)

    # Normalize by z
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels.astype(np.int32)
