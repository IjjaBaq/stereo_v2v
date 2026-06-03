"""Geometry utilities for the stereo_v2v pipeline.

Pure functions — no imports from stages/. Safe to import from any stage
or test without circular dependencies.

All 3D coordinates follow KITTI camera convention:
    X → right, Y → down, Z → forward (into scene)
"""

import logging
import math

import numpy as np

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
# Angle utilities
# ---------------------------------------------------------------------------

def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to the range [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Equivalent angle in [-pi, pi].
    """
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


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
