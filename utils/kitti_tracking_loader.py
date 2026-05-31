"""KITTI Tracking dataset loader.

Loads stereo image pairs, 3D labels, calibration, and oxts ego-motion
from the KITTI Tracking benchmark for end-to-end pipeline evaluation.

Dataset structure expected:
    data/kitti/tracking/training/
    ├── image_02/{seq_id}/{frame_id}.png   left images
    ├── image_03/{seq_id}/{frame_id}.png   right images
    ├── label_02/{seq_id}.txt              3D labels (all frames)
    ├── calib/{seq_id}.txt                 calibration
    └── oxts/{seq_id}.txt                  ego-motion (one line per frame)

Label format (space-separated per line):
    frame track_id class truncated occluded alpha
    x1 y1 x2 y2 h w l x y z rotation_y

oxts format (space-separated per line, one line per frame):
    lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au wx wy wz
    pos_accuracy vel_accuracy navstat numsats posmode velmode orimode
"""

import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

KITTI_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Cyclist")

# oxts field indices
_LAT, _LON, _ALT = 0, 1, 2
_ROLL, _PITCH, _YAW = 3, 4, 5
_VF = 8   # forward velocity m/s


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def load_tracking_image(
    tracking_root: str,
    split: str,
    camera: str,
    seq_id: str,
    frame_id: int,
) -> np.ndarray:
    """Load a single frame from a tracking sequence.

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        camera: 'image_02' (left) or 'image_03' (right).
        seq_id: Zero-padded 4-digit sequence ID e.g. '0000'.
        frame_id: Integer frame index.

    Returns:
        BGR image, shape (H, W, 3), uint8.

    Raises:
        FileNotFoundError: If image file is missing.
    """
    path = (
        Path(tracking_root) / split / camera
        / seq_id / f"{frame_id:06d}.png"
    )
    if not path.exists():
        raise FileNotFoundError(f"Tracking image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV failed to read: {path}")
    return img


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_tracking_calib(
    tracking_root: str,
    split: str,
    seq_id: str,
) -> dict:
    """Load calibration for a tracking sequence.

    Tracking calib format matches detection except Tr_velo_cam is absent
    and keys use R_rect instead of R0_rect.

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        seq_id: Zero-padded 4-digit sequence ID.

    Returns:
        Dict with keys: P2, P3, R_rect, baseline_m, focal_length_px.

    Raises:
        FileNotFoundError: If calib file is missing.
    """
    path = Path(tracking_root) / split / "calib" / f"{seq_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Tracking calib not found: {path}")

    data: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key.strip()] = np.array(
                [float(x) for x in val.split()], dtype=np.float64
            )

    P2 = data["P2"].reshape(3, 4)
    P3 = data["P3"].reshape(3, 4)

    # Baseline from P3 tx (t_x = -f*B → B = -t_x/f)
    focal_length = float(P2[0, 0])
    tx_p3        = float(P3[0, 3])
    baseline_m   = abs(tx_p3) / focal_length

    logger.debug(
        "Tracking calib seq=%s — focal=%.2fpx baseline=%.4fm",
        seq_id, focal_length, baseline_m,
    )

    return {
        "P2":              P2,
        "P3":              P3,
        "focal_length_px": focal_length,
        "baseline_m":      baseline_m,
    }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def load_tracking_labels(
    tracking_root: str,
    split: str,
    seq_id: str,
    frame_id: int,
    classes: tuple[str, ...] = KITTI_CLASSES,
) -> list[dict]:
    """Load 3D GT labels for a single frame of a tracking sequence.

    Filters to requested classes and skips DontCare / invalid entries
    (z <= 0 or h/w/l <= 0).

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        seq_id: Zero-padded 4-digit sequence ID.
        frame_id: Integer frame index.
        classes: Tuple of class names to keep.

    Returns:
        List of label dicts with keys:
            track_id, label, truncated, occluded, alpha,
            x1, y1, x2, y2, h, w, l, x, y, z, rotation_y.
    """
    path = Path(tracking_root) / split / "label_02" / f"{seq_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Tracking labels not found: {path}")

    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue

            fid = int(parts[0])
            if fid != frame_id:
                continue

            label = parts[2]
            if label not in classes:
                continue

            h, w, l = float(parts[10]), float(parts[11]), float(parts[12])
            x, y, z = float(parts[13]), float(parts[14]), float(parts[15])

            # Skip invalid entries
            if z <= 0 or h <= 0 or w <= 0 or l <= 0:
                continue

            labels.append({
                "track_id":  int(parts[1]),
                "label":     label,
                "truncated": float(parts[3]),
                "occluded":  int(parts[4]),
                "alpha":     float(parts[5]),
                "x1":        float(parts[6]),
                "y1":        float(parts[7]),
                "x2":        float(parts[8]),
                "y2":        float(parts[9]),
                "h":         h,
                "w":         w,
                "l":         l,
                "x":         x,
                "y":         y,
                "z":         z,
                "rotation_y": float(parts[16]),
            })

    return labels


def get_sequence_length(
    tracking_root: str,
    split: str,
    seq_id: str,
) -> int:
    """Return number of frames in a tracking sequence.

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        seq_id: Zero-padded 4-digit sequence ID.

    Returns:
        Number of frames (PNG files) in image_02/{seq_id}/.
    """
    img_dir = Path(tracking_root) / split / "image_02" / seq_id
    return len(list(img_dir.glob("*.png")))


# ---------------------------------------------------------------------------
# oxts ego-motion
# ---------------------------------------------------------------------------

def load_oxts_frame(
    tracking_root: str,
    split: str,
    seq_id: str,
    frame_id: int,
) -> dict:
    """Load oxts IMU/GPS data for a single frame.

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        seq_id: Zero-padded 4-digit sequence ID.
        frame_id: Integer frame index (0-based line number).

    Returns:
        Dict with keys: lat, lon, alt, roll, pitch, yaw, forward_vel_ms.

    Raises:
        FileNotFoundError: If oxts file is missing.
        IndexError: If frame_id exceeds number of lines.
    """
    path = Path(tracking_root) / split / "oxts" / f"{seq_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"oxts file not found: {path}")

    with open(path) as f:
        lines = f.readlines()

    if frame_id >= len(lines):
        raise IndexError(
            f"frame_id={frame_id} exceeds oxts length={len(lines)} "
            f"for seq={seq_id}"
        )

    vals = [float(x) for x in lines[frame_id].strip().split()]

    return {
        "lat":            vals[_LAT],
        "lon":            vals[_LON],
        "alt":            vals[_ALT],
        "roll":           vals[_ROLL],
        "pitch":          vals[_PITCH],
        "yaw":            vals[_YAW],
        "forward_vel_ms": vals[_VF],
    }


def compute_ego_transform(
    oxts_a: dict,
    oxts_b: dict,
) -> np.ndarray:
    """Compute rigid transform from frame B to frame A's coordinate system.

    Uses the Mercator projection approach from the KITTI odometry devkit.
    Frame A is treated as the world origin.

    Args:
        oxts_a: oxts dict for frame A (origin / Vehicle A).
        oxts_b: oxts dict for frame B (Vehicle B to transform).

    Returns:
        4x4 homogeneous transform matrix T such that:
            p_A = T @ p_B
        where p_A, p_B are 3D points in their respective camera frames.
    """
    EARTH_RADIUS = 6378137.0  # WGS84 equatorial radius in metres

    def mercator_xy(lat, lon, lat0):
        """Project lat/lon to metric XY relative to lat0."""
        scale = math.cos(math.radians(lat0))
        x = scale * math.radians(lon) * EARTH_RADIUS
        y = scale * EARTH_RADIUS * math.log(
            math.tan(math.pi / 4 + math.radians(lat) / 2)
        )
        return x, y

    lat0 = oxts_a["lat"]

    xa, ya = mercator_xy(oxts_a["lat"], oxts_a["lon"], lat0)
    xb, yb = mercator_xy(oxts_b["lat"], oxts_b["lon"], lat0)

    # Translation in world (ENU) frame
    dx = xb - xa   # east
    dy = yb - ya   # north
    dz = oxts_b["alt"] - oxts_a["alt"]

    # Yaw difference (rotation around up axis)
    dyaw = oxts_b["yaw"] - oxts_a["yaw"]

    # Rotation matrix Rz(-dyaw) — transforms B's heading to A's frame
    cos_y = math.cos(-dyaw)
    sin_y = math.sin(-dyaw)
    R = np.array([
        [ cos_y, -sin_y, 0.0],
        [ sin_y,  cos_y, 0.0],
        [ 0.0,    0.0,   1.0],
    ], dtype=np.float64)

    # Translation vector rotated into A's frame
    t_world = np.array([dx, dy, dz], dtype=np.float64)
    t_a     = R @ t_world

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t_a

    logger.debug(
        "Ego transform — dx=%.2fm dy=%.2fm dz=%.2fm dyaw=%.4frad",
        dx, dy, dz, dyaw,
    )

    return T


# ---------------------------------------------------------------------------
# Convenience: load full frame bundle
# ---------------------------------------------------------------------------

def load_tracking_frame(
    tracking_root: str,
    split: str,
    seq_id: str,
    frame_id: int,
    classes: tuple[str, ...] = KITTI_CLASSES,
) -> dict:
    """Load all data for a single tracking frame in one call.

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        seq_id: Zero-padded 4-digit sequence ID.
        frame_id: Integer frame index.
        classes: Classes to keep in labels.

    Returns:
        Dict with keys:
            seq_id, frame_id,
            left  (BGR ndarray),
            right (BGR ndarray),
            calib (dict),
            labels (list of label dicts),
            oxts  (dict).
    """
    return {
        "seq_id":   seq_id,
        "frame_id": frame_id,
        "left":     load_tracking_image(
                        tracking_root, split, "image_02", seq_id, frame_id),
        "right":    load_tracking_image(
                        tracking_root, split, "image_03", seq_id, frame_id),
        "calib":    load_tracking_calib(tracking_root, split, seq_id),
        "labels":   load_tracking_labels(
                        tracking_root, split, seq_id, frame_id, classes),
        "oxts":     load_oxts_frame(tracking_root, split, seq_id, frame_id),
    }
