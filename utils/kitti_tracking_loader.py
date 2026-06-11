"""KITTI Tracking dataset loader.

Loads stereo image pairs, 3D labels, and calibration from the KITTI Tracking
benchmark for end-to-end Stage 1-3 evaluation.

Dataset structure expected:
    data/kitti/tracking/training/
    ├── image_02/{seq_id}/{frame_id}.png   left images
    ├── image_03/{seq_id}/{frame_id}.png   right images
    ├── label_02/{seq_id}.txt              3D labels (all frames)
    └── calib/{seq_id}.txt                 calibration

Label format (space-separated per line):
    frame track_id class truncated occluded alpha
    x1 y1 x2 y2 h w l x y z rotation_y
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Car-only pipeline (2026-06-10): Pedestrian and Cyclist dropped. Van and Truck
# are kept as car-like vehicle GT. NOTE: the Stage-3 validator filters GT to its
# own ("Car",) set, so Van/Truck labels loaded here are not matched as Car unless
# explicitly remapped.
KITTI_CLASSES = ("Car", "Van", "Truck")


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

    The tracking calib file mixes two line formats: P0–P3 use a colon
    separator (``P2: ...``) while R_rect / Tr_velo_cam / Tr_imu_velo are
    space-separated (``R_rect ...``). Both are parsed; only the projection
    matrices and stereo geometry are returned (the extrinsics are unused now
    that Stage 4 runs on CARLA).

    Args:
        tracking_root: Path to data/kitti/tracking.
        split: 'training' or 'testing'.
        seq_id: Zero-padded 4-digit sequence ID.

    Returns:
        Dict with keys:
            P2, P3 (np.ndarray 3x4): camera projection matrices.
            focal_length_px (float), baseline_m (float).

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
            if not line:
                continue
            if ":" in line:
                key, val = line.split(":", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, val = parts
            try:
                data[key.strip()] = np.array(
                    [float(x) for x in val.split()], dtype=np.float64
                )
            except ValueError:
                continue

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
            labels (list of label dicts).
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
    }
