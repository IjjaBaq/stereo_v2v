"""Source-agnostic V2V fusion core.

The pure fusion algorithm shared by every Stage 4 data source: register
Vehicle B's 3D boxes into Vehicle A's camera frame via a 4x4 transform, greedily
match by BEV centre distance per class, and merge corroborated pairs (noisy-OR
confidence, confidence-weighted pose, circular-mean heading).

This module knows nothing about where the boxes or the transform came from — it
operates on box dicts and a 4x4 matrix. Required box keys are ``label``,
``confidence`` and the centre ``x, y, z``; the 3D extent ``l, w, h`` and
``heading`` are optional. This lets the same core fuse both Stage-3 boxes
(position-only: x, y, z) and full 3D boxes (e.g. CARLA GT with l, w, h,
heading) — matching is always by BEV centre distance, and the extent/heading
fields are merged and emitted only when the inputs carry them. The data loaders
(KITTI, CARLA, ...) and I/O live in ``stages.stage4_fusion``. Boxes use the
KITTI camera convention: x-right, y-down, z-forward; BEV is the x-z plane;
heading is about the y-axis.
"""

import math

import numpy as np

from utils.geometry import wrap_to_pi

# 3D extent + heading — present on full 3D boxes (e.g. CARLA GT), absent on
# Stage-3 position-only boxes. Merged/emitted only when the inputs carry them.
_OPTIONAL_DIMS = ("l", "w", "h")


# ---------------------------------------------------------------------------
# Box registration (B → A)
# ---------------------------------------------------------------------------

def _yaw_about_cam_y(R: np.ndarray) -> float:
    """Extract rotation about the camera Y-axis (down) from a 3x3 rotation.

    For a rotation about camera-Y by θ:
        R_y(θ) = [[cosθ, 0, sinθ], [0, 1, 0], [-sinθ, 0, cosθ]]
    so θ = atan2(R[0,2], R[2,2]). Vehicle ego-motion is dominated by this
    component (planar driving), which is exactly the heading offset.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Yaw angle about camera Y in radians.
    """
    return float(math.atan2(R[0, 2], R[2, 2]))


def transform_box(box: dict, T: np.ndarray) -> dict:
    """Transform a 3D box from one camera frame to another via a 4x4.

    Centre is mapped by T. If the box carries a ``heading`` it is offset by T's
    camera-Y yaw; position-only boxes (no heading) are transformed without it.
    All other keys (dimensions, label, confidence, 2D box) are carried through.

    Args:
        box: Box dict with x, y, z, label, confidence (required) and
             optionally l, w, h, heading.
        T: 4x4 homogeneous transform mapping B's frame to A's.

    Returns:
        New box dict in the target frame (other keys carried through).
    """
    p = T @ np.array([box["x"], box["y"], box["z"], 1.0])
    out = dict(box)
    out["x"], out["y"], out["z"] = float(p[0]), float(p[1]), float(p[2])
    if "heading" in box:
        yaw = _yaw_about_cam_y(T[:3, :3])
        out["heading"] = wrap_to_pi(float(box["heading"]) + yaw)
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def bev_distance(a: dict, b: dict) -> float:
    """BEV (x-z plane) Euclidean distance between two box centres in metres."""
    return float(math.hypot(a["x"] - b["x"], a["z"] - b["z"]))


def match_boxes(
    boxes_a: list[dict],
    boxes_b: list[dict],
    max_dist: dict,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match A↔B by ascending BEV centre distance within class.

    Args:
        boxes_a: Vehicle A 3D boxes.
        boxes_b: Vehicle B 3D boxes, already registered into A's frame.
        max_dist: Per-class max BEV distance for a valid match.

    Returns:
        (matches, unmatched_a, unmatched_b):
            matches      — list of (i_a, i_b) pairs.
            unmatched_a  — A indices with no match.
            unmatched_b  — B indices with no match.
    """
    candidates = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            if a["label"] != b["label"]:
                continue
            d = bev_distance(a, b)
            if d <= float(max_dist.get(a["label"], 0.0)):
                candidates.append((d, i, j))
    candidates.sort()

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        matches.append((i, j))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [i for i in range(len(boxes_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(boxes_b)) if j not in used_b]
    return matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def noisy_or(c1: float, c2: float) -> float:
    """Noisy-OR fusion of two confidences: 1 - (1-c1)(1-c2)."""
    return float(1.0 - (1.0 - c1) * (1.0 - c2))


def merge_static_pair(a: dict, b: dict) -> dict:
    """Merge a corroborated pair into one fused box.

    Centre is a confidence-weighted average and confidence is noisy-OR. The 3D
    extent (l, w, h) and heading are merged only when both inputs carry them —
    dimensions by confidence-weighted average, heading by confidence-weighted
    circular mean. Position-only pairs fuse to a position-only box.

    Args:
        a: Vehicle A box.
        b: Vehicle B box, registered into A's frame.

    Returns:
        Fused box dict with source='fused', is_dynamic=False.
    """
    wa, wb = float(a["confidence"]), float(b["confidence"])
    wsum = wa + wb if (wa + wb) > 0 else 1.0

    def wavg(key):
        return (wa * float(a[key]) + wb * float(b[key])) / wsum

    fused = {
        "label":      a["label"],
        "confidence": round(noisy_or(wa, wb), 4),
        "x":          round(wavg("x"), 3),
        "y":          round(wavg("y"), 3),
        "z":          round(wavg("z"), 3),
    }

    for dim in _OPTIONAL_DIMS:
        if dim in a and dim in b:
            fused[dim] = round(wavg(dim), 3)

    if "heading" in a and "heading" in b:
        # circular weighted mean for heading
        sin_h = wa * math.sin(a["heading"]) + wb * math.sin(b["heading"])
        cos_h = wa * math.cos(a["heading"]) + wb * math.cos(b["heading"])
        fused["heading"] = round(wrap_to_pi(math.atan2(sin_h, cos_h)), 4)

    fused["source"]     = "fused"
    fused["is_dynamic"] = False
    return fused


def _tagged(box: dict, source: str, is_dynamic: bool) -> dict:
    """Return a clean output box carrying only the schema fields + tags.

    The 3D extent (l, w, h) and heading are carried through only when the input
    box has them, so position-only boxes stay position-only.
    """
    out = {
        "label":      box["label"],
        "confidence": round(float(box["confidence"]), 4),
        "x":          round(float(box["x"]), 3),
        "y":          round(float(box["y"]), 3),
        "z":          round(float(box["z"]), 3),
    }
    for dim in _OPTIONAL_DIMS:
        if dim in box:
            out[dim] = round(float(box[dim]), 3)
    if "heading" in box:
        out["heading"] = round(float(box["heading"]), 4)
    out["source"]     = source
    out["is_dynamic"] = is_dynamic
    return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse(
    boxes_a: list[dict],
    boxes_b: list[dict],
    T_b_to_a: np.ndarray,
    stage_cfg: dict,
) -> tuple[list[dict], dict]:
    """Fuse Vehicle A and Vehicle B boxes into A's frame.

    Args:
        boxes_a: Vehicle A 3D boxes (already in A's frame).
        boxes_b: Vehicle B 3D boxes (in B's frame).
        T_b_to_a: 4x4 transform mapping B's camera frame to A's.
        stage_cfg: Loaded stage4.yaml config dict.

    Returns:
        (fused_boxes, stats) where stats has n_a, n_b, n_fused,
        n_dynamic_flagged, n_only_a, n_only_b.

    A matched pair whose post-registration BEV displacement is below
    ``static_filter.max_displacement_m`` is fused. Above it, the pair is kept
    unmerged and flagged ``is_dynamic`` — for simultaneous V2V a large
    displacement signals a bad match or pose error rather than a moving object.
    """
    max_dist      = stage_cfg["matching"]["max_dist"]
    static_thresh = float(stage_cfg["static_filter"]["max_displacement_m"])

    boxes_b_reg = [transform_box(b, T_b_to_a) for b in boxes_b]
    matches, unmatched_a, unmatched_b = match_boxes(boxes_a, boxes_b_reg, max_dist)

    out: list[dict] = []
    n_fused = n_dynamic = 0

    for i, j in matches:
        a, b = boxes_a[i], boxes_b_reg[j]
        displacement = bev_distance(a, b)
        if displacement < static_thresh:
            out.append(merge_static_pair(a, b))
            n_fused += 1
        else:
            # displacement too large to be the same object — keep both,
            # flagged, unmerged (bad match / pose error, or a mover).
            out.append(_tagged(a, "vehicle_A", True))
            out.append(_tagged(b, "vehicle_B", True))
            n_dynamic += 1

    for i in unmatched_a:
        out.append(_tagged(boxes_a[i], "vehicle_A", False))
    for j in unmatched_b:
        out.append(_tagged(boxes_b_reg[j], "vehicle_B", False))

    stats = {
        "n_a":               len(boxes_a),
        "n_b":               len(boxes_b),
        "n_fused":           n_fused,
        "n_dynamic_flagged": n_dynamic,
        "n_only_a":          len(unmatched_a),
        "n_only_b":          len(unmatched_b),
    }
    return out, stats
