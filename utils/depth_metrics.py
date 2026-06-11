"""Shared Stage-1 disparity metrics (EPE, D1, coverage).

Pure metric computation over a predicted disparity map and its ground truth,
kept here so both the KITTI validator (``stages.validate_stage1_depth``) and the
CARLA detector path (``stages.stage4_fusion``) compute depth quality the same
way — production code never imports metrics from a ``validate_*`` script.

Both maps are float32 with ``np.nan`` marking invalid pixels; metrics are taken
over the pixels valid in both.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_epe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute End-Point Error over mutually valid pixels.

    Args:
        pred: Predicted disparity, shape (H, W), float32. np.nan = invalid.
        gt:   GT disparity, shape (H, W), float32. np.nan = invalid.

    Returns:
        EPE in pixels, or nan if no valid pixels.
    """
    valid = ~np.isnan(gt) & ~np.isnan(pred)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def compute_d1(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 3.0,
) -> float:
    """Compute D1 — percentage of pixels with error above threshold.

    Args:
        pred: Predicted disparity, shape (H, W), float32. np.nan = invalid.
        gt:   GT disparity, shape (H, W), float32. np.nan = invalid.
        threshold: Error threshold in pixels (default 3.0, KITTI convention).

    Returns:
        D1 percentage (0–100), or nan if no valid pixels.
    """
    valid = ~np.isnan(gt) & ~np.isnan(pred)
    if not valid.any():
        return float("nan")
    errors = np.abs(pred[valid] - gt[valid])
    return float(np.sum(errors > threshold) / valid.sum() * 100.0)


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Run all metrics for one sample.

    Args:
        pred: Predicted disparity, shape (H, W), float32.
        gt:   GT disparity, shape (H, W), float32.

    Returns:
        Dict with keys: epe, d1, coverage, valid_gt_px, evaluated_px.
    """
    valid    = ~np.isnan(gt) & ~np.isnan(pred)
    gt_count = int((~np.isnan(gt)).sum())

    if not valid.any():
        return {
            "epe": float("nan"), "d1": float("nan"),
            "coverage": 0.0, "valid_gt_px": gt_count, "evaluated_px": 0,
        }

    epe      = compute_epe(pred, gt)
    d1       = compute_d1(pred, gt)
    coverage = float(valid.sum()) / gt_count * 100.0 if gt_count > 0 else 0.0

    logger.info(
        "Metrics — EPE=%.4fpx | D1=%.2f%% | coverage=%.1f%% | %d/%d GT px",
        epe, d1, coverage, int(valid.sum()), gt_count,
    )
    return {
        "epe":          epe,
        "d1":           d1,
        "coverage":     coverage,
        "valid_gt_px":  gt_count,
        "evaluated_px": int(valid.sum()),
    }
