"""Stage 1 Validation — Stereo Depth Estimation.

Compares predicted disparity maps against KITTI GT (disp_noc_0/) and
computes EPE and D1 metrics.

Usage:
    python stages/validate_stage1_depth.py --sample_ids 000000 000001 --method sgbm
    python stages/validate_stage1_depth.py --sample_ids 000000 000001 --method waft
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import mlflow
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stages.stage1_depth import load_configs, run as run_stage1
from utils.kitti_loader import load_disparity_gt

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_epe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute End-Point Error over mutually valid pixels.

    Args:
        pred: Predicted disparity, shape (H, W), float32.
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
        pred: Predicted disparity, shape (H, W), float32.
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_side_by_side(
    pred: np.ndarray,
    gt: np.ndarray,
    sample_id: str,
    epe: float,
    d1: float,
    method: str,
) -> np.ndarray:
    """Render predicted and GT disparity side-by-side with metrics overlay.

    Args:
        pred: Predicted disparity, shape (H, W), float32.
        gt:   GT disparity, shape (H, W), float32.
        sample_id: For title overlay.
        epe: EPE value.
        d1: D1 value.
        method: Method name for label.

    Returns:
        Side-by-side BGR image, shape (H, 2*W, 3), uint8.
    """
    combined = np.concatenate([
        pred[~np.isnan(pred)].ravel(),
        gt[~np.isnan(gt)].ravel(),
    ])
    d_min = float(combined.min()) if combined.size else 0.0
    d_max = float(combined.max()) if combined.size else 1.0

    def _colorize(disp: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(disp)
        norm  = np.zeros_like(disp)
        if d_max > d_min:
            norm[valid] = (disp[valid] - d_min) / (d_max - d_min) * 255.0
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        colored[~valid] = 0
        return colored

    pred_vis = _colorize(pred)
    gt_vis   = _colorize(gt)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    cv2.putText(pred_vis,
                f"{method.upper()}  EPE={epe:.2f}px  D1={d1:.1f}%",
                (10, 25), font, 0.6, white, 1, cv2.LINE_AA)
    cv2.putText(gt_vis,
                f"Ground Truth  [{sample_id}]",
                (10, 25), font, 0.6, white, 1, cv2.LINE_AA)

    return np.concatenate([pred_vis, gt_vis], axis=1)


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def validate_sample(
    sample_id: str,
    base_cfg: dict,
    stage_cfg: dict,
    method: str,
) -> dict:
    """Validate one sample — load predicted disparity, compare to GT.

    Runs Stage 1 first if disparity not already on disk.

    Args:
        sample_id: Zero-padded 6-digit KITTI sample ID.
        base_cfg: Loaded base.yaml config dict.
        stage_cfg: Loaded stage1.yaml config dict.
        method: 'sgbm' | 'waft'.

    Returns:
        Dict with keys: sample_id, epe, d1, coverage, valid_gt_px, evaluated_px.
    """
    output_dir = Path(f"./outputs/depth/object/{method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    disp_path = output_dir / f"{sample_id}_disp.npy"
    if not disp_path.exists():
        logger.info("Disparity not found for %s — running Stage 1.", sample_id)
        run_stage1(sample_id, base_cfg, stage_cfg, method=method)

    pred = np.load(str(disp_path))

    data_root = base_cfg["data"]["data_root"]
    split     = base_cfg["data"]["split"]
    gt        = load_disparity_gt(data_root, split, sample_id)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch for {sample_id}: pred={pred.shape} gt={gt.shape}"
        )

    metrics = evaluate(pred, gt)
    metrics["sample_id"] = sample_id

    vis      = make_side_by_side(pred, gt, sample_id,
                                 epe=metrics["epe"], d1=metrics["d1"],
                                 method=method)
    vis_path = output_dir / f"{sample_id}_val.png"
    cv2.imwrite(str(vis_path), vis)
    logger.info("Saved validation visualization → %s", vis_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 1 Validation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample_id",  type=str,
                       help="Single 6-digit sample ID")
    group.add_argument("--sample_ids", type=str, nargs="+",
                       help="Multiple sample IDs")
    parser.add_argument("--base_config",  default="config/base.yaml")
    parser.add_argument("--stage_config", default="config/stage1.yaml")
    parser.add_argument("--method",       default=None,
                        help="sgbm | waft (default: from config)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    base_cfg, stage_cfg = load_configs(args.base_config, args.stage_config)

    method     = (args.method or stage_cfg["method"]).lower()
    sample_ids = [args.sample_id] if args.sample_id else args.sample_ids

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage1_depth_validation")

    all_results: list[dict] = []

    with mlflow.start_run(
        run_name=f"val_{method}_{len(sample_ids)}samples"
    ):
        mlflow.log_param("method",     method)
        mlflow.log_param("n_samples",  len(sample_ids))
        mlflow.log_param("sample_ids", " ".join(sample_ids))

        for sid in sample_ids:
            try:
                result = validate_sample(sid, base_cfg, stage_cfg, method)
                all_results.append(result)
            except Exception as e:
                logger.error("Failed on sample %s: %s", sid, e)
                all_results.append({"sample_id": sid, "error": str(e)})

        valid_results = [r for r in all_results if "error" not in r]

        summary: dict = {
            "method":   method,
            "n_samples": len(valid_results),
        }

        if valid_results:
            mean_epe      = float(np.mean([r["epe"]      for r in valid_results]))
            mean_d1       = float(np.mean([r["d1"]       for r in valid_results]))
            mean_coverage = float(np.mean([r["coverage"] for r in valid_results]))

            summary["mean_epe"]      = mean_epe
            summary["mean_d1"]       = mean_d1
            summary["mean_coverage"] = mean_coverage

            mlflow.log_metric("mean_epe",      mean_epe)
            mlflow.log_metric("mean_d1",       mean_d1)
            mlflow.log_metric("mean_coverage", mean_coverage)

            logger.info(
                "=== Validation Summary [%s] === "
                "EPE=%.4fpx | D1=%.2f%% | coverage=%.1f%% | n=%d",
                method, mean_epe, mean_d1, mean_coverage, len(valid_results),
            )

        summary["samples"] = all_results

        results_path = Path(f"./outputs/depth/object/{method}") / \
                       "validation_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Validation results saved → %s", results_path)
