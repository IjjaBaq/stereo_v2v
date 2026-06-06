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
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stages.stage1_depth import load_configs, load_waft_model, run as run_stage1
from utils.kitti_loader import load_disparity_gt, load_image
from utils.validation_io import merge_samples

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
    left_img: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    sample_id: str,
    epe: float,
    d1: float,
    method: str,
) -> np.ndarray:
    """Render a 3-panel validation figure for one sample.

    Layout (top to bottom):
        - Top row, full width: the original left camera image.
        - Bottom left:  predicted disparity, colorized.
        - Bottom right: GT disparity, colorized.

    Predicted and GT disparity share a single colour scale so they are
    directly comparable. The top image is resized to the disparity row's
    width (2*W) and height (H) so all three panels are the same height.

    Args:
        left_img: Left camera image, shape (H, W, 3), uint8 BGR.
        pred: Predicted disparity, shape (H, W), float32.
        gt:   GT disparity, shape (H, W), float32.
        sample_id: For the title overlay.
        epe: EPE value.
        d1: D1 value.
        method: Method name for label.

    Returns:
        Stacked BGR figure, shape (2*H, 2*W, 3), uint8.
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

    def _label(img: np.ndarray, text: str, scale: float = 0.6) -> None:
        # Black outline under white text so labels read on any background.
        cv2.putText(img, text, (10, 25), font, scale, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(img, text, (10, 25), font, scale, white,       1, cv2.LINE_AA)

    _label(pred_vis, f"{method.upper()}  EPE={epe:.2f}px  D1={d1:.1f}%")
    _label(gt_vis,   "Ground Truth")

    bottom = np.concatenate([pred_vis, gt_vis], axis=1)

    # Top row spans the full figure width (2*W) at the disparity row's height
    # (H), so all three panels are the same height.
    top_h, top_w = bottom.shape[:2]
    top = cv2.resize(left_img, (top_w, top_h), interpolation=cv2.INTER_AREA)
    _label(top, f"Input Image [{sample_id}]", scale=0.9)

    return np.concatenate([top, bottom], axis=0)


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
        Dict with keys: sample_id, epe, d1, coverage, valid_gt_px, evaluated_px,
        runtime_s. runtime_s is the wall-clock from image load to .npy save
        (model load excluded — caller pre-warms it), or None if the disparity
        was already cached on disk and not recomputed this run.
    """
    output_dir = Path(f"./outputs/depth/object/{method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    disp_path = output_dir / f"{sample_id}_disp.npy"
    runtime_s = None
    if not disp_path.exists():
        logger.info("Disparity not found for %s — running Stage 1.", sample_id)
        # Wall-clock for one image: load → compute → .npy save. The model is
        # pre-warmed (cached) by the caller, so this excludes model-load time.
        t0 = time.perf_counter()
        run_stage1(sample_id, base_cfg, stage_cfg, method=method)
        runtime_s = time.perf_counter() - t0

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
    metrics["runtime_s"] = (
        round(runtime_s, 4) if runtime_s is not None else None
    )

    left_img = load_image(data_root, split, "image_2", sample_id)
    vis      = make_side_by_side(left_img, pred, gt, sample_id,
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

    # Pre-warm the depth model once so per-image runtime excludes model-load
    # time (the model is cached across samples). SGBM has no model to load.
    if method == "waft":
        logger.info("Pre-warming WAFT model (excluded from per-image runtime)...")
        load_waft_model(stage_cfg["waft_config_path"], stage_cfg["waft_model_path"])

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage1_depth_validation")

    all_results: list[dict] = []

    with mlflow.start_run(
        run_name=f"val_{method}_{len(sample_ids)}samples"
    ):
        mlflow.log_param("method",     method)
        mlflow.log_param("n_samples",  len(sample_ids))
        mlflow.log_param("sample_ids", " ".join(sample_ids))

        for i, sid in enumerate(sample_ids):
            try:
                result = validate_sample(sid, base_cfg, stage_cfg, method)
                all_results.append(result)
                if result.get("runtime_s") is not None:
                    mlflow.log_metric("runtime_s", result["runtime_s"], step=i)
            except Exception as e:
                logger.error("Failed on sample %s: %s", sid, e)
                all_results.append({"sample_id": sid, "error": str(e)})

        results_path = Path(f"./outputs/depth/object/{method}") / \
                       "validation_results.json"

        # Merge this run's samples into any results already on disk, then
        # recompute aggregates over the full accumulated set.
        merged_samples = merge_samples(
            results_path, all_results, id_key="sample_id", list_key="samples",
        )
        valid_results = [r for r in merged_samples if "error" not in r]

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

            # Mean per-image runtime over samples that were actually computed
            # this run (cached samples carry runtime_s=None and are excluded).
            runtimes = [r["runtime_s"] for r in valid_results
                        if r.get("runtime_s") is not None]
            if runtimes:
                mean_runtime_s = float(np.mean(runtimes))
                summary["mean_runtime_s"] = mean_runtime_s
                mlflow.log_metric("mean_runtime_s", mean_runtime_s)

            logger.info(
                "=== Validation Summary [%s] === "
                "EPE=%.4fpx | D1=%.2f%% | coverage=%.1f%% | n=%d",
                method, mean_epe, mean_d1, mean_coverage, len(valid_results),
            )

        summary["samples"] = merged_samples

        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Validation results saved → %s", results_path)
