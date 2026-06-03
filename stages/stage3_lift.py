"""Stage 3 — Lift 2D Detections to 3D Bounding Boxes.

Combines Stage 1 disparity + Stage 2 detections + calibration to produce
3D bounding boxes via geometric unprojection.

Usage:
    python stages/stage3_lift.py --sample_id 000000 --method sgbm
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.calib import extract_stereo_params
from utils.config_loader import load_configs
from utils.geometry import unproject_box
from utils.kitti_loader import load_calib

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------

def sample_depth(
    disp: np.ndarray,
    box2d: dict,
    focal_length: float,
    baseline: float,
    method: str = "percentile_75",
    crop_top_frac: float = 1.0,
    min_depth_m: float | None = None,
) -> tuple[float, int]:
    """Sample metric depth from disparity map within a 2D box ROI.

    Two optional filters remove ground/road contamination before sampling:
        crop_top_frac — keeps only the top fraction of the box height,
            discarding ground pixels that appear in the lower portion.
        min_depth_m — nulls out pixels with disparity > f*B/min_depth_m,
            removing implausibly-close hits (road surface, nearby clutter).
    Both are no-ops at their defaults (1.0 / None).

    valid_pixel_count is measured after the vertical crop but before the
    disparity gate, so it remains a stable coverage proxy for confidence.

    Supports 'median' and any 'percentile_N' where N is an integer 0-100.

    Args:
        disp: Disparity map, shape (H, W), float32. np.nan = invalid.
        box2d: Dict with keys x1, y1, x2, y2 (pixel coords).
        focal_length: Camera focal length in pixels.
        baseline: Stereo baseline in metres.
        method: Sampling strategy — 'median' or 'percentile_N' (e.g.
                'percentile_75', 'percentile_90').
        crop_top_frac: Fraction of box height to keep from the top.
                       1.0 = no crop (default). 0.65 = top 65%.
        min_depth_m: Gate out disparity values corresponding to depths
                     closer than this (disparity > f*B/min_depth_m).
                     None = no gate (default).

    Returns:
        Tuple of (Z_metres, valid_pixel_count).

    Raises:
        ValueError: If method is unsupported or percentile out of range.
    """
    x1 = max(0, int(box2d["x1"]))
    y1 = max(0, int(box2d["y1"]))
    x2 = min(disp.shape[1], int(box2d["x2"]))
    y2 = min(disp.shape[0], int(box2d["y2"]))

    # Vertical crop: keep top fraction of box height to reject ground pixels
    if crop_top_frac < 1.0:
        y2 = min(disp.shape[0], y1 + max(1, int((y2 - y1) * crop_top_frac)))

    roi         = disp[y1:y2, x1:x2]
    valid_mask  = ~np.isnan(roi)
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        return float("nan"), 0

    valid_vals = roi[valid_mask]

    # Disparity gate: remove pixels closer than min_depth_m
    if min_depth_m is not None and min_depth_m > 0:
        max_disp   = (focal_length * baseline) / min_depth_m
        valid_vals = valid_vals[valid_vals <= max_disp]
        if len(valid_vals) == 0:
            logger.debug(
                "sample_depth: gate removed all %d valid pixels "
                "(min_depth_m=%.1fm → max_disp=%.1fpx)",
                valid_count, min_depth_m, max_disp,
            )
            return float("nan"), valid_count

    if method == "median":
        d_sample = float(np.median(valid_vals))
    elif method.startswith("percentile_"):
        try:
            p = int(method.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid percentile method '{method}'. "
                f"Expected format: 'percentile_N' where N is 0-100."
            )
        if not (0 <= p <= 100):
            raise ValueError(
                f"Percentile {p} out of range. Must be between 0 and 100."
            )
        d_sample = float(np.percentile(valid_vals, p))
    else:
        raise ValueError(
            f"Unsupported depth sampling method: '{method}'. "
            f"Supported: 'median' or 'percentile_N' (e.g. 'percentile_75')."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (focal_length * baseline) / d_sample

    return float(Z), valid_count


# ---------------------------------------------------------------------------
# Lifting pipeline
# ---------------------------------------------------------------------------

def lift_boxes(
    boxes2d: list[dict],
    disp: np.ndarray,
    calib: dict,
    stage_cfg: dict,
) -> tuple[list[dict], dict]:
    """Lift all 2D detections to 3D positions.

    Each lifted box carries the geometrically reliable quantities only:
    the unprojected 3D center (x, y, z) and the source 2D box. Size and
    heading are deliberately not estimated — they are not recoverable from
    stereo geometry at range (see project notes).

    Args:
        boxes2d: List of 2D detection dicts from Stage 2.
        disp: Disparity map, shape (H, W), float32.
        calib: Calibration dict from load_calib or load_tracking_calib.
        stage_cfg: Loaded stage3.yaml config dict.

    Returns:
        Tuple of (boxes3d, skip_reason_counts).
    """
    P2                       = calib["P2"]
    focal_length, baseline   = extract_stereo_params(calib)
    min_valid_px             = int(stage_cfg["min_valid_pixels"])
    depth_sampling           = stage_cfg["depth_sampling"]
    crop_top_frac            = float(stage_cfg.get("crop_top_frac", 1.0))
    min_depth_m              = stage_cfg.get("min_depth_m", None)

    boxes3d:      list[dict]      = []
    skip_reasons: dict[str, int]  = {}

    for box2d in boxes2d:
        label    = box2d["label"]
        conf_2d  = box2d["confidence"]
        box_area = max(
            (box2d["x2"] - box2d["x1"]) * (box2d["y2"] - box2d["y1"]), 1
        )

        Z, valid_count = sample_depth(
            disp, box2d, focal_length, baseline,
            method=depth_sampling,
            crop_top_frac=crop_top_frac,
            min_depth_m=min_depth_m,
        )

        if valid_count < min_valid_px:
            reason = "insufficient_depth"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            logger.warning("Skipping %s — only %d valid depth pixels (min=%d)",
                           label, valid_count, min_valid_px)
            continue

        if not np.isfinite(Z) or Z <= 0:
            reason = "invalid_depth"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            logger.warning("Skipping %s — invalid depth Z=%.4f", label, Z)
            continue

        X, Y_center, Z, _, _ = unproject_box(box2d, Z, P2)

        coverage_ratio = valid_count / box_area
        confidence_3d  = round(conf_2d * coverage_ratio, 4)

        boxes3d.append({
            "label":      label,
            "confidence": confidence_3d,
            "x":          round(X,        3),
            "y":          round(Y_center, 3),
            "z":          round(Z,        3),
            "x1": box2d["x1"],
            "y1": box2d["y1"],
            "x2": box2d["x2"],
            "y2": box2d["y2"],
        })

    return boxes3d, skip_reasons


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    sample_id: str,
    base_cfg: dict,
    stage_cfg: dict,
    method: str | None = None,
    disp: np.ndarray | None = None,
    boxes2d: list[dict] | None = None,
    calib: dict | None = None,
    output_dir_override: str | None = None,
) -> dict:
    """Run Stage 3 for a single sample.

    Args:
        sample_id: Zero-padded 6-digit KITTI sample ID or tracking frame ID.
        base_cfg: Loaded base.yaml config dict.
        stage_cfg: Loaded stage3.yaml config dict.
        method: Depth method used ('sgbm' | 'waft'). Determines disp source
                path when disp is not provided directly.
        disp: Optional pre-computed disparity array. Skips disk load.
        boxes2d: Optional pre-computed 2D detections. Skips disk load.
        calib: Optional pre-loaded calibration dict. Skips load_calib.
        output_dir_override: Optional full output directory override.

    Returns:
        Dict with keys: sample_id, boxes, n_input_boxes, n_skipped,
        skip_reason, output_path.

    Raises:
        FileNotFoundError: If required Stage 1 or Stage 2 outputs are missing.
    """
    data_root = base_cfg["data"]["data_root"]
    split     = base_cfg["data"]["split"]

    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else Path(stage_cfg["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 3 | sample=%s ===", sample_id)

    # Load Stage 1 output if not provided
    if disp is None:
        if method is None:
            method = stage_cfg.get("method_depth", "sgbm")
        disp_path = Path(f"outputs/depth/object/{method}") / f"{sample_id}_disp.npy"
        if not disp_path.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found: {disp_path}\n"
                f"Run: python stages/stage1_depth.py "
                f"--sample_id {sample_id} --method {method}"
            )
        disp = np.load(str(disp_path))
    logger.info("Loaded disparity — shape=%s", disp.shape)

    # Load Stage 2 output if not provided
    if boxes2d is None:
        det_path = Path("outputs/detections/object") / f"{sample_id}_boxes2d.json"
        if not det_path.exists():
            raise FileNotFoundError(
                f"Stage 2 output not found: {det_path}\n"
                f"Run: python stages/stage2_detect.py --sample_id {sample_id}"
            )
        with open(det_path) as f:
            boxes2d = json.load(f)["boxes"]
    logger.info("Loaded %d 2D detections", len(boxes2d))

    if calib is None:
        calib = load_calib(data_root, split, sample_id)

    boxes3d, skip_reasons = lift_boxes(boxes2d, disp, calib, stage_cfg)

    n_input   = len(boxes2d)
    n_skipped = sum(skip_reasons.values())
    logger.info("Lifting complete — %d input, %d lifted, %d skipped",
                n_input, len(boxes3d), n_skipped)

    output = {
        "sample_id":     sample_id,
        "method":        stage_cfg["method"],
        "n_input_boxes": n_input,
        "n_skipped":     n_skipped,
        "skip_reason":   skip_reasons,
        "boxes":         boxes3d,
    }
    output_path = output_dir / f"{sample_id}_boxes3d.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved %d 3D boxes → %s", len(boxes3d), output_path)

    return {**output, "output_path": output_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 3 — Lift to 3D")
    parser.add_argument("--sample_id",    required=True)
    parser.add_argument("--base_config",  default="config/base.yaml")
    parser.add_argument("--stage_config", default="config/stage3.yaml")
    parser.add_argument("--method",       default=None,
                        help="Depth method: sgbm | waft. "
                             "Determines disp source path.")
    return parser.parse_args()


if __name__ == "__main__":
    import mlflow

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    base_cfg, stage_cfg = load_configs(args.base_config, args.stage_config)

    method = args.method or stage_cfg.get("method_depth", "sgbm")

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage3_lift")

    with mlflow.start_run(run_name=f"{method}_{args.sample_id}"):
        mlflow.log_param("sample_id",      args.sample_id)
        mlflow.log_param("method",         stage_cfg["method"])
        mlflow.log_param("depth_method",   method)
        mlflow.log_param("depth_sampling", stage_cfg["depth_sampling"])
        mlflow.log_param("min_valid_px",   stage_cfg["min_valid_pixels"])

        result = run(args.sample_id, base_cfg, stage_cfg, method=method)

        mlflow.log_metric("n_input_boxes", result["n_input_boxes"])
        mlflow.log_metric("n_lifted",      len(result["boxes"]))
        mlflow.log_metric("n_skipped",     result["n_skipped"])

        for reason, count in result["skip_reason"].items():
            mlflow.log_metric(f"skip_{reason}", count)

        if result["boxes"]:
            depths = [b["z"] for b in result["boxes"]]
            mlflow.log_metric("mean_z", round(float(np.mean(depths)), 3))
            mlflow.log_metric("min_z",  round(float(np.min(depths)),  3))
            mlflow.log_metric("max_z",  round(float(np.max(depths)),  3))

        logger.info("Stage 3 complete — MLflow run logged.")
