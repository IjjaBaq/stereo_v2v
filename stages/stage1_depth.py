"""Stage 1 — Stereo Depth Estimation.

Computes a disparity map from a rectified stereo pair using the selected
method (sgbm | waft), converts it to metric depth via Z = f*B/d, and
saves outputs to outputs/depth/object/{method}/.

Usage:
    python stages/stage1_depth.py --sample_id 000000 --method sgbm
    python stages/stage1_depth.py --sample_id 000000 --method waft
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import mlflow
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.calib import extract_stereo_params
from utils.kitti_loader import load_calib, load_image

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_configs(base_path: str, stage_path: str) -> tuple[dict, dict]:
    """Load and return base and stage YAML configs.

    Args:
        base_path: Path to config/base.yaml.
        stage_path: Path to config/stage1.yaml.

    Returns:
        Tuple of (base_cfg, stage_cfg) as plain dicts.

    Raises:
        FileNotFoundError: If either config file is missing.
    """
    for p in (base_path, stage_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Config file not found: {p}")
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(stage_path) as f:
        stage_cfg = yaml.safe_load(f)
    return base_cfg, stage_cfg


# ---------------------------------------------------------------------------
# Disparity methods
# ---------------------------------------------------------------------------

def compute_disparity_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Compute disparity map using OpenCV StereoSGBM.

    Args:
        left: Left image, shape (H, W, 3), uint8 BGR.
        right: Right image, shape (H, W, 3), uint8 BGR.
        cfg: SGBM parameter dict from stage1.yaml['sgbm'].

    Returns:
        Disparity map, shape (H, W), float32.
        Invalid/occluded pixels are set to np.nan.
    """
    mode_map = {
        "HH":    cv2.StereoSGBM_MODE_HH,
        "SGBM":  cv2.StereoSGBM_MODE_SGBM,
        "3WAY":  cv2.StereoSGBM_MODE_SGBM_3WAY,
        "HH4":   cv2.StereoSGBM_MODE_HH4,
    }
    mode_str = str(cfg.get("mode", "HH")).upper()
    mode     = mode_map.get(mode_str, cv2.StereoSGBM_MODE_HH)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=cfg["min_disparity"],
        numDisparities=cfg["num_disparities"],
        blockSize=cfg["block_size"],
        P1=cfg["p1"],
        P2=cfg["p2"],
        disp12MaxDiff=cfg["disp12_max_diff"],
        preFilterCap=cfg["pre_filter_cap"],
        uniquenessRatio=cfg["uniqueness_ratio"],
        speckleWindowSize=cfg["speckle_window_size"],
        speckleRange=cfg["speckle_range"],
        mode=mode,
    )

    raw  = sgbm.compute(left, right).astype(np.float32) / 16.0
    disp = raw.copy()
    disp[raw <= 0] = np.nan

    logger.info("SGBM — valid pixel ratio: %.3f",
                float(np.sum(~np.isnan(disp))) / disp.size)
    return disp


# ---------------------------------------------------------------------------
# Depth conversion
# ---------------------------------------------------------------------------

def disparity_to_depth(
    disp: np.ndarray,
    focal_length: float,
    baseline: float,
    max_depth_m: float | None = None,
) -> np.ndarray:
    """Convert disparity map to metric depth using Z = f*B/d.

    Args:
        disp: Disparity map, shape (H, W), float32. np.nan = invalid.
        focal_length: Camera focal length in pixels.
        baseline: Stereo baseline in metres.
        max_depth_m: Optional depth clip in metres. None = no clip.

    Returns:
        Depth map, shape (H, W), float32, metres. Invalid pixels = np.nan.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (focal_length * baseline) / disp

    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0]          = np.nan

    if max_depth_m is not None:
        clipped = int(np.sum(depth > max_depth_m))
        depth[depth > max_depth_m] = np.nan
        logger.info("Depth clipping at %.1fm — clipped %d pixels",
                    max_depth_m, clipped)

    valid = ~np.isnan(depth)
    if valid.any():
        logger.info("Depth stats — min=%.2fm max=%.2fm mean=%.2fm",
                    float(np.nanmin(depth)),
                    float(np.nanmax(depth)),
                    float(np.nanmean(depth)))
    return depth


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    """Render a float32 disparity map as a uint8 BGR colormap image.

    Args:
        disp: Disparity map, shape (H, W), float32. np.nan = invalid.

    Returns:
        Colorized disparity image, shape (H, W, 3), uint8.
        Invalid pixels are rendered black.
    """
    valid_mask = ~np.isnan(disp)
    disp_vis   = np.zeros_like(disp)

    if valid_mask.any():
        d_min = float(np.nanmin(disp))
        d_max = float(np.nanmax(disp))
        if d_max > d_min:
            disp_vis[valid_mask] = (
                (disp[valid_mask] - d_min) / (d_max - d_min) * 255.0
            )

    colored = cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_MAGMA)
    colored[~valid_mask] = 0
    return colored


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    sample_id: str,
    base_cfg: dict,
    stage_cfg: dict,
    method: str | None = None,
    image_left: np.ndarray | None = None,
    image_right: np.ndarray | None = None,
    calib: dict | None = None,
    output_dir_override: str | None = None,
) -> dict:
    """Run Stage 1 for a single sample.

    Args:
        sample_id: Zero-padded 6-digit KITTI sample ID or tracking frame ID.
        base_cfg: Loaded base.yaml config dict.
        stage_cfg: Loaded stage1.yaml config dict.
        method: Override method ('sgbm' | 'waft'). If None, uses stage_cfg.
        image_left: Optional pre-loaded left BGR image. Skips load_image.
        image_right: Optional pre-loaded right BGR image. Skips load_image.
        calib: Optional pre-loaded calibration dict. Skips load_calib.
        output_dir_override: Optional output directory. Overrides default.

    Returns:
        Dict with keys: disp, depth, disp_path, disp_png, method.

    Raises:
        ValueError: If method is unsupported or waft used without offline files.
        FileNotFoundError: If required inputs are missing.
    """
    data_root = base_cfg["data"]["data_root"]
    split     = base_cfg["data"]["split"]
    method    = (method or stage_cfg["method"]).lower()

    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else Path(f"./outputs/depth/object/{method}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 1 | sample=%s | method=%s ===", sample_id, method)

    if calib is None:
        calib = load_calib(data_root, split, sample_id)
    focal_length, baseline = extract_stereo_params(calib)

    if method == "sgbm":
        if image_left is None:
            image_left  = load_image(data_root, split, "image_2",
                                     sample_id, suffix="_10.png")
        if image_right is None:
            image_right = load_image(data_root, split, "image_3",
                                     sample_id, suffix="_10.png")
        disp = compute_disparity_sgbm(image_left, image_right, stage_cfg["sgbm"])

    elif method == "waft":
        precomputed_dir = Path(stage_cfg["precomputed_dir"])
        disp_src        = precomputed_dir / f"{sample_id}_disp.npy"
        if not disp_src.exists():
            raise FileNotFoundError(
                f"Pre-computed WAFT disparity not found: {disp_src}\n"
                f"Run scripts/precompute_waft_disparity.py on Colab first, "
                f"then download results to {precomputed_dir}"
            )
        disp        = np.load(str(disp_src))
        valid_ratio = float(np.sum(~np.isnan(disp))) / disp.size
        logger.info("WAFT — loaded %s | valid pixel ratio: %.3f",
                    disp_src, valid_ratio)

    else:
        raise ValueError(
            f"Unsupported method: '{method}'. Supported: ['sgbm', 'waft']."
        )

    max_depth_m = stage_cfg.get("max_depth_m", None)
    depth       = disparity_to_depth(disp, focal_length, baseline, max_depth_m)

    disp_npy_path = output_dir / f"{sample_id}_disp.npy"
    disp_png_path = output_dir / f"{sample_id}_disp.png"

    np.save(str(disp_npy_path), disp)
    cv2.imwrite(str(disp_png_path), colorize_disparity(disp))

    logger.info("Saved disparity → %s", disp_npy_path)

    return {
        "disp":      disp,
        "depth":     depth,
        "disp_path": disp_npy_path,
        "disp_png":  disp_png_path,
        "method":    method,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 1 — Stereo Depth Estimation")
    parser.add_argument("--sample_id",    required=True,
                        help="6-digit KITTI sample ID")
    parser.add_argument("--base_config",  default="config/base.yaml")
    parser.add_argument("--stage_config", default="config/stage1.yaml")
    parser.add_argument("--method",       default=None,
                        help="Override method: sgbm | waft")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    base_cfg, stage_cfg = load_configs(args.base_config, args.stage_config)

    method = (args.method or stage_cfg["method"]).lower()

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage1_depth")

    with mlflow.start_run(run_name=f"{method}_{args.sample_id}"):
        mlflow.log_param("sample_id", args.sample_id)
        mlflow.log_param("method",    method)

        if method == "sgbm":
            for k, v in stage_cfg["sgbm"].items():
                mlflow.log_param(f"sgbm_{k}", v)
        elif method == "waft":
            mlflow.log_param("precomputed_dir", stage_cfg["precomputed_dir"])

        result = run(args.sample_id, base_cfg, stage_cfg, method=method)

        mlflow.log_param("disp_npy", str(result["disp_path"]))
        mlflow.log_param("disp_png", str(result["disp_png"]))

        depth = result["depth"]
        valid = ~np.isnan(depth)
        if valid.any():
            mlflow.log_metric("depth_mean_m",   float(np.nanmean(depth)))
            mlflow.log_metric("depth_min_m",    float(np.nanmin(depth)))
            mlflow.log_metric("depth_max_m",    float(np.nanmax(depth)))
            mlflow.log_metric("valid_px_ratio", float(valid.sum()) / valid.size)

        logger.info("Stage 1 complete — MLflow run logged.")
