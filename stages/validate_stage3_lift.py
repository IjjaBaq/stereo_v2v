"""Stage 3 Validation — End-to-End Pipeline on KITTI Tracking.

Chains Stage 1 → Stage 2 → Stage 3 on KITTI Tracking sequences where
stereo pairs, 3D labels, and calibration are all aligned per frame.

Computes:
    - Mean depth error       : mean |pred_z - gt_z|
    - Mean center distance   : mean sqrt(dx²+dy²+dz²)
    - TP / FP / FN counts    : greedy matching by ascending 3D center
                               distance within class (<= max_dist)

Produces per frame:
    - outputs/depth/tracking/{method}/{seq_id}/{frame_id:06d}_disp.npy
    - outputs/detections/tracking/{seq_id}/{frame_id:06d}_boxes2d.json
    - outputs/lift3d/{method}/{seq_id}/{frame_id:06d}_lift3d.json
    - outputs/lift3d/{method}/{seq_id}/{frame_id:06d}_bev.png
    - outputs/lift3d/{method}/{seq_id}/{frame_id:06d}_2d.png

Produces per run:
    - outputs/lift3d/{method}/{seq_id}/validation_results.json

Usage:
    python stages/validate_stage3_lift.py \\
        --seq_id 0000 \\
        --frame_ids 0 1 2 3 4 \\
        --method sgbm
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stages.stage1_depth import run as run_stage1
from stages.stage2_detect import load_model, run as run_stage2
from stages.stage3_lift import apply_method_overrides, load_configs, run as run_stage3
from utils.geometry import center_distance
from utils.kitti_tracking_loader import load_tracking_frame
from utils.validation_io import merge_samples
from utils.visualization import (
    make_2d_overlay_visualization,
    make_bev_visualization,
    make_detection_overlay,
)

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

KITTI_CLASSES = ("Car", "Pedestrian")

# Per-method depth-sampling parameters (depth_sampling, crop_top_frac,
# min_depth_m) now live in config/stage3.yaml under `per_method_overrides` and
# are applied via stage3_lift.apply_method_overrides(cfg, method) — single
# source of truth shared with the Stage-4 CARLA detector path.
#
# Tuned on a multi-sequence static-car sweep (KITTI tracking 0000-0004; see
# experiments/percentile_choice.md). The two methods need OPPOSITE percentiles
# because their valid-pixel distributions differ:
#  - SGBM is sparse: the left-right consistency check nulls smooth background to
#    NaN, so valid pixels already sit on the car's near surface. percentile_20
#    lands nearest the box centre; the old percentile_75 over-corrected for a
#    background contamination SGBM had already removed (MAE 4.27m@p75 vs 1.72m@p20).
#  - WAFT is dense: a top-40% crop + 6m min-depth gate remove most ground, then
#    percentile_35 matches centre depth with near-zero bias.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_all_configs(
    base_path: str,
    stage1_path: str,
    stage2_path: str,
    stage3_path: str,
) -> tuple[dict, dict, dict, dict]:
    """Load all stage configs.

    Args:
        base_path: Path to config/base.yaml.
        stage1_path: Path to config/stage1.yaml.
        stage2_path: Path to config/stage2.yaml.
        stage3_path: Path to config/stage3.yaml.

    Returns:
        Tuple of (base_cfg, stage1_cfg, stage2_cfg, stage3_cfg).

    Raises:
        FileNotFoundError: If any config file is missing.
    """
    configs = []
    for p in (base_path, stage1_path, stage2_path, stage3_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p) as f:
            configs.append(yaml.safe_load(f))
    return tuple(configs)


# ---------------------------------------------------------------------------
# GT conversion
# ---------------------------------------------------------------------------

def gt_label_to_position3d(obj: dict) -> dict:
    """Convert a KITTI tracking label to a 3D position dict.

    KITTI stores y = bottom of object in camera coordinates (Y points down).
    Convert to center Y before computing metrics. Stage 3 emits position +
    2D box only, so size (l, w, h) and heading are dropped here to match.

    Args:
        obj: Label dict from load_tracking_labels with keys
             x, y, z, h, x1, y1, x2, y2, label.

    Returns:
        3D position dict with keys x, y, z, x1, y1, x2, y2, label.
    """
    return {
        "label":   obj["label"],
        "x":       obj["x"],
        "y":       obj["y"] - obj["h"] / 2.0,
        "z":       obj["z"],
        "x1":      obj["x1"],
        "y1":      obj["y1"],
        "x2":      obj["x2"],
        "y2":      obj["y2"],
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_boxes(
    preds: list[dict],
    gts: list[dict],
    max_dist: dict[str, float] | None = None,
) -> tuple[list[tuple], list[int], list[int]]:
    """Greedy matching of predictions to GT by 3D center distance, per class.

    Mirrors utils.fusion.match_boxes: build all same-class (pred, gt) pairs
    within max_dist, sort by ascending 3D center distance, then greedily
    take pairs whose pred and gt are both still unmatched.

    Args:
        preds: Predicted 3D position dicts with x, y, z, label.
        gts:   GT 3D position dicts with x, y, z, label.
        max_dist: Per-class max 3D center distance for a valid match.
                  Missing classes default to 0.0 (never match).

    Returns:
        Tuple of (matches, fp_indices, fn_indices).
        matches:    List of (pred_idx, gt_idx) TP pairs.
        fp_indices: Pred indices with no GT match.
        fn_indices: GT indices with no pred match.
    """
    max_dist = max_dist or {}

    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if pred["label"] != gt["label"]:
                continue
            d = center_distance(pred, gt)
            if d <= float(max_dist.get(pred["label"], 0.0)):
                candidates.append((d, pi, gi))
    candidates.sort()

    matched_pred: set[int]          = set()
    matched_gt:   set[int]          = set()
    matches:      list[tuple[int,int]] = []
    for _d, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matches.append((pi, gi))
        matched_pred.add(pi)
        matched_gt.add(gi)

    fp_indices = [pi for pi in range(len(preds)) if pi not in matched_pred]
    fn_indices = [gi for gi in range(len(gts))   if gi not in matched_gt]
    return matches, fp_indices, fn_indices


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: list[dict],
    gts: list[dict],
    matches: list[tuple[int, int]],
) -> dict:
    """Compute 3D detection metrics over matched TP pairs.

    Args:
        preds: Predicted 3D position dicts.
        gts:   GT 3D position dicts (Y already converted to center).
        matches: List of (pred_idx, gt_idx) TP pairs.

    Returns:
        Dict with keys: mean_depth_err, mean_center_dist.
        Both nan if no matches.
    """
    if not matches:
        return {
            "mean_depth_err":   float("nan"),
            "mean_center_dist": float("nan"),
        }

    depth_errs   = []
    center_dists = []

    for pi, gi in matches:
        pred = preds[pi]
        gt   = gts[gi]

        depth_errs.append(abs(pred["z"] - gt["z"]))
        center_dists.append(center_distance(pred, gt))

    return {
        "mean_depth_err":   float(np.mean(depth_errs)),
        "mean_center_dist": float(np.mean(center_dists)),
    }


# Depth-range bins for the localization breakdown, keyed by GT depth (gt_z).
# Each bin is [lo, hi) metres; the last is open-ended.
DEPTH_BINS = (
    ("0_10m",    0.0,  10.0),
    ("10_20m",  10.0,  20.0),
    ("20_40m",  20.0,  40.0),
    ("40m_plus", 40.0, float("inf")),
)


def build_pair_records(
    preds: list[dict],
    gts: list[dict],
    matches: list[tuple[int, int]],
) -> list[dict]:
    """Per-matched-pair records for per-class and depth-range aggregation.

    Carries just enough per-TP detail that the run summary can pool errors
    across all frames (by class and by GT-depth bin) instead of averaging
    pre-averaged frame means.

    Args:
        preds: Predicted 3D position dicts.
        gts:   GT 3D position dicts (Y already converted to center).
        matches: List of (pred_idx, gt_idx) TP pairs.

    Returns:
        List of dicts with keys: label, gt_z, depth_err, center_dist.
    """
    records = []
    for pi, gi in matches:
        pred, gt = preds[pi], gts[gi]
        records.append({
            "label":       gt["label"],
            "gt_z":        float(gt["z"]),
            "depth_err":   abs(float(pred["z"]) - float(gt["z"])),
            "center_dist": center_distance(pred, gt),
        })
    return records


# ---------------------------------------------------------------------------
# Per-frame validation
# ---------------------------------------------------------------------------

def validate_frame(
    seq_id: str,
    frame_id: int,
    base_cfg: dict,
    stage1_cfg: dict,
    stage2_cfg: dict,
    stage3_cfg: dict,
    method: str,
    processor,
    model,
) -> dict:
    """Run full pipeline and validate against GT for one tracking frame.

    Args:
        seq_id: Zero-padded 4-digit sequence ID e.g. '0000'.
        frame_id: Integer frame index.
        base_cfg: Loaded base.yaml config dict.
        stage1_cfg: Loaded stage1.yaml config dict.
        stage2_cfg: Loaded stage2.yaml config dict.
        stage3_cfg: Loaded stage3.yaml config dict.
        method: Depth method 'sgbm' | 'waft'. Drives depth sampling,
                output paths, and MLflow logging.
        processor: Pre-loaded RT-DETR processor.
        model: Pre-loaded RT-DETR model.

    Returns:
        Dict with seq_id, frame_id, metrics, n_tp, n_fp, n_fn, n_skipped.
    """
    tracking_root = base_cfg["data"]["tracking_root"]

    depth_out = f"outputs/depth/tracking/{method}/{seq_id}"
    det_out   = f"outputs/detections/tracking/{seq_id}"
    lift3d_out = f"outputs/lift3d/{method}/{seq_id}"

    # Load all aligned data for this frame in one call
    frame = load_tracking_frame(tracking_root, "training", seq_id, frame_id)
    calib = frame["calib"]

    # Stage 1 — stereo depth
    s1 = run_stage1(
        sample_id=f"{frame_id:06d}",
        base_cfg=base_cfg,
        stage_cfg=stage1_cfg,
        method=method,
        image_left=frame["left"],
        image_right=frame["right"],
        calib=calib,
        output_dir_override=depth_out,
    )

    # Stage 2 — 2D detection
    s2 = run_stage2(
        sample_id=f"{frame_id:06d}",
        base_cfg=base_cfg,
        stage_cfg=stage2_cfg,
        processor=processor,
        model=model,
        image=frame["left"],
        output_dir_override=det_out,
    )
    
    # Save detection visualization
    make_detection_overlay(
        frame["left"], s2["boxes"], Path(det_out) / f"{frame_id:06d}_det.png",
    )

    # Stage 3 — lift to 3D
    # Apply per-method sampling overrides from config (empirically validated)
    stage3_cfg_run = apply_method_overrides(stage3_cfg, method)
    s3 = run_stage3(
        sample_id=f"{frame_id:06d}",
        base_cfg=base_cfg,
        stage_cfg=stage3_cfg_run,
        disp=s1["disp"],
        boxes2d=s2["boxes"],
        calib=calib,
        output_dir_override=lift3d_out,
    )

    # GT — filter to active classes, convert Y to center
    gt_boxes = [
        gt_label_to_position3d(obj)
        for obj in frame["labels"]
        if obj["label"] in KITTI_CLASSES
    ]

    pred_boxes = s3["positions"]
    max_dist   = stage3_cfg.get("matching", {}).get("max_dist", {})
    matches, fp_idx, fn_idx = match_boxes(pred_boxes, gt_boxes, max_dist)
    metrics = compute_metrics(pred_boxes, gt_boxes, matches)

    n_tp = len(matches)
    n_fp = len(fp_idx)
    n_fn = len(fn_idx)

    # Per-pair / per-label detail for the run-level per-class and depth-range
    # breakdowns (aggregated across all frames in __main__).
    tp_pairs  = build_pair_records(pred_boxes, gt_boxes, matches)
    fp_labels = [pred_boxes[i]["label"] for i in fp_idx]
    fn_labels = [gt_boxes[i]["label"]   for i in fn_idx]

    logger.info(
        "seq=%s frame=%06d — TP=%d FP=%d FN=%d | "
        "depth_err=%.2fm center_dist=%.2fm",
        seq_id, frame_id, n_tp, n_fp, n_fn,
        metrics["mean_depth_err"]   if not math.isnan(metrics["mean_depth_err"])   else -1,
        metrics["mean_center_dist"] if not math.isnan(metrics["mean_center_dist"]) else -1,
    )

    out_dir = Path(lift3d_out)
    make_bev_visualization(
        pred_boxes, gt_boxes, matches,
        frame_id=frame_id, seq_id=seq_id,
        output_path=out_dir / f"{frame_id:06d}_bev.png",
    )
    make_2d_overlay_visualization(
        frame["left"], pred_boxes, gt_boxes, matches,
        frame_id=frame_id, seq_id=seq_id,
        output_path=out_dir / f"{frame_id:06d}_2d.png",
    )

    return {
        "seq_id":    seq_id,
        "frame_id":  frame_id,
        "n_tp":      n_tp,
        "n_fp":      n_fp,
        "n_fn":      n_fn,
        "n_skipped": s3["n_skipped"],
        "tp_pairs":  tp_pairs,
        "fp_labels": fp_labels,
        "fn_labels": fn_labels,
        **metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 3 Validation — end-to-end pipeline on KITTI Tracking"
    )
    parser.add_argument("--seq_id",    required=True,
                        help="4-digit sequence ID e.g. 0000")
    parser.add_argument("--frame_ids", type=int, nargs="+", required=True,
                        help="Frame indices to validate e.g. 0 1 2 3 4")
    parser.add_argument("--method",    default="sgbm",
                        choices=["sgbm", "waft"],
                        help="Depth method: sgbm | waft")
    parser.add_argument("--base_config",   default="config/base.yaml")
    parser.add_argument("--stage1_config", default="config/stage1.yaml")
    parser.add_argument("--stage2_config", default="config/stage2.yaml")
    parser.add_argument("--stage3_config", default="config/stage3.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    base_cfg, stage1_cfg, stage2_cfg, stage3_cfg = load_all_configs(
        args.base_config,
        args.stage1_config,
        args.stage2_config,
        args.stage3_config,
    )

    # Load RT-DETR once — reused across all frames
    logger.info("Loading RT-DETR model...")
    processor, rt_model = load_model(stage2_cfg["model"])

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("tracking_pipeline")

    all_results: list[dict] = []

    # Resolve the per-method Stage-3 params from config for logging/summary.
    method_cfg = apply_method_overrides(stage3_cfg, args.method)

    with mlflow.start_run(
        run_name=f"seq{args.seq_id}_{args.method}_{len(args.frame_ids)}frames"
    ):
        mlflow.log_param("seq_id",         args.seq_id)
        mlflow.log_param("method",         args.method)
        mlflow.log_param("depth_sampling", method_cfg["depth_sampling"])
        mlflow.log_param("crop_top_frac",  method_cfg["crop_top_frac"])
        mlflow.log_param("min_depth_m",    method_cfg["min_depth_m"])
        mlflow.log_param("n_frames",       len(args.frame_ids))
        mlflow.log_param("frame_ids",      str(args.frame_ids))
        mlflow.log_param("stage3_method",  stage3_cfg["method"])

        for fid in args.frame_ids:
            try:
                result = validate_frame(
                    seq_id=args.seq_id,
                    frame_id=fid,
                    base_cfg=base_cfg,
                    stage1_cfg=stage1_cfg,
                    stage2_cfg=stage2_cfg,
                    stage3_cfg=stage3_cfg,
                    method=args.method,
                    processor=processor,
                    model=rt_model,
                )
                all_results.append(result)
            except Exception as e:
                logger.error("Failed seq=%s frame=%06d: %s",
                             args.seq_id, fid, e)
                all_results.append({
                    "seq_id":   args.seq_id,
                    "frame_id": fid,
                    "error":    str(e),
                })

        results_path = (
            Path("outputs/lift3d")
            / args.method
            / args.seq_id
            / "validation_results.json"
        )

        # Merge this run's frames into any results already on disk, then
        # recompute aggregates over the full accumulated set.
        merged_frames = merge_samples(
            results_path, all_results, id_key="frame_id", list_key="frames",
        )
        valid = [r for r in merged_frames if "error" not in r]

        summary: dict = {
            "seq_id":         args.seq_id,
            "method":         args.method,
            "depth_sampling": method_cfg["depth_sampling"],
            "n_frames":       len(valid),
        }

        if valid:
            for metric in ("mean_depth_err", "mean_center_dist"):
                vals = [r[metric] for r in valid
                        if not math.isnan(r.get(metric, float("nan")))]
                agg  = float(np.mean(vals)) if vals else float("nan")
                summary[metric] = agg
                if not math.isnan(agg):
                    mlflow.log_metric(metric, agg)

            for count_key in ("n_tp", "n_fp", "n_fn", "n_skipped"):
                total = sum(r.get(count_key, 0) for r in valid)
                summary[count_key] = total
                mlflow.log_metric(count_key, total)

            # Per-class breakdown — counts summed, errors pooled over all TP
            # pairs of that class (not an average of per-frame means).
            per_class: dict = {}
            for cls in KITTI_CLASSES:
                de = [p["depth_err"]   for r in valid
                      for p in r.get("tp_pairs", []) if p["label"] == cls]
                cd = [p["center_dist"] for r in valid
                      for p in r.get("tp_pairs", []) if p["label"] == cls]
                n_tp_c = len(de)
                n_fp_c = sum(lab == cls for r in valid
                             for lab in r.get("fp_labels", []))
                n_fn_c = sum(lab == cls for r in valid
                             for lab in r.get("fn_labels", []))
                per_class[cls] = {
                    "n_tp":        n_tp_c,
                    "n_fp":        n_fp_c,
                    "n_fn":        n_fn_c,
                    "depth_err":   float(np.mean(de)) if de else None,
                    "center_dist": float(np.mean(cd)) if cd else None,
                }
                mlflow.log_metric(f"n_tp_{cls}", n_tp_c)
                mlflow.log_metric(f"n_fp_{cls}", n_fp_c)
                mlflow.log_metric(f"n_fn_{cls}", n_fn_c)
                if de:
                    mlflow.log_metric(f"depth_err_{cls}",   per_class[cls]["depth_err"])
                    mlflow.log_metric(f"center_dist_{cls}", per_class[cls]["center_dist"])
            summary["per_class"] = per_class

            # Depth-range breakdown — TP pairs binned by GT depth (gt_z).
            depth_range_breakdown: dict = {}
            for name, lo, hi in DEPTH_BINS:
                de = [p["depth_err"]   for r in valid
                      for p in r.get("tp_pairs", []) if lo <= p["gt_z"] < hi]
                cd = [p["center_dist"] for r in valid
                      for p in r.get("tp_pairs", []) if lo <= p["gt_z"] < hi]
                entry = {
                    "n":           len(de),
                    "depth_err":   float(np.mean(de)) if de else None,
                    "center_dist": float(np.mean(cd)) if cd else None,
                }
                depth_range_breakdown[name] = entry
                if de:  # do not log empty bins to MLflow
                    mlflow.log_metric(f"depth_err_{name}",   entry["depth_err"])
                    mlflow.log_metric(f"center_dist_{name}", entry["center_dist"])
            summary["depth_range_breakdown"] = depth_range_breakdown

            logger.info(
                "=== Validation Summary [seq=%s method=%s] === "
                "depth_err=%.2fm center_dist=%.2fm | "
                "TP=%d FP=%d FN=%d skipped=%d",
                args.seq_id, args.method,
                summary.get("mean_depth_err",   float("nan")),
                summary.get("mean_center_dist", float("nan")),
                summary.get("n_tp",      0),
                summary.get("n_fp",      0),
                summary.get("n_fn",      0),
                summary.get("n_skipped", 0),
            )

        summary["frames"] = merged_frames

        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Validation results saved → %s", results_path)
