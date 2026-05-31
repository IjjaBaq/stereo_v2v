"""Stage 3 Validation — End-to-End Pipeline on KITTI Tracking.

Chains Stage 1 → Stage 2 → Stage 3 on KITTI Tracking sequences where
stereo pairs, 3D labels, and calibration are all aligned per frame.

Computes:
    - Mean depth error       : mean |pred_z - gt_z|
    - Mean center distance   : mean sqrt(dx²+dy²+dz²)
    - Mean heading error     : mean |pred_heading - gt_rotation_y|
    - Mean 3D IoU            : via BEV polygon intersection + height overlap
    - TP / FP / FN counts    : greedy matching by confidence, 2D IoU >= threshold

Produces per frame:
    - outputs/depth/tracking/{method}/{seq_id}/{frame_id:06d}_disp.npy
    - outputs/detections/tracking/{seq_id}/{frame_id:06d}_boxes2d.json
    - outputs/boxes3d/{method}/{seq_id}/{frame_id:06d}_boxes3d.json
    - outputs/boxes3d/{method}/{seq_id}/{frame_id:06d}_bev.png
    - outputs/boxes3d/{method}/{seq_id}/{frame_id:06d}_3d.png

Produces per run:
    - outputs/boxes3d/{method}/{seq_id}/validation_results.json

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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stages.stage1_depth import run as run_stage1
from stages.stage2_detect import load_model, run as run_stage2
from stages.stage3_lift import load_configs, run as run_stage3
from utils.geometry import box3d_iou, center_distance, project_box3d_to_image
from utils.kitti_tracking_loader import load_tracking_frame

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

KITTI_CLASSES = ("Car", "Pedestrian")

# Depth sampling strategy per method — empirically validated.
# percentile_75 for SGBM: background pixels have lower disparity than
# foreground so 75th percentile biases toward the object.
# percentile_90 for WAFT: denser coverage requires higher percentile
# to avoid road/background surface contamination.
DEPTH_SAMPLING_BY_METHOD: dict[str, str] = {
    "sgbm": "percentile_75",
    "waft": "percentile_90",
}


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

def gt_label_to_box3d(obj: dict) -> dict:
    """Convert a KITTI tracking label to a 3D box dict.

    KITTI stores y = bottom of object in camera coordinates (Y points down).
    Convert to center Y before computing metrics.

    Args:
        obj: Label dict from load_tracking_labels with keys
             x, y, z, l, w, h, rotation_y, x1, y1, x2, y2, label.

    Returns:
        3D box dict with keys x, y, z, l, w, h, heading,
        x1, y1, x2, y2, label.
    """
    return {
        "label":   obj["label"],
        "x":       obj["x"],
        "y":       obj["y"] - obj["h"] / 2.0,
        "z":       obj["z"],
        "l":       obj["l"],
        "w":       obj["w"],
        "h":       obj["h"],
        "heading": obj["rotation_y"],
        "x1":      obj["x1"],
        "y1":      obj["y1"],
        "x2":      obj["x2"],
        "y2":      obj["y2"],
    }


# ---------------------------------------------------------------------------
# 2D IoU
# ---------------------------------------------------------------------------

def box2d_iou(pred: dict, gt: dict) -> float:
    """Compute 2D IoU between pred and gt using x1,y1,x2,y2 fields.

    Args:
        pred: Dict with keys x1, y1, x2, y2.
        gt:   Dict with keys x1, y1, x2, y2.

    Returns:
        IoU value in [0, 1].
    """
    ix1 = max(pred["x1"], gt["x1"])
    iy1 = max(pred["y1"], gt["y1"])
    ix2 = min(pred["x2"], gt["x2"])
    iy2 = min(pred["y2"], gt["y2"])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_pred = (pred["x2"] - pred["x1"]) * (pred["y2"] - pred["y1"])
    area_gt   = (gt["x2"]   - gt["x1"])   * (gt["y2"]   - gt["y1"])
    union     = area_pred + area_gt - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_boxes(
    preds: list[dict],
    gts: list[dict],
    iou_thresholds: dict[str, float] | None = None,
) -> tuple[list[tuple], list[int], list[int]]:
    """Greedy matching of predictions to GT by confidence, class, 2D IoU.

    Algorithm:
        1. Sort predictions by confidence descending.
        2. For each prediction find highest-IoU unmatched GT of same class.
        3. If best IoU >= threshold → TP, mark GT matched.
        4. Otherwise → FP.
        5. Unmatched GT → FN.

    Args:
        preds: Predicted 3D box dicts with x1,y1,x2,y2,label,confidence.
        gts:   GT 3D box dicts with x1,y1,x2,y2,label.
        iou_thresholds: Per-class minimum 2D IoU. Defaults to 0.5 per class.

    Returns:
        Tuple of (matches, fp_indices, fn_indices).
        matches:    List of (pred_idx, gt_idx) TP pairs.
        fp_indices: Pred indices with no GT match.
        fn_indices: GT indices with no pred match.
    """
    sorted_pred_idx = sorted(
        range(len(preds)),
        key=lambda i: preds[i]["confidence"],
        reverse=True,
    )

    matched_gt: set[int]           = set()
    matches:    list[tuple[int,int]] = []
    fp_indices: list[int]           = []

    for pi in sorted_pred_idx:
        pred     = preds[pi]
        best_iou = 0.0
        best_gi  = -1

        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            if gt["label"] != pred["label"]:
                continue
            iou = box2d_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi

        threshold = (
            iou_thresholds.get(pred["label"], 0.5)
            if iou_thresholds else 0.5
        )
        if best_iou >= threshold and best_gi >= 0:
            matches.append((pi, best_gi))
            matched_gt.add(best_gi)
        else:
            fp_indices.append(pi)

    fn_indices = [gi for gi in range(len(gts)) if gi not in matched_gt]
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
        preds: Predicted 3D box dicts.
        gts:   GT 3D box dicts (Y already converted to center).
        matches: List of (pred_idx, gt_idx) TP pairs.

    Returns:
        Dict with keys: mean_depth_err, mean_center_dist,
        mean_heading_err, mean_iou3d. All nan if no matches.
    """
    if not matches:
        return {
            "mean_depth_err":   float("nan"),
            "mean_center_dist": float("nan"),
            "mean_heading_err": float("nan"),
            "mean_iou3d":       float("nan"),
        }

    depth_errs   = []
    center_dists = []
    heading_errs = []
    iou3d_vals   = []

    for pi, gi in matches:
        pred = preds[pi]
        gt   = gts[gi]

        depth_errs.append(abs(pred["z"] - gt["z"]))
        center_dists.append(center_distance(pred, gt))

        h_err = abs(pred["heading"] - gt["heading"])
        h_err = min(h_err, 2 * math.pi - h_err)
        heading_errs.append(h_err)

        iou3d_vals.append(box3d_iou(pred, gt))

    return {
        "mean_depth_err":   float(np.mean(depth_errs)),
        "mean_center_dist": float(np.mean(center_dists)),
        "mean_heading_err": float(np.mean(heading_errs)),
        "mean_iou3d":       float(np.mean(iou3d_vals)),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def draw_bev_box(
    ax: plt.Axes,
    box: dict,
    color: str,
    label: str | None = None,
) -> None:
    """Draw a single 3D box footprint on a BEV axes.

    Args:
        ax: Matplotlib axes (BEV — X horizontal, Z vertical/depth).
        box: 3D box dict with keys x, z, l, w, heading.
        color: Matplotlib color string.
        label: Optional text label at box center.
    """
    from matplotlib.patches import Polygon as MplPolygon

    cx, cz = box["x"], box["z"]
    l, w   = box["l"], box["w"]
    yaw    = box["heading"]

    corners_local = np.array([
        [ w / 2,  l / 2],
        [-w / 2,  l / 2],
        [-w / 2, -l / 2],
        [ w / 2, -l / 2],
    ])
    cos_h = math.cos(yaw)
    sin_h = math.sin(yaw)
    R     = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners_local @ R.T + np.array([cx, cz])

    ax.add_patch(MplPolygon(
        corners, closed=True,
        edgecolor=color, facecolor="none", linewidth=1.5,
    ))
    if label:
        ax.text(cx, cz, label, color=color, fontsize=6,
                ha="center", va="center")


def make_bev_visualization(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    matches: list[tuple[int, int]],
    frame_id: int,
    seq_id: str,
    output_path: Path,
) -> None:
    """Render BEV visualization with predicted and GT boxes.

    Predictions in green, matched GT in red, unmatched GT in orange.
    Matched pairs connected with yellow dashed lines.

    Args:
        pred_boxes: Predicted 3D box dicts.
        gt_boxes:   GT 3D box dicts.
        matches:    TP (pred_idx, gt_idx) pairs.
        frame_id:   Frame index for title.
        seq_id:     Sequence ID for title.
        output_path: Path to save PNG.
    """
    if not pred_boxes and not gt_boxes:
        logger.warning("No boxes for seq=%s frame=%06d — skipping BEV.",
                       seq_id, frame_id)
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")

    for pi, box in enumerate(pred_boxes):
        draw_bev_box(ax, box, color="#00ff88",
                     label=f"{box['label'][0]}{pi}")

    matched_gt_idx = {gi for _, gi in matches}
    for gi, box in enumerate(gt_boxes):
        color = "#ff4444" if gi in matched_gt_idx else "#ff8800"
        draw_bev_box(ax, box, color=color,
                     label=f"{box['label'][0]}{gi}")

    for pi, gi in matches:
        p = pred_boxes[pi]
        g = gt_boxes[gi]
        ax.plot([p["x"], g["x"]], [p["z"], g["z"]],
                color="yellow", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.autoscale()
    all_z = [b["z"] for b in pred_boxes + gt_boxes]
    if all_z:
        ax.set_ylim(min(all_z) - 5, max(all_z) + 5)

    x_min, _ = ax.get_xlim()
    z_min, _ = ax.get_ylim()
    ax.plot([x_min + 1, x_min + 11], [z_min + 1, z_min + 1],
            color="white", linewidth=2)
    ax.text(x_min + 6, z_min + 1.3, "10 m",
            color="white", ha="center", fontsize=8)

    ax.legend(
        handles=[
            mpatches.Patch(edgecolor="#00ff88", facecolor="none",
                           label="Predicted"),
            mpatches.Patch(edgecolor="#ff4444", facecolor="none",
                           label="GT (matched)"),
            mpatches.Patch(edgecolor="#ff8800", facecolor="none",
                           label="GT (unmatched)"),
        ],
        loc="upper right", facecolor="#333333",
        labelcolor="white", fontsize=8,
    )
    ax.set_xlabel("X (metres)", color="white")
    ax.set_ylabel("Z — depth (metres)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"BEV — seq {seq_id} frame {frame_id:06d}", color="white")
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved BEV → %s", output_path)


def make_3d_projection_visualization(
    image: np.ndarray,
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    matches: list[tuple[int, int]],
    calib: dict,
    frame_id: int,
    seq_id: str,
    output_path: Path,
) -> None:
    """Draw projected 3D boxes on the left camera image.

    Predictions in green, matched GT in dark red, unmatched GT in orange.

    Args:
        image: Left camera BGR image (H, W, 3) uint8.
        pred_boxes: Predicted 3D box dicts.
        gt_boxes:   GT 3D box dicts (Y converted to center).
        matches:    TP (pred_idx, gt_idx) pairs.
        calib:      Calibration dict with key P2.
        frame_id:   Frame index for overlay.
        seq_id:     Sequence ID for overlay.
        output_path: Path to save PNG.
    """
    import cv2

    P2            = calib["P2"]
    out           = image.copy()
    h_img, w_img  = out.shape[:2]

    # 12 edges of a 3D box — corners 0-3: top face, 4-7: bottom face
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]

    def draw_box(corners_px, color, thickness=2):
        if corners_px is None:
            return
        for i, j in edges:
            pt1 = tuple(corners_px[i].tolist())
            pt2 = tuple(corners_px[j].tolist())
            if (abs(pt1[0]) > w_img * 3 or abs(pt1[1]) > h_img * 3 or
                    abs(pt2[0]) > w_img * 3 or abs(pt2[1]) > h_img * 3):
                continue
            cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

    def draw_label(corners_px, text, color):
        if corners_px is None:
            return
        top = corners_px[[0, 3]].mean(axis=0).astype(int)
        cv2.putText(
            out, text,
            (int(top[0]), max(int(top[1]) - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    matched_gt_idx = {gi for _, gi in matches}

    for gi, gt in enumerate(gt_boxes):
        corners_px = project_box3d_to_image(gt, P2)
        color      = (50, 50, 255) if gi in matched_gt_idx else (0, 100, 255)
        draw_box(corners_px, color)
        draw_label(corners_px, f"GT:{gt['label']}", color)

    for pi, pred in enumerate(pred_boxes):
        corners_px = project_box3d_to_image(pred, P2)
        draw_box(corners_px, (0, 220, 80))
        draw_label(corners_px,
                   f"{pred['label']} {pred['confidence']:.2f}", (0, 220, 80))

    cv2.putText(out, f"seq {seq_id} frame {frame_id:06d}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out,
                "Pred (green)  GT matched (dark red)  GT unmatched (orange)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), out)
    logger.info("Saved 3D projection → %s", output_path)


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
    box3d_out = f"outputs/boxes3d/{method}/{seq_id}"

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
    import cv2 as _cv2
    det_vis = frame["left"].copy()
    for box in s2["boxes"]:
        x1,y1,x2,y2 = int(box["x1"]),int(box["y1"]),int(box["x2"]),int(box["y2"])
        _cv2.rectangle(det_vis, (x1,y1), (x2,y2), (0,255,0), 2)
        _cv2.putText(det_vis, f"{box['label']} {box['confidence']:.2f}",
                    (x1, max(y1-5,15)), _cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,255,0), 1, _cv2.LINE_AA)
    det_png = Path(det_out) / f"{frame_id:06d}_det.png"
    _cv2.imwrite(str(det_png), det_vis)
    logger.info("Saved detection visualization → %s", det_png)

    # Stage 3 — lift to 3D
    # Override depth_sampling based on method — empirically validated
    stage3_cfg_run = {
        **stage3_cfg,
        "depth_sampling": DEPTH_SAMPLING_BY_METHOD[method],
    }
    s3 = run_stage3(
        sample_id=f"{frame_id:06d}",
        base_cfg=base_cfg,
        stage_cfg=stage3_cfg_run,
        disp=s1["disp"],
        boxes2d=s2["boxes"],
        calib=calib,
        output_dir_override=box3d_out,
    )

    # GT — filter to active classes, convert Y to center
    gt_boxes = [
        gt_label_to_box3d(obj)
        for obj in frame["labels"]
        if obj["label"] in KITTI_CLASSES
    ]

    pred_boxes     = s3["boxes"]
    iou_thresholds = stage3_cfg.get("matching", {}).get("iou_threshold", {})
    matches, fp_idx, fn_idx = match_boxes(pred_boxes, gt_boxes, iou_thresholds)
    metrics = compute_metrics(pred_boxes, gt_boxes, matches)

    n_tp = len(matches)
    n_fp = len(fp_idx)
    n_fn = len(fn_idx)

    logger.info(
        "seq=%s frame=%06d — TP=%d FP=%d FN=%d | "
        "depth_err=%.2fm center_dist=%.2fm "
        "heading_err=%.3frad iou3d=%.3f",
        seq_id, frame_id, n_tp, n_fp, n_fn,
        metrics["mean_depth_err"]   if not math.isnan(metrics["mean_depth_err"])   else -1,
        metrics["mean_center_dist"] if not math.isnan(metrics["mean_center_dist"]) else -1,
        metrics["mean_heading_err"] if not math.isnan(metrics["mean_heading_err"]) else -1,
        metrics["mean_iou3d"]       if not math.isnan(metrics["mean_iou3d"])       else -1,
    )

    out_dir = Path(box3d_out)
    make_bev_visualization(
        pred_boxes, gt_boxes, matches,
        frame_id=frame_id, seq_id=seq_id,
        output_path=out_dir / f"{frame_id:06d}_bev.png",
    )
    make_3d_projection_visualization(
        frame["left"], pred_boxes, gt_boxes, matches, calib,
        frame_id=frame_id, seq_id=seq_id,
        output_path=out_dir / f"{frame_id:06d}_3d.png",
    )

    return {
        "seq_id":    seq_id,
        "frame_id":  frame_id,
        "n_tp":      n_tp,
        "n_fp":      n_fp,
        "n_fn":      n_fn,
        "n_skipped": s3["n_skipped"],
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

    with mlflow.start_run(
        run_name=f"seq{args.seq_id}_{args.method}_{len(args.frame_ids)}frames"
    ):
        mlflow.log_param("seq_id",         args.seq_id)
        mlflow.log_param("method",         args.method)
        mlflow.log_param("depth_sampling", DEPTH_SAMPLING_BY_METHOD[args.method])
        mlflow.log_param("n_frames",       len(args.frame_ids))
        mlflow.log_param("frame_ids",      str(args.frame_ids))
        mlflow.log_param("stage3_method",  stage3_cfg["method"])
        mlflow.log_param("heading_method", stage3_cfg["heading_method"])

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

        valid = [r for r in all_results if "error" not in r]

        summary: dict = {
            "seq_id":         args.seq_id,
            "method":         args.method,
            "depth_sampling": DEPTH_SAMPLING_BY_METHOD[args.method],
            "n_frames":       len(valid),
        }

        if valid:
            for metric in (
                "mean_depth_err", "mean_center_dist",
                "mean_heading_err", "mean_iou3d",
            ):
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

            logger.info(
                "=== Validation Summary [seq=%s method=%s] === "
                "depth_err=%.2fm center_dist=%.2fm "
                "heading_err=%.3frad iou3d=%.3f | "
                "TP=%d FP=%d FN=%d skipped=%d",
                args.seq_id, args.method,
                summary.get("mean_depth_err",   float("nan")),
                summary.get("mean_center_dist", float("nan")),
                summary.get("mean_heading_err", float("nan")),
                summary.get("mean_iou3d",       float("nan")),
                summary.get("n_tp",      0),
                summary.get("n_fp",      0),
                summary.get("n_fn",      0),
                summary.get("n_skipped", 0),
            )

        summary["frames"] = all_results

        results_path = (
            Path("outputs/boxes3d")
            / args.method
            / args.seq_id
            / "validation_results.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Validation results saved → %s", results_path)
