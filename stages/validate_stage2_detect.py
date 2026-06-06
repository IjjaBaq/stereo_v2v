"""Stage 2 Validation — 2D Object Detection.

Single-sample mode: IoU overlay visualization only.
Batch mode (10+ samples recommended): per-class AP at IoU=0.5.

Usage:
    python stages/validate_stage2_detect.py --sample_id 000000
    python stages/validate_stage2_detect.py --sample_ids 000000 000001 000002
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
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stages.stage2_detect import load_configs, load_model, run as run_stage2
from utils.kitti_loader import load_image, load_labels
from utils.validation_io import merge_samples

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

KITTI_CLASSES = ("Car", "Pedestrian")


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def box_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two boxes in x1y1x2y2 format.

    Args:
        box_a: Dict with keys x1, y1, x2, y2.
        box_b: Dict with keys x1, y1, x2, y2.

    Returns:
        IoU value in [0, 1].
    """
    ix1 = max(box_a["x1"], box_b["x1"])
    iy1 = max(box_a["y1"], box_b["y1"])
    ix2 = min(box_a["x2"], box_b["x2"])
    iy2 = min(box_a["y2"], box_b["y2"])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

def compute_ap(
    all_preds: list[dict],
    all_gts: list[dict],
    kitti_class: str,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision for one class using 11-point interpolation.

    Args:
        all_preds: List of dicts with keys sample_id, label, confidence,
                   x1, y1, x2, y2.
        all_gts:   List of dicts with keys sample_id, label, x1, y1, x2, y2.
        kitti_class: Class name to evaluate.
        iou_threshold: IoU threshold for TP (default 0.5).

    Returns:
        AP value in [0, 1].
    """
    preds = sorted(
        [p for p in all_preds if p["label"] == kitti_class],
        key=lambda p: p["confidence"], reverse=True,
    )
    gts = [g for g in all_gts if g["label"] == kitti_class]

    if not gts:
        logger.warning("No GT for class '%s' — AP=0.0", kitti_class)
        return 0.0
    if not preds:
        logger.warning("No predictions for class '%s' — AP=0.0", kitti_class)
        return 0.0

    gt_matched: dict[tuple, bool] = {
        (g["sample_id"], i): False for i, g in enumerate(gts)
    }
    gt_by_sample: dict[str, list[tuple[int, dict]]] = {}
    for i, g in enumerate(gts):
        gt_by_sample.setdefault(g["sample_id"], []).append((i, g))

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for pi, pred in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, gt in gt_by_sample.get(pred["sample_id"], []):
            if gt["label"] != kitti_class:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        key = (pred["sample_id"], best_gi)
        if best_iou >= iou_threshold and not gt_matched.get(key, True):
            tp[pi] = 1
            gt_matched[key] = True
        else:
            fp[pi] = 1

    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(fp)
    recall    = cum_tp / len(gts)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    ap = sum(
        (precision[recall >= t].max() if np.any(recall >= t) else 0.0)
        for t in np.linspace(0, 1, 11)
    ) / 11.0
    return float(ap)


def compute_pr_at_threshold(
    all_preds: list[dict],
    all_gts: list[dict],
    kitti_class: str,
    conf_threshold: float,
    iou_threshold: float = 0.5,
) -> dict:
    """Per-class operating-point metrics at a FIXED confidence threshold.

    Unlike `compute_ap` (which sweeps the confidence axis), this reports the
    single operating point the pipeline actually runs at. Detections are first
    filtered to `kitti_class` and `confidence >= conf_threshold`, then greedily
    matched to same-class GT per sample (highest confidence first, one GT per
    detection, IoU >= `iou_threshold` counts as a TP).

    Args:
        all_preds: Dicts with keys sample_id, label, confidence, x1, y1, x2, y2.
        all_gts:   Dicts with keys sample_id, label, x1, y1, x2, y2.
        kitti_class: Class name to evaluate.
        conf_threshold: Minimum detection confidence to keep.
        iou_threshold: IoU for a true positive (default 0.5).

    Returns:
        Dict: n_tp, n_fp, n_fn, n_gt, n_detected, precision, recall.
            n_detected = n_tp + n_fp ; n_gt = n_tp + n_fn.
            precision = n_tp / n_detected (0.0 if none detected).
            recall    = n_tp / n_gt      (0.0 if no GT).
    """
    preds = sorted(
        [p for p in all_preds
         if p["label"] == kitti_class and p["confidence"] >= conf_threshold],
        key=lambda p: p["confidence"], reverse=True,
    )
    gts = [g for g in all_gts if g["label"] == kitti_class]

    gt_by_sample: dict[str, list[tuple[int, dict]]] = {}
    for i, g in enumerate(gts):
        gt_by_sample.setdefault(g["sample_id"], []).append((i, g))
    gt_matched: dict[tuple, bool] = {
        (g["sample_id"], i): False for i, g in enumerate(gts)
    }

    n_tp = n_fp = 0
    for pred in preds:
        best_iou, best_gi = 0.0, -1
        for gi, gt in gt_by_sample.get(pred["sample_id"], []):
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        key = (pred["sample_id"], best_gi)
        if best_iou >= iou_threshold and not gt_matched.get(key, True):
            n_tp += 1
            gt_matched[key] = True
        else:
            n_fp += 1

    n_gt       = len(gts)
    n_fn       = n_gt - n_tp
    n_detected = n_tp + n_fp
    precision  = n_tp / n_detected if n_detected > 0 else 0.0
    recall     = n_tp / n_gt       if n_gt > 0       else 0.0

    return {
        "n_tp":       n_tp,
        "n_fp":       n_fp,
        "n_fn":       n_fn,
        "n_gt":       n_gt,
        "n_detected": n_detected,
        "precision":  float(precision),
        "recall":     float(recall),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_boxes(
    image: np.ndarray,
    boxes: list[dict],
    color: tuple[int, int, int],
    label_prefix: str = "",
) -> np.ndarray:
    """Draw bounding boxes on image copy with label overlay.

    Args:
        image: BGR image, shape (H, W, 3), uint8.
        boxes: List of dicts with keys label, x1, y1, x2, y2,
               and optionally confidence and iou.
        color: BGR color for boxes and text background.
        label_prefix: Prefix for label text.

    Returns:
        Image copy with boxes drawn.
    """
    out  = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        parts = [p for p in [
            label_prefix,
            box["label"],
            f"{box['confidence']:.2f}" if "confidence" in box else "",
            f"IoU={box['iou']:.2f}"    if "iou"        in box else "",
        ] if p]
        text = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, text, (x1, y1 - 2), font, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return out


def make_detection_visualization(
    image: np.ndarray,
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    sample_id: str,
) -> np.ndarray:
    """Render predicted and GT boxes side by side with IoU annotations.

    Args:
        image: BGR left camera image.
        pred_boxes: Predicted boxes from Stage 2.
        gt_boxes: GT boxes.
        sample_id: For title overlay.

    Returns:
        Side-by-side BGR image.
    """
    pred_annotated = [
        {**p, "iou": max(
            (box_iou(p, g) for g in gt_boxes if g["label"] == p["label"]),
            default=0.0,
        )}
        for p in pred_boxes
    ]
    pred_vis = draw_boxes(image, pred_annotated, (0, 255, 0),  "Pred")
    gt_vis   = draw_boxes(image, gt_boxes,       (0, 0, 255),  "GT")

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(pred_vis, f"Predictions [{sample_id}]",
                (10, 25), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(gt_vis, "Ground Truth",
                (10, 25), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return np.concatenate([pred_vis, gt_vis], axis=1)


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def validate_sample(
    sample_id: str,
    base_cfg: dict,
    stage_cfg: dict,
    processor=None,
    model=None,
) -> dict:
    """Run Stage 2 and produce IoU visualization for one sample.

    Runs detection if output JSON not already on disk.

    Args:
        sample_id: Zero-padded 6-digit KITTI sample ID.
        base_cfg: Loaded base.yaml config dict.
        stage_cfg: Loaded stage2.yaml config dict.
        processor: Optional pre-loaded RT-DETR processor.
        model: Optional pre-loaded RT-DETR model.

    Returns:
        Dict with keys: sample_id, pred_boxes, gt_boxes, inference_time_s.
        inference_time_s is the wall-clock of one detection pass (model already
        loaded by the caller, so load is excluded), or None if the detection
        JSON was already cached on disk and not recomputed this run.
    """
    output_dir = Path(stage_cfg.get("output_dir", "outputs/detections/object"))
    json_path  = output_dir / f"{sample_id}_boxes2d.json"

    inference_time_s = None
    if not json_path.exists():
        logger.info("Detection not found for %s — running Stage 2.", sample_id)
        t0 = time.perf_counter()
        run_stage2(sample_id, base_cfg, stage_cfg, processor, model)
        inference_time_s = time.perf_counter() - t0

    with open(json_path) as f:
        pred_boxes = json.load(f)["boxes"]

    data_root = base_cfg["data"]["data_root"]
    split     = base_cfg["data"]["split"]
    gt_boxes  = [
        g for g in load_labels(data_root, split, sample_id)
        if g["label"] in set(KITTI_CLASSES)
    ]

    image    = load_image(data_root, split, "image_2", sample_id, suffix=".png")
    vis      = make_detection_visualization(image, pred_boxes, gt_boxes, sample_id)
    vis_path = output_dir / f"{sample_id}_det.png"
    cv2.imwrite(str(vis_path), vis)
    logger.info("Saved visualization → %s", vis_path)

    return {
        "sample_id":        sample_id,
        "pred_boxes":       pred_boxes,
        "gt_boxes":         gt_boxes,
        "inference_time_s": (
            round(inference_time_s, 4) if inference_time_s is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 2 Validation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample_id",  type=str,
                       help="Single sample ID — visualization only")
    group.add_argument("--sample_ids", type=str, nargs="+",
                       help="Multiple sample IDs — computes AP")
    parser.add_argument("--base_config",  default="config/base.yaml")
    parser.add_argument("--stage_config", default="config/stage2.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    base_cfg, stage_cfg = load_configs(args.base_config, args.stage_config)

    processor, model = load_model(stage_cfg["model"])

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage2_detect_validation")

    if args.sample_id:
        logger.info("Single-sample mode — visualization only")
        with mlflow.start_run(run_name=f"viz_{args.sample_id}"):
            mlflow.log_param("sample_id", args.sample_id)
            mlflow.log_param("mode",      "single_sample_viz")
            result = validate_sample(
                args.sample_id, base_cfg, stage_cfg, processor, model
            )
            mlflow.log_metric("n_pred", len(result["pred_boxes"]))
            mlflow.log_metric("n_gt",   len(result["gt_boxes"]))
        logger.info("Done — pred=%d GT=%d",
                    len(result["pred_boxes"]), len(result["gt_boxes"]))

    else:
        sample_ids = args.sample_ids
        if len(sample_ids) < 10:
            logger.warning(
                "Only %d samples — AP unreliable below ~10. "
                "Add more with --sample_ids.", len(sample_ids),
            )

        samples: list[dict] = []

        with mlflow.start_run(
            run_name=f"ap_eval_{len(sample_ids)}samples"
        ):
            mlflow.log_param("method",         stage_cfg["method"])
            mlflow.log_param("n_samples",      len(sample_ids))
            mlflow.log_param("iou_thresh",     0.5)
            mlflow.log_param("ap_reliability",
                             "low" if len(sample_ids) < 10 else "ok")

            for sid in sample_ids:
                try:
                    result = validate_sample(
                        sid, base_cfg, stage_cfg, processor, model
                    )
                    samples.append({
                        "sample_id":        sid,
                        "pred_boxes":       result["pred_boxes"],
                        "gt_boxes":         result["gt_boxes"],
                        "inference_time_s": result.get("inference_time_s"),
                    })
                except Exception as e:
                    logger.error("Failed on sample %s: %s", sid, e)
                    samples.append({"sample_id": sid, "error": str(e)})

            output_dir   = Path(
                stage_cfg.get("output_dir", "outputs/detections/object")
            )
            results_path = output_dir / "validation_results.json"

            # Merge this run's samples into any results already on disk, then
            # recompute AP over the full accumulated set. AP is a global metric
            # over all boxes, so per-sample boxes are persisted to allow it.
            merged_samples = merge_samples(
                results_path, samples,
                id_key="sample_id", list_key="samples",
            )
            valid_samples = [s for s in merged_samples if "error" not in s]

            all_preds: list[dict] = []
            all_gts:   list[dict] = []
            for s in valid_samples:
                for box in s["pred_boxes"]:
                    all_preds.append({"sample_id": s["sample_id"], **box})
                for box in s["gt_boxes"]:
                    all_gts.append({"sample_id": s["sample_id"], **box})

            ap_results: dict[str, float] = {}
            for cls in KITTI_CLASSES:
                ap              = compute_ap(all_preds, all_gts, cls)
                ap_results[cls] = ap
                mlflow.log_metric(f"AP_{cls}", ap)
                logger.info("AP @ IoU=0.5 | %-12s → %.4f", cls, ap)

            mean_ap = float(np.mean(list(ap_results.values())))
            mlflow.log_metric("mAP", mean_ap)
            logger.info("mAP @ IoU=0.5 → %.4f", mean_ap)

            # Operating-point metrics at the fixed confidence threshold the
            # pipeline runs at (distinct from the AP confidence sweep above).
            conf_threshold = float(stage_cfg["model"]["confidence_threshold"])
            mlflow.log_param("confidence_threshold", conf_threshold)
            per_class: dict[str, dict] = {}
            for cls in KITTI_CLASSES:
                pr = compute_pr_at_threshold(all_preds, all_gts, cls, conf_threshold)
                per_class[cls] = pr
                mlflow.log_metric(f"precision_{cls}",  pr["precision"])
                mlflow.log_metric(f"recall_{cls}",     pr["recall"])
                mlflow.log_metric(f"n_tp_{cls}",       pr["n_tp"])
                mlflow.log_metric(f"n_fp_{cls}",       pr["n_fp"])
                mlflow.log_metric(f"n_fn_{cls}",       pr["n_fn"])
                mlflow.log_metric(f"n_gt_{cls}",       pr["n_gt"])
                mlflow.log_metric(f"n_detected_{cls}", pr["n_detected"])
                logger.info(
                    "Class %-12s @conf>=%.2f IoU>=0.5 | P=%.3f R=%.3f | "
                    "TP=%d FP=%d FN=%d (GT=%d det=%d)",
                    cls, conf_threshold, pr["precision"], pr["recall"],
                    pr["n_tp"], pr["n_fp"], pr["n_fn"],
                    pr["n_gt"], pr["n_detected"],
                )

            # Mean detection wall-clock over samples computed this run.
            inf_times = [s["inference_time_s"] for s in valid_samples
                         if s.get("inference_time_s") is not None]
            mean_inference_time_s = (
                float(np.mean(inf_times)) if inf_times else None
            )
            if mean_inference_time_s is not None:
                mlflow.log_metric("mean_inference_time_s", mean_inference_time_s)

            with open(results_path, "w") as f:
                json.dump({
                    "method":                stage_cfg["method"],
                    "n_samples":             len(valid_samples),
                    "iou_threshold":         0.5,
                    "confidence_threshold":  conf_threshold,
                    "AP":                    ap_results,
                    "mAP":                   mean_ap,
                    "per_class":             per_class,
                    "mean_inference_time_s": mean_inference_time_s,
                    "samples":               merged_samples,
                }, f, indent=2)
            logger.info("Validation results saved → %s", results_path)
