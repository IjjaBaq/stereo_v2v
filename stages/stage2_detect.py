"""Stage 2 — 2D Object Detection.

Runs RT-DETR (pretrained on COCO) on the left KITTI image, filters and
remaps detections to KITTI classes using the config-defined class mapping,
and saves results to outputs/detections/object/.

Usage:
    python stages/stage2_detect.py --sample_id 000000
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
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import load_configs
from utils.kitti_loader import load_image

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_cfg: dict,
) -> tuple[RTDetrImageProcessor, RTDetrForObjectDetection]:
    """Load RT-DETR processor and model from HuggingFace hub or local cache.

    Args:
        model_cfg: The 'model' section of stage2.yaml.

    Returns:
        Tuple of (processor, model) ready for inference.
    """
    model_name       = model_cfg["name"]
    cache_dir        = model_cfg["cache_dir"]
    local_files_only = model_cfg.get("local_files_only", False)

    logger.info("Loading RT-DETR model: %s (cache_dir=%s)",
                model_name, cache_dir)

    processor = RTDetrImageProcessor.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only,
    )
    model = RTDetrForObjectDetection.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only,
    )
    model.eval()
    logger.info("Model loaded — %d parameters",
                sum(p.numel() for p in model.parameters()))
    return processor, model


# ---------------------------------------------------------------------------
# COCO → KITTI mapping
# ---------------------------------------------------------------------------

def build_coco_to_kitti_map(
    class_mapping_cfg: dict,
    model: RTDetrForObjectDetection,
) -> dict[int, str]:
    """Build a mapping from COCO class index → KITTI class name.

    Args:
        class_mapping_cfg: Dict of {coco_name: kitti_name} from stage2.yaml.
        model: Loaded RT-DETR model (provides id2label).

    Returns:
        Dict of {coco_class_index: kitti_class_name}.
    """
    name_to_idx = {v.lower(): k for k, v in model.config.id2label.items()}

    idx_map: dict[int, str] = {}
    for coco_name, kitti_name in class_mapping_cfg.items():
        coco_lower = coco_name.lower()
        if coco_lower not in name_to_idx:
            logger.warning("COCO class '%s' not found in model — skipping",
                           coco_name)
            continue
        idx              = name_to_idx[coco_lower]
        idx_map[idx]     = kitti_name
        logger.debug("Mapped COCO '%s' (idx=%d) → KITTI '%s'",
                     coco_name, idx, kitti_name)

    logger.info("Class map built — %d COCO classes → %d KITTI classes",
                len(idx_map), len(set(idx_map.values())))
    return idx_map


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect(
    image_bgr: np.ndarray,
    processor: RTDetrImageProcessor,
    model: RTDetrForObjectDetection,
    coco_to_kitti: dict[int, str],
    confidence_threshold: float,
) -> list[dict]:
    """Run RT-DETR inference on a single BGR image.

    Args:
        image_bgr: Left camera image, shape (H, W, 3), uint8 BGR.
        processor: HuggingFace RT-DETR image processor.
        model: Loaded RT-DETR model.
        coco_to_kitti: Mapping from COCO class index to KITTI class name.
        confidence_threshold: Minimum confidence to keep a detection.

    Returns:
        List of dicts with keys: label, confidence, x1, y1, x2, y2.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w      = image_rgb.shape[:2]

    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[h, w]]),
        threshold=confidence_threshold,
    )[0]

    detections = []
    for score, label_idx, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_idx = int(label_idx.item())
        if label_idx not in coco_to_kitti:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "label":      coco_to_kitti[label_idx],
            "confidence": round(float(score.item()), 4),
            "x1":         round(x1, 2),
            "y1":         round(y1, 2),
            "x2":         round(x2, 2),
            "y2":         round(y2, 2),
        })

    logger.info("Detection complete — %d boxes kept (threshold=%.2f)",
                len(detections), confidence_threshold)
    return detections


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    sample_id: str,
    base_cfg: dict,
    stage_cfg: dict,
    processor: RTDetrImageProcessor | None = None,
    model: RTDetrForObjectDetection | None = None,
    image: np.ndarray | None = None,
    output_dir_override: str | None = None,
) -> dict:
    """Run Stage 2 detection for a single sample.

    Args:
        sample_id: Zero-padded 6-digit KITTI sample ID or tracking frame ID.
        base_cfg: Loaded base.yaml config dict.
        stage_cfg: Loaded stage2.yaml config dict.
        processor: Optional pre-loaded RT-DETR processor.
        model: Optional pre-loaded RT-DETR model.
        image: Optional pre-loaded BGR image. Skips load_image if provided.
        output_dir_override: Optional output directory override.

    Returns:
        Dict with keys: sample_id, method, boxes, output_path.
    """

    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else Path(stage_cfg.get("output_dir", "outputs/detections/object"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 2 | sample=%s ===", sample_id)

    if processor is None or model is None:
        processor, model = load_model(stage_cfg["model"])

    coco_to_kitti = build_coco_to_kitti_map(stage_cfg["class_mapping"], model)

    if image is None:
        data_root = base_cfg["data"]["data_root"]
        split     = base_cfg["data"]["split"]
        image     = load_image(data_root, split, "image_2",
                               sample_id, suffix=".png")

    detections = detect(
        image, processor, model, coco_to_kitti,
        confidence_threshold=stage_cfg["model"]["confidence_threshold"],
    )

    output = {
        "sample_id": sample_id,
        "method":    stage_cfg["method"],
        "boxes":     detections,
    }
    output_path = output_dir / f"{sample_id}_boxes2d.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved %d detections → %s", len(detections), output_path)
    return {**output, "output_path": output_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 2 — 2D Object Detection")
    parser.add_argument("--sample_id",    required=True,
                        help="6-digit KITTI sample ID")
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

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage2_detect")

    with mlflow.start_run(run_name=f"rtdetr_{args.sample_id}"):
        mlflow.log_param("sample_id",   args.sample_id)
        mlflow.log_param("method",      stage_cfg["method"])
        mlflow.log_param("model_name",  stage_cfg["model"]["name"])
        mlflow.log_param("conf_thresh", stage_cfg["model"]["confidence_threshold"])

        result = run(args.sample_id, base_cfg, stage_cfg)

        mlflow.log_metric("n_detections", len(result["boxes"]))
        class_counts: dict[str, int] = {}
        for box in result["boxes"]:
            class_counts[box["label"]] = class_counts.get(box["label"], 0) + 1
        for cls, count in class_counts.items():
            mlflow.log_metric(f"n_{cls.lower()}", count)

        logger.info("Stage 2 complete — MLflow run logged.")
