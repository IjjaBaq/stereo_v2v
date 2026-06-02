"""Prepare the orientation training dataset from KITTI 3D object labels.

Extracts vehicle crops + their allocentric angle (alpha, KITTI label field 3)
from the object detection split, applies quality filters, and writes resized
crops plus a manifest for the Colab training script.

Runs locally (no GPU). Reads data/ only; writes to outputs/orientation/.

Output layout:
    outputs/orientation/
        train/{label}/{sid}_{idx}.png      resized crops (crop_size square)
        val/{label}/{sid}_{idx}.png
        train_manifest.json                [{path, alpha, bin, label, sid}, ...]
        val_manifest.json
        summary.json                       counts + alpha histogram

The split is BY IMAGE (seed 42) — every object from one image lands in the
same split, so no object leaks between train and val.

Usage:
    python scripts/prepare_orientation_data.py
    python scripts/prepare_orientation_data.py --limit 500   # quick smoke run
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.kitti_loader import load_image, load_labels

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


def load_configs(base_path: str, orient_path: str) -> tuple[dict, dict]:
    """Load base and orientation YAML configs.

    Args:
        base_path: Path to config/base.yaml.
        orient_path: Path to config/orientation.yaml.

    Returns:
        Tuple of (base_cfg, orient_cfg).

    Raises:
        FileNotFoundError: If either config file is missing.
    """
    for p in (base_path, orient_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Config file not found: {p}")
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(orient_path) as f:
        orient_cfg = yaml.safe_load(f)
    return base_cfg, orient_cfg


def alpha_to_bin(alpha: float) -> int:
    """Map alpha to a front/back hemisphere bin (auxiliary label).

    bin 0 = front hemisphere (cos(alpha) >= 0, alpha in [-pi/2, pi/2]),
    bin 1 = rear  hemisphere. The split distinguishes the two ~180-deg
    ambiguous candidates a symmetric vehicle appearance could map to.

    Args:
        alpha: Allocentric angle in radians.

    Returns:
        0 or 1.
    """
    return 0 if math.cos(alpha) >= 0.0 else 1


def list_sample_ids(data_root: str, split: str) -> list[str]:
    """List all label IDs in the object split, sorted.

    Args:
        data_root: KITTI detection root (e.g. ./data/kitti/detection).
        split: 'training'.

    Returns:
        Sorted list of zero-padded sample ID strings.
    """
    label_dir = Path(data_root) / split / "label_2"
    if not label_dir.exists():
        raise FileNotFoundError(f"label_2 not found: {label_dir}")
    return sorted(f.stem for f in label_dir.glob("*.txt"))


def split_by_image(
    sample_ids: list[str],
    val_fraction: float,
) -> tuple[set[str], set[str]]:
    """Deterministically split sample IDs into train/val by image.

    Args:
        sample_ids: All sample IDs.
        val_fraction: Fraction assigned to validation.

    Returns:
        Tuple of (train_ids, val_ids) as sets.
    """
    shuffled = list(sample_ids)
    random.Random(42).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    return train_ids, val_ids


def crop_and_save(
    image: np.ndarray,
    obj: dict,
    crop_size: int,
    out_path: Path,
) -> bool:
    """Crop an object box, resize, and save as PNG.

    Args:
        image: Full BGR image, shape (H, W, 3).
        obj: Label dict with x1, y1, x2, y2.
        crop_size: Square resize side.
        out_path: Destination PNG path.

    Returns:
        True if saved, False if the crop was empty/degenerate.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(obj["x1"]))
    y1 = max(0, int(obj["y1"]))
    x2 = min(w, int(obj["x2"]))
    y2 = min(h, int(obj["y2"]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return True


def prepare(
    base_cfg: dict,
    orient_cfg: dict,
    limit: int | None = None,
) -> dict:
    """Build the orientation dataset.

    Args:
        base_cfg: Loaded base.yaml config dict.
        orient_cfg: Loaded orientation.yaml config dict.
        limit: Optional cap on number of images (for smoke tests).

    Returns:
        Summary dict with counts and the train alpha histogram.
    """
    data_root = base_cfg["data"]["data_root"]
    split     = base_cfg["data"]["split"]

    data_cfg   = orient_cfg["data"]
    classes    = set(data_cfg["classes"])
    max_trunc  = float(data_cfg["max_truncated"])
    max_occ    = int(data_cfg["max_occluded"])
    min_box_px = int(data_cfg["min_box_px"])
    val_frac   = float(data_cfg["val_fraction"])
    out_dir    = Path(data_cfg["output_dir"])
    crop_size  = int(orient_cfg["model"]["crop_size"])

    sample_ids = list_sample_ids(data_root, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    train_ids, val_ids = split_by_image(sample_ids, val_frac)
    logger.info("Images: %d total → %d train, %d val",
                len(sample_ids), len(train_ids), len(val_ids))

    manifests: dict[str, list[dict]] = {"train": [], "val": []}
    n_dropped = 0

    for n, sid in enumerate(sample_ids):
        which = "val" if sid in val_ids else "train"
        try:
            objs  = load_labels(data_root, split, sid)
            image = load_image(data_root, split, "image_2", sid, suffix=".png")
        except FileNotFoundError as e:
            logger.warning("Skipping %s — %s", sid, e)
            continue

        for idx, obj in enumerate(objs):
            if obj["label"] not in classes:
                continue
            if obj["truncated"] > max_trunc or obj["occluded"] > max_occ:
                n_dropped += 1
                continue
            if (obj["x2"] - obj["x1"]) < min_box_px or \
               (obj["y2"] - obj["y1"]) < min_box_px:
                n_dropped += 1
                continue

            rel = Path(which) / obj["label"] / f"{sid}_{idx}.png"
            if not crop_and_save(image, obj, crop_size, out_dir / rel):
                n_dropped += 1
                continue

            manifests[which].append({
                "path":  str(rel),
                "alpha": float(obj["alpha"]),
                "bin":   alpha_to_bin(float(obj["alpha"])),
                "label": obj["label"],
                "sid":   sid,
            })

        if (n + 1) % 1000 == 0:
            logger.info("Processed %d/%d images — train=%d val=%d",
                        n + 1, len(sample_ids),
                        len(manifests["train"]), len(manifests["val"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    for which in ("train", "val"):
        with open(out_dir / f"{which}_manifest.json", "w") as f:
            json.dump(manifests[which], f)
        logger.info("Wrote %s manifest — %d crops", which, len(manifests[which]))

    train_alphas = np.array([m["alpha"] for m in manifests["train"]])
    hist, _ = np.histogram(train_alphas, bins=16, range=(-math.pi, math.pi))

    summary = {
        "crop_size":     crop_size,
        "classes":       sorted(classes),
        "n_train":       len(manifests["train"]),
        "n_val":         len(manifests["val"]),
        "n_dropped":     n_dropped,
        "filters": {
            "max_truncated": max_trunc,
            "max_occluded":  max_occ,
            "min_box_px":    min_box_px,
        },
        "train_alpha_hist_16bins": hist.tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done — train=%d val=%d dropped=%d",
                summary["n_train"], summary["n_val"], n_dropped)
    logger.info("Train alpha histogram (16 bins -pi..pi): %s", hist.tolist())
    logger.info("Next: upload %s/ to Drive, train with "
                "scripts/train_orientation.py on Colab", out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare orientation training crops from KITTI object labels"
    )
    parser.add_argument("--base_config",   default="config/base.yaml")
    parser.add_argument("--orient_config", default="config/orientation.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of images (smoke test)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    base_cfg, orient_cfg = load_configs(args.base_config, args.orient_config)
    prepare(base_cfg, orient_cfg, limit=args.limit)
