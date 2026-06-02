"""Train the orientation head (allocentric alpha) on KITTI vehicle crops.

Runs locally on CPU (no GPU required) or on a GPU/Colab if available.
Consumes the dataset from scripts/prepare_orientation_data.py and trains
OrientationNet to predict alpha as a (sin, cos) pair, with an auxiliary
front/back bin head.

CPU strategy — frozen-backbone feature caching:
    The ImageNet backbone is frozen, so its output for a given crop never
    changes during training. We therefore run the backbone over every crop
    ONCE (the only expensive pass), cache the feature vectors to disk, then
    train the small heads on cached features — milliseconds per epoch. This
    turns a ~16h CPU job into a one-time extraction (~30-60 min) plus fast
    head training. The cache is reused on re-runs (re-extracted only if
    backbone / crop_size / hflip settings change), so head hyperparameters
    can be tuned instantly.

    Horizontal-flip augmentation is baked into the cache: each crop is
    extracted both as-is (alpha) and mirrored (alpha -> -alpha), which is the
    geometrically meaningful augmentation and helps balance the bimodal alpha
    distribution. Color jitter is dropped on the cached path (low value on a
    frozen backbone, and it would require per-epoch re-extraction).

    If freeze_backbone is False (the optional 2nd fine-tuning run), training
    falls back to the standard end-to-end loop with on-the-fly augmentation.

Local usage:
    python scripts/prepare_orientation_data.py            # build crops (once)
    python scripts/train_orientation.py \
        --data_dir ./outputs/orientation \
        --orient_config config/orientation.yaml
    # sanity run on a subset first:
    python scripts/train_orientation.py --data_dir ./outputs/orientation --limit 2000

Colab (GPU) usage is identical — it just auto-detects CUDA. See git history
for the Drive-based cell setup if needed.

Loss:
    angular = mean(1 - cos(pred_alpha - gt_alpha))   on the unit (sin,cos)
    bin     = cross-entropy on front/back logits      (auxiliary)
    total   = angular + bin_loss_weight * bin
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.orientation import OrientationNet, preprocess_crop

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrientationDataset(Dataset):
    """Vehicle crops + alpha targets from a prepared manifest.

    Args:
        data_dir: Directory holding {split}_manifest.json and crop folders.
        split: 'train' or 'val'.
        crop_size: Square resize side (matches prep + inference).
        augment: Apply random hflip + color jitter (end-to-end path only).
        force_hflip: Deterministically mirror every crop (negating alpha).
                     Used to build the flipped half of the feature cache.
        limit: Optional cap on number of crops (smoke/sanity runs).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        crop_size: int = 224,
        augment: bool = False,
        force_hflip: bool = False,
        limit: int | None = None,
    ):
        self.data_dir    = Path(data_dir)
        self.crop_size   = crop_size
        self.augment     = augment
        self.force_hflip = force_hflip
        with open(self.data_dir / f"{split}_manifest.json") as f:
            self.items = json.load(f)
        if limit is not None:
            self.items = self.items[:limit]
        logger.info("%s set — %d crops (augment=%s force_hflip=%s)",
                    split, len(self.items), augment, force_hflip)

    def __len__(self) -> int:
        return len(self.items)

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Light random brightness/contrast jitter on a uint8 BGR crop."""
        contrast   = 1.0 + np.random.uniform(-0.2, 0.2)
        brightness = np.random.uniform(-15, 15)
        return np.clip(img.astype(np.float32) * contrast + brightness,
                       0, 255).astype(np.uint8)

    def __getitem__(self, i: int):
        m     = self.items[i]
        crop  = cv2.imread(str(self.data_dir / m["path"]))
        alpha = float(m["alpha"])
        bin_  = int(m["bin"])

        if self.force_hflip:
            crop  = cv2.flip(crop, 1)
            alpha = -alpha                              # mirror negates alpha
            bin_  = 0 if math.cos(alpha) >= 0 else 1    # invariant, recomputed
        elif self.augment:
            if np.random.rand() < 0.5:
                crop  = cv2.flip(crop, 1)
                alpha = -alpha
                bin_  = 0 if math.cos(alpha) >= 0 else 1
            if np.random.rand() < 0.5:
                crop = self._color_jitter(crop)

        x = preprocess_crop(crop, self.crop_size).squeeze(0)
        target = torch.tensor([math.sin(alpha), math.cos(alpha)], dtype=torch.float32)
        return x, target, torch.tensor(bin_, dtype=torch.long), torch.tensor(alpha)


# ---------------------------------------------------------------------------
# Loss / metric
# ---------------------------------------------------------------------------

def angular_loss(pred_sincos: torch.Tensor, target_sincos: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cosine similarity) between predicted and target unit vectors.

    Args:
        pred_sincos: (B, 2), L2-normalized.
        target_sincos: (B, 2), unit (sin, cos) of GT alpha.

    Returns:
        Scalar loss tensor.
    """
    dot = (pred_sincos * target_sincos).sum(dim=1)   # cos(pred - gt)
    return (1.0 - dot).mean()


def mean_angular_error_deg(pred_sincos: torch.Tensor, gt_alpha: torch.Tensor) -> float:
    """Mean absolute angular error in degrees.

    Args:
        pred_sincos: (B, 2) predicted unit vector.
        gt_alpha: (B,) GT alpha in radians.

    Returns:
        Mean |wrap(pred - gt)| in degrees.
    """
    pred_alpha = torch.atan2(pred_sincos[:, 0], pred_sincos[:, 1])
    diff = pred_alpha - gt_alpha
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))   # wrap to [-pi, pi]
    return float(diff.abs().mean().item() * 180.0 / math.pi)


# ---------------------------------------------------------------------------
# Frozen-backbone feature caching
# ---------------------------------------------------------------------------

def _extract(
    backbone: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen backbone over a dataset, returning cached tensors.

    Returns:
        (feats, sincos_targets, bins, alphas) stacked over the dataset.
    """
    backbone.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    feats, sincos, bins, alphas = [], [], [], []
    with torch.no_grad():
        for n, (x, target, bin_, alpha) in enumerate(loader):
            feats.append(backbone(x.to(device)).cpu())
            sincos.append(target)
            bins.append(bin_)
            alphas.append(alpha)
            if (n + 1) % 50 == 0:
                logger.info("  extracted %d/%d batches", n + 1, len(loader))
    return (torch.cat(feats), torch.cat(sincos),
            torch.cat(bins), torch.cat(alphas))


def get_cached_features(
    data_dir: str,
    split: str,
    backbone: nn.Module,
    crop_size: int,
    device: str,
    batch_size: int,
    num_workers: int,
    cache_hflip: bool,
    limit: int | None,
) -> TensorDataset:
    """Build or load the cached backbone-feature dataset for a split.

    Caches to {data_dir}/features_{split}.pt and reuses it unless the
    backbone arch / crop_size / hflip / limit differ.

    Args:
        data_dir: Prepared dataset dir.
        split: 'train' or 'val'.
        backbone: Frozen backbone module (provides .num_features).
        crop_size: Square resize side.
        device: 'cuda' | 'cpu'.
        batch_size: Extraction batch size.
        num_workers: DataLoader workers.
        cache_hflip: Also cache mirrored crops (train only; aug).
        limit: Optional crop cap.

    Returns:
        TensorDataset of (feats, sincos, bins, alphas).
    """
    backbone_name = backbone.__class__.__name__
    cache_path = Path(data_dir) / f"features_{split}.pt"
    sig = {"backbone": backbone_name, "crop_size": crop_size,
           "hflip": cache_hflip, "limit": limit}

    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        if blob.get("sig") == sig:
            logger.info("Reusing feature cache %s (%d samples)",
                        cache_path, blob["feats"].shape[0])
            return TensorDataset(blob["feats"], blob["sincos"],
                                 blob["bins"], blob["alphas"])
        logger.info("Feature cache signature changed — re-extracting %s", split)

    logger.info("Extracting %s features (one-time backbone pass)...", split)
    ds = OrientationDataset(data_dir, split, crop_size,
                            augment=False, force_hflip=False, limit=limit)
    feats, sincos, bins, alphas = _extract(backbone, ds, device,
                                           batch_size, num_workers)

    if cache_hflip:
        logger.info("Extracting %s features (mirrored, alpha->-alpha)...", split)
        ds_f = OrientationDataset(data_dir, split, crop_size,
                                  augment=False, force_hflip=True, limit=limit)
        f2, s2, b2, a2 = _extract(backbone, ds_f, device,
                                  batch_size, num_workers)
        feats  = torch.cat([feats, f2])
        sincos = torch.cat([sincos, s2])
        bins   = torch.cat([bins, b2])
        alphas = torch.cat([alphas, a2])

    torch.save({"feats": feats, "sincos": sincos, "bins": bins,
                "alphas": alphas, "sig": sig}, cache_path)
    logger.info("Cached %d %s feature vectors → %s",
                feats.shape[0], split, cache_path)
    return TensorDataset(feats, sincos, bins, alphas)


def evaluate_heads(
    model: OrientationNet,
    loader: DataLoader,
    device: str,
) -> dict:
    """Validation on cached features — mean angular error + bin accuracy."""
    model.sincos_head.eval()
    model.bin_head.eval()
    errs, bin_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for feats, target, bin_, alpha in loader:
            feats, alpha = feats.to(device), alpha.to(device)
            sincos = nn.functional.normalize(model.sincos_head(feats), dim=1)
            bins   = model.bin_head(feats)
            errs += mean_angular_error_deg(sincos, alpha) * feats.size(0)
            bin_correct += int((bins.argmax(1).cpu() == bin_).sum().item())
            n += feats.size(0)
    return {
        "val_mean_angular_err_deg": errs / max(n, 1),
        "val_bin_accuracy":         bin_correct / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _start_mlflow(mcfg: dict, tcfg: dict):
    """Start an MLflow run if available; return (mlflow_module | None)."""
    try:
        import mlflow
        mlflow.set_experiment("orientation_head")
        mlflow.start_run(run_name=f"{mcfg['backbone']}_alpha")
        for k in ("backbone", "crop_size", "freeze_backbone"):
            mlflow.log_param(k, mcfg[k])
        for k in ("epochs", "batch_size", "lr", "weight_decay", "bin_loss_weight"):
            mlflow.log_param(k, tcfg[k])
        return mlflow
    except Exception as e:                       # mlflow optional
        logger.warning("MLflow unavailable (%s) — continuing without it", e)
        return None


def _save_ckpt(model, mcfg, crop_size, best_err, epoch, ckpt_path):
    """Save the full model state (frozen backbone + trained heads)."""
    torch.save({
        "model": model.state_dict(),
        "meta":  {"backbone": mcfg["backbone"], "crop_size": crop_size},
        "val_mean_angular_err_deg": best_err,
        "epoch": epoch,
    }, str(ckpt_path))


def train_cached(orient_cfg: dict, data_dir: str, device: str,
                 limit: int | None) -> dict:
    """Train heads on cached frozen-backbone features (the CPU-fast path)."""
    mcfg, tcfg = orient_cfg["model"], orient_cfg["train"]
    crop_size  = int(mcfg["crop_size"])
    bs         = int(tcfg["batch_size"])
    nw         = int(tcfg["num_workers"])
    cache_hflip = bool(tcfg["augment"]["hflip"])

    model = OrientationNet(backbone_name=mcfg["backbone"],
                           pretrained=True, freeze_backbone=True).to(device)

    train_ds = get_cached_features(data_dir, "train", model.backbone, crop_size,
                                   device, bs, nw, cache_hflip, limit)
    val_ds   = get_cached_features(data_dir, "val", model.backbone, crop_size,
                                   device, bs, nw, cache_hflip=False, limit=limit)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=bs, shuffle=False)

    head_params = (list(model.sincos_head.parameters())
                   + list(model.bin_head.parameters()))
    optim   = torch.optim.AdamW(head_params, lr=float(tcfg["lr"]),
                                weight_decay=float(tcfg["weight_decay"]))
    ce_loss = nn.CrossEntropyLoss()
    bin_w   = float(tcfg["bin_loss_weight"])
    epochs  = int(tcfg["epochs"])

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{mcfg['backbone']}_alpha.pth"

    mlflow = _start_mlflow(mcfg, tcfg)
    best_err = float("inf")

    for epoch in range(epochs):
        model.sincos_head.train()
        model.bin_head.train()
        running = 0.0
        for feats, target, bin_, _alpha in train_ld:
            feats, target, bin_ = feats.to(device), target.to(device), bin_.to(device)
            sincos = nn.functional.normalize(model.sincos_head(feats), dim=1)
            bins   = model.bin_head(feats)
            loss   = angular_loss(sincos, target) + bin_w * ce_loss(bins, bin_)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * feats.size(0)

        train_loss = running / len(train_ld.dataset)
        metrics = evaluate_heads(model, val_ld, device)
        logger.info("Epoch %2d/%d — train_loss=%.4f val_ang_err=%.2fdeg bin_acc=%.3f",
                    epoch + 1, epochs, train_loss,
                    metrics["val_mean_angular_err_deg"],
                    metrics["val_bin_accuracy"])

        if mlflow is not None:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_mean_angular_err_deg",
                              metrics["val_mean_angular_err_deg"], step=epoch)
            mlflow.log_metric("val_bin_accuracy",
                              metrics["val_bin_accuracy"], step=epoch)

        if metrics["val_mean_angular_err_deg"] < best_err:
            best_err = metrics["val_mean_angular_err_deg"]
            _save_ckpt(model, mcfg, crop_size, best_err, epoch + 1, ckpt_path)
            logger.info("  ↳ new best (%.2fdeg) — saved %s", best_err, ckpt_path)

    if mlflow is not None:
        mlflow.log_metric("best_val_angular_err_deg", best_err)
        mlflow.end_run()

    logger.info("Training done — best val angular error %.2f deg → %s",
                best_err, ckpt_path)
    return {"best_val_angular_err_deg": best_err, "checkpoint_path": str(ckpt_path)}


def train_end_to_end(orient_cfg: dict, data_dir: str, device: str,
                     limit: int | None) -> dict:
    """Standard end-to-end loop — used when the backbone is unfrozen."""
    mcfg, tcfg = orient_cfg["model"], orient_cfg["train"]
    crop_size  = int(mcfg["crop_size"])

    train_ds = OrientationDataset(data_dir, "train", crop_size,
                                  augment=True, limit=limit)
    val_ds   = OrientationDataset(data_dir, "val", crop_size,
                                  augment=False, limit=limit)
    train_ld = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]),
                          shuffle=True, num_workers=int(tcfg["num_workers"]),
                          drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=int(tcfg["batch_size"]),
                          shuffle=False, num_workers=int(tcfg["num_workers"]))

    model = OrientationNet(backbone_name=mcfg["backbone"],
                           pretrained=bool(mcfg["pretrained"]),
                           freeze_backbone=False).to(device)
    optim   = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                                weight_decay=float(tcfg["weight_decay"]))
    ce_loss = nn.CrossEntropyLoss()
    bin_w   = float(tcfg["bin_loss_weight"])
    epochs  = int(tcfg["epochs"])

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{mcfg['backbone']}_alpha.pth"

    mlflow = _start_mlflow(mcfg, tcfg)
    best_err = float("inf")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, target, bin_, _alpha in train_ld:
            x, target, bin_ = x.to(device), target.to(device), bin_.to(device)
            sincos, bins = model(x)
            loss = angular_loss(sincos, target) + bin_w * ce_loss(bins, bin_)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_ld.dataset)
        # reuse cached-feature evaluator via a manual loop over images
        model.eval()
        errs, bin_correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, target, bin_, alpha in val_ld:
                x, alpha = x.to(device), alpha.to(device)
                sincos, bins = model(x)
                errs += mean_angular_error_deg(sincos, alpha) * x.size(0)
                bin_correct += int((bins.argmax(1).cpu() == bin_).sum().item())
                n += x.size(0)
        val_err = errs / max(n, 1)
        logger.info("Epoch %2d/%d — train_loss=%.4f val_ang_err=%.2fdeg bin_acc=%.3f",
                    epoch + 1, epochs, train_loss, val_err, bin_correct / max(n, 1))

        if mlflow is not None:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_mean_angular_err_deg", val_err, step=epoch)

        if val_err < best_err:
            best_err = val_err
            _save_ckpt(model, mcfg, crop_size, best_err, epoch + 1, ckpt_path)
            logger.info("  ↳ new best (%.2fdeg) — saved %s", best_err, ckpt_path)

    if mlflow is not None:
        mlflow.log_metric("best_val_angular_err_deg", best_err)
        mlflow.end_run()

    logger.info("Training done — best val angular error %.2f deg → %s",
                best_err, ckpt_path)
    return {"best_val_angular_err_deg": best_err, "checkpoint_path": str(ckpt_path)}


def train(orient_cfg: dict, data_dir: str, device: str,
          limit: int | None = None) -> dict:
    """Dispatch to the cached (frozen) or end-to-end (unfrozen) trainer.

    Args:
        orient_cfg: Loaded orientation.yaml config dict.
        data_dir: Prepared dataset directory.
        device: 'cuda' | 'cpu'.
        limit: Optional crop cap for sanity runs.

    Returns:
        Dict with best_val_angular_err_deg and checkpoint_path.
    """
    if bool(orient_cfg["model"]["freeze_backbone"]):
        logger.info("Frozen backbone → cached-feature training (CPU-fast)")
        return train_cached(orient_cfg, data_dir, device, limit)
    logger.info("Unfrozen backbone → end-to-end training")
    return train_end_to_end(orient_cfg, data_dir, device, limit)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train orientation head (local CPU or GPU)")
    parser.add_argument("--data_dir", required=True,
                        help="Prepared dataset dir (outputs/orientation)")
    parser.add_argument("--orient_config", default="config/orientation.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap crops per split for a quick sanity run")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    with open(args.orient_config) as f:
        orient_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=== Orientation training | device=%s ===", device)

    result = train(orient_cfg, args.data_dir, device, limit=args.limit)
    print("\n=== Summary ===")
    print(f"Best val angular error: {result['best_val_angular_err_deg']:.2f} deg")
    print(f"Checkpoint: {result['checkpoint_path']}")
