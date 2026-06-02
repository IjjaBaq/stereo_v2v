"""Learned orientation head — allocentric angle (alpha) estimation.

EXPERIMENTAL — not validated end-to-end. Decoupled behind
``heading_method: learned`` in stage3.yaml (Stage 3 defaults to ray_angle).
Safe to ignore for now.

Predicts the allocentric observation angle `alpha` of a vehicle from its
cropped image patch. The global heading is then recovered geometrically via
`utils.geometry.recover_rotation_y` (rotation_y = alpha + ray_angle).

Design notes:
    - alpha (NOT rotation_y) is the only orientation a crop can determine: a
      patch carries no information about where in the image it came from, so
      it cannot know the global ray angle. This is the durable insight from
      Mousavian et al. 2017; the head/backbone below are modernized.
    - alpha is encoded as (sin, cos) and recovered with atan2 — this avoids
      the wraparound discontinuity of regressing a raw angle (359 deg vs
      1 deg are far apart in L2 but adjacent on the circle).
    - A small front/back (2-bin) head is trained jointly as AUXILIARY
      supervision: it strengthens the shared backbone features on the
      symmetry that pure sin/cos regression tends to hedge on for cars, and
      provides a confidence signal. It is NOT used to alter the predicted
      angle at inference — flipping on a wrong bin would inject 180 deg
      errors, so sin/cos remains the sole predictor.

This module keeps all torch dependencies out of `utils.geometry` (which must
stay a pure-function module). It is imported by both the training script
(scripts/train_orientation.py) and Stage 3 inference.
"""

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ImageNet normalization — backbones are ImageNet-pretrained
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Angle encode / decode — pure, no torch (tested without a model)
# ---------------------------------------------------------------------------

def decode_alpha(sin_val: float, cos_val: float) -> float:
    """Decode an (sin, cos) pair into an angle via atan2.

    Inputs need not be unit-norm — atan2 ignores magnitude, so raw network
    outputs decode correctly without explicit normalization.

    Args:
        sin_val: Predicted sin(alpha) (any scale).
        cos_val: Predicted cos(alpha) (any scale).

    Returns:
        alpha in radians, range [-pi, pi].
    """
    return float(math.atan2(sin_val, cos_val))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OrientationNet(nn.Module):
    """Backbone + (sin, cos) orientation head + front/back confidence head.

    Args:
        backbone_name: timm model name (default 'convnext_tiny').
        pretrained: Load ImageNet-pretrained weights (True for training,
                    False when loading our own checkpoint for inference).
        freeze_backbone: If True, only the heads receive gradients.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        import timm

        # num_classes=0 → backbone returns pooled feature vector
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,
        )
        feat_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone '%s' frozen — training heads only",
                        backbone_name)

        self.sincos_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),          # raw (sin, cos), normalized in forward
        )
        self.bin_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),          # front / back logits
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Args:
            x: Image batch, shape (B, 3, H, W), ImageNet-normalized.

        Returns:
            Tuple of (sincos, bin_logits):
                sincos     — (B, 2), L2-normalized to the unit circle.
                bin_logits — (B, 2), front/back classification logits.
        """
        feats  = self.backbone(x)
        sincos = self.sincos_head(feats)
        sincos = nn.functional.normalize(sincos, dim=1)
        bins   = self.bin_head(feats)
        return sincos, bins


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_crop(crop_bgr: np.ndarray, crop_size: int = 224) -> torch.Tensor:
    """Convert a BGR crop to a normalized CHW tensor batch of size 1.

    Args:
        crop_bgr: Cropped object patch, shape (h, w, 3), uint8 BGR.
        crop_size: Square resize side length.

    Returns:
        Tensor of shape (1, 3, crop_size, crop_size), float32, normalized.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


# ---------------------------------------------------------------------------
# Load / inference
# ---------------------------------------------------------------------------

def load_orientation_model(
    ckpt_path: str,
    device: str | None = None,
) -> tuple[OrientationNet, dict]:
    """Load a trained orientation checkpoint for inference.

    Args:
        ckpt_path: Path to the .pth checkpoint saved by train_orientation.py.
        device: 'cuda' | 'cpu'. Auto-detected if None.

    Returns:
        Tuple of (model in eval mode, metadata dict). Metadata carries at
        least 'backbone' and 'crop_size'.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Orientation checkpoint not found: {ckpt_path}\n"
            f"Train with scripts/train_orientation.py (Colab GPU), then "
            f"download the .pth to this path."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    meta = ckpt.get("meta", {})
    backbone  = meta.get("backbone", "convnext_tiny")
    crop_size = meta.get("crop_size", 224)

    model = OrientationNet(backbone_name=backbone, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    logger.info("Loaded orientation model — backbone=%s crop_size=%d device=%s",
                backbone, crop_size, device)
    return model, {"backbone": backbone, "crop_size": crop_size, "device": device}


@torch.no_grad()
def predict_alpha(
    model: OrientationNet,
    crop_bgr: np.ndarray,
    crop_size: int = 224,
    device: str | None = None,
) -> float:
    """Predict the allocentric angle alpha for one object crop.

    The sin/cos head is the sole predictor. The front/back bin head is
    auxiliary (training-time feature supervision only) and deliberately not
    used to modify the angle here.

    Args:
        model: Loaded OrientationNet in eval mode.
        crop_bgr: Object crop, shape (h, w, 3), uint8 BGR.
        crop_size: Square resize side used at training time.
        device: 'cuda' | 'cpu'. Auto-detected if None.

    Returns:
        alpha in radians, range [-pi, pi].
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = preprocess_crop(crop_bgr, crop_size).to(device)

    sincos, _bins = model(x)
    sin_v, cos_v = sincos[0, 0].item(), sincos[0, 1].item()
    return decode_alpha(sin_v, cos_v)
