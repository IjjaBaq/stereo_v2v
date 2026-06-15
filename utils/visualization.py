"""Shared visualization utilities for the stereo_v2v pipeline.

All functions that render pipeline artifacts (disparity colormaps, detection
overlays, BEV scatters, validation figures) live here so stages and validators
import their visual output from one place rather than from each other.

Pure rendering — no metric computation. Functions are grouped by the stage they
serve, but any stage or validator may reuse any of them.
"""

import logging

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from utils.geometry import box_iou

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — disparity
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
    # Robust shared colour scale: clip to the 2nd–98th percentile of valid
    # disparity so a handful of very-near (large-disparity) pixels don't compress
    # the bulk of the map toward black. CARLA GT in particular has sparse
    # near-field spikes (up to ~330 px) that crush a raw min/max scale.
    if combined.size:
        d_min = float(np.percentile(combined, 2))
        d_max = float(np.percentile(combined, 98))
        if d_max <= d_min:  # degenerate (near-constant disparity) — fall back
            d_min, d_max = float(combined.min()), float(combined.max())
    else:
        d_min, d_max = 0.0, 1.0

    def _colorize(disp: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(disp)
        norm  = np.zeros_like(disp)
        if d_max > d_min:
            norm[valid] = (disp[valid] - d_min) / (d_max - d_min) * 255.0
        # Clip before the uint8 cast: percentile clipping leaves values >255
        # (and <0) which would otherwise wrap around modulo 256.
        norm = np.clip(norm, 0.0, 255.0)
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
# Stage 2 — 2D detection overlays
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
# Stage 3 — lift overlays
# ---------------------------------------------------------------------------

def make_detection_overlay(
    image: np.ndarray,
    boxes: list[dict],
    output_path,
) -> None:
    """Draw predicted 2D detection boxes (green) on the left image and save.

    Predictions-only overlay used by the Stage-3 chain to record what Stage 2
    detected for a frame.

    Args:
        image: Left camera BGR image (H, W, 3) uint8.
        boxes: Detection dicts with keys label, confidence, x1, y1, x2, y2.
        output_path: Path to save the PNG.
    """
    det_vis = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(det_vis, f"{box['label']} {box['confidence']:.2f}",
                    (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), det_vis)
    logger.info("Saved detection visualization → %s", output_path)


def make_bev_visualization(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    matches: list[tuple[int, int]],
    frame_id: int,
    seq_id: str,
    output_path,
) -> None:
    """Render a BEV scatter of predicted and GT 3D centers (X vs Z).

    Stage 3 emits position only (no size/heading), so centers are drawn as
    scatter points rather than rotated footprints. Predictions in green,
    matched GT in red, unmatched GT in orange. Matched pairs are connected
    with yellow dashed lines.

    Args:
        pred_boxes: Predicted 3D position dicts with keys x, z, label.
        gt_boxes:   GT 3D position dicts with keys x, z, label.
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
        ax.scatter(box["x"], box["z"], c="#00ff88", marker="o", s=40)
        ax.text(box["x"], box["z"], f" {box['label'][0]}{pi}",
                color="#00ff88", fontsize=6, ha="left", va="center")

    matched_gt_idx = {gi for _, gi in matches}
    for gi, box in enumerate(gt_boxes):
        color = "#ff4444" if gi in matched_gt_idx else "#ff8800"
        ax.scatter(box["x"], box["z"], c=color, marker="x", s=40)
        ax.text(box["x"], box["z"], f" {box['label'][0]}{gi}",
                color=color, fontsize=6, ha="left", va="center")

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
            mpatches.Patch(color="#00ff88", label="Predicted center"),
            mpatches.Patch(color="#ff4444", label="GT (matched)"),
            mpatches.Patch(color="#ff8800", label="GT (unmatched)"),
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


def make_2d_overlay_visualization(
    image: np.ndarray,
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    matches: list[tuple[int, int]],
    frame_id: int,
    seq_id: str,
    output_path,
) -> None:
    """Draw 2D boxes (x1,y1,x2,y2) on the left camera image.

    Stage 3 no longer emits 3D extents/heading, so this overlays the source
    2D boxes instead of projected 3D wireframes. Predictions in green,
    matched GT in dark red, unmatched GT in orange.

    Args:
        image: Left camera BGR image (H, W, 3) uint8.
        pred_boxes: Predicted box dicts with x1,y1,x2,y2,label,confidence.
        gt_boxes:   GT box dicts with x1,y1,x2,y2,label.
        matches:    TP (pred_idx, gt_idx) pairs.
        frame_id:   Frame index for overlay.
        seq_id:     Sequence ID for overlay.
        output_path: Path to save PNG.
    """
    import cv2

    out = image.copy()

    def draw_box(box, color, text):
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, text, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    matched_gt_idx = {gi for _, gi in matches}

    for gi, gt in enumerate(gt_boxes):
        color = (50, 50, 255) if gi in matched_gt_idx else (0, 100, 255)
        draw_box(gt, color, f"GT:{gt['label']}")

    for pred in pred_boxes:
        draw_box(pred, (0, 220, 80),
                 f"{pred['label']} {pred['z']:.1f}m {pred['confidence']:.2f}")

    cv2.putText(out, f"seq {seq_id} frame {frame_id:06d}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out,
                "Pred (green)  GT matched (dark red)  GT unmatched (orange)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), out)
    logger.info("Saved 2D overlay → %s", output_path)


# ---------------------------------------------------------------------------
# Stage 4 — fusion BEV
# ---------------------------------------------------------------------------

def make_fusion_bev(
    coop_gt: list[dict],
    ego_pred: list[dict],
    fused: list[dict],
    scene_id: str,
    output_path,
    ego_label: str = "A",
) -> None:
    """Render a BEV scatter (X vs Z) from one ego's perspective.

    All inputs must already be expressed in ``ego_label``'s camera frame. The plot
    shows the ego's own (alone) predictions vs the fused predictions, over the
    cooperative GT. The GT is split by ``seen_by``: cars the ego can see itself
    (``seen_by`` ∈ {ego, "both"}) in orange, and cars only the *other* agent can
    see (``seen_by`` == other) highlighted in magenta — the latter are exactly
    what cooperation should let this ego recover, so a fused △ landing on a
    magenta × with no green ○ nearby is the visual proof of V2V gain.

    Position-only points (no footprints) since Stage-3 emits centres.

    Args:
        coop_gt: Cooperative GT boxes in the ego's frame (carry ``seen_by``).
        ego_pred: This ego's own Stage-3 predictions (ego's frame).
        fused: Stage 4 fused boxes in the ego's frame.
        scene_id: Scene identifier for the title.
        output_path: Path to save the PNG.
        ego_label: Which agent's perspective this is ("A" or "B").
    """
    if not (coop_gt or ego_pred or fused):
        logger.warning("No boxes for %s — skipping BEV.", scene_id)
        return

    other_label = "B" if ego_label == "A" else "A"

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")

    for g in coop_gt:
        if g["seen_by"] == other_label:                 # other-only: the gain
            ax.scatter(g["x"], g["z"], c="#ff33cc", marker="x", s=80,
                       linewidths=2.0)
        else:                                            # ego sees it itself
            ax.scatter(g["x"], g["z"], c="#ff8800", marker="x", s=55,
                       linewidths=1.5)
    for p in ego_pred:
        ax.scatter(p["x"], p["z"], facecolors="none", edgecolors="#00ff88",
                   marker="o", s=70, linewidths=1.3)
    for f in fused:
        ax.scatter(f["x"], f["z"], c="#33ccff", marker="^", s=40)

    all_z = [b["z"] for b in coop_gt + ego_pred + fused]
    if all_z:
        ax.set_ylim(min(all_z) - 5, max(all_z) + 5)

    ax.legend(
        handles=[
            mpatches.Patch(color="#ff8800", label=f"Coop GT ({ego_label} or both)"),
            mpatches.Patch(color="#ff33cc", label=f"Coop GT ({other_label}-only) — {ego_label}'s gain"),
            mpatches.Patch(color="#00ff88", label=f"{ego_label}-alone pred"),
            mpatches.Patch(color="#33ccff", label="Fused pred"),
        ],
        loc="upper right", facecolor="#333333", labelcolor="white", fontsize=8,
    )
    ax.set_xlabel("X (metres)", color="white")
    ax.set_ylabel("Z — depth (metres)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"BEV ({ego_label}'s frame) — {scene_id}", color="white")
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved BEV → %s", output_path)
