"""Stage 4 — V2V Cooperative Fusion.

Fuses the per-agent 3D detections from two simultaneous vehicles (CARLA agents A
and B) into Vehicle A's camera coordinate frame. Each detection may be a Stage 3
3D position (x, y, z) or a full CARLA GT box (x, y, z, l, w, h, heading) — the
fusion core handles both.

Pipeline:
    1. Load each agent's boxes and the inter-agent transform T_b_to_a
       (utils.carla_loader.load_carla_pair).
    2. Register B's boxes into A's frame and greedily match by BEV centre
       distance per class (utils.fusion).
    3. Corroborated pairs are fused (noisy-OR confidence, weighted-mean pose);
       matched pairs whose post-registration displacement is too large are kept
       unmerged and flagged (bad match / pose error). Unmatched boxes are kept,
       tagged by source vehicle.

The fusion core (utils.fusion) is source-agnostic; this module is the CARLA
data plumbing + I/O around it. CARLA gives true simultaneous V2V, so there is no
temporal/static caveat.

Usage:
    python stages/stage4_fusion.py --scenario path/to/scenario --timestamp 000000
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import load_configs
from utils.fusion import build_coop_gt, fuse, transform_box
from utils.visualization import make_fusion_bev

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Per-agent detector path (Stages 1-3 on one CARLA agent's stereo)
# ---------------------------------------------------------------------------

def detect_agent_boxes(
    scenario_dir: str,
    agent: str,
    timestamp: str,
    base_cfg: dict,
    stage1_cfg: dict,
    stage2_cfg: dict,
    stage3_cfg: dict,
    method: str,
    processor,
    model,
) -> tuple[list[dict], float, dict | None]:
    """Run Stages 1-3 on one CARLA agent's stereo frame.

    Stereo depth (Stage 1) → 2D detection (Stage 2) → lift to 3D positions
    (Stage 3), all in the agent's own KITTI camera frame. The returned Stage-3
    positions are fusion-ready (carry label, confidence, x, y, z + source 2D
    box). Per-method Stage-3 sampling params come from config via
    ``stage3_lift.apply_method_overrides``. Stage outputs are written under
    ``outputs/<stage>/carla/...`` so nothing lands in the KITTI dirs.

    When the agent has a ground-truth disparity map on disk, a Stage-1 depth
    validation figure (input | predicted | GT, same layout KITTI uses) is written
    beside the disparity and its EPE/D1/coverage are returned for MLflow logging.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent directory name (e.g. 'vehicle_a').
        timestamp: Timestamp string identifying the frame.
        base_cfg: Loaded base.yaml config dict.
        stage1_cfg: Loaded stage1.yaml config dict.
        stage2_cfg: Loaded stage2.yaml config dict.
        stage3_cfg: Loaded stage3.yaml config dict.
        method: Depth method ('sgbm' | 'waft').
        processor: Pre-loaded RT-DETR image processor (Stage 2).
        model: Pre-loaded RT-DETR model (Stage 2).

    Returns:
        (positions, infer_seconds, depth_eval) — the Stage-3 3D positions, the
        wall-clock inference time in seconds (model load excluded), and the
        Stage-1 depth metrics dict (epe, d1, coverage, ...) or None if the agent
        has no GT disparity for this frame.
    """
    import cv2

    from stages.stage1_depth import run as run_stage1
    from stages.stage2_detect import run as run_stage2
    from stages.stage3_lift import apply_method_overrides, run as run_stage3
    from utils.carla_loader import (
        label_box_2d,
        load_carla_disparity_gt,
        load_carla_frame,
    )
    from utils.depth_metrics import evaluate
    from utils.fusion import match_boxes
    from utils.visualization import (
        make_2d_overlay_visualization,
        make_bev_visualization,
        make_detection_visualization,
        make_side_by_side,
    )

    frame = load_carla_frame(scenario_dir, agent, timestamp)
    sample_id = f"{agent}_{frame['frame_id']}"
    depth_dir = f"outputs/depth/carla/{method}/{agent}"

    t0 = time.perf_counter()

    s1 = run_stage1(
        sample_id=sample_id,
        base_cfg=base_cfg,
        stage_cfg=stage1_cfg,
        method=method,
        image_left=frame["left"],
        image_right=frame["right"],
        calib=frame["calib"],
        output_dir_override=depth_dir,
    )

    # Stage-1 depth validation figure (input | predicted | GT), same layout as
    # the KITTI Stage-1 validator. Skipped if this agent has no GT disparity for
    # the frame, so fusion never breaks on a missing/legacy export.
    depth_eval: dict | None = None
    try:
        gt_disp = load_carla_disparity_gt(scenario_dir, agent, frame["frame_id"])
    except FileNotFoundError as e:
        logger.warning("No GT disparity for %s — skipping depth viz (%s).",
                       sample_id, e)
    else:
        if gt_disp.shape != s1["disp"].shape:
            logger.warning(
                "Disparity shape mismatch for %s: pred=%s gt=%s — skipping "
                "depth viz.", sample_id, s1["disp"].shape, gt_disp.shape,
            )
        else:
            depth_eval = evaluate(s1["disp"], gt_disp)
            val_vis = make_side_by_side(
                frame["left"], s1["disp"], gt_disp,
                f"{agent} {frame['frame_id']}",
                epe=depth_eval["epe"], d1=depth_eval["d1"], method=method,
            )
            val_png = Path(depth_dir) / f"{frame['frame_id']}_val.png"
            val_png.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(val_png), val_vis)
            logger.info("Saved depth validation visualization → %s", val_png)

    s2 = run_stage2(
        sample_id=sample_id,
        base_cfg=base_cfg,
        stage_cfg=stage2_cfg,
        processor=processor,
        model=model,
        image=frame["left"],
        output_dir_override=f"outputs/detections/carla/{agent}",
    )

    # Detection visualization (Pred | GT). Use each GT car's pixel-tight 2D box
    # (segmentation-derived, snug to the visible car) when the export provides it,
    # else fall back to the projected-3D-box envelope — same side-by-side overlay
    # KITTI uses.
    calib = frame["calib"]
    gt_2d = [
        label_box_2d(b, calib["P2"], calib["image_w"], calib["image_h"])
        for b in frame["labels"]
    ]
    det_vis = make_detection_visualization(
        frame["left"], s2["boxes"], gt_2d, sample_id,
    )
    det_png = Path(f"outputs/detections/carla/{agent}") / f"{frame['frame_id']}_det.png"
    det_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(det_png), det_vis)
    logger.info("Saved detection visualization → %s", det_png)

    lift_dir = f"outputs/lift3d/carla/{method}/{agent}"
    s3 = run_stage3(
        sample_id=sample_id,
        base_cfg=base_cfg,
        stage_cfg=apply_method_overrides(stage3_cfg, method),
        method=method,
        disp=s1["disp"],
        boxes2d=s2["boxes"],
        calib=frame["calib"],
        output_dir_override=lift_dir,
    )

    # Stage-3 lift visualization (BEV scatter + 2D overlay), same KITTI layout
    # the Stage-3 validator uses. CARLA GT is 3D, so project each GT box to its
    # source 2D rectangle for the overlay; matching is per-class BEV centre
    # distance (config/stage3.yaml matching.max_dist), same criterion as KITTI.
    preds3d = s3["positions"]
    gt_lift = [
        {
            "label": b["label"], "x": b["x"], "y": b["y"], "z": b["z"],
            **label_box_2d(b, calib["P2"], calib["image_w"], calib["image_h"]),
        }
        for b in frame["labels"]
    ]
    max_dist = stage3_cfg.get("matching", {}).get("max_dist", {})
    lift_matches, _, _ = match_boxes(preds3d, gt_lift, max_dist)
    frame_num = int(frame["frame_id"])
    make_bev_visualization(
        preds3d, gt_lift, lift_matches, frame_id=frame_num, seq_id=agent,
        output_path=Path(lift_dir) / f"{frame['frame_id']}_bev.png",
    )
    make_2d_overlay_visualization(
        frame["left"], preds3d, gt_lift, lift_matches,
        frame_id=frame_num, seq_id=agent,
        output_path=Path(lift_dir) / f"{frame['frame_id']}_2d.png",
    )

    infer_s = time.perf_counter() - t0
    return preds3d, infer_s, depth_eval


# ---------------------------------------------------------------------------
# Fuse + write (shared backend tail)
# ---------------------------------------------------------------------------

def fuse_and_write(
    boxes_a: list[dict],
    boxes_b: list[dict],
    T_b_to_a: np.ndarray,
    scene_id: str,
    meta: dict,
    output_dir: str | Path,
    stage_cfg: dict,
) -> dict:
    """Fuse a registered A/B box pair and write the fused-scene JSON.

    Runs the source-agnostic ``utils.fusion.fuse``, logs, assembles the output
    dict and writes it. The backend supplies the loaded boxes, the B→A transform,
    the scene_id, the output directory and any scene-level metadata.

    Args:
        boxes_a: Vehicle A 3D boxes (in A's frame).
        boxes_b: Vehicle B 3D boxes (in B's frame).
        T_b_to_a: 4x4 transform mapping B's frame to A's.
        scene_id: Unique scene identifier (used for the output filename).
        meta: Scene-level fields merged into the output JSON (backend-specific).
        output_dir: Directory to write ``{scene_id}_fused.json`` into.
        stage_cfg: Loaded stage4.yaml config dict.

    Returns:
        The output dict plus ``output_path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fused, stats = fuse(boxes_a, boxes_b, T_b_to_a, stage_cfg)

    logger.info("Fusion — A=%d B=%d → %d fused, %d dynamic-flagged, "
                "%d only-A, %d only-B (%d output boxes)",
                stats["n_a"], stats["n_b"], stats["n_fused"],
                stats["n_dynamic_flagged"], stats["n_only_a"],
                stats["n_only_b"], len(fused))

    output = {
        "scene_id": scene_id,
        "method":   stage_cfg["method"],
        "vehicles": ["vehicle_A", "vehicle_B"],
        **meta,
        **stats,
        "boxes":    fused,
    }
    output_path = output_dir / f"{scene_id}_fused.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved %d fused boxes → %s", len(fused), output_path)

    return {**output, "output_path": output_path}


# ---------------------------------------------------------------------------
# CARLA backend
# ---------------------------------------------------------------------------

def run_carla(
    scenario_dir: str,
    timestamp: str,
    base_cfg: dict,
    stage_cfg: dict,
    agent_a: str | None = None,
    agent_b: str | None = None,
    output_dir_override: str | None = None,
    *,
    stage1_cfg: dict | None = None,
    stage2_cfg: dict | None = None,
    stage3_cfg: dict | None = None,
    method: str = "sgbm",
    processor=None,
    model=None,
) -> dict:
    """Run Stage 4 fusion for one CARLA agent pair at a timestamp.

    Real simultaneous V2V: the two agents are genuine different viewpoints at the
    same instant. Two modes, selected by ``carla.use_gt_boxes`` in stage_cfg:

    - GT path (``use_gt_boxes=True``): fuse each agent's ground-truth vehicles
      (baseline / smoke test) via ``utils.carla_loader.load_carla_pair``.
    - Detector path (``use_gt_boxes=False``): run Stages 1-3 on each agent's
      stereo (``detect_agent_boxes``) and fuse the real 3D detections, with the
      inter-agent transform from ``load_carla_transform``. Requires the stage
      1/2/3 configs and a loaded RT-DETR (``processor``, ``model``).

    Both paths share the ``fuse_and_write`` tail.

    Args:
        scenario_dir: Path to one CARLA scenario folder (contains agent subdirs).
        timestamp: Timestamp string identifying the frame.
        base_cfg: Loaded base.yaml config dict.
        stage_cfg: Loaded stage4.yaml config dict (reads its ``carla`` block).
        agent_a: Vehicle A agent ID. None → config / first agent in scenario.
        agent_b: Vehicle B agent ID. None → config / second agent in scenario.
        output_dir_override: Optional output directory override.
        stage1_cfg: Loaded stage1.yaml (detector path only).
        stage2_cfg: Loaded stage2.yaml (detector path only).
        stage3_cfg: Loaded stage3.yaml (detector path only).
        method: Depth method ('sgbm' | 'waft') for the detector path.
        processor: Pre-loaded RT-DETR processor (detector path only).
        model: Pre-loaded RT-DETR model (detector path only).

    Returns:
        Dict with scene_id, boxes, stats, output_path (+ inference_time_s on the
        detector path).

    Raises:
        ValueError: On the detector path if stage configs or the model are missing.
    """
    from utils.carla_loader import (
        load_carla_ego_boxes,
        load_carla_pair,
        load_carla_transform,
    )

    carla_cfg = stage_cfg.get("carla", {})
    agent_a = agent_a if agent_a is not None else carla_cfg.get("agent_a")
    agent_b = agent_b if agent_b is not None else carla_cfg.get("agent_b")
    use_gt  = bool(carla_cfg.get("use_gt_boxes", True))

    infer_s: float | None = None
    depth_eval: dict = {}
    if use_gt:
        boxes_a, boxes_b, T_b_to_a, scene_id = load_carla_pair(
            scenario_dir, timestamp,
            agent_a=agent_a, agent_b=agent_b, use_gt_boxes=True,
        )
    else:
        if stage1_cfg is None or stage2_cfg is None or stage3_cfg is None \
                or processor is None or model is None:
            raise ValueError(
                "Detector path (use_gt_boxes=False) requires stage1/2/3 configs "
                "and a loaded RT-DETR (processor, model)."
            )
        T_b_to_a, agent_a, agent_b, scene_id = load_carla_transform(
            scenario_dir, timestamp, agent_a=agent_a, agent_b=agent_b,
        )
        boxes_a, t_a, depth_a = detect_agent_boxes(
            scenario_dir, agent_a, timestamp, base_cfg,
            stage1_cfg, stage2_cfg, stage3_cfg, method, processor, model,
        )
        boxes_b, t_b, depth_b = detect_agent_boxes(
            scenario_dir, agent_b, timestamp, base_cfg,
            stage1_cfg, stage2_cfg, stage3_cfg, method, processor, model,
        )
        infer_s = t_a + t_b
        depth_eval = {
            a: m for a, m in ((agent_a, depth_a), (agent_b, depth_b))
            if m is not None
        }

    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else Path(stage_cfg["output_dir"]) / "carla" / method
    )
    logger.info("=== Stage 4 | CARLA scene=%s | %s ===", scene_id,
                "GT" if use_gt else f"detector ({method})")

    meta = {
        "scenario":  Path(scenario_dir).name,
        "timestamp": timestamp,
        "agent_a":   agent_a,
        "agent_b":   agent_b,
        "source":    "gt" if use_gt else "detector",
    }
    if infer_s is not None:
        meta["method_depth"]     = method
        meta["inference_time_s"] = round(infer_s, 3)
    if depth_eval:
        meta["depth_eval"] = depth_eval

    result = fuse_and_write(boxes_a, boxes_b, T_b_to_a, scene_id, meta,
                            output_dir, stage_cfg)

    # Fusion BEV (KITTI-style scatter) from BOTH perspectives: each ego's
    # alone-vs-fused over the coop GT, highlighting the cars only the other agent
    # saw (that ego's V2V gain). The GT path builds cooperative GT from the two
    # agents' GT boxes; the detector path has none loaded here, so it shows
    # alone-vs-fused only. coop_gt and the fused boxes live in A's frame, so they
    # are transformed into B's frame for B's plot (boxes_b already is).
    coop_gt = build_coop_gt(boxes_a, boxes_b, T_b_to_a) if use_gt else []
    # Ego boxes (in A's frame) so the BEV can drop ego detections, matching the
    # validator's ignore-region scoring. Pose-based, so available on both paths.
    ego_boxes = load_carla_ego_boxes(
        scenario_dir, timestamp, agent_a=agent_a, agent_b=agent_b)
    max_dist = stage_cfg["matching"]["max_dist"]
    make_fusion_bev(coop_gt, boxes_a, result["boxes"], scene_id,
                    output_dir / f"{scene_id}_bev_a.png", ego_label="A",
                    ego_boxes=ego_boxes, max_dist=max_dist)
    T_a_to_b = np.linalg.inv(T_b_to_a)
    coop_gt_in_b = [transform_box(g, T_a_to_b) for g in coop_gt]
    fused_in_b   = [transform_box(f, T_a_to_b) for f in result["boxes"]]
    ego_in_b     = [transform_box(e, T_a_to_b) for e in ego_boxes]
    make_fusion_bev(coop_gt_in_b, boxes_b, fused_in_b, scene_id,
                    output_dir / f"{scene_id}_bev_b.png", ego_label="B",
                    ego_boxes=ego_in_b, max_dist=max_dist)

    if infer_s is not None:
        result["inference_time_s"] = round(infer_s, 3)
    if depth_eval:
        result["depth_eval"] = depth_eval
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the CARLA fusion backend."""
    parser = argparse.ArgumentParser(description="Stage 4 — V2V Fusion (CARLA)")
    parser.add_argument("--scenario", required=True,
                        help="Path to one CARLA scenario folder")
    parser.add_argument("--timestamp", required=True,
                        help="Timestamp identifying the frame to fuse")
    parser.add_argument("--agent_a", default=None, help="Vehicle A agent ID")
    parser.add_argument("--agent_b", default=None, help="Vehicle B agent ID")
    parser.add_argument("--method", default="sgbm", choices=["sgbm", "waft"],
                        help="Depth method for the detector path (--no_gt)")
    parser.add_argument("--no_gt", action="store_true",
                        help="Detector path: run Stages 1-3 per agent and fuse "
                             "real detections (default: fuse GT boxes)")
    parser.add_argument("--base_config",   default="config/base.yaml")
    parser.add_argument("--stage_config",  default="config/stage4.yaml")
    parser.add_argument("--stage1_config", default="config/stage1.yaml")
    parser.add_argument("--stage2_config", default="config/stage2.yaml")
    parser.add_argument("--stage3_config", default="config/stage3.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    import mlflow

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    base_cfg, stage_cfg = load_configs(args.base_config, args.stage_config)
    scene = Path(args.scenario).name

    # The --no_gt flag forces the detector path regardless of the config default.
    if args.no_gt:
        stage_cfg.setdefault("carla", {})["use_gt_boxes"] = False
    use_gt = bool(stage_cfg.get("carla", {}).get("use_gt_boxes", True))

    # Detector path needs the stage 1/2/3 configs and a loaded RT-DETR.
    stage1_cfg = stage2_cfg = stage3_cfg = processor = model = None
    if not use_gt:
        _, stage1_cfg = load_configs(args.base_config, args.stage1_config)
        _, stage2_cfg = load_configs(args.base_config, args.stage2_config)
        _, stage3_cfg = load_configs(args.base_config, args.stage3_config)
        from stages.stage2_detect import load_model
        logger.info("Loading RT-DETR model...")
        processor, model = load_model(stage2_cfg["model"])

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage4_fusion")

    with mlflow.start_run(run_name=f"carla_{scene}_{args.timestamp}"):
        mlflow.log_param("data_source",   "carla")
        mlflow.log_param("scenario",      scene)
        mlflow.log_param("timestamp",     args.timestamp)
        mlflow.log_param("fusion_method", stage_cfg["method"])
        mlflow.log_param("box_source",    "gt" if use_gt else "detector")
        if not use_gt:
            mlflow.log_param("depth_method", args.method)

        result = run_carla(args.scenario, args.timestamp, base_cfg, stage_cfg,
                           agent_a=args.agent_a, agent_b=args.agent_b,
                           stage1_cfg=stage1_cfg, stage2_cfg=stage2_cfg,
                           stage3_cfg=stage3_cfg, method=args.method,
                           processor=processor, model=model)

        for k in ("n_a", "n_b", "n_fused", "n_dynamic_flagged",
                  "n_only_a", "n_only_b"):
            mlflow.log_metric(k, result[k])
        mlflow.log_metric("n_output_boxes", len(result["boxes"]))
        if "inference_time_s" in result:
            mlflow.log_metric("inference_time_s", result["inference_time_s"])

        # Stage-1 depth quality per agent (detector path, when GT disparity
        # exists). Skips NaN metrics (no mutually valid pixels) — MLflow rejects
        # them. Agent suffix keeps the two vehicles' metrics distinct.
        for agent, m in result.get("depth_eval", {}).items():
            for key in ("epe", "d1", "coverage"):
                val = m.get(key)
                if val is not None and not math.isnan(val):
                    mlflow.log_metric(f"depth_{key}_{agent}", val)

    logger.info("Stage 4 complete — MLflow run logged.")
