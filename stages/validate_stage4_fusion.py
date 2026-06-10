"""Stage 4 Validation — V2V Cooperative Fusion (CARLA).

Quantifies the value cooperative fusion adds over single-agent perception, on
true-simultaneous CARLA V2V. For each frame it compares three prediction sets,
all expressed in Vehicle A's camera frame, against a **cooperative ground truth**:

    - A-alone : Vehicle A's Stage 1-3 detections.
    - B-alone : Vehicle B's Stage 1-3 detections, registered into A's frame.
    - Fused   : the Stage 4 fused output of the two.

Cooperative GT = every vehicle visible to A *or* B, expressed in A's frame and
deduplicated by ``actor_id`` (an A-visible instance wins — it carries no
registration error). This is the truth fusion should recover; scoring against it
lets ``recall_improvement`` and ``b_unique_tp`` credit objects only B could see.

Matching is greedy BEV (x-z) centre distance within class, reusing the per-class
``matching.max_dist`` thresholds from config/stage4.yaml — the same criterion the
fusion core uses. GT is taken straight from ``utils.carla_loader.load_carla_pair``
(carries ``actor_id``) and registered with ``utils.fusion.transform_box``; it is
never routed through ``fuse`` (which strips ``actor_id``).

Per run it writes ``outputs/fusion/carla/{method}/validation_results.json`` and
logs to MLflow:
    - recall / precision / BEV localization-error per method.
    - recall_improvement = recall_fused - recall_A_alone (headline V2V gain).
    - b_unique_tp = GT recovered only via Vehicle B (outside / occluded for A).
    - per-class (Car, Pedestrian) and GT-depth-range breakdowns.
    - mean inference time per frame (model load excluded).

Usage:
    python stages/validate_stage4_fusion.py \\
        --scenario data/carla \\
        --timestamps 50 100 150 200 250 \\
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stages.stage2_detect import load_model
from stages.stage4_fusion import detect_agent_boxes
from utils.carla_loader import load_carla_pair, load_carla_transform
from utils.config_loader import load_configs
from utils.fusion import bev_distance, fuse, match_boxes, transform_box
from utils.validation_io import merge_samples
from utils.visualization import make_fusion_bev

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# The three prediction sets compared against cooperative GT.
METHODS = ("a_alone", "b_alone", "fused")

# GT-depth bins (by GT z in A's frame) for the localization breakdown — mirrors
# the Stage 3 validator. Each bin is [lo, hi) metres; the last is open-ended.
DEPTH_BINS = (
    ("0_10m",    0.0,  10.0),
    ("10_20m",  10.0,  20.0),
    ("20_40m",  20.0,  40.0),
    ("40m_plus", 40.0, float("inf")),
)


# ---------------------------------------------------------------------------
# Cooperative ground truth
# ---------------------------------------------------------------------------

def build_coop_gt(
    boxes_a_gt: list[dict],
    boxes_b_gt: list[dict],
    T_b_to_a: np.ndarray,
) -> list[dict]:
    """Build cooperative GT in A's frame, deduplicated by actor_id.

    Vehicles visible to A are kept as-is; vehicles visible to B are registered
    into A's frame with ``transform_box`` and added only if their ``actor_id``
    is not already present (the A-visible instance wins — zero registration
    error). Each returned box is tagged ``seen_by`` ∈ {"A", "B", "both"}.

    Args:
        boxes_a_gt: Vehicle A's GT vehicles (A's frame), carry ``actor_id``.
        boxes_b_gt: Vehicle B's GT vehicles (B's frame), carry ``actor_id``.
        T_b_to_a: 4x4 transform mapping B's camera frame into A's.

    Returns:
        Cooperative GT box dicts in A's frame, each with ``seen_by`` added.
    """
    by_actor: dict = {}
    seen: dict = {}

    def key(box: dict, i: int):
        aid = box.get("actor_id")
        return aid if aid is not None else f"_a_{i}"

    for i, box in enumerate(boxes_a_gt):
        k = key(box, i)
        by_actor[k] = dict(box)
        seen[k] = {"A"}

    for i, box in enumerate(boxes_b_gt):
        reg = transform_box(box, T_b_to_a)
        aid = reg.get("actor_id")
        k = aid if aid is not None else f"_b_{i}"
        if k in by_actor:
            seen[k].add("B")
        else:
            by_actor[k] = reg
            seen[k] = {"B"}

    coop = []
    for k, box in by_actor.items():
        s = seen[k]
        box["seen_by"] = "both" if s == {"A", "B"} else next(iter(s))
        coop.append(box)
    return coop


# ---------------------------------------------------------------------------
# Scoring one prediction set against cooperative GT
# ---------------------------------------------------------------------------

def score_against_gt(
    preds: list[dict],
    coop_gt: list[dict],
    max_dist: dict,
) -> dict:
    """Match predictions to cooperative GT and collect TP/FP/FN detail.

    Greedy BEV matching within class (``utils.fusion.match_boxes``). Each TP
    carries the GT depth and BEV error so the run summary can pool by class and
    depth bin. ``matched_gt_keys`` records which GT (by actor_id) each method
    recovered — used for ``b_unique_tp``.

    Args:
        preds: Predicted boxes in A's frame (label, x, y, z, ...).
        coop_gt: Cooperative GT boxes in A's frame (carry actor_id, seen_by).
        max_dist: Per-class BEV match threshold.

    Returns:
        Dict with n_tp, n_fp, n_fn, tp_pairs, fp_labels, fn_labels,
        matched_gt_keys.
    """
    matches, fp_idx, fn_idx = match_boxes(preds, coop_gt, max_dist)

    tp_pairs = []
    matched_gt_keys = set()
    for pi, gi in matches:
        gt = coop_gt[gi]
        tp_pairs.append({
            "label":   gt["label"],
            "gt_z":    float(gt["z"]),
            "bev_err": bev_distance(preds[pi], gt),
        })
        matched_gt_keys.add(gt.get("actor_id", gi))

    return {
        "n_tp":            len(matches),
        "n_fp":            len(fp_idx),
        "n_fn":            len(fn_idx),
        "tp_pairs":        tp_pairs,
        "fp_labels":       [preds[pi]["label"] for pi in fp_idx],
        "fn_labels":       [coop_gt[gi]["label"] for gi in fn_idx],
        "matched_gt_keys": matched_gt_keys,
    }


# ---------------------------------------------------------------------------
# Per-frame validation
# ---------------------------------------------------------------------------

def validate_frame(
    scenario_dir: str,
    timestamp: str,
    base_cfg: dict,
    stage1_cfg: dict,
    stage2_cfg: dict,
    stage3_cfg: dict,
    stage4_cfg: dict,
    method: str,
    agent_a: str | None,
    agent_b: str | None,
    processor,
    model,
    output_dir: Path,
) -> dict:
    """Validate one CARLA frame: A-alone vs B-alone vs Fused against coop GT.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        timestamp: Timestamp string identifying the frame.
        base_cfg: Loaded base.yaml config dict.
        stage1_cfg / stage2_cfg / stage3_cfg: Loaded per-stage configs.
        stage4_cfg: Loaded stage4.yaml config dict (matching + fusion).
        method: Depth method ('sgbm' | 'waft').
        agent_a / agent_b: Agent IDs (None → first/second in scenario).
        processor / model: Pre-loaded RT-DETR handles.
        output_dir: Directory for the per-frame BEV figure.

    Returns:
        Per-frame record dict (keyed by ``timestamp`` for accumulation).
    """
    max_dist = stage4_cfg["matching"]["max_dist"]

    # Resolve agents + transform once, then GT boxes (carry actor_id).
    T_b_to_a, agent_a, agent_b, scene_id = load_carla_transform(
        scenario_dir, timestamp, agent_a=agent_a, agent_b=agent_b,
    )
    boxes_a_gt, boxes_b_gt, _, _ = load_carla_pair(
        scenario_dir, timestamp,
        agent_a=agent_a, agent_b=agent_b, use_gt_boxes=True,
    )
    coop_gt = build_coop_gt(boxes_a_gt, boxes_b_gt, T_b_to_a)

    # Predictions — one detection pass per agent (the expensive part).
    a_pred, t_a = detect_agent_boxes(
        scenario_dir, agent_a, timestamp, base_cfg,
        stage1_cfg, stage2_cfg, stage3_cfg, method, processor, model,
    )
    b_pred, t_b = detect_agent_boxes(
        scenario_dir, agent_b, timestamp, base_cfg,
        stage1_cfg, stage2_cfg, stage3_cfg, method, processor, model,
    )
    b_pred_in_a = [transform_box(b, T_b_to_a) for b in b_pred]
    fused, fuse_stats = fuse(a_pred, b_pred, T_b_to_a, stage4_cfg)

    pred_sets = {"a_alone": a_pred, "b_alone": b_pred_in_a, "fused": fused}
    scored = {m: score_against_gt(p, coop_gt, max_dist)
              for m, p in pred_sets.items()}

    # b_unique_tp — GT visible only to B that fusion recovers but A-alone misses.
    b_only_keys = {
        g.get("actor_id", i) for i, g in enumerate(coop_gt)
        if g["seen_by"] == "B"
    }
    b_unique_tp = len(
        b_only_keys
        & scored["fused"]["matched_gt_keys"]
        - scored["a_alone"]["matched_gt_keys"]
    )

    # BEV figure: coop GT vs A-alone vs fused.
    fig_path = output_dir / f"{scene_id}_bev.png"
    make_fusion_bev(coop_gt, a_pred, fused, scene_id, fig_path)

    record = {
        "timestamp":   timestamp,
        "scene_id":    scene_id,
        "n_coop_gt":   len(coop_gt),
        "b_unique_tp": b_unique_tp,
        "inference_time_s": round(t_a + t_b, 3),
        "fuse_stats":  fuse_stats,
        "methods": {
            m: {k: v for k, v in scored[m].items() if k != "matched_gt_keys"}
            for m in METHODS
        },
    }
    logger.info(
        "scene=%s coop_gt=%d | recall A=%.2f fused=%.2f (+%.2f) | b_unique_tp=%d",
        scene_id, len(coop_gt),
        _recall(scored["a_alone"]), _recall(scored["fused"]),
        _recall(scored["fused"]) - _recall(scored["a_alone"]), b_unique_tp,
    )
    return record


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _recall(s: dict) -> float:
    """Recall = TP / (TP + FN); nan if no GT."""
    denom = s["n_tp"] + s["n_fn"]
    return s["n_tp"] / denom if denom else float("nan")


def _precision(s: dict) -> float:
    """Precision = TP / (TP + FP); nan if no predictions."""
    denom = s["n_tp"] + s["n_fp"]
    return s["n_tp"] / denom if denom else float("nan")


def _pooled_method_summary(valid: list[dict], m: str) -> dict:
    """Pool TP/FP/FN counts and BEV error for one method over all frames."""
    n_tp = sum(r["methods"][m]["n_tp"] for r in valid)
    n_fp = sum(r["methods"][m]["n_fp"] for r in valid)
    n_fn = sum(r["methods"][m]["n_fn"] for r in valid)
    errs = [p["bev_err"] for r in valid for p in r["methods"][m]["tp_pairs"]]
    counts = {"n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn}
    return {
        **counts,
        "recall":    _recall(counts),
        "precision": _precision(counts),
        "loc_error": float(np.mean(errs)) if errs else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Stage 4 CARLA validator."""
    parser = argparse.ArgumentParser(
        description="Stage 4 Validation — V2V cooperative fusion (CARLA)"
    )
    parser.add_argument("--scenario", required=True,
                        help="Path to one CARLA scenario folder (e.g. data/carla)")
    parser.add_argument("--timestamps", nargs="+",
                        default=["50", "100", "150", "200", "250"],
                        help="Frame timestamps to validate (FOV overlaps ~50+)")
    parser.add_argument("--method", default="sgbm", choices=["sgbm", "waft"],
                        help="Depth method (sgbm is CPU-feasible; waft is slow)")
    parser.add_argument("--agent_a", default=None, help="Vehicle A agent ID")
    parser.add_argument("--agent_b", default=None, help="Vehicle B agent ID")
    parser.add_argument("--base_config",   default="config/base.yaml")
    parser.add_argument("--stage1_config", default="config/stage1.yaml")
    parser.add_argument("--stage2_config", default="config/stage2.yaml")
    parser.add_argument("--stage3_config", default="config/stage3.yaml")
    parser.add_argument("--stage4_config", default="config/stage4.yaml")
    return parser.parse_args()


def main() -> None:
    """Run Stage 4 validation over a set of CARLA frames."""
    args = parse_args()

    # Load all configs via the shared loader (no validation-script imports).
    base_cfg, stage1_cfg = load_configs(args.base_config, args.stage1_config)
    _, stage2_cfg = load_configs(args.base_config, args.stage2_config)
    _, stage3_cfg = load_configs(args.base_config, args.stage3_config)
    _, stage4_cfg = load_configs(args.base_config, args.stage4_config)

    classes = tuple(stage4_cfg["matching"]["max_dist"].keys())
    scene = Path(args.scenario).name
    output_dir = Path(stage4_cfg["output_dir"]) / "carla" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RT-DETR model...")
    processor, model = load_model(stage2_cfg["model"])

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage4_fusion_validation")

    all_results: list[dict] = []
    with mlflow.start_run(
        run_name=f"{scene}_{args.method}_{len(args.timestamps)}ts"
    ):
        mlflow.log_param("scenario",   scene)
        mlflow.log_param("method",     args.method)
        mlflow.log_param("timestamps", str(args.timestamps))
        mlflow.log_param("n_frames",   len(args.timestamps))

        for ts in args.timestamps:
            try:
                all_results.append(validate_frame(
                    args.scenario, ts, base_cfg, stage1_cfg, stage2_cfg,
                    stage3_cfg, stage4_cfg, args.method,
                    args.agent_a, args.agent_b, processor, model, output_dir,
                ))
            except Exception as e:
                logger.error("Failed timestamp=%s: %s", ts, e)
                all_results.append({"timestamp": ts, "error": str(e)})

        results_path = output_dir / "validation_results.json"
        merged = merge_samples(results_path, all_results,
                               id_key="timestamp", list_key="frames")
        valid = [r for r in merged if "error" not in r and "methods" in r]

        summary = _build_summary(valid, classes, scene, args.method)
        summary["frames"] = merged
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Validation results saved → %s", results_path)


def _build_summary(
    valid: list[dict],
    classes: tuple[str, ...],
    scene: str,
    method: str,
) -> dict:
    """Aggregate per-method metrics + breakdowns over all valid frames, log MLflow."""
    summary: dict = {"scenario": scene, "method": method, "n_frames": len(valid)}
    if not valid:
        logger.warning("No valid frames — empty summary.")
        return summary

    # Per-method pooled recall/precision/loc-error.
    per_method = {m: _pooled_method_summary(valid, m) for m in METHODS}
    summary["per_method"] = per_method
    for m in METHODS:
        for k in ("n_tp", "n_fp", "n_fn"):
            mlflow.log_metric(f"{k}_{m}", per_method[m][k])
        for k in ("recall", "precision", "loc_error"):
            v = per_method[m][k]
            if v is not None and not math.isnan(v):
                mlflow.log_metric(f"{k}_{m}", v)

    # Headline V2V gains (fused vs A-alone).
    def diff(metric, a, b):
        va, vb = per_method[a][metric], per_method[b][metric]
        if va is None or vb is None or math.isnan(va) or math.isnan(vb):
            return None
        return vb - va

    summary["recall_improvement"]    = diff("recall", "a_alone", "fused")
    summary["precision_change"]      = diff("precision", "a_alone", "fused")
    # loc_error improvement = reduction (A-alone minus fused).
    le = (None if per_method["a_alone"]["loc_error"] is None
          or per_method["fused"]["loc_error"] is None
          else per_method["a_alone"]["loc_error"] - per_method["fused"]["loc_error"])
    summary["loc_error_improvement"] = le
    summary["b_unique_tp"]           = sum(r["b_unique_tp"] for r in valid)
    summary["mean_inference_time_s"] = float(np.mean(
        [r["inference_time_s"] for r in valid]))

    for k in ("recall_improvement", "precision_change", "loc_error_improvement"):
        if summary[k] is not None:
            mlflow.log_metric(k, summary[k])
    mlflow.log_metric("b_unique_tp", summary["b_unique_tp"])
    mlflow.log_metric("mean_inference_time_s", summary["mean_inference_time_s"])

    # Per-class breakdown (counts summed; loc error pooled over TP pairs).
    per_class: dict = {}
    for cls in classes:
        per_class[cls] = {}
        for m in METHODS:
            errs = [p["bev_err"] for r in valid
                    for p in r["methods"][m]["tp_pairs"] if p["label"] == cls]
            n_tp = len(errs)
            n_fp = sum(lab == cls for r in valid
                       for lab in r["methods"][m]["fp_labels"])
            n_fn = sum(lab == cls for r in valid
                       for lab in r["methods"][m]["fn_labels"])
            per_class[cls][m] = {
                "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
                "loc_error": float(np.mean(errs)) if errs else None,
            }
    summary["per_class"] = per_class

    # Depth-range breakdown by GT z, per method.
    depth_breakdown: dict = {}
    for name, lo, hi in DEPTH_BINS:
        depth_breakdown[name] = {}
        for m in METHODS:
            errs = [p["bev_err"] for r in valid
                    for p in r["methods"][m]["tp_pairs"] if lo <= p["gt_z"] < hi]
            depth_breakdown[name][m] = {
                "n": len(errs),
                "loc_error": float(np.mean(errs)) if errs else None,
            }
    summary["depth_range_breakdown"] = depth_breakdown

    logger.info(
        "=== Stage 4 Summary [%s %s] === recall A=%.2f B=%.2f fused=%.2f "
        "(+%.2f) | loc_err A=%.2fm fused=%.2fm | b_unique_tp=%d",
        scene, method,
        _fmt(per_method["a_alone"]["recall"]),
        _fmt(per_method["b_alone"]["recall"]),
        _fmt(per_method["fused"]["recall"]),
        _fmt(summary["recall_improvement"]),
        _fmt(per_method["a_alone"]["loc_error"]),
        _fmt(per_method["fused"]["loc_error"]),
        summary["b_unique_tp"],
    )
    return summary


def _fmt(v) -> float:
    """nan-safe float for logging (None → nan)."""
    return float("nan") if v is None else float(v)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
