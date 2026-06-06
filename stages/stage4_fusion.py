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
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import load_configs
from utils.fusion import fuse

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


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
) -> dict:
    """Run Stage 4 fusion for one CARLA agent pair at a timestamp.

    Real simultaneous V2V: the two agents are genuine different viewpoints at the
    same instant. Loads each agent's boxes and the inter-agent transform via
    ``utils.carla_loader``, then runs the shared ``fuse_and_write`` tail.

    Args:
        scenario_dir: Path to one CARLA scenario folder (contains agent subdirs).
        timestamp: Timestamp string identifying the frame.
        base_cfg: Loaded base.yaml config dict (kept for symmetry).
        stage_cfg: Loaded stage4.yaml config dict (reads its ``carla`` block).
        agent_a: Vehicle A agent ID. None → config / first agent in scenario.
        agent_b: Vehicle B agent ID. None → config / second agent in scenario.
        output_dir_override: Optional output directory override.

    Returns:
        Dict with scene_id, boxes, stats, output_path.
    """
    from utils.carla_loader import load_carla_pair

    carla_cfg = stage_cfg.get("carla", {})
    agent_a = agent_a if agent_a is not None else carla_cfg.get("agent_a")
    agent_b = agent_b if agent_b is not None else carla_cfg.get("agent_b")
    use_gt  = bool(carla_cfg.get("use_gt_boxes", True))

    boxes_a, boxes_b, T_b_to_a, scene_id = load_carla_pair(
        scenario_dir, timestamp,
        agent_a=agent_a, agent_b=agent_b, use_gt_boxes=use_gt,
    )

    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else Path(stage_cfg["output_dir"]) / "carla" / Path(scenario_dir).name
    )
    logger.info("=== Stage 4 | CARLA scene=%s ===", scene_id)

    meta = {
        "scenario":  Path(scenario_dir).name,
        "timestamp": timestamp,
        "agent_a":   agent_a,
        "agent_b":   agent_b,
    }
    return fuse_and_write(boxes_a, boxes_b, T_b_to_a, scene_id, meta,
                          output_dir, stage_cfg)


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
    parser.add_argument("--base_config",  default="config/base.yaml")
    parser.add_argument("--stage_config", default="config/stage4.yaml")
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

    mlflow.set_tracking_uri(base_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("stage4_fusion")

    with mlflow.start_run(run_name=f"carla_{scene}_{args.timestamp}"):
        mlflow.log_param("data_source",   "carla")
        mlflow.log_param("scenario",      scene)
        mlflow.log_param("timestamp",     args.timestamp)
        mlflow.log_param("fusion_method", stage_cfg["method"])

        result = run_carla(args.scenario, args.timestamp, base_cfg, stage_cfg,
                           agent_a=args.agent_a, agent_b=args.agent_b)

        for k in ("n_a", "n_b", "n_fused", "n_dynamic_flagged",
                  "n_only_a", "n_only_b"):
            mlflow.log_metric(k, result[k])
        mlflow.log_metric("n_output_boxes", len(result["boxes"]))

    logger.info("Stage 4 complete — MLflow run logged.")
