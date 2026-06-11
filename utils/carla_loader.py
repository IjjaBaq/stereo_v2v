"""CARLA data loader for Stage 4 cooperative fusion (real simultaneous V2V).

STATUS: IMPLEMENTED against the export produced by
``scripts/collect_carla_data.py`` (Town10HD intersection, two moving ego
vehicles + ambient NPC traffic). The on-disk layout is::

    <scenario_dir>/
    ├── vehicle_a/
    │   ├── image_left/   {frame:06d}.png   — left stereo image
    │   ├── image_right/  {frame:06d}.png   — right stereo image
    │   ├── calib.json    — camera intrinsics + stereo baseline (P2/P3, ...)
    │   └── pose.json     — {frame: {x,y,z,roll,pitch,yaw}} world pose
    ├── vehicle_b/        — same layout
    └── gt_boxes/
        └── {frame:06d}.json  — every vehicle's GT box in CARLA world coords
                                 (label, actor_id, x, y, z, l, w, h, yaw, plus
                                 metrics_metadata.visible_pixels_vA/vB — the
                                 per-agent count of actually-visible pixels)

Per-agent GT is filtered by true visibility: a car counts as seen by an agent
only if ``metrics_metadata.visible_pixels_v{A,B}`` >= ``min_visible_pixels``.
This is occlusion-truthful (occluded / off-screen cars render 0 pixels), unlike
the geometric centre test kept only as a legacy fallback.

What the pipeline needs from a CARLA export
-------------------------------------------
Stage 4 fusion (``stages.stage4_fusion``) consumes, per agent pair at one
timestamp::

    boxes_a, boxes_b, T_b_to_a, scene_id = load_carla_pair(...)

where ``boxes_a`` / ``boxes_b`` are each agent's 3D boxes in the **KITTI camera
convention** (x-right, y-down, z-forward; BEV = x-z plane; heading about the
y-axis) so the source-agnostic core in ``utils.fusion`` works unchanged, and
``T_b_to_a`` maps agent B's camera frame into agent A's.

Stage 3 validation on CARLA frames (``stages.validate_stage3_lift``) uses the
per-frame loaders ``load_carla_frame`` / ``load_carla_calib``, mirroring the
``load_tracking_frame`` / ``load_tracking_calib`` contract in
``utils.kitti_tracking_loader``.

Coordinate conventions honored (CARLA's own, verified numerically)
------------------------------------------------------------------
- World/agent frame: **left-handed**, +x forward, +y right, +z up.
- Agent pose: ``[x, y, z, roll, pitch, yaw]`` in metres and **degrees**. The
  world-from-agent matrix replicates CARLA's ``Transform.get_matrix()`` Euler
  order exactly (see ``_world_from_agent``).
- Object annotations: world ``location`` [x,y,z] (deg ``yaw``); the collector
  already doubles the half-extents into ``l, w, h``.
- LIDAR/agent → KITTI camera maps cam = [y, -z, x] (a left→right-handed flip):

      LIDAR_TO_CAM = [[0, 1, 0, 0],   # cam_x =  agent_y  (right)
                      [0, 0,-1, 0],   # cam_y = -agent_z  (down)
                      [1, 0, 0, 0],   # cam_z =  agent_x  (forward)
                      [0, 0, 0, 1]]

- Inter-agent transform: camB → agentB → world → agentA → camA, i.e.
  ``T_b_to_a = LIDAR_TO_CAM @ inv(T_world_a) @ T_world_b @ inv(LIDAR_TO_CAM)``.
- Heading: a vehicle's camera heading is the relative yaw to the observing
  agent, ``wrap_to_pi(radians(world_yaw - agent_yaw))`` (sign consistent with
  ``utils.fusion.transform_box``'s cam-Y yaw; assumes near-planar vehicles).

Sanity check (implemented in ``_sanity_check_shared_vehicles``): a vehicle seen
by both agents must register to ~0 BEV displacement after ``T_b_to_a`` — a large
displacement means a convention bug (axis map or heading sign).
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from utils.geometry import wrap_to_pi

logger = logging.getLogger(__name__)

# Agent (left-handed: +x fwd, +y right, +z up) → KITTI camera (x-right, y-down,
# z-forward). cam = [agent_y, -agent_z, agent_x].
LIDAR_TO_CAM = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# A GT car counts toward an agent's truth only if at least this many of its
# pixels are actually visible to that agent (gt_boxes ``metrics_metadata``
# visible_pixels_vA/vB). Overridable via stage4.yaml ``carla.min_visible_pixels``.
# This is occlusion-truthful: occluded / out-of-frame cars render 0 pixels and
# the agent's own ego renders ~0 to its own camera, so both drop out naturally.
_DEFAULT_MIN_VISIBLE_PIXELS = 10
# Fallback only — geometric range/FOV gate used when a frame's gt_boxes carry no
# ``metrics_metadata`` (legacy export). A vehicle is "seen" if it is in front
# (z > _MIN_DEPTH_M), within _DEFAULT_MAX_RANGE_M, and its projected centre lands
# inside the image. A box within _SELF_EXCLUDE_M of the agent pose is the ego.
_SELF_EXCLUDE_M = 2.0
_MIN_DEPTH_M = 0.5
_DEFAULT_MAX_RANGE_M = 80.0


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _normalize_ts(timestamp: str) -> str:
    """Normalize a timestamp to the 6-digit frame string used on disk.

    Args:
        timestamp: Frame identifier, numeric ('0', '12') or already padded.

    Returns:
        Zero-padded 6-digit string (e.g. '000012'); returned unchanged if not
        purely numeric.
    """
    ts = str(timestamp)
    return f"{int(ts):06d}" if ts.isdigit() else ts


def _discover_agents(scenario_dir: Path) -> list[str]:
    """List agent sub-directory names in a scenario, sorted.

    An agent directory is any sub-directory containing a ``pose.json``.

    Args:
        scenario_dir: Path to one CARLA scenario folder.

    Returns:
        Sorted list of agent directory names (e.g. ['vehicle_a', 'vehicle_b']).
    """
    return sorted(
        p.name for p in scenario_dir.iterdir()
        if p.is_dir() and (p / "pose.json").exists()
    )


def _resolve_agents(
    scenario_dir: Path,
    agent_a: str | None,
    agent_b: str | None,
) -> tuple[str, str]:
    """Resolve agent A/B IDs, defaulting to the first/second agent on disk.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent_a: Vehicle A agent ID, or None to use the first agent.
        agent_b: Vehicle B agent ID, or None to use the second agent.

    Returns:
        (agent_a, agent_b) directory names.

    Raises:
        ValueError: If the scenario has fewer than two agents.
    """
    agents = _discover_agents(scenario_dir)
    if len(agents) < 2:
        raise ValueError(
            f"Scenario {scenario_dir} has < 2 agents (found {agents}); "
            "Stage 4 V2V fusion needs a pair."
        )
    a = agent_a if agent_a is not None else agents[0]
    b = agent_b if agent_b is not None else agents[1]
    return a, b


def _read_pose(scenario_dir: Path, agent: str, ts: str) -> dict:
    """Read one agent's world pose at a timestamp.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent directory name.
        ts: Normalized 6-digit frame string.

    Returns:
        Pose dict with x, y, z, roll, pitch, yaw (metres / degrees).

    Raises:
        FileNotFoundError: If pose.json is missing.
        KeyError: If the timestamp is absent from pose.json.
    """
    path = scenario_dir / agent / "pose.json"
    if not path.exists():
        raise FileNotFoundError(f"CARLA pose not found: {path}")
    with open(path) as f:
        poses = json.load(f)
    if ts not in poses:
        raise KeyError(f"Timestamp {ts} not in {path}")
    return poses[ts]


def _read_gt_world(scenario_dir: Path, ts: str) -> list[dict]:
    """Read all vehicles' GT boxes (CARLA world coords) at a timestamp.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        ts: Normalized 6-digit frame string.

    Returns:
        List of world-frame box dicts (label, actor_id, x, y, z, l, w, h, yaw).

    Raises:
        FileNotFoundError: If the gt_boxes file is missing.
    """
    path = scenario_dir / "gt_boxes" / f"{ts}.json"
    if not path.exists():
        raise FileNotFoundError(f"CARLA gt_boxes not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _world_from_agent(pose: dict) -> np.ndarray:
    """Build the 4x4 world-from-agent transform from a CARLA pose.

    Replicates CARLA's ``Transform.get_matrix()`` exactly (Euler order: yaw
    about up, then pitch, then roll; left-handed, angles in degrees).

    Args:
        pose: Pose dict with x, y, z, roll, pitch, yaw (metres / degrees).

    Returns:
        4x4 homogeneous transform mapping agent-frame points to world.
    """
    cy, sy = math.cos(math.radians(pose["yaw"])),  math.sin(math.radians(pose["yaw"]))
    cr, sr = math.cos(math.radians(pose["roll"])), math.sin(math.radians(pose["roll"]))
    cp, sp = math.cos(math.radians(pose["pitch"])), math.sin(math.radians(pose["pitch"]))
    return np.array(
        [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, pose["x"]],
         [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, pose["y"]],
         [sp,      -cp * sr,                cp * cr,                 pose["z"]],
         [0.0,      0.0,                    0.0,                     1.0]],
        dtype=np.float64,
    )


def _inter_agent_transform(pose_a: dict, pose_b: dict) -> np.ndarray:
    """Compose T_b_to_a mapping agent B's camera frame into agent A's.

    ``T_b_to_a = LIDAR_TO_CAM @ inv(T_world_a) @ T_world_b @ inv(LIDAR_TO_CAM)``.

    Args:
        pose_a: Vehicle A world pose.
        pose_b: Vehicle B world pose.

    Returns:
        4x4 transform from B's camera frame to A's camera frame.
    """
    t_world_a = _world_from_agent(pose_a)
    t_world_b = _world_from_agent(pose_b)
    lidar_to_cam_inv = np.linalg.inv(LIDAR_TO_CAM)
    return LIDAR_TO_CAM @ np.linalg.inv(t_world_a) @ t_world_b @ lidar_to_cam_inv


def _world_box_to_camera(box_world: dict, pose: dict) -> dict:
    """Convert a world-frame GT box into an observing agent's camera frame.

    Centre is mapped ``cam = LIDAR_TO_CAM @ inv(T_world_agent) @ world``; the
    object size (l, w, h) is frame-independent and carried through; heading is
    the relative yaw ``wrap_to_pi(radians(world_yaw - agent_yaw))``. Confidence
    is 1.0 (ground truth) and ``actor_id`` is carried for the shared-vehicle
    sanity check.

    Args:
        box_world: World-frame GT box (label, actor_id, x, y, z, l, w, h, yaw).
        pose: Observing agent's world pose.

    Returns:
        Camera-frame box dict (label, confidence, x, y, z, l, w, h, heading,
        actor_id).
    """
    t_agent_from_world = np.linalg.inv(_world_from_agent(pose))
    p_world = np.array([box_world["x"], box_world["y"], box_world["z"], 1.0])
    p_cam = LIDAR_TO_CAM @ t_agent_from_world @ p_world
    heading = wrap_to_pi(math.radians(box_world["yaw"] - pose["yaw"]))
    return {
        "label":      box_world["label"],
        "confidence": 1.0,
        "x":          float(p_cam[0]),
        "y":          float(p_cam[1]),
        "z":          float(p_cam[2]),
        "l":          float(box_world["l"]),
        "w":          float(box_world["w"]),
        "h":          float(box_world["h"]),
        "heading":    float(heading),
        "actor_id":   box_world.get("actor_id"),
    }


def _project_center(box_cam: dict, P2: np.ndarray) -> tuple[float, float]:
    """Project a camera-frame box centre to image pixels via P2.

    Args:
        box_cam: Camera-frame box with x, y, z (z assumed > 0).
        P2: 3x4 left-camera projection matrix.

    Returns:
        (u, v) pixel coordinates of the projected centre.
    """
    p = P2 @ np.array([box_cam["x"], box_cam["y"], box_cam["z"], 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def project_box_to_2d(
    box_cam: dict,
    P2: np.ndarray,
    img_w: float,
    img_h: float,
) -> dict:
    """Project a 3D camera-frame box to a 2D image bounding box.

    Builds the box's 8 corners from its centre, dimensions (l, w, h) and
    heading (``rotation_y``, rotation about the camera vertical axis in KITTI
    convention X→right, Y→down, Z→forward), projects each corner with ``P2``,
    and returns the enclosing axis-aligned rectangle clipped to the image. Used
    to draw CARLA GT (which is 3D, not x1y1x2y2) on the detection visualization.

    Args:
        box_cam: Camera-frame box with x, y, z (centre), l, w, h, and
            ``rotation_y`` (or ``heading``) in radians.
        P2: 3x4 left-camera projection matrix.
        img_w: Image width in pixels (for clipping).
        img_h: Image height in pixels (for clipping).

    Returns:
        Dict with label (carried through if present) and x1, y1, x2, y2.
    """
    l, w, h = float(box_cam["l"]), float(box_cam["w"]), float(box_cam["h"])
    ry = float(box_cam.get("rotation_y", box_cam.get("heading", 0.0)))

    # Centre-based corner offsets (object frame): length along X, height along
    # Y (down), width along Z, before the heading rotation about Y.
    x_c = np.array([ l / 2,  l / 2, -l / 2, -l / 2,  l / 2,  l / 2, -l / 2, -l / 2])
    y_c = np.array([ h / 2,  h / 2,  h / 2,  h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    z_c = np.array([ w / 2, -w / 2, -w / 2,  w / 2,  w / 2, -w / 2, -w / 2,  w / 2])

    cos_r, sin_r = math.cos(ry), math.sin(ry)
    R = np.array([[cos_r, 0.0, sin_r], [0.0, 1.0, 0.0], [-sin_r, 0.0, cos_r]])
    corners = R @ np.vstack([x_c, y_c, z_c])
    corners[0] += box_cam["x"]
    corners[1] += box_cam["y"]
    corners[2] += box_cam["z"]

    # Clamp depth to a small positive value so corners that fall behind the
    # image plane do not blow up the projection.
    corners[2] = np.maximum(corners[2], 0.1)

    homog = P2 @ np.vstack([corners, np.ones(8)])
    u = homog[0] / homog[2]
    v = homog[1] / homog[2]

    box2d = {
        "x1": float(np.clip(u.min(), 0.0, img_w)),
        "y1": float(np.clip(v.min(), 0.0, img_h)),
        "x2": float(np.clip(u.max(), 0.0, img_w)),
        "y2": float(np.clip(v.max(), 0.0, img_h)),
    }
    if "label" in box_cam:
        box2d["label"] = box_cam["label"]
    return box2d


# ---------------------------------------------------------------------------
# Per-agent GT detections (visibility-filtered world boxes in camera frame)
# ---------------------------------------------------------------------------

def _visibility_key(agent: str) -> str:
    """Map an agent directory name to its gt_boxes visibility metadata key.

    The export records per-vehicle visible-pixel counts keyed by agent role:
    ``visible_pixels_vA`` for ``vehicle_a``, ``visible_pixels_vB`` for
    ``vehicle_b`` (the suffix after the last underscore, upper-cased).

    Args:
        agent: Agent directory name (e.g. 'vehicle_a').

    Returns:
        The metadata key (e.g. 'visible_pixels_vA').
    """
    return f"visible_pixels_v{agent.split('_')[-1].upper()}"


def _agent_gt_boxes(
    gt_world: list[dict],
    pose: dict,
    calib: dict,
    agent: str,
    min_visible_pixels: int = _DEFAULT_MIN_VISIBLE_PIXELS,
) -> list[dict]:
    """Build one agent's GT detections: world cars actually visible to it.

    Truthful visibility: a car is kept only if at least ``min_visible_pixels`` of
    its pixels are visible to this agent (``metrics_metadata`` visible_pixels_vA/
    vB in gt_boxes). This naturally excludes occluded cars, cars outside the
    image, and the agent's own ego (which renders ~0 pixels to its own camera) —
    none of which the geometric centre test could reject.

    Legacy fallback: if the frame's boxes carry no ``metrics_metadata`` at all,
    fall back to the geometric FOV/range filter (``_agent_gt_boxes_geometric``).

    Args:
        gt_world: All vehicles' world-frame GT boxes at this frame.
        pose: Observing agent's world pose.
        calib: Observing agent's calibration (needs P2, image_w, image_h).
        agent: Observing agent's directory name (selects the visibility key).
        min_visible_pixels: Minimum visible pixels for a car to count as seen.

    Returns:
        List of camera-frame box dicts (carry ``visible_pixels``).
    """
    has_meta = any("metrics_metadata" in b for b in gt_world)
    if not has_meta:
        logger.warning(
            "gt_boxes carry no metrics_metadata — falling back to the geometric "
            "visibility filter (occluded cars may be over-counted)."
        )
        return _agent_gt_boxes_geometric(gt_world, pose, calib, _DEFAULT_MAX_RANGE_M)

    key = _visibility_key(agent)
    boxes: list[dict] = []
    for b in gt_world:
        vis = int(b.get("metrics_metadata", {}).get(key, 0))
        if vis < min_visible_pixels:
            continue
        cam = _world_box_to_camera(b, pose)
        cam["visible_pixels"] = vis
        boxes.append(cam)
    return boxes


def _agent_gt_boxes_geometric(
    gt_world: list[dict],
    pose: dict,
    calib: dict,
    max_range_m: float,
) -> list[dict]:
    """Legacy geometric visibility filter (no occlusion handling).

    Used only as a fallback when gt_boxes lack ``metrics_metadata``. Excludes the
    agent's own vehicle (nearest world box within ``_SELF_EXCLUDE_M``) and keeps
    vehicles in front of and inside the camera FOV within ``max_range_m``.

    Args:
        gt_world: All vehicles' world-frame GT boxes at this frame.
        pose: Observing agent's world pose.
        calib: Observing agent's calibration (needs P2, image_w, image_h).
        max_range_m: Max camera-frame depth (z) to keep a detection.

    Returns:
        List of FOV-visible camera-frame box dicts.
    """
    # Exclude the agent's own vehicle: the single nearest world box to its pose.
    self_idx, self_d = -1, float("inf")
    for i, b in enumerate(gt_world):
        d = math.hypot(b["x"] - pose["x"], b["y"] - pose["y"])
        if d < self_d:
            self_idx, self_d = i, d
    if self_d > _SELF_EXCLUDE_M:
        self_idx = -1  # no box close enough to be self (defensive)

    P2 = np.asarray(calib["P2"], dtype=np.float64)
    img_w, img_h = float(calib["image_w"]), float(calib["image_h"])

    boxes: list[dict] = []
    for i, b in enumerate(gt_world):
        if i == self_idx:
            continue
        cam = _world_box_to_camera(b, pose)
        if cam["z"] <= _MIN_DEPTH_M or cam["z"] > max_range_m:
            continue
        u, v = _project_center(cam, P2)
        if 0.0 <= u <= img_w and 0.0 <= v <= img_h:
            boxes.append(cam)
    return boxes


def _sanity_check_shared_vehicles(
    boxes_a: list[dict],
    boxes_b: list[dict],
    T_b_to_a: np.ndarray,
    tol_m: float = 0.1,
) -> None:
    """Warn if a vehicle seen by both agents fails to register to ~0 BEV.

    Matches A/B detections by ``actor_id``, transforms each B box into A's frame
    and measures BEV (x-z) displacement. A large value signals a convention bug
    (axis map or pose Euler order), not a moving object — the agents are
    simultaneous.

    Args:
        boxes_a: Vehicle A camera-frame boxes (carry actor_id).
        boxes_b: Vehicle B camera-frame boxes (carry actor_id).
        T_b_to_a: 4x4 transform mapping B's camera frame to A's.
        tol_m: Displacement above which a warning is logged.
    """
    by_actor_a = {b["actor_id"]: b for b in boxes_a if b.get("actor_id") is not None}
    max_d, n_shared = 0.0, 0
    for b in boxes_b:
        a = by_actor_a.get(b.get("actor_id"))
        if a is None:
            continue
        p = T_b_to_a @ np.array([b["x"], b["y"], b["z"], 1.0])
        max_d = max(max_d, math.hypot(a["x"] - p[0], a["z"] - p[2]))
        n_shared += 1

    if n_shared == 0:
        logger.warning("Sanity check: no vehicle seen by both agents this frame.")
        return
    level = logging.WARNING if max_d > tol_m else logging.DEBUG
    logger.log(
        level,
        "Sanity check: %d shared vehicles, max BEV registration error %.4f m "
        "(tol %.2f m).", n_shared, max_d, tol_m,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_carla_transform(
    scenario_dir: str,
    timestamp: str,
    agent_a: str | None = None,
    agent_b: str | None = None,
) -> tuple[np.ndarray, str, str, str]:
    """Compose the inter-agent transform for a pair, from poses only.

    Unlike ``load_carla_pair`` this reads no GT boxes and runs no detector — it
    needs only the two agents' poses. The Stage-4 detector path uses it to get
    ``T_b_to_a`` to register Vehicle B's per-agent detections into A's frame.

    Args:
        scenario_dir: Path to one CARLA scenario folder (contains agent subdirs).
        timestamp: Timestamp string identifying the frame.
        agent_a: Vehicle A agent ID. None → first agent in the scenario.
        agent_b: Vehicle B agent ID. None → second agent in the scenario.

    Returns:
        (T_b_to_a, agent_a, agent_b, scene_id) — 4x4 transform mapping B's
        camera frame into A's, the resolved agent IDs, and a unique scene
        identifier (same format as ``load_carla_pair``).
    """
    scenario = Path(scenario_dir)
    ts = _normalize_ts(timestamp)
    agent_a, agent_b = _resolve_agents(scenario, agent_a, agent_b)

    pose_a = _read_pose(scenario, agent_a, ts)
    pose_b = _read_pose(scenario, agent_b, ts)
    T_b_to_a = _inter_agent_transform(pose_a, pose_b)

    scene_id = f"{scenario.name}_{agent_a}_{agent_b}_{ts}"
    return T_b_to_a, agent_a, agent_b, scene_id


def load_carla_pair(
    scenario_dir: str,
    timestamp: str,
    agent_a: str | None = None,
    agent_b: str | None = None,
    use_gt_boxes: bool = True,
    min_visible_pixels: int = _DEFAULT_MIN_VISIBLE_PIXELS,
) -> tuple[list[dict], list[dict], np.ndarray, str]:
    """Load one CARLA agent pair at a timestamp for Stage 4 fusion.

    Args:
        scenario_dir: Path to one CARLA scenario folder (contains agent subdirs).
        timestamp: Timestamp string identifying the frame to load.
        agent_a: Vehicle A agent ID. None → first agent in the scenario.
        agent_b: Vehicle B agent ID. None → second agent in the scenario.
        use_gt_boxes: True → fuse ground-truth boxes (baseline / smoke test);
            False → per-agent detector boxes (run Stages 1-3 per agent).
        min_visible_pixels: Minimum visible pixels for a GT car to count as seen
            by an agent (per-agent visibility filter).

    Returns:
        (boxes_a, boxes_b, T_b_to_a, scene_id) — boxes in each agent's KITTI
        camera frame, T_b_to_a mapping B's camera frame into A's, and a unique
        scene identifier for the output filename.

    Raises:
        NotImplementedError: If use_gt_boxes is False — the per-agent detector
            path (Stages 1-3 on each CARLA agent) is not wired yet.
    """
    scenario = Path(scenario_dir)
    ts = _normalize_ts(timestamp)
    agent_a, agent_b = _resolve_agents(scenario, agent_a, agent_b)

    if not use_gt_boxes:
        raise NotImplementedError(
            "use_gt_boxes=False (per-agent detector fusion) requires running "
            "Stages 1-3 on each CARLA agent's stereo and loading their Stage 3 "
            "3D positions; that pipeline is not wired yet. Use use_gt_boxes=True "
            "for the GT baseline (utils/carla_loader.py)."
        )

    pose_a = _read_pose(scenario, agent_a, ts)
    pose_b = _read_pose(scenario, agent_b, ts)
    gt_world = _read_gt_world(scenario, ts)

    calib_a = load_carla_calib(scenario_dir, agent_a)
    calib_b = load_carla_calib(scenario_dir, agent_b)

    boxes_a = _agent_gt_boxes(gt_world, pose_a, calib_a, agent_a, min_visible_pixels)
    boxes_b = _agent_gt_boxes(gt_world, pose_b, calib_b, agent_b, min_visible_pixels)
    T_b_to_a = _inter_agent_transform(pose_a, pose_b)

    _sanity_check_shared_vehicles(boxes_a, boxes_b, T_b_to_a)

    scene_id = f"{scenario.name}_{agent_a}_{agent_b}_{ts}"
    logger.info(
        "Loaded CARLA pair %s@%s — A(%s)=%d boxes, B(%s)=%d boxes.",
        scenario.name, ts, agent_a, len(boxes_a), agent_b, len(boxes_b),
    )
    return boxes_a, boxes_b, T_b_to_a, scene_id


def load_carla_calib(scenario_dir: str, agent: str) -> dict:
    """Load one CARLA agent's camera calibration.

    Mirrors ``utils.kitti_tracking_loader.load_tracking_calib``: returns the
    projection matrices as arrays plus stereo geometry, with the extra image
    dimensions the CARLA export records.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent directory name whose calibration to load.

    Returns:
        Dict with P2, P3 (np.ndarray 3x4), focal_length_px, baseline_m, cx, cy,
        image_w, image_h.

    Raises:
        FileNotFoundError: If calib.json is missing.
    """
    path = Path(scenario_dir) / agent / "calib.json"
    if not path.exists():
        raise FileNotFoundError(f"CARLA calib not found: {path}")
    with open(path) as f:
        calib = json.load(f)

    P2 = np.asarray(calib["P2"], dtype=np.float64).reshape(3, 4)
    P3 = np.asarray(calib["P3"], dtype=np.float64).reshape(3, 4)
    return {
        "P2":              P2,
        "P3":              P3,
        "focal_length_px": float(calib["focal_length_px"]),
        "baseline_m":      float(calib["baseline_m"]),
        "cx":              float(calib["cx"]),
        "cy":              float(calib["cy"]),
        "image_w":         int(calib["image_w"]),
        "image_h":         int(calib["image_h"]),
    }


def load_carla_image(
    scenario_dir: str,
    agent: str,
    camera: str,
    timestamp: str,
) -> np.ndarray:
    """Load a single CARLA stereo image.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent directory name.
        camera: 'image_left' or 'image_right'.
        timestamp: Timestamp string identifying the frame.

    Returns:
        BGR image, shape (H, W, 3), uint8.

    Raises:
        FileNotFoundError: If the image file is missing.
        ValueError: If OpenCV fails to decode the file.
    """
    ts = _normalize_ts(timestamp)
    path = Path(scenario_dir) / agent / camera / f"{ts}.png"
    if not path.exists():
        raise FileNotFoundError(f"CARLA image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV failed to read: {path}")
    return img


def load_carla_disparity_gt(
    scenario_dir: str,
    agent: str,
    timestamp: str,
) -> np.ndarray:
    """Load one CARLA agent's ground-truth disparity map.

    The export stores GT disparity as a float32 ``.npy`` under the agent's
    ``disp_noc_0/`` directory (mirroring KITTI's ``disp_noc_0`` naming). Pixels
    with value <= 0 are invalid (sky / out of range) and are returned as np.nan,
    matching the invalid-pixel contract of
    ``utils.kitti_loader.load_disparity_gt`` so the Stage-1 metrics work unchanged.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent directory name (e.g. 'vehicle_a').
        timestamp: Timestamp string identifying the frame.

    Returns:
        Disparity map, shape (H, W), float32. Invalid pixels = np.nan.

    Raises:
        FileNotFoundError: If the disparity .npy is missing.
    """
    ts = _normalize_ts(timestamp)
    path = Path(scenario_dir) / agent / "disp_noc_0" / f"{ts}_disp.npy"
    if not path.exists():
        raise FileNotFoundError(f"CARLA disparity GT not found: {path}")

    raw = np.load(str(path)).astype(np.float32)
    disp = raw.copy()
    disp[raw <= 0] = np.nan
    return disp


def load_carla_frame(
    scenario_dir: str,
    agent: str,
    timestamp: str,
    min_visible_pixels: int = _DEFAULT_MIN_VISIBLE_PIXELS,
) -> dict:
    """Load all data for a single CARLA agent frame in one call.

    Mirrors ``utils.kitti_tracking_loader.load_tracking_frame``: returns the
    left/right stereo images, calib, and the agent's GT labels (every other
    vehicle's GT box expressed in this agent's camera frame, visibility-filtered)
    for Stage 3 validation. Labels carry ``rotation_y`` (= camera heading) so they
    match the KITTI label schema; matching is by 3D centre distance.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent directory name.
        timestamp: Timestamp string identifying the frame.
        min_visible_pixels: Minimum visible pixels for a GT car to count as seen.

    Returns:
        Dict with agent, frame_id, left, right, calib, labels.
    """
    ts = _normalize_ts(timestamp)
    calib = load_carla_calib(scenario_dir, agent)
    pose = _read_pose(Path(scenario_dir), agent, ts)
    gt_world = _read_gt_world(Path(scenario_dir), ts)

    boxes = _agent_gt_boxes(gt_world, pose, calib, agent, min_visible_pixels)
    labels = [
        {
            "label":      b["label"],
            "x":          b["x"],
            "y":          b["y"],
            "z":          b["z"],
            "l":          b["l"],
            "w":          b["w"],
            "h":          b["h"],
            "rotation_y": b["heading"],
            "track_id":   b["actor_id"],
        }
        for b in boxes
    ]
    return {
        "agent":    agent,
        "frame_id": ts,
        "left":     load_carla_image(scenario_dir, agent, "image_left", ts),
        "right":    load_carla_image(scenario_dir, agent, "image_right", ts),
        "calib":    calib,
        "labels":   labels,
    }
