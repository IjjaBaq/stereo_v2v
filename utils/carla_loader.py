"""CARLA data loader for Stage 4 cooperative fusion (real simultaneous V2V).

STATUS: STUB. CARLA is not wired yet — the export schema and a sample scenario
are not available at the time of writing, so nothing here is implemented. The
signatures below define the exact contract the rest of the pipeline already
expects; fill them in against a real CARLA export. Until then every function
raises ``NotImplementedError`` (no implementation that could be silently wrong).

What the pipeline needs from a CARLA export
-------------------------------------------
Stage 4 fusion (``stages.stage4_fusion``) consumes, per agent pair at one
timestamp::

    boxes_a, boxes_b, T_b_to_a, scene_id = load_carla_pair(...)

where ``boxes_a`` / ``boxes_b`` are each agent's 3D boxes in the **KITTI camera
convention** (x-right, y-down, z-forward; BEV = x-z plane; heading about the
y-axis) so the source-agnostic core in ``utils.fusion`` works unchanged, and
``T_b_to_a`` maps agent B's camera frame into agent A's.

Stage 3 validation on CARLA frames (``stages.validate_stage3_lift``) needs the
per-frame loaders ``load_carla_frame`` / ``load_carla_calib``, mirroring the
``load_tracking_frame`` / ``load_tracking_calib`` contract in
``utils.kitti_tracking_loader``.

Coordinate conventions to honor (CARLA, verified previously)
------------------------------------------------------------
These are CARLA's own conventions; reuse them when implementing the transforms.
- World/agent frame: **left-handed**, +x forward, +y right, +z up.
- Agent pose: ``[x, y, z, roll, yaw, pitch]`` in metres and **degrees**. The
  world-from-agent matrix follows CARLA's Euler order (yaw about up, then pitch,
  then roll) — replicate it exactly rather than guessing the mixing.
- Object annotations: world ``location`` [x,y,z], ``angle`` [roll,yaw,pitch]
  (deg), ``extent`` (half-sizes) → length=2*ext_x, width=2*ext_y, height=2*ext_z.
- LiDAR/agent → KITTI camera maps cam = [y, -z, x] (a left→right-handed flip):

      LIDAR_TO_CAM = [[0, 1, 0, 0],   # cam_x =  agent_y  (right)
                      [0, 0,-1, 0],   # cam_y = -agent_z  (down)
                      [1, 0, 0, 0],   # cam_z =  agent_x  (forward)
                      [0, 0, 0, 1]]

- Inter-agent transform: camB → agentB → world → agentA → camA, i.e.
  ``T_b_to_a = LIDAR_TO_CAM @ inv(T_world_a) @ T_world_b @ inv(LIDAR_TO_CAM)``.
- Heading: a vehicle's camera heading is the relative yaw to the observing
  agent, ``wrap_to_pi(radians(world_yaw - agent_yaw))`` (sign consistent with
  ``utils.fusion.transform_box``'s cam-Y yaw; assumes near-planar vehicles).

Sanity check to implement alongside: a vehicle seen by both agents must register
to ~0 BEV displacement after ``T_b_to_a`` — a large displacement means a
convention bug (axis map or heading sign).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_carla_pair(
    scenario_dir: str,
    timestamp: str,
    agent_a: str | None = None,
    agent_b: str | None = None,
    use_gt_boxes: bool = True,
) -> tuple[list[dict], list[dict], np.ndarray, str]:
    """Load one CARLA agent pair at a timestamp for Stage 4 fusion.

    Args:
        scenario_dir: Path to one CARLA scenario folder (contains agent subdirs).
        timestamp: Timestamp string identifying the frame to load.
        agent_a: Vehicle A agent ID. None → first agent in the scenario.
        agent_b: Vehicle B agent ID. None → second agent in the scenario.
        use_gt_boxes: True → fuse ground-truth boxes (baseline / smoke test);
            False → per-agent detector boxes (run Stages 1-3 per agent).

    Returns:
        (boxes_a, boxes_b, T_b_to_a, scene_id) — boxes in each agent's KITTI
        camera frame, T_b_to_a mapping B's camera frame into A's, and a unique
        scene identifier for the output filename.

    Raises:
        NotImplementedError: Always — CARLA loader not yet wired.
    """
    # TODO(carla): once a sample export exists —
    #   1. Confirm the on-disk layout and per-agent annotation schema
    #      (pose field name/units, object location/angle/extent, calib).
    #   2. Build T_world_a / T_world_b from each agent's pose (CARLA Euler order).
    #   3. Convert each agent's GT vehicles to camera-frame boxes via LIDAR_TO_CAM
    #      (see module docstring); set length/width/height from extent half-sizes.
    #   4. Compose T_b_to_a = LIDAR_TO_CAM @ inv(T_world_a) @ T_world_b @ inv(LIDAR_TO_CAM).
    #   5. For use_gt_boxes=False, load per-agent Stage 3 detector boxes instead.
    #   6. Add the shared-vehicle ~0-displacement sanity check.
    raise NotImplementedError(
        "CARLA loader is not implemented yet. Provide a sample CARLA export and "
        "fill in load_carla_pair following the conventions in this module's "
        "docstring (utils/carla_loader.py)."
    )


def load_carla_calib(scenario_dir: str, agent: str) -> dict:
    """Load one CARLA agent's camera calibration.

    Mirrors ``utils.kitti_tracking_loader.load_tracking_calib``: should return at
    least ``P2`` (3x4 projection), ``focal_length_px`` and ``baseline_m`` so
    Stages 1-3 can run on CARLA stereo.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent ID whose calibration to load.

    Returns:
        Calibration dict (see load_tracking_calib for the expected keys).

    Raises:
        NotImplementedError: Always — CARLA loader not yet wired.
    """
    # TODO(carla): derive P2/focal/baseline from the CARLA camera intrinsics and
    # the stereo rig baseline used at capture time.
    raise NotImplementedError(
        "CARLA calibration loading is not implemented yet (utils/carla_loader.py)."
    )


def load_carla_frame(scenario_dir: str, agent: str, timestamp: str) -> dict:
    """Load all data for a single CARLA agent frame in one call.

    Mirrors ``utils.kitti_tracking_loader.load_tracking_frame``: should return
    the left/right stereo images, calib, and GT labels for Stage 3 validation.

    Args:
        scenario_dir: Path to one CARLA scenario folder.
        agent: Agent ID.
        timestamp: Timestamp string identifying the frame.

    Returns:
        Frame bundle dict (left, right, calib, labels, ...).

    Raises:
        NotImplementedError: Always — CARLA loader not yet wired.
    """
    # TODO(carla): read the agent's stereo PNGs, calib (load_carla_calib) and GT
    # labels for this timestamp; match the load_tracking_frame return shape.
    raise NotImplementedError(
        "CARLA frame loading is not implemented yet (utils/carla_loader.py)."
    )
