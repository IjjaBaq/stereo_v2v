"""CARLA Stereo V2V Data Collection Script.

Captures a realistic urban V2V sequence at a signalized intersection in
Town10HD using two moving ego vehicles with ambient NPC traffic:
- Vehicle A (Idx 148): Horizontal Street Approach
- Vehicle B (Idx 53):  Vertical Avenue Approach

Both vehicles follow realistic traffic rules including traffic light cycles.
NPC traffic clustered near the intersection provides realistic occlusion
scenarios for V2V cooperative perception evaluation.

Output structure:
    C:/carla_data/
    ├── vehicle_a/
    │   ├── image_left/     {frame:06d}.png  — left stereo image
    │   ├── image_right/    {frame:06d}.png  — right stereo image
    │   ├── disp_noc_0/     {frame:06d}_disp.npy / .png — GT disparity
    │   ├── calib.json      — camera intrinsics + stereo baseline
    │   └── pose.json       — world transform per frame
    ├── vehicle_b/
    │   ├── image_left/
    │   ├── image_right/
    │   ├── disp_noc_0/
    │   ├── calib.json
    │   └── pose.json
    └── gt_boxes/
        └── {frame:06d}.json  — GT 3D boxes with per-vehicle pixel visibility

Usage:
    python scripts/collect_carla_data.py
    python scripts/collect_carla_data.py --frames 300 --output C:/carla_data
"""

import argparse
import json
import logging
import math
import queue
import time
from pathlib import Path

import carla
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spawn configuration
# ---------------------------------------------------------------------------

SPAWN_IDX_A = 148  # Vehicle A: Horizontal Street Approach
SPAWN_IDX_B = 53   # Vehicle B: Vertical Avenue Approach

JUNCTION_CENTER      = carla.Location(x=91.7, y=11.6, z=0.5)
TRAFFIC_SPAWN_RADIUS = 80.0
N_NPC_VEHICLES       = 40

# ---------------------------------------------------------------------------
# Stereo camera configuration (KITTI Tracking compatible)
# ---------------------------------------------------------------------------

BASELINE_M  = 0.54
IMAGE_W     = 1242
IMAGE_H     = 375
FOV         = 90.0

CAM_MOUNT_X = 1.6
CAM_MOUNT_Z = 1.4

# Valid GT-disparity depth range (metres). Pixels nearer than MIN_DEPTH_M are
# CARLA depth-buffer artifacts (scattered ~0 m values at thin silhouettes /
# glass) or the ego's own bodywork; pixels beyond MAX_DEPTH_M are sky / distant
# background. Both are marked invalid (disparity 0) rather than converted, so the
# 1/Z term never explodes into non-physical disparities.
MIN_DEPTH_M = 1.0
MAX_DEPTH_M = 80.0

# CARLA 0.9.16 semantic-segmentation tag palette (Cityscapes-extended). The
# instance-segmentation camera stores the semantic tag in the RED channel and a
# per-object instance id in the GREEN/BLUE channels. Vehicle tags in 0.9.16 are:
#   13 Rider | 14 Car | 15 Truck | 16 Bus | 17 Train | 18 Motorcycle | 19 Bicycle
# NOTE: tag 10 is NOT vehicles in 0.9.16 (it is a static class); "Vehicles = 10"
# is the legacy palette from CARLA <= 0.9.13. Only 4-wheeled vehicles are spawned
# here (Car/Truck/Bus), but the full vehicle set is matched for robustness.
_VEHICLE_SEMANTIC_TAGS = (14, 15, 16, 18, 19)

# Half-size of the patch (pixels) around a projected actor center used
# to count vehicle-tagged pixels for visibility estimation
_VISIBILITY_PATCH_PX = 30

# CARLA agent frame (left-handed: +x fwd, +y right, +z up)
# → KITTI camera frame (x-right, y-down, z-forward)
# cam = [agent_y, -agent_z, agent_x]
_LIDAR_TO_CAM = np.array(
    [[0.0,  1.0,  0.0, 0.0],
     [0.0,  0.0, -1.0, 0.0],
     [1.0,  0.0,  0.0, 0.0],
     [0.0,  0.0,  0.0, 1.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def focal_length_px(image_w: int, fov_deg: float) -> float:
    """Compute focal length in pixels from image width and horizontal FOV.

    Args:
        image_w: Image width in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        Focal length in pixels.
    """
    return image_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))


def build_calib(
    image_w: int,
    image_h: int,
    fov_deg: float,
    baseline_m: float,
) -> dict:
    """Build a calibration dict compatible with utils/carla_loader.py.

    Args:
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        fov_deg: Horizontal field of view in degrees.
        baseline_m: Stereo baseline in metres.

    Returns:
        Calibration dict with focal_length_px, cx, cy, baseline_m, P2, P3.
    """
    f  = focal_length_px(image_w, fov_deg)
    cx = image_w / 2.0
    cy = image_h / 2.0

    P2 = [
        [f,   0.0, cx,  0.0],
        [0.0, f,   cy,  0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    P3 = [
        [f,   0.0, cx,  -f * baseline_m],
        [0.0, f,   cy,   0.0],
        [0.0, 0.0, 1.0,  0.0],
    ]

    return {
        "focal_length_px": f,
        "cx":              cx,
        "cy":              cy,
        "baseline_m":      baseline_m,
        "image_w":         image_w,
        "image_h":         image_h,
        "fov_deg":         fov_deg,
        "P2":              P2,
        "P3":              P3,
    }


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    """Convert a carla.Image (BGRA) to a BGR numpy array.

    Args:
        image: CARLA RGB image (internally stored as BGRA).

    Returns:
        BGR image, shape (H, W, 3), uint8.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def carla_depth_to_meters(carla_image: carla.Image) -> np.ndarray:
    """Convert CARLA raw depth image to a float32 array in metres.

    CARLA encodes depth as a 24-bit value across R, G, B channels:
        normalized = (R + G*256 + B*256^2) / (256^3 - 1)
        depth_m    = normalized * 1000.0  (max range = 1000 m)

    Args:
        carla_image: Raw depth image from sensor.camera.depth.

    Returns:
        Float32 depth array, shape (H, W), in metres.
    """
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))

    r = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    b = array[:, :, 2].astype(np.float32)

    normalized = (r + g * 256.0 + b * 65536.0) / 16777215.0
    return (normalized * 1000.0).astype(np.float32)


def depth_meters_to_kitti_disparity(
    depth_meters: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
) -> np.ndarray:
    """Convert metric depth to KITTI-compatible disparity in pixels.

    Disparity = f * B / Z, computed only where the depth is inside the reliable
    range [MIN_DEPTH_M, MAX_DEPTH_M]; all other pixels are marked invalid (0).
    The near cut-off is essential: CARLA's depth buffer emits scattered ~0 m
    values at thin silhouettes / glass, and dividing by those explodes 1/Z
    (e.g. 0.1 m -> f*B/0.1 = 3353 px for f*B = 335), producing non-physical GT
    disparities far above any stereo matcher's output range. The far cut-off
    drops sky / distant background, which give unreliable matches.

    Args:
        depth_meters: Float32 depth array, shape (H, W).
        focal_length_px: Horizontal focal length in pixels.
        baseline_m: Stereo baseline in metres.

    Returns:
        Float32 disparity array, shape (H, W). 0 = invalid.
    """
    valid     = (depth_meters >= MIN_DEPTH_M) & (depth_meters <= MAX_DEPTH_M)
    disparity = np.zeros_like(depth_meters, dtype=np.float32)
    disparity[valid] = (focal_length_px * baseline_m) / depth_meters[valid]
    return disparity


def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    """Render a disparity map as a MAGMA colormap image.

    Invalid pixels (disp == 0) are rendered black.

    Args:
        disp: Float32 disparity array, shape (H, W). 0 = invalid.

    Returns:
        BGR colormap image, shape (H, W, 3), uint8.
    """
    valid = disp > 0
    vis   = np.zeros_like(disp)

    if valid.any():
        d_min = disp[valid].min()
        d_max = disp[valid].max()
        if d_max > d_min:
            vis[valid] = (disp[valid] - d_min) / (d_max - d_min) * 255.0
        else:
            vis[valid] = 255.0

    colored         = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_MAGMA)
    colored[~valid] = 0
    return colored


# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------

def _world_from_agent(pose: dict) -> np.ndarray:
    """Build 4x4 world-from-agent transform from a CARLA pose dict.

    Replicates CARLA's Transform.get_matrix() Euler order exactly
    (yaw about up, then pitch, then roll; left-handed; degrees).

    Args:
        pose: Dict with x, y, z, roll, pitch, yaw (metres / degrees).

    Returns:
        4x4 homogeneous transform matrix.
    """
    cy = math.cos(math.radians(pose["yaw"]))
    sy = math.sin(math.radians(pose["yaw"]))
    cr = math.cos(math.radians(pose["roll"]))
    sr = math.sin(math.radians(pose["roll"]))
    cp = math.cos(math.radians(pose["pitch"]))
    sp = math.sin(math.radians(pose["pitch"]))

    return np.array([
        [cp*cy,  cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr, pose["x"]],
        [cp*sy,  sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr, pose["y"]],
        [sp,    -cp*sr,             cp*cr,             pose["z"]],
        [0.0,   0.0,               0.0,               1.0      ],
    ], dtype=np.float64)


def calculate_actor_visibility(
    instance_image: carla.Image,
    actor_world_pos: tuple,
    pose: dict,
    calib: dict,
    cam_mount: tuple = (CAM_MOUNT_X, -BASELINE_M / 2.0, CAM_MOUNT_Z),
    patch_px: int = _VISIBILITY_PATCH_PX,
) -> int:
    """Count vehicle-tagged pixels near an actor's projected image location.

    Projects the actor's world position into the left camera's optical frame
    (the vehicle pose composed with the camera mount offset) and counts pixels
    whose semantic tag is a vehicle class within a patch around the projection.

    The instance-segmentation raw buffer is BGRA: the semantic tag is in the RED
    channel (index 2), while the per-object instance id is in the GREEN/BLUE
    channels (indices 1/0). We read the RED channel and match the 0.9.16 vehicle
    tags (_VEHICLE_SEMANTIC_TAGS). Reading index 0 instead compares the
    instance-id byte against a tag value, which is meaningless and registers a
    vehicle only when its instance-id low byte coincidentally equals the tag.

    Args:
        instance_image: Frame from sensor.camera.instance_segmentation.
        actor_world_pos: (x, y, z) world position of the actor in metres.
        pose: Observing agent's vehicle world pose dict (x, y, z, roll, pitch,
            yaw).
        calib: Camera calibration dict with P2, image_w, image_h.
        cam_mount: Left-camera mount offset (x, y, z) in the vehicle frame.
        patch_px: Half-size of patch around projected center in pixels.

    Returns:
        Count of vehicle pixels in the patch around the projected center.
        0 means the actor is not visible or outside the FOV.
    """
    array = np.frombuffer(instance_image.raw_data, dtype=np.uint8)
    array = array.reshape((instance_image.height, instance_image.width, 4))

    # Semantic tag is in the RED channel (index 2) of the BGRA buffer.
    vehicle_mask = np.isin(array[:, :, 2], _VEHICLE_SEMANTIC_TAGS)

    # World → left-camera optical frame. The camera world transform is the
    # vehicle pose composed with the mount translation (the camera carries no
    # rotation relative to the vehicle), so the projection matches where the
    # actor actually appears in this camera's segmentation image.
    t_world_cam        = _world_from_agent(pose).copy()
    t_world_cam[:, 3]  = _world_from_agent(pose) @ np.array([*cam_mount, 1.0])
    p_world = np.array([
        actor_world_pos[0],
        actor_world_pos[1],
        actor_world_pos[2],
        1.0,
    ])
    p_cam = _LIDAR_TO_CAM @ np.linalg.inv(t_world_cam) @ p_world

    # Actor is behind the camera
    if p_cam[2] <= 0:
        return 0

    # Project to pixel coordinates via P2
    P2  = np.asarray(calib["P2"], dtype=np.float64)
    uvw = P2 @ p_cam
    u   = int(round(uvw[0] / uvw[2]))
    v   = int(round(uvw[1] / uvw[2]))

    img_w = calib["image_w"]
    img_h = calib["image_h"]

    # Outside image bounds
    if not (0 <= u < img_w and 0 <= v < img_h):
        return 0

    # Count vehicle pixels in patch around projected center
    y1 = max(0,     v - patch_px)
    y2 = min(img_h, v + patch_px)
    x1 = max(0,     u - patch_px)
    x2 = min(img_w, u + patch_px)

    return int(vehicle_mask[y1:y2, x1:x2].sum())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def transform_to_dict(t: carla.Transform) -> dict:
    """Serialize a carla.Transform to a plain dict.

    Args:
        t: CARLA transform (location + rotation).

    Returns:
        Dict with x, y, z, roll, pitch, yaw (metres / degrees).
    """
    return {
        "x":     t.location.x,
        "y":     t.location.y,
        "z":     t.location.z,
        "roll":  t.rotation.roll,
        "pitch": t.rotation.pitch,
        "yaw":   t.rotation.yaw,
    }


def get_gt_boxes_with_visibility(
    world: carla.World,
    inst_img_a: carla.Image,
    inst_img_b: carla.Image,
    pose_a: dict,
    pose_b: dict,
    calib: dict,
) -> list:
    """Get GT 3D boxes for all vehicles with per-agent pixel visibility counts.

    Uses semantic-tag-based visibility (robust to instance ID encoding issues)
    to count how many vehicle pixels are visible near each actor's projected
    location in each camera frame.

    Args:
        world: CARLA world object.
        inst_img_a: Instance segmentation frame from Vehicle A left camera.
        inst_img_b: Instance segmentation frame from Vehicle B left camera.
        pose_a: Vehicle A world pose dict.
        pose_b: Vehicle B world pose dict.
        calib: Shared camera calibration dict.

    Returns:
        List of GT box dicts with label, actor_id, x, y, z, l, w, h, yaw,
        and metrics_metadata.visible_pixels_vA/vB.
    """
    boxes = []
    for actor in world.get_actors().filter("vehicle.*"):
        t   = actor.get_transform()
        bb  = actor.bounding_box

        # True 3D bounding-box center in world coordinates. The actor origin sits
        # at the vehicle base; bb.location is the center offset in the vehicle
        # frame (~half the height up). Transform it into the world frame —
        # using t.location alone places the box ~h/2 too low (at the wheels).
        center = carla.Location(x=bb.location.x, y=bb.location.y, z=bb.location.z)
        t.transform(center)
        pos = (center.x, center.y, center.z)

        pixels_a = calculate_actor_visibility(inst_img_a, pos, pose_a, calib)
        pixels_b = calculate_actor_visibility(inst_img_b, pos, pose_b, calib)

        boxes.append({
            "label":    "Car",
            "actor_id": actor.id,
            "x":        center.x,
            "y":        center.y,
            "z":        center.z,
            "l":        bb.extent.x * 2.0,
            "w":        bb.extent.y * 2.0,
            "h":        bb.extent.z * 2.0,
            "yaw":      t.rotation.yaw,
            "metrics_metadata": {
                "visible_pixels_vA": pixels_a,
                "visible_pixels_vB": pixels_b,
            },
        })
    return boxes


# ---------------------------------------------------------------------------
# Sensor setup
# ---------------------------------------------------------------------------

def spawn_stereo_suite(
    world: carla.World,
    vehicle: carla.Actor,
    side: str,
    image_w: int,
    image_h: int,
    fov: float,
    baseline_m: float,
) -> tuple:
    """Spawn RGB camera + optional depth and instance segmentation sensors.

    Left cameras get companion depth and instance segmentation sensors
    for GT disparity and visibility metadata. Right cameras get RGB only.

    Args:
        world: CARLA world.
        vehicle: Parent vehicle actor.
        side: 'left' or 'right'.
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        fov: Horizontal field of view in degrees.
        baseline_m: Stereo baseline in metres.

    Returns:
        Tuple (rgb_cam, rgb_q, depth_cam, depth_q, inst_cam, inst_q).
        depth_cam, depth_q, inst_cam, inst_q are None for side='right'.
    """
    bp_lib   = world.get_blueprint_library()
    y_offset = (baseline_m / 2.0) if side == "right" else -(baseline_m / 2.0)

    mount = carla.Transform(
        carla.Location(x=CAM_MOUNT_X, y=y_offset, z=CAM_MOUNT_Z),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    def make_cam(blueprint_id: str) -> tuple:
        bp = bp_lib.find(blueprint_id)
        bp.set_attribute("image_size_x", str(image_w))
        bp.set_attribute("image_size_y", str(image_h))
        bp.set_attribute("fov",          str(fov))
        bp.set_attribute("sensor_tick",  "0.0")
        cam = world.spawn_actor(bp, mount, attach_to=vehicle)
        q   = queue.Queue()
        cam.listen(q.put)
        return cam, q

    rgb_cam, rgb_q = make_cam("sensor.camera.rgb")

    if side == "right":
        return rgb_cam, rgb_q, None, None, None, None

    depth_cam, depth_q = make_cam("sensor.camera.depth")
    inst_cam,  inst_q  = make_cam("sensor.camera.instance_segmentation")

    return rgb_cam, rgb_q, depth_cam, depth_q, inst_cam, inst_q


# ---------------------------------------------------------------------------
# Main collection pipeline
# ---------------------------------------------------------------------------

def collect(
    output_dir: str,
    n_frames: int,
    host: str = "localhost",
    port: int  = 2000,
) -> None:
    """Run the full CARLA data collection pipeline.

    Args:
        output_dir: Root directory to save collected data.
        n_frames: Number of synchronized frame pairs to collect.
        host: CARLA server host.
        port: CARLA server port.
    """
    out = Path(output_dir)

    for vehicle in ("vehicle_a", "vehicle_b"):
        for folder in ("image_left", "image_right", "disp_noc_0"):
            (out / vehicle / folder).mkdir(parents=True, exist_ok=True)
    (out / "gt_boxes").mkdir(parents=True, exist_ok=True)

    client = carla.Client(host, port)
    client.set_timeout(60.0)

    logger.info("Loading Town10HD...")
    world = client.load_world("Town10HD")
    time.sleep(4.0)

    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    actors_to_destroy = []

    try:
        bp_lib       = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        sp_a = spawn_points[SPAWN_IDX_A]
        sp_b = spawn_points[SPAWN_IDX_B]

        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        vehicle_a = world.spawn_actor(vehicle_bp, sp_a)
        vehicle_b = world.spawn_actor(vehicle_bp, sp_b)
        vehicle_a.set_simulate_physics(True)
        vehicle_b.set_simulate_physics(True)
        vehicle_a.set_autopilot(True, 8000)
        vehicle_b.set_autopilot(True, 8000)
        actors_to_destroy.extend([vehicle_a, vehicle_b])

        logger.info(
            "Spawned Vehicle A (Horizontal, Idx %d) and "
            "Vehicle B (Vertical, Idx %d)",
            SPAWN_IDX_A, SPAWN_IDX_B,
        )

        # --- Spawn NPC vehicles near the intersection ---
        local_spawn_points = [
            sp for sp in spawn_points
            if sp.location.distance(JUNCTION_CENTER) <= TRAFFIC_SPAWN_RADIUS
            and sp.location.distance(sp_a.location) > 12.0
            and sp.location.distance(sp_b.location) > 12.0
        ]

        vehicle_blueprints = bp_lib.filter("vehicle.*")
        n_spawned = 0
        for i, sp in enumerate(local_spawn_points[:N_NPC_VEHICLES]):
            bp = vehicle_blueprints[i % len(vehicle_blueprints)]
            if int(bp.get_attribute("number_of_wheels")) == 4:
                npc = world.try_spawn_actor(bp, sp)
                if npc is not None:
                    npc.set_autopilot(True, 8000)
                    traffic_manager.ignore_lights_percentage(npc, 30)
                    traffic_manager.set_desired_speed(npc, 20)
                    actors_to_destroy.append(npc)
                    n_spawned += 1

        logger.info("Spawned %d NPC vehicles near the intersection.", n_spawned)

        # --- Attach sensor suites ---
        cam_a_left,  q_a_left,  cam_a_depth, q_a_depth, cam_a_inst, q_a_inst = \
            spawn_stereo_suite(world, vehicle_a, "left",  IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        cam_a_right, q_a_right, *_ = \
            spawn_stereo_suite(world, vehicle_a, "right", IMAGE_W, IMAGE_H, FOV, BASELINE_M)

        cam_b_left,  q_b_left,  cam_b_depth, q_b_depth, cam_b_inst, q_b_inst = \
            spawn_stereo_suite(world, vehicle_b, "left",  IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        cam_b_right, q_b_right, *_ = \
            spawn_stereo_suite(world, vehicle_b, "right", IMAGE_W, IMAGE_H, FOV, BASELINE_M)

        actors_to_destroy.extend([
            cam_a_left, cam_a_right, cam_a_depth, cam_a_inst,
            cam_b_left, cam_b_right, cam_b_depth, cam_b_inst,
        ])

        calib = build_calib(IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        for v_name in ("vehicle_a", "vehicle_b"):
            with open(out / v_name / "calib.json", "w") as f:
                json.dump(calib, f, indent=2)
        logger.info(
            "Saved calibration — focal=%.2fpx baseline=%.3fm",
            calib["focal_length_px"], calib["baseline_m"],
        )

        # Warmup — let physics settle and drain initial sensor frames
        logger.info("Warming up (40 ticks)...")
        all_queues = (
            q_a_left, q_a_right, q_a_depth, q_a_inst,
            q_b_left, q_b_right, q_b_depth, q_b_inst,
        )
        for _ in range(40):
            world.tick()
            for q in all_queues:
                try:
                    q.get(timeout=2.0)
                except queue.Empty:
                    pass

        # --- Main collection loop ---
        pose_a_all: dict = {}
        pose_b_all: dict = {}
        f_px = calib["focal_length_px"]

        logger.info("Collecting %d frames...", n_frames)
        for frame_idx in range(n_frames):
            world.tick()

            # Keep spectator overhead for monitoring
            spectator = world.get_spectator()
            spectator.set_transform(carla.Transform(
                carla.Location(
                    x=vehicle_a.get_transform().location.x,
                    y=vehicle_a.get_transform().location.y,
                    z=50.0,
                ),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            ))

            try:
                img_a_left  = q_a_left.get(timeout=5.0)
                img_a_right = q_a_right.get(timeout=5.0)
                raw_depth_a = q_a_depth.get(timeout=5.0)
                raw_inst_a  = q_a_inst.get(timeout=5.0)

                img_b_left  = q_b_left.get(timeout=5.0)
                img_b_right = q_b_right.get(timeout=5.0)
                raw_depth_b = q_b_depth.get(timeout=5.0)
                raw_inst_b  = q_b_inst.get(timeout=5.0)
            except queue.Empty:
                logger.warning("Frame timeout at index %d — skipping.", frame_idx)
                continue

            frame_str = f"{frame_idx:06d}"

            # Record poses before GT boxes (needed for visibility projection)
            pose_a_all[frame_str] = transform_to_dict(vehicle_a.get_transform())
            pose_b_all[frame_str] = transform_to_dict(vehicle_b.get_transform())

            # 1. Save RGB stereo pairs
            cv2.imwrite(
                str(out / "vehicle_a" / "image_left"  / f"{frame_str}.png"),
                carla_image_to_bgr(img_a_left),
            )
            cv2.imwrite(
                str(out / "vehicle_a" / "image_right" / f"{frame_str}.png"),
                carla_image_to_bgr(img_a_right),
            )
            cv2.imwrite(
                str(out / "vehicle_b" / "image_left"  / f"{frame_str}.png"),
                carla_image_to_bgr(img_b_left),
            )
            cv2.imwrite(
                str(out / "vehicle_b" / "image_right" / f"{frame_str}.png"),
                carla_image_to_bgr(img_b_right),
            )

            # 2. Convert depth to GT disparity and save
            depth_m_a    = carla_depth_to_meters(raw_depth_a)
            depth_m_b    = carla_depth_to_meters(raw_depth_b)
            disp_kitti_a = depth_meters_to_kitti_disparity(depth_m_a, f_px, BASELINE_M)
            disp_kitti_b = depth_meters_to_kitti_disparity(depth_m_b, f_px, BASELINE_M)

            np.save(
                str(out / "vehicle_a" / "disp_noc_0" / f"{frame_str}_disp.npy"),
                disp_kitti_a,
            )
            np.save(
                str(out / "vehicle_b" / "disp_noc_0" / f"{frame_str}_disp.npy"),
                disp_kitti_b,
            )
            cv2.imwrite(
                str(out / "vehicle_a" / "disp_noc_0" / f"{frame_str}_disp.png"),
                colorize_disparity(disp_kitti_a),
            )
            cv2.imwrite(
                str(out / "vehicle_b" / "disp_noc_0" / f"{frame_str}_disp.png"),
                colorize_disparity(disp_kitti_b),
            )

            # 3. Save GT boxes with semantic-tag-based visibility counts
            gt = get_gt_boxes_with_visibility(
                world,
                raw_inst_a,
                raw_inst_b,
                pose_a=pose_a_all[frame_str],
                pose_b=pose_b_all[frame_str],
                calib=calib,
            )
            with open(out / "gt_boxes" / f"{frame_str}.json", "w") as f:
                json.dump(gt, f, indent=2)

            if frame_idx % 20 == 0:
                logger.info("Frame %d/%d", frame_idx + 1, n_frames)

        # Save all poses in one file per vehicle
        with open(out / "vehicle_a" / "pose.json", "w") as f:
            json.dump(pose_a_all, f, indent=2)
        with open(out / "vehicle_b" / "pose.json", "w") as f:
            json.dump(pose_b_all, f, indent=2)

        logger.info("Collection complete — %d frames saved to %s", n_frames, out)

    finally:
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        logger.info("Destroying %d actors...", len(actors_to_destroy))
        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except Exception:
                pass
        logger.info("Teardown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect stereo V2V data from CARLA Town10HD"
    )
    parser.add_argument("--output", default="C:/carla_data",
                        help="Root output directory")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to collect")
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int, default=2000)
    args = parser.parse_args()

    collect(
        output_dir=args.output,
        n_frames=args.frames,
        host=args.host,
        port=args.port,
    )



