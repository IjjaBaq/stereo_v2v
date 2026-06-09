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
    │   ├── calib.json      — camera intrinsics + stereo baseline
    │   └── pose.json       — world transform per frame
    ├── vehicle_b/
    │   ├── image_left/
    │   ├── image_right/
    │   ├── calib.json
    │   └── pose.json
    └── gt_boxes/
        └── {frame:06d}.json  — GT 3D boxes for all vehicles in world coords

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

# Validated spawn point indices in Town10HD at the target intersection
SPAWN_IDX_A = 148  # Vehicle A: Horizontal Street Approach
SPAWN_IDX_B = 53   # Vehicle B: Vertical Avenue Approach

# NPC traffic — spawned within TRAFFIC_SPAWN_RADIUS of the junction center
# Junction center is the midpoint between Vehicle A and Vehicle B spawn points
JUNCTION_CENTER      = carla.Location(x=91.7, y=11.6, z=0.5)
TRAFFIC_SPAWN_RADIUS = 80.0   # metres
N_NPC_VEHICLES       = 40

# ---------------------------------------------------------------------------
# Stereo camera configuration (KITTI Tracking compatible)
# ---------------------------------------------------------------------------

BASELINE_M  = 0.54   # stereo baseline in metres — matches KITTI
IMAGE_W     = 1242   # image width in pixels
IMAGE_H     = 375    # image height in pixels
FOV         = 90.0   # horizontal field of view in degrees

# Camera mount position on vehicle — above hood, looking forward
CAM_MOUNT_X = 1.6    # metres forward from vehicle center
CAM_MOUNT_Z = 1.4    # metres above vehicle center


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

    Produces P2 (left camera) and P3 (right camera) projection matrices
    in the same format as KITTI calibration files.

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

    # P2 — left camera projection matrix (3x4), no lateral offset
    P2 = [
        [f,   0.0, cx,  0.0],
        [0.0, f,   cy,  0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    # P3 — right camera projection matrix (3x4), offset by baseline
    # P3[0,3] = -f * B encodes the stereo baseline in pixels
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
# Data helpers
# ---------------------------------------------------------------------------

def transform_to_dict(t: carla.Transform) -> dict:
    """Serialize a carla.Transform to a plain dict.

    Used to record vehicle world poses for Stage 4 coordinate transform.

    Args:
        t: CARLA transform (location + rotation).

    Returns:
        Dict with x, y, z, roll, pitch, yaw.
    """
    return {
        "x":     t.location.x,
        "y":     t.location.y,
        "z":     t.location.z,
        "roll":  t.rotation.roll,
        "pitch": t.rotation.pitch,
        "yaw":   t.rotation.yaw,
    }


def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    """Convert a carla.Image (BGRA) to a BGR numpy array.

    Args:
        image: CARLA RGB image (internally stored as BGRA).

    Returns:
        BGR image, shape (H, W, 3), uint8.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    return array[:, :, :3].copy()                          # drop alpha


def get_gt_boxes(world: carla.World) -> list:
    """Get ground truth 3D bounding boxes for all vehicles in the world.

    Returns boxes in CARLA world coordinates. carla_loader.py is
    responsible for projecting these into each vehicle's camera frame.

    Args:
        world: CARLA world object.

    Returns:
        List of dicts with label, actor_id, x, y, z, l, w, h, yaw.
    """
    boxes = []
    for actor in world.get_actors().filter("vehicle.*"):
        t  = actor.get_transform()
        bb = actor.bounding_box
        boxes.append({
            "label":    "Car",
            "actor_id": actor.id,
            "x":        t.location.x,
            "y":        t.location.y,
            "z":        t.location.z,
            "l":        bb.extent.x * 2.0,  # extent is half-size
            "w":        bb.extent.y * 2.0,
            "h":        bb.extent.z * 2.0,
            "yaw":      t.rotation.yaw,
        })
    return boxes


# ---------------------------------------------------------------------------
# Sensor setup
# ---------------------------------------------------------------------------

def spawn_stereo_camera(
    world: carla.World,
    vehicle: carla.Actor,
    side: str,
    image_w: int,
    image_h: int,
    fov: float,
    baseline_m: float,
) -> tuple:
    """Spawn a single RGB camera attached to a vehicle.

    Left and right cameras are offset symmetrically by half the baseline
    along the vehicle's Y axis.

    Args:
        world: CARLA world.
        vehicle: Parent vehicle actor to attach camera to.
        side: 'left' or 'right' — determines lateral offset direction.
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        fov: Horizontal field of view in degrees.
        baseline_m: Stereo baseline in metres.

    Returns:
        Tuple of (camera_actor, data_queue).
    """
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(image_w))
    bp.set_attribute("image_size_y", str(image_h))
    bp.set_attribute("fov",          str(fov))
    bp.set_attribute("sensor_tick",  "0.0")  # capture every simulation tick

    # Right camera offset positively, left camera negatively along Y
    y_offset = (baseline_m / 2.0) if side == "right" else -(baseline_m / 2.0)

    local_transform = carla.Transform(
        carla.Location(x=CAM_MOUNT_X, y=y_offset, z=CAM_MOUNT_Z),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(bp, local_transform, attach_to=vehicle)
    q   = queue.Queue()
    cam.listen(q.put)
    return cam, q


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

    # Create output directory structure
    for vehicle in ("vehicle_a", "vehicle_b"):
        for camera in ("image_left", "image_right"):
            (out / vehicle / camera).mkdir(parents=True, exist_ok=True)
    (out / "gt_boxes").mkdir(parents=True, exist_ok=True)

    client = carla.Client(host, port)
    client.set_timeout(60.0)

    logger.info("Loading Town10HD...")
    world = client.load_world("Town10HD")
    time.sleep(4.0)

    # Synchronous mode — guarantees all sensors tick together
    # every world.tick() advances simulation by fixed_delta_seconds
    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # Traffic manager must also be in sync mode
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    actors_to_destroy = []

    try:
        bp_lib       = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        sp_a = spawn_points[SPAWN_IDX_A]
        sp_b = spawn_points[SPAWN_IDX_B]

        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        # Spawn ego vehicles
        vehicle_a = world.spawn_actor(vehicle_bp, sp_a)
        vehicle_b = world.spawn_actor(vehicle_bp, sp_b)

        vehicle_a.set_simulate_physics(True)
        vehicle_b.set_simulate_physics(True)

        # Autopilot — vehicles follow traffic rules including traffic lights
        vehicle_a.set_autopilot(True, 8000)
        vehicle_b.set_autopilot(True, 8000)

        actors_to_destroy.extend([vehicle_a, vehicle_b])
        logger.info(
            "Spawned Vehicle A (Horizontal, Idx %d) and "
            "Vehicle B (Vertical, Idx %d)",
            SPAWN_IDX_A, SPAWN_IDX_B,
        )

        # --- Spawn NPC vehicles near the intersection ---
        # Clustered within TRAFFIC_SPAWN_RADIUS of the junction center
        # to ensure occlusion scenarios are relevant to both ego vehicles.
        # Minimum distance from ego vehicles avoids NPCs spawning directly
        # in front of A or B and blocking their view from frame 0.
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
                    # Keep NPCs moving through lights so they reach
                    # the intersection during the collection window
                    traffic_manager.ignore_lights_percentage(npc, 30)
                    traffic_manager.set_desired_speed(npc, 20)
                    actors_to_destroy.append(npc)
                    n_spawned += 1

        logger.info("Spawned %d NPC vehicles near the intersection.", n_spawned)

        # --- Attach stereo cameras to both vehicles ---
        cam_a_left,  q_a_left  = spawn_stereo_camera(
            world, vehicle_a, "left",  IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        cam_a_right, q_a_right = spawn_stereo_camera(
            world, vehicle_a, "right", IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        cam_b_left,  q_b_left  = spawn_stereo_camera(
            world, vehicle_b, "left",  IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        cam_b_right, q_b_right = spawn_stereo_camera(
            world, vehicle_b, "right", IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        actors_to_destroy.extend(
            [cam_a_left, cam_a_right, cam_b_left, cam_b_right]
        )

        # Save calibration — same intrinsics for both vehicles
        calib = build_calib(IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        for v_name in ("vehicle_a", "vehicle_b"):
            with open(out / v_name / "calib.json", "w") as f:
                json.dump(calib, f, indent=2)
        logger.info(
            "Saved calibration — focal=%.2fpx baseline=%.3fm",
            calib["focal_length_px"], calib["baseline_m"],
        )

        # Warmup — let physics settle and drain initial sensor frames
        logger.info(
            "Warming up physics and draining initial sensor frames "
            "(40 ticks)..."
        )
        for _ in range(40):
            world.tick()
            for q in (q_a_left, q_a_right, q_b_left, q_b_right):
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass

        # --- Main collection loop ---
        pose_a_all = {}
        pose_b_all = {}

        logger.info("Collecting %d frames...", n_frames)
        for frame_idx in range(n_frames):
            world.tick()

            # Move spectator to bird's eye view above Vehicle A
            spectator = world.get_spectator()
            spectator.set_transform(carla.Transform(
                carla.Location(
                    x=vehicle_a.get_transform().location.x,
                    y=vehicle_a.get_transform().location.y,
                    z=50.0,
                ),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            ))

            # Retrieve synchronized images from all 4 cameras
            try:
                img_a_left  = q_a_left.get(timeout=5.0)
                img_a_right = q_a_right.get(timeout=5.0)
                img_b_left  = q_b_left.get(timeout=5.0)
                img_b_right = q_b_right.get(timeout=5.0)
            except queue.Empty:
                logger.warning(
                    "Frame timeout at index %d — skipping.", frame_idx
                )
                continue

            frame_str = f"{frame_idx:06d}"

            # Save synchronized stereo pairs for each vehicle
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

            # Record vehicle world poses for Stage 4 coordinate transform
            pose_a_all[frame_str] = transform_to_dict(
                vehicle_a.get_transform()
            )
            pose_b_all[frame_str] = transform_to_dict(
                vehicle_b.get_transform()
            )

            # Record GT 3D boxes for all vehicles in world coordinates
            # Note: carla_loader.py filters these to each vehicle's FOV
            gt = get_gt_boxes(world)
            with open(out / "gt_boxes" / f"{frame_str}.json", "w") as f:
                json.dump(gt, f, indent=2)

            if frame_idx % 20 == 0:
                logger.info("Frame %d/%d", frame_idx + 1, n_frames)

        # Save all poses in one file per vehicle
        with open(out / "vehicle_a" / "pose.json", "w") as f:
            json.dump(pose_a_all, f, indent=2)
        with open(out / "vehicle_b" / "pose.json", "w") as f:
            json.dump(pose_b_all, f, indent=2)

        logger.info(
            "Collection complete — %d frames saved to %s", n_frames, out
        )

    finally:
        # Always restore async mode and destroy all spawned actors
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





