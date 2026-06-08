"""CARLA Stereo V2V Data Collection Script.

Spawns two stationary vehicles in Town05 with synchronized stereo cameras.
Collects 100 frame pairs and saves them in a KITTI-compatible format for
direct use with the stereo V2V pipeline (Stages 1-4).

Scenario:
    Vehicle A (spawn point 1): x=-44.2, y=-39.6, yaw=-90.3 (facing south)
    Vehicle B (spawn point 42): x=-50.7, y=53.9,  yaw=90.4  (facing north)
    Both on the same street, ~93m apart, facing each other.
    20 NPC vehicles spawned between them as detection targets.

Output structure:
    C:/carla_data/
    ├── vehicle_a/
    │   ├── image_left/     {frame:06d}.png
    │   ├── image_right/    {frame:06d}.png
    │   ├── calib.json      camera intrinsics + baseline
    │   └── pose.json       world transform per frame
    ├── vehicle_b/
    │   ├── image_left/
    │   ├── image_right/
    │   ├── calib.json
    │   └── pose.json
    └── gt_boxes/
        └── {frame:06d}.json   GT 3D boxes for all vehicles in world coords

Usage (carla_env activated):
    python scripts/collect_carla_data.py
    python scripts/collect_carla_data.py --frames 200 --output D:/my_data
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
# Scene configuration
# ---------------------------------------------------------------------------

# Spawn point indices from Town05 — confirmed valid road positions
# Vehicle A: x=-44.2, y=-39.6, yaw=-90.3 (facing south)
# Vehicle B: x=-50.7, y= 53.9, yaw= 90.4 (facing north)
# Same street, ~93m apart, facing each other
VEHICLE_A_SPAWN_IDX = 1
VEHICLE_B_SPAWN_IDX = 42

# NPC traffic — spawned between A and B as detection targets
# Use spawn points 60-79 to avoid overlapping with A and B
NPC_SPAWN_START_IDX = 60
N_NPC_VEHICLES      = 20

# Stereo camera baseline — 0.54m matches KITTI for pipeline compatibility
BASELINE_M = 0.54

# Camera intrinsics — KITTI-compatible resolution
IMAGE_W = 1242
IMAGE_H = 375
FOV     = 90.0  # degrees — horizontal field of view

# Camera mount on vehicle — above hood, looking forward
CAM_MOUNT_X = 1.5   # metres forward from vehicle center
CAM_MOUNT_Z = 1.5   # metres above vehicle center


# ---------------------------------------------------------------------------
# Calibration helpers
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

    Matches the format expected by extract_stereo_params() in utils/calib.py:
    requires P2 (left camera) and P3 (right camera) as 3x4 matrices.

    Args:
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        fov_deg: Horizontal FOV in degrees.
        baseline_m: Stereo baseline in metres.

    Returns:
        Calibration dict with focal_length_px, cx, cy, baseline_m, P2, P3.
    """
    f  = focal_length_px(image_w, fov_deg)
    cx = image_w / 2.0
    cy = image_h / 2.0

    # P2 — left camera projection matrix (3x4)
    P2 = [
        [f,   0.0, cx,  0.0],
        [0.0, f,   cy,  0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    # P3 — right camera projection matrix (3x4)
    # tx = -f * baseline encodes the stereo offset
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

    Args:
        t: CARLA transform (location + rotation).

    Returns:
        Dict with x, y, z, roll, pitch, yaw (degrees).
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
        image: CARLA RGB camera image.

    Returns:
        BGR image, shape (H, W, 3), uint8.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    return array[:, :, :3].copy()                          # BGR, drop alpha


def get_gt_boxes(world: carla.World) -> list:
    """Get GT 3D bounding boxes for all vehicles in the world.

    Boxes are in CARLA world coordinates (left-handed, metres).
    carla_loader.py is responsible for transforming them into
    each vehicle's camera frame.

    Args:
        world: CARLA world object.

    Returns:
        List of dicts with keys:
            label, actor_id, x, y, z, l, w, h, yaw.
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
            "l":        bb.extent.x * 2.0,
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

    Left camera is offset -baseline/2 along Y, right camera +baseline/2.
    CARLA vehicle Y axis points right (from driver perspective).

    Args:
        world: CARLA world.
        vehicle: Parent vehicle actor.
        side: 'left' or 'right'.
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        fov: Horizontal FOV in degrees.
        baseline_m: Stereo baseline in metres.

    Returns:
        Tuple of (camera_actor, data_queue).
    """
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(image_w))
    bp.set_attribute("image_size_y", str(image_h))
    bp.set_attribute("fov",          str(fov))
    bp.set_attribute("sensor_tick",  "0.0")  # capture every simulation tick

    # Y offset: left camera at -baseline/2, right camera at +baseline/2
    y_offset = (baseline_m / 2.0) if side == "right" else -(baseline_m / 2.0)

    transform = carla.Transform(
        carla.Location(x=CAM_MOUNT_X, y=y_offset, z=CAM_MOUNT_Z),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    cam = world.spawn_actor(bp, transform, attach_to=vehicle)
    q   = queue.Queue()
    cam.listen(q.put)

    logger.debug("Spawned %s camera for %s", side, vehicle.type_id)
    return cam, q


# ---------------------------------------------------------------------------
# Main collection loop
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

    logger.info("Loading Town05...")
    world = client.load_world("Town05")
    time.sleep(8.0)  # wait for world to fully initialize

    # Synchronous mode — guarantees all 4 cameras tick together
    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    logger.info("Synchronous mode enabled — 20 FPS")

    # Traffic manager also needs synchronous mode
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    actors_to_destroy = []

    try:
        bp_lib      = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        logger.info("Available spawn points: %d", len(spawn_points))

        # --- Spawn ego vehicles (stationary) ---
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        vehicle_a = world.spawn_actor(
            vehicle_bp, spawn_points[VEHICLE_A_SPAWN_IDX]
        )
        vehicle_b = world.spawn_actor(
            vehicle_bp, spawn_points[VEHICLE_B_SPAWN_IDX]
        )
        vehicle_a.set_autopilot(False)
        vehicle_b.set_autopilot(False)
        actors_to_destroy.extend([vehicle_a, vehicle_b])

        logger.info(
            "Vehicle A spawned — x=%.1f y=%.1f yaw=%.1f",
            vehicle_a.get_transform().location.x,
            vehicle_a.get_transform().location.y,
            vehicle_a.get_transform().rotation.yaw,
        )
        logger.info(
            "Vehicle B spawned — x=%.1f y=%.1f yaw=%.1f",
            vehicle_b.get_transform().location.x,
            vehicle_b.get_transform().location.y,
            vehicle_b.get_transform().rotation.yaw,
        )

        # --- Spawn NPC traffic between A and B ---
        vehicle_blueprints = bp_lib.filter("vehicle.*")
        n_spawned = 0
        for i in range(N_NPC_VEHICLES):
            sp_idx = (NPC_SPAWN_START_IDX + i) % len(spawn_points)
            # Skip spawn points too close to A or B
            if sp_idx in (VEHICLE_A_SPAWN_IDX, VEHICLE_B_SPAWN_IDX):
                continue
            bp  = vehicle_blueprints[i % len(vehicle_blueprints)]
            npc = world.try_spawn_actor(bp, spawn_points[sp_idx])
            if npc is not None:
                npc.set_autopilot(True, 8000)
                actors_to_destroy.append(npc)
                n_spawned += 1

        logger.info("Spawned %d NPC vehicles", n_spawned)

        # --- Spawn stereo cameras on both vehicles ---
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
        logger.info("Stereo cameras spawned on both vehicles")

        # --- Save calibration ---
        calib = build_calib(IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        for v_name in ("vehicle_a", "vehicle_b"):
            with open(out / v_name / "calib.json", "w") as f:
                json.dump(calib, f, indent=2)
        logger.info(
            "Calibration saved — focal=%.2fpx baseline=%.3fm",
            calib["focal_length_px"], calib["baseline_m"],
        )

        # --- Warm up — let NPC traffic settle before collecting ---
        logger.info("Warming up — letting traffic settle (30 ticks)...")
        for _ in range(30):
            world.tick()
            # Drain all camera queues during warmup
            for q in (q_a_left, q_a_right, q_b_left, q_b_right):
                try:
                    q.get(timeout=2.0)
                except queue.Empty:
                    pass

        # --- Collection loop ---
        pose_a_all: dict = {}
        pose_b_all: dict = {}

        logger.info("Starting collection — %d frames...", n_frames)
        for frame_idx in range(n_frames):
            world.tick()

            # Retrieve synchronized images from all 4 cameras
            try:
                img_a_left  = q_a_left.get(timeout=5.0)
                img_a_right = q_a_right.get(timeout=5.0)
                img_b_left  = q_b_left.get(timeout=5.0)
                img_b_right = q_b_right.get(timeout=5.0)
            except queue.Empty:
                logger.warning(
                    "Frame %d: camera timeout — skipping", frame_idx
                )
                continue

            frame_str = f"{frame_idx:06d}"

            # Save images
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

            # Save vehicle poses
            pose_a_all[frame_str] = transform_to_dict(
                vehicle_a.get_transform()
            )
            pose_b_all[frame_str] = transform_to_dict(
                vehicle_b.get_transform()
            )

            # Save GT boxes for this frame
            gt = get_gt_boxes(world)
            with open(out / "gt_boxes" / f"{frame_str}.json", "w") as f:
                json.dump(gt, f, indent=2)

            if frame_idx % 10 == 0:
                logger.info(
                    "  Frame %d/%d — %d GT boxes",
                    frame_idx + 1, n_frames, len(gt),
                )

        # Save all poses as single files
        with open(out / "vehicle_a" / "pose.json", "w") as f:
            json.dump(pose_a_all, f, indent=2)
        with open(out / "vehicle_b" / "pose.json", "w") as f:
            json.dump(pose_b_all, f, indent=2)

        logger.info(
            "Collection complete — %d frames saved to %s",
            n_frames, out,
        )

    finally:
        # Always restore async mode and clean up actors
        logger.info("Cleaning up...")
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except Exception:
                pass
        logger.info("Destroyed %d actors. Done.", len(actors_to_destroy))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Collect stereo V2V data from CARLA Town05"
    )
    parser.add_argument(
        "--output", default="C:/carla_data",
        help="Root output directory (default: C:/carla_data)",
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of synchronized frame pairs to collect (default: 100)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(
        output_dir=args.output,
        n_frames=args.frames,
        host=args.host,
        port=args.port,
    )
