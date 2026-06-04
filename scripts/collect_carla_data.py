"""CARLA Stereo V2V Data Collection Script.

Spawns two stationary vehicles in Town05 with synchronized stereo cameras.
Collects 100 frame pairs and saves them in a KITTI-compatible format for
direct use with the stereo V2V pipeline (Stages 1-4).

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
        └── {frame:06d}.json   GT 3D boxes in world coordinates

Usage (from stereo_v2v project root, carla_env activated):
    python scripts/collect_carla_data.py
    python scripts/collect_carla_data.py --frames 200 --output D:/my_data
"""

import argparse
import json
import logging
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

# Vehicle A — facing down the main street
VEHICLE_A_TRANSFORM = carla.Transform(
    carla.Location(x=10.0, y=0.0, z=0.5),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
)

# Vehicle B — at a 90° intersection, facing perpendicular
# Close enough for overlapping FOV, far enough for independent detections
VEHICLE_B_TRANSFORM = carla.Transform(
    carla.Location(x=40.0, y=-20.0, z=0.5),
    carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
)

# Stereo camera baseline — 0.54m matches KITTI for pipeline compatibility
BASELINE_M = 0.54

# Camera intrinsics — KITTI-compatible resolution and FOV
IMAGE_W    = 1242
IMAGE_H    = 375
FOV        = 90.0   # degrees

# Camera mount on vehicle — above hood, looking forward
CAM_OFFSET = carla.Transform(
    carla.Location(x=1.5, y=0.0, z=1.5),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def focal_length_px(image_w: int, fov_deg: float) -> float:
    """Compute focal length in pixels from image width and horizontal FOV.

    Args:
        image_w: Image width in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        Focal length in pixels.
    """
    import math
    return image_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))


def build_calib(image_w: int, image_h: int,
                fov_deg: float, baseline_m: float) -> dict:
    """Build a calibration dict compatible with utils/carla_loader.py.

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

    # P2 — left camera projection matrix (3x4), no translation
    P2 = [
        [f,   0.0, cx,  0.0],
        [0.0, f,   cy,  0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    # P3 — right camera projection matrix (3x4), baseline offset in x
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


def transform_to_dict(t: carla.Transform) -> dict:
    """Serialize a carla.Transform to a plain dict.

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
    """Convert a carla.Image to a BGR numpy array.

    Args:
        image: CARLA RGB image.

    Returns:
        BGR image, shape (H, W, 3), uint8.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    return array[:, :, :3]  # drop alpha → BGR


def get_gt_boxes(world: carla.World) -> list[dict]:
    """Get ground truth 3D bounding boxes for all vehicles in the world.

    Args:
        world: CARLA world object.

    Returns:
        List of dicts with label, x, y, z, l, w, h, yaw (world coordinates).
    """
    boxes = []
    for actor in world.get_actors().filter("vehicle.*"):
        t   = actor.get_transform()
        bb  = actor.bounding_box
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
    side: str,          # 'left' or 'right'
    image_w: int,
    image_h: int,
    fov: float,
    baseline_m: float,
) -> tuple[carla.Actor, queue.Queue]:
    """Spawn a single RGB camera attached to a vehicle.

    Args:
        world: CARLA world.
        vehicle: Parent vehicle actor.
        side: 'left' or 'right' — determines lateral offset.
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        fov: Horizontal field of view in degrees.
        baseline_m: Stereo baseline in metres (right cam offset).

    Returns:
        Tuple of (camera_actor, data_queue).
    """
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(image_w))
    bp.set_attribute("image_size_y", str(image_h))
    bp.set_attribute("fov",          str(fov))
    bp.set_attribute("sensor_tick",  "0.0")   # capture every tick

    # Right camera offset by baseline along vehicle Y axis
    y_offset = baseline_m / 2.0 if side == "right" else -baseline_m / 2.0
    transform = carla.Transform(
        carla.Location(
            x=CAM_OFFSET.location.x,
            y=y_offset,
            z=CAM_OFFSET.location.z,
        ),
        CAM_OFFSET.rotation,
    )

    cam   = world.spawn_actor(bp, transform, attach_to=vehicle)
    q     = queue.Queue()
    cam.listen(q.put)
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
    """Run the full data collection pipeline.

    Args:
        output_dir: Root directory to save collected data.
        n_frames: Number of synchronized frame pairs to collect.
        host: CARLA server host.
        port: CARLA server port.
    """
    out = Path(output_dir)

    # Create output directories
    for vehicle in ("vehicle_a", "vehicle_b"):
        for camera in ("image_left", "image_right"):
            (out / vehicle / camera).mkdir(parents=True, exist_ok=True)
    (out / "gt_boxes").mkdir(parents=True, exist_ok=True)

    client = carla.Client(host, port)
    client.set_timeout(30.0)

    logger.info("Loading Town05...")
    world = client.load_world("Town05")
    time.sleep(5.0)   # wait for world to fully load

    # Use synchronous mode — guarantees all sensors tick together
    settings                        = world.get_settings()
    settings.synchronous_mode       = True
    settings.fixed_delta_seconds    = 0.05   # 20 FPS
    world.apply_settings(settings)
    logger.info("Synchronous mode enabled at 20 FPS")

    actors_to_destroy = []

    try:
        # --- Spawn vehicles ---
        bp_lib     = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "vehicle_a")

        vehicle_a = world.spawn_actor(vehicle_bp, VEHICLE_A_TRANSFORM)
        vehicle_b = world.spawn_actor(vehicle_bp, VEHICLE_B_TRANSFORM)
        vehicle_a.set_autopilot(False)
        vehicle_b.set_autopilot(False)
        actors_to_destroy.extend([vehicle_a, vehicle_b])
        logger.info("Spawned Vehicle A at %s", VEHICLE_A_TRANSFORM.location)
        logger.info("Spawned Vehicle B at %s", VEHICLE_B_TRANSFORM.location)

        # --- Spawn stereo cameras ---
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

        # --- Save calibration (same for both vehicles — same camera config) ---
        calib = build_calib(IMAGE_W, IMAGE_H, FOV, BASELINE_M)
        for vehicle in ("vehicle_a", "vehicle_b"):
            with open(out / vehicle / "calib.json", "w") as f:
                json.dump(calib, f, indent=2)
        logger.info("Saved calibration — focal=%.2fpx baseline=%.3fm",
                    calib["focal_length_px"], calib["baseline_m"])

        # --- Warm up — tick a few times before collecting ---
        logger.info("Warming up...")
        for _ in range(10):
            world.tick()
            for q in (q_a_left, q_a_right, q_b_left, q_b_right):
                q.get(timeout=5.0)  # drain warmup frames

        # --- Collection loop ---
        pose_a_all = {}
        pose_b_all = {}

        logger.info("Collecting %d frames...", n_frames)
        for frame_idx in range(n_frames):
            world.tick()   # advance simulation one step

            # Retrieve synchronized images from all 4 cameras
            img_a_left  = q_a_left.get(timeout=5.0)
            img_a_right = q_a_right.get(timeout=5.0)
            img_b_left  = q_b_left.get(timeout=5.0)
            img_b_right = q_b_right.get(timeout=5.0)

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

            # Save GT boxes
            gt = get_gt_boxes(world)
            with open(out / "gt_boxes" / f"{frame_str}.json", "w") as f:
                json.dump(gt, f, indent=2)

            if frame_idx % 10 == 0:
                logger.info("Frame %d/%d", frame_idx + 1, n_frames)

        # Save all poses in one file per vehicle
        with open(out / "vehicle_a" / "pose.json", "w") as f:
            json.dump(pose_a_all, f, indent=2)
        with open(out / "vehicle_b" / "pose.json", "w") as f:
            json.dump(pose_b_all, f, indent=2)

        logger.info("Collection complete — %d frames saved to %s",
                    n_frames, out)

    finally:
        # Always restore async mode and destroy actors
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        logger.info("Destroying %d actors...", len(actors_to_destroy))
        for actor in actors_to_destroy:
            actor.destroy()
        logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Collect stereo V2V data from CARLA"
    )
    parser.add_argument("--output",  default="C:/carla_data",
                        help="Output directory")
    parser.add_argument("--frames",  type=int, default=100,
                        help="Number of frames to collect")
    parser.add_argument("--host",    default="localhost")
    parser.add_argument("--port",    type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(
        output_dir=args.output,
        n_frames=args.frames,
        host=args.host,
        port=args.port,
    )
