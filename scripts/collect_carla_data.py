"""CARLA Stereo V2V Data Collection Script.

Captures a realistic urban V2V sequence at a signalized intersection in
Town10HD_Opt (the layered map, with the ParkedVehicles layer unloaded) using two
moving ego vehicles with ambient NPC traffic:
- Vehicle A (Idx 148): Horizontal Street Approach
- Vehicle B (Idx 53):  Vertical Avenue Approach

Both vehicles follow realistic traffic rules including traffic light cycles.
NPC traffic clustered near the intersection provides realistic occlusion
scenarios for V2V cooperative perception evaluation. Parked vehicles are unloaded
(they are static map-layer meshes with no GT box — phantom vehicles in the images),
so every vehicle in the scene is a spawned actor with a ground-truth box.

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
import random
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
N_AHEAD_PER_EGO      = 6     # spawn points reserved directly ahead of each ego

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

# Visibility is occlusion-truthful: a GT car's visible-pixel count is the number
# of pixels inside its projected 2D box that are (a) tagged a vehicle class AND
# (b) at a depth consistent with the car's own distance. The depth gate rejects
# nearer occluders and far background that fall within the 2D box, and clipping
# the box to the image counts only the on-screen part — so partially visible
# cars (centre off-screen, body in frame) are kept rather than dropped. Tolerance
# = half the car's largest horizontal extent plus this margin (metres).
_VISIBILITY_DEPTH_MARGIN_M = 1.5

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

    CARLA encodes depth as a 24-bit value, least→most significant byte = R, G, B:
        normalized = (R + G*256 + B*256^2) / (256^3 - 1)
        depth_m    = normalized * 1000.0  (max range = 1000 m)

    The raw buffer is BGRA, so in the reshaped (H, W, 4) array index 0 = B,
    index 1 = G, index 2 = R (same layout the instance-seg reader relies on).
    The RED byte is least significant and the BLUE byte most significant —
    reading index 0 as R (and index 2 as B) swaps the high/low bytes and makes
    the decoded depth garbage.

    Args:
        carla_image: Raw depth image from sensor.camera.depth.

    Returns:
        Float32 depth array, shape (H, W), in metres.
    """
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))

    r = array[:, :, 2].astype(np.float32)  # RED   — least significant byte
    g = array[:, :, 1].astype(np.float32)  # GREEN
    b = array[:, :, 0].astype(np.float32)  # BLUE  — most significant byte

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
    depth_meters: np.ndarray,
    center_world: tuple,
    dims: tuple,
    world_yaw: float,
    pose: dict,
    calib: dict,
    cam_mount: tuple = (CAM_MOUNT_X, -BASELINE_M / 2.0, CAM_MOUNT_Z),
    depth_margin_m: float = _VISIBILITY_DEPTH_MARGIN_M,
) -> tuple[int, list | None]:
    """Measure an actor's visible pixels + tight 2D box (occlusion-truthful).

    Projects the actor's 3D box into the left camera, clips its 2D bounding box
    to the image, and selects pixels inside it that are both vehicle-tagged AND at
    a depth consistent with the actor's own distance. The depth-consistency gate
    rejects nearer occluders and far background that share the 2D box; clipping
    to the image keeps the on-screen portion of a partially visible car instead
    of dropping it (the old centre-patch test returned 0 whenever the projected
    centre fell outside the image).

    Returns both the count of those pixels (visibility) and their pixel-tight
    axis-aligned bounding box (KITTI-style 2D GT box — snug to the *visible* car,
    not the looser envelope of the 3D box's projected corners).

    The instance-segmentation raw buffer is BGRA: the semantic tag is in the RED
    channel (index 2). We match the 0.9.16 vehicle tags (_VEHICLE_SEMANTIC_TAGS).
    The camera world transform is the vehicle pose composed with the mount
    translation (the camera carries no rotation relative to the vehicle), so the
    projection matches where the actor appears in this camera's images — the same
    convention utils.carla_loader uses to place GT.

    Args:
        instance_image: Frame from sensor.camera.instance_segmentation.
        depth_meters: Left-camera depth (metres, planar Z), shape (H, W).
        center_world: (x, y, z) world position of the actor's box centre (m).
        dims: Actor box (l, w, h) in metres.
        world_yaw: Actor yaw in world frame (degrees).
        pose: Observing agent's vehicle world pose dict (x, y, z, roll, pitch,
            yaw).
        calib: Camera calibration dict with P2, image_w, image_h.
        cam_mount: Left-camera mount offset (x, y, z) in the vehicle frame.
        depth_margin_m: Slack added to half the largest horizontal extent when
            testing depth consistency.

    Returns:
        (visible_pixels, bbox_2d): the count of the actor's actually-visible
        pixels and their tight 2D box ``[x1, y1, x2, y2]`` (ints). Both are
        (0, None) when the actor is out of frame, behind the camera, or fully
        occluded.
    """
    h_img, w_img = instance_image.height, instance_image.width
    array = np.frombuffer(instance_image.raw_data, dtype=np.uint8)
    array = array.reshape((h_img, w_img, 4))

    # Semantic tag is in the RED channel (index 2) of the BGRA buffer.
    vehicle_mask = np.isin(array[:, :, 2], _VEHICLE_SEMANTIC_TAGS)

    # World → left-camera optical frame (vehicle rotation, camera position).
    t_world_cam       = _world_from_agent(pose).copy()
    t_world_cam[:, 3] = _world_from_agent(pose) @ np.array([*cam_mount, 1.0])
    t_cam_from_world  = np.linalg.inv(t_world_cam)
    p_cam = _LIDAR_TO_CAM @ t_cam_from_world @ np.array([*center_world, 1.0])
    z_box = float(p_cam[2])
    if z_box <= 0:  # behind the camera
        return 0, None

    # 3D box corners in the camera frame (KITTI convention: l along X, h along Y
    # down, w along Z), rotated by the actor's relative yaw about camera-Y.
    l, w, h = dims
    ry = math.radians(world_yaw - pose["yaw"])
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2])
    z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    cos_r, sin_r = math.cos(ry), math.sin(ry)
    R = np.array([[cos_r, 0.0, sin_r], [0.0, 1.0, 0.0], [-sin_r, 0.0, cos_r]])
    corners = R @ np.vstack([x_c, y_c, z_c])
    corners[0] += p_cam[0]
    corners[1] += p_cam[1]
    corners[2] += p_cam[2]
    corners[2] = np.maximum(corners[2], 0.1)

    P2 = np.asarray(calib["P2"], dtype=np.float64)
    homog = P2 @ np.vstack([corners, np.ones(8)])
    u = homog[0] / homog[2]
    v = homog[1] / homog[2]

    x1 = int(np.clip(math.floor(u.min()), 0, w_img))
    x2 = int(np.clip(math.ceil(u.max()),  0, w_img))
    y1 = int(np.clip(math.floor(v.min()), 0, h_img))
    y2 = int(np.clip(math.ceil(v.max()),  0, h_img))
    if x2 <= x1 or y2 <= y1:  # projects fully off-screen
        return 0, None

    # Vehicle pixels inside the 2D box whose depth matches the actor's distance.
    # tol spans the car's own near/far surfaces (half its largest horizontal
    # extent) plus a margin, so an occluder in front or background behind is
    # excluded but the whole car body counts.
    region_mask  = vehicle_mask[y1:y2, x1:x2]
    region_depth = depth_meters[y1:y2, x1:x2]
    tol = max(l, w) / 2.0 + depth_margin_m
    visible = np.logical_and(region_mask, depth_ok := np.abs(region_depth - z_box) <= tol)

    n_visible = int(visible.sum())
    if n_visible == 0:
        return 0, None

    # Pixel-tight 2D box = bounds of the visible pixels (offset back into full
    # image coords). Snug to the visible car, unlike the 3D-box-corner envelope.
    rows = np.where(visible.any(axis=1))[0]
    cols = np.where(visible.any(axis=0))[0]
    bbox = [x1 + int(cols[0]), y1 + int(rows[0]),
            x1 + int(cols[-1]) + 1, y1 + int(rows[-1]) + 1]
    return n_visible, bbox


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def spawn_points_ahead(
    ego_sp: carla.Transform,
    spawn_points: list,
    exclude_idx: set,
    max_dist_m: float = 60.0,
    max_lateral_m: float = 6.0,
) -> list:
    """Return spawn-point indices lying on the road directly ahead of an ego.

    A point counts as "ahead" when its offset from the ego projects positively
    onto the ego's forward vector (within ``max_dist_m``) and sits within
    ``max_lateral_m`` of the ego's heading line (so it is on the same lane /
    carriageway, not a side street). Results are nearest-first so the closest
    in-view traffic is spawned preferentially.

    Args:
        ego_sp: The ego vehicle's spawn transform.
        spawn_points: All map spawn points (index order preserved).
        exclude_idx: Indices to skip (e.g. too close to either ego).
        max_dist_m: Furthest ahead to consider.
        max_lateral_m: Max perpendicular distance from the ego's heading line.

    Returns:
        List of spawn-point indices, nearest ego first.
    """
    fwd = ego_sp.get_forward_vector()
    ahead = []
    for idx, sp in enumerate(spawn_points):
        if idx in exclude_idx:
            continue
        dx = sp.location.x - ego_sp.location.x
        dy = sp.location.y - ego_sp.location.y
        along   = dx * fwd.x + dy * fwd.y
        lateral = abs(dx * -fwd.y + dy * fwd.x)
        if 0.0 < along <= max_dist_m and lateral <= max_lateral_m:
            ahead.append((along, idx))
    ahead.sort()
    return [idx for _, idx in ahead]


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
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    pose_a: dict,
    pose_b: dict,
    calib: dict,
    ego_ids: tuple = (),
) -> list:
    """Get GT 3D boxes for all vehicles with per-agent visible-pixel counts.

    Visibility is occlusion- and partial-truthful: for each agent it counts the
    pixels inside the actor's projected 2D box that are vehicle-tagged and at a
    depth consistent with the actor's distance (see ``calculate_actor_visibility``).

    The two ego vehicles are flagged ``is_ego: True``. An ego is invisible to its
    own camera but visible to the other agent, so without this flag it survives
    the per-agent visibility filter via the other agent and pollutes the
    cooperative GT (one agent "detecting" the other's car). Downstream loaders
    drop ``is_ego`` boxes so the cooperative GT is real perceived objects only.

    Args:
        world: CARLA world object.
        inst_img_a: Instance segmentation frame from Vehicle A left camera.
        inst_img_b: Instance segmentation frame from Vehicle B left camera.
        depth_a: Vehicle A left-camera depth (metres), shape (H, W).
        depth_b: Vehicle B left-camera depth (metres), shape (H, W).
        pose_a: Vehicle A world pose dict.
        pose_b: Vehicle B world pose dict.
        calib: Shared camera calibration dict.
        ego_ids: Actor IDs of the two ego vehicles, flagged ``is_ego``.

    Returns:
        List of GT box dicts with label, actor_id, is_ego, x, y, z, l, w, h, yaw,
        and metrics_metadata.{visible_pixels_vA/vB, bbox_2d_vA/vB}. ``bbox_2d_v*``
        is the pixel-tight 2D box ``[x1,y1,x2,y2]`` of the car's visible pixels in
        that agent's view (null if not visible) — KITTI-style, snug to the car.
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
        dims = (bb.extent.x * 2.0, bb.extent.y * 2.0, bb.extent.z * 2.0)

        pixels_a, bbox_a = calculate_actor_visibility(
            inst_img_a, depth_a, pos, dims, t.rotation.yaw, pose_a, calib)
        pixels_b, bbox_b = calculate_actor_visibility(
            inst_img_b, depth_b, pos, dims, t.rotation.yaw, pose_b, calib)

        boxes.append({
            "label":    "Car",
            "actor_id": actor.id,
            "is_ego":   actor.id in ego_ids,
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
                "bbox_2d_vA":        bbox_a,
                "bbox_2d_vB":        bbox_b,
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

    # SPAWN_IDX_A / SPAWN_IDX_B were validated on the BASE Town10HD map. The
    # layered Town10HD_Opt (which we need below to unload parked vehicles) shares
    # the same road network but orders get_spawn_points() DIFFERENTLY, so the
    # same indices land in the wrong place there (e.g. B near idx 51, A near idx
    # 79). To stay pinned to the validated poses we read the canonical ego spawn
    # transforms from the base map here, then match them by world position on the
    # Opt map after loading it. (The road geometry is identical, so the poses are
    # valid on both.)
    logger.info("Reading canonical ego spawn poses from base Town10HD...")
    base_world = client.load_world("Town10HD")
    base_sps   = base_world.get_map().get_spawn_points()
    target_a   = base_sps[SPAWN_IDX_A]
    target_b   = base_sps[SPAWN_IDX_B]

    # Use the layered (_Opt) map and unload the ParkedVehicles layer. Parked cars
    # are static MAP-LAYER meshes, not spawned actors, so they render in the
    # RGB/segmentation/depth images but never get a GT box — phantom vehicles that
    # cannot be removed by destroying actors. Only the _Opt map allows toggling
    # layers; its road network / junction geometry is identical to Town10HD (only
    # the spawn-point ORDER differs, handled above). Result: the only vehicles in
    # the scene are the spawned ego + NPC actors, which all carry GT boxes.
    logger.info("Loading Town10HD_Opt and unloading parked vehicles...")
    world = client.load_world("Town10HD_Opt")
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    time.sleep(4.0)  # let the layer unload propagate (server still async here)

    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    # Deterministic NPC routing/lane choices so the scene (and both ego
    # trajectories) reproduce run-to-run. Without this the traffic manager picks
    # turns/speeds from an unseeded RNG, so each collection differs.
    traffic_manager.set_random_device_seed(42)

    actors_to_destroy = []

    try:
        bp_lib       = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        # Match the canonical base-map poses to the nearest Opt spawn point by
        # world position (not by index, which is reordered on the Opt map). This
        # pins A/B to the validated physical locations regardless of ordering.
        idx_a = min(range(len(spawn_points)),
                    key=lambda i: spawn_points[i].location.distance(target_a.location))
        idx_b = min(range(len(spawn_points)),
                    key=lambda i: spawn_points[i].location.distance(target_b.location))
        sp_a = spawn_points[idx_a]
        sp_b = spawn_points[idx_b]
        logger.info(
            "Matched canonical spawns -> Opt idx A=%d (%.2f m off), B=%d (%.2f m off)",
            idx_a, sp_a.location.distance(target_a.location),
            idx_b, sp_b.location.distance(target_b.location),
        )

        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        vehicle_a = world.spawn_actor(vehicle_bp, sp_a)
        vehicle_b = world.spawn_actor(vehicle_bp, sp_b)
        vehicle_a.set_simulate_physics(True)
        vehicle_b.set_simulate_physics(True)
        # Autopilot is enabled AFTER warmup (below) so frame 0 is exactly the
        # spawn pose — B at the validated idx-53 location (the intersection), A
        # at idx-148 — and the cars then drive on through the recording ("start
        # at 53 and keep going"). Driving during warmup instead would offset the
        # recorded start away from the spawn point.
        actors_to_destroy.extend([vehicle_a, vehicle_b])

        logger.info(
            "Spawned Vehicle A at (%.1f, %.1f) yaw %.0f | "
            "Vehicle B at (%.1f, %.1f) yaw %.0f",
            sp_a.location.x, sp_a.location.y, sp_a.rotation.yaw,
            sp_b.location.x, sp_b.location.y, sp_b.rotation.yaw,
        )

        # --- Spawn NPC vehicles: guarantee traffic ahead of BOTH egos ---
        # With the ParkedVehicles layer unloaded these NPCs are the ONLY cars in
        # the scene, so their placement alone decides how populated each approach
        # looks (the parked meshes used to fill the streets). Selecting purely by
        # distance to JUNCTION_CENTER packed every car around the intersection
        # and left Vehicle A's lane empty. Instead we first RESERVE spawn points
        # directly ahead of each ego (so both see leading/oncoming traffic from
        # frame 0), then fill the remainder from the junction-area pool. Indices
        # are used throughout so a point is never spawned twice.
        near_ego = {
            idx for idx, sp in enumerate(spawn_points)
            if sp.location.distance(sp_a.location) <= 12.0
            or sp.location.distance(sp_b.location) <= 12.0
        }
        ahead_a = spawn_points_ahead(sp_a, spawn_points, near_ego)
        ahead_b = spawn_points_ahead(sp_b, spawn_points, near_ego)

        junction_pool = [
            idx for idx, sp in enumerate(spawn_points)
            if idx not in near_ego
            and sp.location.distance(JUNCTION_CENTER) <= TRAFFIC_SPAWN_RADIUS
        ]
        random.shuffle(junction_pool)  # even spread; deterministic via seed(42)

        selected_idx: list = []
        seen: set = set()
        for idx in (ahead_a[:N_AHEAD_PER_EGO] + ahead_b[:N_AHEAD_PER_EGO]
                    + junction_pool):
            if len(selected_idx) >= N_NPC_VEHICLES:
                break
            if idx not in seen:
                selected_idx.append(idx)
                seen.add(idx)
        n_reserved = min(len(ahead_a), N_AHEAD_PER_EGO) + \
            min(len(ahead_b), N_AHEAD_PER_EGO)

        # Cars only — drop vans / trucks / buses / bikes (base_type == "car").
        car_blueprints = [
            bp for bp in bp_lib.filter("vehicle.*")
            if bp.has_attribute("base_type")
            and bp.get_attribute("base_type") == "car"
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]

        # Autopilot for NPCs is enabled after warmup with the egos, so frame 0 is
        # a clean snapshot (everyone at their spawn positions) and the whole
        # scene starts driving together from the first recorded frame.
        npc_actors = []
        for i, idx in enumerate(selected_idx):
            bp = car_blueprints[i % len(car_blueprints)]
            npc = world.try_spawn_actor(bp, spawn_points[idx])
            if npc is not None:
                npc_actors.append(npc)
                actors_to_destroy.append(npc)

        logger.info(
            "Spawned %d car NPCs (%d reserved ahead of egos, rest near junction).",
            len(npc_actors), n_reserved,
        )

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

        # Warmup — settle physics and drain initial sensor frames. All vehicles
        # stay parked (autopilot enabled afterwards), so frame 0 is exactly the
        # spawn poses.
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

        # Warmup done (frame 0 pinned to the spawn poses) — hand every vehicle to
        # the traffic manager so the whole scene starts driving from frame 1 on.
        vehicle_a.set_autopilot(True, 8000)
        vehicle_b.set_autopilot(True, 8000)
        for npc in npc_actors:
            npc.set_autopilot(True, 8000)
            traffic_manager.ignore_lights_percentage(npc, 30)
            traffic_manager.set_desired_speed(npc, 20)

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

            # 3. Save GT boxes with occlusion-truthful visibility counts
            gt = get_gt_boxes_with_visibility(
                world,
                raw_inst_a,
                raw_inst_b,
                depth_m_a,
                depth_m_b,
                pose_a=pose_a_all[frame_str],
                pose_b=pose_b_all[frame_str],
                calib=calib,
                ego_ids=(vehicle_a.id, vehicle_b.id),
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
        description="Collect stereo V2V data from CARLA Town10HD_Opt "
                    "(parked-vehicle layer unloaded)"
    )
    parser.add_argument("--output", default="C:/carla_data",
                        help="Root output directory")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to collect")
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int, default=2000)
    args = parser.parse_args()

    # Reproducible NPC placement (random.shuffle of spawn points) — paired with
    # traffic_manager.set_random_device_seed(42) inside collect() for routing.
    random.seed(42)
    np.random.seed(42)

    collect(
        output_dir=args.output,
        n_frames=args.frames,
        host=args.host,
        port=args.port,
    )



