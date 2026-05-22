"""
Track robot torso_link and box poses in real-time (single-camera shadow logger).

Tag layout (current robot setup):
  - Head tag (default id=9): mounted on the head, ~25 cm above torso_link
      → torso_pos_head = head_tag_pos + world_z * head_z_offset
  - Pelvis tags (default ids=8,7): mounted slightly below pelvis, ~10 cm below root
      → root_pos = pelvis_tag_pos + world_z * pelvis_to_root_z
      → torso_pos_pelvis = root_pos + world_z * root_to_torso_z
  - Box tags: ids loaded from --box-tag-map (does NOT include floor origin tag).

Why two paths?
  Phase A (shadow mode) of apriltag2obs.md: log BOTH torso candidates so we can
  compare them offline against the npz reference and decide a fusion weight.

Outputs:
  - GUI overlay (per-tag axes, computed torso/box, fused if both available).
  - Optional CSV (--csv-out) with one row per frame, all candidates included.
"""
import argparse
import csv
import json
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from utils.apriltag_config import (
    parse_tag_size_map,
    load_tag_size_config,
    merge_tag_sizes,
    detect_with_tag_sizes,
)

# =========================================================
# Config
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--cam-serial", type=str, default="115222071236")
parser.add_argument("--cam-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--box-tag-map", type=str, default="box_tag_map.npz")
parser.add_argument("--head-tag-calib", type=str, default="T_tag_torso.npz",
                    help="Optional: precise T_tag_torso npz. If missing, falls back to --head-z-offset")
parser.add_argument("--head-tag-id", type=int, default=9)
parser.add_argument("--pelvis-tag-ids", type=str, default="8,7",
                    help="Comma-separated pelvis tag ids, e.g. '8,7'")
parser.add_argument("--head-z-offset", type=float, default=-0.25,
                    help="World-z offset from head tag to torso_link [m] (fallback when no T_tag_torso.npz)")
parser.add_argument("--pelvis-to-root-z", type=float, default=0.10,
                    help="World-z offset from pelvis tag to root (pelvis link) [m]")
parser.add_argument("--root-to-torso-z", type=float, default=0.20,
                    help="World-z offset from root to torso_link [m]")
parser.add_argument("--torso-source", type=str, default="fused",
                    choices=["head", "pelvis", "fused"],
                    help="Which torso estimate to display in GUI (CSV always logs all paths)")
parser.add_argument("--csv-out", type=str, default="",
                    help="If set, write per-frame shadow log to this CSV path")
parser.add_argument("--print-every", type=int, default=30,
                    help="Print obs line every N frames (0=never)")
parser.add_argument("--tag-size", type=float, default=0.077, help="Default AprilTag size in meters")
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--origin-id", type=int, default=-1,
                    help="Floor origin tag id (e.g. 1). When set, all coords are reported in origin frame. -1 disables.")
parser.add_argument("--anchor-config", type=str, default="config/floor_anchor_transforms_cam2.json",
                    help="JSON file with T_origin_anchor entries for fallback anchors")
parser.add_argument("--fallback-anchor-ids", type=str, default="10",
                    help="Comma-separated fallback anchor ids when origin tag is hidden, e.g. '10'")
args = parser.parse_args()

CAM_SERIAL = args.cam_serial
CAM_CALIB = args.cam_calib
BOX_TAG_MAP_FILE = args.box_tag_map
HEAD_TAG_CALIB_FILE = args.head_tag_calib

ROBOT_HEAD_TAG_ID = args.head_tag_id
PELVIS_TAG_IDS = [int(x) for x in args.pelvis_tag_ids.split(",") if x.strip()]

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

# =========================================================
# Utils
# =========================================================

def load_camera_params(path):
    data = np.load(path)
    K = data["camera_matrix"]
    return K, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]


def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def average_rotation(R_list):
    R_avg = np.mean(np.stack(R_list, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def rotation_to_euler(R):
    """Extract roll, pitch, yaw from rotation matrix (XYZ convention)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def rotation_to_quat_wxyz(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    q = np.empty(4, dtype=float)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    n = np.linalg.norm(q)
    if n > 1e-9:
        q /= n
    return q


def world_z_translate(T_world_X, dz):
    """Return new pose with same rotation but translated by dz along WORLD z-axis."""
    out = T_world_X.copy()
    out[2, 3] += dz
    return out


def load_anchor_transforms(path: str, origin_id: int):
    """Load fallback anchors keyed by tag id from a floor_anchor_transforms*.json."""
    if not path or not os.path.exists(path):
        return {}
    data = json.loads(open(path).read())
    cfg_origin = data.get("origin_id", origin_id)
    if int(cfg_origin) != int(origin_id):
        print(f"[origin] WARN anchor config origin_id={cfg_origin} != --origin-id={origin_id}")
    out = {}
    for raw_id, meta in (data.get("anchors") or {}).items():
        if "T_origin_anchor" in meta:
            out[int(raw_id)] = np.asarray(meta["T_origin_anchor"], dtype=float)
    return out


def draw_detection(img, det, color, label=""):
    corners = det.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, color, 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    text = label if label else f"ID:{det.tag_id}"
    cv2.putText(img, text, (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_axes(img, K, T_cam_frame, length=0.1):
    dist_coeffs = np.zeros(5)
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3]
    points_3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    pts = points_2d.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 2)
    return origin

# =========================================================
# Load calibration data
# =========================================================

K, cam_params = load_camera_params(CAM_CALIB)

# Box tag map (ids come from npz, never collide with floor origin tag).
map_data = np.load(BOX_TAG_MAP_FILE)
BOX_T_TAG = {}
for tag_id, T in zip(map_data["tag_ids"], map_data["box_T_tags"]):
    BOX_T_TAG[int(tag_id)] = T
# Defensive: never let floor origin / anchor ids be used as box-face candidates.
# This protects against a stale box_tag_map that picked up a floor tag during registration.
_origin_excludes = set()
if int(args.origin_id) >= 0:
    _origin_excludes.add(int(args.origin_id))
_origin_excludes |= {int(x) for x in args.fallback_anchor_ids.split(",") if x.strip()}
_dropped = sorted(_origin_excludes & set(BOX_T_TAG.keys()))
for _tid in _dropped:
    BOX_T_TAG.pop(_tid, None)
if _dropped:
    print(f"[box-map] dropped ids {_dropped} (origin/anchor) from box candidates "
          f"to avoid identity collision; check box_tag_map.npz.")
BOX_TAG_IDS = sorted(BOX_T_TAG.keys())

# Head tag → torso calibration (optional in shadow mode).
T_tag_torso = None
if HEAD_TAG_CALIB_FILE and os.path.exists(HEAD_TAG_CALIB_FILE):
    head_calib = np.load(HEAD_TAG_CALIB_FILE)
    T_tag_torso = head_calib["T_tag_torso"]
    print(f"Loaded T_tag_torso (translation: {T_tag_torso[:3, 3]})")
else:
    print(f"[shadow] {HEAD_TAG_CALIB_FILE} not found; "
          f"falling back to head_z_offset={args.head_z_offset:+.3f} m for head→torso.")

# Origin / floor anchor (optional). When --origin-id >= 0, all coords are reported
# in the origin tag's local frame instead of the camera optical frame.
ORIGIN_ID = int(args.origin_id)
ORIGIN_ENABLED = ORIGIN_ID >= 0
FALLBACK_ANCHOR_IDS = [int(x) for x in args.fallback_anchor_ids.split(",") if x.strip()]
ANCHOR_MAP = load_anchor_transforms(args.anchor_config, ORIGIN_ID) if ORIGIN_ENABLED else {}
FALLBACK_ANCHOR_SET = set(FALLBACK_ANCHOR_IDS)
T_cam_origin_cache = None  # latched once origin (or anchor) seen, kept under static-cam assumption
if ORIGIN_ENABLED:
    print(f"[origin] enabled origin_id={ORIGIN_ID}, fallback_anchors={FALLBACK_ANCHOR_IDS}, "
          f"anchors_loaded={sorted(ANCHOR_MAP.keys())}")
else:
    print("[origin] disabled (camera optical frame is world). "
          "Pass --origin-id 1 to use floor origin frame.")

# =========================================================
# Detector + cameras
# =========================================================

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(CAM_SERIAL)
config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline.start(config)

print("\n======================================")
print("Tracking: Robot torso + Box (shadow logger)")
print("Single camera = WORLD frame")
print(f"stream={args.width}x{args.height}@{args.fps}")
print(f"head_tag_id={ROBOT_HEAD_TAG_ID}, pelvis_tag_ids={PELVIS_TAG_IDS}")
print(f"head_z_offset={args.head_z_offset:+.3f} m  (used when T_tag_torso unavailable)")
print(f"pelvis_to_root_z={args.pelvis_to_root_z:+.3f} m, root_to_torso_z={args.root_to_torso_z:+.3f} m")
print(f"box_tag_ids={BOX_TAG_IDS}")
print(f"torso_source(GUI)={args.torso_source}, csv_out={args.csv_out or '(off)'}")
print(f"tag_config={args.tag_config}, default_tag_size={TAG_SIZE}, tag_size_map={TAG_SIZE_MAP}")
print("Press ESC to quit")
print("======================================\n")

# =========================================================
# CSV logger
# =========================================================

CSV_COLUMNS = [
    "frame_idx", "t_sec",
    # Raw tag world poses (translations only for compactness)
    "head_visible", "head_pos_x", "head_pos_y", "head_pos_z",
    "head_quat_w", "head_quat_x", "head_quat_y", "head_quat_z", "head_margin",
    "pelvis_visible", "pelvis_used_id",
    "pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z",
    "pelvis_quat_w", "pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z",
    "pelvis_margin_max",
    # Derived poses (root and 3 torso candidates)
    "root_pos_x", "root_pos_y", "root_pos_z",
    "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "torso_head_pos_x", "torso_head_pos_y", "torso_head_pos_z",
    "torso_head_quat_w", "torso_head_quat_x", "torso_head_quat_y", "torso_head_quat_z",
    "torso_pelvis_pos_x", "torso_pelvis_pos_y", "torso_pelvis_pos_z",
    "torso_pelvis_quat_w", "torso_pelvis_quat_x", "torso_pelvis_quat_y", "torso_pelvis_quat_z",
    "torso_pos_x", "torso_pos_y", "torso_pos_z",
    "torso_quat_w", "torso_quat_x", "torso_quat_y", "torso_quat_z",
    "torso_path",
    # Box
    "box_visible", "box_n_tags",
    "obj_pos_x", "obj_pos_y", "obj_pos_z",
    "obj_quat_w", "obj_quat_x", "obj_quat_y", "obj_quat_z",
]

csv_writer = None
csv_file = None
if args.csv_out:
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    csv_file = open(args.csv_out, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()
    print(f"[shadow] CSV logging -> {args.csv_out}")


def csv_row(values):
    row = {k: "" for k in CSV_COLUMNS}
    row.update(values)
    return row


def fill_pose(prefix, T):
    if T is None:
        return {}
    p = T[:3, 3]
    q = rotation_to_quat_wxyz(T[:3, :3])
    return {
        f"{prefix}_pos_x": float(p[0]),
        f"{prefix}_pos_y": float(p[1]),
        f"{prefix}_pos_z": float(p[2]),
        f"{prefix}_quat_w": float(q[0]),
        f"{prefix}_quat_x": float(q[1]),
        f"{prefix}_quat_y": float(q[2]),
        f"{prefix}_quat_z": float(q[3]),
    }

# =========================================================
# Main loop
# =========================================================

t_start = time.time()
frame_idx = 0

try:
    while True:
        frame_idx += 1
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detect_with_tag_sizes(detector, gray, cam_params, TAG_SIZE, TAG_SIZE_MAP)
        det_by_id = {det.tag_id: det for det in dets}

        # ------ Determine T_cam_origin (= camera-frame pose of the origin frame) ------
        # Priority: visible origin tag -> visible fallback anchor -> cached value.
        T_cam_origin_now = None
        origin_source = None
        if ORIGIN_ENABLED:
            if ORIGIN_ID in det_by_id:
                origin_det = det_by_id[ORIGIN_ID]
                T_cam_origin_now = pose_to_T(origin_det.pose_R, origin_det.pose_t)
                origin_source = ORIGIN_ID
            else:
                for aid in FALLBACK_ANCHOR_IDS:
                    if aid in det_by_id and aid in ANCHOR_MAP:
                        T_cam_anchor = pose_to_T(det_by_id[aid].pose_R, det_by_id[aid].pose_t)
                        T_anchor_origin = np.linalg.inv(ANCHOR_MAP[aid])
                        T_cam_origin_now = T_cam_anchor @ T_anchor_origin
                        origin_source = aid
                        break
            if T_cam_origin_now is not None:
                T_cam_origin_cache = T_cam_origin_now

        T_cam_origin = T_cam_origin_cache if ORIGIN_ENABLED else None

        def to_world(T_cam_X):
            """Convert a cam-frame pose to origin frame. Falls back to identity when origin disabled."""
            if T_cam_origin is None:
                return T_cam_X
            return np.linalg.inv(T_cam_origin) @ T_cam_X

        def to_cam(T_world_X):
            """Inverse: origin-frame pose back to cam frame for rendering."""
            if T_cam_origin is None:
                return T_world_X
            return T_cam_origin @ T_world_X

        # ------ Collect detections ------
        world_T_boxtags = {}
        world_T_headtag = None
        head_margin = 0.0
        world_T_pelvistags = {}
        pelvis_margins = {}

        # Heuristic for "this is a box tag I just don't have in my map yet":
        # - not the head/pelvis tags
        # - tag id is small enough to plausibly be a box face (<= 9 covers ids 0..5)
        # Used only for GUI diagnostics; never feeds box pose estimation.
        UNMAPPED_BOX_HINT = set(range(0, 10)) - set(BOX_T_TAG.keys()) - {ROBOT_HEAD_TAG_ID} - set(PELVIS_TAG_IDS)
        if ORIGIN_ENABLED:
            UNMAPPED_BOX_HINT.discard(ORIGIN_ID)
            UNMAPPED_BOX_HINT -= FALLBACK_ANCHOR_SET

        for det in dets:
            T_cam_tag = pose_to_T(det.pose_R, det.pose_t)
            T_world_tag = to_world(T_cam_tag)
            margin = float(getattr(det, "decision_margin", 0.0))

            if det.tag_id in BOX_T_TAG:
                world_T_boxtags[det.tag_id] = T_world_tag
                draw_detection(img, det, (0, 255, 255))
                draw_axes(img, K, T_cam_tag, length=0.05)
            elif det.tag_id in UNMAPPED_BOX_HINT:
                draw_detection(img, det, (160, 160, 160), f"BOX?:{det.tag_id} (unmapped)")
                draw_axes(img, K, T_cam_tag, length=0.04)
            elif det.tag_id == ROBOT_HEAD_TAG_ID:
                world_T_headtag = T_world_tag
                head_margin = margin
                draw_detection(img, det, (255, 0, 255), f"HEAD:{det.tag_id}")
                draw_axes(img, K, T_cam_tag, length=0.05)
            elif det.tag_id in PELVIS_TAG_IDS:
                world_T_pelvistags[det.tag_id] = T_world_tag
                pelvis_margins[det.tag_id] = margin
                draw_detection(img, det, (0, 200, 255), f"PELVIS:{det.tag_id}")
                draw_axes(img, K, T_cam_tag, length=0.05)
            elif ORIGIN_ENABLED and det.tag_id == ORIGIN_ID:
                draw_detection(img, det, (0, 255, 0), f"ORIGIN:{det.tag_id}")
                draw_axes(img, K, T_cam_tag, length=0.10)
            elif ORIGIN_ENABLED and det.tag_id in FALLBACK_ANCHOR_SET:
                draw_detection(img, det, (0, 255, 128), f"ANCHOR:{det.tag_id}")
                draw_axes(img, K, T_cam_tag, length=0.06)

        # ------ Path A: head-tag → torso ------
        world_T_torso_head = None
        if world_T_headtag is not None:
            if T_tag_torso is not None:
                world_T_torso_head = world_T_headtag @ T_tag_torso
            else:
                world_T_torso_head = world_z_translate(world_T_headtag, args.head_z_offset)

        # ------ Path B: pelvis-tag(s) → root → torso ------
        world_T_pelvis_avg = None
        pelvis_margin_max = 0.0
        pelvis_used_id = -1
        if len(world_T_pelvistags) > 0:
            # If multiple pelvis tags visible, pick the one with highest margin (more robust than average for now).
            best_id, best_margin = max(pelvis_margins.items(), key=lambda kv: kv[1])
            pelvis_used_id = int(best_id)
            pelvis_margin_max = float(best_margin)
            world_T_pelvis_avg = world_T_pelvistags[best_id]

        world_T_root = None
        world_T_torso_pelvis = None
        if world_T_pelvis_avg is not None:
            world_T_root = world_z_translate(world_T_pelvis_avg, args.pelvis_to_root_z)
            world_T_torso_pelvis = world_z_translate(world_T_root, args.root_to_torso_z)

        # ------ Fused torso (simple position avg, rotation from head if available) ------
        world_T_torso = None
        torso_path = ""
        if world_T_torso_head is not None and world_T_torso_pelvis is not None:
            world_T_torso = world_T_torso_head.copy()
            world_T_torso[:3, 3] = 0.5 * (world_T_torso_head[:3, 3] + world_T_torso_pelvis[:3, 3])
            torso_path = "fused"
        elif world_T_torso_head is not None:
            world_T_torso = world_T_torso_head
            torso_path = "head"
        elif world_T_torso_pelvis is not None:
            world_T_torso = world_T_torso_pelvis
            torso_path = "pelvis"

        # GUI choice override
        gui_torso = world_T_torso
        if args.torso_source == "head" and world_T_torso_head is not None:
            gui_torso = world_T_torso_head
        elif args.torso_source == "pelvis" and world_T_torso_pelvis is not None:
            gui_torso = world_T_torso_pelvis

        if gui_torso is not None:
            draw_axes(img, K, to_cam(gui_torso), length=0.15)

        # Draw root and pelvis-derived torso so both paths are visible at once.
        if world_T_root is not None:
            draw_axes(img, K, to_cam(world_T_root), length=0.08)
        if world_T_torso_pelvis is not None and args.torso_source != "pelvis":
            draw_axes(img, K, to_cam(world_T_torso_pelvis), length=0.08)
        if world_T_torso_head is not None and args.torso_source != "head":
            draw_axes(img, K, to_cam(world_T_torso_head), length=0.08)

        # ------ Box pose ------
        world_T_box = None
        if len(world_T_boxtags) > 0:
            positions = []
            rotations = []
            for tag_id, T_world_tag in world_T_boxtags.items():
                if tag_id not in BOX_T_TAG:
                    continue
                world_T_box_candidate = T_world_tag @ np.linalg.inv(BOX_T_TAG[tag_id])
                positions.append(world_T_box_candidate[:3, 3])
                rotations.append(world_T_box_candidate[:3, :3])
            if len(positions) > 0:
                world_T_box = np.eye(4)
                world_T_box[:3, 3] = np.mean(np.stack(positions), axis=0)
                world_T_box[:3, :3] = average_rotation(rotations)
        if world_T_box is not None:
            draw_axes(img, K, to_cam(world_T_box), length=0.10)

        # ------ GUI overlay ------
        y = 30
        if ORIGIN_ENABLED:
            if T_cam_origin is None:
                origin_msg = "ORIGIN: NOT YET (need tag {} or anchor {})".format(
                    ORIGIN_ID, FALLBACK_ANCHOR_IDS)
                origin_color = (0, 0, 255)
            elif origin_source == ORIGIN_ID:
                origin_msg = f"ORIGIN: visible (id={ORIGIN_ID})"
                origin_color = (0, 255, 0)
            elif origin_source is not None:
                origin_msg = f"ORIGIN: via anchor {origin_source}"
                origin_color = (0, 255, 128)
            else:
                origin_msg = f"ORIGIN: cached (latched)"
                origin_color = (0, 200, 255)
            cv2.putText(img, origin_msg, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, origin_color, 2)
        else:
            cv2.putText(img, "ORIGIN: disabled (cam frame = world)", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        y += 22
        cv2.putText(img, f"head:{ROBOT_HEAD_TAG_ID} {'OK' if world_T_headtag is not None else '--'}  "
                         f"pelvis:{PELVIS_TAG_IDS} {pelvis_used_id if pelvis_used_id>=0 else '--'}  "
                         f"box_tags:{len(world_T_boxtags)}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 25
        if world_T_headtag is not None:
            hp = world_T_headtag[:3, 3]
            cv2.putText(img,
                        f"HeadTag(id={ROBOT_HEAD_TAG_ID}): [{hp[0]:+.3f}, {hp[1]:+.3f}, {hp[2]:+.3f}]  m={head_margin:.0f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            y += 22
        if world_T_pelvis_avg is not None:
            pp = world_T_pelvis_avg[:3, 3]
            cv2.putText(img,
                        f"PelvisTag(id={pelvis_used_id}): [{pp[0]:+.3f}, {pp[1]:+.3f}, {pp[2]:+.3f}]  m={pelvis_margin_max:.0f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            y += 22
        if world_T_root is not None:
            rp = world_T_root[:3, 3]
            cv2.putText(img,
                        f"Root (pelvis+{args.pelvis_to_root_z:+.2f}z): [{rp[0]:+.3f}, {rp[1]:+.3f}, {rp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2)
            y += 22
        if world_T_torso_head is not None:
            thp = world_T_torso_head[:3, 3]
            cv2.putText(img,
                        f"Torso<head>: [{thp[0]:+.3f}, {thp[1]:+.3f}, {thp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 130, 220), 2)
            y += 20
        if world_T_torso_pelvis is not None:
            tpp = world_T_torso_pelvis[:3, 3]
            cv2.putText(img,
                        f"Torso<pelvis>: [{tpp[0]:+.3f}, {tpp[1]:+.3f}, {tpp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 200, 220), 2)
            y += 20
        if gui_torso is not None:
            tp = gui_torso[:3, 3]
            cv2.putText(img,
                        f"Torso({args.torso_source}): [{tp[0]:+.3f}, {tp[1]:+.3f}, {tp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            y += 22
        if world_T_box is not None:
            bp = world_T_box[:3, 3]
            cv2.putText(img,
                        f"Box: [{bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y += 22
            if gui_torso is not None:
                rel = bp - gui_torso[:3, 3]
                cv2.putText(img,
                            f"box-torso: [{rel[0]:+.3f}, {rel[1]:+.3f}, {rel[2]:+.3f}]",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                y += 22
        if world_T_box is None:
            cv2.putText(img, "Box: NOT VISIBLE", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # ------ Console (throttled) ------
        if args.print_every > 0 and frame_idx % args.print_every == 0 and gui_torso is not None and world_T_box is not None:
            tp = gui_torso[:3, 3]
            bp = world_T_box[:3, 3]
            rel = bp - tp
            print(f"[shadow {frame_idx:5d}] torso({torso_path})=[{tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f}] "
                  f"box=[{bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f}] rel=[{rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f}]")

        # ------ CSV row ------
        # When origin mode is on but origin is not yet acquired, skip logging:
        # otherwise frames in cam-frame would silently mix with origin-frame frames.
        skip_csv = ORIGIN_ENABLED and T_cam_origin is None
        if csv_writer is not None and not skip_csv:
            row = csv_row({
                "frame_idx": frame_idx,
                "t_sec": time.time() - t_start,
                "head_visible": int(world_T_headtag is not None),
                "head_margin": head_margin,
                "pelvis_visible": int(world_T_pelvis_avg is not None),
                "pelvis_used_id": pelvis_used_id,
                "pelvis_margin_max": pelvis_margin_max,
                "torso_path": torso_path,
                "box_visible": int(world_T_box is not None),
                "box_n_tags": len(world_T_boxtags),
            })
            row.update(fill_pose("head", world_T_headtag))
            row.update(fill_pose("pelvis", world_T_pelvis_avg))
            row.update(fill_pose("root", world_T_root))
            row.update(fill_pose("torso_head", world_T_torso_head))
            row.update(fill_pose("torso_pelvis", world_T_torso_pelvis))
            row.update(fill_pose("torso", world_T_torso))
            row.update(fill_pose("obj", world_T_box))
            csv_writer.writerow(row)

        cv2.imshow("WORLD", img)
        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()
        print(f"[shadow] CSV closed: {args.csv_out}")
