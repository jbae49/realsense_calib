"""
Multi-camera (cam1/cam2/cam3) shadow logger for robot torso + box.

Design:
  - cam2 is the WORLD frame (origin). Required.
  - cam1 and cam3 are optional. Their detections are mapped into cam2/world via
    the corresponding extrinsic npz (key 'T_c2_c1' regardless of source camera).
  - For each tag_id, candidates from all visible cameras are fused using
    decision_margin as weight (filter by --margin-min).
  - Two torso estimation paths logged independently (head, pelvis) plus a fused.
  - CSV schema is compatible with track_robot_and_box.py and
    compute_ref_alignment_yaw_only.py.

Tag layout (current robot setup):
  - head tag (default 9), pelvis tags (default 8,7), box tags (from box_tag_map.npz).
"""
import argparse
import csv
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
# Helpers
# =========================================================

def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def rotation_to_quat_wxyz(R):
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
    out = T_world_X.copy()
    out[2, 3] += dz
    return out


def weighted_avg_rotation(rotations, weights):
    if len(rotations) == 1:
        return rotations[0]
    w = np.asarray(weights, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w_sum
    R = np.zeros((3, 3), dtype=float)
    for rot, wi in zip(rotations, w):
        R += wi * rot
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def fuse_tag_pose(candidates):
    """candidates: list of {'T': 4x4, 'w': float}. Returns fused 4x4."""
    if len(candidates) == 1:
        return candidates[0]["T"]
    rots = [c["T"][:3, :3] for c in candidates]
    poss = np.stack([c["T"][:3, 3] for c in candidates], axis=0)
    ws = np.asarray([c["w"] for c in candidates], dtype=float)
    R = weighted_avg_rotation(rots, ws)
    s = float(np.sum(ws))
    if s <= 1e-9:
        wn = np.ones_like(ws) / len(ws)
    else:
        wn = ws / s
    p = np.sum(poss * wn[:, None], axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def load_camera_params(path):
    data = np.load(path)
    K = data["camera_matrix"]
    return K, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]


def draw_detection(img, det, color, label=""):
    corners = det.corners.astype(int)
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    text = label if label else f"ID:{det.tag_id}"
    cv2.putText(img, text, (center[0] + 8, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_axes(img, K, T_cam_frame, length=0.06):
    dist = np.zeros(5)
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3]
    pts3 = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    pts2, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
    pts = pts2.reshape(-1, 2).astype(int)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(img, tuple(pts[0]), tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(img, tuple(pts[0]), tuple(pts[3]), (255, 0, 0), 2)


# =========================================================
# Args
# =========================================================

parser = argparse.ArgumentParser()
# Cam2 = WORLD (required).
parser.add_argument("--cam2-serial", type=str, default="115222071236")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
# Cam1 (optional).
parser.add_argument("--cam1-serial", type=str, default="935322072654")
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz",
                    help="cam1->cam2 extrinsic npz (key T_c2_c1)")
parser.add_argument("--no-cam1", action="store_true")
# Cam3 (optional).
parser.add_argument("--cam3-serial", type=str, default="")
parser.add_argument("--cam3-calib", type=str, default="")
parser.add_argument("--extrinsic-cam3-to-c2", type=str, default="",
                    help="cam3->cam2 extrinsic npz (key T_c2_c1)")

# Tag layout.
parser.add_argument("--head-tag-id", type=int, default=9)
parser.add_argument("--pelvis-tag-ids", type=str, default="8,7")
parser.add_argument("--box-tag-map", type=str, default="box_tag_map.npz")
parser.add_argument("--head-tag-calib", type=str, default="T_tag_torso.npz",
                    help="Optional precise T_tag_torso npz; falls back to --head-z-offset")

# Offsets.
parser.add_argument("--head-z-offset", type=float, default=-0.25)
parser.add_argument("--pelvis-to-root-z", type=float, default=0.10)
parser.add_argument("--root-to-torso-z", type=float, default=0.20)

# Fusion.
parser.add_argument("--margin-min", type=float, default=40.0)

# Output.
parser.add_argument("--torso-source", type=str, default="fused",
                    choices=["head", "pelvis", "fused"])
parser.add_argument("--csv-out", type=str, default="")
parser.add_argument("--print-every", type=int, default=30)

# Stream.
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)

# AprilTag config.
parser.add_argument("--tag-size", type=float, default=0.077)
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")

args = parser.parse_args()

# Decide which cams are active.
USE_CAM1 = (not args.no_cam1) and bool(args.cam1_serial.strip())
USE_CAM3 = bool(args.cam3_serial.strip())
if USE_CAM1 and not args.extrinsic.strip():
    raise ValueError("--extrinsic is required when cam1 is enabled")
if USE_CAM3:
    if not args.cam3_calib.strip():
        raise ValueError("--cam3-calib is required when --cam3-serial is set")
    if not args.extrinsic_cam3_to_c2.strip():
        raise ValueError("--extrinsic-cam3-to-c2 is required when --cam3-serial is set")

ROBOT_HEAD_TAG_ID = args.head_tag_id
PELVIS_TAG_IDS = [int(x) for x in args.pelvis_tag_ids.split(",") if x.strip()]

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)


# =========================================================
# Load calibrations
# =========================================================

K2, cam2_params = load_camera_params(args.cam2_calib)
if USE_CAM1:
    K1, cam1_params = load_camera_params(args.cam1_calib)
    T_c2_c1 = np.load(args.extrinsic)["T_c2_c1"]
else:
    K1 = None
    cam1_params = None
    T_c2_c1 = None
if USE_CAM3:
    K3, cam3_params = load_camera_params(args.cam3_calib)
    T_c2_c3 = np.load(args.extrinsic_cam3_to_c2)["T_c2_c1"]
else:
    K3 = None
    cam3_params = None
    T_c2_c3 = None

# Box tag map.
map_data = np.load(args.box_tag_map)
BOX_T_TAG = {int(tid): T for tid, T in zip(map_data["tag_ids"], map_data["box_T_tags"])}
BOX_TAG_IDS = sorted(BOX_T_TAG.keys())

# Head tag → torso calibration (optional).
T_tag_torso = None
if args.head_tag_calib and os.path.exists(args.head_tag_calib):
    T_tag_torso = np.load(args.head_tag_calib)["T_tag_torso"]
    print(f"Loaded T_tag_torso (translation: {T_tag_torso[:3, 3]})")
else:
    print(f"[shadow] {args.head_tag_calib} not found; "
          f"falling back to head_z_offset={args.head_z_offset:+.3f} m for head→torso.")


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
    debug=False,
)


def make_pipeline(serial):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe.start(cfg)
    return pipe


pipe2 = make_pipeline(args.cam2_serial)
pipe1 = make_pipeline(args.cam1_serial) if USE_CAM1 else None
pipe3 = make_pipeline(args.cam3_serial) if USE_CAM3 else None

print("\n======================================")
print("Tracking: Robot torso + Box (multi-cam shadow logger)")
print(f"cam2(world)={args.cam2_serial}")
if USE_CAM1:
    print(f"cam1={args.cam1_serial}, extrinsic={args.extrinsic}")
if USE_CAM3:
    print(f"cam3={args.cam3_serial}, extrinsic_cam3_to_c2={args.extrinsic_cam3_to_c2}")
print(f"head={ROBOT_HEAD_TAG_ID}, pelvis={PELVIS_TAG_IDS}, box={BOX_TAG_IDS}")
print(f"margin_min={args.margin_min:.1f}, torso_source(GUI)={args.torso_source}, "
      f"csv_out={args.csv_out or '(off)'}")
print("Press ESC to quit")
print("======================================\n")


# =========================================================
# CSV setup (compatible schema with single-cam version + per-cam visibility)
# =========================================================

CSV_COLUMNS = [
    "frame_idx", "t_sec",
    "head_visible", "head_pos_x", "head_pos_y", "head_pos_z",
    "head_quat_w", "head_quat_x", "head_quat_y", "head_quat_z",
    "head_margin_max", "head_seen_cams",
    "pelvis_visible", "pelvis_used_id",
    "pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z",
    "pelvis_quat_w", "pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z",
    "pelvis_margin_max", "pelvis_seen_cams",
    "root_pos_x", "root_pos_y", "root_pos_z",
    "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "torso_head_pos_x", "torso_head_pos_y", "torso_head_pos_z",
    "torso_head_quat_w", "torso_head_quat_x", "torso_head_quat_y", "torso_head_quat_z",
    "torso_pelvis_pos_x", "torso_pelvis_pos_y", "torso_pelvis_pos_z",
    "torso_pelvis_quat_w", "torso_pelvis_quat_x", "torso_pelvis_quat_y", "torso_pelvis_quat_z",
    "torso_pos_x", "torso_pos_y", "torso_pos_z",
    "torso_quat_w", "torso_quat_x", "torso_quat_y", "torso_quat_z",
    "torso_path",
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

def detect_for_cam(pipe, cam_params):
    """Returns (img, dets) for a given pipeline, or (None, []) if pipe is None."""
    if pipe is None:
        return None, []
    frames = pipe.wait_for_frames()
    color = frames.get_color_frame()
    if not color:
        return None, []
    img = np.asanyarray(color.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detect_with_tag_sizes(detector, gray, cam_params, TAG_SIZE, TAG_SIZE_MAP)
    return img, dets


t_start = time.time()
frame_idx = 0

try:
    while True:
        frame_idx += 1

        img2, dets2 = detect_for_cam(pipe2, cam2_params)
        img1, dets1 = detect_for_cam(pipe1, cam1_params) if USE_CAM1 else (None, [])
        img3, dets3 = detect_for_cam(pipe3, cam3_params) if USE_CAM3 else (None, [])

        if img2 is None:
            continue

        # Per-tag candidate buckets (T in world/c2 frame).
        tag_candidates = {}  # tag_id -> list[{T, w, src}]
        head_seen_cams = []
        pelvis_seen_cams_per_id = {}

        def push(tag_id, T_world_tag, w, src):
            tag_candidates.setdefault(int(tag_id), []).append({"T": T_world_tag, "w": max(w, 1e-3), "src": src})

        # cam1 → world
        for det in dets1:
            T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
            T_world_tag = T_c2_c1 @ T_c1_tag
            w = float(getattr(det, "decision_margin", 1.0))
            color = (0, 255, 200)
            label = None
            if det.tag_id == ROBOT_HEAD_TAG_ID:
                color = (255, 0, 255); label = f"HEAD:{det.tag_id}"
                if w >= args.margin_min:
                    head_seen_cams.append("cam1")
            elif det.tag_id in PELVIS_TAG_IDS:
                color = (0, 200, 255); label = f"PELVIS:{det.tag_id}"
                if w >= args.margin_min:
                    pelvis_seen_cams_per_id.setdefault(int(det.tag_id), []).append("cam1")
            elif det.tag_id in BOX_T_TAG:
                color = (0, 255, 255); label = f"BOX:{det.tag_id}"
            draw_detection(img1, det, color, label)
            draw_axes(img1, K1, T_c1_tag, length=0.05)
            if w >= args.margin_min:
                push(det.tag_id, T_world_tag, w, "cam1")

        # cam2 (world)
        for det in dets2:
            T_world_tag = pose_to_T(det.pose_R, det.pose_t)
            w = float(getattr(det, "decision_margin", 1.0))
            color = (0, 255, 200)
            label = None
            if det.tag_id == ROBOT_HEAD_TAG_ID:
                color = (255, 0, 255); label = f"HEAD:{det.tag_id}"
                if w >= args.margin_min:
                    head_seen_cams.append("cam2")
            elif det.tag_id in PELVIS_TAG_IDS:
                color = (0, 200, 255); label = f"PELVIS:{det.tag_id}"
                if w >= args.margin_min:
                    pelvis_seen_cams_per_id.setdefault(int(det.tag_id), []).append("cam2")
            elif det.tag_id in BOX_T_TAG:
                color = (0, 255, 255); label = f"BOX:{det.tag_id}"
            draw_detection(img2, det, color, label)
            draw_axes(img2, K2, T_world_tag, length=0.05)
            if w >= args.margin_min:
                push(det.tag_id, T_world_tag, w, "cam2")

        # cam3 → world
        for det in dets3:
            T_c3_tag = pose_to_T(det.pose_R, det.pose_t)
            T_world_tag = T_c2_c3 @ T_c3_tag
            w = float(getattr(det, "decision_margin", 1.0))
            color = (0, 255, 200)
            label = None
            if det.tag_id == ROBOT_HEAD_TAG_ID:
                color = (255, 0, 255); label = f"HEAD:{det.tag_id}"
                if w >= args.margin_min:
                    head_seen_cams.append("cam3")
            elif det.tag_id in PELVIS_TAG_IDS:
                color = (0, 200, 255); label = f"PELVIS:{det.tag_id}"
                if w >= args.margin_min:
                    pelvis_seen_cams_per_id.setdefault(int(det.tag_id), []).append("cam3")
            elif det.tag_id in BOX_T_TAG:
                color = (0, 255, 255); label = f"BOX:{det.tag_id}"
            draw_detection(img3, det, color, label)
            draw_axes(img3, K3, T_c3_tag, length=0.05)
            if w >= args.margin_min:
                push(det.tag_id, T_world_tag, w, "cam3")

        # ---- Fuse per-tag world poses ----
        fused_world = {tid: fuse_tag_pose(cands) for tid, cands in tag_candidates.items()}
        fused_margin = {tid: max(c["w"] for c in cands) for tid, cands in tag_candidates.items()}

        # Head
        world_T_headtag = fused_world.get(ROBOT_HEAD_TAG_ID)
        head_margin_max = float(fused_margin.get(ROBOT_HEAD_TAG_ID, 0.0))

        # Pelvis: pick best-margin id
        pelvis_used_id = -1
        world_T_pelvis_avg = None
        pelvis_margin_max = 0.0
        for pid in PELVIS_TAG_IDS:
            if pid in fused_world and fused_margin[pid] > pelvis_margin_max:
                pelvis_margin_max = float(fused_margin[pid])
                pelvis_used_id = int(pid)
                world_T_pelvis_avg = fused_world[pid]

        # ---- Path A: head → torso ----
        world_T_torso_head = None
        if world_T_headtag is not None:
            if T_tag_torso is not None:
                world_T_torso_head = world_T_headtag @ T_tag_torso
            else:
                world_T_torso_head = world_z_translate(world_T_headtag, args.head_z_offset)

        # ---- Path B: pelvis → root → torso ----
        world_T_root = None
        world_T_torso_pelvis = None
        if world_T_pelvis_avg is not None:
            world_T_root = world_z_translate(world_T_pelvis_avg, args.pelvis_to_root_z)
            world_T_torso_pelvis = world_z_translate(world_T_root, args.root_to_torso_z)

        # ---- Fused torso (simple position avg, rotation from head if available) ----
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

        gui_torso = world_T_torso
        if args.torso_source == "head" and world_T_torso_head is not None:
            gui_torso = world_T_torso_head
        elif args.torso_source == "pelvis" and world_T_torso_pelvis is not None:
            gui_torso = world_T_torso_pelvis

        # ---- Draw fused torso on cam2 image ----
        if gui_torso is not None:
            draw_axes(img2, K2, gui_torso, length=0.15)

        # ---- Box pose ----
        world_T_box = None
        n_box_tags = 0
        if BOX_T_TAG:
            positions, rotations = [], []
            for tag_id, T_world_tag in fused_world.items():
                if tag_id not in BOX_T_TAG:
                    continue
                if fused_margin.get(tag_id, 0.0) < args.margin_min:
                    continue
                cand = T_world_tag @ np.linalg.inv(BOX_T_TAG[tag_id])
                positions.append(cand[:3, 3])
                rotations.append(cand[:3, :3])
                n_box_tags += 1
            if positions:
                world_T_box = np.eye(4)
                world_T_box[:3, 3] = np.mean(np.stack(positions), axis=0)
                # equal-weight box rotation average is fine for now
                w_eq = np.ones(len(rotations), dtype=float)
                world_T_box[:3, :3] = weighted_avg_rotation(rotations, w_eq)
            if world_T_box is not None:
                draw_axes(img2, K2, world_T_box, length=0.10)

        # ---- GUI overlay (cam2 panel header) ----
        y = 30
        cv2.putText(img2, f"head={ROBOT_HEAD_TAG_ID} cams:{','.join(head_seen_cams) if head_seen_cams else '--'}  "
                          f"pelvis={pelvis_used_id if pelvis_used_id >= 0 else '--'}  "
                          f"box_tags={n_box_tags}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 25
        if gui_torso is not None:
            tp = gui_torso[:3, 3]
            cv2.putText(img2,
                        f"Torso({args.torso_source}): [{tp[0]:+.3f}, {tp[1]:+.3f}, {tp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            y += 22
        if world_T_box is not None:
            bp = world_T_box[:3, 3]
            cv2.putText(img2,
                        f"Box: [{bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y += 22
            if gui_torso is not None:
                rel = bp - gui_torso[:3, 3]
                cv2.putText(img2,
                            f"box-torso: [{rel[0]:+.3f}, {rel[1]:+.3f}, {rel[2]:+.3f}]",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # ---- Throttled console ----
        if args.print_every > 0 and frame_idx % args.print_every == 0 and gui_torso is not None and world_T_box is not None:
            tp = gui_torso[:3, 3]
            bp = world_T_box[:3, 3]
            rel = bp - tp
            print(f"[shadow {frame_idx:5d}] torso({torso_path}) seen={head_seen_cams} "
                  f"=[{tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f}] "
                  f"box=[{bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f}] "
                  f"rel=[{rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f}]")

        # ---- CSV row ----
        if csv_writer is not None:
            row = csv_row({
                "frame_idx": frame_idx,
                "t_sec": time.time() - t_start,
                "head_visible": int(world_T_headtag is not None),
                "head_margin_max": head_margin_max,
                "head_seen_cams": ",".join(head_seen_cams),
                "pelvis_visible": int(world_T_pelvis_avg is not None),
                "pelvis_used_id": pelvis_used_id,
                "pelvis_margin_max": pelvis_margin_max,
                "pelvis_seen_cams": ",".join(pelvis_seen_cams_per_id.get(pelvis_used_id, []))
                                    if pelvis_used_id >= 0 else "",
                "torso_path": torso_path,
                "box_visible": int(world_T_box is not None),
                "box_n_tags": n_box_tags,
            })
            row.update(fill_pose("head", world_T_headtag))
            row.update(fill_pose("pelvis", world_T_pelvis_avg))
            row.update(fill_pose("root", world_T_root))
            row.update(fill_pose("torso_head", world_T_torso_head))
            row.update(fill_pose("torso_pelvis", world_T_torso_pelvis))
            row.update(fill_pose("torso", world_T_torso))
            row.update(fill_pose("obj", world_T_box))
            csv_writer.writerow(row)

        # ---- Show panels ----
        cv2.imshow("WORLD (cam2)", img2)
        if img1 is not None:
            cv2.imshow("cam1 -> WORLD", img1)
        if img3 is not None:
            cv2.imshow("cam3 -> WORLD", img3)

        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
            break

finally:
    pipe2.stop()
    if pipe1 is not None:
        pipe1.stop()
    if pipe3 is not None:
        pipe3.stop()
    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()
        print(f"[shadow] CSV closed: {args.csv_out}")
