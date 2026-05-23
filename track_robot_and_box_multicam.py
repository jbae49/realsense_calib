"""
track_robot_and_box_multicam.py
Multi-camera (1~3 RealSense) shadow logger for robot torso + box.

Per-camera origin establishment (priority order):
  1) DIRECT      — that camera sees the primary origin tag (id=1) directly.
  2) VIA_ANCHOR  — primary not seen, but a secondary floor anchor (e.g. tag 10) is.
                   We pre-stored T_origin_anchor (the anchor tag's pose in origin frame),
                   and update each camera's own copy whenever both tag1 and the anchor
                   are visible in the same frame.  Then:
                       T_camN_origin = T_camN_anchor @ inv(T_origin_anchor)
                   — no inter-camera extrinsic involved.
  3) HELD        — last good T_camN_origin, kept for up to --origin-hold-max-frames.
  4) NONE        — that camera doesn't contribute this frame.

Once any camera has an origin pose, all of its tag detections become DIRECT candidates
(per-camera weighted average across all cameras that have an origin):
       T_origin_tag(N) = inv(T_camN_origin) @ T_camN_tag
       weight          = min(origin_path_w, tag_margin)
Translation: weighted mean. Rotation: SVD-projected weighted mean.

Robot pose math is then identical to the single-camera tracker:
  head_tag → torso_head  (T_tag_torso npz, or world-z offset fallback)
  pelvis_tag → root → torso_pelvis  (world-z offsets)
  box       → mean over per-tag candidates with BOX_T_TAG

World frame (Z sign):
  When --origin-id >= 0, we use the floor anchor tag as origin. pupil_apriltags
  follows the OpenCV solvePnP convention, so the tag's local Z axis points "into the
  page" (away from the camera). For a face-up floor tag this means
        +Z = downward,  -Z = upward
  All default z-offsets below are signed for that convention.
"""
import argparse
import csv
import datetime
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def world_z_translate(T, dz):
    out = T.copy()
    out[2, 3] += dz
    return out


_AXIS_VEC = {"+x": np.array([1.0,0.0,0.0]), "-x": np.array([-1.0,0.0,0.0]),
             "+y": np.array([0.0,1.0,0.0]), "-y": np.array([0.0,-1.0,0.0]),
             "+z": np.array([0.0,0.0,1.0]), "-z": np.array([0.0,0.0,-1.0])}


def parse_axis(axis_str):
    """Parse '+x'/'-y'/'+z' etc. into a unit vector. 'x' assumed '+x'."""
    s = axis_str.strip().lower()
    if s and s[0] not in "+-":
        s = "+" + s
    if s not in _AXIS_VEC:
        raise ValueError(f"Bad axis '{axis_str}', expect one of {list(_AXIS_VEC)}")
    return _AXIS_VEC[s]


def local_axis_translate(T, axis_str, distance):
    """Move T along its local axis by `distance` meters (in T's own frame).
    Orientation is preserved; only the translation column changes.
    """
    v_local = parse_axis(axis_str)
    v_world = T[:3, :3] @ v_local
    out = T.copy()
    out[:3, 3] = T[:3, 3] + v_world * float(distance)
    return out


def average_rotation(R_list, weights=None):
    if len(R_list) == 1:
        return R_list[0]
    if weights is None:
        weights = np.ones(len(R_list), dtype=float)
    w = np.asarray(weights, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w_sum
    R_avg = np.zeros((3, 3), dtype=float)
    for Ri, wi in zip(R_list, w):
        R_avg += wi * Ri
    U, _, Vt = np.linalg.svd(R_avg)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def ema_pose(prev_T, new_T, alpha):
    """Pose EMA: linear EMA on translation + SVD-mean (a "weighted SLERP" that
    works for any number of inputs) on rotation.

    alpha = weight of the NEW measurement, in [0, 1].
        alpha = 1.0  -> passthrough (returns new_T)
        alpha = 0.3  -> ~3-frame time constant
        alpha = 0.1  -> ~10-frame time constant

    Edge cases:
      - prev_T is None  : nothing to mix with, return new_T (or None).
      - new_T is None   : "carry-over" the previous estimate (e.g. box not
                          detected this frame). Returns prev_T.

    Note: SVD-projected weighted mean is exact only when both rotations are
    "close" (within a few tens of degrees). For setup-phase static scenes this
    is always true; the SVD step renormalises to SO(3) regardless.
    """
    if new_T is None:
        return prev_T
    if prev_T is None or alpha >= 1.0:
        return new_T.copy()
    out = np.eye(4)
    out[:3, 3] = (1.0 - alpha) * prev_T[:3, 3] + alpha * new_T[:3, 3]
    out[:3, :3] = average_rotation(
        [prev_T[:3, :3], new_T[:3, :3]], [1.0 - alpha, alpha]
    )
    return out


def fuse_poses(candidates):
    """candidates = list of dict {T, w, src}.  Returns fused 4x4 T (and source list)."""
    if not candidates:
        return None, []
    if len(candidates) == 1:
        return candidates[0]["T"].copy(), [candidates[0]["src"]]
    weights = np.array([max(c["w"], 1e-3) for c in candidates], dtype=float)
    positions = np.stack([c["T"][:3, 3] for c in candidates], axis=0)
    rotations = [c["T"][:3, :3] for c in candidates]
    w_norm = weights / weights.sum()
    pos = (positions * w_norm[:, None]).sum(axis=0)
    rot = average_rotation(rotations, weights)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T, [c["src"] for c in candidates]


def rotation_to_rot6d(R):
    """6D rotation representation (Zhou et al. 2019): first two columns of R.

    Layout: [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]] -- column-major
    of the first two columns. This matches IsaacLab / mjlab `_ori6_*` outputs.
    """
    return np.concatenate([R[:, 0], R[:, 1]]).astype(float)


def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


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


def draw_detection(img, det, color, label=""):
    corners = det.corners.astype(int)
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    text = label if label else f"ID:{det.tag_id}"
    cv2.putText(img, text, (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def parse_int_list(raw):
    if not raw or not raw.strip():
        return []
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def load_anchor_transforms(path, origin_id):
    """Load T_origin_anchor for each anchor id from JSON.

    Format (config/floor_anchor_transforms.json):
        {
          "origin_id": 1,
          "anchors": {
            "10": {"T_origin_anchor": <4x4 matrix>}
          }
        }
    Returns: dict {anchor_id (int) -> 4x4 numpy array}
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"[anchor] failed to load {path}: {exc}")
        return {}
    cfg_origin = int(data.get("origin_id", origin_id))
    if cfg_origin != int(origin_id):
        print(
            f"[anchor] WARN config origin_id={cfg_origin} != --origin-id={origin_id}; "
            "using entries anyway."
        )
    out = {}
    for raw_id, meta in (data.get("anchors") or {}).items():
        try:
            aid = int(raw_id)
            T = np.asarray(meta["T_origin_anchor"], dtype=float)
            if T.shape == (4, 4):
                out[aid] = T
        except Exception as exc:
            print(f"[anchor] skipping entry {raw_id}: {exc}")
    return out


def draw_axes_cam(img, K, T_cam_frame, length=0.06):
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3]
    points_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, np.zeros(5))
    pts = points_2d.reshape(-1, 2).astype(int)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(img, tuple(pts[0]), tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(img, tuple(pts[0]), tuple(pts[3]), (255, 0, 0), 2)


# =========================================================
# CLI
# =========================================================

parser = argparse.ArgumentParser()
# Cameras (cam1 + cam2 required, cam3 optional)
parser.add_argument("--cam1-serial", type=str, default="935322072654")
parser.add_argument("--cam2-serial", type=str, default="115222071236")
parser.add_argument("--cam3-serial", type=str, default="")
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--cam3-calib", type=str, default="camera3_112322072671_calibration.npz")
# Tag setup
parser.add_argument("--origin-id", type=int, default=1,
                    help="Primary floor-anchor tag id used as world origin")
parser.add_argument("--anchor-ids", type=str, default="10",
                    help="Comma-separated secondary floor anchor tag ids (used when origin id is hidden). "
                         "Each anchor uses its T_origin_anchor entry from --anchor-config.")
parser.add_argument("--anchor-config", type=str, default="config/floor_anchor_transforms.json",
                    help="JSON with anchor transforms: anchors[id].T_origin_anchor (4x4)")
parser.add_argument("--head-tag-id", type=int, default=9)
parser.add_argument("--pelvis-tag-ids", type=str, default="8,7")
parser.add_argument("--box-tag-map", type=str, default="box_tag_map.npz")
# Box body-frame correction: mjlab's large_box.xml defines the body frame
# along the *mesh AABB* axes (because the mesh itself is drawn with the
# box's real faces tilted ~46° w.r.t. the AABB). The inertial quat in
# largebox.xml encodes that tilt:
#   <inertial ... quat="0.920083 -0.118744 -0.106429 0.357797" .../>
# Our box_tag_map (from box_tag_calib) measures the box's *real-face* OBB
# frame, which equals the principal-inertial axes for a uniform box. So
# the rotation that maps "csv-frame box quat" -> "mjlab-frame box quat"
# is exactly the inverse of the inertial quat:
#   q_world_box_mjlab = q_world_box_csv * inertial_quat.conjugate()
# Without this correction the policy sees a constant ~46° offset between
# the npz-derived ref_object quat (mjlab frame) and camera-derived
# object quat (csv frame), which prevents alignment from passing.
parser.add_argument("--box-inertial-quat", type=str,
                    default="0.920083 -0.118744 -0.106429 0.357797",
                    help="(wxyz) Inertial quat from largebox.xml's <inertial> "
                         "tag, used to convert the OBB-aligned tracker pose "
                         "into the mjlab body-frame the npz / policy expect. "
                         "Pass 'identity' to disable. Read the comment in "
                         "track_robot_and_box_multicam.py near this flag.")
parser.add_argument("--head-tag-calib", type=str, default="T_tag_torso.npz",
                    help="Optional. If present we use T_tag_torso, else fall back to head_z_offset.")
# Z offsets (origin-tag world frame: +Z = down, -Z = up; see docstring above)
parser.add_argument("--head-z-offset", type=float, default=0.25,
                    help="(world-z mode) World-z offset from head tag to torso_link [m] "
                         "(fallback when no T_tag_torso.npz)")
parser.add_argument("--head-up-mode", type=str, default="tag-axis",
                    choices=["tag-axis", "world-z"],
                    help=("How to move from head tag down along the body axis to torso_link "
                          "when no T_tag_torso.npz is loaded. 'tag-axis' (default): along the "
                          "head tag's local axis given by --head-tag-down-axis (correct when "
                          "the head is tilted, e.g. while bending). 'world-z': legacy, uses "
                          "world +/-z (only valid when head stays roughly upright). Ignored "
                          "when --head-tag-calib is loaded."))
parser.add_argument("--head-tag-down-axis", type=str, default="+z",
                    help=("Which LOCAL axis of the head AprilTag points toward body-DOWN "
                          "(i.e., toward torso/pelvis). For G1 with the head tag on top of "
                          "the head facing up, pupil_apriltags +z is INTO the tag = INTO the "
                          "head = body-down, so default '+z'."))
parser.add_argument("--head-to-torso-body", type=float, default=0.25,
                    help="(tag-axis mode) Distance from head tag along --head-tag-down-axis "
                         "to torso_link [m] (default 25 cm)")
parser.add_argument("--pelvis-to-root-z", type=float, default=-0.05,
                    help="(world-z mode only) Offset from pelvis tag up to root in WORLD z [m]")
parser.add_argument("--root-to-torso-z", type=float, default=-0.20,
                    help="(world-z mode only) Offset from root up to torso_link in WORLD z [m]")
parser.add_argument("--pelvis-up-mode", type=str, default="tag-axis",
                    choices=["tag-axis", "world-z"],
                    help=("How to move from pelvis tag along the body axis. "
                          "'tag-axis' (default): along the pelvis tag's local "
                          "axis given by --pelvis-tag-up-axis (correct when robot is "
                          "bent or tilted). 'world-z': legacy behaviour, uses world +/-z, "
                          "only valid when robot stays roughly upright."))
parser.add_argument("--pelvis-tag-up-axis", type=str, default="-y",
                    help=("Which LOCAL axis of the pelvis AprilTag points toward "
                          "body-up. Default '-y' matches both tag 7 and tag 8 on the G1 "
                          "(verified empirically in the lab)."))
parser.add_argument("--pelvis-to-root-body", type=float, default=0.05,
                    help="(tag-axis mode) Distance UP along the body axis from pelvis tag to root [m]")
parser.add_argument("--root-to-torso-body", type=float, default=0.20,
                    help="(tag-axis mode) Distance UP along the body axis from root to torso_link [m]")
parser.add_argument("--torso-source", type=str, default="fused",
                    choices=["head", "pelvis", "fused"])
# Pose smoothing (EMA on translation + SVD-mean SLERP on rotation, applied
# AFTER fusion, BEFORE GUI/CSV/UDP). Useful for setup phase: the robot and box
# are static so jitter from sub-pixel apriltag noise can be heavily attenuated.
# alpha = weight of the NEW measurement:
#   1.0  -> passthrough (no smoothing, default for backwards compat)
#   0.3  -> moderate (~3-frame effective time constant)
#   0.1  -> heavy    (~10-frame effective time constant; good for static box)
parser.add_argument("--ema-alpha-torso", type=float, default=1.0,
                    help="EMA factor for torso pose (1=raw, 0.3=moderate, 0.1=heavy). "
                         "Use ~0.3 during setup; ~0.5 during deploy when robot moves.")
parser.add_argument("--ema-alpha-box", type=float, default=1.0,
                    help="EMA factor for box pose (1=raw, 0.3=moderate, 0.1=heavy). "
                         "Box on a stand is static -> ~0.1-0.2 recommended.")
# Fusion
parser.add_argument("--margin-min", type=float, default=40.0,
                    help="Minimum decision_margin for a detection to enter fusion")
parser.add_argument("--origin-hold", action="store_true", default=True,
                    help="When a camera briefly loses the origin tag, keep using its last "
                         "T_camN_origin so direct path is still available")
parser.add_argument("--no-origin-hold", dest="origin_hold", action="store_false")
parser.add_argument("--origin-hold-max-frames", type=int, default=30,
                    help="Drop held origin pose after this many frames without a fresh detection")
# Display / IO
parser.add_argument("--show-box-tags", action="store_true",
                    help="Show per-box-tag world position and per-tag candidates on fused panel")
parser.add_argument("--show-robot-tags", action="store_true", default=True)
parser.add_argument("--no-show-robot-tags", dest="show_robot_tags", action="store_false")
parser.add_argument("--show-axes", action="store_true", default=True,
                    help="Draw per-detection RGB axes on each camera image")
parser.add_argument("--no-show-axes", dest="show_axes", action="store_false")
parser.add_argument("--csv-out", type=str, default="",
                    help="CSV output path. If non-empty, a timestamp suffix "
                         "(_YYYYMMDD_HHMMSS) is auto-inserted before the extension "
                         "to prevent overwrite. Use --csv-no-timestamp to keep the "
                         "exact path.")
parser.add_argument("--csv-no-timestamp", action="store_true",
                    help="Do NOT auto-append _YYYYMMDD_HHMMSS to --csv-out filename")
parser.add_argument("--print-every", type=int, default=30)
parser.add_argument("--debug-margins", action="store_true",
                    help="On every frame, print per-camera per-tag decision_margin "
                         "and whether it passed --margin-min. Helps debug e.g. pelvis "
                         "tags being filtered.")
# Stream
parser.add_argument("--tag-size", type=float, default=0.077)
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
# Detector tuning (loop bottleneck — at 960x540 with quad_decimate=1.0 and
# 3 cameras the loop runs ~4-5 fps; quad_decimate=2.0 typically yields 3-4x
# speedup with negligible margin loss for tag sizes >= ~7 cm at our distances)
parser.add_argument("--detector-quad-decimate", type=float, default=2.0,
                    help="apriltag quad_decimate (1.0=full res, 2.0=half res for "
                         "detection. Bigger = faster but smaller tags drop out.)")
parser.add_argument("--detector-nthreads", type=int, default=9,
                    help="Total apriltag thread budget. When --detect-parallel is ON "
                         "(default), this is divided across N parallel detectors so "
                         "total threads = nthreads. Sweet spot for 14-core / 20-thread "
                         "CPUs (e.g. i5-13500HX) is 8-9 (3 cam x 3 threads). Going "
                         "higher (12+) tends to hurt due to E-core / cache contention.")
parser.add_argument("--detector-decode-sharpening", type=float, default=0.25)
parser.add_argument("--detect-parallel", action="store_true", default=True,
                    help="Run apriltag detection on all cameras concurrently using "
                         "Python threads. pupil_apriltags releases the GIL during "
                         "detection so true parallelism is achieved. Default: ON.")
parser.add_argument("--no-detect-parallel", dest="detect_parallel", action="store_false",
                    help="Disable parallel detection (run cameras serially). Useful "
                         "for benchmarking or low-core CPUs.")
parser.add_argument("--no-show-cam-windows", action="store_true",
                    help="Skip cv2.imshow for per-camera windows (saves a few ms/frame). "
                         "FUSED panel still shown.")
# UDP publisher for the tag-history policy. When enabled, emits a single
# ASCII packet per loop with current torso & box poses (lab frame).
# Wire format matches deploy/include/camera_pose_subscriber.h.
#
# Default target = 127.0.0.1 because the supported deploy mode is
# **PC-only**: g1_ctrl runs on the same machine as this tracker, talking
# to the robot over a wired ethernet link to 192.168.123.{15,18}. UDP
# loopback gives <1ms latency and zero packet loss, so the staleness
# logic is essentially a no-op.
#
# To run g1_ctrl on the Jetson instead (legacy split-machine mode),
# pass `--udp-host 192.168.123.164`.
parser.add_argument("--udp-publish", action="store_true",
                    help="Publish each loop's torso+box pose to a UDP target. "
                         "Required for sub8_45_tag_history policy.")
parser.add_argument("--udp-host", type=str, default="127.0.0.1",
                    help="Destination for UDP pose packets. "
                         "Default 127.0.0.1 (g1_ctrl on this PC). "
                         "Use 192.168.123.164 to send to the Jetson.")
parser.add_argument("--udp-port", type=int, default=9999,
                    help="UDP port matching CAMERA_POSE_PORT in deploy.")
args = parser.parse_args()


# =========================================================
# Setup
# =========================================================

K1, cam1_params = load_camera_params(args.cam1_calib)
K2, cam2_params = load_camera_params(args.cam2_calib)
use_cam3 = bool(args.cam3_serial.strip())
if use_cam3:
    K3, cam3_params = load_camera_params(args.cam3_calib)
else:
    K3, cam3_params = None, None

# Anchor transforms: T_origin_anchor[anchor_id] = pose of anchor tag in origin frame.
ANCHOR_IDS = parse_int_list(args.anchor_ids)
ANCHOR_ID_SET = set(ANCHOR_IDS)
anchor_global = load_anchor_transforms(args.anchor_config, args.origin_id)
print(f"[anchor] loaded {len(anchor_global)} anchor(s) from {args.anchor_config}: {sorted(anchor_global.keys())}")
for aid in ANCHOR_IDS:
    if aid not in anchor_global:
        print(f"[anchor] WARN id {aid} listed in --anchor-ids but missing from config; "
              "it will only become usable after a camera sees both id={origin} and id={aid} simultaneously.")

# Tag sizes
cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

# Robot tag ids
ROBOT_HEAD_TAG_ID = args.head_tag_id
PELVIS_TAG_IDS = [int(x) for x in args.pelvis_tag_ids.split(",") if x.strip()]
PELVIS_TAG_SET = set(PELVIS_TAG_IDS)

# Box tag map
map_data = np.load(args.box_tag_map)
BOX_T_TAG = {int(tid): T for tid, T in zip(map_data["tag_ids"], map_data["box_T_tags"])}
BOX_TAG_IDS = sorted(BOX_T_TAG.keys())
BOX_TAG_SET = set(BOX_TAG_IDS)

# Box OBB→body correction (see comment on --box-inertial-quat).
# T_obb_to_body is a pure rotation (translation 0) applied as right-multiply
# on the world->box pose: T_world_box_mjlab = T_world_box_obb @ T_obb_to_body.
def _parse_box_inertial_quat(s):
    s = s.strip().lower()
    if s in ("", "identity", "none", "off", "0"):
        return np.eye(4)
    parts = s.replace(",", " ").split()
    if len(parts) != 4:
        raise SystemExit(f"--box-inertial-quat expects 4 numbers (wxyz), got: {s}")
    w, x, y, z = (float(p) for p in parts)
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n < 1e-9:
        raise SystemExit("--box-inertial-quat has zero norm")
    w, x, y, z = w/n, x/n, y/n, z/n
    # quat (wxyz) -> R(quat)
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
    # We need the INVERSE (body -> obb is the inertial quat; we want
    # obb -> body which is the transpose of a rotation matrix).
    T = np.eye(4)
    T[:3, :3] = R.T
    return T

T_OBB_TO_BODY = _parse_box_inertial_quat(args.box_inertial_quat)
_box_correction_active = not np.allclose(T_OBB_TO_BODY[:3, :3], np.eye(3), atol=1e-6)
if _box_correction_active:
    print(f"[box-frame] applying mesh-tilt correction "
          f"(inertial quat='{args.box_inertial_quat}'). "
          "Tracker output will be in mjlab body frame, matching the npz.")
else:
    print("[box-frame] mesh-tilt correction DISABLED — tracker output is in "
          "raw OBB frame from box_tag_map. The npz frame may differ by "
          "~46° if mjlab was used for training. Pass --box-inertial-quat to fix.")

# Head tag → torso calibration
T_tag_torso = None
if args.head_tag_calib and os.path.exists(args.head_tag_calib):
    T_tag_torso = np.load(args.head_tag_calib)["T_tag_torso"]
    print(f"[multicam] Loaded T_tag_torso (translation: {T_tag_torso[:3, 3]})")
else:
    print(f"[multicam] {args.head_tag_calib} not found; using head_z_offset={args.head_z_offset:+.3f} m")

# Detector(s)
# When detect-parallel is on, we instantiate one Detector per camera (the
# C-level apriltag context is NOT thread-safe to share) and split nthreads
# evenly so total_threads = args.detector_nthreads (fair vs serial baseline).
n_active_cams = 3 if use_cam3 else 2
if args.detect_parallel:
    nthreads_per_det = max(1, int(args.detector_nthreads) // n_active_cams)
else:
    nthreads_per_det = int(args.detector_nthreads)


def _make_detector():
    return Detector(
        families="tag36h11",
        nthreads=nthreads_per_det,
        quad_decimate=float(args.detector_quad_decimate),
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=float(args.detector_decode_sharpening),
        debug=False,
    )


# Per-camera detector: { "cam1": Detector, "cam2": Detector, "cam3": Detector? }
# In serial mode all three could share one, but having N independent detectors
# with the same total thread budget costs nothing and lets us toggle parallel
# at runtime without re-init.
cam_detectors = {
    "cam1": _make_detector(),
    "cam2": _make_detector(),
    "cam3": _make_detector() if use_cam3 else None,
}
# Back-compat alias for any old code that referenced `detector` directly.
detector = cam_detectors["cam1"]
print(f"[detector] quad_decimate={args.detector_quad_decimate}  "
      f"nthreads={args.detector_nthreads} (per-detector={nthreads_per_det})  "
      f"decode_sharpening={args.detector_decode_sharpening}  "
      f"parallel={'ON' if args.detect_parallel else 'OFF'}  "
      f"n_cams={n_active_cams}")

# Pipelines
def open_pipeline(serial):
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    p.start(cfg)
    return p


pipeline1 = open_pipeline(args.cam1_serial)
pipeline2 = open_pipeline(args.cam2_serial)
pipeline3 = open_pipeline(args.cam3_serial) if use_cam3 else None

print("\n======================================================")
print("Multicam shadow logger (robot torso + box)")
print(f"cams: cam1={args.cam1_serial}  cam2={args.cam2_serial}"
      + (f"  cam3={args.cam3_serial}" if use_cam3 else ""))
print(f"origin_id={args.origin_id}  anchor_ids={ANCHOR_IDS}  margin_min={args.margin_min}")
print(f"head_tag_id={ROBOT_HEAD_TAG_ID}  pelvis_tag_ids={PELVIS_TAG_IDS}  box_tags={BOX_TAG_IDS}")
print(f"head_z_offset={args.head_z_offset:+.3f}  pelvis_to_root_z={args.pelvis_to_root_z:+.3f}  "
      f"root_to_torso_z={args.root_to_torso_z:+.3f}")
print(f"origin_hold={args.origin_hold} (max {args.origin_hold_max_frames} frames)")
print(f"csv_out={args.csv_out or '(off)'}  print_every={args.print_every}")
print(f"stream={args.width}x{args.height}@{args.fps}  axes={'on' if args.show_axes else 'off'}")
print("======================================================\n")

# CSV
CSV_COLUMNS = [
    "frame_idx", "t_sec",
    # origin tracking per camera (origin_source: "direct" / "anchor:<id>" / "hold" / "none")
    "cam1_origin_source", "cam2_origin_source", "cam3_origin_source",
    "cam1_origin_held_frames", "cam2_origin_held_frames", "cam3_origin_held_frames",
    "n_direct_cams", "n_anchor_cams", "n_held_cams", "n_total_tags",
    # robot
    "head_visible", "head_pos_x", "head_pos_y", "head_pos_z",
    "head_quat_w", "head_quat_x", "head_quat_y", "head_quat_z", "head_n_cams",
    "pelvis_visible", "pelvis_used_id",
    "pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z",
    "pelvis_quat_w", "pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z", "pelvis_n_cams",
    "root_pos_x", "root_pos_y", "root_pos_z",
    "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "torso_head_pos_x", "torso_head_pos_y", "torso_head_pos_z",
    "torso_head_quat_w", "torso_head_quat_x", "torso_head_quat_y", "torso_head_quat_z",
    "torso_pelvis_pos_x", "torso_pelvis_pos_y", "torso_pelvis_pos_z",
    "torso_pelvis_quat_w", "torso_pelvis_quat_x", "torso_pelvis_quat_y", "torso_pelvis_quat_z",
    "torso_pos_x", "torso_pos_y", "torso_pos_z",
    "torso_quat_w", "torso_quat_x", "torso_quat_y", "torso_quat_z",
    "torso_path",
    # box
    "box_visible", "box_n_tags",
    "obj_pos_x", "obj_pos_y", "obj_pos_z",
    "obj_quat_w", "obj_quat_x", "obj_quat_y", "obj_quat_z",
    # ----- ACTOR-OBS-READY DERIVED FIELDS (lab/origin frame; no T_ref_lab applied) -----
    # 6D rotation (Zhou et al. 2019, first two columns of R, column-major)
    "torso_rot6d_0", "torso_rot6d_1", "torso_rot6d_2",
    "torso_rot6d_3", "torso_rot6d_4", "torso_rot6d_5",
    "obj_rot6d_0", "obj_rot6d_1", "obj_rot6d_2",
    "obj_rot6d_3", "obj_rot6d_4", "obj_rot6d_5",
    # object expressed in torso frame (= what `object_pos_torso` / `object_ori6_torso` see)
    "obj_in_torso_pos_x", "obj_in_torso_pos_y", "obj_in_torso_pos_z",
    "obj_in_torso_rot6d_0", "obj_in_torso_rot6d_1", "obj_in_torso_rot6d_2",
    "obj_in_torso_rot6d_3", "obj_in_torso_rot6d_4", "obj_in_torso_rot6d_5",
    # quick alignment debug
    "torso_yaw_rad", "obj_yaw_rad",
]
csv_writer = None
csv_file = None
csv_path_resolved = ""
if args.csv_out:
    csv_path_resolved = args.csv_out
    if not args.csv_no_timestamp:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path(args.csv_out)
        # Insert timestamp before the extension; keep parent dir.
        csv_path_resolved = str(p.with_name(f"{p.stem}_{ts}{p.suffix or '.csv'}"))
    os.makedirs(os.path.dirname(csv_path_resolved) or ".", exist_ok=True)
    if os.path.exists(csv_path_resolved):
        # Extra safety: if user asked to keep exact name AND file exists, don't clobber.
        raise FileExistsError(
            f"Refusing to overwrite existing CSV: {csv_path_resolved}. "
            "Remove it or drop --csv-no-timestamp."
        )
    csv_file = open(csv_path_resolved, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()
    print(f"[multicam] CSV logging -> {csv_path_resolved}")


# UDP publisher (for deploy-side camera_pose_subscriber). One ASCII line per
# loop. Format matches the sscanf in camera_pose_subscriber.h:
#   "<ts_ns> <torso_v> <tx> <ty> <tz> <tqw> <tqx> <tqy> <tqz>
#    <box_v>   <bx> <by> <bz> <bqw> <bqx> <bqy> <bqz>\n"
udp_sock = None
udp_target = None
udp_packet_count = 0
if args.udp_publish:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = (args.udp_host, args.udp_port)
    print(f"[multicam] UDP publish -> {args.udp_host}:{args.udp_port}")


def udp_publish_pose(world_T_torso, world_T_box):
    """Send the latest fused torso+box poses to the Jetson over UDP.
    world_T_* may be None if the pose isn't available this frame; in that
    case the corresponding `valid` flag is set to 0 and pose fields are
    zero-filled (the deploy-side reader treats invalid poses as 'use last').
    """
    global udp_packet_count
    if udp_sock is None:
        return

    def _flatten(T):
        if T is None:
            return 0, [0.0]*3, [1.0, 0.0, 0.0, 0.0]
        p = T[:3, 3].astype(float).tolist()
        q = rotation_to_quat_wxyz(T[:3, :3]).astype(float).tolist()
        return 1, p, q

    t_v, t_p, t_q = _flatten(world_T_torso)
    b_v, b_p, b_q = _flatten(world_T_box)
    ts_ns = time.time_ns()
    msg = (
        f"{ts_ns} {t_v} "
        f"{t_p[0]:.6f} {t_p[1]:.6f} {t_p[2]:.6f} "
        f"{t_q[0]:.6f} {t_q[1]:.6f} {t_q[2]:.6f} {t_q[3]:.6f} "
        f"{b_v} "
        f"{b_p[0]:.6f} {b_p[1]:.6f} {b_p[2]:.6f} "
        f"{b_q[0]:.6f} {b_q[1]:.6f} {b_q[2]:.6f} {b_q[3]:.6f}\n"
    )
    try:
        udp_sock.sendto(msg.encode("ascii"), udp_target)
        udp_packet_count += 1
    except OSError as e:
        # Network blip — log once per ~1k frames, don't crash the tracker.
        if udp_packet_count % 1000 == 0:
            print(f"[multicam] UDP send failed: {e}")


def csv_blank():
    return {k: "" for k in CSV_COLUMNS}


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


def fill_actor_obs_fields(world_T_torso, world_T_box):
    """Compute actor-style derived fields (lab/origin frame).

    Mirrors mjlab/IsaacLab observation funcs:
      * `*_rot6d_*` = first two columns of rotation matrix (Zhou et al. 2019)
      * `obj_in_torso_*` = box expressed in torso frame
        (== `object_pos_torso` / `object_ori6_torso` at deploy time)

    These are still in LAB/ORIGIN frame; npz reference alignment (T_ref_lab)
    is applied later in the obs builder.
    """
    out = {}
    if world_T_torso is not None:
        rot6 = rotation_to_rot6d(world_T_torso[:3, :3])
        for i, v in enumerate(rot6):
            out[f"torso_rot6d_{i}"] = float(v)
        out["torso_yaw_rad"] = quat_wxyz_to_yaw(rotation_to_quat_wxyz(world_T_torso[:3, :3]))
    if world_T_box is not None:
        rot6 = rotation_to_rot6d(world_T_box[:3, :3])
        for i, v in enumerate(rot6):
            out[f"obj_rot6d_{i}"] = float(v)
        out["obj_yaw_rad"] = quat_wxyz_to_yaw(rotation_to_quat_wxyz(world_T_box[:3, :3]))
    if world_T_torso is not None and world_T_box is not None:
        T_torso_obj = np.linalg.inv(world_T_torso) @ world_T_box
        p = T_torso_obj[:3, 3]
        out["obj_in_torso_pos_x"] = float(p[0])
        out["obj_in_torso_pos_y"] = float(p[1])
        out["obj_in_torso_pos_z"] = float(p[2])
        rot6 = rotation_to_rot6d(T_torso_obj[:3, :3])
        for i, v in enumerate(rot6):
            out[f"obj_in_torso_rot6d_{i}"] = float(v)
    return out


# =========================================================
# Main loop
# =========================================================

# Per-camera state. T_origin_anchor_self[aid] starts at the global JSON value (if any)
# and is overwritten with the camera's own observation whenever it sees both id={origin}
# and id={aid} simultaneously, giving each camera its own bias-absorbing copy.
def _init_cam(name, K, params):
    return {
        "name": name, "K": K, "params": params, "img": None,
        "T_cN_origin_held": None, "held_frames": 0,
        "origin_source": "none",   # "direct" / "anchor:<id>" / "hold" / "none"
        "T_origin_anchor_self": {aid: anchor_global.get(aid).copy() if aid in anchor_global else None
                                 for aid in ANCHOR_IDS},
        "anchor_self_n_obs": {aid: 0 for aid in ANCHOR_IDS},
    }

cam_state = {
    "cam1": _init_cam("cam1", K1, cam1_params),
    "cam2": _init_cam("cam2", K2, cam2_params),
    "cam3": _init_cam("cam3", K3, cam3_params) if use_cam3 else _init_cam("cam3", None, None),
}

# Per-tag-id margin statistics: tag_id -> dict with counters/sums.
# Used to print a summary at end so we can spot tags whose margins routinely fall
# below --margin-min and get filtered out of fusion despite being detected.
margin_stats = {}
margin_stats_lock = threading.Lock()


def _record_margin(tag_id, margin):
    with margin_stats_lock:
        s = margin_stats.setdefault(tag_id, {"n_seen": 0, "n_pass": 0, "sum": 0.0,
                                             "min": float("inf"), "max": -float("inf")})
        s["n_seen"] += 1
        s["sum"]    += float(margin)
        s["min"]     = min(s["min"], float(margin))
        s["max"]     = max(s["max"], float(margin))
        if margin >= args.margin_min:
            s["n_pass"] += 1


def grab(pipeline):
    return np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())


def detect_one_camera(name, img, draw_color):
    """Run detector on this camera's frame, draw boxes + (optional) axes,
    return dict tag_id -> (T_cN_tag, margin).

    Thread-safe when called concurrently for different `name`s, because:
      * Each camera has its OWN Detector instance (cam_detectors[name])
      * cv2 ops on `img` write only to that camera's buffer
      * margin_stats updates go through margin_stats_lock
    """
    det_inst = cam_detectors[name]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detect_with_tag_sizes(det_inst, gray, cam_state[name]["params"], TAG_SIZE, TAG_SIZE_MAP)
    out = {}
    for det in dets:
        T_cN_tag = pose_to_T(det.pose_R, det.pose_t)
        margin = float(getattr(det, "decision_margin", 0.0))
        out[det.tag_id] = (T_cN_tag, margin)
        _record_margin(det.tag_id, margin)
        # Highlight robot/box/origin specifically.
        if det.tag_id == args.origin_id:
            color = (0, 255, 0)
            label = f"ORIGIN id{det.tag_id}"
        elif det.tag_id == ROBOT_HEAD_TAG_ID:
            color = (255, 0, 255)
            label = f"HEAD id{det.tag_id}"
        elif det.tag_id in PELVIS_TAG_SET:
            color = (0, 200, 255)
            label = f"PELVIS id{det.tag_id}"
        elif det.tag_id in BOX_TAG_SET:
            color = (0, 255, 255)
            label = f"BOX id{det.tag_id}"
        else:
            color = draw_color
            label = f"id{det.tag_id}"
        draw_detection(img, det, color, label)
        if args.show_axes:
            draw_axes_cam(img, cam_state[name]["K"], T_cN_tag, length=0.06)
    return out


t_start = time.time()
frame_idx = 0
# Stage timing accumulators (printed every --print-every frames)
stage_acc = {"grab": 0.0, "detect": 0.0, "fuse": 0.0, "gui": 0.0, "csv": 0.0, "total": 0.0}
stage_n = 0
# Pose-smoothing state: latest smoothed 4x4 for torso / box, or None until the
# first valid measurement. Updated in-place after fusion when --ema-alpha-* < 1.
# When alpha >= 1 (default) this dict is never touched, so smoothing is a no-op.
pose_ema_state = {"torso": None, "box": None}
if args.ema_alpha_torso < 1.0 or args.ema_alpha_box < 1.0:
    print(f"[multicam] pose smoothing: alpha_torso={args.ema_alpha_torso:.2f}  "
          f"alpha_box={args.ema_alpha_box:.2f}  (1.0=raw)")

# Thread pool for parallel per-camera detection. pupil_apriltags releases
# the GIL during detection so Python threads achieve true CPU parallelism.
# Pool is built ONCE outside the loop to avoid per-frame thread spawn cost.
detect_executor = (
    ThreadPoolExecutor(max_workers=n_active_cams,
                       thread_name_prefix="detect")
    if args.detect_parallel else None
)

try:
    while True:
        frame_idx += 1
        _t_loop = time.time()
        _t_stage = time.time()
        cam_state["cam1"]["img"] = grab(pipeline1)
        cam_state["cam2"]["img"] = grab(pipeline2)
        if use_cam3:
            cam_state["cam3"]["img"] = grab(pipeline3)
        stage_acc["grab"] += time.time() - _t_stage

        _t_stage = time.time()
        if detect_executor is not None:
            # Submit all cameras concurrently; wait for all results.
            f1 = detect_executor.submit(detect_one_camera, "cam1",
                                        cam_state["cam1"]["img"], (200, 200, 200))
            f2 = detect_executor.submit(detect_one_camera, "cam2",
                                        cam_state["cam2"]["img"], (200, 200, 200))
            if use_cam3:
                f3 = detect_executor.submit(detect_one_camera, "cam3",
                                            cam_state["cam3"]["img"], (200, 200, 200))
            det_cam1 = f1.result()
            det_cam2 = f2.result()
            det_cam3 = f3.result() if use_cam3 else {}
        else:
            det_cam1 = detect_one_camera("cam1", cam_state["cam1"]["img"], (200, 200, 200))
            det_cam2 = detect_one_camera("cam2", cam_state["cam2"]["img"], (200, 200, 200))
            det_cam3 = detect_one_camera("cam3", cam_state["cam3"]["img"], (200, 200, 200)) if use_cam3 else {}
        stage_acc["detect"] += time.time() - _t_stage

        if args.debug_margins:
            parts = []
            for nm, dd in [("c1", det_cam1), ("c2", det_cam2), ("c3", det_cam3)]:
                if nm == "c3" and not use_cam3:
                    continue
                segs = []
                for tid, (_T, m) in sorted(dd.items()):
                    flag = "P" if m >= args.margin_min else "f"
                    role = ("O" if tid == args.origin_id
                            else "H" if tid == ROBOT_HEAD_TAG_ID
                            else "L" if tid in PELVIS_TAG_SET
                            else "B" if tid in BOX_TAG_SET
                            else ".")
                    segs.append(f"{tid}{role}{m:.0f}{flag}")
                parts.append(f"{nm}=[{','.join(segs) or '-'}]")
            print(f"[margins f{frame_idx:5d}] " + "  ".join(parts))

        # ---- Step 1: Refresh per-camera anchor calibration when both id={origin}
        #              and id={anchor} are visible in the same camera frame. ----
        for name, det_dict in [("cam1", det_cam1), ("cam2", det_cam2), ("cam3", det_cam3)]:
            if name == "cam3" and not use_cam3:
                continue
            st = cam_state[name]
            if args.origin_id not in det_dict:
                continue
            T_cN_origin, m_o = det_dict[args.origin_id]
            if m_o < args.margin_min:
                continue
            inv_origin = np.linalg.inv(T_cN_origin)
            for aid in ANCHOR_IDS:
                if aid not in det_dict:
                    continue
                T_cN_anchor, m_a = det_dict[aid]
                if m_a < args.margin_min:
                    continue
                # T_origin_anchor = pose of anchor tag in origin frame, derived this frame.
                st["T_origin_anchor_self"][aid] = inv_origin @ T_cN_anchor
                st["anchor_self_n_obs"][aid] = st["anchor_self_n_obs"].get(aid, 0) + 1

        _t_stage = time.time()
        # ---- Step 2: Establish T_camN_origin per camera with priority ----
        #   1) DIRECT: tag {origin} visible
        #   2) VIA_ANCHOR: tag {anchor} visible AND T_origin_anchor available
        #   3) HELD: last good T_camN_origin within hold window
        per_cam_origin_T = {}
        per_cam_origin_w = {}
        per_cam_origin_src = {}    # name -> "direct" / "anchor:<id>" / "hold" / "none"

        def get_self_or_global_anchor(st, aid):
            T = st["T_origin_anchor_self"].get(aid)
            if T is not None:
                return T
            return anchor_global.get(aid)

        for name, det_dict in [("cam1", det_cam1), ("cam2", det_cam2), ("cam3", det_cam3)]:
            if name == "cam3" and not use_cam3:
                continue
            st = cam_state[name]
            assigned = False
            # --- 1) DIRECT
            if args.origin_id in det_dict:
                T_cN_origin, m_o = det_dict[args.origin_id]
                if m_o >= args.margin_min:
                    st["T_cN_origin_held"] = T_cN_origin.copy()
                    st["held_frames"] = 0
                    st["origin_source"] = "direct"
                    per_cam_origin_T[name] = T_cN_origin
                    per_cam_origin_w[name] = m_o
                    per_cam_origin_src[name] = "direct"
                    assigned = True

            # --- 2) VIA_ANCHOR
            if not assigned:
                for aid in ANCHOR_IDS:
                    if aid not in det_dict:
                        continue
                    T_cN_anchor, m_a = det_dict[aid]
                    if m_a < args.margin_min:
                        continue
                    T_origin_anchor = get_self_or_global_anchor(st, aid)
                    if T_origin_anchor is None:
                        continue
                    T_cN_origin = T_cN_anchor @ np.linalg.inv(T_origin_anchor)
                    st["T_cN_origin_held"] = T_cN_origin.copy()
                    st["held_frames"] = 0
                    st["origin_source"] = f"anchor:{aid}"
                    per_cam_origin_T[name] = T_cN_origin
                    # weight: anchor margin clipped, but slightly de-rated so DIRECT cameras win
                    per_cam_origin_w[name] = max(m_a * 0.8, 1.0)
                    per_cam_origin_src[name] = f"anchor:{aid}"
                    assigned = True
                    break

            # --- 3) HELD
            if not assigned:
                if args.origin_hold and st["T_cN_origin_held"] is not None:
                    st["held_frames"] += 1
                    if st["held_frames"] <= args.origin_hold_max_frames:
                        per_cam_origin_T[name] = st["T_cN_origin_held"]
                        per_cam_origin_w[name] = max(args.margin_min * 0.4, 1.0)
                        per_cam_origin_src[name] = "hold"
                        st["origin_source"] = "hold"
                        assigned = True
                if not assigned:
                    st["origin_source"] = "none"

        # ---- Step 3: DIRECT-style candidates from every camera that has T_cN_origin ----
        rel_candidates = {}     # tag_id -> list of {T, w, src}
        cam_n_for_tag = {}      # tag_id -> set of camera names contributing
        cam_src_count = {"direct": 0, "anchor": 0, "hold": 0}

        for name, det_dict in [("cam1", det_cam1), ("cam2", det_cam2), ("cam3", det_cam3)]:
            if name == "cam3" and not use_cam3:
                continue
            if name not in per_cam_origin_T:
                continue
            inv_origin_cN = np.linalg.inv(per_cam_origin_T[name])
            origin_w = per_cam_origin_w[name]
            src_label = per_cam_origin_src[name]
            cam_src_count_key = "anchor" if src_label.startswith("anchor") else src_label
            cam_src_count[cam_src_count_key] = cam_src_count.get(cam_src_count_key, 0) + 1
            for tag_id, (T_cN_tag, m_tag) in det_dict.items():
                if tag_id == args.origin_id:
                    continue
                if m_tag < args.margin_min:
                    continue
                rel_T = inv_origin_cN @ T_cN_tag
                w = max(min(origin_w, m_tag), 1e-3)
                rel_candidates.setdefault(tag_id, []).append({
                    "T": rel_T, "w": w, "src": f"{name}:{src_label}",
                })
                cam_n_for_tag.setdefault(tag_id, set()).add(name)

        # ---- Step 4: Final fusion per tag ----
        rel_fused = {}
        rel_n_cams = {}
        rel_srcs = {}
        for tag_id, cands in rel_candidates.items():
            T_fused, srcs = fuse_poses(cands)
            rel_fused[tag_id] = T_fused
            rel_n_cams[tag_id] = len(cam_n_for_tag.get(tag_id, set()))
            rel_srcs[tag_id] = srcs

        n_total_tags = len(rel_fused)
        n_via_anchor_cams = cam_src_count.get("anchor", 0)
        n_held_cams = cam_src_count.get("hold", 0)
        n_direct_cams = cam_src_count.get("direct", 0)

        # ---- Robot poses (origin frame) ----
        world_T_headtag = rel_fused.get(ROBOT_HEAD_TAG_ID)
        head_n_cams = rel_n_cams.get(ROBOT_HEAD_TAG_ID, 0) if world_T_headtag is not None else 0

        pelvis_visible_ids = [tid for tid in PELVIS_TAG_IDS if tid in rel_fused]
        world_T_pelvis = None
        pelvis_used_id = -1
        pelvis_n_cams = 0
        if pelvis_visible_ids:
            # Pick the one whose underlying detection had the highest summed margin across cameras.
            best_id = max(pelvis_visible_ids,
                          key=lambda tid: sum(c["w"] for c in rel_candidates.get(tid, [])))
            pelvis_used_id = int(best_id)
            world_T_pelvis = rel_fused[best_id]
            pelvis_n_cams = rel_n_cams.get(best_id, 0)

        # head -> torso_head
        world_T_torso_head = None
        if world_T_headtag is not None:
            if T_tag_torso is not None:
                # Calibrated rigid transform: orientation tilt is automatically respected.
                world_T_torso_head = world_T_headtag @ T_tag_torso
            elif args.head_up_mode == "tag-axis":
                # Move along the head tag's LOCAL body-down axis. Tracks head tilt,
                # so when the robot bends and the head pitches forward, the inferred
                # torso position swings along the body axis instead of along world-Z.
                world_T_torso_head = local_axis_translate(
                    world_T_headtag, args.head_tag_down_axis, args.head_to_torso_body
                )
            else:
                # Legacy world-z fallback: only correct when head is upright.
                world_T_torso_head = world_z_translate(world_T_headtag, args.head_z_offset)

        # pelvis -> root -> torso_pelvis
        world_T_root = None
        world_T_torso_pelvis = None
        if world_T_pelvis is not None:
            if args.pelvis_up_mode == "tag-axis":
                # Move along the pelvis tag's local body-up axis. This is robust
                # when the body is tilted/bent: --pelvis-tag-up-axis points
                # body-up regardless of world orientation.
                world_T_root = local_axis_translate(
                    world_T_pelvis, args.pelvis_tag_up_axis, args.pelvis_to_root_body
                )
                world_T_torso_pelvis = local_axis_translate(
                    world_T_root, args.pelvis_tag_up_axis, args.root_to_torso_body
                )
            else:
                # Legacy world-z mode: only correct when robot is upright.
                world_T_root = world_z_translate(world_T_pelvis, args.pelvis_to_root_z)
                world_T_torso_pelvis = world_z_translate(world_T_root, args.root_to_torso_z)

        # fused torso (avg position; rotation from head if available else pelvis)
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

        # ---- Box pose ----
        per_tag_box_candidate = {}
        positions, rotations, weights = [], [], []
        for tag_id in BOX_TAG_IDS:
            T_world_tag = rel_fused.get(tag_id)
            if T_world_tag is None or tag_id not in BOX_T_TAG:
                continue
            T_world_box = T_world_tag @ np.linalg.inv(BOX_T_TAG[tag_id])
            per_tag_box_candidate[tag_id] = T_world_box
            positions.append(T_world_box[:3, 3])
            rotations.append(T_world_box[:3, :3])
            # use sum of underlying margins as weight
            weights.append(sum(c["w"] for c in rel_candidates.get(tag_id, [])))

        world_T_box = None
        if positions:
            w = np.asarray(weights, dtype=float)
            w_sum = w.sum() if w.sum() > 1e-9 else float(len(w))
            w_norm = w / w_sum
            world_T_box = np.eye(4)
            world_T_box[:3, 3] = (np.stack(positions, axis=0) * w_norm[:, None]).sum(axis=0)
            world_T_box[:3, :3] = average_rotation(rotations, w)

            # Right-multiply by T_OBB_TO_BODY to convert from box_tag_map's
            # OBB axes (= real-face normals) to mjlab body-frame axes (mesh
            # AABB axes), so the published quat matches the npz convention.
            # Translation is unaffected because T_OBB_TO_BODY has zero
            # translation (inertial pos in largebox.xml is ~µm, ignored).
            if _box_correction_active:
                world_T_box = world_T_box @ T_OBB_TO_BODY

        # ---- Pose smoothing (EMA + SVD-mean SLERP) ----
        # Applied AFTER fusion, BEFORE downstream consumers. Carries over the
        # previous estimate when this frame's measurement is None (e.g. box
        # tag dropout) so GUI/CSV/UDP keep getting a stable readout.
        if args.ema_alpha_torso < 1.0:
            world_T_torso = ema_pose(
                pose_ema_state["torso"], world_T_torso, args.ema_alpha_torso
            )
            if world_T_torso is not None:
                pose_ema_state["torso"] = world_T_torso
            # Keep `gui_torso` consistent: re-resolve from the chosen source
            # AFTER smoothing so the GUI shows the same pose as CSV/UDP.
            if args.torso_source == "fused":
                gui_torso = world_T_torso
        if args.ema_alpha_box < 1.0:
            world_T_box = ema_pose(
                pose_ema_state["box"], world_T_box, args.ema_alpha_box
            )
            if world_T_box is not None:
                pose_ema_state["box"] = world_T_box

        stage_acc["fuse"] += time.time() - _t_stage

        # ---- UDP publish (timed under "csv" stage to avoid GUI conflation) ----
        # Sent BEFORE GUI so deploy gets the freshest pose with minimal
        # added latency (GUI can spike to ~10ms).
        if udp_sock is not None:
            udp_publish_pose(world_T_torso, world_T_box)

        _t_stage = time.time()
        # ---- GUI: per-camera windows and fused panel ----
        if not args.no_show_cam_windows:
            for name, win in [("cam1", "cam1"), ("cam2", "cam2"), ("cam3", "cam3")]:
                if name == "cam3" and not use_cam3:
                    continue
                img = cam_state[name]["img"]
                st = cam_state[name]
                src = st["origin_source"]
                if src == "direct":
                    ostr, ocol = f"ORIGIN id{args.origin_id}: DIRECT", (0, 255, 0)
                elif src.startswith("anchor:"):
                    aid = src.split(":", 1)[1]
                    ostr, ocol = f"ORIGIN via ANCHOR id{aid}", (0, 255, 200)
                elif src == "hold":
                    ostr = f"ORIGIN id{args.origin_id}: HOLD ({st['held_frames']}f)"
                    ocol = (0, 200, 255)
                else:
                    ostr = f"ORIGIN id{args.origin_id}: NOT SEEN"
                    ocol = (0, 0, 255)
                cv2.putText(img, ostr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ocol, 2)
                cv2.imshow(win, img)

        # Fused panel
        panel_h = 600
        panel = np.zeros((panel_h, 880, 3), dtype=np.uint8)
        cv2.putText(panel, "Fused (origin-tag) frame    [direct + anchor-fallback, no extrinsic]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        def _cam_glyph(n):
            src = cam_state[n]["origin_source"]
            if src == "direct":
                return f"{n}=D"
            if src.startswith("anchor:"):
                aid = src.split(":", 1)[1]
                return f"{n}=A{aid}"
            if src == "hold":
                return f"{n}=H{cam_state[n]['held_frames']}"
            return f"{n}=X"

        cam_names = ["cam1", "cam2"] + (["cam3"] if use_cam3 else [])
        cv2.putText(panel,
                    f"frame={frame_idx}  cams: "
                    + " ".join(_cam_glyph(n) for n in cam_names)
                    + f"   src(D/A/H)={n_direct_cams}/{n_via_anchor_cams}/{n_held_cams}"
                    + f"  fused_tags={n_total_tags}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        py = 80

        def panel_line(label, T, color, extra=""):
            global py
            if T is None:
                txt = f"{label}: --"
            else:
                p = T[:3, 3]
                q = rotation_to_quat_wxyz(T[:3, :3])
                txt = (f"{label}: pos=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]  "
                       f"q(wxyz)=[{q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f},{q[3]:+.2f}]")
                if extra:
                    txt += "  " + extra
            cv2.putText(panel, txt, (10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            py += 18

        if args.show_robot_tags:
            cv2.putText(panel, "Robot pose math (z-sign: +z=down, -z=up)",
                        (10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 255), 2)
            py += 22
            head_src = "head_calib(npz)" if T_tag_torso is not None else f"head + z*{args.head_z_offset:+.2f}"
            cv2.putText(panel, f"  torso_head = {head_src}", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            py += 16
            cv2.putText(panel, f"  root        = pelvis + z*{args.pelvis_to_root_z:+.2f}", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            py += 16
            cv2.putText(panel, f"  torso_pelvis= root + z*{args.root_to_torso_z:+.2f}", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            py += 22

            panel_line(f"head(id{ROBOT_HEAD_TAG_ID})", world_T_headtag, (255, 0, 255),
                       extra=f"n_cams={head_n_cams}" if world_T_headtag is not None else "")
            panel_line(f"pelvis(id{pelvis_used_id if pelvis_used_id>=0 else '-'})",
                       world_T_pelvis, (0, 200, 255),
                       extra=f"n_cams={pelvis_n_cams}" if world_T_pelvis is not None else "")
            panel_line("root         ", world_T_root, (0, 200, 255))
            panel_line("torso_head   ", world_T_torso_head, (255, 100, 255))
            panel_line("torso_pelvis ", world_T_torso_pelvis, (255, 100, 255))
            panel_line(f"torso({torso_path or '--'})", world_T_torso, (255, 0, 255))
            py += 6

        cv2.putText(panel, "Object", (10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        py += 22
        panel_line("box(fused)", world_T_box, (0, 255, 255),
                   extra=f"from {len(per_tag_box_candidate)} tags" if per_tag_box_candidate else "")
        if args.show_box_tags and per_tag_box_candidate:
            for tag_id in sorted(per_tag_box_candidate.keys()):
                panel_line(f"  via id{tag_id}", per_tag_box_candidate[tag_id], (180, 220, 220))

        cv2.imshow("FUSED (origin frame)", panel)

        # ---- Console (throttled) ----
        if (args.print_every > 0 and frame_idx % args.print_every == 0
                and gui_torso is not None and world_T_box is not None):
            tp = gui_torso[:3, 3]
            bp = world_T_box[:3, 3]
            rel = bp - tp
            print(f"[multicam {frame_idx:5d}] torso({torso_path})="
                  f"[{tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f}] "
                  f"box=[{bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f}] "
                  f"rel=[{rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f}] "
                  f"src(D/A/H)={n_direct_cams}/{n_via_anchor_cams}/{n_held_cams}")

        stage_acc["gui"] += time.time() - _t_stage

        _t_stage = time.time()
        # ---- CSV ----
        if csv_writer is not None:
            row = csv_blank()
            row.update({
                "frame_idx": frame_idx,
                "t_sec": time.time() - t_start,
                "cam1_origin_source": cam_state["cam1"]["origin_source"],
                "cam2_origin_source": cam_state["cam2"]["origin_source"],
                "cam3_origin_source": cam_state["cam3"]["origin_source"] if use_cam3 else "",
                "cam1_origin_held_frames": int(cam_state["cam1"]["held_frames"]),
                "cam2_origin_held_frames": int(cam_state["cam2"]["held_frames"]),
                "cam3_origin_held_frames": int(cam_state["cam3"]["held_frames"]) if use_cam3 else "",
                "n_direct_cams": int(n_direct_cams),
                "n_anchor_cams": int(n_via_anchor_cams),
                "n_held_cams": int(n_held_cams),
                "n_total_tags": int(n_total_tags),
                "head_visible": int(world_T_headtag is not None),
                "head_n_cams": head_n_cams,
                "pelvis_visible": int(world_T_pelvis is not None),
                "pelvis_used_id": pelvis_used_id,
                "pelvis_n_cams": pelvis_n_cams,
                "torso_path": torso_path,
                "box_visible": int(world_T_box is not None),
                "box_n_tags": len(per_tag_box_candidate),
            })
            row.update(fill_pose("head", world_T_headtag))
            row.update(fill_pose("pelvis", world_T_pelvis))
            row.update(fill_pose("root", world_T_root))
            row.update(fill_pose("torso_head", world_T_torso_head))
            row.update(fill_pose("torso_pelvis", world_T_torso_pelvis))
            row.update(fill_pose("torso", world_T_torso))
            row.update(fill_pose("obj", world_T_box))
            row.update(fill_actor_obs_fields(world_T_torso, world_T_box))
            csv_writer.writerow(row)
        stage_acc["csv"] += time.time() - _t_stage

        stage_acc["total"] += time.time() - _t_loop
        stage_n += 1
        if stage_n >= max(1, args.print_every):
            n = stage_n
            ms = {k: 1000.0 * v / n for k, v in stage_acc.items()}
            print(f"[timing avg/{n}f] grab={ms['grab']:5.1f}  "
                  f"detect={ms['detect']:5.1f}  fuse={ms['fuse']:5.1f}  "
                  f"gui={ms['gui']:5.1f}  csv={ms['csv']:5.1f}  "
                  f"TOTAL={ms['total']:5.1f}ms ({1000.0/max(ms['total'],0.1):4.1f}fps)")
            stage_acc = {k: 0.0 for k in stage_acc}
            stage_n = 0

        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break
finally:
    if detect_executor is not None:
        detect_executor.shutdown(wait=True)
    if udp_sock is not None:
        try:
            udp_sock.close()
            print(f"[multicam] UDP packets sent: {udp_packet_count}")
        except OSError:
            pass
    pipeline1.stop()
    pipeline2.stop()
    if pipeline3 is not None:
        pipeline3.stop()
    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()
        print(f"[multicam] CSV closed: {csv_path_resolved}")
    # ---- Per-tag margin summary ----
    if margin_stats:
        print("")
        print(f"================ Per-tag margin summary (margin_min={args.margin_min}) ================")
        print(f"{'id':>4}  {'role':<7}  {'n_seen':>7}  {'n_pass':>7}  {'pass%':>6}  "
              f"{'min':>6}  {'mean':>6}  {'max':>6}")
        # role classifier (mirror detect_one_camera labels)
        def _role(tid):
            if tid == args.origin_id: return "ORIGIN"
            if tid == ROBOT_HEAD_TAG_ID: return "HEAD"
            if tid in PELVIS_TAG_SET: return "PELVIS"
            if tid in BOX_TAG_SET: return "BOX"
            if tid in ANCHOR_IDS: return "ANCHOR"
            return "other"
        for tid in sorted(margin_stats.keys()):
            s = margin_stats[tid]
            mean_m = s["sum"] / max(1, s["n_seen"])
            pct = 100.0 * s["n_pass"] / max(1, s["n_seen"])
            print(f"{tid:>4}  {_role(tid):<7}  {s['n_seen']:>7d}  {s['n_pass']:>7d}  {pct:>5.1f}%  "
                  f"{s['min']:>6.1f}  {mean_m:>6.1f}  {s['max']:>6.1f}")
        # Tags whose mean margin is below threshold are likely the ones being filtered.
        bad = [tid for tid, s in margin_stats.items()
               if s["n_seen"] > 0 and s["n_pass"] / s["n_seen"] < 0.5]
        if bad:
            print("")
            print(f"[multicam] Tags with <50% pass rate at margin_min={args.margin_min}: {sorted(bad)}")
            print("           Consider lowering --margin-min or improving lighting / tag size.")
        print("=========================================================================")
