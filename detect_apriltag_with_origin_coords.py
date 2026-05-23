"""
AprilTag visualization in an origin-tag coordinate frame.

This script is intentionally separate from detect_apriltag_with_axes.py
to avoid mixing camera-frame and origin-frame displays.

What it shows:
  - Tag outlines + IDs
  - Per-tag coordinates in origin frame
  - Origin tag coordinate is always [0, 0, 0]
  - Lines from origin tag center to other tag centers

Quick run commands (cam1/cam2/cam3):
  # cam1 (D435i: 935322072654)
  # python detect_apriltag_with_origin_coords.py \
  #   --serial 935322072654 \
  #   --calib camera1_935322072654_calibration.npz \
  #   --width 960 --height 540 --fps 60
  #
  # cam2 (D435: 115222071236)
  # python detect_apriltag_with_origin_coords.py \
  #   --serial 115222071236 \
  #   --calib camera2_115222071236_calibration.npz \
  #   --width 960 --height 540 --fps 60
  #
  # cam3 (D435: 112322072671)
  # python detect_apriltag_with_origin_coords.py \
  #   --serial 112322072671 \
  #   --calib camera3_112322072671_calibration.npz \
  #   --width 960 --height 540 --fps 60
"""
import argparse
import json
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


def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def parse_int_list(raw: str):
    if not raw.strip():
        return []
    out = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def load_anchor_transforms(path: str, origin_id: int):
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    anchors = data.get("anchors", {})
    cfg_origin_id = data.get("origin_id", origin_id)
    if int(cfg_origin_id) != int(origin_id):
        print(
            f"[WARN] anchor config origin_id={cfg_origin_id} != --origin-id={origin_id}. "
            "Using entries anyway."
        )
    out = {}
    if not isinstance(anchors, dict):
        return out
    for raw_id, meta in anchors.items():
        try:
            aid = int(raw_id)
            T = np.asarray(meta["T_origin_anchor"], dtype=float)
            if T.shape == (4, 4):
                out[aid] = T
        except Exception:
            continue
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--serial", type=str, default="115222071236",
                    help="RealSense serial number")
parser.add_argument("--calib", type=str, default="camera2_115222071236_calibration.npz",
                    help="Calibration .npz file path")
parser.add_argument("--origin-id", type=int, default=1,
                    help="Tag ID used as origin")
parser.add_argument("--tag-size", type=float, default=0.077,
                    help="Default AprilTag size in meters")
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json",
                    help="JSON file for default/per-tag tag sizes")
parser.add_argument("--tag-size-map", type=str, default="",
                    help='Per-tag size map, e.g. "0:0.145,1:0.145"')
parser.add_argument("--anchor-config", type=str, default="config/floor_anchor_transforms.json",
                    help="JSON file containing T_origin_anchor entries for fallback anchors")
parser.add_argument("--fallback-anchor-ids", type=str, default="",
                    help='Fallback anchor IDs when origin is hidden, e.g. "10,11"')
parser.add_argument("--width", type=int, default=640, help="Color stream width")
parser.add_argument("--height", type=int, default=480, help="Color stream height")
parser.add_argument("--fps", type=int, default=60, help="Color stream FPS")
parser.add_argument(
    "--resizable-window",
    action="store_true",
    help="Enable resizable preview window (may look softer when scaled)",
)
parser.add_argument(
    "--show-camera-coords",
    action="store_true",
    help="Also display each tag's position in the camera frame (for debugging)",
)
parser.add_argument(
    "--show-distance-check",
    action="store_true",
    help="Show |t_cam-tag - t_cam-origin| vs |t_origin-tag| (must match if math is correct)",
)
parser.add_argument(
    "--debug-print-every",
    type=int,
    default=0,
    help="If >0, every N frames print T_cam_origin, R_origin_cam, and per-tag positions to console",
)
args = parser.parse_args()

calib = np.load(args.calib)
K = calib["camera_matrix"]
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)
fallback_anchor_ids = parse_int_list(args.fallback_anchor_ids)
fallback_anchor_id_set = set(fallback_anchor_ids)
anchor_map = load_anchor_transforms(args.anchor_config, args.origin_id)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False,
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(args.serial)
config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline.start(config)

print("AprilTag Origin-Frame Viewer")
print("============================")
print(f"serial={args.serial} calib={args.calib}")
print(f"origin_id={args.origin_id}")
print(f"tag_config={args.tag_config}")
print(f"default_tag_size={TAG_SIZE}")
print(f"tag_size_map={TAG_SIZE_MAP if TAG_SIZE_MAP else '{}'}")
print(f"anchor_config={args.anchor_config}")
print(f"fallback_anchor_ids={fallback_anchor_ids if fallback_anchor_ids else '[]'}")
print("Coordinates shown under each tag are in ORIGIN frame.")
print("Press ESC to quit.")
print(f"window_mode={'resizable' if args.resizable_window else 'fixed (autosize)'}")

if args.resizable_window:
    cv2.namedWindow("AprilTag Origin Coordinates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Origin Coordinates", args.width, args.height)

frame_idx = 0
try:
    while True:
        frame_idx += 1
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections = detect_with_tag_sizes(
            detector,
            gray,
            [fx, fy, cx, cy],
            TAG_SIZE,
            TAG_SIZE_MAP,
        )
        det_by_id = {det.tag_id: det for det in detections}
        origin_det = det_by_id.get(args.origin_id)

        origin_center = None
        T_cam_origin = None
        origin_source = None
        if origin_det is not None:
            origin_center = tuple(origin_det.center.astype(int))
            T_cam_origin = pose_to_T(origin_det.pose_R, origin_det.pose_t)
            origin_source = args.origin_id
        else:
            for aid in fallback_anchor_ids:
                anchor_det = det_by_id.get(aid)
                T_origin_anchor = anchor_map.get(aid)
                if anchor_det is None or T_origin_anchor is None:
                    continue
                T_cam_anchor = pose_to_T(anchor_det.pose_R, anchor_det.pose_t)
                T_anchor_origin = np.linalg.inv(T_origin_anchor)
                T_cam_origin = T_cam_anchor @ T_anchor_origin
                origin_center = tuple(anchor_det.center.astype(int))
                origin_source = aid
                break

        for det in detections:
            corners = det.corners.astype(int)
            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(img, p1, p2, (0, 255, 255), 2)

            center = tuple(det.center.astype(int))
            cv2.circle(img, center, 5, (0, 255, 255), -1)

            if det.tag_id == args.origin_id or det.tag_id in fallback_anchor_id_set:
                color = (0, 255, 0)
                label = f"ID:{det.tag_id} ORIGIN/ANCHOR"
            else:
                color = (0, 255, 255)
                label = f"ID:{det.tag_id}"

            cv2.putText(
                img,
                label,
                (center[0] + 10, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            T_cam_tag_self = pose_to_T(det.pose_R, det.pose_t)
            cam_pos = T_cam_tag_self[:3, 3]

            text_y = center[1] + 5
            if T_cam_origin is not None:
                if det.tag_id == args.origin_id:
                    rel = np.zeros(3)
                    d_origin = 0.0
                    d_cam = 0.0
                else:
                    T_origin_tag = np.linalg.inv(T_cam_origin) @ T_cam_tag_self
                    rel = T_origin_tag[:3, 3]
                    d_origin = float(np.linalg.norm(rel))
                    d_cam = float(np.linalg.norm(cam_pos - T_cam_origin[:3, 3]))

                cv2.putText(
                    img,
                    f"rel: [{rel[0]:+.3f}, {rel[1]:+.3f}, {rel[2]:+.3f}] m",
                    (center[0] + 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )
                text_y += 16
                if origin_center is not None and det.tag_id != args.origin_id:
                    cv2.line(img, origin_center, center, (255, 255, 0), 2)

                if args.show_distance_check and det.tag_id != args.origin_id:
                    color = (0, 255, 0) if abs(d_cam - d_origin) < 0.005 else (0, 0, 255)
                    cv2.putText(
                        img,
                        f"|d|: cam={d_cam:.3f}  origin={d_origin:.3f}  diff={d_cam - d_origin:+.3f} m",
                        (center[0] + 10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                    )
                    text_y += 16
            else:
                cv2.putText(
                    img,
                    "rel: N/A (origin not visible)",
                    (center[0] + 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    1,
                )
                text_y += 16

            if args.show_camera_coords:
                cv2.putText(
                    img,
                    f"cam: [{cam_pos[0]:+.3f}, {cam_pos[1]:+.3f}, {cam_pos[2]:+.3f}] m",
                    (center[0] + 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 200, 255),
                    1,
                )

        if origin_det is None:
            if origin_source is not None and origin_source != args.origin_id:
                cv2.putText(
                    img,
                    f"Origin ID {args.origin_id} recovered via anchor {origin_source}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
            else:
                cv2.putText(
                    img,
                    f"Origin tag ID {args.origin_id} not visible",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                img,
                f"Origin ID {args.origin_id} visible",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if args.debug_print_every > 0 and frame_idx % args.debug_print_every == 0:
            print(f"\n[debug frame {frame_idx}] ----------------------------------------")
            if T_cam_origin is None:
                print("  origin not visible (and no fallback anchor)")
            else:
                R_co = T_cam_origin[:3, :3]
                t_co = T_cam_origin[:3, 3]
                R_oc = R_co.T
                print("  T_cam_origin:")
                print(f"    R_cam_origin =\n{R_co}")
                print(f"    t_cam_origin = {t_co}")
                print(f"    R_origin_cam[2,2] = {R_oc[2, 2]:+.4f}  "
                      f"(== cos(angle between cam_z and origin_z))")
                for det in detections:
                    if det.tag_id == args.origin_id:
                        continue
                    T_ct = pose_to_T(det.pose_R, det.pose_t)
                    t_ct = T_ct[:3, 3]
                    delta_cam = t_ct - t_co
                    t_ot = (np.linalg.inv(T_cam_origin) @ T_ct)[:3, 3]
                    print(
                        f"  id{det.tag_id}: "
                        f"t_cam={t_ct.round(3).tolist()}  "
                        f"Δt_cam={delta_cam.round(3).tolist()}  |Δ|={np.linalg.norm(delta_cam):.3f}  "
                        f"t_origin={t_ot.round(3).tolist()}  |t_o|={np.linalg.norm(t_ot):.3f}"
                    )
                    print(
                        f"     check  |Δt_cam| - |t_origin| = "
                        f"{np.linalg.norm(delta_cam) - np.linalg.norm(t_ot):+.6f} m  "
                        f"(must be ~0 if math is correct)"
                    )

        cv2.imshow("AprilTag Origin Coordinates", img)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
