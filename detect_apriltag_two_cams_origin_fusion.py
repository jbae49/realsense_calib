import argparse
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


def rotation_to_euler_xyz_deg(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


def rotation_to_quat_wxyz(R):
    q = np.empty(4, dtype=float)  # w, x, y, z
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


def orientation_text(R, fmt):
    parts = []
    if fmt in ("euler", "both"):
        eul = rotation_to_euler_xyz_deg(R)
        parts.append(f"eul[{eul[0]:+.1f},{eul[1]:+.1f},{eul[2]:+.1f}]")
    if fmt in ("quat", "both"):
        q = rotation_to_quat_wxyz(R)
        parts.append(f"quat[wxyz]=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]")
    return " ".join(parts)


def project_tag_point_to_pixel(T_cam_tag, p_tag, cam_params):
    fx, fy, cx, cy = cam_params
    p_cam = T_cam_tag[:3, :3] @ np.asarray(p_tag, dtype=float).reshape(3) + T_cam_tag[:3, 3]
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    u = int(round(fx * (p_cam[0] / z) + cx))
    v = int(round(fy * (p_cam[1] / z) + cy))
    return (u, v)


def draw_pose_axes(img, T_cam_tag, cam_params, axis_len=0.08):
    origin = project_tag_point_to_pixel(T_cam_tag, [0.0, 0.0, 0.0], cam_params)
    x_tip = project_tag_point_to_pixel(T_cam_tag, [axis_len, 0.0, 0.0], cam_params)
    y_tip = project_tag_point_to_pixel(T_cam_tag, [0.0, axis_len, 0.0], cam_params)
    z_tip = project_tag_point_to_pixel(T_cam_tag, [0.0, 0.0, axis_len], cam_params)
    if origin is None:
        return
    if x_tip is not None:
        cv2.line(img, origin, x_tip, (0, 0, 255), 2)  # X: red
    if y_tip is not None:
        cv2.line(img, origin, y_tip, (0, 255, 0), 2)  # Y: green
    if z_tip is not None:
        cv2.line(img, origin, z_tip, (255, 0, 0), 2)  # Z: blue


def weighted_avg_rotation(rotations, weights):
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-9:
        return rotations[0]
    R = np.zeros((3, 3))
    for rot, w in zip(rotations, weights):
        R += (w / w_sum) * rot
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def fuse_tag_poses(candidates):
    if len(candidates) == 1:
        return candidates[0]["T"]
    weights = np.array([c["w"] for c in candidates], dtype=float)
    positions = np.stack([c["T"][:3, 3] for c in candidates], axis=0)
    rotations = [c["T"][:3, :3] for c in candidates]
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-9:
        w = np.ones_like(weights) / len(weights)
    else:
        w = weights / w_sum
    pos = np.sum(positions * w[:, None], axis=0)
    rot = weighted_avg_rotation(rotations, weights)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def rel_map_from_tag_dict(tag_dict, origin_id):
    if origin_id not in tag_dict:
        return None
    T_origin = tag_dict[origin_id]
    inv_T_origin = np.linalg.inv(T_origin)
    rel = {}
    for tag_id, T in tag_dict.items():
        rel[tag_id] = inv_T_origin @ T
    return rel


def add_direct_rel_candidates(rel_candidates, tag_dict, margin_dict, origin_id, margin_min, source_name):
    if origin_id not in tag_dict:
        return
    origin_margin = margin_dict.get(origin_id, 0.0)
    if origin_margin < margin_min:
        return
    inv_origin = np.linalg.inv(tag_dict[origin_id])
    for tag_id, T in tag_dict.items():
        if tag_id == origin_id:
            continue
        tag_margin = margin_dict.get(tag_id, 0.0)
        if tag_margin < margin_min:
            continue
        rel_T = inv_origin @ T
        rel_candidates.setdefault(tag_id, []).append(
            {
                "T": rel_T,
                "w": max(min(origin_margin, tag_margin), 1e-3),
                "src": f"{source_name}:direct",
            }
        )


parser = argparse.ArgumentParser()
parser.add_argument("--cam1-serial", type=str, default="935322072654")
parser.add_argument("--cam2-serial", type=str, default="115222071236")
parser.add_argument("--cam3-serial", type=str, default="")
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--cam3-calib", type=str, default="")
parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
parser.add_argument(
    "--extrinsic-cam3-to-c2",
    type=str,
    default="",
    help="Extrinsic npz that maps cam3 pose into C2/world frame (expects key T_c2_c1).",
)
parser.add_argument("--origin-id", type=int, default=1)
parser.add_argument("--tag-size", type=float, default=0.077)
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--margin-min", type=float, default=40.0)
parser.add_argument("--show-orientation", action="store_true", help="Show fused orientation per tag")
parser.add_argument(
    "--orientation-format",
    type=str,
    default="euler",
    choices=["euler", "quat", "both"],
    help="Orientation display format when --show-orientation is enabled",
)
parser.add_argument("--show-axes", action="store_true", help="Draw RGB XYZ axes on each camera image")
parser.add_argument("--axis-length", type=float, default=0.08, help="Axis length in meters for --show-axes")
args = parser.parse_args()

K1 = np.load(args.cam1_calib)["camera_matrix"]
K2 = np.load(args.cam2_calib)["camera_matrix"]
cam1_params = [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]
cam2_params = [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]
T_c2_c1 = np.load(args.extrinsic)["T_c2_c1"]
use_cam3 = bool(args.cam3_serial.strip())
if use_cam3:
    if not args.cam3_calib.strip():
        raise ValueError("--cam3-calib is required when --cam3-serial is set")
    if not args.extrinsic_cam3_to_c2.strip():
        raise ValueError("--extrinsic-cam3-to-c2 is required when --cam3-serial is set")
    K3 = np.load(args.cam3_calib)["camera_matrix"]
    cam3_params = [K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]]
    T_c2_c3 = np.load(args.extrinsic_cam3_to_c2)["T_c2_c1"]

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False,
)

pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device(args.cam1_serial)
config1.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline1.start(config1)

pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device(args.cam2_serial)
config2.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline2.start(config2)

pipeline3 = None
if use_cam3:
    pipeline3 = rs.pipeline()
    config3 = rs.config()
    config3.enable_device(args.cam3_serial)
    config3.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline3.start(config3)

print("Multi-Cam Origin Fusion Viewer")
print("=============================")
if use_cam3:
    print(f"cam1={args.cam1_serial}, cam2(world)={args.cam2_serial}, cam3={args.cam3_serial}")
else:
    print(f"cam1={args.cam1_serial}, cam2(world)={args.cam2_serial}")
print(f"origin_id={args.origin_id}")
print(f"extrinsic={args.extrinsic}")
if use_cam3:
    print(f"extrinsic_cam3_to_c2={args.extrinsic_cam3_to_c2}")
print(f"margin_min={args.margin_min:.1f} (used for direct/fallback fusion)")
print(f"stream={args.width}x{args.height}@{args.fps}")
print(
    "show_orientation="
    + ("on" if args.show_orientation else "off")
    + (f", format={args.orientation_format}" if args.show_orientation else "")
)
print("show_axes=" + ("on" if args.show_axes else "off") + (f", axis_length={args.axis_length:.3f}m" if args.show_axes else ""))
print("Press ESC to quit.")

try:
    while True:
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()
        img1 = np.asanyarray(frames1.get_color_frame().get_data())
        img2 = np.asanyarray(frames2.get_color_frame().get_data())
        img3 = None
        if use_cam3:
            frames3 = pipeline3.wait_for_frames()
            img3 = np.asanyarray(frames3.get_color_frame().get_data())
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) if use_cam3 else None

        dets1 = detect_with_tag_sizes(detector, gray1, cam1_params, TAG_SIZE, TAG_SIZE_MAP)
        dets2 = detect_with_tag_sizes(detector, gray2, cam2_params, TAG_SIZE, TAG_SIZE_MAP)
        dets3 = detect_with_tag_sizes(detector, gray3, cam3_params, TAG_SIZE, TAG_SIZE_MAP) if use_cam3 else []

        cam1_local_tags = {}
        cam2_local_tags = {}
        cam3_local_tags = {}
        cam1_c2_tags = {}
        cam2_c2_tags = {}
        cam3_c2_tags = {}
        cam1_margin = {}
        cam2_margin = {}
        cam3_margin = {}
        fused_candidates_c2 = {}

        for det in dets1:
            T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
            cam1_local_tags[det.tag_id] = T_c1_tag
            w = float(getattr(det, "decision_margin", 1.0))
            cam1_margin[det.tag_id] = w
            T_c2_tag = T_c2_c1 @ T_c1_tag
            cam1_c2_tags[det.tag_id] = T_c2_tag
            if w >= args.margin_min:
                fused_candidates_c2.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3), "src": "cam1:extrinsic"})
            if args.show_axes:
                draw_pose_axes(img1, T_c1_tag, cam1_params, axis_len=args.axis_length)

            center = tuple(det.center.astype(int))
            cv2.circle(img1, center, 5, (0, 255, 0), -1)
            cv2.putText(img1, f"ID:{det.tag_id} m:{w:.1f}", (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for det in dets2:
            T_c2_tag = pose_to_T(det.pose_R, det.pose_t)
            cam2_local_tags[det.tag_id] = T_c2_tag
            w = float(getattr(det, "decision_margin", 1.0))
            cam2_margin[det.tag_id] = w
            cam2_c2_tags[det.tag_id] = T_c2_tag
            if w >= args.margin_min:
                fused_candidates_c2.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3), "src": "cam2:world"})
            if args.show_axes:
                draw_pose_axes(img2, T_c2_tag, cam2_params, axis_len=args.axis_length)

            center = tuple(det.center.astype(int))
            cv2.circle(img2, center, 5, (0, 255, 255), -1)
            cv2.putText(img2, f"ID:{det.tag_id} m:{w:.1f}", (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for det in dets3:
            T_c3_tag = pose_to_T(det.pose_R, det.pose_t)
            cam3_local_tags[det.tag_id] = T_c3_tag
            w = float(getattr(det, "decision_margin", 1.0))
            cam3_margin[det.tag_id] = w
            T_c2_tag = T_c2_c3 @ T_c3_tag
            cam3_c2_tags[det.tag_id] = T_c2_tag
            if w >= args.margin_min:
                fused_candidates_c2.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3), "src": "cam3:extrinsic"})
            if args.show_axes:
                draw_pose_axes(img3, T_c3_tag, cam3_params, axis_len=args.axis_length)

            center = tuple(det.center.astype(int))
            cv2.circle(img3, center, 5, (255, 0, 255), -1)
            cv2.putText(img3, f"ID:{det.tag_id} m:{w:.1f}", (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 1) Prefer direct per-camera relative estimates (no extrinsic path).
        rel_candidates = {}
        add_direct_rel_candidates(rel_candidates, cam1_local_tags, cam1_margin, args.origin_id, args.margin_min, "cam1")
        add_direct_rel_candidates(rel_candidates, cam2_local_tags, cam2_margin, args.origin_id, args.margin_min, "cam2")
        if use_cam3:
            add_direct_rel_candidates(rel_candidates, cam3_local_tags, cam3_margin, args.origin_id, args.margin_min, "cam3")

        rel_fused = {tag_id: fuse_tag_poses(cands) for tag_id, cands in rel_candidates.items()}
        direct_tag_count = len(rel_fused)
        fallback_tag_count = 0

        # 2) Fallback only for tags missing from direct map: use extrinsic-merged C2/world tags.
        fused_tags_c2 = {tag_id: fuse_tag_poses(cands) for tag_id, cands in fused_candidates_c2.items()}
        if args.origin_id in fused_tags_c2:
            inv_origin_c2 = np.linalg.inv(fused_tags_c2[args.origin_id])
            for tag_id, T_c2_tag in fused_tags_c2.items():
                if tag_id == args.origin_id or tag_id in rel_fused:
                    continue
                rel_fused[tag_id] = inv_origin_c2 @ T_c2_tag
                fallback_tag_count += 1

        rel_cam1 = rel_map_from_tag_dict(cam1_c2_tags, args.origin_id)
        rel_cam2 = rel_map_from_tag_dict(cam2_c2_tags, args.origin_id)
        rel_cam3 = rel_map_from_tag_dict(cam3_c2_tags, args.origin_id) if use_cam3 else None

        if rel_cam1 is None:
            cv2.putText(img1, f"CAM1 origin ID {args.origin_id} not visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(img1, f"CAM1->C2 origin ID {args.origin_id} OK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            idx = 0
            for tag_id in sorted(rel_cam1.keys()):
                if tag_id == args.origin_id:
                    continue
                p = rel_cam1[tag_id][:3, 3]
                cv2.putText(img1, f"rel[{args.origin_id}->{tag_id}] [{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]",
                            (10, 55 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                idx += 1
                if idx >= 12:
                    break

        if rel_cam2 is None:
            cv2.putText(img2, f"CAM2 origin ID {args.origin_id} not visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(img2, f"CAM2(world) origin ID {args.origin_id} OK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            idx = 0
            for tag_id in sorted(rel_cam2.keys()):
                if tag_id == args.origin_id:
                    continue
                p = rel_cam2[tag_id][:3, 3]
                cv2.putText(img2, f"rel[{args.origin_id}->{tag_id}] [{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]",
                            (10, 55 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                idx += 1
                if idx >= 12:
                    break

        if use_cam3:
            if rel_cam3 is None:
                cv2.putText(img3, f"CAM3 origin ID {args.origin_id} not visible", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                cv2.putText(img3, f"CAM3->C2 origin ID {args.origin_id} OK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                idx = 0
                for tag_id in sorted(rel_cam3.keys()):
                    if tag_id == args.origin_id:
                        continue
                    p = rel_cam3[tag_id][:3, 3]
                    cv2.putText(img3, f"rel[{args.origin_id}->{tag_id}] [{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]",
                                (10, 55 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    idx += 1
                    if idx >= 12:
                        break

        fused_panel = np.zeros((360, 820, 3), dtype=np.uint8)
        cv2.putText(fused_panel, f"FUSED in C2/world, origin ID {args.origin_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(
            fused_panel,
            f"sources: cam1({len(dets1)}) cam2({len(dets2)})" + (f" cam3({len(dets3)})" if use_cam3 else ""),
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        if len(rel_fused) == 0:
            cv2.putText(fused_panel, "Origin not visible in fused set", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(
                fused_panel,
                f"mode: direct-first, fallback-via-extrinsic  direct={direct_tag_count} fallback={fallback_tag_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 255, 180),
                1,
            )
            cv2.putText(fused_panel, f"rel[{args.origin_id}->{args.origin_id}] [0.000,0.000,0.000]", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            row = 0
            for tag_id in sorted(rel_fused.keys()):
                if tag_id == args.origin_id:
                    continue
                p = rel_fused[tag_id][:3, 3]
                cv2.putText(
                    fused_panel,
                    f"rel[{args.origin_id}->{tag_id}] x:{p[0]:+.3f} y:{p[1]:+.3f} z:{p[2]:+.3f}"
                    + (
                        " " + orientation_text(rel_fused[tag_id][:3, :3], args.orientation_format)
                        if args.show_orientation
                        else ""
                    ),
                    (10, 95 + row * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                row += 1
                if row >= 12:
                    break

        cv2.imshow("Camera1 -> transformed to C2", img1)
        cv2.imshow("Camera2 (C2/world)", img2)
        if use_cam3:
            cv2.imshow("Camera3 -> transformed to C2", img3)
        cv2.imshow("Fused Origin Coordinates (C2/world)", fused_panel)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
finally:
    pipeline1.stop()
    pipeline2.stop()
    if pipeline3 is not None:
        pipeline3.stop()
    cv2.destroyAllWindows()
