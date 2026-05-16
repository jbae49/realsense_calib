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


parser = argparse.ArgumentParser()
parser.add_argument("--cam1-serial", type=str, default="935322072654")
parser.add_argument("--cam2-serial", type=str, default="115222071236")
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
parser.add_argument("--origin-id", type=int, default=1)
parser.add_argument("--tag-size", type=float, default=0.077)
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

K1 = np.load(args.cam1_calib)["camera_matrix"]
K2 = np.load(args.cam2_calib)["camera_matrix"]
cam1_params = [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]
cam2_params = [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]
T_c2_c1 = np.load(args.extrinsic)["T_c2_c1"]

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

print("Two-Cam Origin Fusion Viewer")
print("============================")
print(f"cam1={args.cam1_serial}, cam2(world)={args.cam2_serial}")
print(f"origin_id={args.origin_id}")
print(f"extrinsic={args.extrinsic}")
print(f"stream={args.width}x{args.height}@{args.fps}")
print("Press ESC to quit.")

try:
    while True:
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()
        img1 = np.asanyarray(frames1.get_color_frame().get_data())
        img2 = np.asanyarray(frames2.get_color_frame().get_data())
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        dets1 = detect_with_tag_sizes(detector, gray1, cam1_params, TAG_SIZE, TAG_SIZE_MAP)
        dets2 = detect_with_tag_sizes(detector, gray2, cam2_params, TAG_SIZE, TAG_SIZE_MAP)

        cam1_c2_tags = {}
        cam2_c2_tags = {}
        fused_candidates = {}

        for det in dets1:
            T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
            T_c2_tag = T_c2_c1 @ T_c1_tag
            cam1_c2_tags[det.tag_id] = T_c2_tag
            w = float(getattr(det, "decision_margin", 1.0))
            fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3)})

            center = tuple(det.center.astype(int))
            cv2.circle(img1, center, 5, (0, 255, 0), -1)
            cv2.putText(img1, f"ID:{det.tag_id}", (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for det in dets2:
            T_c2_tag = pose_to_T(det.pose_R, det.pose_t)
            cam2_c2_tags[det.tag_id] = T_c2_tag
            w = float(getattr(det, "decision_margin", 1.0))
            fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3)})

            center = tuple(det.center.astype(int))
            cv2.circle(img2, center, 5, (0, 255, 255), -1)
            cv2.putText(img2, f"ID:{det.tag_id}", (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        fused_tags = {tag_id: fuse_tag_poses(cands) for tag_id, cands in fused_candidates.items()}

        rel_cam1 = rel_map_from_tag_dict(cam1_c2_tags, args.origin_id)
        rel_cam2 = rel_map_from_tag_dict(cam2_c2_tags, args.origin_id)
        rel_fused = rel_map_from_tag_dict(fused_tags, args.origin_id)

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

        fused_panel = np.zeros((360, 820, 3), dtype=np.uint8)
        cv2.putText(fused_panel, f"FUSED in C2/world, origin ID {args.origin_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if rel_fused is None:
            cv2.putText(fused_panel, "Origin not visible in fused set", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(fused_panel, f"rel[{args.origin_id}->{args.origin_id}] [0.000,0.000,0.000]", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            row = 0
            for tag_id in sorted(rel_fused.keys()):
                if tag_id == args.origin_id:
                    continue
                p = rel_fused[tag_id][:3, 3]
                cv2.putText(
                    fused_panel,
                    f"rel[{args.origin_id}->{tag_id}] x:{p[0]:+.3f} y:{p[1]:+.3f} z:{p[2]:+.3f}",
                    (10, 85 + row * 20),
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
        cv2.imshow("Fused Origin Coordinates (C2/world)", fused_panel)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
finally:
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()
