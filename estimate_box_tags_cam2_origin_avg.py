import argparse
from dataclasses import dataclass, field

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


def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def rotation_to_quat_xyzw(R):
    q = np.empty(4, dtype=float)  # x, y, z, w
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    return q


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


def weighted_avg_rotation(rotations, weights):
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-9:
        return rotations[0]
    R = np.zeros((3, 3), dtype=float)
    for rot, w in zip(rotations, weights):
        R += (w / w_sum) * rot
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


@dataclass
class TagAccumulator:
    positions: list = field(default_factory=list)
    rotations: list = field(default_factory=list)
    margins: list = field(default_factory=list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="115222071236")
    parser.add_argument("--calib", type=str, default="camera2_115222071236_calibration.npz")
    parser.add_argument("--origin-id", type=int, default=1)
    parser.add_argument("--target-ids", type=str, default="0,2,3,4,5")
    parser.add_argument("--num-frames", type=int, default=10, help="Per-tag valid samples")
    parser.add_argument("--max-frames", type=int, default=1000, help="Safety limit for total frames")
    parser.add_argument("--tag-size", type=float, default=0.077)
    parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
    parser.add_argument("--tag-size-map", type=str, default="")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--show", action="store_true", help="Show camera window while collecting")
    args = parser.parse_args()

    target_ids = parse_int_list(args.target_ids)
    if not target_ids:
        raise ValueError("--target-ids is empty")

    calib = np.load(args.calib)
    K = calib["camera_matrix"]
    cam_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
    tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
    cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
    tag_size, tag_size_map = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe.start(cfg)

    acc = {tid: TagAccumulator() for tid in target_ids}
    frame_idx = 0

    print("Collecting origin-relative tag poses (camera2)...")
    print(f"origin_id={args.origin_id}, target_ids={target_ids}")
    print(f"target samples per id={args.num_frames}, max_frames={args.max_frames}")
    print(f"stream={args.width}x{args.height}@{args.fps}")
    print("Press ESC to stop early.")

    try:
        while frame_idx < args.max_frames:
            frame_idx += 1
            color = pipe.wait_for_frames().get_color_frame()
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dets = detect_with_tag_sizes(detector, gray, cam_params, tag_size, tag_size_map)
            det_by_id = {d.tag_id: d for d in dets}
            origin_det = det_by_id.get(args.origin_id)
            if origin_det is None:
                if args.show:
                    cv2.putText(img, f"origin {args.origin_id}: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("cam2 origin avg collector", img)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                continue

            T_cam_origin = pose_to_T(origin_det.pose_R, origin_det.pose_t)
            origin_inv = np.linalg.inv(T_cam_origin)

            for tid in target_ids:
                if len(acc[tid].positions) >= args.num_frames:
                    continue
                det = det_by_id.get(tid)
                if det is None:
                    continue
                T_cam_tag = pose_to_T(det.pose_R, det.pose_t)
                T_origin_tag = origin_inv @ T_cam_tag
                acc[tid].positions.append(T_origin_tag[:3, 3].copy())
                acc[tid].rotations.append(T_origin_tag[:3, :3].copy())
                acc[tid].margins.append(float(getattr(det, "decision_margin", 0.0)))

            counts = {tid: len(acc[tid].positions) for tid in target_ids}
            print(f"[frame {frame_idx}] collected " + " ".join(f"{tid}:{counts[tid]}/{args.num_frames}" for tid in target_ids))

            if all(len(acc[tid].positions) >= args.num_frames for tid in target_ids):
                break

            if args.show:
                for det in dets:
                    corners = det.corners.astype(int)
                    for i in range(4):
                        p1 = tuple(corners[i])
                        p2 = tuple(corners[(i + 1) % 4])
                        color = (0, 255, 0) if det.tag_id == args.origin_id else (0, 255, 255)
                        cv2.line(img, p1, p2, color, 2)
                cv2.imshow("cam2 origin avg collector", img)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    print("\n=== RESULT (origin frame) ===")
    for tid in target_ids:
        n = len(acc[tid].positions)
        if n == 0:
            print(f"ID {tid}: no valid samples")
            continue
        pos = np.stack(acc[tid].positions, axis=0)
        margins = np.asarray(acc[tid].margins, dtype=float)
        rot = weighted_avg_rotation(acc[tid].rotations, margins if np.sum(margins) > 1e-9 else np.ones_like(margins))
        mean_pos = pos.mean(axis=0)
        std_pos = pos.std(axis=0)
        quat_xyzw = rotation_to_quat_xyzw(rot)
        euler_xyz_deg = rotation_to_euler_xyz_deg(rot)
        print(f"\nID {tid} (n={n})")
        print(f"  mean pos [m] : [{mean_pos[0]:+.4f}, {mean_pos[1]:+.4f}, {mean_pos[2]:+.4f}]")
        print(f"  std  pos [m] : [{std_pos[0]:+.4f}, {std_pos[1]:+.4f}, {std_pos[2]:+.4f}]")
        print(f"  mean margin  : {margins.mean():.2f}")
        print(f"  mean quat(xyzw): [{quat_xyzw[0]:+.5f}, {quat_xyzw[1]:+.5f}, {quat_xyzw[2]:+.5f}, {quat_xyzw[3]:+.5f}]")
        print(f"  mean euler xyz [deg]: [{euler_xyz_deg[0]:+.2f}, {euler_xyz_deg[1]:+.2f}, {euler_xyz_deg[2]:+.2f}]")


if __name__ == "__main__":
    main()
