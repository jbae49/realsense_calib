import argparse
import numpy as np
import cv2
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


def rel_vec(tag_dict, origin_id, target_id):
    if origin_id not in tag_dict or target_id not in tag_dict:
        return None
    T_origin = tag_dict[origin_id]
    T_target = tag_dict[target_id]
    T_rel = np.linalg.inv(T_origin) @ T_target
    return T_rel[:3, 3]


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


def print_stats(name, values):
    if len(values) == 0:
        print(f"{name}: no samples")
        return
    arr = np.stack(values, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    norm = np.linalg.norm(arr, axis=1)
    print(f"\n[{name}]")
    print(f"  samples: {len(values)}")
    print(f"  mean xyz [m]: [{mean[0]:+.4f}, {mean[1]:+.4f}, {mean[2]:+.4f}]")
    print(f"  std  xyz [m]: [{std[0]:+.4f}, {std[1]:+.4f}, {std[2]:+.4f}]")
    print(f"  mean |rel| [m]: {norm.mean():.4f}")
    print(f"  std  |rel| [m]: {norm.std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam1-serial", type=str, default="935322072654")
    parser.add_argument("--cam2-serial", type=str, default="115222071236")
    parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
    parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
    parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
    parser.add_argument("--origin-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=500)
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

    p1 = rs.pipeline()
    c1 = rs.config()
    c1.enable_device(args.cam1_serial)
    c1.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    p1.start(c1)

    p2 = rs.pipeline()
    c2 = rs.config()
    c2.enable_device(args.cam2_serial)
    c2.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    p2.start(c2)

    cam1_vals = []
    cam2_vals = []
    fused_vals = []

    print("Collecting rel stats...")
    print(f"origin_id={args.origin_id}, target_id={args.target_id}, num_samples={args.num_samples}")
    print(f"stream={args.width}x{args.height}@{args.fps}")
    print("Press ESC to stop early.")

    try:
        while (
            len(cam1_vals) < args.num_samples
            or len(cam2_vals) < args.num_samples
            or len(fused_vals) < args.num_samples
        ):
            f1 = p1.wait_for_frames().get_color_frame()
            f2 = p2.wait_for_frames().get_color_frame()
            img1 = np.asanyarray(f1.get_data())
            img2 = np.asanyarray(f2.get_data())
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            dets1 = detect_with_tag_sizes(detector, gray1, cam1_params, tag_size, tag_size_map)
            dets2 = detect_with_tag_sizes(detector, gray2, cam2_params, tag_size, tag_size_map)

            cam1_tags_c2 = {}
            cam2_tags_c2 = {}
            fused_candidates = {}

            for det in dets1:
                T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
                T_c2_tag = T_c2_c1 @ T_c1_tag
                cam1_tags_c2[det.tag_id] = T_c2_tag
                w = float(getattr(det, "decision_margin", 1.0))
                fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3)})

            for det in dets2:
                T_c2_tag = pose_to_T(det.pose_R, det.pose_t)
                cam2_tags_c2[det.tag_id] = T_c2_tag
                w = float(getattr(det, "decision_margin", 1.0))
                fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3)})

            v1 = rel_vec(cam1_tags_c2, args.origin_id, args.target_id)
            v2 = rel_vec(cam2_tags_c2, args.origin_id, args.target_id)
            if v1 is not None and len(cam1_vals) < args.num_samples:
                cam1_vals.append(v1)
            if v2 is not None and len(cam2_vals) < args.num_samples:
                cam2_vals.append(v2)

            fused_tags = {tid: fuse_tag_poses(cands) for tid, cands in fused_candidates.items()}
            vf = rel_vec(fused_tags, args.origin_id, args.target_id)
            if vf is not None and len(fused_vals) < args.num_samples:
                fused_vals.append(vf)

            if (len(cam1_vals) + len(cam2_vals) + len(fused_vals)) % 30 == 0:
                print(
                    f"\rcam1={len(cam1_vals)}/{args.num_samples}  "
                    f"cam2={len(cam2_vals)}/{args.num_samples}  "
                    f"fused={len(fused_vals)}/{args.num_samples}",
                    end="",
                    flush=True,
                )

            cv2.imshow("cam1", img1)
            cv2.imshow("cam2", img2)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        p1.stop()
        p2.stop()
        cv2.destroyAllWindows()

    print("\n\n=== RESULT ===")
    print_stats("cam1_only (transformed to C2)", cam1_vals)
    print_stats("cam2_only (C2/world)", cam2_vals)
    print_stats("fused", fused_vals)


if __name__ == "__main__":
    main()
