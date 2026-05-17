import argparse
import csv
import json
from dataclasses import dataclass
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


def rel_vec_from_T(T_origin, T_target):
    T_rel = np.linalg.inv(T_origin) @ T_target
    return T_rel[:3, 3]


def draw_detection(img, det, color):
    corners = det.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, color, 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    cv2.putText(
        img,
        f"ID:{det.tag_id}",
        (center[0] + 10, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
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


def load_anchor_transforms(path: str, origin_id: int):
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        return {}
    cfg_origin_id = data.get("origin_id", origin_id)
    if int(cfg_origin_id) != int(origin_id):
        print(
            f"[WARN] anchor config origin_id={cfg_origin_id} != --origin-id={origin_id}. "
            "Using entries anyway."
        )
    anchors = data.get("anchors", {})
    if not isinstance(anchors, dict):
        return {}
    out = {}
    for raw_id, meta in anchors.items():
        try:
            aid = int(raw_id)
            T = np.asarray(meta["T_origin_anchor"], dtype=float)
            if T.shape == (4, 4):
                out[aid] = T
        except Exception:
            continue
    return out


def inject_origin_from_fallback(tag_dict, origin_id, fallback_anchor_ids, anchor_map):
    """
    Returns a copied dict where origin_id exists if either:
    - origin_id was already detected, or
    - fallback anchor is detected and T_origin_anchor is known.
    """
    out = dict(tag_dict)
    if origin_id in out:
        return out, origin_id
    for aid in fallback_anchor_ids:
        if aid in out and aid in anchor_map:
            # T_cam_origin = T_cam_anchor @ T_anchor_origin
            T_anchor_origin = np.linalg.inv(anchor_map[aid])
            out[origin_id] = out[aid] @ T_anchor_origin
            return out, aid
    return out, None


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


def kalman_filter_cv(meas_xyz, dt, process_var=0.05, meas_var=0.01):
    """
    Constant-velocity Kalman filter for 3D position.
    state = [x, y, z, vx, vy, vz]
    """
    n = len(meas_xyz)
    if n == 0:
        return np.empty((0, 3))

    x = np.zeros((6, 1))
    x[:3, 0] = meas_xyz[0]

    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    H = np.zeros((3, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0

    q = float(process_var)
    Q = np.eye(6) * q
    r = float(meas_var)
    R = np.eye(3) * r
    P = np.eye(6) * 1.0

    out = []
    for z in meas_xyz:
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        z = z.reshape(3, 1)
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

        out.append(x[:3, 0].copy())

    return np.array(out)


@dataclass
class SeriesMetrics:
    mean_xyz: np.ndarray
    std_xyz: np.ndarray
    mean_norm: float
    std_norm: float
    p95_jump: float
    max_jump: float
    median_jump: float
    mad_jump: float
    spike_count: int
    spike_ratio: float


def compute_metrics(series_xyz, spike_k=6.0):
    if len(series_xyz) < 2:
        z = np.zeros(3)
        return SeriesMetrics(z, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    arr = np.asarray(series_xyz)
    mean_xyz = arr.mean(axis=0)
    std_xyz = arr.std(axis=0)
    norms = np.linalg.norm(arr, axis=1)

    jumps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    p95_jump = float(np.percentile(jumps, 95))
    max_jump = float(np.max(jumps))
    median_jump = float(np.median(jumps))
    mad_jump = float(np.median(np.abs(jumps - median_jump))) + 1e-9
    spike_thresh = median_jump + spike_k * mad_jump
    spike_count = int(np.sum(jumps > spike_thresh))
    spike_ratio = float(spike_count / len(jumps))

    return SeriesMetrics(
        mean_xyz=mean_xyz,
        std_xyz=std_xyz,
        mean_norm=float(norms.mean()),
        std_norm=float(norms.std()),
        p95_jump=p95_jump,
        max_jump=max_jump,
        median_jump=median_jump,
        mad_jump=mad_jump,
        spike_count=spike_count,
        spike_ratio=spike_ratio,
    )


def print_metrics(name, m: SeriesMetrics):
    print(f"\n[{name}]")
    print(f"  mean xyz [m]: [{m.mean_xyz[0]:+.4f}, {m.mean_xyz[1]:+.4f}, {m.mean_xyz[2]:+.4f}]")
    print(f"  std  xyz [m]: [{m.std_xyz[0]:+.4f}, {m.std_xyz[1]:+.4f}, {m.std_xyz[2]:+.4f}]")
    print(f"  mean |rel| [m]: {m.mean_norm:.4f}")
    print(f"  std  |rel| [m]: {m.std_norm:.4f}")
    print(f"  jump p95 [m/frame]: {m.p95_jump:.5f}")
    print(f"  jump max [m/frame]: {m.max_jump:.5f}")
    print(f"  jump median [m/frame]: {m.median_jump:.5f}")
    print(f"  jump MAD [m/frame]: {m.mad_jump:.5f}")
    print(f"  spike count: {m.spike_count}")
    print(f"  spike ratio: {m.spike_ratio*100:.2f}%")


def maybe_write_csv(path: str, raw_xyz, kf_xyz):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "raw_x",
                "raw_y",
                "raw_z",
                "kf_x",
                "kf_y",
                "kf_z",
            ]
        )
        for i, (r, k) in enumerate(zip(raw_xyz, kf_xyz)):
            writer.writerow([i, r[0], r[1], r[2], k[0], k[1], k[2]])
    print(f"\nSaved CSV: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam1-serial", type=str, default="935322072654")
    parser.add_argument("--cam2-serial", type=str, default="115222071236")
    parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
    parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
    parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
    parser.add_argument("--origin-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument(
        "--target-ids",
        type=str,
        default="",
        help='Optional target ID list, e.g. "1,2,3,4,5". First ID is primary for rel metrics.',
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--tag-size", type=float, default=0.077)
    parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
    parser.add_argument("--tag-size-map", type=str, default="")
    parser.add_argument("--anchor-config", type=str, default="config/floor_anchor_transforms.json")
    parser.add_argument(
        "--fallback-anchor-ids",
        type=str,
        default="",
        help='Fallback floor anchors when origin is missing, e.g. "10,11"',
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--kalman-process-var", type=float, default=0.05)
    parser.add_argument("--kalman-meas-var", type=float, default=0.01)
    parser.add_argument("--spike-k", type=float, default=6.0, help="MAD multiplier for spike threshold")
    parser.add_argument("--csv-out", type=str, default="", help="Optional output CSV path")
    parser.add_argument(
        "--print-z-ids",
        type=str,
        default="",
        help='Comma-separated tag IDs to print origin-frame z every frame, e.g. "1,2,3,4,5"',
    )
    parser.add_argument(
        "--print-margin-ids",
        type=str,
        default="",
        help='Comma-separated tag IDs to print per-camera decision_margin every frame, e.g. "1,10,0,2,3,4,5"',
    )
    parser.add_argument(
        "--print-occlusion-events",
        type=int,
        default=20,
        help="Max number of occlusion events to print (0 to disable)",
    )
    args = parser.parse_args()
    target_ids = parse_int_list(args.target_ids)
    if not target_ids:
        target_ids = [args.target_id]
    primary_target_id = target_ids[0]
    target_id_set = set(target_ids)
    fallback_anchor_ids = parse_int_list(args.fallback_anchor_ids)
    fallback_anchor_id_set = set(fallback_anchor_ids)
    print_z_ids = parse_int_list(args.print_z_ids)
    print_margin_ids = parse_int_list(args.print_margin_ids)

    K1 = np.load(args.cam1_calib)["camera_matrix"]
    K2 = np.load(args.cam2_calib)["camera_matrix"]
    cam1_params = [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]
    cam2_params = [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]
    T_c2_c1 = np.load(args.extrinsic)["T_c2_c1"]

    cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
    tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
    cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
    tag_size, tag_size_map = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)
    anchor_map = load_anchor_transforms(args.anchor_config, args.origin_id)
    missing_fallback_ids = [aid for aid in fallback_anchor_ids if aid not in anchor_map]
    if fallback_anchor_ids:
        print(f"fallback_anchor_ids={fallback_anchor_ids}")
        if missing_fallback_ids:
            print(f"[WARN] missing anchor transforms for IDs: {missing_fallback_ids}")

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

    fused_rel = []
    frame_idx = 0

    vis_stats = {
        "both_cameras_have_origin_and_any_target": 0,
        "cam1_only_has_origin_and_any_target": 0,
        "cam2_only_has_origin_and_any_target": 0,
        "neither_has_origin_and_any_target": 0,
        "both_cameras_missing_all_targets": 0,
        "origin_recovered_by_fallback": 0,
        "origin_missing_even_after_fallback": 0,
    }
    occlusion_events = []

    print("Collecting fused rel samples...")
    print(
        f"origin_id={args.origin_id}, target_ids={target_ids}, "
        f"primary_target_id={primary_target_id}, num_samples={args.num_samples}"
    )
    if print_z_ids:
        print(f"print_z_ids={print_z_ids} (origin frame, fused)")
    if print_margin_ids:
        print(f"print_margin_ids={print_margin_ids} (per camera decision_margin)")
    print(f"anchor_config={args.anchor_config}")
    print(f"stream={args.width}x{args.height}@{args.fps}")
    print("Move box tags while origin is visible (or recoverable via fallback anchors).")
    print("Press ESC to stop early.")

    try:
        while len(fused_rel) < args.num_samples:
            frame_idx += 1
            f1 = p1.wait_for_frames().get_color_frame()
            f2 = p2.wait_for_frames().get_color_frame()
            img1 = np.asanyarray(f1.get_data())
            img2 = np.asanyarray(f2.get_data())
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            dets1 = detect_with_tag_sizes(detector, gray1, cam1_params, tag_size, tag_size_map)
            dets2 = detect_with_tag_sizes(detector, gray2, cam2_params, tag_size, tag_size_map)

            fused_candidates = {}
            cam1_seen = set()
            cam2_seen = set()
            cam1_tags_c2 = {}
            cam2_tags_c2 = {}
            cam1_margin = {}
            cam2_margin = {}
            for det in dets1:
                T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
                T_c2_tag = T_c2_c1 @ T_c1_tag
                w = float(getattr(det, "decision_margin", 1.0))
                fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3)})
                cam1_seen.add(det.tag_id)
                cam1_tags_c2[det.tag_id] = T_c2_tag
                cam1_margin[det.tag_id] = max(cam1_margin.get(det.tag_id, -1e9), w)
                if det.tag_id == args.origin_id or det.tag_id in fallback_anchor_id_set:
                    draw_detection(img1, det, (0, 255, 0))      # origin/fallback anchors: green
                elif det.tag_id in target_id_set:
                    draw_detection(img1, det, (0, 255, 255))    # targets: yellow
                else:
                    draw_detection(img1, det, (180, 180, 180))  # others: gray
            for det in dets2:
                T_c2_tag = pose_to_T(det.pose_R, det.pose_t)
                w = float(getattr(det, "decision_margin", 1.0))
                fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(w, 1e-3)})
                cam2_seen.add(det.tag_id)
                cam2_tags_c2[det.tag_id] = T_c2_tag
                cam2_margin[det.tag_id] = max(cam2_margin.get(det.tag_id, -1e9), w)
                if det.tag_id == args.origin_id or det.tag_id in fallback_anchor_id_set:
                    draw_detection(img2, det, (0, 255, 0))      # origin/fallback anchors: green
                elif det.tag_id in target_id_set:
                    draw_detection(img2, det, (0, 255, 255))    # targets: yellow
                else:
                    draw_detection(img2, det, (180, 180, 180))  # others: gray

            fused_tags = {tid: fuse_tag_poses(cands) for tid, cands in fused_candidates.items()}
            cam1_eval, cam1_origin_source = inject_origin_from_fallback(
                cam1_tags_c2, args.origin_id, fallback_anchor_ids, anchor_map
            )
            cam2_eval, cam2_origin_source = inject_origin_from_fallback(
                cam2_tags_c2, args.origin_id, fallback_anchor_ids, anchor_map
            )
            fused_eval, fused_origin_source = inject_origin_from_fallback(
                fused_tags, args.origin_id, fallback_anchor_ids, anchor_map
            )

            if (
                args.origin_id not in fused_tags
                and args.origin_id in fused_eval
                and fused_origin_source is not None
                and fused_origin_source != args.origin_id
            ):
                vis_stats["origin_recovered_by_fallback"] += 1
            if args.origin_id not in fused_eval:
                vis_stats["origin_missing_even_after_fallback"] += 1

            c1_has_pair = args.origin_id in cam1_eval and any(t in cam1_eval for t in target_id_set)
            c2_has_pair = args.origin_id in cam2_eval and any(t in cam2_eval for t in target_id_set)
            if c1_has_pair and c2_has_pair:
                vis_stats["both_cameras_have_origin_and_any_target"] += 1
            elif c1_has_pair and not c2_has_pair:
                vis_stats["cam1_only_has_origin_and_any_target"] += 1
            elif c2_has_pair and not c1_has_pair:
                vis_stats["cam2_only_has_origin_and_any_target"] += 1
                if len(occlusion_events) < args.print_occlusion_events:
                    occlusion_events.append(
                        f"frame {frame_idx}: cam1 missing origin/any-target pair, cam2 has pair"
                    )
            else:
                vis_stats["neither_has_origin_and_any_target"] += 1
                if len(occlusion_events) < args.print_occlusion_events:
                    occlusion_events.append(
                        f"frame {frame_idx}: both cameras missing origin/any-target pair"
                    )
            if (
                args.origin_id in fused_eval
                and all(t not in cam1_seen for t in target_id_set)
                and all(t not in cam2_seen for t in target_id_set)
            ):
                vis_stats["both_cameras_missing_all_targets"] += 1

            v_cam1 = rel_vec(cam1_eval, args.origin_id, primary_target_id)
            v_cam2 = rel_vec(cam2_eval, args.origin_id, primary_target_id)
            v = rel_vec(fused_eval, args.origin_id, primary_target_id)
            if v is not None:
                fused_rel.append(v)
            if print_z_ids:
                if args.origin_id in fused_eval:
                    T_origin = fused_eval[args.origin_id]
                    parts = []
                    for tid in print_z_ids:
                        if tid in fused_eval:
                            z = rel_vec_from_T(T_origin, fused_eval[tid])[2]
                            parts.append(f"{tid}:{z:+.4f}m")
                        else:
                            parts.append(f"{tid}:N/A")
                    print(f"[frame {frame_idx}] z(origin {args.origin_id}) " + " ".join(parts))
                else:
                    print(f"[frame {frame_idx}] z(origin {args.origin_id}) origin missing")
            if print_margin_ids:
                m1_parts = []
                m2_parts = []
                for tid in print_margin_ids:
                    if tid in cam1_margin:
                        m1_parts.append(f"{tid}:{cam1_margin[tid]:.2f}")
                    else:
                        m1_parts.append(f"{tid}:N/A")
                    if tid in cam2_margin:
                        m2_parts.append(f"{tid}:{cam2_margin[tid]:.2f}")
                    else:
                        m2_parts.append(f"{tid}:N/A")
                print(
                    f"[frame {frame_idx}] margin cam1 " + " ".join(m1_parts)
                    + " | cam2 " + " ".join(m2_parts)
                )

            cv2.putText(
                img1,
                f"origin={args.origin_id} targets={target_ids}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if v_cam1 is not None:
                cv2.putText(
                    img1,
                    f"rel[{args.origin_id}->{primary_target_id}] [{v_cam1[0]:+.3f}, {v_cam1[1]:+.3f}, {v_cam1[2]:+.3f}]",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            else:
                cv2.putText(
                    img1,
                    f"rel[{args.origin_id}->{primary_target_id}] N/A",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.putText(
                img2,
                f"fused samples: {len(fused_rel)}/{args.num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if v_cam2 is not None:
                cv2.putText(
                    img2,
                    f"rel[{args.origin_id}->{primary_target_id}] [{v_cam2[0]:+.3f}, {v_cam2[1]:+.3f}, {v_cam2[2]:+.3f}]",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            else:
                cv2.putText(
                    img2,
                    f"rel[{args.origin_id}->{primary_target_id}] N/A",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            if v is not None:
                cv2.putText(
                    img2,
                    f"fused rel [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
            origin_label = "origin:direct" if fused_origin_source == args.origin_id else (
                f"origin:fallback({fused_origin_source})" if fused_origin_source is not None else "origin:missing"
            )
            cv2.putText(
                img2,
                origin_label,
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255) if fused_origin_source is not None else (0, 0, 255),
                1,
            )
            cv2.imshow("cam1", img1)
            cv2.imshow("cam2", img2)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        p1.stop()
        p2.stop()
        cv2.destroyAllWindows()

    if len(fused_rel) < 2:
        print("Not enough samples.")
        return

    fused_rel = np.asarray(fused_rel)
    dt = 1.0 / float(args.fps)
    kf_rel = kalman_filter_cv(
        fused_rel,
        dt=dt,
        process_var=args.kalman_process_var,
        meas_var=args.kalman_meas_var,
    )

    m_raw = compute_metrics(fused_rel, spike_k=args.spike_k)
    m_kf = compute_metrics(kf_rel, spike_k=args.spike_k)

    print("\n=== METRICS (fused raw vs Kalman) ===")
    print_metrics("fused_raw", m_raw)
    print_metrics("fused_kalman", m_kf)

    p95_improve = (m_raw.p95_jump - m_kf.p95_jump) / max(m_raw.p95_jump, 1e-9) * 100.0
    max_improve = (m_raw.max_jump - m_kf.max_jump) / max(m_raw.max_jump, 1e-9) * 100.0
    std_norm_improve = (m_raw.std_norm - m_kf.std_norm) / max(m_raw.std_norm, 1e-9) * 100.0
    print("\n=== IMPROVEMENT ===")
    print(f"  p95 jump reduction: {p95_improve:.2f}%")
    print(f"  max jump reduction: {max_improve:.2f}%")
    print(f"  std |rel| reduction: {std_norm_improve:.2f}%")

    total_frames = frame_idx
    if total_frames > 0:
        print("\n=== VISIBILITY / OCCLUSION ===")
        pair_keys = [
            "both_cameras_have_origin_and_any_target",
            "cam1_only_has_origin_and_any_target",
            "cam2_only_has_origin_and_any_target",
            "neither_has_origin_and_any_target",
        ]
        for k in pair_keys:
            v = vis_stats[k]
            print(f"  {k}: {v} ({100.0 * v / total_frames:.2f}%)")
        print("  -- supplemental --")
        for k in ["both_cameras_missing_all_targets", "origin_recovered_by_fallback", "origin_missing_even_after_fallback"]:
            v = vis_stats[k]
            print(f"  {k}: {v} ({100.0 * v / total_frames:.2f}%)")
        if occlusion_events:
            print("\n  sample occlusion events:")
            for e in occlusion_events:
                print(f"   - {e}")

    maybe_write_csv(args.csv_out, fused_rel, kf_rel)


if __name__ == "__main__":
    main()
