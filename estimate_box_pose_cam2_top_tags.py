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


def weighted_avg_rotation(rotations, weights):
    if len(rotations) == 1:
        return rotations[0]
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-9:
        w = np.ones(len(weights), dtype=float) / len(weights)
    else:
        w = np.asarray(weights, dtype=float) / w_sum
    R = np.zeros((3, 3), dtype=float)
    for rot, wi in zip(rotations, w):
        R += wi * rot
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


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
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


class RotMapAverager:
    def __init__(self):
        self.sum_R = {}
        self.count = {}

    def add(self, tag_id: int, R_0_i: np.ndarray):
        if tag_id not in self.sum_R:
            self.sum_R[tag_id] = np.array(R_0_i, dtype=float)
            self.count[tag_id] = 1
        else:
            self.sum_R[tag_id] += R_0_i
            self.count[tag_id] += 1

    def has(self, tag_id: int):
        return tag_id in self.sum_R and self.count.get(tag_id, 0) > 0

    def avg(self, tag_id: int):
        R = self.sum_R[tag_id] / float(self.count[tag_id])
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho

    def n(self, tag_id: int):
        return int(self.count.get(tag_id, 0))


class KalmanCV3D:
    def __init__(self, dt: float, process_var: float, meas_var: float):
        self.dt = float(dt)
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.Q = np.eye(6) * float(process_var)
        self.R = np.eye(3) * float(meas_var)
        self.P = np.eye(6) * 1.0
        self.x = None

    def update(self, z_xyz: np.ndarray):
        z = np.asarray(z_xyz, dtype=float).reshape(3, 1)
        if self.x is None:
            self.x = np.zeros((6, 1), dtype=float)
            self.x[:3, 0] = z.reshape(3)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3, 0].copy()


class Ema3D:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state = None

    def update(self, z_xyz: np.ndarray):
        z = np.asarray(z_xyz, dtype=float).reshape(3)
        if self.state is None:
            self.state = z.copy()
        else:
            self.state = self.alpha * z + (1.0 - self.alpha) * self.state
        return self.state.copy()


def print_series_metrics(name: str, series):
    if len(series) < 2:
        print(f"\n[{name}] not enough samples for metrics")
        return
    arr = np.asarray(series, dtype=float)
    mean_xyz = arr.mean(axis=0)
    std_xyz = arr.std(axis=0)
    norms = np.linalg.norm(arr, axis=1)
    jumps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    print(f"\n[{name}]")
    print(f"  n_samples: {len(arr)}")
    print(f"  mean xyz [m]: [{mean_xyz[0]:+.4f}, {mean_xyz[1]:+.4f}, {mean_xyz[2]:+.4f}]")
    print(f"  std  xyz [m]: [{std_xyz[0]:+.4f}, {std_xyz[1]:+.4f}, {std_xyz[2]:+.4f}]")
    print(f"  std |pos| [m]: {float(norms.std()):.5f}")
    print(f"  jump mean [m/frame]: {float(jumps.mean()):.5f}")
    print(f"  jump p95  [m/frame]: {float(np.percentile(jumps, 95)):.5f}")
    print(f"  jump max  [m/frame]: {float(jumps.max()):.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="115222071236")
    parser.add_argument("--calib", type=str, default="camera2_115222071236_calibration.npz")
    parser.add_argument("--origin-id", type=int, default=1)
    parser.add_argument("--primary-id", type=int, default=0, help="Preferred top-face tag ID for box pose")
    parser.add_argument("--fallback-ids", type=str, default="2,3,4,5")
    parser.add_argument("--primary-margin-min", type=float, default=50.0)
    parser.add_argument("--fallback-margin-min", type=float, default=40.0)
    parser.add_argument("--box-half-height", type=float, default=0.16, help="Subtract from tag z to get box center z [m]")
    parser.add_argument("--tag-size", type=float, default=0.077)
    parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
    parser.add_argument("--tag-size-map", type=str, default="")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop automatically after this many frames (0=until ESC)")
    parser.add_argument("--filter-mode", type=str, default="none", choices=["none", "kalman", "ema"])
    parser.add_argument("--kalman-process-var", type=float, default=0.05)
    parser.add_argument("--kalman-meas-var", type=float, default=0.01)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    args = parser.parse_args()

    fallback_ids = parse_int_list(args.fallback_ids)

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

    rot_map = RotMapAverager()  # stores R_primary_fallback
    kf = KalmanCV3D(
        dt=1.0 / float(args.fps),
        process_var=args.kalman_process_var,
        meas_var=args.kalman_meas_var,
    ) if args.filter_mode == "kalman" else None
    ema = Ema3D(alpha=args.ema_alpha) if args.filter_mode == "ema" else None
    frame_idx = 0
    filtered_series = []

    print("Box pose estimator (camera2 only)")
    print(f"origin_id={args.origin_id}, primary_id={args.primary_id}, fallback_ids={fallback_ids}")
    print(
        f"margin thresholds: primary>={args.primary_margin_min:.1f}, "
        f"fallback>={args.fallback_margin_min:.1f}"
    )
    print(f"box center z offset: +{args.box_half_height:.3f} m")
    print(f"filter_mode={args.filter_mode}")
    if args.filter_mode == "kalman":
        print(
            f"kalman enabled: process_var={args.kalman_process_var:.4f}, "
            f"meas_var={args.kalman_meas_var:.4f}"
        )
    elif args.filter_mode == "ema":
        print(f"ema enabled: alpha={args.ema_alpha:.3f}")
    print("Rule: use primary when visible+strong; otherwise use fallback tags with learned orientation mapping.")
    print("Press ESC to quit.")

    try:
        while True:
            frame_idx += 1
            color = pipe.wait_for_frames().get_color_frame()
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dets = detect_with_tag_sizes(detector, gray, cam_params, tag_size, tag_size_map)
            det_by_id = {d.tag_id: d for d in dets}

            for det in dets:
                if det.tag_id == args.origin_id:
                    draw_detection(img, det, (0, 255, 0))
                elif det.tag_id == args.primary_id:
                    draw_detection(img, det, (255, 255, 0))
                elif det.tag_id in fallback_ids:
                    draw_detection(img, det, (0, 255, 255))
                else:
                    draw_detection(img, det, (180, 180, 180))

            origin_det = det_by_id.get(args.origin_id)
            if origin_det is None:
                cv2.putText(img, f"origin {args.origin_id}: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("box pose cam2", img)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            T_cam_origin = pose_to_T(origin_det.pose_R, origin_det.pose_t)
            T_origin_cam = np.linalg.inv(T_cam_origin)

            T_origin = {}
            margin = {}
            for tid, det in det_by_id.items():
                T_origin[tid] = T_origin_cam @ pose_to_T(det.pose_R, det.pose_t)
                margin[tid] = float(getattr(det, "decision_margin", 0.0))

            # Update orientation mapping R_primary_fallback when primary is visible
            if args.primary_id in T_origin:
                R_o_p = T_origin[args.primary_id][:3, :3]
                for fid in fallback_ids:
                    if fid in T_origin and margin.get(fid, 0.0) >= args.fallback_margin_min:
                        R_o_f = T_origin[fid][:3, :3]
                        R_p_f = R_o_p.T @ R_o_f
                        rot_map.add(fid, R_p_f)

            mode = "none"
            used = []
            box_pos = None
            box_rot = None

            # 1) Primary tag 0 with strict margin
            if args.primary_id in T_origin and margin.get(args.primary_id, 0.0) >= args.primary_margin_min:
                mode = "primary"
                used = [args.primary_id]
                box_pos = T_origin[args.primary_id][:3, 3].copy()
                box_rot = T_origin[args.primary_id][:3, :3].copy()
            else:
                # 2) Fallback tags (2,3,4,5) with learned orientation mapping to primary
                pos_candidates = []
                rot_candidates = []
                ws = []
                for fid in fallback_ids:
                    if fid not in T_origin:
                        continue
                    if margin.get(fid, 0.0) < args.fallback_margin_min:
                        continue
                    if not rot_map.has(fid):
                        continue
                    T_o_f = T_origin[fid]
                    p_o_f = T_o_f[:3, 3]
                    R_o_f = T_o_f[:3, :3]
                    R_p_f = rot_map.avg(fid)
                    R_f_p = R_p_f.T
                    R_o_p_est = R_o_f @ R_f_p

                    pos_candidates.append(p_o_f)
                    rot_candidates.append(R_o_p_est)
                    ws.append(margin[fid])
                    used.append(fid)

                if len(pos_candidates) > 0:
                    mode = "fallback"
                    w = np.asarray(ws, dtype=float)
                    w_sum = float(np.sum(w))
                    if w_sum <= 1e-9:
                        wn = np.ones_like(w) / len(w)
                    else:
                        wn = w / w_sum
                    box_pos = np.sum(np.stack(pos_candidates, axis=0) * wn[:, None], axis=0)
                    box_rot = weighted_avg_rotation(rot_candidates, w)

            if box_pos is not None and box_rot is not None:
                # For this setup, "up" is negative z, so center is top-tag z + half-height.
                box_center = box_pos.copy()
                box_center[2] += args.box_half_height
                if kf is not None:
                    box_center_out = kf.update(box_center)
                elif ema is not None:
                    box_center_out = ema.update(box_center)
                else:
                    box_center_out = box_center
                filtered_series.append(box_center_out.copy())
                eul = rotation_to_euler_xyz_deg(box_rot)
                if frame_idx % max(1, args.print_every) == 0:
                    used_str = ",".join(map(str, used)) if used else "-"
                    print(
                        f"[frame {frame_idx}] mode={mode} used={used_str} "
                        f"pos=[{box_center_out[0]:+.4f},{box_center_out[1]:+.4f},{box_center_out[2]:+.4f}] "
                        f"euler_xyz_deg=[{eul[0]:+.2f},{eul[1]:+.2f},{eul[2]:+.2f}]"
                    )
                cv2.putText(
                    img,
                    f"mode={mode} used={used}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if mode == "primary" else (0, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"box center xyz [{box_center_out[0]:+.3f}, {box_center_out[1]:+.3f}, {box_center_out[2]:+.3f}]",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    img,
                    f"box euler xyz [{eul[0]:+.1f}, {eul[1]:+.1f}, {eul[2]:+.1f}]",
                    (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
            else:
                cv2.putText(
                    img,
                    "box pose: N/A (need tag0>=thr or mapped fallback tags)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("box pose cam2", img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    print_series_metrics(f"box_pose_{args.filter_mode}", filtered_series)


if __name__ == "__main__":
    main()
