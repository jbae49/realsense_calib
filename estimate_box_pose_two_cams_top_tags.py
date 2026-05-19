import argparse
import csv
import json
from pathlib import Path
from itertools import permutations, product

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
    if len(candidates) == 1:
        return candidates[0]["T"]
    rots = [c["T"][:3, :3] for c in candidates]
    poss = np.stack([c["T"][:3, 3] for c in candidates], axis=0)
    ws = np.asarray([c["w"] for c in candidates], dtype=float)
    R = weighted_avg_rotation(rots, ws)
    w_sum = float(np.sum(ws))
    if w_sum <= 1e-9:
        wn = np.ones_like(ws) / len(ws)
    else:
        wn = ws / w_sum
    p = np.sum(poss * wn[:, None], axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
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


def quat_wxyz_to_rotation(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_wxyz_normalize(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n > 1e-9:
        q = q / n
    return q


def rotation_angle_deg(R_a, R_b):
    R = R_a.T @ R_b
    c = float((np.trace(R) - 1.0) * 0.5)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def generate_axis_mappings():
    """
    Generate all right-handed signed permutation matrices (24 total),
    then group by where source +X is mapped (6 groups: ±X, ±Y, ±Z).
    """
    mats = []
    I = np.eye(3, dtype=float)
    for perm in permutations([0, 1, 2]):
        P = I[:, perm]
        for signs in product([-1.0, 1.0], repeat=3):
            S = np.diag(signs)
            M = S @ P
            if np.linalg.det(M) > 0.5:
                mats.append(M)
    axis_groups = {}
    axis_keys = [
        ("+X", np.array([1.0, 0.0, 0.0])),
        ("-X", np.array([-1.0, 0.0, 0.0])),
        ("+Y", np.array([0.0, 1.0, 0.0])),
        ("-Y", np.array([0.0, -1.0, 0.0])),
        ("+Z", np.array([0.0, 0.0, 1.0])),
        ("-Z", np.array([0.0, 0.0, -1.0])),
    ]
    for name, v in axis_keys:
        axis_groups[name] = [M for M in mats if np.allclose(M[:, 0], v)]
    return axis_groups


def generate_axis24_mappings():
    """
    All right-handed signed permutation matrices (24 total).
    """
    mats = []
    I = np.eye(3, dtype=float)
    for perm in permutations([0, 1, 2]):
        P = I[:, perm]
        for signs in product([-1.0, 1.0], repeat=3):
            S = np.diag(signs)
            M = S @ P
            if np.linalg.det(M) > 0.5:
                mats.append(M)
    return mats


def vec_to_axis_name(v):
    if np.allclose(v, [1.0, 0.0, 0.0]):
        return "+X"
    if np.allclose(v, [-1.0, 0.0, 0.0]):
        return "-X"
    if np.allclose(v, [0.0, 1.0, 0.0]):
        return "+Y"
    if np.allclose(v, [0.0, -1.0, 0.0]):
        return "-Y"
    if np.allclose(v, [0.0, 0.0, 1.0]):
        return "+Z"
    if np.allclose(v, [0.0, 0.0, -1.0]):
        return "-Z"
    return "?"


def matrix_to_mapping_name(M):
    # Columns indicate where source basis axes map in target frame.
    return (
        f"x->{vec_to_axis_name(M[:,0])},"
        f"y->{vec_to_axis_name(M[:,1])},"
        f"z->{vec_to_axis_name(M[:,2])}"
    )


def get_axis24_mapping_by_name(name: str):
    if not name:
        return None
    for M in generate_axis24_mappings():
        if matrix_to_mapping_name(M) == name:
            return M
    raise ValueError(f"Unknown axis24 mapping name: {name}")


def get_x_map_matrix(mode: str):
    if mode == "none":
        return np.eye(3, dtype=float)
    if mode == "-X":
        # Right-handed mapping with x->-x, y->+y, z->-z.
        return np.diag([-1.0, 1.0, -1.0])
    raise ValueError(f"Unsupported x-map mode: {mode}")


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
            f"[WARN] anchor config origin_id={cfg_origin_id} != --origin-id={origin_id}. Using entries anyway."
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
    out = dict(tag_dict)
    if origin_id in out:
        return out, origin_id
    for aid in fallback_anchor_ids:
        if aid in out and aid in anchor_map:
            T_anchor_origin = np.linalg.inv(anchor_map[aid])
            out[origin_id] = out[aid] @ T_anchor_origin
            return out, aid
    return out, None


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


class EmaQuat:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state = None  # wxyz

    def update(self, q_wxyz: np.ndarray):
        q = np.asarray(q_wxyz, dtype=float).reshape(4)
        n = np.linalg.norm(q)
        if n > 1e-9:
            q = q / n
        if self.state is None:
            self.state = q.copy()
            return self.state.copy()
        # Hemisphere alignment to avoid sign-flip discontinuity.
        if float(np.dot(self.state, q)) < 0.0:
            q = -q
        self.state = self.alpha * q + (1.0 - self.alpha) * self.state
        n2 = np.linalg.norm(self.state)
        if n2 > 1e-9:
            self.state /= n2
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
    parser.add_argument("--cam1-serial", type=str, default="935322072654")
    parser.add_argument("--cam2-serial", type=str, default="115222071236")
    parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
    parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
    parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
    parser.add_argument("--origin-id", type=int, default=1)
    parser.add_argument("--primary-id", type=int, default=0, help="Preferred top-face tag ID for box pose")
    parser.add_argument("--fallback-ids", type=str, default="2,3,4,5")
    parser.add_argument("--primary-margin-min", type=float, default=50.0)
    parser.add_argument("--fallback-margin-min", type=float, default=40.0)
    parser.add_argument("--box-half-height", type=float, default=0.16, help="Subtract from tag z to get box center z [m]")
    parser.add_argument("--tag-size", type=float, default=0.077)
    parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
    parser.add_argument("--tag-size-map", type=str, default="")
    parser.add_argument("--anchor-config", type=str, default="config/floor_anchor_transforms.json")
    parser.add_argument("--fallback-anchor-ids", type=str, default="")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop automatically after this many frames (0=until ESC)")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="If set, write per-frame box pose to CSV and suppress per-frame pose terminal print",
    )
    parser.add_argument(
        "--csv-axis24-out",
        type=str,
        default="",
        help="If set, write all 24 axis-mapping candidates per frame to CSV",
    )
    parser.add_argument("--filter-mode", type=str, default="none", choices=["none", "kalman", "ema"])
    parser.add_argument("--kalman-process-var", type=float, default=0.05)
    parser.add_argument("--kalman-meas-var", type=float, default=0.01)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--quat-order", type=str, default="wxyz", choices=["wxyz", "xyzw"])
    parser.add_argument("--apply-x-map", type=str, default="none", choices=["none", "-X"])
    parser.add_argument(
        "--print-axis6-candidates",
        action="store_true",
        help="Print 6 candidate mappings (our +X -> ±X/±Y/±Z) and orientation error to target quat",
    )
    parser.add_argument(
        "--print-axis24-candidates",
        action="store_true",
        help="Print all 24 right-handed axis mappings and orientation error to target quat",
    )
    parser.add_argument(
        "--target-ref-quat-wxyz",
        type=str,
        default="-0.0728,0.9474,0.3096,0.0354",
        help='Target reference object quat in wxyz, e.g. "-0.0728,0.9474,0.3096,0.0354"',
    )
    parser.add_argument(
        "--overlay-converted-map",
        type=str,
        default="",
        help='Optional axis24 mapping name to overlay converted quat, e.g. "x->-Y,y->-X,z->-Z"',
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize terminal output (viewer overlay still shown)",
    )
    args = parser.parse_args()

    fallback_ids = parse_int_list(args.fallback_ids)
    fallback_anchor_ids = parse_int_list(args.fallback_anchor_ids)

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

    rot_map = RotMapAverager()  # stores R_primary_fallback
    axis_map_groups = generate_axis_mappings() if args.print_axis6_candidates else None
    need_axis24 = bool(args.print_axis24_candidates or args.csv_axis24_out)
    axis24_maps = generate_axis24_mappings() if need_axis24 else None
    target_ref_quat = None
    target_ref_rot = None
    if args.print_axis6_candidates or need_axis24:
        target_ref_quat = quat_wxyz_normalize([float(x) for x in args.target_ref_quat_wxyz.split(",")])
        target_ref_rot = quat_wxyz_to_rotation(target_ref_quat)
    kf = KalmanCV3D(
        dt=1.0 / float(args.fps),
        process_var=args.kalman_process_var,
        meas_var=args.kalman_meas_var,
    ) if args.filter_mode == "kalman" else None
    ema = Ema3D(alpha=args.ema_alpha) if args.filter_mode == "ema" else None
    ema_q = EmaQuat(alpha=args.ema_alpha) if args.filter_mode == "ema" else None
    x_map = get_x_map_matrix(args.apply_x_map)
    frame_idx = 0
    filtered_series = []
    csv_rows = []
    axis24_csv_rows = []
    overlay_map = get_axis24_mapping_by_name(args.overlay_converted_map) if args.overlay_converted_map else None

    if not args.quiet:
        print("Box pose estimator (two cameras fused, C2 frame)")
        print(f"origin_id={args.origin_id}, primary_id={args.primary_id}, fallback_ids={fallback_ids}")
        print(
            f"margin thresholds: primary>={args.primary_margin_min:.1f}, "
            f"fallback>={args.fallback_margin_min:.1f}"
        )
        print(f"fallback_anchor_ids={fallback_anchor_ids}")
        print(f"box center z offset: +{args.box_half_height:.3f} m")
        print(f"filter_mode={args.filter_mode}")
        if args.filter_mode == "kalman":
            print(
                f"kalman enabled: process_var={args.kalman_process_var:.4f}, "
                f"meas_var={args.kalman_meas_var:.4f}"
            )
        elif args.filter_mode == "ema":
            print(f"ema enabled: alpha={args.ema_alpha:.3f}")
        print(f"quat_order={args.quat_order}")
        print(f"apply_x_map={args.apply_x_map}")
        if args.csv_out:
            print(f"csv_out={args.csv_out} (per-frame pose print suppressed)")
        if args.csv_axis24_out:
            print(f"csv_axis24_out={args.csv_axis24_out} (save all 24 axis candidates per frame)")
        if args.overlay_converted_map:
            print(f"overlay_converted_map={args.overlay_converted_map}")
        if args.print_axis6_candidates:
            print(f"axis6 compare target_ref_quat_wxyz={target_ref_quat.tolist()}")
        if need_axis24:
            print(f"axis24 compare target_ref_quat_wxyz={target_ref_quat.tolist()}")
        print("Rule: use primary when visible+strong; otherwise use fallback tags with learned orientation mapping.")
        print("Press ESC to quit.")

    try:
        while True:
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
            for det in dets1:
                draw_detection(img1, det, (0, 255, 255))
                T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
                T_c2_tag = T_c2_c1 @ T_c1_tag
                w = float(getattr(det, "decision_margin", 1.0))
                fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(1e-3, w)})

            for det in dets2:
                draw_detection(img2, det, (0, 255, 255))
                T_c2_tag = pose_to_T(det.pose_R, det.pose_t)
                w = float(getattr(det, "decision_margin", 1.0))
                fused_candidates.setdefault(det.tag_id, []).append({"T": T_c2_tag, "w": max(1e-3, w)})

            if not fused_candidates:
                cv2.putText(img2, "no tags", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("cam1", img1)
                cv2.imshow("cam2", img2)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            fused_tags_c2 = {tid: fuse_tag_pose(cands) for tid, cands in fused_candidates.items()}
            fused_margin = {tid: max(c["w"] for c in cands) for tid, cands in fused_candidates.items()}

            fused_eval, origin_source = inject_origin_from_fallback(
                fused_tags_c2, args.origin_id, fallback_anchor_ids, anchor_map
            )
            if args.origin_id not in fused_eval:
                cv2.putText(img2, f"origin {args.origin_id}: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("cam1", img1)
                cv2.imshow("cam2", img2)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            T_o_c2 = np.linalg.inv(fused_eval[args.origin_id])
            T_o_tag = {tid: T_o_c2 @ T for tid, T in fused_eval.items()}

            # Update orientation mapping R_primary_fallback when primary is visible
            if args.primary_id in T_o_tag:
                R_o_p = T_o_tag[args.primary_id][:3, :3]
                for fid in fallback_ids:
                    if fid in T_o_tag and fused_margin.get(fid, 0.0) >= args.fallback_margin_min:
                        R_o_f = T_o_tag[fid][:3, :3]
                        R_p_f = R_o_p.T @ R_o_f
                        rot_map.add(fid, R_p_f)

            mode = "none"
            used = []
            box_pos = None
            box_rot = None

            if args.primary_id in T_o_tag and fused_margin.get(args.primary_id, 0.0) >= args.primary_margin_min:
                mode = "primary"
                used = [args.primary_id]
                box_pos = T_o_tag[args.primary_id][:3, 3].copy()
                box_rot = T_o_tag[args.primary_id][:3, :3].copy()
            else:
                pos_candidates = []
                rot_candidates = []
                ws = []
                for fid in fallback_ids:
                    if fid not in T_o_tag:
                        continue
                    if fused_margin.get(fid, 0.0) < args.fallback_margin_min:
                        continue
                    if not rot_map.has(fid):
                        continue
                    T_o_f = T_o_tag[fid]
                    p_o_f = T_o_f[:3, 3]
                    R_o_f = T_o_f[:3, :3]
                    R_p_f = rot_map.avg(fid)
                    R_f_p = R_p_f.T
                    R_o_p_est = R_o_f @ R_f_p
                    pos_candidates.append(p_o_f)
                    rot_candidates.append(R_o_p_est)
                    ws.append(fused_margin[fid])
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
                box_center = box_pos.copy()
                # For this setup, "up" is negative z, so center is top-tag z + half-height.
                box_center[2] += args.box_half_height
                if kf is not None:
                    box_center_out = kf.update(box_center)
                elif ema is not None:
                    box_center_out = ema.update(box_center)
                else:
                    box_center_out = box_center
                filtered_series.append(box_center_out.copy())
                q_raw = rotation_to_quat_wxyz(box_rot)
                q_out_wxyz = ema_q.update(q_raw) if ema_q is not None else q_raw
                R_out = quat_wxyz_to_rotation(q_out_wxyz)
                R_disp = x_map @ R_out
                q_disp_wxyz = rotation_to_quat_wxyz(R_disp)
                eul = rotation_to_euler_xyz_deg(R_disp)
                if args.quat_order == "xyzw":
                    q_print = np.array([q_disp_wxyz[1], q_disp_wxyz[2], q_disp_wxyz[3], q_disp_wxyz[0]])
                else:
                    q_print = q_disp_wxyz
                if args.csv_out:
                    used_str = ",".join(map(str, used)) if used else "-"
                    csv_rows.append(
                        {
                            "frame_idx": frame_idx,
                            "origin_source": origin_source if origin_source is not None else "",
                            "mode": mode,
                            "used_tags": used_str,
                            "obj_pos_x": float(box_center_out[0]),
                            "obj_pos_y": float(box_center_out[1]),
                            "obj_pos_z": float(box_center_out[2]),
                            "obj_quat_w": float(q_disp_wxyz[0]),
                            "obj_quat_x": float(q_disp_wxyz[1]),
                            "obj_quat_y": float(q_disp_wxyz[2]),
                            "obj_quat_z": float(q_disp_wxyz[3]),
                            "obj_euler_x_deg": float(eul[0]),
                            "obj_euler_y_deg": float(eul[1]),
                            "obj_euler_z_deg": float(eul[2]),
                            "filter_mode": args.filter_mode,
                            "quat_order": args.quat_order,
                            "apply_x_map": args.apply_x_map,
                        }
                    )
                should_print_frame = args.print_every > 0 and frame_idx % args.print_every == 0
                if (not args.quiet) and should_print_frame and (not args.csv_out or args.print_axis6_candidates or args.print_axis24_candidates):
                    used_str = ",".join(map(str, used)) if used else "-"
                    if not args.csv_out:
                        print(
                            f"[frame {frame_idx}] origin_src={origin_source} mode={mode} used={used_str} "
                            f"obj_pos=[{box_center_out[0]:+.4f},{box_center_out[1]:+.4f},{box_center_out[2]:+.4f}] "
                            f"obj_quat_{args.quat_order}=[{q_print[0]:+.5f},{q_print[1]:+.5f},{q_print[2]:+.5f},{q_print[3]:+.5f}] "
                            f"euler_xyz_deg=[{eul[0]:+.2f},{eul[1]:+.2f},{eul[2]:+.2f}]"
                        )
                    elif args.print_axis6_candidates or args.print_axis24_candidates:
                        print(f"[frame {frame_idx}] origin_src={origin_source} mode={mode} used={used_str}")
                    if args.print_axis6_candidates:
                        cand_parts = []
                        for name in ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]:
                            best = None
                            for M in axis_map_groups[name]:
                                R_cand = M @ R_out
                                err = rotation_angle_deg(target_ref_rot, R_cand)
                                if best is None or err < best["err"]:
                                    q_c = rotation_to_quat_wxyz(R_cand)
                                    best = {"err": err, "q": q_c}
                            cand_parts.append(
                                f"x->{name}: err={best['err']:.2f}deg q=[{best['q'][0]:+.3f},{best['q'][1]:+.3f},{best['q'][2]:+.3f},{best['q'][3]:+.3f}]"
                            )
                        print("  axis6 best-per-x-map | " + " | ".join(cand_parts))
                    if args.print_axis24_candidates:
                        cand24 = []
                        for M in axis24_maps:
                            R_cand = M @ R_out
                            err = rotation_angle_deg(target_ref_rot, R_cand)
                            cand24.append((err, M, R_cand))
                        cand24.sort(key=lambda x: x[0])
                        # Print all 24, sorted by error (best first).
                        print("  axis24 all-candidates (sorted by err):")
                        for idx, (err, M, R_cand) in enumerate(cand24, start=1):
                            q_c = rotation_to_quat_wxyz(R_cand)
                            print(
                                f"    {idx:02d}) {matrix_to_mapping_name(M)} "
                                f"err={err:.2f}deg "
                                f"q=[{q_c[0]:+.3f},{q_c[1]:+.3f},{q_c[2]:+.3f},{q_c[3]:+.3f}]"
                            )
                if need_axis24:
                    cand24 = []
                    for M in axis24_maps:
                        R_cand = M @ R_out
                        err = rotation_angle_deg(target_ref_rot, R_cand)
                        q_c = rotation_to_quat_wxyz(R_cand)
                        cand24.append((err, M, q_c))
                    cand24.sort(key=lambda x: x[0])
                    if args.csv_axis24_out:
                        used_str = ",".join(map(str, used)) if used else "-"
                        for rank, (err, M, q_c) in enumerate(cand24, start=1):
                            axis24_csv_rows.append(
                                {
                                    "frame_idx": frame_idx,
                                    "origin_source": origin_source if origin_source is not None else "",
                                    "mode": mode,
                                    "used_tags": used_str,
                                    "rank": rank,
                                    "mapping_name": matrix_to_mapping_name(M),
                                    "err_deg": float(err),
                                    "quat_w": float(q_c[0]),
                                    "quat_x": float(q_c[1]),
                                    "quat_y": float(q_c[2]),
                                    "quat_z": float(q_c[3]),
                                    "target_ref_quat_wxyz": args.target_ref_quat_wxyz,
                                    "filter_mode": args.filter_mode,
                                    "apply_x_map": args.apply_x_map,
                                }
                            )
                cv2.putText(
                    img2,
                    f"origin_src={origin_source} mode={mode} used={used} xmap={args.apply_x_map}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if mode == "primary" else (0, 255, 255),
                    2,
                )
                cv2.putText(
                    img2,
                    f"obj_pos xyz [{box_center_out[0]:+.3f}, {box_center_out[1]:+.3f}, {box_center_out[2]:+.3f}]",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    img2,
                    f"obj_quat {args.quat_order}: [{q_print[0]:+.3f}, {q_print[1]:+.3f}, {q_print[2]:+.3f}, {q_print[3]:+.3f}]",
                    (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    img2,
                    f"obj_euler xyz [{eul[0]:+.1f}, {eul[1]:+.1f}, {eul[2]:+.1f}]",
                    (10, 101),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                )
                if overlay_map is not None:
                    R_conv = overlay_map @ R_out
                    q_conv_wxyz = rotation_to_quat_wxyz(R_conv)
                    if args.quat_order == "xyzw":
                        q_conv = np.array(
                            [q_conv_wxyz[1], q_conv_wxyz[2], q_conv_wxyz[3], q_conv_wxyz[0]]
                        )
                    else:
                        q_conv = q_conv_wxyz
                    cv2.putText(
                        img2,
                        f"converted_quat {args.quat_order}: [{q_conv[0]:+.3f}, {q_conv[1]:+.3f}, {q_conv[2]:+.3f}, {q_conv[3]:+.3f}]",
                        (10, 124),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )
            else:
                cv2.putText(
                    img2,
                    "box pose: N/A (need tag0>=thr or mapped fallback tags)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("cam1", img1)
            cv2.imshow("cam2", img2)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        p1.stop()
        p2.stop()
        cv2.destroyAllWindows()
    if args.csv_out:
        csv_path = Path(args.csv_out)
        if csv_path.parent and not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "frame_idx",
            "origin_source",
            "mode",
            "used_tags",
            "obj_pos_x",
            "obj_pos_y",
            "obj_pos_z",
            "obj_quat_w",
            "obj_quat_x",
            "obj_quat_y",
            "obj_quat_z",
            "obj_euler_x_deg",
            "obj_euler_y_deg",
            "obj_euler_z_deg",
            "filter_mode",
            "quat_order",
            "apply_x_map",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if csv_rows:
                writer.writerows(csv_rows)
        if not args.quiet:
            print(f"Wrote {len(csv_rows)} rows to {csv_path}")
    if args.csv_axis24_out:
        axis24_path = Path(args.csv_axis24_out)
        if axis24_path.parent and not axis24_path.parent.exists():
            axis24_path.parent.mkdir(parents=True, exist_ok=True)
        axis24_fieldnames = [
            "frame_idx",
            "origin_source",
            "mode",
            "used_tags",
            "rank",
            "mapping_name",
            "err_deg",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
            "target_ref_quat_wxyz",
            "filter_mode",
            "apply_x_map",
        ]
        with axis24_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=axis24_fieldnames)
            writer.writeheader()
            if axis24_csv_rows:
                writer.writerows(axis24_csv_rows)
        if not args.quiet:
            print(f"Wrote {len(axis24_csv_rows)} rows to {axis24_path}")
    if not args.quiet:
        print_series_metrics(f"box_pose_{args.filter_mode}", filtered_series)


if __name__ == "__main__":
    main()
