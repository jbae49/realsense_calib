import argparse
import json
from datetime import datetime, timezone
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


def weighted_avg_rotation(rotations):
    R = np.mean(np.stack(rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def find_by_id(detections, tag_id):
    for det in detections:
        if det.tag_id == tag_id:
            return det
    return None


def draw_det(img, det, color):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, required=True, help="Camera serial used for anchor calibration")
    parser.add_argument("--calib", type=str, required=True, help="Camera calibration .npz")
    parser.add_argument("--origin-id", type=int, default=0)
    parser.add_argument("--anchor-id", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--tag-size", type=float, default=0.077)
    parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
    parser.add_argument("--tag-size-map", type=str, default="")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--out-config", type=str, default="config/floor_anchor_transforms.json")
    args = parser.parse_args()

    if args.anchor_id == args.origin_id:
        raise ValueError("anchor-id must be different from origin-id")

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

    samples = []

    print("======================================")
    print("Calibrate floor anchor transform")
    print(f"origin_id={args.origin_id}, anchor_id={args.anchor_id}")
    print(f"num_samples={args.num_samples}, stream={args.width}x{args.height}@{args.fps}")
    print(f"output={args.out_config}")
    print("Keep both floor tags fixed and visible.")
    print("Press ESC to stop early.")
    print("======================================")

    try:
        while len(samples) < args.num_samples:
            color = pipe.wait_for_frames().get_color_frame()
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dets = detect_with_tag_sizes(detector, gray, cam_params, tag_size, tag_size_map)
            origin_det = find_by_id(dets, args.origin_id)
            anchor_det = find_by_id(dets, args.anchor_id)

            if origin_det is not None:
                draw_det(img, origin_det, (0, 255, 0))
            if anchor_det is not None:
                draw_det(img, anchor_det, (0, 255, 255))

            if origin_det is not None and anchor_det is not None:
                T_cam_origin = pose_to_T(origin_det.pose_R, origin_det.pose_t)
                T_cam_anchor = pose_to_T(anchor_det.pose_R, anchor_det.pose_t)
                T_origin_anchor = np.linalg.inv(T_cam_origin) @ T_cam_anchor
                samples.append(T_origin_anchor)

            cv2.putText(
                img,
                f"samples: {len(samples)}/{args.num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("floor_anchor_calibration", img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    if len(samples) == 0:
        print("No valid samples collected (origin+anchor both visible required).")
        return

    t_stack = np.stack([s[:3, 3] for s in samples], axis=0)
    r_list = [s[:3, :3] for s in samples]
    t_avg = np.mean(t_stack, axis=0)
    R_avg = weighted_avg_rotation(r_list)

    T_origin_anchor = np.eye(4)
    T_origin_anchor[:3, :3] = R_avg
    T_origin_anchor[:3, 3] = t_avg

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if out_path.exists():
        data = json.loads(out_path.read_text())
    if not isinstance(data, dict):
        data = {}

    data["origin_id"] = args.origin_id
    anchors = data.get("anchors")
    if not isinstance(anchors, dict):
        anchors = {}

    anchors[str(args.anchor_id)] = {
        "T_origin_anchor": T_origin_anchor.tolist(),
        "num_samples": len(samples),
        "source": {
            "serial": args.serial,
            "calib": args.calib,
            "stream": {"width": args.width, "height": args.height, "fps": args.fps},
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    data["anchors"] = anchors

    out_path.write_text(json.dumps(data, indent=2))

    print("\nT_origin_anchor:")
    print(T_origin_anchor)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
