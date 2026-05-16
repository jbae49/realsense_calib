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

parser = argparse.ArgumentParser()
parser.add_argument("--cam1-serial", type=str, default="935322072654")
parser.add_argument("--cam2-serial", type=str, default="115222071236")  # world origin
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
parser.add_argument("--tag-size", type=float, default=0.077, help="Default AprilTag size in meters")
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json",
                    help="JSON file for default/per-tag tag sizes")
parser.add_argument("--tag-size-map", type=str, default="",
                    help='Per-tag size map, e.g. "0:0.145,1:0.145"')
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

CAM1_SERIAL = args.cam1_serial
CAM2_SERIAL = args.cam2_serial
CAM1_CALIB = args.cam1_calib
CAM2_CALIB = args.cam2_calib
EXTRINSIC_FILE = args.extrinsic

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

def load_camera_params(path):
    data = np.load(path)
    K = data["camera_matrix"]
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def draw_detection(img, det, color, label):
    corners = det.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, color, 2)

    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    cv2.putText(
        img,
        label,
        (center[0] + 10, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

cam1_params = load_camera_params(CAM1_CALIB)
cam2_params = load_camera_params(CAM2_CALIB)

T_c2_c1 = np.load(EXTRINSIC_FILE)["T_c2_c1"]

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device(CAM1_SERIAL)
config1.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline1.start(config1)

pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device(CAM2_SERIAL)
config2.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline2.start(config2)

print("Camera2 frame is WORLD.")
print("World/OpenCV camera convention: x right, y down, z forward.")
print(f"stream={args.width}x{args.height}@{args.fps}")
print(f"tag_config={args.tag_config}, default_tag_size={TAG_SIZE}, tag_size_map={TAG_SIZE_MAP}")
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

        world_poses = []

        for det in dets1:
            T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
            T_world_tag = T_c2_c1 @ T_c1_tag

            p = T_world_tag[:3, 3]
            world_poses.append(("CAM1", det.tag_id, T_world_tag))

            draw_detection(
                img1,
                det,
                (0, 255, 0),
                f"ID:{det.tag_id}"
            )

            print(
                f"[CAM1->WORLD] Tag {det.tag_id}: "
                f"x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}"
            )

        for det in dets2:
            T_world_tag = pose_to_T(det.pose_R, det.pose_t)

            p = T_world_tag[:3, 3]
            world_poses.append(("CAM2", det.tag_id, T_world_tag))

            draw_detection(
                img2,
                det,
                (0, 255, 255),
                f"ID:{det.tag_id}"
            )

            print(
                f"[CAM2=WORLD] Tag {det.tag_id}: "
                f"x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}"
            )

        cv2.imshow("Camera 1", img1)
        cv2.imshow("Camera 2 / WORLD", img2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()
