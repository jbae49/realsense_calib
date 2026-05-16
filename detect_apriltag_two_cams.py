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

# =========================
# Camera serials
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--cam1-serial", type=str, default="935322072654", help="D435i serial")
parser.add_argument("--cam2-serial", type=str, default="115222071236", help="D435 serial")
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--tag-size", type=float, default=0.077, help="Default AprilTag size (m)")
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json",
                    help="JSON file for default/per-tag tag sizes")
parser.add_argument("--tag-size-map", type=str, default="",
                    help='Per-tag size map, e.g. "0:0.145,1:0.145"')
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

CAM1_SERIAL = args.cam1_serial
CAM2_SERIAL = args.cam2_serial

# =========================
# Calibration files
# =========================

CAM1_CALIB = args.cam1_calib
CAM2_CALIB = args.cam2_calib

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

# =========================
# Load calibration
# =========================

cam1_calib = np.load(CAM1_CALIB)
cam2_calib = np.load(CAM2_CALIB)

K1 = cam1_calib["camera_matrix"]
K2 = cam2_calib["camera_matrix"]

fx1, fy1 = K1[0, 0], K1[1, 1]
cx1, cy1 = K1[0, 2], K1[1, 2]

fx2, fy2 = K2[0, 0], K2[1, 1]
cx2, cy2 = K2[0, 2], K2[1, 2]

# =========================
# AprilTag detector
# =========================

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

# =========================
# Camera 1 pipeline
# =========================

pipeline1 = rs.pipeline()
config1 = rs.config()

config1.enable_device(CAM1_SERIAL)
config1.enable_stream(
    rs.stream.color,
    args.width,
    args.height,
    rs.format.bgr8,
    args.fps
)

pipeline1.start(config1)

# =========================
# Camera 2 pipeline
# =========================

pipeline2 = rs.pipeline()
config2 = rs.config()

config2.enable_device(CAM2_SERIAL)
config2.enable_stream(
    rs.stream.color,
    args.width,
    args.height,
    rs.format.bgr8,
    args.fps
)

pipeline2.start(config2)

print("Press ESC to quit.")
print(f"cam1={CAM1_SERIAL}, cam2={CAM2_SERIAL}, stream={args.width}x{args.height}@{args.fps}")
print(f"tag_config={args.tag_config}, default_tag_size={TAG_SIZE}, tag_size_map={TAG_SIZE_MAP}")

try:

    while True:

        # =====================
        # Camera 1 frame
        # =====================

        frames1 = pipeline1.wait_for_frames()
        color_frame1 = frames1.get_color_frame()

        img1 = np.asanyarray(color_frame1.get_data())

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        detections1 = detect_with_tag_sizes(
            detector,
            gray1,
            [fx1, fy1, cx1, cy1],
            TAG_SIZE,
            TAG_SIZE_MAP,
        )

        for det in detections1:

            corners = det.corners.astype(int)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])

                cv2.line(img1, p1, p2, (0, 255, 0), 2)

            center = tuple(det.center.astype(int))

            det_tag_size = TAG_SIZE_MAP.get(det.tag_id, TAG_SIZE)
            cv2.putText(
                img1,
                f"ID:{det.tag_id} size:{det_tag_size:.3f}",
                (center[0], center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            z = det.pose_t[2][0]

            print(f"[CAM1] Tag {det.tag_id} z={z:.3f} m")

        # =====================
        # Camera 2 frame
        # =====================

        frames2 = pipeline2.wait_for_frames()
        color_frame2 = frames2.get_color_frame()

        img2 = np.asanyarray(color_frame2.get_data())

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        detections2 = detect_with_tag_sizes(
            detector,
            gray2,
            [fx2, fy2, cx2, cy2],
            TAG_SIZE,
            TAG_SIZE_MAP,
        )

        for det in detections2:

            corners = det.corners.astype(int)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])

                cv2.line(img2, p1, p2, (0, 255, 255), 2)

            center = tuple(det.center.astype(int))

            det_tag_size = TAG_SIZE_MAP.get(det.tag_id, TAG_SIZE)
            cv2.putText(
                img2,
                f"ID:{det.tag_id} size:{det_tag_size:.3f}",
                (center[0], center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            z = det.pose_t[2][0]

            print(f"[CAM2] Tag {det.tag_id} z={z:.3f} m")

        # =====================
        # Show windows
        # =====================

        cv2.imshow("Camera 1", img1)
        cv2.imshow("Camera 2", img2)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

finally:

    pipeline1.stop()
    pipeline2.stop()

    cv2.destroyAllWindows()
