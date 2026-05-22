import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
import argparse
from utils.apriltag_config import (
    parse_tag_size_map,
    load_tag_size_config,
    merge_tag_sizes,
)

# =========================================================
# Argument parser
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tag-id",
    type=int,
    default=1,
    help="AprilTag ID used for extrinsic calibration"
)

parser.add_argument(
    "--num-samples",
    type=int,
    default=100,
    help="Number of samples to average"
)
parser.add_argument(
    "--tag-size",
    type=float,
    default=0.077,
    help="Default AprilTag size in meters"
)
parser.add_argument(
    "--tag-config",
    type=str,
    default="config/tag_sizes.json",
    help="JSON file for default/per-tag tag sizes"
)
parser.add_argument(
    "--tag-size-map",
    type=str,
    default="",
    help='Per-tag size map, e.g. "0:0.145,1:0.145"'
)
parser.add_argument(
    "--width",
    type=int,
    default=640,
    help="Color stream width"
)
parser.add_argument(
    "--height",
    type=int,
    default=480,
    help="Color stream height"
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Color stream FPS"
)
parser.add_argument(
    "--cam1-serial",
    type=str,
    default="935322072654",
    help="Source camera serial (mapped into cam2/world frame)"
)
parser.add_argument(
    "--cam2-serial",
    type=str,
    default="115222071236",
    help="World/reference camera serial"
)
parser.add_argument(
    "--cam1-calib",
    type=str,
    default="camera1_935322072654_calibration.npz",
    help="Calibration npz for cam1"
)
parser.add_argument(
    "--cam2-calib",
    type=str,
    default="camera2_115222071236_calibration.npz",
    help="Calibration npz for cam2"
)
parser.add_argument(
    "--output",
    type=str,
    default="camera1_to_camera2_extrinsic.npz",
    help="Output npz path for extrinsic (key: T_c2_c1)"
)

args = parser.parse_args()

TARGET_TAG_ID = args.tag_id
NUM_SAMPLES = args.num_samples

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
_, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)
TAG_SIZE = TAG_SIZE_MAP.get(TARGET_TAG_ID, tag_default)

CAM1_SERIAL = args.cam1_serial
CAM2_SERIAL = args.cam2_serial
CAM1_CALIB = args.cam1_calib
CAM2_CALIB = args.cam2_calib
OUTPUT_FILE = args.output

# =========================================================
# Utility functions
# =========================================================

def load_camera_params(path):
    data = np.load(path)
    K = data["camera_matrix"]

    return [
        K[0, 0],
        K[1, 1],
        K[0, 2],
        K[1, 2]
    ]

def pose_to_T(R, t):
    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)

    return T

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
        0.7,
        color,
        2
    )

def find_target_detection(detections, tag_id):

    for det in detections:

        if det.tag_id == tag_id:
            return det

    return None

# =========================================================
# Load calibration
# =========================================================

cam1_params = load_camera_params(CAM1_CALIB)
cam2_params = load_camera_params(CAM2_CALIB)

# =========================================================
# AprilTag detector
# =========================================================

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

# =========================================================
# Camera 1 pipeline
# =========================================================

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

# =========================================================
# Camera 2 pipeline
# =========================================================

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

# =========================================================
# Main
# =========================================================

T_c2_c1_samples = []

print("======================================")
print("Camera2 is WORLD origin")
print(f"Target tag ID : {TARGET_TAG_ID}")
print(f"Collecting    : {NUM_SAMPLES} samples")
print(f"Tag size      : {TAG_SIZE:.3f} m (for target ID)")
print(f"Stream        : {args.width}x{args.height}@{args.fps}")
print("Keep cameras and tag fixed")
print("Press ESC to quit")
print("======================================")

try:

    while len(T_c2_c1_samples) < NUM_SAMPLES:

        # -------------------------------------------------
        # Frames
        # -------------------------------------------------

        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()

        img1 = np.asanyarray(
            frames1.get_color_frame().get_data()
        )

        img2 = np.asanyarray(
            frames2.get_color_frame().get_data()
        )

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------
        # Detect tags
        # -------------------------------------------------

        dets1 = detector.detect(
            gray1,
            estimate_tag_pose=True,
            camera_params=cam1_params,
            tag_size=TAG_SIZE
        )

        dets2 = detector.detect(
            gray2,
            estimate_tag_pose=True,
            camera_params=cam2_params,
            tag_size=TAG_SIZE
        )

        det1 = find_target_detection(
            dets1,
            TARGET_TAG_ID
        )

        det2 = find_target_detection(
            dets2,
            TARGET_TAG_ID
        )

        # -------------------------------------------------
        # Draw detections
        # -------------------------------------------------

        if det1 is not None:
            draw_detection(
                img1,
                det1,
                (0, 255, 0)
            )

        if det2 is not None:
            draw_detection(
                img2,
                det2,
                (0, 255, 255)
            )

        # -------------------------------------------------
        # Compute extrinsic
        # -------------------------------------------------

        if det1 is not None and det2 is not None:

            T_c1_tag = pose_to_T(
                det1.pose_R,
                det1.pose_t
            )

            T_c2_tag = pose_to_T(
                det2.pose_R,
                det2.pose_t
            )

            # Camera1 -> Camera2/world
            T_c2_c1 = (
                T_c2_tag @
                np.linalg.inv(T_c1_tag)
            )

            T_c2_c1_samples.append(T_c2_c1)

            # ---------------------------------------------
            # Sample text
            # ---------------------------------------------

            cv2.putText(
                img1,
                f"samples: {len(T_c2_c1_samples)}/{NUM_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img2,
                f"samples: {len(T_c2_c1_samples)}/{NUM_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            print(
                f"Collected "
                f"{len(T_c2_c1_samples)}/"
                f"{NUM_SAMPLES}"
            )

        # -------------------------------------------------
        # Show windows
        # -------------------------------------------------

        cv2.imshow("Camera 1", img1)

        cv2.imshow(
            "Camera 2 / WORLD",
            img2
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

finally:

    pipeline1.stop()
    pipeline2.stop()

    cv2.destroyAllWindows()

# =========================================================
# Save result
# =========================================================

if len(T_c2_c1_samples) == 0:

    print("No valid samples collected.")
    exit()

T_stack = np.stack(
    T_c2_c1_samples,
    axis=0
)

T_avg = np.mean(
    T_stack,
    axis=0
)

# ---------------------------------------------------------
# Re-orthogonalize rotation matrix
# ---------------------------------------------------------

U, _, Vt = np.linalg.svd(
    T_avg[:3, :3]
)

R_avg = U @ Vt

if np.linalg.det(R_avg) < 0:

    U[:, -1] *= -1

    R_avg = U @ Vt

T_avg[:3, :3] = R_avg

T_avg[3, :] = np.array(
    [0, 0, 0, 1]
)

# =========================================================
# Print result
# =========================================================

print("\n======================================")
print("Extrinsic Result")
print("======================================")

print("\nT_c2_c1:")
print(T_avg)

print("\nTranslation Camera1 in Camera2/world:")
print(T_avg[:3, 3])

# =========================================================
# Save
# =========================================================

np.savez(
    OUTPUT_FILE,

    T_c2_c1=T_avg,

    cam1_serial=CAM1_SERIAL,
    cam2_serial=CAM2_SERIAL,

    target_tag_id=TARGET_TAG_ID,

    tag_size=TAG_SIZE,

    num_samples=len(T_c2_c1_samples)
)

print("\nSaved:")
print(OUTPUT_FILE)
