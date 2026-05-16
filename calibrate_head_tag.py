"""
Calibrate T_tag_torso: the fixed transform from AprilTag (on robot head) to torso_link.

How to use:
  1. Place robot in front of camera (standing or held still)
  2. Make sure the head AprilTag is visible
  3. This script shows the detected tag axes overlaid on the image
  4. You visually confirm the tag orientation matches expectations
  5. The script computes and saves T_tag_torso.npz

The transform is composed of:
  T_torso_tag = T_torso_headlink @ T_headlink_tagtop @ R_mounting
  T_tag_torso = inv(T_torso_tag)

Where:
  T_torso_headlink: URDF fixed joint (torso_link -> head_link)
  T_headlink_tagtop: mesh measurement (head_link origin -> head top center)
  R_mounting: tag orientation relative to head_link frame
"""
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

# =========================================================
# Config — EDIT THESE
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--head-tag-id", type=int, default=10)
parser.add_argument("--cam-serial", type=str, default="115222071236")
parser.add_argument("--cam-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--tag-size", type=float, default=0.077, help="Default AprilTag size in meters")
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

ROBOT_HEAD_TAG_ID = args.head_tag_id
CAM_SERIAL = args.cam_serial
CAM_CALIB = args.cam_calib

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

# =========================================================
# URDF-derived constants (from G1 mesh/URDF analysis)
# =========================================================

# torso_link -> head_link (URDF head_joint, fixed)
T_torso_headlink = np.eye(4)
T_torso_headlink[:3, 3] = [0.0039635, 0.0, -0.054]

# head_link origin -> head mesh top center (from STL analysis)
T_headlink_tagtop = np.eye(4)
T_headlink_tagtop[:3, 3] = [0.001, 0.0, 0.526]

# =========================================================
# Tag mounting rotation options
# =========================================================

def R_mounting_top_up_topforward():
    """Tag on top of head, facing UP, tag image "top" points to robot FRONT.

    AprilTag frame (looking at tag from camera):
      x = tag right, y = tag down, z = out of surface (toward camera)

    Robot frame (URDF):
      x = forward, y = left, z = up

    Mapping when tag is flat on head, tag's printed top = robot forward:
      robot_x (forward) = -tag_y (tag "up" = -y)
      robot_y (left)    = -tag_x (tag "left" = -x)
      robot_z (up)      = +tag_z (surface normal = up)
    """
    return np.array([
        [ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0,  1],
    ])


def R_mounting_top_up_topright():
    """Tag on top of head, facing UP, tag image "top" points to robot RIGHT.

    Mapping:
      robot_x (forward) = +tag_x
      robot_y (left)    = +tag_y ... wait, tag_y = down, robot_y = left
    Actually:
      robot_x (forward) = -tag_x (tag right ≠ forward)
    Let's derive properly:
      tag "top" = -tag_y direction → points to robot right (-robot_y)
      So: -tag_y = -robot_y → tag_y = robot_y ... no

    Easier: rotate the topforward case by -90° around z.
    """
    Rz = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1],
    ])
    return Rz @ R_mounting_top_up_topforward()


def R_mounting_front_facing_forward():
    """Tag on forehead, facing FORWARD (tag surface normal = robot forward).

    Mapping:
      robot_x (forward) = +tag_z (surface normal)
      robot_y (left)    = -tag_x (tag left)
      robot_z (up)      = -tag_y (tag up)
    """
    return np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0],
    ])


# =========================================================
# SELECT YOUR MOUNTING OPTION HERE
# =========================================================
R_mount = R_mounting_top_up_topforward()  # <-- CHANGE IF NEEDED (see options above)

# =========================================================
# Build T_torso_tag
# =========================================================

T_tagtop_tag = np.eye(4)
T_tagtop_tag[:3, :3] = R_mount

T_torso_tag = T_torso_headlink @ T_headlink_tagtop @ T_tagtop_tag
T_tag_torso = np.linalg.inv(T_torso_tag)

print("=== Computed transforms ===")
print(f"T_torso_tag translation: {T_torso_tag[:3, 3]}")
print(f"T_torso_tag rotation:\n{T_torso_tag[:3, :3]}")
print(f"\nT_tag_torso (what we save):")
print(f"  translation: {T_tag_torso[:3, 3]}")
print()

# =========================================================
# Utils
# =========================================================

def load_camera_params(path):
    data = np.load(path)
    K = data["camera_matrix"]
    return K, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]


def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def draw_axes(img, K, T_cam_frame, length=0.1):
    """Draw XYZ axes of a frame in camera image."""
    dist_coeffs = np.zeros(5)
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3]

    points_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ])

    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2).astype(int)

    origin = tuple(points_2d[0])
    cv2.line(img, origin, tuple(points_2d[1]), (0, 0, 255), 3)   # X = red
    cv2.line(img, origin, tuple(points_2d[2]), (0, 255, 0), 3)   # Y = green
    cv2.line(img, origin, tuple(points_2d[3]), (255, 0, 0), 3)   # Z = blue

    return origin


# =========================================================
# Camera + detector
# =========================================================

K, cam_params = load_camera_params(CAM_CALIB)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(CAM_SERIAL)
config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
pipeline.start(config)

print("\n======================================")
print(f"Looking for tag ID={ROBOT_HEAD_TAG_ID}")
print(f"stream={args.width}x{args.height}@{args.fps}")
print(f"tag_config={args.tag_config}, default_tag_size={TAG_SIZE}, tag_size_map={TAG_SIZE_MAP}")
print("You should see 3 sets of axes:")
print("  - TAG axes (at tag surface)")
print("  - TORSO axes (estimated torso_link)")
print("  Red=X  Green=Y  Blue=Z")
print()
print("Press 's' to SAVE calibration")
print("Press 'q' to quit without saving")
print("======================================")

saved = False

try:
    while True:
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detect_with_tag_sizes(detector, gray, cam_params, TAG_SIZE, TAG_SIZE_MAP)

        for det in dets:
            if det.tag_id != ROBOT_HEAD_TAG_ID:
                continue

            T_cam_tag = pose_to_T(det.pose_R, det.pose_t)
            T_cam_torso = T_cam_tag @ T_tag_torso

            # Draw tag corners
            corners = det.corners.astype(int)
            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(img, p1, p2, (0, 255, 255), 2)

            # Draw tag axes (small)
            draw_axes(img, K, T_cam_tag, length=0.05)
            cv2.putText(img, "TAG", (corners[0][0], corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw estimated torso axes (larger)
            torso_origin = draw_axes(img, K, T_cam_torso, length=0.15)
            if torso_origin:
                cv2.putText(img, "TORSO", (torso_origin[0] + 10, torso_origin[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Print info
            tag_pos = T_cam_tag[:3, 3]
            torso_pos = T_cam_torso[:3, 3]
            cv2.putText(img,
                        f"Tag: [{tag_pos[0]:.3f}, {tag_pos[1]:.3f}, {tag_pos[2]:.3f}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img,
                        f"Torso: [{torso_pos[0]:.3f}, {torso_pos[1]:.3f}, {torso_pos[2]:.3f}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Head Tag Calibration", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            np.savez("T_tag_torso.npz",
                     T_tag_torso=T_tag_torso,
                     T_torso_tag=T_torso_tag,
                     robot_head_tag_id=ROBOT_HEAD_TAG_ID,
                     tag_size=TAG_SIZE)
            print("\nSaved T_tag_torso.npz!")
            saved = True
            break
        elif key == ord('q') or key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

if not saved:
    print("Exited without saving.")
