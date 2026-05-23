import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
import argparse
from utils.apriltag_config import (
    parse_tag_size_map,
    load_tag_size_config,
    merge_tag_sizes,
    detect_with_tag_sizes,
)

# =========================================================
# Args
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--ref-tag-id", type=int, default=0)
parser.add_argument("--num-samples", type=int, default=100)
parser.add_argument("--cam1-serial", type=str, default="935322072654")
parser.add_argument("--cam2-serial", type=str, default="115222071236")
parser.add_argument("--cam1-calib", type=str, default="camera1_935322072654_calibration.npz")
parser.add_argument("--cam2-calib", type=str, default="camera2_115222071236_calibration.npz")
parser.add_argument("--extrinsic", type=str, default="camera1_to_camera2_extrinsic.npz")
parser.add_argument("--tag-size", type=float, default=0.077, help="Default AprilTag size in meters")
parser.add_argument("--tag-config", type=str, default="config/tag_sizes.json")
parser.add_argument("--tag-size-map", type=str, default="")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

REF_TAG_ID = args.ref_tag_id
NUM_SAMPLES = args.num_samples

# =========================================================
# Settings
# =========================================================

CAM1_SERIAL = args.cam1_serial
CAM2_SERIAL = args.cam2_serial  # Camera2 = world origin

CAM1_CALIB = args.cam1_calib
CAM2_CALIB = args.cam2_calib
EXTRINSIC_FILE = args.extrinsic

cfg_default_size, cfg_tag_size_map = load_tag_size_config(args.tag_config)
tag_default = cfg_default_size if cfg_default_size is not None else args.tag_size
cli_tag_size_map = parse_tag_size_map(args.tag_size_map)
TAG_SIZE, TAG_SIZE_MAP = merge_tag_sizes(tag_default, cfg_tag_size_map, cli_tag_size_map)

BOX_TAG_IDS = [0, 1, 2, 3, 4, 5]

# Box dimensions [m]
# User box is a 34 × 34 × 34 cm cube. (Earlier values 0.37/0.32/0.29 were
# rough guesses and only affected: (a) the z-coordinate of tag 0 in the box
# body frame via REF_TAG_TO_BOX_TRANSLATION below, and (b) the dims drawn
# in the GUI bbox. The xy positions of side-face tags are still measured
# directly from detections so they are unaffected.)
BOX_W = 0.34
BOX_D = 0.34
BOX_H = 0.34

# ID 0 = top face.
# box center is H/2 "below" tag 0 in tag0 local frame.
# If center appears above the box later, change -BOX_H/2 to +BOX_H/2.
REF_TAG_TO_BOX_TRANSLATION = np.array([0.0, 0.0, +BOX_H / 2.0])

# =========================================================
# Utils
# =========================================================

def load_camera_params(path):
    data = np.load(path)
    K = data["camera_matrix"]
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

def pose_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def make_translation_T(offset):
    T = np.eye(4)
    T[:3, 3] = offset
    return T

def invert_T(T):
    return np.linalg.inv(T)

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

def average_transforms(T_list):
    T_stack = np.stack(T_list, axis=0)
    T_avg = np.mean(T_stack, axis=0)

    U, _, Vt = np.linalg.svd(T_avg[:3, :3])
    R_avg = U @ Vt

    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    T_avg[:3, :3] = R_avg
    T_avg[3, :] = np.array([0, 0, 0, 1])
    return T_avg

# =========================================================
# Load calibration/extrinsic
# =========================================================

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

# =========================================================
# Camera pipelines
# =========================================================

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

# tag_id -> list of box_T_tag samples
box_T_tag_samples = {tag_id: [] for tag_id in BOX_TAG_IDS}

print("======================================")
print("Registering box tag map")
print(f"Reference tag ID: {REF_TAG_ID}")
print(f"Samples: {NUM_SAMPLES}")
print("Camera2 frame is WORLD")
print(f"stream={args.width}x{args.height}@{args.fps}")
print(f"tag_config={args.tag_config}, default_tag_size={TAG_SIZE}, tag_size_map={TAG_SIZE_MAP}")
print("Keep box and cameras fixed")
print("Press ESC to quit early")
print("======================================")

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

        # -------------------------------------------------
        # Convert all detections to world(Camera2) frame
        # -------------------------------------------------

        world_T_tags = {}

        for det in dets1:
            if det.tag_id not in BOX_TAG_IDS:
                continue

            T_c1_tag = pose_to_T(det.pose_R, det.pose_t)
            T_world_tag = T_c2_c1 @ T_c1_tag

            world_T_tags[det.tag_id] = T_world_tag
            draw_detection(img1, det, (0, 255, 0))

        for det in dets2:
            if det.tag_id not in BOX_TAG_IDS:
                continue

            T_world_tag = pose_to_T(det.pose_R, det.pose_t)

            # If same tag was seen by cam1 and cam2, prefer cam2 for now.
            world_T_tags[det.tag_id] = T_world_tag
            draw_detection(img2, det, (0, 255, 255))

        # -------------------------------------------------
        # Need reference tag
        # -------------------------------------------------

        if REF_TAG_ID in world_T_tags:
            world_T_ref = world_T_tags[REF_TAG_ID]

            # Define current box frame from reference tag.
            # world_T_box = world_T_ref @ ref_tag_T_box
            ref_tag_T_box = make_translation_T(REF_TAG_TO_BOX_TRANSLATION)
            world_T_box = world_T_ref @ ref_tag_T_box

            box_T_world = invert_T(world_T_box)

            # Register every visible tag relative to box frame.
            for tag_id, world_T_tag in world_T_tags.items():
                box_T_tag = box_T_world @ world_T_tag
                box_T_tag_samples[tag_id].append(box_T_tag)

            n_ref_samples = len(box_T_tag_samples[REF_TAG_ID])

            cv2.putText(
                img1,
                f"ref samples: {n_ref_samples}/{NUM_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img2,
                f"ref samples: {n_ref_samples}/{NUM_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            print(
                f"Collected ref {n_ref_samples}/{NUM_SAMPLES}, "
                f"visible tags: {sorted(world_T_tags.keys())}"
            )

            if n_ref_samples >= NUM_SAMPLES:
                break

        cv2.imshow("Camera 1", img1)
        cv2.imshow("Camera 2 / WORLD", img2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()

# =========================================================
# Average and save map
# =========================================================

box_T_tag_map = {}

print("\n======================================")
print("Registered tag map")
print("======================================")

for tag_id, samples in box_T_tag_samples.items():
    if len(samples) == 0:
        print(f"Tag {tag_id}: no samples")
        continue

    T_avg = average_transforms(samples)
    box_T_tag_map[tag_id] = T_avg

    print(f"\nTag {tag_id}: {len(samples)} samples")
    print("box_T_tag:")
    print(T_avg)
    print("tag position in box frame [m]:")
    print(T_avg[:3, 3])

if REF_TAG_ID not in box_T_tag_map:
    print("\nReference tag was not registered. Failed.")
    exit()

# Save as object array dictionary
np.savez(
    "box_tag_map.npz",
    tag_ids=np.array(list(box_T_tag_map.keys()), dtype=int),
    box_T_tags=np.array([box_T_tag_map[k] for k in box_T_tag_map.keys()]),
    ref_tag_id=REF_TAG_ID,
    tag_size=TAG_SIZE,
    box_dims=np.array([BOX_W, BOX_D, BOX_H]),
)

print("\nSaved: box_tag_map.npz")
