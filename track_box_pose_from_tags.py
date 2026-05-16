import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# =========================================================
# Settings
# =========================================================

CAM1_SERIAL = "935322072654"
CAM2_SERIAL = "115222071236"  # Camera2 = WORLD

CAM1_CALIB = "camera1_935322072654_calibration.npz"
CAM2_CALIB = "camera2_115222071236_calibration.npz"

EXTRINSIC_FILE = "camera1_to_camera2_extrinsic.npz"
BOX_TAG_MAP_FILE = "box_tag_map.npz"

TAG_SIZE = 0.077

BOX_TAG_IDS = [0, 1, 2, 3, 4, 5]

# =========================================================
# Utils
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

def invert_T(T):
    return np.linalg.inv(T)

def average_rotation(R_list):

    R_stack = np.stack(R_list, axis=0)

    R_avg = np.mean(R_stack, axis=0)

    U, _, Vt = np.linalg.svd(R_avg)

    R_avg = U @ Vt

    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    return R_avg

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

# =========================================================
# Load calibration
# =========================================================

cam1_params = load_camera_params(CAM1_CALIB)
cam2_params = load_camera_params(CAM2_CALIB)

T_c2_c1 = np.load(EXTRINSIC_FILE)["T_c2_c1"]

# =========================================================
# Load box tag map
# =========================================================

map_data = np.load(BOX_TAG_MAP_FILE)

tag_ids = map_data["tag_ids"]
box_T_tags = map_data["box_T_tags"]

BOX_T_TAG = {}

for tag_id, T in zip(tag_ids, box_T_tags):
    BOX_T_TAG[int(tag_id)] = T

print("Loaded box tag map:")
print(sorted(BOX_T_TAG.keys()))

# =========================================================
# Detector
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
# Camera pipelines
# =========================================================

pipeline1 = rs.pipeline()

config1 = rs.config()

config1.enable_device(CAM1_SERIAL)

config1.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

pipeline1.start(config1)

pipeline2 = rs.pipeline()

config2 = rs.config()

config2.enable_device(CAM2_SERIAL)

config2.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

pipeline2.start(config2)

print("\n======================================")
print("Tracking box pose")
print("Camera2 frame = WORLD")
print("Press ESC to quit")
print("======================================")

# =========================================================
# Main loop
# =========================================================

try:

    while True:

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
        # Detect
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

        # -------------------------------------------------
        # Build world_T_tag dict
        # -------------------------------------------------

        world_T_tags = {}

        # ---------- CAM1 ----------

        for det in dets1:

            if det.tag_id not in BOX_TAG_IDS:
                continue

            T_c1_tag = pose_to_T(
                det.pose_R,
                det.pose_t
            )

            T_world_tag = (
                T_c2_c1 @
                T_c1_tag
            )

            world_T_tags[det.tag_id] = T_world_tag

            draw_detection(
                img1,
                det,
                (0, 255, 0)
            )

        # ---------- CAM2 ----------

        for det in dets2:

            if det.tag_id not in BOX_TAG_IDS:
                continue

            T_world_tag = pose_to_T(
                det.pose_R,
                det.pose_t
            )

            # Prefer CAM2 if duplicated
            world_T_tags[det.tag_id] = T_world_tag

            draw_detection(
                img2,
                det,
                (0, 255, 255)
            )

        # -------------------------------------------------
        # Compute box pose candidates
        # -------------------------------------------------

        box_pose_candidates = []

        for tag_id, world_T_tag in world_T_tags.items():

            if tag_id not in BOX_T_TAG:
                continue

            box_T_tag = BOX_T_TAG[tag_id]

            world_T_box = (
                world_T_tag @
                invert_T(box_T_tag)
            )

            box_pose_candidates.append(
                (tag_id, world_T_box)
            )

        # -------------------------------------------------
        # Fuse candidates
        # -------------------------------------------------

        if len(box_pose_candidates) > 0:

            positions = []
            rotations = []

            for tag_id, T in box_pose_candidates:

                p = T[:3, 3]
                R = T[:3, :3]

                positions.append(p)
                rotations.append(R)

                print(
                    f"[Tag {tag_id}] "
                    f"candidate position: "
                    f"{p}"
                )

            # Position average
            p_avg = np.mean(
                np.stack(positions, axis=0),
                axis=0
            )

            # Rotation average
            R_avg = average_rotation(rotations)

            world_T_box_final = np.eye(4)

            world_T_box_final[:3, :3] = R_avg
            world_T_box_final[:3, 3] = p_avg

            print("\n================================")
            print("FINAL BOX POSE")
            print("================================")

            print("Position [m]:")
            print(p_avg)

            print("\nRotation:")
            print(R_avg)

            print(
                "\nVisible tags:",
                sorted(world_T_tags.keys())
            )

            print("================================\n")

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
