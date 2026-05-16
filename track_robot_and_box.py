"""
Track robot torso_link and box poses in real-time.

Uses:
  - AprilTag on robot head → torso_link pose (via T_tag_torso calibration)
  - AprilTags on box → box pose (via box_tag_map)
  - Both expressed in world frame (camera2)

Outputs the relative pose (torso vs box) as observation for RL policy.
"""
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# =========================================================
# Config
# =========================================================

CAM_SERIAL = "115222071236"  # Camera2 = WORLD (single cam)
CAM_CALIB = "camera2_115222071236_calibration.npz"

BOX_TAG_MAP_FILE = "box_tag_map.npz"
HEAD_TAG_CALIB_FILE = "T_tag_torso.npz"

TAG_SIZE = 0.077

BOX_TAG_IDS = [0, 1, 2, 3, 4, 5]
ROBOT_HEAD_TAG_ID = 10   # CHANGE to your actual head tag ID

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


def average_rotation(R_list):
    R_avg = np.mean(np.stack(R_list, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def rotation_to_euler(R):
    """Extract roll, pitch, yaw from rotation matrix (XYZ convention)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def draw_detection(img, det, color, label=""):
    corners = det.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, color, 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    text = label if label else f"ID:{det.tag_id}"
    cv2.putText(img, text, (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_axes(img, K, T_cam_frame, length=0.1):
    dist_coeffs = np.zeros(5)
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3]
    points_3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    pts = points_2d.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 2)
    return origin

# =========================================================
# Load calibration data
# =========================================================

K, cam_params = load_camera_params(CAM_CALIB)

# Box tag map
map_data = np.load(BOX_TAG_MAP_FILE)
BOX_T_TAG = {}
for tag_id, T in zip(map_data["tag_ids"], map_data["box_T_tags"]):
    BOX_T_TAG[int(tag_id)] = T

# Head tag → torso calibration
head_calib = np.load(HEAD_TAG_CALIB_FILE)
T_tag_torso = head_calib["T_tag_torso"]
print(f"Loaded T_tag_torso (translation: {T_tag_torso[:3, 3]})")

# =========================================================
# Detector + cameras
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

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(CAM_SERIAL)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("\n======================================")
print("Tracking: Robot torso + Box")
print("Single camera = WORLD frame")
print("Press ESC to quit")
print("======================================\n")

# =========================================================
# Main loop
# =========================================================

try:
    while True:
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=cam_params, tag_size=TAG_SIZE)

        # ------ Collect detections ------

        world_T_boxtags = {}
        world_T_headtag = None

        for det in dets:
            T_world_tag = pose_to_T(det.pose_R, det.pose_t)

            if det.tag_id in BOX_TAG_IDS:
                world_T_boxtags[det.tag_id] = T_world_tag
                draw_detection(img, det, (0, 255, 255))
            elif det.tag_id == ROBOT_HEAD_TAG_ID:
                world_T_headtag = T_world_tag
                draw_detection(img, det, (255, 0, 255), "HEAD")

        # ------ Compute torso pose ------

        world_T_torso = None
        if world_T_headtag is not None:
            world_T_torso = world_T_headtag @ T_tag_torso
            torso_pos = world_T_torso[:3, 3]

            draw_axes(img, K, world_T_torso, length=0.15)

        # ------ Compute box pose ------

        world_T_box = None
        if len(world_T_boxtags) > 0:
            positions = []
            rotations = []
            for tag_id, T_world_tag in world_T_boxtags.items():
                if tag_id not in BOX_T_TAG:
                    continue
                world_T_box_candidate = T_world_tag @ np.linalg.inv(BOX_T_TAG[tag_id])
                positions.append(world_T_box_candidate[:3, 3])
                rotations.append(world_T_box_candidate[:3, :3])

            if len(positions) > 0:
                world_T_box = np.eye(4)
                world_T_box[:3, 3] = np.mean(np.stack(positions), axis=0)
                world_T_box[:3, :3] = average_rotation(rotations)

        # ------ Compute relative pose (observation) ------

        if world_T_torso is not None and world_T_box is not None:
            torso_pos = world_T_torso[:3, 3]
            box_pos = world_T_box[:3, 3]

            relative_pos = torso_pos - box_pos
            relative_T = np.linalg.inv(world_T_box) @ world_T_torso
            relative_rpy = rotation_to_euler(relative_T[:3, :3])

            print(f"[OBS] torso-box relative pos: "
                  f"[{relative_pos[0]:+.3f}, {relative_pos[1]:+.3f}, {relative_pos[2]:+.3f}]  "
                  f"rpy: [{relative_rpy[0]:+.1f}, {relative_rpy[1]:+.1f}, {relative_rpy[2]:+.1f}]")

            cv2.putText(img,
                        f"Torso: [{torso_pos[0]:.3f}, {torso_pos[1]:.3f}, {torso_pos[2]:.3f}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(img,
                        f"Box:   [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img,
                        f"Rel:   [{relative_pos[0]:+.3f}, {relative_pos[1]:+.3f}, {relative_pos[2]:+.3f}]",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif world_T_torso is not None:
            torso_pos = world_T_torso[:3, 3]
            cv2.putText(img,
                        f"Torso: [{torso_pos[0]:.3f}, {torso_pos[1]:.3f}, {torso_pos[2]:.3f}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(img, "Box: NOT VISIBLE", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(img, "Head tag NOT VISIBLE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("WORLD", img)

        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
