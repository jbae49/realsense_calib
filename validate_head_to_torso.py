"""
Validate the head tag → torso_link transform.

Method: Stick a SECOND AprilTag on the robot's torso (chest area),
then compare:
  1. Torso position estimated from HEAD tag + URDF offset
  2. Torso position directly from TORSO tag

If the two match closely, our URDF-based transform is correct.

Setup:
  - Tag on head (top): ROBOT_HEAD_TAG_ID
  - Tag on torso (chest): ROBOT_TORSO_TAG_ID (temporary, for validation only)
  - Single camera (cam2 = world)
"""
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# =========================================================
# Config — EDIT THESE
# =========================================================

ROBOT_HEAD_TAG_ID = 10     # head tag ID (change to yours)
ROBOT_TORSO_TAG_ID = 11    # temporary torso tag ID (change to yours)
TAG_SIZE = 0.077            # meters

CAM_SERIAL = "115222071236"
CAM_CALIB = "camera2_115222071236_calibration.npz"

# =========================================================
# URDF-based transform: tag(head) → torso_link
# =========================================================

T_torso_headlink = np.eye(4)
T_torso_headlink[:3, 3] = [0.0039635, 0.0, -0.054]

T_headlink_tagtop = np.eye(4)
T_headlink_tagtop[:3, 3] = [0.001, 0.0, 0.526]

T_torso_tag_head = T_torso_headlink @ T_headlink_tagtop  # R_mounting = I for now
T_tag_head_torso = np.linalg.inv(T_torso_tag_head)

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


def draw_detection(img, det, color, label):
    corners = det.corners.astype(int)
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i+1)%4]), color, 2)
    center = tuple(det.center.astype(int))
    cv2.circle(img, center, 5, color, -1)
    cv2.putText(img, label, (center[0]+10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_axes(img, K, T, length=0.1):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    pts_3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, T[:3,3], K, np.zeros(5))
    pts = pts_2d.reshape(-1, 2).astype(int)
    o = tuple(pts[0])
    cv2.line(img, o, tuple(pts[1]), (0,0,255), 2)
    cv2.line(img, o, tuple(pts[2]), (0,255,0), 2)
    cv2.line(img, o, tuple(pts[3]), (255,0,0), 2)
    return o

# =========================================================
# Camera + detector
# =========================================================

K, cam_params = load_camera_params(CAM_CALIB)

detector = Detector(
    families="tag36h11", nthreads=4, quad_decimate=1.0,
    quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(CAM_SERIAL)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("======================================")
print("VALIDATION: Head tag vs Torso tag")
print(f"  Head tag ID:  {ROBOT_HEAD_TAG_ID}")
print(f"  Torso tag ID: {ROBOT_TORSO_TAG_ID}")
print()
print("Place both tags on robot, keep robot still.")
print("Compare estimated vs actual torso position.")
print("Press 'c' to capture & print comparison")
print("Press 'q' to quit")
print("======================================")

errors = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=cam_params, tag_size=TAG_SIZE)

        T_cam_headtag = None
        T_cam_torsotag = None

        for det in dets:
            T = pose_to_T(det.pose_R, det.pose_t)

            if det.tag_id == ROBOT_HEAD_TAG_ID:
                T_cam_headtag = T
                draw_detection(img, det, (255, 0, 255), "HEAD")
                draw_axes(img, K, T, 0.05)

            elif det.tag_id == ROBOT_TORSO_TAG_ID:
                T_cam_torsotag = T
                draw_detection(img, det, (0, 255, 0), "TORSO(real)")
                draw_axes(img, K, T, 0.05)

        # Compute estimated torso from head tag
        if T_cam_headtag is not None:
            T_cam_torso_est = T_cam_headtag @ T_tag_head_torso
            est_pos = T_cam_torso_est[:3, 3]

            origin = draw_axes(img, K, T_cam_torso_est, 0.12)
            if origin:
                cv2.putText(img, "TORSO(est)", (origin[0]+10, origin[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.putText(img, f"Est:  [{est_pos[0]:.3f}, {est_pos[1]:.3f}, {est_pos[2]:.3f}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        if T_cam_torsotag is not None:
            real_pos = T_cam_torsotag[:3, 3]
            cv2.putText(img, f"Real: [{real_pos[0]:.3f}, {real_pos[1]:.3f}, {real_pos[2]:.3f}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Compare if both visible
        if T_cam_headtag is not None and T_cam_torsotag is not None:
            est_pos = (T_cam_headtag @ T_tag_head_torso)[:3, 3]
            real_pos = T_cam_torsotag[:3, 3]
            err = np.linalg.norm(est_pos - real_pos)
            diff = est_pos - real_pos

            cv2.putText(img, f"Diff: [{diff[0]:+.3f}, {diff[1]:+.3f}, {diff[2]:+.3f}]  err={err:.3f}m",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            color = (0, 255, 0) if err < 0.05 else (0, 165, 255) if err < 0.10 else (0, 0, 255)
            cv2.putText(img, f"{'OK' if err < 0.05 else 'ADJUST NEEDED' if err < 0.10 else 'BAD'}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Validation", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and T_cam_headtag is not None and T_cam_torsotag is not None:
            est_pos = (T_cam_headtag @ T_tag_head_torso)[:3, 3]
            real_pos = T_cam_torsotag[:3, 3]
            err = np.linalg.norm(est_pos - real_pos)
            errors.append(err)
            print(f"\n--- Capture #{len(errors)} ---")
            print(f"  Estimated (from head):  {est_pos}")
            print(f"  Actual (torso tag):     {real_pos}")
            print(f"  Difference:             {est_pos - real_pos}")
            print(f"  Error:                  {err:.4f} m ({err*100:.1f} cm)")

        elif key in [ord('q'), 27]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

if errors:
    print(f"\n=== Summary ({len(errors)} captures) ===")
    print(f"  Mean error: {np.mean(errors)*100:.1f} cm")
    print(f"  Max error:  {np.max(errors)*100:.1f} cm")
    print(f"  Min error:  {np.min(errors)*100:.1f} cm")
    if np.mean(errors) < 0.05:
        print("  → URDF offset is good enough!")
    else:
        print("  → Consider adjusting the offset or doing 2-tag calibration")
        print("  → To auto-calibrate, press 's' next time to save measured offset")
