import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# =========================
# Load camera calibration
# =========================

calib = np.load("camera2_115222071236_calibration.npz")

K = calib["camera_matrix"]
D = calib["dist_coeffs"]

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

print("Camera params:")
print("fx, fy, cx, cy =", fx, fy, cx, cy)

# =========================
# AprilTag setting
# =========================

TAG_SIZE = 0.145  # meter 단위. 예: 8cm면 0.08로 수정

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
# RealSense RGB stream
# =========================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("Press ESC to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=TAG_SIZE
        )

        for det in detections:
            corners = det.corners.astype(int)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(img, p1, p2, (0, 255, 0), 2)

            center = tuple(det.center.astype(int))
            cv2.circle(img, center, 5, (0, 0, 255), -1)

            cv2.putText(
                img,
                f"ID: {det.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print("-------------")
            print("Tag ID:", det.tag_id)
            print("Position t [m]:")
            print(det.pose_t.reshape(3))
            print("Rotation R:")
            print(det.pose_R)

        cv2.imshow("AprilTag Detection", img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
