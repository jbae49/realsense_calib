"""
AprilTag detection with 6DOF axes visualization.

Each tag shows RGB axes:
  Red   = tag X axis (tag's right side)
  Green = tag Y axis (tag's down)
  Blue  = tag Z axis (out of tag surface, toward camera)

Use this to visually confirm tag orientation before calibration.
"""
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

TAG_SIZE = 0.077

# =========================
# Detector
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
# Axis drawing
# =========================

def draw_tag_axes(img, det, axis_length=0.06):
    """Draw XYZ axes on detected tag.

    AprilTag coordinate frame:
      X (red)   = tag's right side
      Y (green) = tag's downward
      Z (blue)  = out of tag surface (toward camera)
    """
    rvec, _ = cv2.Rodrigues(det.pose_R)
    tvec = det.pose_t.reshape(3, 1)

    axis_points = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length],
    ])

    img_pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, D)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    origin = tuple(img_pts[0])
    cv2.line(img, origin, tuple(img_pts[1]), (0, 0, 255), 3)   # X = red
    cv2.line(img, origin, tuple(img_pts[2]), (0, 255, 0), 3)   # Y = green
    cv2.line(img, origin, tuple(img_pts[3]), (255, 0, 0), 3)   # Z = blue

    cv2.putText(img, "X", tuple(img_pts[1] + [5, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, "Y", tuple(img_pts[2] + [5, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, "Z", tuple(img_pts[3] + [5, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return origin

# =========================
# RealSense
# =========================

pipeline = rs.pipeline()
config = rs.config()
config.enable_device("115222071236")
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("AprilTag 6DOF Axis Viewer")
print("========================")
print("  Red (X)   = tag right")
print("  Green (Y) = tag down")
print("  Blue (Z)  = out of tag surface")
print()
print("Use this to check tag orientation.")
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
                cv2.line(img, p1, p2, (0, 255, 255), 2)

            center = tuple(det.center.astype(int))
            cv2.circle(img, center, 5, (0, 255, 255), -1)

            draw_tag_axes(img, det, axis_length=TAG_SIZE * 0.8)

            t = det.pose_t.reshape(3)
            cv2.putText(img, f"ID:{det.tag_id}",
                        (center[0] + 10, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
                        (center[0] + 10, center[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Legend
        cv2.putText(img, "X(red)=tag right", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(img, "Y(green)=tag down", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(img, "Z(blue)=surface normal", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        cv2.imshow("AprilTag 6DOF Axes", img)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
