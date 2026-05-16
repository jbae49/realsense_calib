import cv2
import numpy as np
import pyrealsense2 as rs
import os

SERIAL = "115222071236"   # Camera 2: D435
SAVE_DIR = "checker_images_cam2"

os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_device(SERIAL)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

count = 0

print(f"Using RealSense serial: {SERIAL}")
print("SPACE : save image")
print("ESC   : quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        display = img.copy()

        cv2.putText(display, f"Cam2 Saved: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera2 Checkerboard Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            filename = os.path.join(SAVE_DIR, f"cam2_img_{count:03d}.png")
            cv2.imwrite(filename, img)
            print(f"Saved {filename}")
            count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
