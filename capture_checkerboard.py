import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import os

parser = argparse.ArgumentParser()
parser.add_argument("--serial", type=str, default="", help="RealSense serial; empty uses default device")
parser.add_argument("--save-dir", type=str, default="checker_images", help="Directory to save checkerboard images")
parser.add_argument("--width", type=int, default=640, help="Color stream width")
parser.add_argument("--height", type=int, default=480, help="Color stream height")
parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
args = parser.parse_args()

SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

if args.serial:
    config.enable_device(args.serial)
config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

pipeline.start(config)

count = 0

if args.serial:
    print(f"Using RealSense serial: {args.serial}")
print(f"Stream: {args.width}x{args.height}@{args.fps}")
print(f"Save dir: {SAVE_DIR}")
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

        cv2.putText(
            display,
            f"Saved: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Checkerboard Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == 32:
            filename = os.path.join(
                SAVE_DIR,
                f"img_{count:03d}.png"
            )

            cv2.imwrite(filename, img)

            print(f"Saved {filename}")

            count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
