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
parser.add_argument("--no-gui", action="store_true", help="Run without OpenCV window (headless auto-capture)")
parser.add_argument("--max-images", type=int, default=30, help="Target number of images to save")
parser.add_argument("--save-every-n-frames", type=int, default=15, help="Auto-save interval in frames for --no-gui")
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
latest_img = None

if args.serial:
    print(f"Using RealSense serial: {args.serial}")
print(f"Stream: {args.width}x{args.height}@{args.fps}")
print(f"Save dir: {SAVE_DIR}")
if args.no_gui:
    print(f"Headless mode: ON (max_images={args.max_images}, save_every_n_frames={args.save_every_n_frames})")
else:
    print("SPACE / Left Click : save image")
    print("ESC   : quit")


def save_current_frame(img):
    global count
    filename = os.path.join(
        SAVE_DIR,
        f"img_{count:03d}.png"
    )
    cv2.imwrite(filename, img)
    print(f"Saved {filename}")
    count += 1


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and latest_img is not None:
        save_current_frame(latest_img)


if not args.no_gui:
    cv2.namedWindow("Checkerboard Capture")
    cv2.setMouseCallback("Checkerboard Capture", on_mouse)

try:
    frame_idx = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        latest_img = img.copy()
        frame_idx += 1

        if args.no_gui:
            if frame_idx % max(1, args.save_every_n_frames) == 0:
                save_current_frame(img)
            if count >= args.max_images:
                print(f"Reached max_images={args.max_images}. Exiting.")
                break
            continue

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
            save_current_frame(img)

finally:
    pipeline.stop()
    if not args.no_gui:
        cv2.destroyAllWindows()
