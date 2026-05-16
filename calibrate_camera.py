import argparse
import cv2
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--images-glob", type=str, default="checker_images/*.png",
                    help="Glob for checkerboard images")
parser.add_argument("--output", type=str, default="camera_calibration.npz",
                    help="Output calibration npz path")
parser.add_argument("--serial", type=str, default="",
                    help="Optional camera serial to save in output")
parser.add_argument("--checkerboard-cols", type=int, default=7,
                    help="Checkerboard inner corners (cols)")
parser.add_argument("--checkerboard-rows", type=int, default=10,
                    help="Checkerboard inner corners (rows)")
parser.add_argument("--square-size", type=float, default=0.025,
                    help="Square size in meters")
args = parser.parse_args()

# =========================
# Checkerboard settings
# =========================

CHECKERBOARD = (args.checkerboard_cols, args.checkerboard_rows)   # 내부 코너 개수
SQUARE_SIZE = args.square_size      # meter

# =========================
# Prepare object points
# =========================

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# =========================
# Arrays to store points
# =========================

objpoints = []
imgpoints = []

images = glob.glob(args.images_glob)

print(f"Found {len(images)} images")

# =========================
# Detect corners
# =========================

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners2,
            ret
        )

        cv2.imshow("Corners", img)
        cv2.waitKey(100)

    else:
        print(f"Failed: {fname}")

cv2.destroyAllWindows()

# =========================
# Calibration
# =========================

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n=== Calibration Result ===\n")

print("Camera Matrix:\n")
print(camera_matrix)

print("\nDistortion Coefficients:\n")
print(dist_coeffs)

print("\nReprojection Error:\n")
print(ret)

# =========================
# Save result
# =========================

np.savez(
    args.output,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    serial=args.serial
)

print(f"\nSaved to {args.output}")
