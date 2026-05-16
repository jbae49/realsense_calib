import cv2
import numpy as np
import glob

CHECKERBOARD = (7, 10)   # 내부 코너 개수
SQUARE_SIZE = 0.025      # 2.5 cm = 0.025 m

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob("checker_images_cam2/*.png")
print(f"Found {len(images)} images")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

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
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Cam2 Corners", img)
        cv2.waitKey(100)

    else:
        print(f"Failed: {fname}")

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n=== Camera 2 Calibration Result ===\n")
print("Camera Matrix:\n")
print(camera_matrix)
print("\nDistortion Coefficients:\n")
print(dist_coeffs)
print("\nReprojection Error:\n")
print(ret)

np.savez(
    "camera2_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    serial="115222071236"
)

print("\nSaved to camera2_calibration.npz")
