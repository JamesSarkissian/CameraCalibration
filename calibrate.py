import numpy as np
import cv2 as cv
import os
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Our board has 8 x 8 squares, so it has 7 x 7 internal corners.
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

images = sorted(glob.glob('DistortedImages/*.png'))
if not images:
    raise RuntimeError('No images found in DistortedImages.')
os.makedirs('CorrectedImages', exist_ok=True)

# These photos contain two image sizes. Calibrate each set separately.
groups = {}
for fname in images:
    img = cv.imread(fname)
    size = (img.shape[1], img.shape[0])
    groups.setdefault(size, []).append(fname)

for size, filenames in groups.items():
    objpoints = []
    imgpoints = []

    # 1. Find corners in every image in this set.
    for fname in filenames:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Smaller images make corner detection faster.
        scale = min(1.0, 1200 / max(size))
        small = cv.resize(gray, None, fx=scale, fy=scale)
        ret, corners = cv.findChessboardCorners(small, (7, 7), None)

        if ret:
            # Convert the detected positions back to the original image size.
            corners = corners.reshape(-1, 1, 2)
            corners[:, :, 0] *= size[0] / small.shape[1]
            corners[:, :, 1] *= size[1] / small.shape[0]
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
        print(fname, 'corners found:', ret)

    if not objpoints:
        raise RuntimeError(f'No chessboard corners detected for size {size}.')

    # 2. Calibrate once using all detected boards in this set.
    # Keep k3 at zero to prevent excessive distortion near the image edges.
    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, size, None, None, flags=cv.CALIB_FIX_K3
    )
    print('Image size:', size, 'Reprojection error:', rms)

    # 3. Undistort and save each image.
    for fname in filenames:
        img = cv.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        x, y, crop_w, crop_h = roi
        if crop_w > 0 and crop_h > 0:
            dst = dst[y:y + crop_h, x:x + crop_w]

        output = 'CorrectedImages/' + os.path.basename(fname)
        if not cv.imwrite(output, dst):
            raise RuntimeError('Could not save: ' + output)
        print('Saved:', output)
