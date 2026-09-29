import cv2 as cv
import numpy as np
import glob
import os

# Find the training images and create the output folder.
images = sorted(glob.glob('TrainingImages/*.png'))
os.makedirs('DistortedImages', exist_ok=True)

# Distortion settings: k1, k2, p1, p2, k3.
dist = np.array([-0.30, 0.10, 0.015, -0.015, 0.02], dtype=np.float32)

for index, fname in enumerate(images, start=1):
    img = cv.imread(fname)
    if img is None:
        print('Could not read:', fname)
        continue

    h, w = img.shape[:2]

    # Approximate camera matrix, based on the image size.
    mtx = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Simple distortion simulation.
    # Negating dist is an approximation, not an exact inverse.
    map1, map2 = cv.initUndistortRectifyMap(
        mtx, -dist, None, mtx, (w, h), cv.CV_32FC1
    )
    distorted = cv.remap(img, map1, map2, cv.INTER_LINEAR)

    # Save each image with its index: distorted_001.png, etc.
    output = f'DistortedImages/distorted_{index:03d}.png'
    if not cv.imwrite(output, distorted):
        raise RuntimeError('Could not save: ' + output)
    print(fname, '->', output)
