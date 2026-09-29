import cv2
import numpy as np

# Load image
img = cv2.imread("Images/ChessBoard1.jpg")

if img is None:
    print("Image not found")
    exit()

h, w = img.shape[:2]

# Approximate camera matrix
K = np.array([
    [w, 0, w / 2],
    [0, w, h / 2],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients:
# [k1, k2, p1, p2, k3]
dist = np.array([
    -0.30,   # k1 - radial distortion
     0.10,   # k2 - radial distortion
     0.015,  # p1 - tangential distortion
    -0.015,  # p2 - tangential distortion
     0.02    # k3 - radial distortion
], dtype=np.float32)

# Negating the coefficients here lets us simulate distortion
map1, map2 = cv2.initUndistortRectifyMap(
    K,
    -dist,
    None,
    K,
    (w, h),
    cv2.CV_32FC1
)

# Apply distortion
distorted = cv2.remap(
    img,
    map1,
    map2,
    cv2.INTER_LINEAR
)

# Save result
cv2.imwrite("Images/distorted_photo.jpg", distorted)

# Show before and after
cv2.imshow("Original", img)
cv2.imshow("Distorted", distorted)

cv2.waitKey(0)
cv2.destroyAllWindows()