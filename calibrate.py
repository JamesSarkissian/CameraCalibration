"""Learn a profile's calibration and save it in simulation.json."""
from pathlib import Path
import json

import cv2 as cv
import numpy as np


def calibrate(profile_name):
    """Calibrate PNG chessboard photos, independently for each image size."""
    if not profile_name or Path(profile_name).name != profile_name or profile_name in ('.', '..'):
        raise ValueError('Provide only a profile name.')
    profile = Path(__file__).resolve().parent / 'Profiles' / profile_name
    metadata_path = profile / 'simulation.json'
    metadata = json.loads(metadata_path.read_text())
    images = sorted((profile / 'calibration_photos').glob('*.png'))
    if not images:
        raise RuntimeError('No calibration photos found.')

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Our board has 8 x 8 squares: 7 x 7 internal corners.
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    groups = {}
    for path in images:
        img = cv.imread(str(path))
        if img is None:
            raise RuntimeError(f'Could not read: {path}')
        size = (img.shape[1], img.shape[0])
        groups.setdefault(size, []).append(path)

    results = {}
    for size, filenames in groups.items():
        objpoints = []
        imgpoints = []
        used_photos = []
        for path in filenames:
            img = cv.imread(str(path))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            scale = min(1.0, 1200 / max(size))
            small = cv.resize(gray, None, fx=scale, fy=scale)
            ret, corners = cv.findChessboardCorners(small, (7, 7), None)
            if ret:
                corners = corners.reshape(-1, 1, 2)
                corners[:, :, 0] *= size[0] / small.shape[1]
                corners[:, :, 1] *= size[1] / small.shape[0]
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                used_photos.append(path.name)
            print(path.name, 'corners found:', ret)

        if not objpoints:
            raise RuntimeError(f'No chessboard corners detected for size {size}.')
        # Estimate k3 as well as the other coefficients; do not fix it to zero.
        rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, size, None, None, flags=0
        )
        print('Image size:', size, 'Reprojection error:', rms)
        results[f'{size[0]}x{size[1]}'] = {
            'image_size': list(size),
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'reprojection_error': float(rms),
            'used_photos': used_photos,
        }

    metadata['calibration'] = {
        'board_internal_corners': [7, 7],
        'flags': [],
        'by_image_size': results,
    }
    # Preserve generation metadata and replace calibration only after success.
    temporary_path = metadata_path.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary_path.replace(metadata_path)
    return metadata['calibration']
