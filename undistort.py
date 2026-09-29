"""Correct profile photos using learned calibration, not simulation settings."""
from pathlib import Path
import json

import cv2 as cv
import numpy as np
from photo_paths import find_photo


def undistort_calibration_photos(profile_name):
    """Correct every calibration PNG; return the indexed output paths."""
    if not profile_name or Path(profile_name).name != profile_name or profile_name in ('.', '..'):
        raise ValueError('Provide only a profile name.')
    profile = Path(__file__).resolve().parent / 'Profiles' / profile_name
    metadata = json.loads((profile / 'simulation.json').read_text())
    if 'calibration' not in metadata:
        raise RuntimeError('Calibrate this profile first.')
    calibrations = metadata['calibration']['by_image_size']
    images = sorted((profile / 'calibration_photos').glob('*.png'))
    if not images:
        raise RuntimeError('No calibration photos found.')
    output_dir = profile / 'corrected_photos'
    output_dir.mkdir(exist_ok=True)
    indices = [
        int(path.stem.removeprefix('calibration_'))
        for path in output_dir.glob('calibration_*.png')
        if path.stem.removeprefix('calibration_').isdigit()
    ]
    next_index = max(indices, default=0) + 1
    outputs = []
    for path in images:
        img = cv.imread(str(path))
        if img is None:
            raise RuntimeError(f'Could not read: {path}')
        h, w = img.shape[:2]
        key = f'{w}x{h}'
        if key not in calibrations:
            raise RuntimeError(f'No saved calibration for {key}.')
        saved = calibrations[key]
        mtx = np.array(saved['camera_matrix'], dtype=np.float64)
        dist = np.array(saved['distortion_coefficients'], dtype=np.float64)
        newcameramtx, _ = cv.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # Preserve the full canvas and original dimensions, including black borders.
        output = output_dir / f'calibration_{next_index:03d}.png'
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite: {output}')
        if not cv.imwrite(str(output), dst):
            raise RuntimeError(f'Could not save: {output}')
        print(f'{path.name} -> {output.name}')
        outputs.append(output)
        next_index += 1
    return outputs


def undistort_photo(photo_name, profile_name):
    """Correct one photo from either input folder; return its indexed output path."""
    if not profile_name or Path(profile_name).name != profile_name or profile_name in ('.', '..'):
        raise ValueError('Provide only a profile name.')
    if not photo_name or Path(photo_name).name != photo_name or photo_name in ('.', '..'):
        raise ValueError("Provide only the photo's filename.")
    profile = Path(__file__).resolve().parent / 'Profiles' / profile_name
    metadata = json.loads((profile / 'simulation.json').read_text())
    if 'calibration' not in metadata:
        raise RuntimeError('Calibrate this profile first.')
    path = find_photo(profile, photo_name)
    img = cv.imread(str(path))
    if img is None:
        raise RuntimeError(f'Could not read: {path}')
    h, w = img.shape[:2]
    key = f'{w}x{h}'
    calibrations = metadata['calibration']['by_image_size']
    if key not in calibrations:
        raise RuntimeError(f'No saved calibration for {key}.')
    saved = calibrations[key]
    mtx = np.array(saved['camera_matrix'], dtype=np.float64)
    dist = np.array(saved['distortion_coefficients'], dtype=np.float64)
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # Preserve the full canvas and original dimensions, including black borders.
    output_dir = profile / 'corrected_photos'
    output_dir.mkdir(exist_ok=True)
    prefix = 'test' if path.parent.name == 'test_photos' else 'calibration'
    indices = [
        int(item.stem.removeprefix(prefix + '_'))
        for item in output_dir.glob(f'{prefix}_*.png')
        if item.stem.removeprefix(prefix + '_').isdigit()
    ]
    next_index = max(indices, default=0) + 1
    output = output_dir / f'{prefix}_{next_index:03d}.png'
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite: {output}')
    if not cv.imwrite(str(output), dst):
        raise RuntimeError(f'Could not save: {output}')
    print(f'{photo_name} -> {output.name}')
    return output
