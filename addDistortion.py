"""Simulate distortion on an image or on a profile's calibration photos."""
from pathlib import Path
import json
import tempfile

import cv2 as cv
import numpy as np
from photo_paths import find_photo


DISTORTION = np.array([-0.30, 0.10, 0.015, -0.015, 0.02], dtype=np.float32)


def distort_image(image):
    """Return a distorted OpenCV image without changing the input array."""
    if image is None or image.size == 0:
        raise ValueError('Provide a nonempty image loaded with cv.imread().')
    h, w = image.shape[:2]
    matrix = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1],
    ], dtype=np.float32)
    # Preserve the original approximation: negating coefficients is not an
    # exact inverse of the undistortion model.
    map1, map2 = cv.initUndistortRectifyMap(
        matrix, -DISTORTION, None, matrix, (w, h), cv.CV_32FC1
    )
    return cv.remap(image, map1, map2, cv.INTER_LINEAR)


def distort_photo(photo_name, profile_name):
    """Distort one named photo in place, searching both profile input folders."""
    if not profile_name or Path(profile_name).name != profile_name or profile_name in ('.', '..'):
        raise ValueError('Provide only a profile name.')
    profile = Path(__file__).resolve().parent / 'Profiles' / profile_name
    path = find_photo(profile, photo_name)
    image = cv.imread(str(path))
    if image is None:
        raise RuntimeError(f'Could not read: {path}')
    metadata_path = profile / 'simulation.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    h, w = image.shape[:2]
    # Do not silently use a different simulated lens for a test image.
    if 'image_size' in metadata and metadata['image_size'] != [w, h]:
        raise RuntimeError(f'Photo is {w}x{h}; simulation expects {metadata["image_size"]}.')
    if 'simulation_coefficients' in metadata and not np.allclose(
        np.asarray(metadata['simulation_coefficients']).reshape(-1), DISTORTION
    ):
        raise RuntimeError('Saved simulation settings differ from the current distortion settings.')
    distorted = distort_image(image)
    with tempfile.TemporaryDirectory(prefix='.distortion-', dir=profile) as temporary:
        staged = Path(temporary) / path.name
        if not cv.imwrite(str(staged), distorted):
            raise RuntimeError(f'Could not prepare distorted photo: {path}')
        # Editing training data invalidates calibration; editing a test photo
        # must preserve it so the live demo can immediately correct that photo.
        if path.parent.name == 'calibration_photos' and 'calibration' in metadata:
            metadata.pop('calibration')
            staged_metadata = Path(temporary) / 'simulation.json'
            staged_metadata.write_text(json.dumps(metadata, indent=2) + '\n')
            staged_metadata.replace(metadata_path)
        staged.replace(path)
    print(f'Distorted: {path.relative_to(profile)}')
    return path


def distort_calibration_photos(profile_name):
    """Replace calibration PNGs with distorted versions at the same paths.

    Each call adds distortion again. Recalibrate after using this function;
    any previously learned calibration is removed from simulation.json.
    """
    if not profile_name or Path(profile_name).name != profile_name or profile_name in ('.', '..'):
        raise ValueError('Provide only a profile name.')
    root = Path(__file__).resolve().parent
    profile = root / 'Profiles' / profile_name
    images = sorted((profile / 'calibration_photos').glob('*.png'))
    if not images:
        raise RuntimeError(f'No calibration PNG photos found in {profile}.')
    metadata_path = profile / 'simulation.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    # Prepare every output before replacing any input, so unreadable images,
    # mismatched dimensions, or encoding failures leave the photos unchanged.
    with tempfile.TemporaryDirectory(prefix='.distortion-', dir=profile) as temporary:
        staging = Path(temporary)
        expected_size = None
        records = []
        for path in images:
            image = cv.imread(str(path))
            if image is None:
                raise RuntimeError(f'Could not read: {path}')
            h, w = image.shape[:2]
            if expected_size is not None and expected_size != (w, h):
                raise RuntimeError('All simulation inputs must have the same dimensions.')
            expected_size = (w, h)
            distorted = distort_image(image)
            if not cv.imwrite(str(staging / path.name), distorted):
                raise RuntimeError(f'Could not prepare distorted photo: {path}')
            records.append({'source': str(path.relative_to(root)),
                            'output': str(path.relative_to(root))})

        metadata.pop('calibration', None)
        metadata.update({
            'method': 'approximate distortion using initUndistortRectifyMap with negated coefficients',
            'simulation_coefficients': DISTORTION.tolist(),
            'simulation_camera_matrix': [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]],
            'image_size': list(expected_size),
            'note': 'Settings for the latest in-place distortion pass, not a learned calibration. Repeated calls compound distortion.',
            'photos': records,
        })
        staged_metadata = staging / 'simulation.json'
        staged_metadata.write_text(json.dumps(metadata, indent=2) + '\n')
        # Invalidate old calibration before replacing photos, even if a later
        # filesystem error interrupts the replacement of the batch.
        staged_metadata.replace(metadata_path)
        for path in images:
            (staging / path.name).replace(path)
            print(f'Distorted: {path.relative_to(root)}', flush=True)
    return images
