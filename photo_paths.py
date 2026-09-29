"""Find a named photo in either of a profile's input folders."""
from pathlib import Path


def find_photo(profile, photo_name):
    if not photo_name or Path(photo_name).name != photo_name or photo_name in ('.', '..'):
        raise ValueError("Provide only a photo name, not a path.")
    name = photo_name.casefold()
    candidates = [
        path
        for folder in ('calibration_photos', 'test_photos')
        for path in (profile / folder).glob('*')
        if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    ]
    matches = [path for path in candidates if path.name.casefold() == name]
    if not matches and not Path(photo_name).suffix:
        matches = [path for path in candidates if path.stem.casefold() == name]
    if not matches:
        raise FileNotFoundError(f"'{photo_name}' was not found in calibration_photos or test_photos.")
    if len(matches) > 1:
        raise ValueError('Ambiguous photo name; use the full filename or give the photos unique names: '
                         + ', '.join(str(path.relative_to(profile)) for path in matches))
    return matches[0]
