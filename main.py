"""Terminal menu for camera profile calibration and photo correction."""
from pathlib import Path
import json

from calibrate import calibrate
from addDistortion import distort_calibration_photos, distort_photo
from undistort import undistort_calibration_photos, undistort_photo


def print_values(values, indent=0):
    """Display saved fields with labels, units, and compact numeric tables."""
    padding = " " * indent
    labels = {
        "calibration": "Learned calibration (used to correct photos)",
        "simulation_coefficients": "Simulation distortion coefficients",
        "simulation_camera_matrix": "Simulation camera matrix",
        "by_image_size": "Calibration results by image size",
        "reprojection_error": "Reprojection error (RMS)",
        "board_internal_corners": "Chessboard internal corners",
        "photos": "Simulation source and output photos",
        "used_photos": "Photos used for calibration",
    }
    for key, value in values.items():
        label = labels.get(key, key.replace("_", " ").capitalize())
        if key in ("camera_matrix", "simulation_camera_matrix"):
            print(f"{padding}{label}:")
            for row in value:
                print(padding + "  " + "  ".join(f"{number:14.6f}" for number in row))
            print(f"{padding}  First two diagonal values: focal lengths in pixels.")
            print(f"{padding}  Last column, first two rows: optical center in pixels.")
        elif key in ("distortion_coefficients", "simulation_coefficients"):
            coefficients = value[0] if value and isinstance(value[0], list) else value
            print(f"{padding}{label}:")
            for index, number in enumerate(coefficients):
                name = ("k1", "k2", "p1", "p2", "k3")[index] if index < 5 else f"Coefficient {index + 1}"
                print(f"{padding}  {name:>2} = {number: .8f}")
            print(f"{padding}  k values: radial distortion; p values: tangential distortion.")
        elif key in ("image_size", "board_internal_corners"):
            unit = " pixels (width × height)" if key == "image_size" else ""
            print(f"{padding}{label}: {value[0]} × {value[1]}{unit}")
        elif key == "reprojection_error":
            print(f"{padding}{label}: {value:.6f} pixels")
        elif isinstance(value, dict):
            print(f"\n{padding}{label}:")
            print_values(value, indent + 2)
        elif isinstance(value, list):
            if not value:
                print(f"{padding}{label}: None")
                continue
            print(f"{padding}{label} ({len(value)}):")
            for index, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    print(f"{padding}  Photo {index}:")
                    print_values(item, indent + 4)
                else:
                    print(f"{padding}  {index}. {item}")
        else:
            print(f"{padding}{label}: {value}")


def main():
    profiles_dir = Path(__file__).resolve().parent / "Profiles"

    while True:
        profile_name = input("\nEnter profile name (or 'exit'): ").strip()
        if profile_name.lower() == "exit":
            return

        if (
            not profile_name
            or Path(profile_name).name != profile_name
            or profile_name in (".", "..")
        ):
            print("Please enter a profile name, not a path.")
            continue

        profile = profiles_dir / profile_name
        if not profile.is_dir():
            print(f"Profile '{profile_name}' does not exist.")
            continue

        while True:
            print(f"\nProfile: {profile_name}")
            print("1. Calibrate this profile")
            print("2. Correct all calibration photos")
            print("3. Correct one photo (calibration_photos or test_photos)")
            print("4. Switch profiles")
            print("5. View all saved profile values")
            print("6. Add distortion to calibration photos (overwrites photos)")
            print("7. Add distortion to one photo (overwrites photo)")
            print("8. Exit")
            choice = input("Choose an option: ").strip()

            try:
                if choice == "1":
                    calibrate(profile_name)
                    print("Calibration saved to simulation.json.")
                elif choice == "2":
                    outputs = undistort_calibration_photos(profile_name)
                    print(f"Saved {len(outputs)} images to corrected_photos.")
                elif choice == "3":
                    photo_name = input(
                        "Photo name in calibration_photos or test_photos (extension optional): "
                    ).strip()
                    output = undistort_photo(photo_name, profile_name)
                    print(f"Saved to corrected_photos/{output.name}")
                elif choice == "4":
                    break
                elif choice == "5":
                    metadata = json.loads((profile / "simulation.json").read_text())
                    print(f"\nSaved values for {profile_name}")
                    print("Numbers are rounded for display; saved values are unchanged.\n")
                    print_values(metadata)
                    if "calibration" not in metadata:
                        print("\nNo learned calibration saved. Choose option 1 to calibrate.")
                elif choice == "6":
                    outputs = distort_calibration_photos(profile_name)
                    print(f"Distorted {len(outputs)} calibration photos in place.")
                    print("Choose option 1 to recalibrate before correcting photos.")
                elif choice == "7":
                    photo_name = input("Photo name to distort (extension optional): ").strip()
                    output = distort_photo(photo_name, profile_name)
                    if output.parent.name == "test_photos":
                        print("Choose option 3 and the same name to correct it using saved calibration.")
                    else:
                        print("Calibration photos changed. Choose option 1 to recalibrate.")
                elif choice == "8":
                    return
                else:
                    print("Please choose a number from 1 to 8.")
            except Exception as error:
                print(f"Could not complete the action: {error}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
