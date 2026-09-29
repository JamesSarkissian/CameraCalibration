# Simulated camera demo

`simulated_camera` is the only active camera profile:

- `calibration_photos/`: six iPhone chessboard photos with artificial distortion added.
- `test_photos/`: reserved for separate photos with the same simulated distortion.
- `corrected_photos/`: reserved for correction results.
- `simulation.json`: generation settings; this is not a learned calibration.
- `calibration.npz`: will be created when calibration is implemented for this profile.

Original iPhone photos are preserved in `source_photos/iPhone/` at the repository
root. MacBook originals are preserved in `archive/Macbook/` and are not part of
any active profile. Original checksums are in `source_photos/photo_manifest.json`.

`python3 addDistortion.py` generates the six calibration inputs and refuses to
overwrite existing output photos. It retains the original approximate simulation
method (negated distortion coefficients). The iPhone inputs are not guaranteed to
be distortion-free, and simulation settings are not exact calibration ground truth.

The board has 7 × 7 internal corners. Six images support a preliminary demo;
aim for at least ten varied usable calibration images and separate test images.
The existing `calibrate.py` still uses its old paths and needs updating before
calibrating this profile. No calibration or correction has been run yet.
