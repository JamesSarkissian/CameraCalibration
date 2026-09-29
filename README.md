This Python program uses OpenCV to estimate camera lens distortion from photos of an 8 × 8 chessboard. 
It finds the board’s inner corners, saves the calibration settings, and uses them to correct other photos.

Run python3 main.py, enter a profile name from the Profiles folder, and follow the menu to calibrate it or correct photos. 
Calibration images go in calibration_photos, and separate images go in test_photos. 
Corrected images are saved in corrected_photos. 
Photos must match a calibrated image size.


You can also view saved settings or add artificial distortion to test the correction process. 
Adding distortion overwrites the selected input photos, so keep copies of your originals.
