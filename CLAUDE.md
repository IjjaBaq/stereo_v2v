# Stereo V2V Project Rules
- Environment: Python 3.11+, venv in `stereo_v2v_env`
- Coding Style: Functional, type-hinted, Google-style docstrings.
- Architecture: Focus on modular scripts; avoid monolithic classes.
- Libraries: OpenCV for classical stereo/processing.
- Data Path: All data relative to `./data` (KITTI structure).
- Calibration: Always check for `calib` files before processing pairs.
