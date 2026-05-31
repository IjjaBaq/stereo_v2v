import logging
logger = logging.getLogger(__name__)

def extract_stereo_params(calib: dict) -> tuple[float, float]:
    """Extract focal length and baseline from calibration dict.

    Supports both detection split format (P2/P3 matrices) and tracking
    split format (focal_length_px + baseline_m pre-extracted by
    load_tracking_calib).

    Args:
        calib: Calibration dict from load_calib or load_tracking_calib.

    Returns:
        Tuple of (focal_length_px, baseline_m) as floats.

    Raises:
        ValueError: If computed baseline is non-positive (detection format only).
    """
    # Tracking format — already extracted by load_tracking_calib
    if "focal_length_px" in calib and "baseline_m" in calib:
        logger.info(
            "Stereo params — focal_length=%.2f px, baseline=%.4f m",
            calib["focal_length_px"], calib["baseline_m"],
        )
        return float(calib["focal_length_px"]), float(calib["baseline_m"])

    # Detection format — extract from P2 and P3 matrices
    f        = float(calib["P2"][0, 0])
    tx_left  = float(calib["P2"][0, 3])
    tx_right = float(calib["P3"][0, 3])

    baseline = (tx_left - tx_right) / f
    if baseline <= 0:
        raise ValueError(
            f"Computed baseline is non-positive ({baseline:.6f} m). "
            f"Check calibration file — P2[0,3]={tx_left}, P3[0,3]={tx_right}, f={f}"
        )
    logger.info("Stereo params — focal_length=%.2f px, baseline=%.4f m", f, baseline)
    return f, baseline
