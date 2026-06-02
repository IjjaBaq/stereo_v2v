"""Tests for the learned orientation heading path.

Covers the pure geometry that recovers global heading (rotation_y) from a
learned allocentric angle (alpha), plus the orientation model module's
angle encoding/decoding.

The geometry tests run with no GPU and no checkpoint — they pin the
correctness foundation (allocentric recovery + wraparound) that the whole
learned-heading approach rests on.

Run with: pytest tests/test_orientation.py -v
"""

import math

import numpy as np
import pytest

from utils.geometry import compute_heading, recover_rotation_y, wrap_to_pi


# ---------------------------------------------------------------------------
# wrap_to_pi
# ---------------------------------------------------------------------------

class TestWrapToPi:
    def test_within_range_unchanged(self):
        for a in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert wrap_to_pi(a) == pytest.approx(a, abs=1e-9)

    def test_above_pi_wraps(self):
        assert wrap_to_pi(math.pi + 0.5) == pytest.approx(-math.pi + 0.5, abs=1e-9)

    def test_below_neg_pi_wraps(self):
        assert wrap_to_pi(-math.pi - 0.5) == pytest.approx(math.pi - 0.5, abs=1e-9)

    def test_full_turn_wraps_to_self(self):
        assert wrap_to_pi(0.3 + 2.0 * math.pi) == pytest.approx(0.3, abs=1e-9)

    def test_result_always_in_range(self):
        for a in np.linspace(-20, 20, 200):
            w = wrap_to_pi(float(a))
            assert -math.pi <= w <= math.pi


# ---------------------------------------------------------------------------
# recover_rotation_y — the allocentric → egocentric recovery
# ---------------------------------------------------------------------------

class TestRecoverRotationY:
    @pytest.fixture(scope="class")
    def P2(self):
        # KITTI tracking seq 0000 left projection intrinsics
        p = np.zeros((3, 4), dtype=np.float64)
        p[0, 0] = 721.5377   # fx
        p[1, 1] = 721.5377   # fy
        p[0, 2] = 609.5593   # cx
        p[1, 2] = 172.854    # cy
        return p

    # KITTI stores alpha and rotation_y as independent annotations; they
    # satisfy rotation_y = alpha + atan2(x, z) only to ~1 deg (label noise,
    # confirmed varying by object type and unaffected by the P2 tx offset).
    # GT round-trips therefore use a 0.02 rad (~1.1 deg) tolerance.
    _LABEL_NOISE_RAD = 0.02

    def test_reproduces_gt_rotation_y_van(self, P2):
        """GT round-trip on seq 0000 frame 0 Van (full-precision label_02).

        GT: x=-4.552284, z=13.410495, alpha=-1.793451, rotation_y=-2.115488.
        cx_2d derived from the GT ray: x/z = (cx_2d - cx)/fx.
        Recovering rotation_y from GT alpha must reproduce GT rotation_y
        to within KITTI annotation precision.
        """
        fx = float(P2[0, 0])
        cx = float(P2[0, 2])
        x, z = -4.552284, 13.410495
        gt_alpha = -1.793451
        gt_roty = -2.115488

        cx_2d = cx + (x / z) * fx
        roty = recover_rotation_y(gt_alpha, cx_2d, P2)

        assert roty == pytest.approx(gt_roty, abs=self._LABEL_NOISE_RAD)

    def test_reproduces_gt_rotation_y_pedestrian(self, P2):
        """GT round-trip on seq 0000 frame 0 Pedestrian (full-precision).

        GT: x=6.301919, z=8.455685, alpha=-2.523309, rotation_y=-1.900245.
        """
        fx = float(P2[0, 0])
        cx = float(P2[0, 2])
        x, z = 6.301919, 8.455685
        gt_alpha = -2.523309
        gt_roty = -1.900245

        cx_2d = cx + (x / z) * fx
        roty = recover_rotation_y(gt_alpha, cx_2d, P2)

        assert roty == pytest.approx(gt_roty, abs=self._LABEL_NOISE_RAD)

    def test_exact_identity(self, P2):
        """Machine-precision check of the pure formula (no label noise).

        recover_rotation_y must equal wrap_to_pi(alpha + atan2(cx_2d-cx, fx))
        exactly — this is the contract the GT round-trips approximate.
        """
        fx = float(P2[0, 0])
        cx = float(P2[0, 2])
        for alpha in (-1.793451, -2.523309, 0.5, -0.5):
            for cx_2d in (364.63, 1147.31, 600.0):
                expected = wrap_to_pi(alpha + math.atan2(cx_2d - cx, fx))
                assert recover_rotation_y(alpha, cx_2d, P2) == pytest.approx(
                    expected, abs=1e-12
                )

    def test_zero_alpha_equals_ray_angle(self, P2):
        """alpha=0 must collapse to the old ray_angle behaviour."""
        cx_2d = 800.0
        assert recover_rotation_y(0.0, cx_2d, P2) == pytest.approx(
            compute_heading(cx_2d, P2, method="ray_angle"), abs=1e-9
        )

    def test_output_wrapped(self, P2):
        """Large alpha + ray must still land in [-pi, pi]."""
        for alpha in (3.0, -3.0, 2.9, -2.9):
            for cx_2d in (0.0, 600.0, 1240.0):
                roty = recover_rotation_y(alpha, cx_2d, P2)
                assert -math.pi <= roty <= math.pi


# ---------------------------------------------------------------------------
# Orientation model — sin/cos angle encoding round-trip
# ---------------------------------------------------------------------------

class TestAngleEncoding:
    """sin/cos encoding must round-trip any angle without wraparound loss."""

    def test_sincos_atan2_round_trip(self):
        from utils.orientation import decode_alpha

        for a in np.linspace(-math.pi, math.pi, 50):
            sin_v, cos_v = math.sin(a), math.cos(a)
            decoded = decode_alpha(sin_v, cos_v)
            # compare on the circle (handle ±pi identity)
            diff = wrap_to_pi(decoded - float(a))
            assert abs(diff) < 1e-6

    def test_decode_normalizes_unnormalized_input(self):
        """Raw network outputs need not be unit-norm; decode must still work."""
        from utils.orientation import decode_alpha

        a = 0.7
        scale = 5.0
        decoded = decode_alpha(scale * math.sin(a), scale * math.cos(a))
        assert wrap_to_pi(decoded - a) == pytest.approx(0.0, abs=1e-6)
