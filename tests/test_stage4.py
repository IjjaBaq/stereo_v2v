"""Tests for the Stage 4 fusion core (utils.fusion).

Covers the source-agnostic fusion core with synthetic data (no GPU, no dataset
needed): box registration, matching, noisy-OR confidence, static-pair merging,
and the fused/flagged/unmatched routing in fuse().

Run with: pytest tests/test_stage4.py -v
"""

import math

import numpy as np
import pytest
import yaml

from utils.fusion import (
    bev_distance,
    fuse,
    match_boxes,
    merge_static_pair,
    noisy_or,
    transform_box,
)

STAGE4_CONFIG = "config/stage4.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(STAGE4_CONFIG) as f:
        return yaml.safe_load(f)


def _box(label="Car", x=0.0, z=10.0, conf=0.5, heading=0.0,
         y=1.0, l=4.2, w=1.8, h=1.5):
    return {"label": label, "x": x, "y": y, "z": z, "l": l, "w": w, "h": h,
            "heading": heading, "confidence": conf}


# ---------------------------------------------------------------------------
# transform_box
# ---------------------------------------------------------------------------

class TestTransformBox:
    def test_identity_unchanged(self):
        b = _box(x=1.0, z=12.0, heading=0.3)
        t = transform_box(b, np.eye(4))
        assert t["x"] == pytest.approx(1.0)
        assert t["z"] == pytest.approx(12.0)
        assert t["heading"] == pytest.approx(0.3)

    def test_translation_moves_center(self):
        T = np.eye(4)
        T[0, 3] = 2.0   # +2m in camera x
        T[2, 3] = -3.0  # -3m in camera z
        t = transform_box(_box(x=0.0, z=10.0), T)
        assert t["x"] == pytest.approx(2.0)
        assert t["z"] == pytest.approx(7.0)

    def test_yaw_rotates_heading(self):
        th = math.radians(90)
        T = np.eye(4)
        T[:3, :3] = np.array([[math.cos(th), 0, math.sin(th)],
                              [0, 1, 0],
                              [-math.sin(th), 0, math.cos(th)]])
        t = transform_box(_box(heading=0.0), T)
        assert t["heading"] == pytest.approx(math.pi / 2, abs=1e-6)

    def test_dimensions_preserved(self):
        b = _box(l=4.2, w=1.8, h=1.5, conf=0.7)
        t = transform_box(b, np.eye(4))
        assert (t["l"], t["w"], t["h"], t["confidence"]) == (4.2, 1.8, 1.5, 0.7)


# ---------------------------------------------------------------------------
# bev_distance / noisy_or
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_bev_distance_ignores_y(self):
        a = _box(x=0.0, z=0.0, y=0.0)
        b = _box(x=3.0, z=4.0, y=100.0)
        assert bev_distance(a, b) == pytest.approx(5.0)

    def test_noisy_or_formula(self):
        assert noisy_or(0.6, 0.5) == pytest.approx(0.8)

    def test_noisy_or_monotonic(self):
        # combining never decreases confidence below either input
        assert noisy_or(0.9, 0.1) >= 0.9
        assert noisy_or(0.0, 0.0) == pytest.approx(0.0)
        assert noisy_or(1.0, 0.3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# match_boxes
# ---------------------------------------------------------------------------

class TestMatchBoxes:
    def test_close_same_class_matches(self):
        a = [_box("Car", x=0, z=10)]
        b = [_box("Car", x=0.5, z=10)]
        m, ua, ub = match_boxes(a, b, {"Car": 2.0})
        assert m == [(0, 0)] and not ua and not ub

    def test_beyond_threshold_no_match(self):
        a = [_box("Car", x=0, z=10)]
        b = [_box("Car", x=3.0, z=10)]
        m, ua, ub = match_boxes(a, b, {"Car": 2.0})
        assert not m and ua == [0] and ub == [0]

    def test_different_class_no_match(self):
        a = [_box("Car", x=0, z=10)]
        b = [_box("Pedestrian", x=0, z=10, l=0.8, w=0.8, h=1.7)]
        m, _, _ = match_boxes(a, b, {"Car": 2.0, "Pedestrian": 1.0})
        assert not m

    def test_greedy_picks_nearest(self):
        a = [_box("Car", x=0, z=10)]
        b = [_box("Car", x=1.5, z=10), _box("Car", x=0.2, z=10)]
        m, _, ub = match_boxes(a, b, {"Car": 2.0})
        assert m == [(0, 1)]    # nearest (index 1) wins
        assert ub == [0]

    def test_one_to_one(self):
        """Each prediction matches at most one GT and vice versa."""
        a = [_box("Car", x=0, z=10), _box("Car", x=0.1, z=10)]
        b = [_box("Car", x=0.05, z=10)]
        m, ua, ub = match_boxes(a, b, {"Car": 2.0})
        assert len(m) == 1 and len(ua) == 1 and not ub


# ---------------------------------------------------------------------------
# merge_static_pair
# ---------------------------------------------------------------------------

class TestMergeStaticPair:
    def test_confidence_weighted_center(self):
        a = _box(x=0.0, z=10.0, conf=0.6)
        b = _box(x=0.3, z=10.2, conf=0.5)
        m = merge_static_pair(a, b)
        # (0.6*0 + 0.5*0.3) / 1.1
        assert m["x"] == pytest.approx((0.5 * 0.3) / 1.1, abs=1e-3)

    def test_noisy_or_confidence(self):
        m = merge_static_pair(_box(conf=0.6), _box(conf=0.5))
        assert m["confidence"] == pytest.approx(0.8, abs=1e-4)

    def test_source_and_flag(self):
        m = merge_static_pair(_box(), _box())
        assert m["source"] == "fused" and m["is_dynamic"] is False

    def test_circular_heading_mean(self):
        # headings straddling ±pi average correctly via circular mean
        a = _box(heading=math.pi - 0.1, conf=0.5)
        b = _box(heading=-math.pi + 0.1, conf=0.5)
        m = merge_static_pair(a, b)
        assert abs(abs(m["heading"]) - math.pi) < 0.11


# ---------------------------------------------------------------------------
# fuse — routing
# ---------------------------------------------------------------------------

class TestFuse:
    def test_static_pair_fused(self, cfg):
        a = [_box("Car", x=0, z=10, conf=0.6)]
        b = [_box("Car", x=0.3, z=10.2, conf=0.5)]  # within match + static thresh
        out, stats = fuse(a, b, np.eye(4), cfg)
        assert stats["n_fused"] == 1
        assert len(out) == 1 and out[0]["source"] == "fused"

    def test_dynamic_pair_flagged_unmerged(self, cfg):
        # 1.5m apart: matches (<2.0) but exceeds static thresh (1.0) → dynamic
        a = [_box("Car", x=0, z=10, conf=0.6)]
        b = [_box("Car", x=1.5, z=10, conf=0.5)]
        out, stats = fuse(a, b, np.eye(4), cfg)
        assert stats["n_dynamic_flagged"] == 1 and stats["n_fused"] == 0
        assert {o["source"] for o in out} == {"vehicle_A", "vehicle_B"}
        assert all(o["is_dynamic"] for o in out)

    def test_unmatched_kept_with_source(self, cfg):
        a = [_box("Car", x=0, z=10)]
        b = [_box("Car", x=50, z=10)]   # far apart → both unmatched
        out, stats = fuse(a, b, np.eye(4), cfg)
        assert stats["n_only_a"] == 1 and stats["n_only_b"] == 1
        sources = sorted(o["source"] for o in out)
        assert sources == ["vehicle_A", "vehicle_B"]
        assert all(o["is_dynamic"] is False for o in out)

    def test_stats_account_for_all_inputs(self, cfg):
        a = [_box("Car", x=0, z=10), _box("Pedestrian", x=5, z=8, l=.8, w=.8, h=1.7)]
        b = [_box("Car", x=0.2, z=10)]
        out, stats = fuse(a, b, np.eye(4), cfg)
        assert stats["n_a"] == 2 and stats["n_b"] == 1
        # every input is represented: fused pairs + leftovers
        assert stats["n_fused"] + stats["n_only_a"] == 2 - 0  # A's 2 accounted
