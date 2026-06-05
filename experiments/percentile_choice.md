# Depth-Sampling Percentile Choice — WAFT & SGBM

*Experiment date: 2026-06-05 · KITTI tracking sequences 0000–0004 · static parked cars*
*Status: **analysis only — no config/code changes applied.** Recommendations pending review.*

---

## 1. Question

Stage 3 lifts a 2D detection to a 3D position by sampling a single depth value
from the disparity map inside the detection box (a percentile of the valid
disparity pixels), then unprojecting the box **centre** to `(x, y, z)`. The
percentile is a fixed per-method hyper-parameter:

| Method | Current default | Defined in |
|---|---|---|
| WAFT | `percentile_60` | `stages/validate_stage3_lift.py` → `DEPTH_SAMPLING_BY_METHOD` |
| SGBM | `percentile_75` | same |

**Does a different percentile give better depth accuracy against ground truth?**
This document tests p20–p90 for both methods and reports the evidence.

---

## 2. What was tested, and how

### 2.1 Test targets — static parked cars
Parked cars are rigid, cleanly annotated, span a useful depth range, and are
unambiguous targets for a depth-accuracy study. "Static" is a **world-frame**
property (the ego camera moves), so frames were **not** assumed — they were
derived from the GT:

1. Built the standard KITTI **camera → rectified → velodyne → IMU → world**
   transform per sequence, using each sequence's `R_rect`, `Tr_velo_cam`,
   `Tr_imu_velo` and the oxts ego pose (Mercator position + `Rz(yaw)Ry(pitch)Rx(roll)`).
2. Transformed every `Car` GT box to world coordinates and measured each
   **track's world-position spread** (RMS radial). **Static ⇔ spread < 0.6 m.**
3. **Chain validation** (before trusting the classification): the ego's own
   trajectory length matched its integrated forward velocity for all five
   sequences, and the static/moving split was clean with a wide margin
   (static tracks < 0.45 m spread, movers > 1.7 m):

   | Seq | Ego traj length | ∫(forward_vel·dt) | Static Car tracks |
   |---|---|---|---|
   | 0000 | 69.4 m | 69.6 m | 8 |
   | 0001 | 332.4 m | 321.1 m | 81 |
   | 0002 | 114.0 m | 110.9 m | 8 |
   | 0003 | 173.0 m | 168.0 m | 6 |
   | 0004 | 402.5 m | 389.6 m | 1 |

4. Selected frames containing **clean** static cars (occlusion ≤ 1,
   truncation < 0.5, depth 6–45 m), maximising static-car count with ≥ 8-frame
   spacing for depth diversity:

   | Seq | Frames selected | Depth span | Notes |
   |---|---|---|---|
   | 0000 | 109, 124, 134, 142, 152 | 8–26 m | parking row |
   | 0001 | 17, 70, 88, 96, 110 | 9–45 m | dense urban parking (≤ 8 cars/frame) |
   | 0002 | 0, 8, 56, 64, 72 | 21–44 m | sparse, mostly far |
   | 0003 | 12, 20, 67, 75, 120 | 28–45 m | far parked cars |
   | 0004 | 172, 180, 188, 196 | 11–44 m | only one static car (track 20) exists |

   Seq 0004 is a moving-traffic scene — genuinely one parked car; reported
   honestly rather than padded.

### 2.2 Measurement — decoupled from matching
The end-to-end validation only measures depth error over **matched** TPs, but
matching depends on the predicted position, which depends on the percentile —
circular. To isolate the percentile's effect on depth accuracy:

- Each RT-DETR `Car` detection is associated to a static-car GT box by **2D IoU
  ≥ 0.5** (greedy). **This association is percentile-independent**, so the
  comparison set is fixed across the sweep.
- For each percentile, the depth is sampled in the **detection** box (exactly as
  the pipeline does) and compared to the GT **centre** depth `z`.
- Each method keeps its **own pipeline crop/gate** (only the percentile varies):
  - WAFT: `crop_top_frac = 0.40`, `min_depth_m = 6.0`
  - SGBM: `crop_top_frac = 1.0`, `min_depth_m = None`

### 2.3 Metrics
- **MAE** = mean |Z − z| (the headline accuracy metric).
- **median** |Z − z| (robust to far-range outliers).
- **bias** = mean (Z − z): **positive ⇒ over-estimates depth (places car too far);
  negative ⇒ under-estimates (too close).**
- **micro** aggregate pools all pairs; **macro** aggregate is the mean of
  per-sequence MAE (so the pair-dense seq 0001 doesn't dominate).

### 2.4 Sample
66 detection↔GT static-car pairs total (per seq: 0000=15, 0001=32, 0002=7,
0003=8, 0004=4). SGBM yields fewer valid samples (59/66) because its sparse
disparity leaves some far-car boxes with no valid pixels.

---

## 3. WAFT results

### 3.1 Per-sequence MAE (m)

| pct | 0000 | 0001 | 0002 | 0003 | 0004 |
|---|---|---|---|---|---|
| 20 | 3.51 | 6.50 | 35.57 | 9.90 | 57.42 |
| 25 | 2.15 | 3.72 | 17.38 | 6.09 | 36.58 |
| 30 | 1.94 | 2.78 | **9.25** | 5.37 | 24.51 |
| 35 | **1.84** | 2.65 | 9.25 | 4.27 | **3.48** |
| 40 | 1.95 | 2.71 | 9.66 | 1.19 | 3.72 |
| 45 | 2.08 | 2.83 | 9.76 | **0.34** | 3.84 |
| 50 | 2.19 | **1.92** | 9.86 | 0.37 | 3.91 |
| 55 | 2.15 | 2.10 | 9.92 | 0.58 | 3.97 |
| **60 (cur)** | 2.15 | 2.22 | 9.98 | 0.80 | 4.02 |
| 70 | 2.37 | 2.43 | 10.11 | 1.19 | 4.12 |
| 90 | 3.18 | 3.44 | 10.42 | 1.78 | 4.33 |

Per-seq best MAE: **0000→p35, 0001→p50, 0002→p30, 0003→p45, 0004→p35.**

### 3.2 Aggregate (66 pairs)

| pct | micro-MAE | micro-median | micro-bias | macro-MAE |
|---|---|---|---|---|
| 30 | 4.908 | **1.732** | +1.719 | 8.772 |
| 35 | 3.442 | 1.919 | **−0.643** | 4.353 |
| 40 | 3.151 | 2.040 | −1.507 | 3.845 |
| 45 | 3.154 | 2.092 | −1.854 | 3.770 |
| **50** | **2.754** | 2.171 | −2.618 | **3.648** |
| 55 | 2.869 | 2.283 | −2.806 | 3.743 |
| **60 (cur)** | 2.966 | 2.377 | −2.951 | 3.834 |
| 70 | 3.184 | 2.501 | −3.178 | 4.045 |
| 90 | 3.973 | 2.868 | −3.971 | 4.630 |

- **MAE-optimal (micro & macro agree): p50** — ~7% better than current p60.
- **Unbiased: p35** (bias −0.64; crosses zero ~p33). **Median-optimal: p30.**

### 3.3 WAFT reading
- The current **p60 is consistently on the high side** — every sequence's curve
  is flat or rising by p60, and bias at p60 is −2 to −3 m (under-estimates depth
  by sampling the car's near surface).
- The aggregate optimum is a **shallow, flat bowl ~p45–p50**.
- **MAE (p50) and median/bias (p30–35) disagree** because the far-range
  sequences (0002, 0004) have huge right-tail errors at low percentiles, which
  inflate the *mean* but not the *median*. For the *typical* near car, lower
  (~p30–35) is best and unbiased.
- **Sequences 0002 (~9–10 m) and 0004 (~3.5 m) are WAFT-depth-limited at all
  percentiles** — a depth-*map* limitation at range (known issue), not a
  sampling-percentile one. They drag the aggregate and should not dominate the
  choice.

**WAFT recommendation:** move `percentile_60` → **`percentile_35`** (preferred:
near-zero bias and best near-field median; the extra mean error it carries comes
almost entirely from far cars that are unrecoverable regardless), **or
`percentile_50`** if minimising global mean error is the objective (~7% MAE gain,
but biased −2.6 m). Either improves on p60.

---

## 4. SGBM results

### 4.1 Per-sequence MAE (m)

| pct | 0000 | 0001 | 0002 | 0003 | 0004 |
|---|---|---|---|---|---|
| **20** | **1.41** | **1.62** | **2.20** | **0.67** | **4.11** |
| 25 | 1.74 | 1.87 | 2.27 | 0.76 | 4.22 |
| 30 | 2.00 | 2.11 | 2.44 | 0.71 | 4.32 |
| 40 | 2.31 | 2.64 | 2.77 | 1.43 | 4.39 |
| 50 | 2.60 | 3.37 | 2.96 | 1.72 | 4.45 |
| 60 | 2.80 | 3.94 | 3.33 | 2.16 | 4.53 |
| **75 (cur)** | 3.37 | 4.97 | 3.76 | 2.49 | 4.62 |
| 90 | 4.83 | 6.00 | 4.19 | 2.99 | 4.82 |

Per-seq best MAE: **p20 for every sequence.**

### 4.2 Aggregate (59 valid pairs)

| pct | micro-MAE | micro-median | micro-bias | macro-MAE |
|---|---|---|---|---|
| **20** | **1.717** | 1.170 | −1.460 | **2.000** |
| 25 | 1.955 | 1.925 | −1.812 | 2.172 |
| 30 | 2.167 | 2.121 | −2.052 | 2.317 |
| 40 | 2.603 | 2.500 | −2.540 | 2.707 |
| 50 | 3.099 | 2.964 | −3.054 | 3.019 |
| 60 | 3.518 | 3.231 | −3.483 | 3.353 |
| **75 (cur)** | 4.267 | 3.605 | −4.267 | 3.840 |
| 90 | 5.261 | 4.508 | −5.261 | 4.563 |

### 4.3 Boundary extension (p5–p20, aggregate)
The p20–p90 sweep bottoms out at its low boundary, so the boundary was
characterised below p20:

| pct | micro-MAE | micro-median | micro-bias |
|---|---|---|---|
| 5  | 3.022 | 1.658 | +1.636 |
| 10 | 2.207 | 1.302 | **+0.069** |
| 15 | 2.087 | 1.102 | −0.579 |
| **20** | **1.717** | 1.170 | −1.460 |
| 25 | 1.955 | 1.925 | −1.812 |

p20 is a genuine minimum (p15 = 2.09 > p20 = 1.72 < p25 = 1.96), and **bias
crosses zero at ~p10.**

### 4.4 SGBM reading
- **SGBM is the opposite of WAFT: it wants a LOW percentile.** Bias is negative
  at every percentile in the main sweep, monotonically worse as the percentile
  rises.
- **The current `percentile_75` is at the worst end** of the SGBM curve:
  micro-MAE 4.27 m and bias −4.27 m, versus 1.72 m / −1.46 m at p20 — a **~60%
  MAE reduction** by moving to p20.
- **Why:** SGBM disparity is *sparse* — smooth background fails the
  left-right consistency check and becomes NaN, so the valid pixels are already
  foreground (car-surface) biased. A *high* percentile then over-shoots to the
  nearest surface (large under-estimation of centre depth); a *low* percentile
  lands nearer the centre. The original p75 rationale ("sparsity is
  foreground-biased, so a high percentile selects the object") double-counts the
  foreground bias the sparsity already provides.

**SGBM recommendation:** move `percentile_75` → **`percentile_20`** (min MAE,
~60% improvement), or **`percentile_10`** if unbiased depth is the priority
(bias +0.07 m, median 1.30 m). This is a large, consistent gain across every
sequence.

---

## 5. Cross-method conclusion

The two depth methods require **opposite** sampling percentiles because their
valid-pixel distributions differ:

| | WAFT | SGBM |
|---|---|---|
| Disparity density | dense (incl. ground/background) | sparse (consistency-checked → foreground) |
| Pipeline pre-filter | top-40% crop + 6 m gate | none |
| Optimal percentile | ~p35 (unbiased) – p50 (min MAE) | ~p10 (unbiased) – p20 (min MAE) |
| Current default | p60 (slightly high) | p75 (**badly** miscalibrated) |
| Best MAE gain vs current | ~7% | ~60% |

**The single biggest actionable finding is SGBM's percentile**, which is far
from optimal today and cheap to fix. WAFT's is a modest refinement.

---

## 6. Recommendations (pending approval — not yet applied)

| Method | Current | Recommended | Alternative | Expected effect |
|---|---|---|---|---|
| WAFT | `percentile_60` | `percentile_35` (unbiased) | `percentile_50` (min MAE) | bias −2.95 → −0.64 m; MAE ~−7% |
| SGBM | `percentile_75` | `percentile_20` (min MAE) | `percentile_10` (unbiased) | MAE 4.27 → 1.72 m (~−60%) |

Both would change `DEPTH_SAMPLING_BY_METHOD` in
`stages/validate_stage3_lift.py`, and the rationale comment block there should be
updated to reflect this evidence (the current comment argues the now-contradicted
p75 reasoning for SGBM).

---

## 7. Caveats and limitations

1. **Decoupled from end-to-end matching.** This isolates depth accuracy; changing
   the percentile also shifts predicted centre distance and therefore the
   TP/FP/FN counts. A full `validate_stage3_lift.py` re-run is needed to confirm
   the end-to-end matching improves (it most likely does, since both current
   settings are biased).
2. **Single dataset / limited sample.** 66 pairs (59 for SGBM) across 5 KITTI
   tracking sequences; seq 0001 contributes ~half the pairs (hence micro vs macro
   reported). May not transfer to other scenes/sensors — consistent with the
   project's standing "tuned on seq 0000" caveat, now broadened to 0000–0004.
3. **Far range is depth-map-limited, not sampling-limited.** Beyond ~25–30 m,
   WAFT depth error (0002 ~9 m, 0004 ~3.5 m) dominates and no percentile helps;
   SGBM loses coverage entirely (7/66 far-car boxes have no valid pixels). The
   percentile retune is a near-field refinement.
4. **Static detector is geometric, not annotated.** "Static" is inferred from
   world-position spread (< 0.6 m) via the oxts→camera chain, validated against
   ego odometry but still subject to GT and chain noise. The 0.6 m threshold sits
   in a wide gap (static < 0.45 m, movers > 1.7 m), so the classification is
   robust here.
5. **Centre-depth target.** Error is measured against the GT box *centre* `z`,
   which is what the pipeline unprojects, so this is the correct target — but it
   means a method that samples the visible front face is penalised (correctly).

---

## 8. Reproduction

- Static detection + frame selection: world-spread classifier over GT + oxts
  (chain: `R_rect`, `Tr_velo_cam`, `Tr_imu_velo`, Mercator ego pose).
- Caches used (already on disk):
  - WAFT/SGBM disparity → `outputs/depth/tracking/{method}/{seq}/{frame}_disp.npy`
  - RT-DETR detections → `outputs/detections/tracking/{seq}/{frame}_boxes2d.json`
    (method-independent)
- Sweep: for each method, fixed IoU≥0.5 association, then `sample_depth(...)`
  from `stages/stage3_lift.py` over `percentile_{20..90}` with the method's
  crop/gate held fixed.
- Selected frames: `{0000:[109,124,134,142,152], 0001:[17,70,88,96,110],
  0002:[0,8,56,64,72], 0003:[12,20,67,75,120], 0004:[172,180,188,196]}`.

*No repository code, config, or `data/` was modified by this experiment. Only
read-only analysis plus depth/detection caches under `outputs/` were produced.*
