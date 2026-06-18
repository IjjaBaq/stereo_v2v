# Depth-Sampling Percentile Choice — WAFT & SGBM

*Experiment date: 2026-06-05 · KITTI tracking sequences 0000–0004 · static parked cars*
*Status: **analysis only — no config/code changes applied.** Recommendations pending review.*

> **2026-06-18 re-check on workstation data — see [Section 9](#9-re-check-on-full-workstation-data-2026-06-18).**
> The p20 (SGBM) / p35 (WAFT) values from this 5-sequence near-field study were
> re-tested on the full workstation run (21 KITTI sequences + CARLA, ~10× the
> sample, both decoupled **and** end-to-end). **SGBM p20 holds up and is kept.
> WAFT p35 did not** — it over-estimated depth on the broader data and crippled
> CARLA fusion. **WAFT was changed `p35` → `p50`** (the balanced KITTI/CARLA
> optimum), which lifts CARLA WAFT fused recall from 0.31 to 0.78 (now on par with
> SGBM). Applied in `config/stage3.yaml`; corrected outputs regenerated under
> `outputs_waft_p50/` (originals untouched).

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

---

## 9. Re-check on full workstation data (2026-06-18)

*Re-check date: 2026-06-18 · source: `outputs_workstation/` + `mlflow_workstation.db` ·
KITTI tracking **all 21 sequences** (673 IoU-associated pairs) **and CARLA** (both
agents, 698 pairs) · **analysis only — config still reads SGBM p20 / WAFT p35,
nothing changed.***

### 9.1 Why re-check

Sections 3–6 chose p20 (SGBM) / p35 (WAFT) from **66 pairs across 5 sequences of
near-field static parked cars**, decoupled from matching — and flagged exactly two
risks (caveats #1 and #2): the sample might not transfer, and the percentile needs
an **end-to-end** re-run to confirm it improves matching, not just decoupled depth
MAE. The workstation run now provides both: ~10× the sample, the full depth range,
a second domain (CARLA), and cached disparity + detections that let the choice be
re-tested two ways.

### 9.2 Method (two criteria)

Using the cached WAFT/SGBM disparity (`*_disp.npy`) and RT-DETR detections
(`*_boxes2d.json`) — no re-inference:

1. **Decoupled depth accuracy** (as in §2): detection↔GT by 2D IoU ≥ 0.5
   (percentile-independent), sample depth in the box at each percentile, compare to
   GT centre depth → MAE / median / bias. Each method keeps its own crop/gate
   (SGBM 1.0 / none; WAFT 0.40 / 6 m).
2. **End-to-end Stage 3** (the criterion the pipeline optimises, and the §7 caveat-#1
   gap): lift every detection at each percentile (`sample_depth` → `unproject_box`,
   skip `<10` px), match to GT by 3D centre distance ≤ 2 m → TP/FP/FN, recall, F1,
   centre-dist.

**Harness validation:** at the current settings the end-to-end sweep reproduces the
workstation Stage-3 numbers exactly — SGBM p20 → TP 349 / R 0.348 / cdist 1.08;
WAFT p35 → TP 271 / R 0.270 / cdist 1.13 — so the off-line re-lift is trustworthy.

### 9.3 SGBM — p20 confirmed (near-optimal, low-bias)

| Criterion | Optimum | p20 (current) |
|---|---|---|
| KITTI decoupled MAE (673 pr) | p15 = 2.60 m | 2.62 m (tie); bias −1.85, p10 bias +0.13 |
| KITTI end-to-end **F1** | p15 = 0.353 | 0.319 (R 0.348 vs 0.385) |
| CARLA decoupled MAE (698 pr) | p25 = 1.06 m | **1.13 m, bias +0.10** |

p20 sits in a flat low-bias bowl in both domains (KITTI marginally prefers p10–15,
CARLA prefers p20–25). **A robust compromise — keep.** The only refinement on the
table is a small recall/F1 gain on KITTI by dropping to p15, partly offset by a
slightly worse centre-dist; not worth disturbing the CARLA-optimal p20–25.

### 9.4 WAFT — p35 is **too low** in both domains

| Criterion | Optimum | p35 (current) |
|---|---|---|
| KITTI end-to-end **F1** | **p45 = 0.267** (p40 0.265) | 0.236 (R 0.270 vs **0.305** at p45) |
| KITTI decoupled MAE (673 pr) | p60 = 2.74 m | 9.82 m, **bias +7.7 m** |
| CARLA decoupled MAE (698 pr) | p60 = 0.92 m | 14.95 m, **bias +14.8 m** |

WAFT p35 **systematically over-estimates depth** (places cars too far) on the
broader data: the dense WAFT map keeps far-background pixels inside the box, and a
low percentile latches onto them. The near-field static study masked this because
its boxes were close and background-free.

End-to-end KITTI F1 by percentile (21 seqs): p35 **0.236** → p40 0.265 → **p45
0.267** → p50 0.253 → p60 0.215. The end-to-end optimum (~p45) is *lower* than the
decoupled-MAE optimum (~p60) because a high percentile keeps shrinking the far-car
mean error (which dominates the mean) but starts under-shooting the *matchable*
near/mid cars, pushing them outside the 2 m gate. **For the pipeline, ~p45 is the
right target.**

This also helps explain the poor Stage-4 WAFT fusion (Chapter 5: recall 0.31,
corroboration 0.004): with p35 the follower's far cars are lifted ~15 m too far, so
almost nothing falls within the 1 m merge gate.

### 9.5 Updated cross-method picture

| | WAFT | SGBM |
|---|---|---|
| Current config | p35 | p20 |
| End-to-end (KITTI) optimum | **~p45** (+≈13 % F1/recall) | ~p15 (p20 ≈ optimum) |
| Depth-MAE optimum (KITTI / CARLA) | ~p60 / ~p60 | ~p15 / ~p25 |
| Verdict | **sub-optimal — raise to ~p45–50** | **keep p20** |

### 9.6 Decision and action taken

The new sample is **sufficient to decide** (≈10× larger, full range, two domains,
end-to-end + decoupled).

- **SGBM `p20`: kept.** Validated near-optimal and low-bias in both domains.
- **WAFT `p35` → `p50`: changed.** p35 was too low and systematically
  over-estimated depth. The two domains' optima differ — KITTI Stage-3 end-to-end
  peaks at ~p45, while CARLA fusion keeps improving to ~p55–60 — so **p50 is the
  balanced choice**: KITTI Stage-3 F1 = 0.253 (vs 0.236 at p35, near the p45 peak of
  0.267) and CARLA fused recall = 0.78 / precision 0.76 / corroboration 0.27.

The CARLA effect is the decisive one. WAFT fusion at each percentile (150 frames,
cached disparity/detections, real Stage-4 scoring):

| WAFT pct | fused recall | fused prec | corrob. | merges | fused loc-err (m) |
|---|---|---|---|---|---|
| 35 (old) | 0.31 | 0.27 | 0.004 | 1 | 0.96 |
| 45 | 0.68 | 0.62 | 0.10 | 28 | 0.81 |
| **50 (new)** | **0.78** | **0.76** | **0.27** | **74** | **0.61** |
| 55 | 0.79 | 0.81 | 0.36 | 99 | 0.53 |
| 60 | 0.81 | 0.84 | 0.38 | 105 | 0.62 |

The earlier "WAFT collapses for fusion" result (Chapter 5 draft: corroboration
0.004, 1 merge) was therefore **a depth-sampling artefact of p35, not an inherent
property** — at p50 WAFT fusion is on par with SGBM (SGBM: recall 0.77, prec 0.79,
corroboration 0.26). KITTI Stage-3 lifting still modestly favours SGBM (F1 0.32 vs
0.25), so "confident sparsity helps lifting" survives, but only weakly.

**Action:** `config/stage3.yaml` `per_method_overrides.waft.depth_sampling` set to
`percentile_50`. Corrected WAFT Stage-3 (KITTI) and Stage-4 (CARLA) outputs were
regenerated **from the cached disparity + detections** (no WAFT/RT-DETR re-inference)
into `outputs_waft_p50/`; `outputs/`, `outputs_workstation/`, and the workstation
MLflow store were left untouched. SGBM outputs are unchanged.
