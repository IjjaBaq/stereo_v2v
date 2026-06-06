# Stereo V2V — Project Status Report
*As of 2026-06-05 · branch `main`*

## 1. What the project is

The goal is **cooperative perception between two vehicles (V2V)**: each vehicle
perceives with its own **stereo camera**, lifts what it sees to 3D, and the two
vehicles **share and fuse** their detections so that together they perceive more
than either alone.

The system is a **4-stage pipeline**. Each stage has its own config file, MLflow
logging, and a standalone validation step.

| Stage | Name | Input → Output |
|------|------|----------------|
| 1 | Depth | Stereo image pair → dense disparity / metric depth |
| 2 | Detect | Left image → 2D object boxes (RT-DETR) |
| 3 | Lift | 2D boxes + depth → **3D position** (x, y, z) + source 2D box |
| 4 | Fusion | Two agents' 3D boxes → one fused, registered scene |

**Two data sources, by design:**
- **KITTI** — real-image stereo, used for Stages 1–3 (object split for Stages 1–2
  single-frame validation; tracking sequences 0000–0004 for the Stage 3 chain).
- **CARLA** — intended source for Stage 4 (true simultaneous multi-agent V2V).
  **Not wired yet:** there is no `data/carla/` and the CARLA loader is a stub.

> **Scope note.** Stage 3 emits **position only** — `label, confidence, x, y, z,
> x1, y1, x2, y2`. Object **size** and **heading** are not produced: neither is
> recoverable from stereo geometry at range (see §4). Earlier work on a
> learned-orientation head was removed.

---

## 2. Current validated metrics

All numbers below come from `validation_results.json` files in `outputs/` and
from the depth-sampling study in `experiments/percentile_choice.md`. Sample
sizes are stated; small-sample numbers are flagged as indicative, not
generalization estimates.

### Stage 1 — Depth (KITTI object split, **5 samples**, frames 000000–000004)
`outputs/depth/object/{sgbm,waft}/validation_results.json`

| Metric | SGBM | WAFT |
|---|---|---|
| EPE (end-point error) | 4.80 px | **0.89 px** |
| D1 (outlier rate) | 16.00 % | **2.69 %** |
| Coverage (valid GT px) | 32.1 % | **100 %** |

**WAFT is the trusted, more accurate depth method** (~5.4× lower EPE, ~6× lower
D1, full coverage). This supersedes the earlier "WAFT output is suspect" status —
that was a bug in the retired offline-precompute path, now fixed (WAFT runs
inference directly; see §3). SGBM remains a valid sparse baseline.

### Stage 2 — 2D detection (RT-DETR, object split, **10 samples**, IoU ≥ 0.5)
`outputs/detections/object/validation_results.json`

| Class | AP |
|---|---|
| Car | 0.860 |
| Pedestrian | 1.000 |
| **mAP** | **0.930** |

Confirms the detector runs and maps COCO→KITTI correctly. 10 frames is still a
small evaluation set — indicative, not a benchmark accuracy.

### Stage 3 — 3D lifting (position-only)
The end-to-end **matching** metrics (TP/FP/FN, centre distance) have **not been
re-run cleanly at the newly tuned depth-sampling percentiles** (the
`outputs/lift3d/**/validation_results.json` files accumulate frames from several
prior runs at the *old* percentiles and are superseded — do not cite them).

What **is** current and clean is the **depth-sampling accuracy study**
(`experiments/percentile_choice.md`): a multi-sequence sweep on static parked
cars (KITTI tracking 0000–0004, 66 detection↔GT pairs, association fixed by 2D
IoU so the percentile is isolated from matching). Headline depth-error (|Z − GT
centre z|) at the chosen percentiles:

| Method | Percentile (new ← old) | Aggregate MAE | Notes |
|---|---|---|---|
| SGBM | **p20** ← p75 | **1.72 m** | ~60 % better than the old p75 (4.27 m) |
| WAFT | **p35** ← p60 | 3.44 m (near-zero bias) | mean dominated by far cars; near-field much lower |

Per-sequence MAE shows the near/far split clearly (WAFT @p35): 0000 = 1.84 m,
0001 = 2.65 m, 0003 = 4.27 m, but 0002 = 9.25 m and 0004 = 3.48 m at 30–45 m
range. Beyond ~30 m, depth error is governed by the depth **map** at range
(disparity error amplified by `Z = f·B/d`), not by the sampling percentile.

- **Status:** ⚙️ Code current and unit-tested; depth-sampling percentiles freshly
  re-tuned (2026-06-05); end-to-end matching re-validation at the new config is
  pending.

### Stage 4 — V2V fusion: **not yet validated** (CARLA not wired)
The fusion **core** (`utils/fusion.py`) is implemented and unit-tested (26 tests),
but it cannot run end-to-end: `utils/carla_loader.py` and
`stages/validate_stage4_fusion.py` are stubs that raise `NotImplementedError`.
Any files under `outputs/fusion/**` are leftovers from the retired KITTI
temporal-simulation backend and are not produced by the current code.

---

## 3. Stage-by-stage: what it does, how, and status

### Stage 1 — Depth (`stages/stage1_depth.py`)
Computes a disparity map from a rectified stereo pair, converts to metric depth
via `Z = f·B/d`, saves `.npy` + a colorized `.png`.
- **SGBM** — OpenCV `StereoSGBM`; invalid pixels → NaN (sparse, ~32 % coverage).
- **WAFT** — a learned deep-stereo net (WAFT-Stereo) run **directly in-process**
  (`get_cfg` + `WAFT(cfg)` + checkpoint, PEFT `merge_and_unload`, module-level
  cached; CUDA if available else CPU, ~85 s/image on CPU). The old offline
  precompute path and its three bugs (÷255 input, positional call, wrong output
  tensor) are gone.
- **Status:** ✅ Both work. **WAFT is the trusted, more accurate method**; SGBM is
  the sparse baseline. Run `--method waft` from the project root so WAFT-Stereo
  imports resolve.

### Stage 2 — Detection (`stages/stage2_detect.py`)
Runs pretrained **RT-DETR** on the left image, keeps COCO classes mapped to KITTI
(`car→Car`, `person→Pedestrian`), thresholds by confidence. Outputs 2D boxes
`{label, confidence, x1, y1, x2, y2}`.
- **Status:** ✅ Working (mAP 0.930 over 10 frames).

### Stage 3 — Lift to 3D position (`stages/stage3_lift.py`)
For each 2D box: samples a single disparity value inside the box (a per-method
**percentile**, with an optional top-crop + min-depth gate to reject road/ground),
converts to depth, and **unprojects the box centre** to a 3D point `(x, y, z)`.
Confidence is propagated as `conf_2d × coverage_ratio`. The source 2D box is
carried for matching. Validation (`validate_stage3_lift.py`) matches
predictions↔GT by **3D centre distance** per class (`matching.max_dist`: Car 2.0,
Ped 1.0) and reports `mean_depth_err` + `mean_center_dist`.
- **Depth-sampling percentiles (tuned 2026-06-05, `experiments/percentile_choice.md`):**
  the two methods need **opposite** percentiles because their valid-pixel
  distributions differ. **SGBM = `percentile_20`** (sparse; the consistency check
  already drops background, so valid pixels sit on the car's near surface — a low
  percentile best matches centre depth; the old `percentile_75` over-corrected and
  was the worst end of the curve). **WAFT = `percentile_35`** (dense; after a
  top-40 % crop + 6 m gate, a mid-low percentile gives near-zero bias; the old
  `percentile_60` sampled the near surface).
- **Status:** ⚙️ Code current and unit-tested; end-to-end matching re-validation at
  the new percentiles pending.

### Stage 4 — V2V fusion (`stages/stage4_fusion.py` + `utils/fusion.py`)
The **source-agnostic core** registers Vehicle B's boxes into Vehicle A's frame
via a 4×4 transform, greedily matches by BEV centre distance per class, and merges
corroborated pairs (noisy-OR confidence, confidence-weighted centre; size/heading
merged only if present — the core handles both position-only and full-3D boxes).
The CARLA backend (`run_carla`) is the data plumbing around it.
- **Status:** ⚙️ Core implemented + unit-tested (26 tests). 🔲 **Cannot run
  end-to-end** — CARLA loader and Stage 4 validation are stubs pending a real
  CARLA export.

---

## 4. Known issues

1. **Depth error grows sharply with range.** WAFT's disparity map is accurate
   (EPE 0.89 px on the object split), but lifting *distant* cars still incurs
   multi-metre error because `Z = f·B/d` amplifies small disparity errors at low
   disparity. The Stage-3 study shows static-car depth MAE rising from ~1.8 m
   near (≤ 20 m) to ~9 m at 30–45 m. This is geometry + far-range depth, **not** a
   WAFT bug, and no sampling percentile fixes it. SGBM additionally loses coverage
   at range (some far-car boxes have no valid pixels).
2. **Stage 3 end-to-end metrics need a clean re-run.** The
   `outputs/lift3d/**/validation_results.json` files accumulate frames from
   multiple runs at the *old* percentiles (p75/p60) and must be regenerated at the
   new config (p20/p35) before any Stage-3 TP/FP/centre-distance numbers are cited.
3. **Object split is not used for Stage 3.** Object-split stereo and detection
   frames are different scenes; Stage 3 is chained only on the tracking split.
4. **CARLA not wired.** `utils/carla_loader.py` and
   `stages/validate_stage4_fusion.py` raise `NotImplementedError`. Stage 4 has no
   end-to-end run or validation until a CARLA export exists. The verified CARLA
   coordinate conventions are documented in the loader's docstring.
5. **Heading/orientation is intentionally out of scope.** Stereo cannot recover
   per-object heading at range (ray-angle assumes the object faces the camera ray;
   pseudo-LiDAR PCA locks onto depth noise; a learned head only reached a ~69°
   road-aligned prior). Stage 3 reports position only by design.

---

## 5. Test health

Full suite: **157 tests, all passing** (`stereo_v2v_env`):

| File | Count |
|---|---|
| `test_loader.py` | 36 |
| `test_stage1.py` | 29 |
| `test_stage2.py` | 33 |
| `test_stage3.py` | 33 |
| `test_stage4.py` | 26 |

Stage 4 includes coverage for **both** box schemas through the fusion core
(position-only and full 3D).

---

## 6. Done · pending · honest one-liners

| Stage | Status | One line |
|---|---|---|
| 1 Depth | ✅ working | WAFT accurate (EPE 0.89 px, 5 frames) and trusted; SGBM is the sparse baseline. |
| 2 Detect | ✅ working | RT-DETR runs and maps to KITTI; mAP 0.930 over 10 frames. |
| 3 Lift | ⚙️ code current, depth-sampling re-tuned | Emits 3D position + 2D box; percentiles tuned (SGBM p20, WAFT p35); end-to-end re-run pending. |
| 4 Fusion | ⚙️ core done, 🔲 e2e blocked | Source-agnostic fusion unit-tested; cannot run until CARLA is wired. |

### Takeaway
> The 4-stage scaffold is in place and unit-tested (157 tests green). Stage 1
> (WAFT now accurate and trusted, SGBM baseline) and Stage 2 run with spot-check
> validation. Stage 3 produces honest stereo-recoverable output (3D position + 2D
> box); its per-method depth-sampling percentiles were just re-tuned on a
> multi-sequence study (`experiments/percentile_choice.md`), and an end-to-end
> matching re-validation at the new config is the immediate next step. Stage 4's
> fusion core is complete and schema-flexible, but **the V2V result cannot be
> demonstrated until CARLA data is wired in** — that remains the single biggest
> gap between the current state and the project goal.
