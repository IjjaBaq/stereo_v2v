# Stereo V2V — Project Status Report
*As of 2026-06-03 · branch `master` · commit `48753c1`*

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
  single-frame validation; tracking sequences 0000–0002 for the Stage 3 chain).
- **CARLA** — intended source for Stage 4 (true simultaneous multi-agent V2V).
  **Not wired yet:** there is no `data/carla/` and the CARLA loader is a stub.

> **Scope note (changed 2026-06-03).** Stage 3 was simplified to emit **position
> only** — `label, confidence, x, y, z, x1, y1, x2, y2`. Object **size** and
> **heading** are no longer produced: neither is recoverable from stereo geometry
> at range (see §4). Earlier work on a learned-orientation head was removed.

---

## 2. Current validated metrics

These come directly from `validation_results.json` files in `outputs/`. Where a
stage has no current measurement, it is marked **not yet validated** rather than
quoting old numbers.

### Stage 1 — Depth (SGBM, object split, **1 sample: frame 000000**)
`outputs/depth/object/sgbm/validation_results.json`

| Metric | Value |
|---|---|
| EPE (end-point error) | **4.33 px** |
| D1 (outlier rate) | **19.83 %** |
| Coverage (valid GT px) | **36.57 %** |

Single-frame only — indicative, not a generalization estimate. **WAFT depth has no
GT validation on record** (`outputs/depth/object/waft/validation_results.json` does
not exist).

### Stage 2 — 2D detection (RT-DETR, object split, **3 samples**, IoU≥0.5)
`outputs/detections/object/validation_results.json`

| Class | AP |
|---|---|
| Car | **1.000** |
| Pedestrian | **1.000** |
| **mAP** | **1.000** |

AP is 1.0 over only **3 frames** — this confirms the detector runs and maps COCO→KITTI
correctly, but is **not** a meaningful accuracy estimate at this sample size.

### Stage 3 — 3D lifting: **not yet validated under the current pipeline**
The `validation_results.json` files under `outputs/boxes3d/{sgbm,waft}/000{0,1,2}/`
and the per-frame `*_boxes3d.json` boxes are **stale**: they were produced *before*
the position-only refactor. They still contain `l/w/h/heading` and `mean_heading_err`/
`mean_iou3d`, and were matched by **2D-IoU**, whereas the current code emits
position-only boxes and matches by **3D centre distance**. Their aggregate numbers
therefore do **not** describe the current code and are **not quoted here**. A re-run
of `validate_stage3_lift.py` is pending. (The one robust, matching-independent signal
from those old runs — WAFT depth failing to generalize across sequences — is recorded
as a known issue in §4.)

### Stage 4 — V2V fusion: **not yet validated** (CARLA not wired)
The fusion **core** (`utils/fusion.py`) is implemented and unit-tested, but it cannot
run end-to-end: `utils/carla_loader.py` and `stages/validate_stage4_fusion.py` are
stubs that raise `NotImplementedError`. The files under `outputs/fusion/sgbm/0000/`
are leftovers from the **retired KITTI temporal-simulation** backend and are **not**
produced by the current code — disregard them.

---

## 3. Stage-by-stage: what it does, how, and status

### Stage 1 — Depth (`stages/stage1_depth.py`)
Computes a disparity map from a rectified stereo pair, converts to metric depth via
`Z = f·B/d`, saves `.npy` + a colorized `.png`.
- **SGBM** — OpenCV `StereoSGBM` (params in `config/stage1.yaml`); invalid pixels → NaN.
- **WAFT** — a learned deep-stereo net, loaded from pre-computed `.npy` (Colab); no
  in-repo inference.
- **Status:** ✅ SGBM works and is the trusted method. ⚠️ WAFT output is suspect (see §4).

### Stage 2 — Detection (`stages/stage2_detect.py`)
Runs pretrained **RT-DETR** on the left image, keeps COCO classes mapped to KITTI
(`car→Car`, `person→Pedestrian`), thresholds by confidence. Outputs 2D boxes
`{label, confidence, x1, y1, x2, y2}`.
- **Status:** ✅ Working. Accuracy validated on only 3 frames (mAP 1.0 — see caveat above).

### Stage 3 — Lift to 3D position (`stages/stage3_lift.py`)
For each 2D box: samples disparity inside the box (per-method percentile, with an
optional top-crop + min-depth gate to reject road/ground pixels), converts to depth,
and **unprojects the box centre** to a 3D point `(x, y, z)`. Confidence is propagated
as `conf_2d × coverage_ratio`. Output carries the source 2D box for matching.
Validation (`validate_stage3_lift.py`) matches predictions↔GT by **3D centre distance**
per class (`matching.max_dist`: Car 2.0 m, Ped 1.0 m) and reports `mean_depth_err` +
`mean_center_dist`.
- **Status:** ⚙️ Code current and unit-tested; **end-to-end metrics not yet re-run**
  since the simplification.

### Stage 4 — V2V fusion (`stages/stage4_fusion.py` + `utils/fusion.py`)
The **source-agnostic core** registers Vehicle B's boxes into Vehicle A's frame via a
4×4 transform, greedily matches by BEV centre distance per class, and merges
corroborated pairs (noisy-OR confidence, confidence-weighted centre; size/heading
merged **only if present** — the core handles both position-only and full-3D boxes).
The CARLA backend (`run_carla`) is the data plumbing around it.
- **Status:** ⚙️ Core implemented + unit-tested (26 tests). 🔲 **Cannot run end-to-end** —
  CARLA loader and Stage 4 validation are stubs pending a real CARLA export.

---

## 4. Known issues

1. **WAFT depth is suspect.** It reports ~100% coverage and an implausibly narrow
   depth range regardless of scene; likely the wrong output tensor is read in
   `scripts/precompute_waft_disparity.py`. Pre-refactor Stage 3 runs showed WAFT depth
   error ~1.3 m on tracking seq 0000 but ~15–16 m on seqs 0001/0002 — i.e. it does not
   generalize across scenes. **SGBM is the trusted depth method downstream.**
2. **Stage 3 metrics are stale on disk.** All `outputs/boxes3d/**` results predate the
   position-only refactor (old schema, old 2D-IoU matching). They must be regenerated
   before any Stage 3 numbers are cited.
3. **CARLA not wired.** `utils/carla_loader.py` (load pair/frame/calib) and
   `stages/validate_stage4_fusion.py` raise `NotImplementedError`. Stage 4 has no
   end-to-end run or validation until a CARLA export exists. The verified CARLA
   coordinate conventions are documented in the loader's docstring, ready to implement.
4. **Heading/orientation is intentionally out of scope.** Geometry from stereo cannot
   recover per-object heading at range (ray-angle assumes the object faces the camera
   ray → large error; pseudo-LiDAR PCA locks onto depth noise; a learned head only
   reached a ~69° road-aligned prior). Stage 3 reports position only by design.
5. **Stale leftover outputs.** `outputs/fusion/sgbm/0000/` is from the retired
   KITTI-temporal backend; not regenerated by current code.

---

## 5. Test health

Full suite: **154 tests, all passing** (`stereo_v2v_env`):

| File | Count |
|---|---|
| `test_loader.py` | 36 |
| `test_stage1.py` | 26 |
| `test_stage2.py` | 33 |
| `test_stage3.py` | 33 |
| `test_stage4.py` | 26 |

Stage 4 includes coverage for **both** box schemas through the fusion core
(position-only and full 3D).

---

## 6. Done · pending · honest one-liners

| Stage | Status | One line |
|---|---|---|
| 1 Depth | ✅ working (SGBM) | SGBM solid (EPE 4.33 px on 1 frame); WAFT output suspect, untrusted. |
| 2 Detect | ✅ working | RT-DETR runs and maps to KITTI; accuracy only spot-checked on 3 frames. |
| 3 Lift | ⚙️ code current, metrics pending | Emits 3D position + 2D box; on-disk metrics are stale, re-run needed. |
| 4 Fusion | ⚙️ core done, 🔲 e2e blocked | Source-agnostic fusion unit-tested; cannot run until CARLA is wired. |

### Takeaway
> The 4-stage scaffold is in place and unit-tested (154 tests green). Stage 1 (SGBM)
> and Stage 2 run and have spot-check validation; **WAFT depth is untrusted**. Stage 3
> now produces honest stereo-recoverable output (3D position + 2D box) but its
> end-to-end metrics need re-running after the simplification. Stage 4's fusion core is
> complete and schema-flexible, but **the V2V result cannot be demonstrated until CARLA
> data is wired in** — that is the single biggest gap between the current state and the
> project goal.
