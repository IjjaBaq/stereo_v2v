# Stereo V2V — Project Status Report
*As of 2026-06-16 · branch `main`*

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
- **CARLA** — source for Stage 4 (true simultaneous multi-agent V2V). **Wired
  in:** `data/carla` (Town10HD intersection, 300 frames, two moving ego
  vehicles); the loader, detector path, and Stage 4 validation all run
  end-to-end.

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

### Stage 1 — Depth (KITTI object split, **10 samples**, frames 000000–000009)
`outputs/depth/object/{sgbm,waft}/validation_results.json`

| Metric | SGBM | WAFT |
|---|---|---|
| EPE (end-point error) | 4.57 px | **0.87 px** |
| D1 (outlier rate) | 13.81 % | **2.34 %** |
| Coverage (valid GT px) | 37.0 % | **100 %** |

**WAFT is the trusted, more accurate depth method** (~5.2× lower EPE, ~5.9× lower
D1, full coverage). This supersedes the earlier "WAFT output is suspect" status —
that was a bug in the retired offline-precompute path, now fixed (WAFT runs
inference directly; see §3). SGBM remains a valid sparse baseline.

### Stage 2 — 2D detection (RT-DETR, object split, **10 samples**, IoU ≥ 0.5)
`outputs/detections/object/validation_results.json`

| Class | AP |
|---|---|
| Car | 0.860 |

Car AP **0.860** is the headline under the Car-only pipeline (= mAP with a single
class). The earlier mAP of 0.930 averaged in a Pedestrian AP of 1.000; Pedestrian
was dropped 2026-06-10, so the recorded `validation_results.json` predates Car-only
and Stage 2 reports Car-only on the next run. Confirms the detector runs and maps
COCO→KITTI correctly. 10 frames is still a small evaluation set — indicative, not a
benchmark accuracy.

### Stage 3 — 3D lifting (position-only)
The end-to-end **matching** metrics (TP/FP/FN, centre distance) are regenerated at
the tuned percentiles — the `outputs/lift3d/**/validation_results.json` files carry
`depth_sampling = percentile_20` (SGBM) / `percentile_35` (WAFT) and the
per-sequence numbers are in VALIDATION_SUMMARY.md (Stage 3). Still pending are the
**per-class** and **depth-range breakdown** blocks: the validator now emits them,
but the on-disk JSONs predate that code and will populate on the next run.

The controlled, isolated analysis is the **depth-sampling accuracy study**
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

- **Status:** Code current and unit-tested; depth-sampling percentiles re-tuned
  (2026-06-05) and the matching JSONs regenerated at them; the per-class /
  depth-range breakdown blocks will populate on the next run.

### Stage 4 — V2V fusion (CARLA, **150 frames**, cooperative GT)
`outputs/fusion/carla/{sgbm,waft}/validation_results.json`

Each frame scores three prediction sets — **A-alone**, **B-alone** (registered
into A's frame), **Fused** — against a *cooperative* GT (every **non-ego** vehicle
visible to A *or* B, deduped by `actor_id`). Coop-GT is **9 distinct cars** across
the 150 frames = **586 instances** (`n_coop_gt_distinct` / `n_coop_gt_instances`).
The two agents form a **leader–follower** pair: A is close to the shared cars
(median ~11 m), B is ~3.3× further behind (median ~36 m). Matching is greedy BEV
centre distance per class. The fused output is **shared**, so both agents' gains
are reported.

| Metric | A-alone | B-alone | **Fused** |
|---|---|---|---|
| Recall — SGBM | 0.66 | 0.32 | **0.77** |
| Recall — WAFT | 0.11 | 0.21 | **0.31** |
| Precision — SGBM | 0.98 | 0.75 | 0.79 |
| Precision — WAFT | 0.16 | 0.41 | 0.27 |
| Loc-err (m) — SGBM | 0.78 | 0.82 | **0.76** |
| Loc-err (m) — WAFT | 1.09 | 0.90 | 0.96 |

**Cooperative gains** (fused vs each agent's own single-agent baseline, SGBM):

| Gain | value |
|---|---|
| **A gains from B** — Δrecall_a | **+0.11** (`b_unique_tp` = 40) |
| **B gains from A** — Δrecall_b | **+0.44** (`a_unique_tp` = 185) |
| Loc-err improvement (A / B) | +0.02 m / +0.06 m |
| Corroboration rate (merges / co-observed cars) | **72 / 274 = 0.26** |

**The headline V2V result is coverage, not localization.** Cooperation
substantially extends each vehicle's reach — the follower's recall rises
**0.32 → 0.77 (+0.44)**, recovering **185 vehicle-instances** it could not perceive
alone; the leader gains +0.11. **Localization barely changes** (SGBM 0.78/0.82 →
0.76): a diagnostic over the 150 frames shows the two agents view shared cars at a
**median ~12° angular separation** (never > 20°), so their depth-uncertainty axes
are near-parallel and cross-view triangulation — the mechanism that would tighten
depth — is largely unavailable. On the subset of cars seen by *both* agents,
averaging the two estimates does reduce error by ~0.15 m, but this dilutes
fleet-wide because most fused boxes are single-agent. This is a clean, quantified
**operating limit** of camera-only V2V in a leader–follower geometry, not a bug.

- **SGBM clearly beats WAFT** (WAFT is dense → no sparsity filter → far more FPs,
  precision 0.27; it also essentially never corroborates — 1 merge in 150 frames).
  SGBM is the method to cite for Stage 4.
- **Precision is the recall/precision trade-off of late fusion:** A-alone is very
  precise (0.98) but limited in reach; fusion unions in B's detections (and B's
  FPs), so fused precision settles at 0.79. The **PR sweep** (`pr_curve_sgbm.png`)
  shows the fused operating curve reaching recall (~0.74) that neither agent
  attains alone, trading up to precision ~0.95 at lower recall.
- **Caveats (indicative — single scenario):** 150 frames but one Town10HD
  intersection, 9 Car-only coop-GT vehicles, all within ~0–40 m. The result is a
  characterised feasibility study, not a multi-scenario benchmark.

---

## 3. Stage-by-stage: what it does, how, and status

### Stage 1 — Depth (`stages/stage1_depth.py`)
Computes a disparity map from a rectified stereo pair, converts to metric depth
via `Z = f·B/d`, saves `.npy` + a colorized `.png`.
- **SGBM** — OpenCV `StereoSGBM`; invalid pixels → NaN (sparse, ~37 % coverage).
- **WAFT** — a learned deep-stereo net (WAFT-Stereo) run **directly in-process**
  (`get_cfg` + `WAFT(cfg)` + checkpoint, PEFT `merge_and_unload`, module-level
  cached; CUDA if available else CPU, ~85 s/image on CPU). The old offline
  precompute path and its three bugs (÷255 input, positional call, wrong output
  tensor) are gone.
- **Status:** Both work. **WAFT is the trusted, more accurate method**; SGBM is
  the sparse baseline. Run `--method waft` from the project root so WAFT-Stereo
  imports resolve.

### Stage 2 — Detection (`stages/stage2_detect.py`)
Runs pretrained **RT-DETR** on the left image, keeps COCO classes mapped to KITTI
(`car/truck/bus→Car`; Car-only pipeline), thresholds by confidence. Outputs 2D
boxes `{label, confidence, x1, y1, x2, y2}`.
- **Status:** Working (Car AP 0.860 over 10 frames).

### Stage 3 — Lift to 3D position (`stages/stage3_lift.py`)
For each 2D box: samples a single disparity value inside the box (a per-method
**percentile**, with an optional top-crop + min-depth gate to reject road/ground),
converts to depth, and **unprojects the box centre** to a 3D point `(x, y, z)`.
Confidence is propagated as `conf_2d × coverage_ratio`. The source 2D box is
carried for matching. Validation (`validate_stage3_lift.py`) matches
predictions↔GT by **3D centre distance** per class (`matching.max_dist`: Car 2.0;
Car-only pipeline) and reports `mean_depth_err` + `mean_center_dist`.
- **Depth-sampling percentiles (tuned 2026-06-05, `experiments/percentile_choice.md`):**
  the two methods need **opposite** percentiles because their valid-pixel
  distributions differ. **SGBM = `percentile_20`** (sparse; the consistency check
  already drops background, so valid pixels sit on the car's near surface — a low
  percentile best matches centre depth; the old `percentile_75` over-corrected and
  was the worst end of the curve). **WAFT = `percentile_35`** (dense; after a
  top-40 % crop + 6 m gate, a mid-low percentile gives near-zero bias; the old
  `percentile_60` sampled the near surface).
- **Status:** Code current and unit-tested; matching JSONs are at the new
  percentiles; the per-class / depth-range breakdown blocks populate on the next run.

### Stage 4 — V2V fusion (`stages/stage4_fusion.py` + `utils/fusion.py`)
The **source-agnostic core** registers Vehicle B's boxes into Vehicle A's frame
via a 4×4 transform, greedily matches by BEV centre distance per class, and merges
corroborated pairs (noisy-OR confidence, confidence-weighted centre; size/heading
merged only if present — the core handles both position-only and full-3D boxes).
The CARLA backend (`run_carla`) is the data plumbing around it: it runs Stages
1-3 per agent (`detect_agent_boxes`), registers B into A's frame, and fuses.
- **Status:** Core implemented + unit-tested (31 tests) and **validated
  end-to-end on CARLA** (150 frames). The evaluation is **symmetric** — both
  agents' gains are scored (`recall_improvement_a`/`b_unique_tp` for A, and
  `recall_improvement_b`/`a_unique_tp` for B). Fusion lifts A's recall 0.66→0.77
  (SGBM, 40 B-unique objects) and B's recall 0.32→0.77 (185 A-unique objects);
  loc-error stays ~flat (0.78/0.82 → 0.76 m) — the gain is **coverage, not
  localization** (~12° agent separation precludes triangulation). See §2.

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
4. **Stage 4 is a single scenario.** 150 frames, but one Town10HD intersection
   with a leader–follower agent pair and 9 Car-only coop-GT vehicles (0–40 m) — a
   characterised feasibility study, not a multi-scenario benchmark. The
   localization gain is structurally limited by the ~12° angular separation
   between the two agents (near-collinear viewpoints preclude triangulation); a
   perpendicular-approach scenario would be needed to test the triangulation
   regime.
5. **Heading/orientation is intentionally out of scope.** Stereo cannot recover
   per-object heading at range (ray-angle assumes the object faces the camera ray;
   pseudo-LiDAR PCA locks onto depth noise; a learned head only reached a ~69°
   road-aligned prior). Stage 3 reports position only by design.

---

## 5. Test health

Full suite: **162 tests, all passing** (`stereo_v2v_env`):

| File | Count |
|---|---|
| `test_loader.py` | 36 |
| `test_stage1.py` | 29 |
| `test_stage2.py` | 33 |
| `test_stage3.py` | 33 |
| `test_stage4.py` | 31 |

Stage 4 covers **both** box schemas through the fusion core (position-only and
full 3D) and the **symmetric** cooperation metrics (`build_coop_gt`, `unique_tp`,
`score_against_gt` — A's and B's gains computed the same way).

---

## 6. Done · pending · honest one-liners

| Stage | Status | One line |
|---|---|---|
| 1 Depth | working | WAFT accurate (EPE 0.87 px, 10 frames) and trusted; SGBM is the sparse baseline. |
| 2 Detect | working | RT-DETR runs and maps to KITTI; Car AP 0.860 over 10 frames. |
| 3 Lift | code current, depth-sampling re-tuned | Emits 3D position + 2D box; percentiles tuned (SGBM p20, WAFT p35); per-class/depth-range breakdown pending. |
| 4 Fusion | validated on CARLA (150 frames) | Coverage gain: A recall 0.66→0.77 (40 B-unique TPs), B recall 0.32→0.77 (185 A-unique TPs) (SGBM); loc-err ~flat 0.78/0.82→0.76 m — localization limited by ~12° agent separation. |

### Takeaway
> The full 4-stage pipeline runs end-to-end and is unit-tested (162 tests green).
> Stage 1 (WAFT accurate and trusted, SGBM baseline) and Stage 2 run with
> spot-check validation. Stage 3 produces honest stereo-recoverable output (3D
> position + 2D box) at re-tuned per-method depth-sampling percentiles. **Stage 4
> — the project goal — is validated on 150 frames of CARLA V2V data:** cooperative
> fusion delivers a large **coverage** gain to both vehicles — the follower's
> recall 0.32→0.77 (recovering 185 objects only the leader saw) and the leader's
> 0.66→0.77 (40 objects only the follower saw). **Localization stays ~flat**
> (0.78/0.82→0.76 m): the two agents view shared cars at only ~12° separation, so
> triangulation — the mechanism that would tighten depth — is geometrically
> unavailable. This is a clean, quantified operating limit, and the central honest
> finding: camera-only V2V late fusion extends *coverage* strongly, while
> *localization* improvement is gated by inter-agent viewpoint geometry.
