# Validation Summary

Validated results from the end-to-end pipeline. Stages 1–3 are from the
regeneration on **2026-06-06**; **Stage 4 (V2V fusion) was re-run on 2026-06-12**
after the CARLA pipeline fixes (camera-mount extrinsic, validator/visualization
fixes). Numbers below are taken verbatim from the per-stage
`validation_results.json` files — nothing was recalculated.

Seeds fixed at every entry point (`random`, `numpy`, `torch` = 42).

---

## Stage 1 — Stereo Depth (KITTI object split, 10 samples)

Source: `outputs/depth/object/{method}/validation_results.json`

| Method | Samples | EPE (px) ↓ | D1 (%) ↓ | Coverage (%) ↑ |
|--------|:------:|:----------:|:--------:|:--------------:|
| SGBM   | 10 | 4.569 | 13.81 | 37.0 |
| **WAFT** | 10 | **0.874** | **2.34** | **100.0** |

- **Improvement (WAFT vs SGBM):** EPE 4.569 → 0.874 px ≈ **5.2× lower**;
  D1 13.81% → 2.34% ≈ **5.9× lower**. WAFT also produces dense disparity
  (100% coverage) vs SGBM's sparse, consistency-checked 37.0%.
- **Samples evaluated:** 10 per method (`000000`–`000009`).
- **Runtime per image:** *not logged* as a pipeline metric **this run**.
  Observed wall-clock from screen timestamps: WAFT ≈ **80–85 s/image** (single
  process, model loaded once and cached across samples, CPU); SGBM ≈ **~1 s/image**
  of compute (each sample was a separate process that also incurred ~5 s
  Python/MLflow startup).

> **Pending next run:** the validator now records per-image `runtime_s` (MLflow,
> model-load excluded via pre-warm) and `mean_runtime_s` (summary JSON + MLflow).
> These will populate automatically on the next regeneration — no numbers yet.

### Per-sample EPE / D1 (WAFT)
| Sample | EPE (px) | D1 (%) | Coverage (%) |
|--------|:--------:|:------:|:------------:|
| 000000 | 0.608 | 1.65 | 100.0 |
| 000001 | 0.795 | 2.02 | 100.0 |
| 000002 | 0.868 | 2.59 | 100.0 |
| 000003 | 1.456 | 5.11 | 100.0 |
| 000004 | 0.744 | 2.09 | 100.0 |
| 000005 | 0.764 | 1.72 | 100.0 |
| 000006 | 1.055 | 2.46 | 100.0 |
| 000007 | 0.806 | 2.47 | 100.0 |
| 000008 | 0.813 | 1.71 | 100.0 |
| 000009 | 0.831 | 1.60 | 100.0 |

---

## Stage 2 — 2D Detection (RT-DETR, KITTI object split, 10 samples)

Source: `outputs/detections/object/validation_results.json` + MLflow params.
Model: `PekingU/rtdetr_r50vd` (42,891,372 params).

| Metric | Value |
|--------|:-----:|
| mAP @ IoU=0.5 | **0.9302** |
| AP — Car | 0.8604 |
| AP — Pedestrian | 1.0000 |
| IoU threshold | 0.5 |
| Confidence threshold | 0.5 (`config/stage2.yaml: model.confidence_threshold`) |
| Samples evaluated | 10 |
| `ap_reliability` flag | **`ok`** |

> **Caveat — predates the Car-only decision.** These numbers are verbatim from a
> `validation_results.json` produced with Pedestrian detection still enabled
> (`person → Pedestrian` in stage2.yaml). Pedestrian was dropped 2026-06-10
> (Car-only pipeline), so the Pedestrian AP (1.0000) and the two-class mAP (0.9302)
> will not appear on the next run — Car AP (0.8604) is the Car-only headline. Numbers
> are not silently changed here; they will refresh on the next full validation run.

- **AP method:** 11-point interpolated AP per class.
- **`ap_reliability` + sample-size caveat:** the validator sets this flag to
  `low` when fewer than 10 samples are evaluated, otherwise `ok`. This run used
  exactly **10 samples → `ok`**, i.e. the minimum threshold for the flag to clear.
  AP on 10 images is still a small-sample estimate and should be read as
  indicative, not definitive — the flag marks the floor, not statistical
  sufficiency.

### Per-class precision, recall, GT-vs-detected counts, false positives
**Not computed this run.** At the time of this run the Stage 2 validator emitted
only AP-per-class, mAP, the IoU threshold, and the raw per-sample
`pred_boxes` / `gt_boxes` lists. Per-class precision, recall, GT-object vs
detected counts, and false-positive counts were not produced or logged.

> **Pending next run:** the validator now computes, at the fixed confidence
> threshold (`config/stage2.yaml: model.confidence_threshold`), a per-class block
> — `precision`, `recall`, `n_tp`, `n_fp`, `n_fn`, `n_gt`, `n_detected` — plus
> `mean_inference_time_s`, all written to `validation_results.json` and MLflow.
> These will populate automatically on the next regeneration — no numbers yet.

---

## Stage 3 — Lift to 3D Positions (KITTI tracking, 5 sequences, 24 frames)

Source: `outputs/lift3d/{method}/{seq}/validation_results.json`.
Matching: greedy by 3D center distance, per class (Car ≤ 2.0 m; Car-only pipeline).
Depth-error and center-distance figures are TP-weighted means over matched pairs.

### Aggregate (all 5 sequences combined)
| Method | Frames | TP | FP | FN | Skipped | Precision | Recall | F1 | Depth err (m) ↓ | Center dist (m) ↓ |
|--------|:------:|:--:|:--:|:--:|:------:|:---------:|:------:|:----:|:---------------:|:-----------------:|
| **SGBM** | 24 | 59 | 93 | 78 | 21 | 0.388 | 0.431 | **0.408** | **0.880** | **1.022** |
| WAFT | 24 | 56 | 117 | 81 | 0 | 0.324 | 0.409 | 0.361 | 0.914 | 1.082 |

### Per-sequence detail (`depth_err m / center_dist m · TP/FP/FN`)
| Seq | Frames | SGBM | WAFT |
|-----|:------:|------|------|
| 0000 | 5 | 0.90 / 1.18 · 19/36/11 | 1.02 / 1.27 · 14/42/16 |
| 0001 | 5 | 0.88 / 0.96 · 27/27/31 | 1.03 / 1.17 · 28/30/30 |
| 0002 | 5 | 0.75 / 0.81 · 3/11/14  | 0.41 / 0.62 · 4/15/13 |
| 0003 | 5 | 0.93 / 1.12 · 7/7/9    | 0.65 / 0.85 · 6/14/10 |
| 0004 | 4 | 0.86 / 1.10 · 3/12/13  | 1.14 / 1.31 · 4/16/12 |

### Depth-sampling percentiles used
- **SGBM → `percentile_20`** (confirmed in each `validation_results.json`
  `depth_sampling` field).
- **WAFT → `percentile_35`**.
- Tuned per `experiments/percentile_choice.md`: SGBM's valid pixels are sparse
  and already background-free (consistency check), so a low percentile matches
  box-centre depth; WAFT is dense (top-crop + 6 m gate), so a mid-low percentile
  is best.

### Coverage rate (n_lifted / n_input) per method
Derived from the aggregate counts above (`n_lifted = TP + FP`,
`n_input = n_lifted + skipped`; the same Stage 2 detections feed both methods):

| Method | Lifted | Skipped | Input | Coverage |
|--------|:------:|:------:|:-----:|:--------:|
| SGBM | 152 | 21 | 173 | **87.9%** |
| WAFT | 173 | 0  | 173 | **100.0%** |

### Interpretation — why SGBM edges out WAFT on Stage-3 F1
Despite WAFT's far superior **raw depth** (Stage 1: 0.87 vs 4.57 px EPE), SGBM
wins on end-to-end localization F1 (0.408 vs 0.361) and on aggregate depth/center
error. Two compounding reasons:

1. **SGBM's sparse, consistency-checked pixels already sit on the car's near
   surface.** The left↔right consistency check drops background/occluded pixels,
   so the surviving disparities cluster on the nearest visible vehicle surface —
   a low percentile (p20) lands close to the true box-centre depth.
2. **WAFT generates more false positives because it skips nothing.** WAFT's dense
   depth means *every* detection gets lifted (**0 skips → 173 lifted, 117 FP**),
   whereas SGBM's sparsity causes **21 low-evidence detections to be skipped**
   (insufficient valid depth pixels → only 152 lifted, 93 FP). Those skips act as
   a free precision filter, removing weak/empty-frustum detections that WAFT
   keeps. Net effect: WAFT has higher coverage but lower precision, and SGBM's
   surface-biased sampling gives it the better localization error.

WAFT still wins cleanly where depth is the limiting factor — sequences **0002**
(0.41 / 0.62 m) and **0003** (0.65 / 0.85 m) — but loses the aggregate on the
FP-heavy sequences.

> Note: at this stage "position" means 3D centre `(x, y, z)` + the source 2D box
> only — no size or heading (not recoverable from stereo at range).

> **Pending next run:** the validator now emits a **per-class** block
> (`per_class.{Car}`: `n_tp`, `n_fp`, `n_fn`, `depth_err`,
> `center_dist`) and a **depth-range breakdown** (`depth_range_breakdown`: bins
> `0_10m` / `10_20m` / `20_40m` / `40m_plus`, each with `n`, `depth_err`,
> `center_dist`; empty bins → `null` and are not logged to MLflow). Errors are
> pooled over all TP pairs, not averaged from per-frame means. These will
> populate automatically on the next regeneration — no numbers yet.

---

## Stage 4 — V2V Cooperative Fusion (CARLA, 5 frames)

Source: `outputs/fusion/carla/{sgbm,waft}/validation_results.json` (run 2026-06-13,
`--scenario data/carla`, frames `000050/000100/000150/000200/000250`, the
default FOV-overlap set). Scenario: Town10HD intersection, two moving ego vehicles.
This run includes the 2026-06-12 fixes (most importantly the **camera-mount
extrinsic** fix — GT in each agent's left-camera frame, not the vehicle origin)
**plus the 2026-06-13 ego-exclusion fix**: the export listed both ego vehicles in
`gt_boxes`, and since an ego is invisible to its own camera but visible to the
other agent it leaked into the cooperative GT (one agent "detecting" the other's
car). Excluding the egos drops the coop-GT from 19 to **15 instances** and removes
the inflated unique-TP counts (e.g. SGBM `b_unique_tp` 6 → 3).

> **Note on counts.** "coop-GT objects" below are **per-frame instances summed
> over the 5 frames**, not distinct vehicles. Only **3 distinct cars** (actors
> 121, 124, 125) appear across these frames — 3 per frame × 5 = 15 instances. The
> validation JSON reports both as `n_coop_gt_distinct` (3) and
> `n_coop_gt_instances` (15). The scene has 29 vehicle actors but only 3–4 per
> frame pass the visibility filter; the rest are out of FOV / too far / occluded.

Each frame compares three prediction sets — **A-alone**, **B-alone** (B's Stages
1–3 output registered into A's frame), **Fused** — against a **cooperative GT**:
every (non-ego) vehicle visible to A *or* B, deduplicated by `actor_id` (A-visible
instance wins). Matching is greedy BEV (x-z) centre distance per class,
`matching.max_dist` from `config/stage4.yaml` (Car ≤ 2.0 m; Car-only pipeline).
All 15 coop-GT instances (3 distinct cars × 5 frames) are Cars.

Cooperation is **mutual** — the fused output is shared, so both agents benefit and
both gains are measured against the same cooperative GT: **A's gain from B**
(fused vs A-alone; `b_unique_tp` = GT only B saw) and **B's gain from A** (fused vs
B-alone; `a_unique_tp` = GT only A saw).

### Headline — recall per set
| Method | Recall A-alone | Recall B-alone | **Recall Fused** | Mean infer (s/frame) |
|--------|:--------------:|:--------------:|:----------------:|:--------------------:|
| **SGBM** | 0.8000 | 0.4667 | **1.0000** | 6.81 |
| WAFT | 0.3333 | 0.2000 | **0.5333** | 150.8 |

The fused set is shared, so `Recall Fused` is the common target both agents reach.
SGBM fusion recovers **all 15** coop-GT cars (recall 1.00, 0 FN) from single-agent
recalls of 0.80 / 0.47 — each agent recovers cars only the other saw.

### Symmetric V2V gains — both agents benefit
| Perspective | Baseline | Δ recall | Δ precision | Δ loc-err (m) ↓ | Unique TPs recovered |
|-------------|----------|:--------:|:-----------:|:---------------:|:--------------------:|
| **SGBM — A gains from B** | A-alone | **+0.2000** | −0.0682 | +0.0600 | **3** (`b_unique_tp`) |
| **SGBM — B gains from A** | B-alone | **+0.5333** | −0.1932 | +0.1640 | **4** (`a_unique_tp`) |
| **WAFT — A gains from B** | A-alone | **+0.2000** | −0.0278 | −0.0769 | **3** (`b_unique_tp`) |
| **WAFT — B gains from A** | B-alone | **+0.3333** | +0.0357 | +0.1287 | **2** (`a_unique_tp`) |

Δ recall / Δ precision = fused minus that agent's single-agent baseline; Δ loc-err
= baseline loc-err minus fused (positive = fusion tightens localization). Both
agents gain recall; the precision dip is each agent absorbing the *other's* genuine
false positives (the fused set pools both), and localization tightens in 3 of the 4
perspectives. Both `a_unique_tp` and `b_unique_tp` are captured.

**Ego detections are an ignore region.** Each agent's detector still detects the
*other ego* (e.g. B sees A ahead). The egos are not coop-GT targets (their poses
are shared over V2V), so such detections are scored as **ignored — neither TP nor
FP** (KITTI `DontCare` semantics), counted as `n_ignored` rather than penalizing
precision. This lifts B-alone precision from 0.636 to **0.875** (SGBM: 3 ego-A
detections ignored) without touching recall or FN.

### Full per-method breakdown (TP / FP / FN · ignored · precision · BEV loc-err)
| Method | Set | TP | FP | FN | Ign | Precision | Loc-err (m) ↓ |
|--------|-----|:--:|:--:|:--:|:---:|:---------:|:-------------:|
| SGBM | A-alone | 12 | 4 | 3 | 0 | 0.750 | 0.815 |
| SGBM | B-alone | 7 | 1 | 8 | 3 | 0.875 | 0.919 |
| SGBM | **Fused** | 15 | 7 | 0 | 3 | 0.682 | 0.755 |
| WAFT | A-alone | 5 | 13 | 10 | 0 | 0.278 | 0.678 |
| WAFT | B-alone | 3 | 11 | 12 | 2 | 0.214 | 0.884 |
| WAFT | **Fused** | 8 | 24 | 7 | 2 | 0.250 | 0.755 |

"Ign" = ego detections ignored (not counted TP/FP). A-alone has 0 (A can't see its
own ego and detects no NPC at an ego's location); B-alone has 3 (B detects ego-A).

- A's gain: `recall_improvement_a` = +0.2000 (SGBM) / +0.2000 (WAFT);
  `loc_error_improvement_a` = +0.060 m / −0.077 m.
- B's gain: `recall_improvement_b` = +0.5333 (SGBM) / +0.3333 (WAFT);
  `loc_error_improvement_b` = +0.164 m / +0.129 m.
- Fusion improves recall for **both** agents. Fused precision (SGBM 0.68) reflects
  the genuine false positives each agent contributes to the shared set; the
  other-ego detections no longer count against it.

### Per-class (Car · TP/FP/FN — fused)
Coop-GT is non-ego vehicles only and the pipeline is **Car-only** (Pedestrian
dropped 2026-06-10), so **Car** is the entire signal; fused FPs are genuine car
mis/duplicate detections (the other-ego detections are ignored, not FPs), not
pedestrian hallucinations.

| Method | Car fused TP | Car fused FP | Car fused FN | Car loc-err (m) |
|--------|:------------:|:------------:|:------------:|:---------------:|
| SGBM | 15 | 7 | 0 | 0.755 |
| WAFT | 8 | 24 | 7 | 0.755 |

### GT-depth-range breakdown (fused TP count · loc-err m)
All cooperative GT is within 0–20 m (close-range intersection); the 20–40 m and
40 m+ bins are empty.

| Range | SGBM fused (n · m) | WAFT fused (n · m) |
|-------|:------------------:|:------------------:|
| 0–10 m  | 3 · 1.384 | 2 · 1.233 |
| 10–20 m | 10 · 0.687 | 5 · 0.678 |

### Why SGBM beats WAFT (again)
Same mechanism as Stage 3: WAFT's dense depth lifts *every* detection (more FPs,
lower precision and recall after matching), while SGBM's sparse,
consistency-checked depth filters weak detections — here SGBM fused recall 1.00 vs
WAFT 0.53 at comparable localization. SGBM is also ~22× faster (6.81 vs 150.8
s/frame on CPU). **SGBM is the method to cite for Stage 4.**

> Caveats: 5 frames, one scenario, close range (all GT 0–20 m), vehicle-only GT —
> an indicative V2V demonstration, not a benchmark. Ego vehicles are excluded from
> coop-GT via a **temporary proximity filter** in `carla_loader._drop_ego_boxes`
> (a re-collection will tag them `is_ego` and retire the hack — see CLAUDE.md
> "Temporary code"). Per-agent GT visibility still uses the pre-fix binary
> `visible_pixels` on disk (the occlusion-truthful collector rewrite needs a CARLA
> re-collection to take effect), so partially visible cars may be under-counted.
> The fusion core (`utils/fusion.py`) is schema-agnostic (handles Stage-3
> position-only and full CARLA GT boxes); this run uses the detector path.

---

## Output file inventory
- Stage 1: 10 `*_disp.npy` + 10 `*_val.png` per method, `validation_results.json` ×2
- Stage 2: 10 `*_boxes2d.json` + 10 `*_det.png`, `validation_results.json`
- Stage 3: 24 `*_lift3d.json` + 24 `*_2d.png` + 24 `*_bev.png` per method,
  `validation_results.json` ×5 per method (one per sequence)
- Stage 4: per method (`carla/{sgbm,waft}/`) 5 `carla_*_bev.png` +
  `validation_results.json`; per-agent `carla/` subtrees under `depth/`,
  `detections/`, `lift3d/` (each with `_val.png` / `_det.png` / `_bev.png` +
  `_2d.png` for both `vehicle_a` and `vehicle_b`).

See `outputs/README.md` for the full directory structure and file formats.
