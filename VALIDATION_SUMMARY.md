# Validation Summary

Validated results from the end-to-end pipeline. Stages 1–3 are from the
regeneration on **2026-06-06**; **Stage 4 (V2V fusion) was run on 2026-06-10**
once CARLA data was wired in. Numbers below are taken verbatim from the per-stage
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
Matching: greedy by 3D center distance, per class (Car ≤ 2.0 m, Ped ≤ 1.0 m).
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
> (`per_class.{Car,Pedestrian}`: `n_tp`, `n_fp`, `n_fn`, `depth_err`,
> `center_dist`) and a **depth-range breakdown** (`depth_range_breakdown`: bins
> `0_10m` / `10_20m` / `20_40m` / `40m_plus`, each with `n`, `depth_err`,
> `center_dist`; empty bins → `null` and are not logged to MLflow). Errors are
> pooled over all TP pairs, not averaged from per-frame means. These will
> populate automatically on the next regeneration — no numbers yet.

---

## Stage 4 — V2V Cooperative Fusion (CARLA, 20 frames)

Source: `outputs/fusion/carla/{sgbm,waft}/validation_results.json` (run 2026-06-10,
`--scenario data/carla`, frames `000054`–`000282`, spread across the FOV-overlap
range). Scenario: Town10HD intersection, two moving ego vehicles.

Each frame compares three prediction sets — **A-alone**, **B-alone** (B's Stages
1–3 output registered into A's frame), **Fused** — against a **cooperative GT**:
every vehicle visible to A *or* B, deduplicated by `actor_id` (A-visible instance
wins). Matching is greedy BEV (x-z) centre distance per class, `matching.max_dist`
from `config/stage4.yaml` (Car ≤ 2.0 m, Ped ≤ 1.0 m). All 85 coop-GT objects
across the 20 frames are Cars.

Cooperation is **mutual** — the fused output is shared, so both agents benefit and
both gains are measured against the same cooperative GT: **A's gain from B**
(fused vs A-alone; `b_unique_tp` = GT only B saw) and **B's gain from A** (fused vs
B-alone; `a_unique_tp` = GT only A saw).

### Headline — recall per set
| Method | Recall A-alone | Recall B-alone | **Recall Fused** | Mean infer (s/frame) |
|--------|:--------------:|:--------------:|:----------------:|:--------------------:|
| **SGBM** | 0.2118 | 0.3765 | **0.5412** | 6.79 |
| WAFT | 0.0824 | 0.2941 | **0.3765** | 160.4 |

The fused set is shared, so `Recall Fused` is the common target both agents reach.
B-alone already out-recalls A-alone (Vehicle B sees more of this intersection), so
A gains more from cooperation than B does — but **both gain**.

### Symmetric V2V gains — both agents benefit
| Perspective | Baseline | Δ recall | Δ precision | Δ loc-err (m) ↓ | Unique TPs recovered |
|-------------|----------|:--------:|:-----------:|:---------------:|:--------------------:|
| **SGBM — A gains from B** | A-alone | **+0.3294** | +0.0766 | +0.0212 | **23** (`b_unique_tp`) |
| **SGBM — B gains from A** | B-alone | **+0.1647** | −0.1254 | −0.0098 | _pending re-run_ (`a_unique_tp`) |
| **WAFT — A gains from B** | A-alone | **+0.2941** | +0.0830 | +0.2941 | 15 (`b_unique_tp`) |
| **WAFT — B gains from A** | B-alone | **+0.0824** | −0.1040 | −0.0830 | _pending re-run_ (`a_unique_tp`) |

Δ recall / Δ precision = fused minus that agent's single-agent baseline; Δ loc-err
= baseline loc-err minus fused (positive = fusion tightens localization). Both
agents gain recall; each pays a precision cost (the other agent's false positives
enter the fused set). `a_unique_tp` (GT only A saw, recovered for B) requires
re-running `validate_stage4_fusion.py` — the symmetric counterpart to `b_unique_tp`
is now computed by the validator but was not captured in the 2026-06-10 run.

### Full per-method breakdown (TP / FP / FN · precision · BEV loc-err)
| Method | Set | TP | FP | FN | Precision | Loc-err (m) ↓ |
|--------|-----|:--:|:--:|:--:|:---------:|:-------------:|
| SGBM | A-alone | 18 | 102 | 67 | 0.150 | 1.326 |
| SGBM | B-alone | 32 | 59 | 53 | 0.352 | 1.295 |
| SGBM | **Fused** | 46 | 157 | 39 | 0.227 | 1.305 |
| WAFT | A-alone | 7 | 126 | 78 | 0.053 | 1.755 |
| WAFT | B-alone | 25 | 79 | 60 | 0.240 | 1.378 |
| WAFT | **Fused** | 32 | 204 | 53 | 0.136 | 1.461 |

- A's gain: `recall_improvement_a` = +0.3294 (SGBM) / +0.2941 (WAFT);
  `precision_change_a` = +0.0766 / +0.0830; `loc_error_improvement_a` =
  +0.0212 m / +0.2941 m.
- B's gain: `recall_improvement_b` = +0.1647 (SGBM) / +0.0824 (WAFT);
  `precision_change_b` = −0.1254 / −0.1040; `loc_error_improvement_b` =
  −0.0098 m / −0.0830 m.
- Fusion improves recall for **both** agents; the precision dip is each agent
  absorbing the other's false positives, and localization stays essentially flat.

### Per-class (Car · TP/FP/FN — fused)
Coop-GT is vehicles only, so **Car** is the entire real signal. **Pedestrian** has
**0 GT**, so every ped detection is a false positive (RT-DETR hallucinations) —
SGBM fused 86 ped FP, WAFT 92 — which is what drags overall precision down.

| Method | Car fused TP | Car fused FP | Car fused FN | Car loc-err (m) |
|--------|:------------:|:------------:|:------------:|:---------------:|
| SGBM | 46 | 71 | 39 | 1.305 |
| WAFT | 32 | 112 | 53 | 1.461 |

### GT-depth-range breakdown (fused TP count · loc-err m)
All cooperative GT is within 0–20 m (close-range intersection); the 20–40 m and
40 m+ bins are empty.

| Range | SGBM fused (n · m) | WAFT fused (n · m) |
|-------|:------------------:|:------------------:|
| 0–10 m  | 19 · 1.018 | 8 · 1.463 |
| 10–20 m | 16 · 1.469 | 15 · 1.548 |

### Why SGBM beats WAFT (again)
Same mechanism as Stage 3: WAFT's dense depth lifts *every* detection (more FPs,
lower precision and recall after matching), while SGBM's sparse,
consistency-checked depth filters weak detections. SGBM is also ~24× faster
(6.79 vs 160.4 s/frame on CPU). **SGBM is the method to cite for Stage 4.**

> Caveats: 20 frames, one scenario, close range, vehicle-only GT — an indicative
> V2V demonstration, not a benchmark. The fusion core (`utils/fusion.py`) is
> schema-agnostic (handles Stage-3 position-only and full CARLA GT boxes with
> `l/w/h/heading`); this run uses the detector path (Stages 1–3 per agent).

---

## Output file inventory
- Stage 1: 10 `*_disp.npy` + 10 `*_val.png` per method, `validation_results.json` ×2
- Stage 2: 10 `*_boxes2d.json` + 10 `*_det.png`, `validation_results.json`
- Stage 3: 24 `*_lift3d.json` + 24 `*_2d.png` + 24 `*_bev.png` per method,
  `validation_results.json` ×5 per method (one per sequence)
- Stage 4: per method (`carla/{sgbm,waft}/`) 20 `carla_*_bev.png` +
  `validation_results.json`; per-agent `carla/` subtrees under `depth/`,
  `detections/`, `lift3d/`.

See `outputs/README.md` for the full directory structure and file formats.
