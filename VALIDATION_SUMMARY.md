# Validation Summary

Validated results from the end-to-end pipeline. Stages 1–3 are from the
regeneration on **2026-06-06**; **Stage 4 (V2V fusion) is the 150-frame CARLA run
(2026-06-16)** with the ego-from-BEV fix, the corroboration-rate metric, and the
PR-sweep / loc-error-vs-range figures. Numbers below are taken verbatim from the
per-stage
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

## Stage 4 — V2V Cooperative Fusion (CARLA, 150 frames)

Source: `outputs/fusion/carla/{sgbm,waft}/validation_results.json`
(`--scenario data/carla`, 150 frames). Scenario: Town10HD intersection, two moving
ego vehicles forming a **leader–follower** pair — A is close to the shared cars
(median ~11 m), B is ~3.3× further behind (median ~36 m). Egos are tagged `is_ego`
and excluded from coop-GT; an agent's detection of the *other* ego is an ignore
region (neither TP nor FP). Coop-GT = **9 distinct cars** / **586 instances**
(`n_coop_gt_distinct` / `n_coop_gt_instances`).

Each frame compares three prediction sets — **A-alone**, **B-alone** (B's Stages
1–3 output registered into A's frame), **Fused** — against a **cooperative GT**:
every (non-ego) vehicle visible to A *or* B, deduplicated by `actor_id`. Matching
is greedy BEV (x-z) centre distance per class, `matching.max_dist` (Car ≤ 2.0 m).

Cooperation is **mutual** — the fused output is shared, so both gains are measured
against the same coop-GT: **A's gain from B** (`b_unique_tp` = GT only B saw) and
**B's gain from A** (`a_unique_tp` = GT only A saw).

### Headline — recall per set
| Method | Recall A-alone | Recall B-alone | **Recall Fused** |
|--------|:--------------:|:--------------:|:----------------:|
| **SGBM** | 0.66 | 0.32 | **0.77** |
| WAFT | 0.11 | 0.21 | **0.31** |

The big result is **coverage**: the follower's recall rises **0.32 → 0.77**,
recovering **185** cars it could not perceive alone; the leader gains +0.11.

### Cooperative gains (SGBM, fused vs each single-agent baseline)
| Perspective | Δ recall | Δ loc-err (m) ↓ | Unique TPs recovered |
|-------------|:--------:|:---------------:|:--------------------:|
| **A gains from B** | **+0.11** | +0.02 | **40** (`b_unique_tp`) |
| **B gains from A** | **+0.44** | +0.06 | **185** (`a_unique_tp`) |

**Localization is essentially flat** (see per-method table). A diagnostic over the
150 frames shows the two agents view shared cars at a **median ~12° angular
separation** (never > 20°): their depth-uncertainty axes are near-parallel, so
cross-view triangulation — the mechanism that would tighten depth — is largely
unavailable. The **corroboration rate** is **0.26** (72 merges of 274 co-observed
cars): most co-observed cars are not actually merged (one agent misses, or the two
detections fall > 2 m apart), so the fused scene is closer to a union than a
deduplicated set. This is a clean, quantified **operating limit**, not a bug.

### Full per-method breakdown (TP / FP / FN · precision · BEV loc-err)
| Method | Set | TP | FP | FN | Precision | Loc-err (m) ↓ |
|--------|-----|:--:|:--:|:--:|:---------:|:-------------:|
| SGBM | A-alone | 384 | 6 | 202 | 0.985 | 0.785 |
| SGBM | B-alone | 190 | 63 | 396 | 0.751 | 0.819 |
| SGBM | **Fused** | 449 | 122 | 137 | 0.786 | 0.763 |
| WAFT | A-alone | 63 | 330 | 523 | 0.160 | 1.091 |
| WAFT | B-alone | 122 | 174 | 464 | 0.412 | 0.900 |
| WAFT | **Fused** | 184 | 504 | 402 | 0.267 | 0.962 |

The precision/recall trade-off of late fusion: A-alone is very precise (0.985) but
limited in reach; fusion unions in B's detections (and B's FPs), so fused precision
settles at 0.786. The **PR sweep** (`pr_curve_{method}.png`) traces this — the
fused curve reaches recall (~0.74) neither agent attains alone, and trades up to
precision ~0.95 at lower recall.

### GT-depth-range breakdown (fused TP count · loc-err m) — SGBM
Localization error grows with range (the central Ch1 claim); fusion is at or below
both agents at 10–20 m. Plotted in `loc_error_vs_range_sgbm.png`.

| Range | A-alone | B-alone | Fused |
|-------|:-------:|:-------:|:-----:|
| 0–10 m  | 0.67 (94) | 0.51 (41) | 0.57 (94) |
| 10–20 m | 0.81 (246) | 0.82 (84) | 0.75 (246) |
| 20–40 m | 0.87 (44) | 0.97 (29) | 0.91 (73) |

### Why SGBM beats WAFT
WAFT's dense depth lifts *every* detection → far more FPs (fused precision 0.27)
and it essentially never corroborates (**1 merge in 150 frames**), whereas SGBM's
sparse, consistency-checked depth filters weak detections (fused recall 0.77 vs
0.31). SGBM is also far faster on CPU. **SGBM is the method to cite for Stage 4.**

> Caveats: 150 frames but **one scenario** (Town10HD intersection), a
> leader–follower pair, 9 Car-only coop-GT vehicles (0–40 m) — a characterised
> feasibility study, not a multi-scenario benchmark. The localization limit is
> structural (~12° agent separation precludes triangulation); a perpendicular
> approach would be needed to test the triangulation regime. The fusion core
> (`utils/fusion.py`) is schema-agnostic; this run uses the detector path.

---

## Output file inventory
- Stage 1: 10 `*_disp.npy` + 10 `*_val.png` per method, `validation_results.json` ×2
- Stage 2: 10 `*_boxes2d.json` + 10 `*_det.png`, `validation_results.json`
- Stage 3: 24 `*_lift3d.json` + 24 `*_2d.png` + 24 `*_bev.png` per method,
  `validation_results.json` ×5 per method (one per sequence)
- Stage 4: per method (`carla/{sgbm,waft}/`) per-frame `carla_*_bev_a.png` /
  `carla_*_bev_b.png` (ego detections hidden; fused TP cyan / FP red),
  `loc_error_vs_range_{method}.png`, `pr_curve_{method}.png`, and
  `validation_results.json` (now carries `corroboration_rate`, `n_co_observed`,
  `pr_sweep`); per-agent `carla/` subtrees under `depth/`, `detections/`,
  `lift3d/` (each with `_val.png` / `_det.png` / `_bev.png` + `_2d.png` for both
  `vehicle_a` and `vehicle_b`).

See `outputs/README.md` for the full directory structure and file formats.
