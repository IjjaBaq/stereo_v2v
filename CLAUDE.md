# Stereo V2V — Project Rules

## Code Style
- Functional, type-hinted, Google-style docstrings
- No hardcoded paths — everything via config or CLI
- Modular scripts, avoid monolithic classes
- `logging` module only (not print)
- `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` at every entry point
- `basicConfig` only in `__main__` blocks

## Pipeline Rules
- Each stage has standalone validation before next stage begins
- MLflow logs all params and metrics — never just print
- Per-stage config files (stage1.yaml … stage4.yaml)
- Output paths: `outputs/{stage_name}/`
- Never modify `data/` — read only
- Tests in `tests/`, one file per stage, pytest
- Shared config loading: `utils.config.load_configs` (do not re-copy per stage)

## Class scope (Car-only, 2026-06-10)
- The pipeline detects/evaluates **Car only**. Pedestrian was dropped everywhere:
  no `person → Pedestrian` in stage2.yaml; no `Pedestrian` threshold in
  stage3/stage4 `matching.max_dist`; `KITTI_CLASSES = ("Car",)` in all stages and
  validators. Rationale: CARLA GT is all `Car`, and stereo pedestrian
  detection/lifting was unreliable. COCO `truck`/`bus` still map to `Car`.
- Exception: `utils/kitti_tracking_loader.KITTI_CLASSES = ("Car","Van","Truck")`
  keeps car-like KITTI GT available (Van/Truck are still filtered out by the
  Car-only Stage-3 validator unless explicitly remapped). To re-enable
  Pedestrian, restore the stage2 mapping + stage3/stage4 thresholds + the
  KITTI_CLASSES constants.

## Data Sources (two)
- **KITTI** → real-image stereo for Stages 1-3 (depth → detect → lift). Only
  non-synthetic camera data. Stage 3 lifts each 2D detection to a 3D position
  (x, y, z) + carries the source 2D box — no size/heading (not recoverable
  from stereo at range). Validation matches preds↔GT by 3D center distance.
- **CARLA** → full pipeline incl. Stage 4 V2V fusion (true simultaneous
  multi-agent; can also produce stereo for Stages 1-3). Data is wired in at
  `data/carla` (Town10HD intersection, 300 frames, two moving ego vehicles);
  the CARLA loader (`utils/carla_loader.py`) is fully implemented. Per-agent GT
  is filtered by **true visibility**: a car counts as seen by an agent only if
  `gt_boxes` `metrics_metadata.visible_pixels_v{A,B}` >= `carla.min_visible_pixels`
  (stage4.yaml, default 10) — occlusion-truthful, not a geometric FOV guess.

## KITTI Conventions
- Calib keys: P2, P3, R_rect_00, Tr_velo_to_cam
  (file uses R0_rect — aliased in load_calib)
- Image suffix: _10.png for stereo split, .png for object split
- GT y = bottom of object (not center) — subtract h/2 for center
- Disparity GT: uint16 / 256.0 → float32, raw==0 → np.nan

- Depth sampling inside 2D boxes (per-method, tuned in
  experiments/percentile_choice.md): SGBM → percentile_20, WAFT → percentile_35.
  SGBM is sparse (consistency check already drops background), so its valid
  pixels sit on the car's near surface — a LOW percentile best matches box-centre
  depth; a high percentile under-shoots. WAFT is dense (top-40% crop + 6m gate
  first), so a mid-low percentile is best. (Supersedes the old percentile_75
  rule, which over-corrected for background SGBM had already removed.)

## CLI Conventions
- `--method` for depth method: `sgbm` | `waft` (Stages 1-3)
- `--sample_id` / `--sample_ids` for object split
- `--seq_id` + `--frame_ids` for tracking split
- Stage 4 is CARLA-only: `--scenario` + `--timestamp` (+ optional `--agent_a/--agent_b`)

## Output Structure
- `outputs/depth/object/{method}/`       Stage 1 object split
- `outputs/depth/tracking/{method}/{seq_id}/`  Stage 1 tracking
- `outputs/detections/object/`           Stage 2 object split
- `outputs/detections/tracking/{seq_id}/` Stage 2 tracking
- `outputs/lift3d/{method}/{seq_id}/`   Stage 3 tracking only
- `outputs/fusion/carla/{method}/`       Stage 4 CARLA fusion

## WAFT Notes
- Run `--method waft` from the project root so WAFT-Stereo imports
  (`algorithms`, `bridgedepth`, `peft`) resolve correctly.
- WAFT runs inference directly (no offline precompute): ~85s/image on
  CPU, ~1-2s on GPU. Model is loaded once and cached across samples.

## Known Issues
- Stage 3 only evaluated on tracking split — object split has misaligned
  stereo and detection frames (different scenes)
- Stage 4 is wired and validated on CARLA (`data/carla`): `carla_loader.py`,
  `stage4_fusion.py` (detector path), and `validate_stage4_fusion.py` all run
  end-to-end; the fusion core (`utils/fusion.py`) is complete and unit-tested.
  The validator scores cooperation **symmetrically** — both agents gain:
  `recall_improvement_a`/`b_unique_tp` (A's gain from B) and
  `recall_improvement_b`/`a_unique_tp` (B's gain from A). Latest SGBM run (20
  frames) shows A recall 0.21→0.54 (+0.33, 23 B-unique TPs) and B recall
  0.38→0.54 (+0.16). `a_unique_tp` is newly added and needs a re-run to populate
  (the 2026-06-10 JSON predates it) — see VALIDATION_SUMMARY.md.
