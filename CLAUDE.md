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

## Data Sources (two)
- **KITTI** → real-image stereo for Stages 1-3 (depth → detect → lift). Only
  non-synthetic camera data.
- **CARLA** → full pipeline incl. Stage 4 V2V fusion (true simultaneous
  multi-agent; can also produce stereo for Stages 1-3). The CARLA loader
  (`utils/carla_loader.py`) is a documented STUB until a real export is wired in.

## KITTI Conventions
- Calib keys: P2, P3, R_rect_00, Tr_velo_to_cam
  (file uses R0_rect — aliased in load_calib)
- Image suffix: _10.png for stereo split, .png for object split
- GT y = bottom of object (not center) — subtract h/2 for center
- Disparity GT: uint16 / 256.0 → float32, raw==0 → np.nan

- Depth sampling inside 2D boxes: use percentile_75 not median
  (background pixels have lower disparity than foreground objects)

## CLI Conventions
- `--method` for depth method: `sgbm` | `waft` (Stages 1-3)
- `--sample_id` / `--sample_ids` for object split
- `--seq_id` + `--frame_ids` for tracking split
- No `--tag`, no `--depth_method` — use `--method` everywhere
- Stage 4 is CARLA-only: `--scenario` + `--timestamp` (+ optional `--agent_a/--agent_b`)

## Output Structure
- `outputs/depth/object/{method}/`       Stage 1 object split
- `outputs/depth/tracking/{method}/{seq_id}/`  Stage 1 tracking
- `outputs/detections/object/`           Stage 2 object split
- `outputs/detections/tracking/{seq_id}/` Stage 2 tracking
- `outputs/boxes3d/{method}/{seq_id}/`   Stage 3 tracking only
- `outputs/fusion/carla/{scenario}/`     Stage 4 CARLA fusion

## Known Issues
- WAFT disparity output suspected incorrect — 100% coverage and
  depth range 2-8m regardless of scene. Likely wrong output tensor
  read in precompute_waft_disparity.py.
- Heading estimation (theta_ray) systematic ~90° error for side-on vehicles.
  Learned orientation head (utils/orientation.py) is EXPERIMENTAL / not
  validated end-to-end — Stage 3 defaults to ray_angle.
- Stage 3 only evaluated on tracking split — object split has misaligned
  stereo and detection frames (different scenes)
- Stage 4 (`utils/carla_loader.py`, `validate_stage4_fusion.py`) is stubbed
  pending a CARLA export; the fusion core (`utils/fusion.py`) is complete and
  unit-tested.
