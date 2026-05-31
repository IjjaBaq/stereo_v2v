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
- Per-stage config files (stage1.yaml, stage2.yaml, stage3.yaml)
- Output paths: `outputs/{stage_name}/`
- Never modify `data/` — read only
- Tests in `tests/`, one file per stage, pytest

## CLI Conventions
- `--method` for depth method: `sgbm` | `waft`
- `--sample_id` / `--sample_ids` for object split
- `--seq_id` + `--frame_ids` for tracking split
- No `--tag`, no `--depth_method` — use `--method` everywhere

## Output Structure
- `outputs/depth/object/{method}/`       Stage 1 object split
- `outputs/depth/tracking/{method}/{seq_id}/`  Stage 1 tracking
- `outputs/detections/object/`           Stage 2 object split
- `outputs/detections/tracking/{seq_id}/` Stage 2 tracking
- `outputs/boxes3d/{method}/{seq_id}/`   Stage 3 tracking only

## Known Issues
- WAFT disparity output suspected incorrect — 100% coverage and
  depth range 2-8m regardless of scene. Likely wrong output tensor
  read in precompute_waft_disparity.py. Investigate before Stage 4.
- Heading estimation (theta_ray) systematic ~90° error for side-on vehicles
- Stage 3 only evaluated on tracking split — object split has misaligned
  stereo and detection frames (different scenes)
