#!/usr/bin/env bash
#
# run_validation.sh — run the complete stereo_v2v validation pipeline from scratch.
#
# Usage:
#   bash scripts/run_validation.sh [--method sgbm|waft|both]   # default: both
#
# Runs every stage in order, across both data sources, each logging to MLflow:
#   Stage 1  Depth   (KITTI object split, samples 000000-000199)        per method
#   Stage 2  Detect  (KITTI object split, samples 000000-000199)        once
#   Stage 3  Lift    (KITTI tracking, all sequences, 15 frames/seq)     per method
#   Stage 4  Fusion  (CARLA data/carla, 150 frames evenly spread)       per method
#
# Behaviour:
#   - Sample/frame selection is read from data/ (not outputs/): the tracking
#     sequence list and per-sequence / CARLA frame counts are enumerated live.
#   - "From scratch" = clean metrics: before each stage the script removes only
#     that stage's validation_results.json SUMMARY file(s) so the reported numbers
#     reflect exactly this run's sample set (the validators merge results by ID; a
#     prior run with different IDs would otherwise pollute the summary). Heavy
#     cached artifacts (*_disp.npy, detection JSON, images) are NOT deleted — the
#     validators reuse them correctly by ID and regenerate anything missing.
#   - Continues on error: a failure in any step is recorded, never aborts the run;
#     all failures are reported in a summary at the end. Exit code is non-zero if
#     any step failed.
#
# Standard output locations (unchanged):
#   outputs/depth/object/{method}/        outputs/detections/object/
#   outputs/lift3d/{method}/{seq}/        outputs/fusion/carla/{method}/

# Note: intentionally NOT using `set -e` — we must continue past per-step failures.
set -o pipefail

# --------------------------------------------------------------------------
# Locate project root and Python interpreter
# --------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "ERROR: cannot cd to project root '$ROOT'"; exit 2; }


if [[ "$OS" == "Windows_NT" ]]; then
    PY="$ROOT/stereo_v2v_env/Scripts/python.exe"
elif [[ -x "stereo_v2v_env/bin/python" ]]; then
    PY="$ROOT/stereo_v2v_env/bin/python"
else
    PY="python3"
fi

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
METHOD_ARG="both"

usage() {
    cat <<EOF
Usage: bash scripts/run_validation.sh [--method sgbm|waft|both]

  --method   Depth method(s) to run for Stages 1, 3, 4 (default: both).
             Stage 2 (detection) is method-agnostic and always runs once.
  -h, --help Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            shift
            METHOD_ARG="${1:-}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'"
            usage
            exit 2
            ;;
    esac
    shift
done

case "$METHOD_ARG" in
    sgbm) METHODS=(sgbm) ;;
    waft) METHODS=(waft) ;;
    both) METHODS=(sgbm waft) ;;
    *)
        echo "ERROR: --method must be sgbm | waft | both (got '$METHOD_ARG')"
        exit 2
        ;;
esac

# --------------------------------------------------------------------------
# Selection parameters (read from data/)
# --------------------------------------------------------------------------
KITTI_OBJECT_DIR="data/kitti/object/training/image_2"
TRACKING_IMG_DIR="data/kitti/tracking/training/image_02"
CARLA_DIR="data/carla"

OBJ_LO=0
OBJ_HI=199                 # samples 000000-000199 (200)
TRACK_FRAMES_PER_SEQ=15    # frames per tracking sequence, evenly spread
CARLA_N_FRAMES=150         # frames evenly spread across available CARLA frames

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

# gen_range LO HI -> space-separated zero-padded 6-digit IDs in [LO, HI].
gen_range() {
    "$PY" - "$1" "$2" <<'PYEOF'
import sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
print(' '.join(f'{i:06d}' for i in range(lo, hi + 1)))
PYEOF
}

# even_spread N K WIDTH -> K indices evenly spread over [0, N-1] (deduped, sorted),
# zero-padded to WIDTH (WIDTH=0 -> plain integers). If N<=K, returns 0..N-1.
even_spread() {
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import sys
n, k, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if n <= 0:
    idx = []
elif n <= k:
    idx = list(range(n))
else:
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
fmt = (lambda x: format(x, f'0{w}d')) if w > 0 else str
print(' '.join(fmt(i) for i in idx))
PYEOF
}

print_stage() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

print_step() {
    echo ""
    echo "------------------------------------------------------------------"
    echo "  >> $1"
    echo "------------------------------------------------------------------"
}

PASSED=()
FAILED=()

# run_step LABEL CMD...  — run CMD, record pass/fail by exit code, never abort.
run_step() {
    local label="$1"; shift
    print_step "$label"
    if "$@"; then
        PASSED+=("$label")
        echo ">> [PASS] $label"
    else
        local rc=$?
        FAILED+=("$label")
        echo ">> [FAIL] $label (exit $rc)"
    fi
}

# --------------------------------------------------------------------------
# Pre-flight: data presence
# --------------------------------------------------------------------------
for d in "$KITTI_OBJECT_DIR" "$TRACKING_IMG_DIR" "$CARLA_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: required data directory not found: $d"
        echo "       Run this from a checkout with data/ populated."
        exit 2
    fi
done

# Tracking sequences (all available), sorted.
mapfile -t SEQS < <(ls -1 "$TRACKING_IMG_DIR" | sort)
N_SEQS=${#SEQS[@]}

# Available CARLA frames.
CARLA_AVAIL=$(ls -1 "$CARLA_DIR/gt_boxes" 2>/dev/null | wc -l | tr -d ' ')

# --------------------------------------------------------------------------
# Startup workload echo
# --------------------------------------------------------------------------
N_OBJ=$((OBJ_HI - OBJ_LO + 1))
N_METHODS=${#METHODS[@]}
S3_TOTAL=$((N_SEQS * TRACK_FRAMES_PER_SEQ * N_METHODS))

print_stage "stereo_v2v — full validation pipeline"
cat <<EOF
  Project root : $ROOT
  Python       : $PY
  Started      : $(date '+%Y-%m-%d %H:%M:%S')

  Depth method(s)        : ${METHODS[*]}
  MLflow tracking        : sqlite:///mlflow.db (per config/base.yaml; each stage logs automatically)

  Planned workload
  ----------------
  Stage 1  Depth  (KITTI) : ${N_OBJ} samples x ${N_METHODS} method(s) = $((N_OBJ * N_METHODS)) runs-worth of images
  Stage 2  Detect (KITTI) : ${N_OBJ} samples x 1 (method-agnostic)
  Stage 3  Lift   (KITTI) : ${N_SEQS} sequences x ${TRACK_FRAMES_PER_SEQ} frames x ${N_METHODS} method(s) = ${S3_TOTAL} frame-validations
  Stage 4  Fusion (CARLA) : ${CARLA_N_FRAMES} frames (of ${CARLA_AVAIL} available) x ${N_METHODS} method(s)

  Selection is read live from data/. Summary JSONs are cleared per stage for a
  clean run; cached per-frame artifacts are reused. Steps continue on error and
  are tallied at the end.
EOF

# --------------------------------------------------------------------------
# Stage 1 — Depth (KITTI object split)
# --------------------------------------------------------------------------
print_stage "STAGE 1 — Depth  (KITTI object split, samples $(printf '%06d' $OBJ_LO)-$(printf '%06d' $OBJ_HI))"
read -ra OBJ_SAMPLES <<< "$(gen_range "$OBJ_LO" "$OBJ_HI")"

for m in "${METHODS[@]}"; do
    rm -f "outputs/depth/object/$m/validation_results.json"
    run_step "Stage1-KITTI-depth-$m" \
        "$PY" stages/validate_stage1_depth.py \
        --sample_ids "${OBJ_SAMPLES[@]}" --method "$m"
done

# --------------------------------------------------------------------------
# Stage 2 — Detection (KITTI object split, method-agnostic)
# --------------------------------------------------------------------------
print_stage "STAGE 2 — Detection  (KITTI object split, samples $(printf '%06d' $OBJ_LO)-$(printf '%06d' $OBJ_HI))"
rm -f "outputs/detections/object/validation_results.json"
run_step "Stage2-KITTI-detect" \
    "$PY" stages/validate_stage2_detect.py \
    --sample_ids "${OBJ_SAMPLES[@]}"

# --------------------------------------------------------------------------
# Stage 3 — Lift to 3D (KITTI tracking, all sequences)
# --------------------------------------------------------------------------
print_stage "STAGE 3 — Lift  (KITTI tracking, ${N_SEQS} sequences x ${TRACK_FRAMES_PER_SEQ} frames)"
for m in "${METHODS[@]}"; do
    rm -f "outputs/lift3d/$m"/*/validation_results.json
    for seq in "${SEQS[@]}"; do
        n_frames=$(ls -1 "$TRACKING_IMG_DIR/$seq" 2>/dev/null | wc -l | tr -d ' ')
        read -ra FRAME_IDS <<< "$(even_spread "$n_frames" "$TRACK_FRAMES_PER_SEQ" 0)"
        if [[ ${#FRAME_IDS[@]} -eq 0 ]]; then
            echo ">> [SKIP] Stage3-KITTI-seq${seq}-${m} (no frames found)"
            FAILED+=("Stage3-KITTI-seq${seq}-${m} (no frames)")
            continue
        fi
        run_step "Stage3-KITTI-seq${seq}-${m} (${#FRAME_IDS[@]} frames of ${n_frames})" \
            "$PY" stages/validate_stage3_lift.py \
            --seq_id "$seq" --frame_ids "${FRAME_IDS[@]}" --method "$m"
    done
done

# --------------------------------------------------------------------------
# Stage 4 — V2V Fusion (CARLA)
# --------------------------------------------------------------------------
print_stage "STAGE 4 — Fusion  (CARLA, ${CARLA_N_FRAMES} of ${CARLA_AVAIL} frames)"
read -ra CARLA_TS <<< "$(even_spread "$CARLA_AVAIL" "$CARLA_N_FRAMES" 6)"
if [[ ${#CARLA_TS[@]} -eq 0 ]]; then
    echo ">> [SKIP] Stage4-CARLA (no CARLA frames found)"
    FAILED+=("Stage4-CARLA (no frames)")
else
    for m in "${METHODS[@]}"; do
        rm -f "outputs/fusion/carla/$m/validation_results.json"
        run_step "Stage4-CARLA-fusion-${m} (${#CARLA_TS[@]} frames)" \
            "$PY" stages/validate_stage4_fusion.py \
            --scenario "$CARLA_DIR" --timestamps "${CARLA_TS[@]}" --method "$m"
    done
fi

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
print_stage "VALIDATION SUMMARY"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  PASSED (${#PASSED[@]}):"
if [[ ${#PASSED[@]} -eq 0 ]]; then
    echo "    (none)"
else
    for s in "${PASSED[@]}"; do echo "    [PASS] $s"; done
fi
echo ""
echo "  FAILED (${#FAILED[@]}):"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "    (none)"
else
    for s in "${FAILED[@]}"; do echo "    [FAIL] $s"; done
fi
echo ""
echo "  Results: outputs/{depth,detections,lift3d,fusion}/...  |  MLflow: sqlite:///mlflow.db"
echo "=================================================================="

[[ ${#FAILED[@]} -eq 0 ]] && exit 0 || exit 1
