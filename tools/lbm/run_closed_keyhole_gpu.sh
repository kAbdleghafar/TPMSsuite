#!/usr/bin/env bash
# Closed-keyhole production runs on a CUDA GPU (CuPy with the fused CUDA kernels for the flow,
# numba for the rays). Each run checkpoints every CKPT minutes (default 10). After an
# interruption, rerun the same command: finished cases are skipped and the interrupted one
# resumes from its last checkpoint. Results go to tools/lbm/keyhole_runs/; the closure figure is
# drawn at the end.
#
#   bash tools/lbm/run_closed_keyhole_gpu.sh                 # everything below, at DX (default 3.33 um)
#   DX=2.5e-6 bash tools/lbm/run_closed_keyhole_gpu.sh turn  # one group at 2.5 um
#   PYTHON=.venv/Scripts/python bash tools/lbm/run_closed_keyhole_gpu.sh   # a specific interpreter
#   bash tools/lbm/run_closed_keyhole_gpu.sh precision                     # is float32 good enough?
#   DTYPE=float32 bash tools/lbm/run_closed_keyhole_gpu.sh turn overhang   # then the rest in float32
#
# Groups:
#   off      Ti-6Al-4V T3 track, laser off abruptly at the end vs ramped down over 100 um
#   turn     Martin et al. (2019) turn points: M200 at constant power and with the
#            constant-keyhole-number power schedule first, then M100 and M300
#   overhang CoCrFeMnNi slab over powder at the gyroid settings: 90, 150, 250 um slabs
#   precision  timing breakdown (300 steps of M200, float64 and float32), then T3 abrupt and
#            ramped in float32, compared with the float64 runs (compare_precision.py): PASS means
#            the remaining runs can use DTYPE=float32 (float32 results go to *_f32* files)
set -euo pipefail
cd "$(dirname "$0")/../.."
export TPMSFEM_BACKEND=cupy
DX=${DX:-3.33e-6}
CKPT=${CKPT:-10}
DTYPE=${DTYPE:-float64}
PY=${PYTHON:-python}
SETS=("$@")
[ ${#SETS[@]} -eq 0 ] && SETS=(off turn overhang)
for g in "${SETS[@]}"; do
  case "$g" in
    off)
      "$PY" tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --checkpoint "$CKPT" --dtype "$DTYPE"
      "$PY" tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --scenario ramp --ramp 100e-6 --checkpoint "$CKPT" --dtype "$DTYPE"
      ;;
    turn)
      # 700 um plates: at 400 um the M200 keyhole reached the bottom wall 88 us after the turn
      "$PY" tools/lbm/keyhole_ti64.py M200 --dx "$DX" --scenario turn --plate 700e-6 --cool 200e-6 --checkpoint "$CKPT" --dtype "$DTYPE"
      "$PY" tools/lbm/keyhole_ti64.py M200 --dx "$DX" --scenario turn-ke --plate 700e-6 --cool 200e-6 --checkpoint "$CKPT" --dtype "$DTYPE"
      for c in M100 M300; do "$PY" tools/lbm/keyhole_ti64.py "$c" --dx "$DX" --scenario turn --plate 700e-6 --cool 200e-6 --checkpoint "$CKPT" --dtype "$DTYPE"; done
      ;;
    overhang)
      for t in 90e-6 150e-6 250e-6; do "$PY" tools/lbm/keyhole_overhang.py --slab "$t" --dx "$DX" --checkpoint "$CKPT" --dtype "$DTYPE"; done
      ;;
    precision)
      for d in float64 float32; do
        "$PY" tools/lbm/keyhole_ti64.py M200 --dx "$DX" --scenario turn --plate 700e-6 --dtype "$d" --profile 300
      done
      "$PY" tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --checkpoint "$CKPT" --dtype float32
      "$PY" tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --scenario ramp --ramp 100e-6 --checkpoint "$CKPT" --dtype float32
      "$PY" tools/lbm/compare_precision.py --all tools/lbm/keyhole_runs || true
      ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done
"$PY" tools/lbm/keyhole_figure.py closure tools/lbm/keyhole_runs/T3_dx*um.json tools/lbm/keyhole_runs/T3_ramp_dx*um.json \
  tools/lbm/keyhole_runs/M*_turn*_dx*um.json || true
