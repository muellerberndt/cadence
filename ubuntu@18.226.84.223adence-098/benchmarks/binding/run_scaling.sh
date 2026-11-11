#!/usr/bin/env bash
# The scaling evaluation of the recall memory (SCALING_PROTOCOL.md) on one box.
#
# From the repository root, with Cadence's dependencies installed for `python`:
#   bash benchmarks/binding/run_scaling.sh OUT [WORKERS] [GRID]
# OUT is a directory outside the repository; WORKERS defaults to the box's cores, GRID to
# `trimmed`. The run resumes: a second call skips every life whose receipt is in OUT/lives.
# Another grid (`supplement`, amendment 1) runs into the same OUT/lives with a job list and a
# log of its own (OUT/jobs-GRID.json, OUT/run-GRID.log); the summary reads every receipt.
# With PROBE=1 the box first measures the grid's cost classes and the plan prints its own
# estimate before the lives start.
set -euo pipefail

OUT=${1:?usage: run_scaling.sh OUT [WORKERS] [GRID]}
WORKERS=${2:-$(getconf _NPROCESSORS_ONLN)}
GRID=${3:-trimmed}
PYTHON=${PYTHON:-python}

export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CADENCE_FUSED=0

TAG=$([ "$GRID" = trimmed ] && echo "" || echo "-$GRID")  # the declared grid keeps its names
mkdir -p "$OUT"
COSTS=()
if [ "${PROBE:-0}" = 1 ]; then
  [ -f "$OUT/costs$TAG.json" ] || "$PYTHON" benchmarks/binding/scaling.py probe \
    --grid "$GRID" --out "$OUT/costs$TAG.json" --workers "$WORKERS"
  COSTS=(--costs "$OUT/costs$TAG.json")
fi
[ -f "$OUT/jobs$TAG.json" ] || "$PYTHON" benchmarks/binding/scaling.py plan --grid "$GRID" \
  --out "$OUT/jobs$TAG.json" ${COSTS[@]+"${COSTS[@]}"}
"$PYTHON" benchmarks/binding/scaling.py run --jobs "$OUT/jobs$TAG.json" --out "$OUT" \
  --workers "$WORKERS" 2>&1 | tee -a "$OUT/run$TAG.log" || echo "lives failed: see $OUT/errors"
"$PYTHON" benchmarks/binding/scaling.py summary --out "$OUT" > /dev/null
echo "summary: $OUT/summary.md"
