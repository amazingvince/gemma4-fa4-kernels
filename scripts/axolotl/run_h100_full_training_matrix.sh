#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ID=${AXOLOTL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
MATRIX_ROOT=${AXOLOTL_MATRIX_ROOT:-$ROOT/agent_space/axolotl-full-training-matrix/$RUN_ID}
LOCK_PATH=${H100_LOCK_PATH:-/workspace/.h100-codex.lock}
SEEDS=(1729 31415 65537)

mkdir -p "$MATRIX_ROOT"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "H100 lease is held by another task" >&2
  exit 75
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
  echo "H100 already has an active compute process" >&2
  exit 75
fi

if [[ -z ${GEMMA4_FA4_SOURCE_REVISION:-} ]]; then
  if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
    export GEMMA4_FA4_SOURCE_REVISION
    GEMMA4_FA4_SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
  else
    echo "GEMMA4_FA4_SOURCE_REVISION is required for a source tree without Git metadata" >&2
    exit 2
  fi
fi

for seed in "${SEEDS[@]}"; do
  AXOLOTL_SEED="$seed" \
  AXOLOTL_RUN_ROOT="$MATRIX_ROOT/seed-$seed" \
  AXOLOTL_DEFER_PAIR_GATE=1 \
  REMOTE_FA4_CACHE_DIR="$MATRIX_ROOT/seed-$seed/fa4-cache" \
    bash "$ROOT/scripts/axolotl/run_h100_full_training.sh"
done

VENV=${AXOLOTL_PROJECT_VENV_DIR:-/workspace/gemma4-fa4-kernels/.venv-h100-axolotl}
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" "$ROOT/scripts/compare_axolotl_training_matrix.py" \
  "$MATRIX_ROOT" --output "$MATRIX_ROOT/comparison.json"

echo "Three-seed full-training matrix: $MATRIX_ROOT"
