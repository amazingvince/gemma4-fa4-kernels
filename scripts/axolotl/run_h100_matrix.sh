#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG=${AXOLOTL_CONFIG:-$ROOT/configs/axolotl/gemma4-12b-smoke.yaml}
PROJECT_VENV=${AXOLOTL_PROJECT_VENV_DIR:-$ROOT/.venv-h100-axolotl}
BASELINE_VENV=${AXOLOTL_BASELINE_VENV_DIR:-$ROOT/.venv-h100-axolotl-baseline}
RUN_ID=${AXOLOTL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${AXOLOTL_RUN_ROOT:-$ROOT/agent_space/axolotl-exp0036/$RUN_ID}
DATASET=$RUN_ROOT/gemma4-12b-smoke.jsonl
CACHE_DIR=${REMOTE_FA4_CACHE_DIR:-$RUN_ROOT/fa4-cache}

mkdir -p "$RUN_ROOT" "$CACHE_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
export CUTE_DSL_ARCH=sm_90a
export FLASH_ATTENTION_ARCH=sm_90
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$CACHE_DIR"
export GEMMA4_FA4_DATASET_PATH="$DATASET"

"$PROJECT_VENV/bin/python" "$ROOT/scripts/axolotl/check_env.py" \
  --json "$RUN_ROOT/project-preflight.json"
"$BASELINE_VENV/bin/python" "$ROOT/scripts/axolotl/check_env.py" \
  --skip-fa4 --json "$RUN_ROOT/baseline-preflight.json"
"$PROJECT_VENV/bin/python" "$ROOT/scripts/axolotl/make_smoke_dataset.py" --output "$DATASET"

run_one() {
  local backend=$1
  local venv=$PROJECT_VENV
  if [[ "$backend" != project_12b_compat ]]; then
    venv=$BASELINE_VENV
  fi
  local backend_root=$RUN_ROOT/$backend
  mkdir -p "$backend_root"
  export GEMMA4_FA4_HARNESS_BACKEND="$backend"
  export GEMMA4_FA4_REPORT_PATH="$backend_root/report.json"
  export GEMMA4_FA4_OUTPUT_DIR="$backend_root/output"
  "$venv/bin/axolotl" train "$CONFIG" --launcher python 2>&1 | tee "$backend_root/train.log"
  test -s "$backend_root/report.json"
}

run_one "hybrid"
run_one "sdpa"
run_one "project_12b_compat"

comparison_status=0
if ! "$PROJECT_VENV/bin/python" "$ROOT/scripts/compare_axolotl_runs.py" \
  "$RUN_ROOT/hybrid/report.json" "$RUN_ROOT/project_12b_compat/report.json" \
  --output "$RUN_ROOT/hybrid-vs-project.json"; then
  comparison_status=1
fi
if ! "$PROJECT_VENV/bin/python" "$ROOT/scripts/compare_axolotl_runs.py" \
  "$RUN_ROOT/sdpa/report.json" "$RUN_ROOT/project_12b_compat/report.json" \
  --output "$RUN_ROOT/sdpa-vs-project.json"; then
  comparison_status=1
fi

if (( comparison_status != 0 )); then
  echo "EXP-0036 comparison rejected; reports: $RUN_ROOT" >&2
  exit "$comparison_status"
fi

echo "EXP-0036 reports: $RUN_ROOT"
