#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG=${AXOLOTL_CONFIG:-$ROOT/configs/axolotl/gemma4-12b-full-bf16-100steps.yaml}
VENV=${AXOLOTL_PROJECT_VENV_DIR:-/workspace/gemma4-fa4-kernels/.venv-h100-axolotl}
FA4_ROOT=${FLASH_ATTENTION_SOURCE_ROOT:-/workspace/flash-attention-fa4-d512-pr}
FA2_EXTENSION_SITE=${FLASH_ATTENTION_2_EXTENSION_SITE:-/workspace/gemma4-fa4-kernels/.venv-h100-axolotl-baseline/lib/python3.12/site-packages}
RUN_ID=${AXOLOTL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${AXOLOTL_RUN_ROOT:-$ROOT/agent_space/axolotl-full-training/$RUN_ID}
DATASET=$RUN_ROOT/alpaca.jsonl
CACHE_DIR=${REMOTE_FA4_CACHE_DIR:-$RUN_ROOT/fa4-cache}

mkdir -p "$RUN_ROOT" "$CACHE_DIR"
export PYTHONPATH="$FA4_ROOT:$FA2_EXTENSION_SITE:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
export CUTE_DSL_ARCH=sm_90a
export FLASH_ATTENTION_ARCH=sm_90
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$CACHE_DIR"
export GEMMA4_FA4_TRAINING_DATASET_PATH="$DATASET"
export GEMMA4_FA4_SOURCE_REVISION=${GEMMA4_FA4_SOURCE_REVISION:-955e906a6441e3f618335d200be39a03ca201c40}
export FLASH_ATTENTION_SOURCE_REVISION=${FLASH_ATTENTION_SOURCE_REVISION:-17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1}

"$VENV/bin/python" "$ROOT/scripts/axolotl/check_env.py" \
  --expected-fa4-revision "$FLASH_ATTENTION_SOURCE_REVISION" \
  --expected-fa4-root "$FA4_ROOT" \
  --expected-fa2-extension-root "$FA2_EXTENSION_SITE" \
  --json "$RUN_ROOT/preflight.json"
"$VENV/bin/python" "$ROOT/scripts/axolotl/make_training_dataset.py" \
  --output "$DATASET" --records 256 --seed 4721 \
  | tee "$RUN_ROOT/dataset.log"

run_one() {
  local backend=$1
  local backend_root=$RUN_ROOT/$backend
  mkdir -p "$backend_root"
  export GEMMA4_FA4_TRAINING_BACKEND="$backend"
  export GEMMA4_FA4_TRAINING_REPORT_PATH="$backend_root/report.json"
  export GEMMA4_FA4_OUTPUT_DIR="$backend_root/output"
  "$VENV/bin/axolotl" train "$CONFIG" --launcher python \
    2>&1 | tee "$backend_root/train.log"
  test -s "$backend_root/report.json"
}

run_one "sdpa"
run_one "native"

"$VENV/bin/python" "$ROOT/scripts/compare_axolotl_training.py" \
  "$RUN_ROOT/sdpa/report.json" "$RUN_ROOT/native/report.json" \
  --output "$RUN_ROOT/comparison.json"

echo "Full-training reports: $RUN_ROOT"
