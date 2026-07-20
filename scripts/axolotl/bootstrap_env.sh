#!/usr/bin/env bash
# Build a separate, experimental Axolotl environment without touching .venv-h100.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/configs/env/h100-axolotl-experimental.env"
PYTHON_BIN=${PYTHON_BIN:-python3.12}
VENV_DIR=${AXOLOTL_VENV_DIR:-$ROOT/.venv-h100-axolotl}
UPSTREAM_DIR=${AXOLOTL_UPSTREAM_DIR:-$ROOT/.upstream/axolotl-exp0036}
INSTALL_AXOLOTL_FA2=${INSTALL_AXOLOTL_FA2:-0}
INSTALL_PROJECT_FA4=${INSTALL_PROJECT_FA4:-1}
export CUTE_DSL_ARCH FLASH_ATTENTION_ARCH
export MAX_JOBS=${MAX_JOBS:-8}

if [[ "$INSTALL_AXOLOTL_FA2" == 1 && "$INSTALL_PROJECT_FA4" == 1 ]]; then
  echo "FA2 and FA4 both own the flash_attn package; use separate baseline and project venvs" >&2
  exit 2
fi

command -v "$PYTHON_BIN" >/dev/null || { echo "missing $PYTHON_BIN" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc / CUDA 12.8 is required" >&2; exit 1; }

ensure_checkout() {
  local url=$1
  local revision=$2
  local path=$3
  local allowed_patch=${4:-}
  if [[ ! -d "$path/.git" ]]; then
    git clone --filter=blob:none "$url" "$path"
  fi
  if ! git -C "$path" diff --quiet || ! git -C "$path" diff --cached --quiet; then
    local head changed expected
    head=$(git -C "$path" rev-parse HEAD)
    changed=$(git -C "$path" diff --name-only | sort)
    expected=
    if [[ -n "$allowed_patch" && -f "$ROOT/$allowed_patch" ]]; then
      expected=$(git -C "$path" apply --numstat "$ROOT/$allowed_patch" | awk '{print $3}' | sort)
    fi
    if [[ "$head" != "$revision" || -z "$expected" || "$changed" != "$expected" ]] || \
       ! git -C "$path" apply --reverse --check "$ROOT/$allowed_patch"; then
      echo "refusing to replace a dirty managed checkout: $path" >&2
      exit 1
    fi
    echo "retaining the exact locked patch in $path"
  fi
  git -C "$path" fetch --depth 1 origin "$revision"
  git -C "$path" checkout --detach "$revision"
}

apply_locked_patch() {
  local checkout=$1
  local relative_patch=$2
  local expected_hash=$3
  local patch=$ROOT/$relative_patch
  local actual_hash
  actual_hash=$(sha256sum "$patch" | awk '{print $1}')
  [[ "$actual_hash" == "$expected_hash" ]] || {
    echo "patch hash mismatch for $relative_patch: $actual_hash" >&2
    exit 1
  }
  if git -C "$checkout" apply --reverse --check "$patch"; then
    return
  fi
  git -C "$checkout" apply --check "$patch"
  git -C "$checkout" apply "$patch"
}

mkdir -p "$UPSTREAM_DIR"
ensure_checkout https://github.com/axolotl-ai-cloud/axolotl.git "$AXOLOTL_REV" "$UPSTREAM_DIR/axolotl"
ensure_checkout https://github.com/huggingface/transformers.git "$TRANSFORMERS_REV" "$UPSTREAM_DIR/transformers" "$TRANSFORMERS_PATCH_PATH"
ensure_checkout https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_REV" "$UPSTREAM_DIR/flash-attention" "$FLASH_ATTN_PATCH_PATH"
apply_locked_patch "$UPSTREAM_DIR/transformers" "$TRANSFORMERS_PATCH_PATH" "$TRANSFORMERS_PATCH_SHA256"
apply_locked_patch "$UPSTREAM_DIR/flash-attention" "$FLASH_ATTN_PATCH_PATH" "$FLASH_ATTN_PATCH_SHA256"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip 'setuptools<82' wheel uv
uv pip install "torch==$PYTORCH_VERSION" torchvision --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA_WHEEL"
# Install Axolotl's tested dependency set and its FA2 hybrid baseline first.
uv pip install numpy ninja psutil
if [[ "$INSTALL_AXOLOTL_FA2" == 1 ]]; then
  uv pip install --no-build-isolation -e "$UPSTREAM_DIR/axolotl[flash-attn]"
else
  echo "installing Axolotl without optional FA2; hybrid baseline will be unavailable"
  uv pip install -e "$UPSTREAM_DIR/axolotl"
  # Remove a partial/previous FA2 install before installing FA4 into this venv.
  uv pip uninstall flash-attn >/dev/null 2>&1 || true
fi
# Replace only Transformers and FA4 with the project-pinned, patched sources.
uv pip install -e "$UPSTREAM_DIR/transformers"
if [[ "$INSTALL_PROJECT_FA4" == 1 ]]; then
  uv pip install --prerelease=allow "quack-kernels==$QUACK_KERNELS_VERSION"
  uv pip install --prerelease=allow -e "$UPSTREAM_DIR/flash-attention/flash_attn/cute[dev]"
fi
uv pip install -e "$ROOT[dev]"

CHECK_ARGS=(--skip-model-access)
if [[ "$INSTALL_PROJECT_FA4" != 1 ]]; then
  CHECK_ARGS+=(--skip-fa4)
fi
python "$ROOT/scripts/axolotl/check_env.py" "${CHECK_ARGS[@]}"

cat <<MSG
Experimental EXP-0036 environment ready:
  source $VENV_DIR/bin/activate

Authenticate to Hugging Face for the gated model, then rerun the full preflight:
  export HF_TOKEN=...
  python scripts/axolotl/check_env.py

Axolotl pins Transformers 5.14.1, while this experiment deliberately replaces it
with the project's patched revision. This boundary is experimental and does not
change configs/env/h100-compatible.env.
MSG
