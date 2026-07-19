#!/usr/bin/env bash
# User-space development environment. Does not install or replace GPU drivers.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROFILE=${1:-${TARGET_PROFILE:-b300}}
case "$PROFILE" in
  h100) POLICY_FILE="$ROOT/configs/env/h100-compatible.env" ;;
  b300) POLICY_FILE="$ROOT/configs/env/latest-compatible.env" ;;
  *) echo "usage: $0 [h100|b300]" >&2; exit 2 ;;
esac
# shellcheck disable=SC1091
# shellcheck source=/dev/null
source "$POLICY_FILE"
PYTHON_BIN=${PYTHON_BIN:-python3.12}
VENV_DIR=${VENV_DIR:-$ROOT/.venv-$PROFILE}
UPSTREAM_DIR=${UPSTREAM_DIR:-$ROOT/.upstream}
INSTALL_HF_ORACLE=${INSTALL_HF_ORACLE:-1}
VERIFY_ONLINE=${VERIFY_ONLINE:-0}
export CUTE_DSL_ARCH FLASH_ATTENTION_ARCH

command -v "$PYTHON_BIN" >/dev/null || {
  echo "missing $PYTHON_BIN; install Python $PYTHON_VERSION first" >&2
  exit 1
}
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v nvcc >/dev/null || {
  echo "nvcc is required. See docs/environment.md and scripts/remote/install_cuda_ubuntu.sh" >&2
  exit 1
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==$PYTORCH_VERSION" \
  --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA_WHEEL"
python -m pip install -e "${ROOT}[dev]"

mkdir -p "$UPSTREAM_DIR"
ensure_checkout() {
  local url=$1
  local rev=$2
  local path=$3
  if [[ ! -d "$path/.git" ]]; then
    git clone --filter=blob:none "$url" "$path"
  fi
  git -C "$path" fetch --depth 1 origin "$rev"
  git -C "$path" checkout --detach "$rev"
}

ensure_checkout \
  "${FLASH_ATTN_REPO_URL:-https://github.com/Dao-AILab/flash-attention.git}" \
  "$FLASH_ATTN_REV" \
  "$UPSTREAM_DIR/flash-attention"
if [[ -n ${FLASH_ATTN_PATCH_PATH:-} ]]; then
  PATCH_PATH="$ROOT/$FLASH_ATTN_PATCH_PATH"
  [[ -f "$PATCH_PATH" ]] || {
    echo "missing required FlashAttention patch: $PATCH_PATH" >&2
    exit 1
  }
  PATCH_SHA256=$(sha256sum "$PATCH_PATH" | awk '{print $1}')
  [[ "$PATCH_SHA256" == "$FLASH_ATTN_PATCH_SHA256" ]] || {
    echo "FlashAttention patch hash mismatch: $PATCH_SHA256" >&2
    exit 1
  }
  if git -C "$UPSTREAM_DIR/flash-attention" apply --reverse --check "$PATCH_PATH"; then
    echo "required FlashAttention patch already applied: $FLASH_ATTN_PATCH_PATH"
  elif git -C "$UPSTREAM_DIR/flash-attention" apply --check "$PATCH_PATH"; then
    git -C "$UPSTREAM_DIR/flash-attention" apply "$PATCH_PATH"
    echo "applied FlashAttention patch: $FLASH_ATTN_PATCH_PATH"
  else
    echo "required FlashAttention patch does not apply cleanly" >&2
    exit 1
  fi
fi
python -m pip install "quack-kernels==$QUACK_KERNELS_VERSION"
python -m pip install -e "$UPSTREAM_DIR/flash-attention/flash_attn/cute[$FA4_EXTRAS]"

VERIFY_ARGS=()
if [[ "$VERIFY_ONLINE" == 1 ]]; then
  VERIFY_ARGS+=(--online)
fi
if [[ "$INSTALL_HF_ORACLE" == 1 ]]; then
  ensure_checkout \
    "${TRANSFORMERS_REPO_URL:-https://github.com/huggingface/transformers.git}" \
    "$TRANSFORMERS_REV" \
    "$UPSTREAM_DIR/transformers"
  python -m pip install -e "$UPSTREAM_DIR/transformers"
  VERIFY_ARGS+=(--transformers)
fi

python "$ROOT/scripts/verify_model_contract.py" "${VERIFY_ARGS[@]}"
CHECK_ARGS=(--profile "$PROFILE" --expect-arch "$EXPECTED_ARCH" --strict)
if [[ "$INSTALL_HF_ORACLE" == 1 ]]; then
  CHECK_ARGS+=(--require-transformers)
fi
python "$ROOT/scripts/check_env.py" "${CHECK_ARGS[@]}"
bash "$ROOT/scripts/verify_bundle.sh"

cat <<MSG
Environment ready:
  source $VENV_DIR/bin/activate
  upstream FA4: $UPSTREAM_DIR/flash-attention @ $FLASH_ATTN_REV
  Transformers oracle installed: $INSTALL_HF_ORACLE
  target profile: $PROFILE

The host toolkit policy is CUDA $CUDA_TOOLKIT_VERSION. PyTorch uses its official
$PYTORCH_CUDA_WHEEL wheel runtime. FA4 extras: [$FA4_EXTRAS].
MSG
