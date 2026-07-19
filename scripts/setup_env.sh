#!/usr/bin/env bash
# User-space development environment. Does not install or replace GPU drivers.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/configs/env/latest-compatible.env"
PYTHON_BIN=${PYTHON_BIN:-python3.12}
VENV_DIR=${VENV_DIR:-$ROOT/.venv}
UPSTREAM_DIR=${UPSTREAM_DIR:-$ROOT/.upstream}
INSTALL_HF_ORACLE=${INSTALL_HF_ORACLE:-1}
VERIFY_ONLINE=${VERIFY_ONLINE:-0}

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
python -m pip install -e "$UPSTREAM_DIR/flash-attention/flash_attn/cute[dev,cu13]"

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
python "$ROOT/scripts/check_env.py"
bash "$ROOT/scripts/verify_bundle.sh"

cat <<MSG
Environment ready:
  source $VENV_DIR/bin/activate
  upstream FA4: $UPSTREAM_DIR/flash-attention @ $FLASH_ATTN_REV
  Transformers oracle installed: $INSTALL_HF_ORACLE

The host toolkit policy is CUDA $CUDA_TOOLKIT_VERSION. PyTorch uses its official
$PYTORCH_CUDA_WHEEL wheel runtime; those minor versions are intentionally not identical.
MSG
