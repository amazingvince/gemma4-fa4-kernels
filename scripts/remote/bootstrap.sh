#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?usage: bootstrap.sh <profile>}
"$(dirname "$0")/sync.sh" "$PROFILE"
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"
remote_exec_tty \
  'bash scripts/setup_env.sh && source .venv/bin/activate && python scripts/check_env.py --expect-arch '"$EXPECTED_ARCH"' --strict --require-transformers'
