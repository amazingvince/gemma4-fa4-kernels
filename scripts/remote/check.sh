#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?usage: check.sh <profile>}
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"
remote_exec \
  'if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; python scripts/check_env.py --expect-arch '"$EXPECTED_ARCH"' --strict --require-transformers'
