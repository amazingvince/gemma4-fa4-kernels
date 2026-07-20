#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?usage: run.sh <profile> <command...>}
shift
[[ $# -gt 0 ]] || { echo "command required" >&2; exit 2; }
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"
COMMAND=$(quote_command "$@")
remote_exec "source .venv-$PROFILE/bin/activate && $COMMAND"
