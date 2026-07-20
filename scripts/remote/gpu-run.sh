#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:?usage: gpu-run.sh <profile> [--no-venv] <command...>}
shift
USE_VENV=1
if [[ ${1:-} == --no-venv ]]; then
  USE_VENV=0
  shift
fi
[[ $# -gt 0 ]] || { echo "command required" >&2; exit 2; }

# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"

GPU_LOCK_FILE=${REMOTE_GPU_LOCK_FILE:-/tmp/gemma4-fa4-$PROFILE.gpu.lock}
GPU_LOCK_WAIT_SECONDS=${REMOTE_GPU_LOCK_WAIT_SECONDS:-0}
GPU_REQUIRE_IDLE=${REMOTE_GPU_REQUIRE_IDLE:-1}

if [[ "$GPU_LOCK_FILE" != /* ]] || [[ "$GPU_LOCK_FILE" =~ [^A-Za-z0-9._/:-] ]]; then
  echo "REMOTE_GPU_LOCK_FILE must be a simple absolute path without spaces" >&2
  exit 2
fi
if [[ ! "$GPU_LOCK_WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "REMOTE_GPU_LOCK_WAIT_SECONDS must be a nonnegative integer" >&2
  exit 2
fi
if [[ "$GPU_REQUIRE_IDLE" != 0 && "$GPU_REQUIRE_IDLE" != 1 ]]; then
  echo "REMOTE_GPU_REQUIRE_IDLE must be 0 or 1" >&2
  exit 2
fi

COMMAND=$(quote_command "$@")
VENV_COMMAND=$COMMAND
if [[ "$USE_VENV" == 1 ]]; then
  printf -v VENV_COMMAND 'source .venv-%q/bin/activate && %s' "$PROFILE" "$COMMAND"
fi
printf -v VENV_COMMAND_Q '%q' "$VENV_COMMAND"

IDLE_CHECK=:
if [[ "$GPU_REQUIRE_IDLE" == 1 ]]; then
  # Expanded only by the remote lease shell after flock succeeds.
  # shellcheck disable=SC2016
  IDLE_CHECK='busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true); if [[ -n "$busy" ]]; then echo "GPU lease acquired, but an existing compute process is still present:" >&2; echo "$busy" >&2; exit 76; fi'
fi

# Dollar expressions in this format string intentionally belong to the
# generated remote shell body, not this process.
# shellcheck disable=SC2016
printf -v LEASE_BODY '%s; printf "pid=%%s host=%%s started=%%s command=%%q\\n" "$$$$" "$(hostname)" "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" %q > %q.owner; trap '\''rm -f %q.owner'\'' EXIT; bash -lc %s' \
  "$IDLE_CHECK" "$COMMAND" "$GPU_LOCK_FILE" "$GPU_LOCK_FILE" "$VENV_COMMAND_Q"
printf -v LEASE_BODY_Q '%q' "$LEASE_BODY"

LEASE_COMMAND="flock --exclusive --conflict-exit-code 75 --wait $GPU_LOCK_WAIT_SECONDS $GPU_LOCK_FILE bash -lc $LEASE_BODY_Q"
LEASE_COMMAND+=" || { rc=\$?; if [[ \$rc -eq 75 ]]; then echo 'GPU lease unavailable: $GPU_LOCK_FILE' >&2; fi; exit \$rc; }"

remote_exec "$LEASE_COMMAND"
