#!/usr/bin/env bash
set -euo pipefail

REMOTE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$REMOTE_SCRIPT_DIR/../.." && pwd)
PROFILE=${1:?usage: $0 <profile> [args...]}
PROFILE_FILE="$REPO_ROOT/remote/$PROFILE.env"
[[ -f "$PROFILE_FILE" ]] || {
  echo "Missing $PROFILE_FILE. Copy remote/$PROFILE.env.example and fill it in." >&2
  exit 2
}
# shellcheck disable=SC1090
source "$PROFILE_FILE"
: "${REMOTE_HOST:?REMOTE_HOST required}"
: "${REMOTE_USER:?REMOTE_USER required}"
: "${REMOTE_PORT:=22}"
: "${REMOTE_ROOT:='~/work/gemma4-fa4-kernels'}"
if { [[ "$REMOTE_ROOT" != "~/"* ]] && [[ "$REMOTE_ROOT" != /* ]]; } || \
   [[ "$REMOTE_ROOT" =~ [^A-Za-z0-9._/~:-] ]]; then
  echo "REMOTE_ROOT must be a simple absolute path or ~/relative/path without spaces" >&2
  exit 2
fi
: "${EXPECTED_ARCH:?EXPECTED_ARCH required}"
: "${CUTE_DSL_ARCH:?CUTE_DSL_ARCH required}"
: "${FLASH_ATTENTION_ARCH:?FLASH_ATTENTION_ARCH required}"
REMOTE_FA4_CACHE_DIR=${REMOTE_FA4_CACHE_DIR:-/tmp/$REMOTE_USER/fa4-cache-$EXPECTED_ARCH}
REMOTE_INIT_COMMAND=${REMOTE_INIT_COMMAND:-:}
REMOTE_LAUNCHER=${REMOTE_LAUNCHER:-}

SSH_ARGS=(-p "$REMOTE_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
if [[ -n ${REMOTE_IDENTITY_FILE:-} ]]; then
  SSH_ARGS+=(-i "$REMOTE_IDENTITY_FILE")
fi
if [[ -n ${REMOTE_PROXY_JUMP:-} ]]; then
  SSH_ARGS+=(-J "$REMOTE_PROXY_JUMP")
fi
if [[ -n ${REMOTE_EXTRA_SSH_ARGS:-} ]]; then
  # Trusted local profile; shell-style words are intentional here.
  # shellcheck disable=SC2206
  EXTRA_ARGS=($REMOTE_EXTRA_SSH_ARGS)
  SSH_ARGS+=("${EXTRA_ARGS[@]}")
fi
REMOTE_TARGET="$REMOTE_USER@$REMOTE_HOST"
printf -v REMOTE_ENV \
  'export CUTE_DSL_ARCH=%q FLASH_ATTENTION_ARCH=%q FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=%q' \
  "$CUTE_DSL_ARCH" "$FLASH_ATTENTION_ARCH" "$REMOTE_FA4_CACHE_DIR"

quote_command() {
  printf '%q ' "$@"
}

remote_exec() {
  local command=$1
  local payload="cd $REMOTE_ROOT && $REMOTE_ENV && $command"
  local wrapped="$REMOTE_INIT_COMMAND && $payload"
  if [[ -n "$REMOTE_LAUNCHER" ]]; then
    printf -v payload_q '%q' "$payload"
    wrapped="$REMOTE_INIT_COMMAND && $REMOTE_LAUNCHER bash -lc $payload_q"
  fi
  printf -v wrapped_q '%q' "$wrapped"
  ssh "${SSH_ARGS[@]}" "$REMOTE_TARGET" "bash -lc $wrapped_q"
}

remote_exec_tty() {
  local command=$1
  local payload="cd $REMOTE_ROOT && $REMOTE_ENV && $command"
  local wrapped="$REMOTE_INIT_COMMAND && $payload"
  if [[ -n "$REMOTE_LAUNCHER" ]]; then
    printf -v payload_q '%q' "$payload"
    wrapped="$REMOTE_INIT_COMMAND && $REMOTE_LAUNCHER bash -lc $payload_q"
  fi
  printf -v wrapped_q '%q' "$wrapped"
  ssh -t "${SSH_ARGS[@]}" "$REMOTE_TARGET" "bash -lc $wrapped_q"
}

rsync_rsh() {
  local result="ssh"
  local arg
  for arg in "${SSH_ARGS[@]}"; do
    printf -v quoted '%q' "$arg"
    result+=" $quoted"
  done
  printf '%s' "$result"
}
