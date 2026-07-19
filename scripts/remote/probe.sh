#!/usr/bin/env bash
# Verify SSH, scheduler initialization, and GPU visibility before syncing source.
set -euo pipefail
PROFILE=${1:?usage: probe.sh <profile>}
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"
probe='nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader && command -v nvcc || true'
wrapped="$REMOTE_INIT_COMMAND && $probe"
if [[ -n "$REMOTE_LAUNCHER" ]]; then
  printf -v probe_q '%q' "$probe"
  wrapped="$REMOTE_INIT_COMMAND && $REMOTE_LAUNCHER bash -lc $probe_q"
fi
printf -v wrapped_q '%q' "$wrapped"
ssh -t "${SSH_ARGS[@]}" "$REMOTE_TARGET" "bash -lc $wrapped_q"
