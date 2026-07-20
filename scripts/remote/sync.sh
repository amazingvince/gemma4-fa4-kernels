#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?usage: sync.sh <profile>}
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"

# REMOTE_ROOT is validated in common.sh; keep a leading ~/ unescaped so the
# remote login shell expands it.
mkdir_command="mkdir -p $REMOTE_ROOT"
# mkdir_command is intentionally expanded locally for evaluation by the remote shell.
# shellcheck disable=SC2029
ssh "${SSH_ARGS[@]}" "$REMOTE_TARGET" "$mkdir_command"
RSH=$(rsync_rsh)
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '.upstream/' \
  --include 'agent_space/' \
  --include 'agent_space/README.md' \
  --include 'agent_space/h100-check-exp0006.json' \
  --include 'agent_space/h100-check-exp0012.json' \
  --include 'agent_space/h100-check-exp0013.json' \
  --include 'agent_space/h100-check-exp0014.json' \
  --include 'agent_space/h100-check-precommit.json' \
  --exclude 'agent_space/*' \
  --exclude 'remote/*.env' \
  -e "$RSH" \
  "$REPO_ROOT/" "$REMOTE_TARGET:$REMOTE_ROOT/"
echo "synced $REPO_ROOT -> $REMOTE_TARGET:$REMOTE_ROOT"
