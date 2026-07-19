#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?usage: sync.sh <profile>}
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"

# REMOTE_ROOT is validated in common.sh; keep a leading ~/ unescaped so the
# remote login shell expands it.
mkdir_command="mkdir -p $REMOTE_ROOT"
ssh "${SSH_ARGS[@]}" "$REMOTE_TARGET" "$mkdir_command"
RSH=$(rsync_rsh)
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.upstream/' \
  --exclude 'agent_space/' \
  --exclude 'remote/*.env' \
  -e "$RSH" \
  "$REPO_ROOT/" "$REMOTE_TARGET:$REMOTE_ROOT/"
echo "synced $REPO_ROOT -> $REMOTE_TARGET:$REMOTE_ROOT"
