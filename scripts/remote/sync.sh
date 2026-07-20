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

# Evidence under agent_space is ignored by default because raw profiler and
# codegen artifacts can be very large. Transfer only files already committed
# to Git, plus every parent directory rsync must traverse to reach them. This
# keeps the remote checksum bundle complete without maintaining a stale manual
# allowlist or uploading ignored scratch artifacts.
declare -a AGENT_SPACE_FILES=()
declare -A AGENT_SPACE_DIRS=()
while IFS= read -r -d '' tracked_path; do
  AGENT_SPACE_FILES+=("--include=$tracked_path")
  tracked_dir=${tracked_path%/*}
  while [[ "$tracked_dir" == agent_space/* ]]; do
    AGENT_SPACE_DIRS["$tracked_dir/"]=1
    tracked_dir=${tracked_dir%/*}
  done
done < <(git -C "$REPO_ROOT" ls-files -z -- agent_space)

AGENT_SPACE_INCLUDES=("--include=agent_space/")
for tracked_dir in "${!AGENT_SPACE_DIRS[@]}"; do
  AGENT_SPACE_INCLUDES+=("--include=$tracked_dir")
done
AGENT_SPACE_INCLUDES+=("${AGENT_SPACE_FILES[@]}")

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '.upstream/' \
  "${AGENT_SPACE_INCLUDES[@]}" \
  --exclude 'agent_space/*' \
  --exclude 'remote/*.env' \
  -e "$RSH" \
  "$REPO_ROOT/" "$REMOTE_TARGET:$REMOTE_ROOT/"
echo "synced $REPO_ROOT -> $REMOTE_TARGET:$REMOTE_ROOT"
