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
tracked_list=$(mktemp)
trap 'rm -f "$tracked_list"' EXIT

# WSL's Git cannot resolve the Windows-absolute gitdir stored by a linked
# Windows worktree. Resolve that pointer explicitly so a sync from an isolated
# worktree cannot silently omit tracked agent_space evidence.
if ! git -C "$REPO_ROOT" ls-files -z -- agent_space >"$tracked_list" 2>/dev/null; then
  gitdir=
  if [[ -f "$REPO_ROOT/.git" ]] && command -v wslpath >/dev/null 2>&1; then
    gitdir=$(sed -n 's/^gitdir: //p' "$REPO_ROOT/.git")
    if [[ "$gitdir" =~ ^[A-Za-z]:[\\/] ]]; then
      gitdir=$(wslpath -u "$gitdir")
    fi
  fi
  if [[ -z "$gitdir" ]] || \
     ! git --git-dir="$gitdir" --work-tree="$REPO_ROOT" \
       ls-files -z -- agent_space >"$tracked_list"; then
    echo "Unable to enumerate tracked agent_space files from $REPO_ROOT" >&2
    exit 2
  fi
fi
while IFS= read -r -d '' tracked_path; do
  AGENT_SPACE_FILES+=("--include=$tracked_path")
  tracked_dir=${tracked_path%/*}
  while [[ "$tracked_dir" == agent_space/* ]]; do
    AGENT_SPACE_DIRS["$tracked_dir/"]=1
    tracked_dir=${tracked_dir%/*}
  done
done <"$tracked_list"

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
