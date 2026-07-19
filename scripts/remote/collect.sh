#!/usr/bin/env bash
set -euo pipefail
PROFILE=${1:?usage: collect.sh <profile> [destination]}
DEST=${2:-agent_space/remote-$PROFILE}
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh" "$PROFILE"
mkdir -p "$REPO_ROOT/$DEST"
RSH=$(rsync_rsh)
rsync -az -e "$RSH" \
  "$REMOTE_TARGET:$REMOTE_ROOT/agent_space/" "$REPO_ROOT/$DEST/"
echo "collected remote artifacts in $DEST"
