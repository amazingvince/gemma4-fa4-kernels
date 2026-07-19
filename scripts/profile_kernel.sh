#!/usr/bin/env bash
# Usage: scripts/profile_kernel.sh <python command and arguments...>
set -euo pipefail
[[ $# -gt 0 ]] || { echo "command required" >&2; exit 2; }
mkdir -p agent_space
OUT=agent_space/ncu_$(date +%Y%m%d_%H%M%S)
ncu   --section SpeedOfLight   --section MemoryWorkloadAnalysis   --section SchedulerStats   --section WarpStateStats   --section Occupancy   --import-source on   -o "$OUT"   "$@"
echo "profile: $OUT.ncu-rep"
echo "Run --set full once per major kernel structure; use targeted sections while iterating."
