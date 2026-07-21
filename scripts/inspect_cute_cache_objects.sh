#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CACHE_SUBDIRECTORY" >&2
  exit 2
fi

cache_subdirectory=$1
if [[ ! -d "$cache_subdirectory" ]]; then
  echo "cache directory does not exist: $cache_subdirectory" >&2
  exit 2
fi

declare -A seen=()
while IFS= read -r -d '' object; do
  digest=$(sha256sum "$object" | cut -d' ' -f1)
  if [[ -n "${seen[$digest]:-}" ]]; then
    continue
  fi
  seen[$digest]=1

  short_digest=${digest:0:12}
  object_copy=$(mktemp "/tmp/cute-object-${short_digest}-XXXXXX.o")
  device_image=$(mktemp "/tmp/cute-${short_digest}-XXXXXX.bin")
  trap 'rm -f "$object_copy" "$device_image"' EXIT
  cp -- "$object" "$object_copy"
  objcopy --dump-section .lrodata="$device_image" "$object_copy"

  echo "object=$object sha256=$digest bytes=$(stat -c%s "$object")"
  cuobjdump --dump-resource-usage "$device_image" 2>&1 | grep 'REG:' || true
  sass=$(cuobjdump --dump-sass "$device_image" 2>/dev/null || true)
  echo "sass_hgmma=$(grep -c HGMMA <<<"$sass" || true) utma=$(grep -c UTMA <<<"$sass" || true) cpasync=$(grep -c CPASYNC <<<"$sass" || true) mbarrier=$(grep -c MBARRIER <<<"$sass" || true) ldl=$(grep -c LDL <<<"$sass" || true) stl=$(grep -c STL <<<"$sass" || true) call=$(grep -c CALL <<<"$sass" || true)"

  digest_after=$(sha256sum "$object" | cut -d' ' -f1)
  if [[ "$digest_after" != "$digest" ]]; then
    echo "cache object changed during inspection: $object" >&2
    exit 1
  fi
  rm -f "$object_copy" "$device_image"
  trap - EXIT
done < <(find "$cache_subdirectory" -maxdepth 1 -type f -name '*.o' -print0 | sort -z)
