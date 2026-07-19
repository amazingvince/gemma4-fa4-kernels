#!/usr/bin/env bash
# Install non-driver host prerequisites on Ubuntu 24.04.
set -euo pipefail

[[ ${ALLOW_SYSTEM_CHANGES:-0} == 1 ]] || {
  echo "Refusing system changes. Re-run with ALLOW_SYSTEM_CHANGES=1." >&2
  exit 2
}
[[ $(id -u) == 0 ]] || { echo "run as root or with sudo" >&2; exit 2; }
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || {
  echo "This script supports Ubuntu 24.04 only; found $PRETTY_NAME" >&2
  exit 2
}

apt-get update
apt-get install -y \
  build-essential ca-certificates curl git jq ninja-build pkg-config rsync \
  python3.12 python3.12-dev python3.12-venv shellcheck wget
