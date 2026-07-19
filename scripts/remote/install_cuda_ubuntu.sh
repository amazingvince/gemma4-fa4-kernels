#!/usr/bin/env bash
# Install the selected target's CUDA Toolkit on Ubuntu 24.04 without changing the driver.
# Driver changes can require a reboot and must be coordinated with the machine owner.
set -euo pipefail

PROFILE=${1:-b300}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case "$PROFILE" in
  h100) POLICY_FILE="$ROOT/configs/env/h100-compatible.env" ;;
  b300) POLICY_FILE="$ROOT/configs/env/latest-compatible.env" ;;
  *) echo "usage: $0 [h100|b300]" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "$POLICY_FILE"

[[ ${ALLOW_SYSTEM_CHANGES:-0} == 1 ]] || {
  echo "Refusing system changes. Re-run with ALLOW_SYSTEM_CHANGES=1." >&2
  exit 2
}
[[ $(id -u) == 0 ]] || { echo "run as root or with sudo" >&2; exit 2; }
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || {
  echo "This script supports Ubuntu 24.04 only; follow NVIDIA's guide for $PRETTY_NAME" >&2
  exit 2
}

apt-get update
apt-get install -y wget ca-certificates gnupg
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
wget -q -O "$TMP/cuda-keyring.deb"   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i "$TMP/cuda-keyring.deb"
apt-get update
apt-get install -y "cuda-toolkit-$CUDA_TOOLKIT_PACKAGE"

cat <<EOF
CUDA Toolkit $CUDA_TOOLKIT_VERSION installed for $PROFILE. Add
/usr/local/cuda-$CUDA_TOOLKIT_VERSION/bin to PATH. Verify the driver separately
with scripts/check_env.py --profile $PROFILE. This script does not replace a
working datacenter driver or reboot the host.
EOF
