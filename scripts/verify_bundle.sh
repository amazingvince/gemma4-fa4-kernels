#!/usr/bin/env bash
# Offline/static verification for the starter repository itself.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
cleanup() {
  find "$ROOT" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) \
    -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT" -type f -name '*.pyc' -delete 2>/dev/null || true
}
trap cleanup EXIT

python -m compileall -q src tests scripts benchmarks
python scripts/verify_model_contract.py
pytest -q -p no:cacheprovider
if python -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
  python -m ruff check --no-cache .
  python -m ruff format --check src tests scripts benchmarks
else
  echo "WARNING: ruff not installed (pip install -e '.[dev]'); lint checks SKIPPED" >&2
fi

for script in scripts/*.sh scripts/remote/*.sh; do
  bash -n "$script"
done
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck scripts/*.sh scripts/remote/*.sh
else
  echo "NOTE: shellcheck not installed; bash -n completed" >&2
fi

python skills/writing-cute-dsl-kernels/scripts/validate_skill.py
(
  cd skills/writing-cute-dsl-kernels
  sha256sum -c SHA256SUMS.txt
)

python - <<'PY'
import json
from pathlib import Path
for path in Path('.').rglob('*.json'):
    json.loads(path.read_text())
print('JSON validation passed')
PY

echo "starter bundle verification passed"
