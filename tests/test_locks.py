import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_lock_has_full_shas():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    for section in ("flash_attention", "transformers", "gemma4_31b"):
        assert len(lock[section]["revision"]) == 40


def test_latest_environment_policy_is_explicit():
    text = (ROOT / "configs/env/latest-compatible.env").read_text()
    assert "CUDA_TOOLKIT_VERSION=13.3" in text
    assert "PYTORCH_VERSION=2.13.0" in text
    assert "PYTORCH_CUDA_WHEEL=cu132" in text
    assert "PYTORCH_CUDA_RUNTIME=13.2" in text
    assert "CUTLASS_DSL_VERSION=4.6.0.dev0" in text
