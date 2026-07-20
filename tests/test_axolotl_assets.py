from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_axolotl_yaml_freezes_short_zero_update_workload():
    text = (ROOT / "configs/axolotl/gemma4-12b-smoke.yaml").read_text()
    required = (
        "base_model: google/gemma-4-12B-it",
        "revision_of_model: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "gemma4_fa4.axolotl_plugin.Fa4AxolotlHarnessPlugin",
        "sequence_len: 1024",
        "sample_packing: false",
        "learning_rate: 0.0",
        "weight_decay: 0.0",
        "lora_dropout: 0.0",
        "gradient_accumulation_steps: 1",
        "micro_batch_size: 1",
        "max_steps: 8",
        "fa4_harness_warmup_steps: 3",
        "torch_compile: false",
        "seed: 3600",
    )
    for fragment in required:
        assert fragment in text
    assert "wandb_project:" not in text
    assert "save_strategy: steps" not in text


def test_dataset_generator_is_byte_deterministic_and_long(tmp_path):
    generator = _load_script(
        "make_axolotl_smoke_dataset",
        "scripts/axolotl/make_smoke_dataset.py",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_result = generator.write_dataset(first, records=12)
    second_result = generator.write_dataset(second, records=12)

    assert first.read_bytes() == second.read_bytes()
    assert {key: value for key, value in first_result.items() if key != "path"} == {
        key: value for key, value in second_result.items() if key != "path"
    }
    assert first_result["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(rows) == 12
    assert all(len(row["conversations"][0]["content"].split()) > 1800 for row in rows)


def test_environment_policy_rejects_wrong_gpu_and_revisions():
    check = _load_script("check_axolotl_env", "scripts/axolotl/check_env.py")
    good = {
        "python_version": "3.12.4",
        "torch_version": "2.12.1",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_capability": [9, 0],
        "device_name": "NVIDIA H100 80GB HBM3",
        "total_memory_bytes": 80 << 30,
        "axolotl_revision": check.AXOLOTL_REVISION,
        "transformers_revision": check.TRANSFORMERS_REVISION,
        "flash_attention_revision": check.FLASH_ATTENTION_REVISION,
        "transformers_patch_applied": True,
        "flash_attention_patch_applied": True,
        "model_access": True,
    }
    assert check.validate_snapshot(good) == []

    bad = dict(good)
    bad.update(
        cuda_capability=[8, 0],
        axolotl_revision="bad",
        model_access=False,
    )
    errors = check.validate_snapshot(bad)
    assert any("SM90" in error for error in errors)
    assert any("Axolotl revision" in error for error in errors)
    assert any("gated model" in error for error in errors)


def test_matrix_runner_contains_all_backends_and_two_correctness_comparisons():
    text = (ROOT / "scripts/axolotl/run_h100_matrix.sh").read_text()
    for backend in ("hybrid", "sdpa", "project_12b_compat"):
        assert f'run_one "{backend}"' in text
    assert text.count("scripts/compare_axolotl_runs.py") == 2
    assert "set -euo pipefail" in text


def test_bootstrap_is_isolated_and_pins_the_experimental_stack():
    text = (ROOT / "scripts/axolotl/bootstrap_env.sh").read_text()
    assert ".venv-h100-axolotl" in text
    assert "h100-axolotl-experimental.env" in text
    assert "axolotl[flash-attn]" in text
    assert "scripts/setup_env.sh" not in text


@pytest.mark.parametrize("module", ["axolotl_harness", "gemma4_12b_compat"])
def test_new_modules_export_documented_public_symbols(module):
    imported = __import__(f"gemma4_fa4.{module}", fromlist=["__all__"])
    assert imported.__all__
