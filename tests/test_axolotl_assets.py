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
        "skip_prepare_dataset: true",
        "remove_unused_columns: false",
        "field_messages: messages",
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


def test_dataset_generator_is_byte_deterministic_and_bounded(tmp_path):
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
    assert all(set(row) == {"messages"} for row in rows)
    prompt_word_counts = [len(row["messages"][0]["content"].split()) for row in rows]
    assert all(750 <= count <= 850 for count in prompt_word_counts)
    assert all(row["messages"][0]["content"].endswith(" padding" * 9) for row in rows)


def test_real_training_dataset_conversion_is_deterministic_and_message_only():
    generator = _load_script(
        "make_axolotl_training_dataset",
        "scripts/axolotl/make_training_dataset.py",
    )
    source = [
        {"instruction": f"instruction {index}", "input": "", "output": f"answer {index}"}
        for index in range(20)
    ]
    first, first_indices = generator.build_records(source, records=10, seed=4721)
    second, second_indices = generator.build_records(source, records=10, seed=4721)

    assert first == second
    assert first_indices == second_indices
    assert len(set(first_indices)) == 10
    assert all(set(record) == {"messages"} for record in first)
    assert all(
        [message["role"] for message in record["messages"]] == ["user", "assistant"]
        for record in first
    )


def test_full_training_yaml_locks_real_updates_and_memory_controls():
    text = (ROOT / "configs/axolotl/gemma4-12b-full-bf16-100steps.yaml").read_text()
    required = (
        "gemma4_fa4.axolotl_training_plugin.Fa4FullTrainingPlugin",
        "fa4_training_backend: native",
        "sequence_len: 512",
        "dataset_num_proc: 1",
        "skip_prepare_dataset: true",
        "micro_batch_size: 1",
        "gradient_accumulation_steps: 1",
        "max_steps: 100",
        "optimizer: adafactor",
        "learning_rate: 0.00001",
        "max_grad_norm: 1.0",
        "bf16: true",
        "gradient_checkpointing: true",
        "use_reentrant: false",
        "sample_packing: false",
    )
    for fragment in required:
        assert fragment in text
    assert "adapter:" not in text
    assert "learning_rate: 0.0\n" not in text


def test_full_training_runner_uses_matched_control_and_all_fa4_candidate():
    text = (ROOT / "scripts/axolotl/run_h100_full_training.sh").read_text()
    assert 'run_one "sdpa"' in text
    assert 'run_one "native"' in text
    assert "make_training_dataset.py" in text
    assert "compare_axolotl_training.py" in text
    assert "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1" in text
    assert "SEED=${AXOLOTL_SEED:-4721}" in text
    assert 'GEMMA4_FA4_TRAINING_SEED="$SEED"' in text
    assert "AXOLOTL_DEFER_PAIR_GATE" in text


def test_full_training_matrix_runner_locks_three_fresh_seed_pairs():
    text = (ROOT / "scripts/axolotl/run_h100_full_training_matrix.sh").read_text()
    assert "SEEDS=(1729 31415 65537)" in text
    assert "flock -n 9" in text
    assert "--query-compute-apps=pid" in text
    assert "run_h100_full_training.sh" in text
    assert "compare_axolotl_training_matrix.py" in text
    assert "AXOLOTL_DEFER_PAIR_GATE=1" in text


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


def test_environment_policy_accepts_exact_clean_generic_fa4_candidate():
    check = _load_script("check_axolotl_env_candidate", "scripts/axolotl/check_env.py")
    candidate = {
        "python_version": "3.12.4",
        "torch_version": "2.11.0",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_capability": [9, 0],
        "total_memory_bytes": 80 << 30,
        "axolotl_revision": check.AXOLOTL_REVISION,
        "transformers_revision": check.TRANSFORMERS_REVISION,
        "flash_attention_revision": "17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1",
        "transformers_patch_applied": True,
        "flash_attention_patch_applied": False,
        "flash_attention_worktree_clean": True,
        "model_access": True,
    }
    assert (
        check.validate_snapshot(
            candidate,
            expected_flash_revision="17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1",
            require_flash_patch=False,
        )
        == []
    )


def test_matrix_runner_contains_all_backends_and_two_correctness_comparisons():
    text = (ROOT / "scripts/axolotl/run_h100_matrix.sh").read_text()
    for backend in ("hybrid", "sdpa", "project_12b_compat"):
        assert f'run_one "{backend}"' in text
    assert text.count("scripts/compare_axolotl_runs.py") == 2
    assert "comparison_status=0" in text
    assert "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1" in text
    assert 'FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$CACHE_DIR"' in text
    assert "set -euo pipefail" in text


def test_comparison_cli_writes_a_rejected_result_artifact(tmp_path, monkeypatch):
    compare = _load_script("compare_axolotl_runs", "scripts/compare_axolotl_runs.py")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text("{}")
    candidate.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_axolotl_runs.py",
            str(baseline),
            str(candidate),
            "--output",
            str(output),
        ],
    )

    assert compare.main() == 1
    payload = json.loads(output.read_text())
    assert payload["passed"] is False
    assert payload["schema_version"] == 2
    assert "unsupported schema" in payload["error"]


def test_plugin_resets_and_records_backend_independent_lora_initialization():
    text = (ROOT / "src/gemma4_fa4/axolotl_plugin.py").read_text()
    assert "initialize_lora_parameters" in text
    assert '"lora_initialization": self.lora_initialization' in text


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
