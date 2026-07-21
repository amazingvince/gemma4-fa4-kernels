from __future__ import annotations

import copy

import pytest
import torch

from gemma4_fa4.axolotl_training_harness import (
    PROSPECTIVE_MATRIX_SEEDS,
    TrainingComparisonError,
    align_trainable_parameters_to_bf16,
    compare_full_training_matrix,
    compare_full_training_reports,
    mutate_full_training_config,
)


def _report(
    backend: str,
    *,
    loss_delta: float = 0.0,
    norm_scale: float = 1.0,
    seed: int = 4721,
    step_ms: float = 4.0,
):
    trainable = 11_000_000_000
    metrics = [
        {
            "step": step,
            "loss": 2.5 - 0.005 * step + loss_delta,
            "grad_norm": (1.0 + 0.001 * step) * norm_scale,
            "learning_rate": 1e-5,
        }
        for step in range(1, 101)
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "errors": [],
        "backend": backend,
        "workload": {
            "model_id": "google/gemma-4-12B-it",
            "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "sequence_len": 512,
            "max_steps": 100,
            "seed": seed,
        },
        "dataset": {"sha256": "a" * 64, "records": 256},
        "parameters": {
            "trainable": trainable,
            "frozen": 400_000_000,
            "adapter_parameters": 0,
            "trainable_dtypes": {"torch.bfloat16": trainable},
        },
        "routes": {},
        "layer_routes": {},
        "native_geometries": {},
        "training_metrics": metrics,
        "step_times_ms": [{"step": step, "milliseconds": step_ms} for step in range(1, 101)],
        "timing_warmup_steps": 5,
        "peak_memory": {
            "allocated_bytes": 64_000_000_000,
            "reserved_bytes": 68_000_000_000,
        },
    }


def test_full_training_config_selects_all_fa4_and_memory_safe_contract():
    cfg = {
        "fa4_training_backend": "native",
        "base_model": "google/gemma-4-12B-it",
        "revision_of_model": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "max_steps": 100,
        "fa4_training_expected_steps": 100,
    }
    mutate_full_training_config(cfg)
    assert cfg["attn_implementation"] == "gemma4_fa4_h100_native"
    assert cfg["micro_batch_size"] == 1
    assert cfg["optimizer"] == "adafactor"
    assert cfg["gradient_checkpointing"] is True
    assert cfg["bf16"] is True
    assert cfg["skip_prepare_dataset"] is True


def test_full_training_config_allows_short_explicit_memory_probe():
    cfg = {
        "fa4_training_backend": "sdpa",
        "max_steps": 2,
        "fa4_training_expected_steps": 2,
        "fa4_training_timing_warmup_steps": 0,
    }
    mutate_full_training_config(cfg)
    assert cfg["attn_implementation"] == "sdpa"


def test_full_training_config_requires_matching_training_and_data_seeds():
    with pytest.raises(ValueError, match="seed and data_seed"):
        mutate_full_training_config({"seed": 1729, "data_seed": 31415})


def test_full_training_precision_alignment_converts_only_trainable_storage():
    model = torch.nn.Module()
    model.trainable = torch.nn.Linear(4, 3, dtype=torch.float32)
    model.frozen = torch.nn.Linear(4, 3, dtype=torch.float32)
    model.frozen.requires_grad_(False)

    evidence = align_trainable_parameters_to_bf16(model)

    assert evidence["converted_elements"] == 15
    assert {item["name"] for item in evidence["converted_parameters"]} == {
        "trainable.weight",
        "trainable.bias",
    }
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.trainable.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.frozen.parameters())


def test_full_training_precision_alignment_is_idempotent():
    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)

    evidence = align_trainable_parameters_to_bf16(model)

    assert evidence["converted_elements"] == 0
    assert evidence["converted_parameters"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter", "lora"),
        ("bf16", False),
        ("micro_batch_size", 2),
        ("optimizer", "adamw_torch"),
        ("learning_rate", 0.0),
    ],
)
def test_full_training_config_rejects_contract_drift(field, value):
    with pytest.raises(ValueError):
        mutate_full_training_config({field: value})


def test_full_training_comparison_accepts_normal_100_step_curves():
    result = compare_full_training_reports(
        _report("sdpa"),
        _report("native", loss_delta=0.01, norm_scale=1.05),
    )
    assert result["passed"] is True
    assert result["grad_norm_median_ratio"] == pytest.approx(1.05)


def test_full_training_comparison_rejects_nonfinite_candidate():
    candidate = _report("native")
    candidate["training_metrics"][40]["grad_norm"] = float("nan")
    with pytest.raises(TrainingComparisonError, match="finite"):
        compare_full_training_reports(_report("sdpa"), candidate)


def test_full_training_comparison_rejects_wrong_precision_evidence():
    candidate = _report("native")
    candidate["parameters"] = copy.deepcopy(candidate["parameters"])
    candidate["parameters"]["trainable_dtypes"] = {"torch.float32": 11_000_000_000}
    with pytest.raises(TrainingComparisonError, match="BF16"):
        compare_full_training_reports(_report("sdpa"), candidate)


@pytest.mark.parametrize("field", ["routes", "layer_routes", "native_geometries"])
def test_full_training_comparison_rejects_fa4_evidence_in_sdpa_control(field):
    control = _report("sdpa")
    control[field] = {"fa4_native/local_fixed": 1}

    with pytest.raises(TrainingComparisonError, match="SDPA control"):
        compare_full_training_reports(control, _report("native"))


def test_prospective_training_matrix_accepts_three_close_faster_pairs():
    reports = {
        seed: (
            _report("sdpa", seed=seed, step_ms=4.0),
            _report("native", seed=seed, loss_delta=0.01, norm_scale=1.05, step_ms=1.0),
        )
        for seed in PROSPECTIVE_MATRIX_SEEDS
    }

    result = compare_full_training_matrix(reports)

    assert result["passed"] is True
    assert result["aggregate"]["median_step_speedup"] == pytest.approx(4.0)
    assert set(result["seed_results"]) == {str(seed) for seed in PROSPECTIVE_MATRIX_SEEDS}


def test_prospective_training_matrix_rejects_one_bad_seed():
    reports = {
        seed: (
            _report("sdpa", seed=seed, step_ms=4.0),
            _report(
                "native",
                seed=seed,
                loss_delta=0.5 if seed == PROSPECTIVE_MATRIX_SEEDS[-1] else 0.01,
                step_ms=1.0,
            ),
        )
        for seed in PROSPECTIVE_MATRIX_SEEDS
    }

    result = compare_full_training_matrix(reports)

    assert result["passed"] is False
    assert result["seed_results"][str(PROSPECTIVE_MATRIX_SEEDS[-1])]["passed"] is False
    assert "one or more seed pairs failed" in result["reasons"][0]


def test_prospective_training_matrix_requires_exact_predeclared_seeds():
    with pytest.raises(TrainingComparisonError, match="matrix seeds"):
        compare_full_training_matrix({})
