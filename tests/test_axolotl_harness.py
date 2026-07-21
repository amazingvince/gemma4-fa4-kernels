from __future__ import annotations

import pytest
import torch

from gemma4_fa4.axolotl_harness import (
    HarnessComparisonError,
    compare_reports,
    initialize_lora_parameters,
    mutate_axolotl_config,
    summarize_timings,
)


def _report(backend: str, *, timings=None, loss_delta=0.0, sketch_delta=0.0, routes=None):
    timings = timings or [11.0, 10.0, 9.0, 10.5, 9.5]
    base_sketch = [1.0, -2.0, 0.5, 4.0]
    return {
        "schema_version": 2,
        "status": "complete",
        "backend": backend,
        "workload": {
            "model_id": "google/gemma-4-12B-it",
            "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "sequence_len": 1024,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 8,
            "warmup_steps": 3,
            "seed": 3600,
            "dataset_sha256": "a" * 64,
        },
        "measured_step_ms": timings,
        "losses": [
            2.0 + loss_delta,
            1.9 + loss_delta,
            1.8 + loss_delta,
            1.7 + loss_delta,
            1.6 + loss_delta,
        ],
        "gradient_probe": {
            "parameter_names": ["lora_A", "lora_B"],
            "values": [value + sketch_delta for value in base_sketch],
        },
        "lora_initialization": {
            "scheme": "name-seeded-kaiming-uniform-a-zero-b-v1",
            "seed": 3600,
            "sha256": "b" * 64,
            "parameter_count": 2,
            "value_count": 8,
        },
        "routes": routes or {},
        "layer_routes": {},
    }


def _valid_layer_routes():
    return {
        str(layer_idx): {
            (
                "fa4_12b_compat/fa4_global_fixed"
                if (layer_idx + 1) % 6 == 0
                else "fa4_12b_compat/fa4_local_fixed"
            ): 8
        }
        for layer_idx in range(48)
    }


def _valid_global_native_routes():
    return {
        str(layer_idx): {
            ("fa4_native/global_fixed" if (layer_idx + 1) % 6 == 0 else "fa2_local/fixed"): 8
        }
        for layer_idx in range(48)
    }


def test_timing_summary_uses_linear_percentiles():
    summary = summarize_timings([1.0, 2.0, 3.0, 4.0, 9.0])
    assert summary == {"count": 5, "median_ms": 3.0, "p25_ms": 2.0, "p75_ms": 4.0, "iqr_ms": 2.0}


@pytest.mark.parametrize(
    ("backend", "expected_attn", "expected_hybrid"),
    [
        ("native", "gemma4_fa4_h100_native", False),
        ("global_native", "gemma4_fa4_h100_global_native", False),
        ("project_12b_compat", "gemma4_fa4_h100_12b_compat", False),
        ("hybrid", "flash_attention_2", True),
        ("sdpa", "sdpa", False),
    ],
)
def test_config_mutation_selects_exactly_one_backend(backend, expected_attn, expected_hybrid):
    cfg = {"fa4_harness_backend": backend}
    mutate_axolotl_config(cfg)
    assert cfg["attn_implementation"] == expected_attn
    assert cfg["gemma4_hybrid_attn_impl"] is expected_hybrid


@pytest.mark.parametrize("sequence_len", [512, 1024, 2048, 4096])
def test_config_mutation_accepts_validated_sequence_lengths(sequence_len):
    cfg = {"fa4_harness_backend": "native", "sequence_len": sequence_len}
    mutate_axolotl_config(cfg)
    assert cfg["sequence_len"] == sequence_len


def test_config_mutation_rejects_unvalidated_sequence_length():
    with pytest.raises(ValueError, match="512, 1024, 2048, or 4096"):
        mutate_axolotl_config({"fa4_harness_backend": "native", "sequence_len": 8192})


def test_config_mutation_rejects_semantic_benchmark_drift():
    cfg = {
        "fa4_harness_backend": "hybrid",
        "learning_rate": 1e-4,
        "sample_packing": "yes",
    }
    with pytest.raises(ValueError, match="learning_rate.*0"):
        mutate_axolotl_config(cfg)


def test_lora_initialization_is_name_seeded_and_byte_reproducible():
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})

    first = Adapter()
    torch.manual_seed(9999)
    second = Adapter()

    first_fingerprint = initialize_lora_parameters(first, seed=3600)
    second_fingerprint = initialize_lora_parameters(second, seed=3600)

    assert first_fingerprint == second_fingerprint
    torch.testing.assert_close(
        first.lora_A["default"].weight,
        second.lora_A["default"].weight,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(first.lora_A["default"].weight).item() > 0
    assert torch.count_nonzero(first.lora_B["default"].weight).item() == 0


def test_comparator_accepts_equivalent_project_routes_and_reports_speedup():
    baseline = _report("hybrid", timings=[20, 21, 19, 20, 20])
    candidate = _report(
        "project_12b_compat",
        timings=[10, 11, 9, 10, 10],
        loss_delta=2e-4,
        sketch_delta=1e-4,
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    result = compare_reports(baseline, candidate)
    assert result["passed"] is True
    assert result["speedup"] == pytest.approx(2.0)
    assert result["candidate_timing"]["median_ms"] == 10.0


def test_comparator_accepts_global_native_route_mix():
    baseline = _report("hybrid", timings=[20, 21, 19, 20, 20])
    candidate = _report(
        "global_native",
        timings=[18, 19, 17, 18, 18],
        routes={"fa2_local/fixed": 320, "fa4_native/global_fixed": 64},
    )
    candidate["layer_routes"] = _valid_global_native_routes()
    result = compare_reports(baseline, candidate)
    assert result["passed"] is True
    assert result["candidate_backend"] == "global_native"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("losses", [2.0, 1.9, float("nan"), 1.7, 1.6]),
        ("measured_step_ms", [10.0, 10.0, 10.0, float("inf"), 10.0]),
    ],
)
def test_comparator_rejects_nonfinite_measurements(field, invalid_value):
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    candidate[field] = invalid_value

    with pytest.raises(HarnessComparisonError, match="finite"):
        compare_reports(baseline, candidate)


def test_comparator_rejects_nonfinite_gradient_values():
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    candidate["gradient_probe"]["values"][2] = float("inf")

    with pytest.raises(HarnessComparisonError, match="finite"):
        compare_reports(baseline, candidate)


@pytest.mark.parametrize("field", ["losses", "measured_step_ms"])
def test_comparator_rejects_short_measurement_sequences(field):
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    candidate[field] = candidate[field][:-1]

    with pytest.raises(HarnessComparisonError, match="exactly 5"):
        compare_reports(baseline, candidate)


def test_comparator_requires_every_layer_route_on_every_step():
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    candidate["layer_routes"]["0"]["fa4_12b_compat/fa4_local_fixed"] = 1

    with pytest.raises(HarnessComparisonError, match="8 calls"):
        compare_reports(baseline, candidate)


def test_comparator_requires_initialization_fingerprints_in_both_reports():
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    baseline.pop("lora_initialization")
    candidate.pop("lora_initialization")

    with pytest.raises(HarnessComparisonError, match="initialization"):
        compare_reports(baseline, candidate)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report: report["workload"].update(sequence_len=512), "workload"),
        (lambda report: report.update(losses=[3.0] * 5), "loss"),
        (
            lambda report: report["gradient_probe"].update(values=[10.0, 10.0, 10.0, 10.0]),
            "gradient",
        ),
        (lambda report: report.update(routes={}), "route"),
        (
            lambda report: report["lora_initialization"].update(sha256="c" * 64),
            "initialization",
        ),
        (
            lambda report: report.update(routes={"fa4_12b_compat/flex_attention": 240}),
            "fallback",
        ),
    ],
)
def test_comparator_rejects_invalid_candidate(mutator, message):
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 320,
            "fa4_12b_compat/fa4_global_fixed": 64,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    mutator(candidate)
    with pytest.raises(HarnessComparisonError, match=message):
        compare_reports(baseline, candidate)
