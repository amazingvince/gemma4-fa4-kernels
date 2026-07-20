from __future__ import annotations

import pytest

from gemma4_fa4.axolotl_harness import (
    HarnessComparisonError,
    compare_reports,
    mutate_axolotl_config,
    summarize_timings,
)


def _report(backend: str, *, timings=None, loss_delta=0.0, sketch_delta=0.0, routes=None):
    timings = timings or [11.0, 10.0, 9.0, 10.5, 9.5]
    base_sketch = [1.0, -2.0, 0.5, 4.0]
    return {
        "schema_version": 1,
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
        "losses": [2.0 + loss_delta, 1.9 + loss_delta, 1.8 + loss_delta],
        "gradient_probe": {
            "parameter_names": ["lora_A", "lora_B"],
            "values": [value + sketch_delta for value in base_sketch],
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
            ): 5
        }
        for layer_idx in range(48)
    }


def test_timing_summary_uses_linear_percentiles():
    summary = summarize_timings([1.0, 2.0, 3.0, 4.0, 9.0])
    assert summary == {"count": 5, "median_ms": 3.0, "p25_ms": 2.0, "p75_ms": 4.0, "iqr_ms": 2.0}


@pytest.mark.parametrize(
    ("backend", "expected_attn", "expected_hybrid"),
    [
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


def test_config_mutation_rejects_semantic_benchmark_drift():
    cfg = {
        "fa4_harness_backend": "hybrid",
        "learning_rate": 1e-4,
        "sample_packing": True,
    }
    with pytest.raises(ValueError, match="learning_rate.*0"):
        mutate_axolotl_config(cfg)


def test_comparator_accepts_equivalent_project_routes_and_reports_speedup():
    baseline = _report("hybrid", timings=[20, 21, 19, 20, 20])
    candidate = _report(
        "project_12b_compat",
        timings=[10, 11, 9, 10, 10],
        loss_delta=2e-4,
        sketch_delta=1e-4,
        routes={
            "fa4_12b_compat/fa4_local_fixed": 200,
            "fa4_12b_compat/fa4_global_fixed": 40,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    result = compare_reports(baseline, candidate)
    assert result["passed"] is True
    assert result["speedup"] == pytest.approx(2.0)
    assert result["candidate_timing"]["median_ms"] == 10.0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report: report["workload"].update(sequence_len=512), "workload"),
        (lambda report: report.update(losses=[3.0, 3.0, 3.0]), "loss"),
        (
            lambda report: report["gradient_probe"].update(values=[10.0, 10.0, 10.0, 10.0]),
            "gradient",
        ),
        (lambda report: report.update(routes={}), "route"),
        (
            lambda report: report.update(
                routes={"fa4_12b_compat/flex_attention": 240}
            ),
            "fallback",
        ),
    ],
)
def test_comparator_rejects_invalid_candidate(mutator, message):
    baseline = _report("hybrid")
    candidate = _report(
        "project_12b_compat",
        routes={
            "fa4_12b_compat/fa4_local_fixed": 200,
            "fa4_12b_compat/fa4_global_fixed": 40,
        },
    )
    candidate["layer_routes"] = _valid_layer_routes()
    mutator(candidate)
    with pytest.raises(HarnessComparisonError, match=message):
        compare_reports(baseline, candidate)
