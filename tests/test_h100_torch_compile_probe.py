from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_torch_compile",
    ROOT / "scripts/probe_h100_torch_compile.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _comparison(*, max_abs: float = 0.0, mean_abs: float = 0.0):
    return {
        "exact": True,
        "bitwise": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def _backend_result(lengths: tuple[int, ...], family: str) -> dict:
    return {
        "status": "passed",
        "fullgraph": True,
        "backend_delegate": "stock_eager",
        "shape_policy": {
            "torch_compile_dynamic": True,
            "mark_dynamic_sequence_dim": True,
            "backed_size_oblivious": True,
            "is_pytorch_default": False,
        },
        "graph_count": 1,
        "graph_break_count": 0,
        "custom_op_node": True,
        "weight_snapshot_node": False,
        "graph_nodes": [
            {
                "nodes": [
                    {
                        "op": "call_function",
                        "target": f"gemma4_fa4.h100_{family}_layer_fwd.default",
                    }
                ]
            }
        ],
        "cases": [
            {
                "seqlen": length,
                **_comparison(),
                "output_shape": [1, length, 5376],
                "output_dtype": "torch.bfloat16",
            }
            for length in lengths
        ],
        "nondefault_stream": _comparison(),
        "reset_positions": {"status": "rejected"},
        "cache": {"status": "rejected"},
    }


def _public_dynamic_result(lengths: tuple[int, ...], family: str) -> dict:
    expected_graphs = 2 if 1 in lengths and any(length > 1 for length in lengths) else 1
    return {
        "status": "passed",
        "fullgraph": True,
        "backend_delegate": "stock_eager",
        "shape_policy": {
            "torch_compile_dynamic": True,
            "mark_dynamic_sequence_dim": False,
            "backed_size_oblivious": False,
            "is_pytorch_default": True,
        },
        "graph_count": expected_graphs,
        "expected_graph_count": expected_graphs,
        "graph_classes": [
            *([{"class": "S1", "lengths": [1]}] if 1 in lengths else []),
            *(
                [{"class": "S>1", "lengths": [length for length in lengths if length > 1]}]
                if any(length > 1 for length in lengths)
                else []
            ),
        ],
        "bounded_exactly_s1_or_gt1": True,
        "one_graph_requirement_met": expected_graphs == 1,
        "graph_break_count": 0,
        "custom_op_node": True,
        "weight_snapshot_node": False,
        "graph_nodes": [
            {
                "nodes": [
                    {
                        "op": "call_function",
                        "target": f"gemma4_fa4.h100_{family}_layer_fwd.default",
                    }
                ]
            }
        ],
        "cases": [
            {
                "seqlen": length,
                **_comparison(),
            }
            for length in lengths
        ],
    }


def _family_result(lengths: tuple[int, ...], family: str, backends: tuple[str, ...]) -> dict:
    layer_idx = PROBE.FAMILY_LAYERS[family]
    spec = PROBE.GEMMA4_31B.spec_for_layer(layer_idx)
    return {
        "status": "passed",
        "layer_idx": layer_idx,
        "layer_type": spec.kind,
        "hidden_size": PROBE.GEMMA4_31B.hidden_size,
        "geometry": {
            "q_heads": spec.num_q_heads,
            "kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim_qk,
            "scale": 1.0,
        },
        "fake_tensor": {"status": "passed", "real_body_entered": False},
        "opcheck": {"status": "passed", "tests": {"test_schema": "SUCCESS"}},
        "fa4_forward_application_keys": {
            "before": [],
            "warmed": [f"{family}-digest"],
            "after": [f"{family}-digest"],
            "added_family_class": [f"{family}-digest"],
            "new_class_count": 1,
            "reused_across_compiler_matrix": True,
        },
        "direct_reference": [
            {
                "seqlen": length,
                "projection_transport_bitwise": True,
                "whole_layer_output_bitwise": True,
                "whole_layer_lse_bitwise": True,
                "dormant_weight_output_bitwise": True,
                "dormant_weight_lse_bitwise": True,
            }
            for length in lengths
        ],
        "backends": {
            backend: {
                "public_dynamic_default": _public_dynamic_result(lengths, family),
                "scoped_backed_size_oblivious": _backend_result(lengths, family),
            }
            for backend in backends
        },
    }


def _report(
    *,
    lengths: tuple[int, ...] = PROBE.DEFAULT_LENGTHS,
    families: tuple[str, ...] = ("local", "global"),
    backends: tuple[str, ...] = PROBE.BACKENDS,
) -> dict:
    return {
        "schema_version": PROBE.SCHEMA_VERSION,
        "experiment": PROBE.EXPERIMENT,
        "status": "passed",
        "request": {
            "families": list(families),
            "backends": list(backends),
            "lengths": list(lengths),
            "declared_matrix_complete": lengths == PROBE.DEFAULT_LENGTHS,
            "shape_policies": [
                "pytorch_default_dynamic",
                "scoped_backed_size_oblivious",
            ],
            "seed": 17,
        },
        "environment": {
            "torch": PROBE.PINNED_TORCH_VERSION,
            "device_name": "NVIDIA H100 80GB HBM3",
            "capability": [9, 0],
            "caches": {
                "fa4": {"path": "/tmp/fa4", "file_count": 2, "total_bytes": 1024},
                **(
                    {
                        "inductor": {
                            "path": "/tmp/inductor",
                            "file_count": 3,
                            "total_bytes": 2048,
                        }
                    }
                    if "inductor" in backends
                    else {}
                ),
            },
        },
        "families": {family: _family_result(lengths, family, backends) for family in families},
    }


def _cache_snapshot(family: str, cache_type: str, state: str) -> dict:
    layer_idx = PROBE.FAMILY_LAYERS[family]
    length = 1 if state == "nonempty" else 0
    lengths = [0] * 60
    lengths[layer_idx] = length
    return {
        "cache_type": cache_type,
        "cache_identity": 1234,
        "layer_idx": layer_idx,
        "layer_count": 60,
        "logical_lengths": lengths,
        "initialized_layers": [layer_idx] if state == "nonempty" else [],
        "target_layer": {
            "type": f"{cache_type}Layer",
            "is_initialized": state == "nonempty",
            "logical_length": length,
            "counters": {"cumulative_length": length},
            "tensors": {},
            "storage_identity": {},
            "content_sha256": "0" * 64,
            "allocation": {
                "allocated_tensor_names": [],
                "unique_storage_count": 0,
                "storage_nbytes": 0,
            },
        },
    }


def _negative_case(family: str, backend: str, cache_type: str, state: str) -> dict:
    snapshot = _cache_snapshot(family, cache_type, state)
    zero = {"layer_entry": 0, "cache_update": 0, "custom_op_entry": 0}
    return {
        "status": "rejected",
        "family": family,
        "layer_idx": PROBE.FAMILY_LAYERS[family],
        "backend": backend,
        "cache_type": cache_type,
        "cache_state": state,
        "rejection_class": "exp0018_mask_cache_rejection_dynamo_wrapper",
        "error_type": "Unsupported",
        "error": ("Observed exception: UserDefinedExceptionObjectVariable(UnsupportedH100Path)"),
        "cache_unchanged": True,
        "cache_before": snapshot,
        "cache_after": copy.deepcopy(snapshot),
        "entry_counters_before": zero,
        "entry_counters_after": dict(zero),
        "compiler_backend_graph_count": 0,
        "dynamo_graph_break_count": 0,
    }


def _negative_cache_report(
    *,
    families: tuple[str, ...] = ("local", "global"),
    backends: tuple[str, ...] = PROBE.BACKENDS,
) -> dict:
    cache_types = ("DynamicCache", "StaticCache")
    cache_states = ("empty", "nonempty")
    results = [
        _negative_case(family, backend, cache_type, state)
        for family in families
        for backend in backends
        for cache_type in cache_types
        for state in cache_states
    ]
    return {
        "schema_version": PROBE.SCHEMA_VERSION,
        "experiment": PROBE.EXPERIMENT,
        "status": "passed",
        "request": {
            "mode": "negative_cache_object",
            "families": list(families),
            "backends": list(backends),
            "cache_types": list(cache_types),
            "cache_states": list(cache_states),
            "seed": 17,
        },
        "environment": {
            "torch": PROBE.PINNED_TORCH_VERSION,
            "device_name": "NVIDIA H100 80GB HBM3",
            "capability": [9, 0],
            "caches": {
                "fa4": {"path": "/tmp/fa4", "file_count": 0, "total_bytes": 0},
                **(
                    {
                        "inductor": {
                            "path": "/tmp/inductor",
                            "file_count": 0,
                            "total_bytes": 0,
                        }
                    }
                    if "inductor" in backends
                    else {}
                ),
            },
        },
        "summary": {
            "case_count": len(results),
            "rejected_before_entry_count": len(results),
        },
        "results": results,
    }


def _raw_comparison(*, bitwise: bool, max_abs: float, mean_abs: float) -> dict:
    return {
        "bitwise": bitwise,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "shape": [1, 1023, 5376],
        "dtype": "torch.bfloat16",
    }


def _localization_report() -> dict:
    exact = _raw_comparison(bitwise=True, max_abs=0.0, mean_abs=0.0)
    drift = _raw_comparison(bitwise=False, max_abs=0.0703125, mean_abs=0.0074)
    return {
        "schema_version": PROBE.SCHEMA_VERSION,
        "experiment": PROBE.EXPERIMENT,
        "status": "passed",
        "request": {
            "mode": "localize_outer_drift",
            "family": "local",
            "backend": "inductor",
            "seqlen": 1023,
            "runtime_seed": 17017,
            "input_seed": 19020,
        },
        "environment": {
            "torch": PROBE.PINNED_TORCH_VERSION,
            "device_name": "NVIDIA H100 80GB HBM3",
            "capability": [9, 0],
            "caches": {
                "fa4": {"path": "/tmp/fa4", "file_count": 2, "total_bytes": 1024},
                "inductor": {
                    "path": "/tmp/inductor",
                    "file_count": 3,
                    "total_bytes": 2048,
                },
            },
        },
        "fa4_forward_application_keys": {
            "before": [],
            "after": ["local-digest"],
            "added": ["local-digest"],
            "new_class_count": 1,
        },
        "localization": {
            "qkv": {
                "graph_count": 1,
                "graph_break_count": 0,
                "distinct_storage": True,
                "components": {"q": drift, "k": exact, "v": exact},
            },
            "prepared_opaque": {
                "graph_count": 1,
                "graph_break_count": 0,
                "custom_op_node": True,
                "output": exact,
                "lse": exact,
                "reference": {
                    "output": {"passed": True, "max_abs": 0.01, "mean_abs": 0.001},
                    "lse": {"passed": True, "max_abs": 0.02, "mean_abs": 0.002},
                },
            },
            "output_projection": {
                "graph_count": 1,
                "graph_break_count": 0,
                "output": drift,
            },
            "hybrid_compiled_qkv": {"output": drift},
            "whole_layer": {
                "graph_count": 1,
                "graph_break_count": 0,
                "custom_op_node": True,
                "output": {
                    **drift,
                    "within_exp0018_frozen_tolerance": False,
                },
            },
            "prepared_opaque_bitwise": True,
            "outer_drift_observed": True,
            "exp0018_failure_reproduced": True,
            "hypothesis_supported": True,
        },
    }


def test_declared_matrix_and_actual_layer_indices_are_locked():
    assert PROBE.EXPERIMENT == "EXP-0021"
    assert PROBE.DEFAULT_LENGTHS == (1, 32, 33, 1023, 1024)
    assert PROBE.FAMILY_LAYERS == {"local": 0, "global": 5}
    assert PROBE.BACKENDS == ("eager", "inductor")
    assert PROBE.GEMMA4_31B.hidden_size == 5376
    assert PROBE.FULL_LAYER_OUTPUT_ATOL == 0.0625
    assert PROBE.FULL_LAYER_OUTPUT_RTOL == 0.02


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", (1,)),
        ("1,32,33,1023,1024", PROBE.DEFAULT_LENGTHS),
        (" 32, 1024 ", (32, 1024)),
    ],
)
def test_length_parser_accepts_only_bounded_unique_lengths(raw, expected):
    assert PROBE._parse_lengths(raw) == expected


@pytest.mark.parametrize("raw", ["", "x", "0", "1025", "32,32"])
def test_length_parser_rejects_invalid_matrices(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        PROBE._parse_lengths(raw)


def test_cli_defaults_to_complete_two_family_two_backend_matrix():
    args = PROBE._build_parser().parse_args([])

    assert args.family == "all"
    assert args.backend == "all"
    assert args.lengths == PROBE.DEFAULT_LENGTHS
    assert not args.negative_cache_object
    assert not args.localize_outer_drift


def test_localization_and_cache_negative_modes_are_mutually_exclusive():
    parser = PROBE._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--negative-cache-object", "--localize-outer-drift"])


def test_outer_drift_localization_cli_mode_routes_without_normal_matrix(monkeypatch):
    expected = _localization_report()
    calls = []

    def run(args):
        calls.append(args)
        return expected

    monkeypatch.setattr(PROBE, "_run_outer_drift_localization", run)
    args = PROBE._build_parser().parse_args(["--localize-outer-drift"])

    assert PROBE._run_probe(args) is expected
    assert calls == [args]


def test_negative_cache_object_cli_mode_is_explicit_and_routes_without_normal_matrix(
    monkeypatch,
):
    expected = _negative_cache_report()
    calls = []

    def run(args):
        calls.append(args)
        return expected

    monkeypatch.setattr(PROBE, "_run_negative_cache_object_probe", run)
    args = PROBE._build_parser().parse_args(["--negative-cache-object", "--seed", "23"])

    assert PROBE._run_probe(args) is expected
    assert calls == [args]


def test_cache_object_callable_passes_same_real_cache_to_mask_and_layer():
    cache = object()
    calls = []
    counters = {"layer_entry": 0}

    def mask_builder(_config, **kwargs):
        calls.append(("mask", kwargs["past_key_values"]))
        return "mask-plan"

    class Layer:
        def __call__(self, _hidden, _position_embeddings, mask, _shared, **kwargs):
            calls.append(("layer", kwargs["past_key_values"], mask))
            return "output", None

    runtime = SimpleNamespace(
        config=object(),
        layer_idx=0,
        mask_builder=mask_builder,
        layer=Layer(),
    )

    output = PROBE._cache_object_callable(runtime, cache, counters)(
        "hidden", "cos", "sin", "positions"
    )

    assert output == "output"
    assert calls == [("mask", cache), ("layer", cache, "mask-plan")]
    assert counters == {"layer_entry": 1}


@pytest.mark.parametrize("family", ["local", "global"])
def test_custom_op_node_detection_requires_project_namespace_and_family(family):
    fragment = f"h100_{family}_layer_fwd"
    graphs = _backend_result((33,), family)["graph_nodes"]

    assert PROBE._contains_custom_op(graphs, fragment)
    assert not PROBE._contains_custom_op(graphs, "unrelated")
    graphs[0]["nodes"][0]["target"] = f"other_project.{fragment}.default"
    assert not PROBE._contains_custom_op(graphs, fragment)


def test_graph_break_counter_sums_all_recorded_reasons(monkeypatch):
    monkeypatch.setattr(
        PROBE.torch._dynamo.utils,
        "counters",
        {"graph_break": {"reason-a": 2, "reason-b": 3}},
    )

    assert PROBE._graph_break_count() == 5


def test_fresh_cache_preflight_creates_and_inventories_both_isolated_dirs(monkeypatch, tmp_path):
    fa4 = tmp_path / "fa4"
    inductor = tmp_path / "inductor"
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(fa4))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inductor))

    paths = PROBE._prepare_fresh_cache_dirs(("eager", "inductor"))
    (fa4 / "object.bin").write_bytes(b"abc")

    assert paths == {"fa4": fa4.resolve(), "inductor": inductor.resolve()}
    assert PROBE._cache_inventory(fa4) == {
        "path": str(fa4.resolve()),
        "file_count": 1,
        "total_bytes": 3,
    }


def test_fresh_cache_preflight_rejects_nonempty_or_missing_dirs(monkeypatch, tmp_path):
    fa4 = tmp_path / "fa4"
    fa4.mkdir()
    (fa4 / "existing").write_text("occupied")
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(fa4))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

    with pytest.raises(RuntimeError, match="fresh and empty"):
        PROBE._prepare_fresh_cache_dirs(("eager",))

    (fa4 / "existing").unlink()
    with pytest.raises(RuntimeError, match="inductor cache"):
        PROBE._prepare_fresh_cache_dirs(("inductor",))


def test_application_key_digest_snapshot_is_order_independent():
    first = ("local", 256)
    second = ("global", 512)

    assert PROBE._application_key_digests((first, second)) == PROBE._application_key_digests(
        (second, first)
    )
    assert len(PROBE._application_key_digests((first, second))) == 2


@pytest.mark.parametrize(
    "message",
    [
        "EXP-0018 FakeTensor/torch.compile does not accept a cache",
        (
            "Observed exception: developer context "
            "UserDefinedExceptionObjectVariable(UnsupportedH100Path)"
        ),
    ],
)
def test_cache_marker_rejection_classifier_accepts_declared_failures(message):
    assert PROBE._is_expected_cache_marker_rejection(message)


@pytest.mark.parametrize(
    "message",
    [
        "ConstraintViolationError: unrelated shape",
        "Inductor code cache write failed",
        "EXP-0017 unrelated compiler assertion",
        "cache_position is unsupported",
    ],
)
def test_cache_marker_rejection_classifier_rejects_unrelated_compiler_errors(message):
    assert not PROBE._is_expected_cache_marker_rejection(message)


@pytest.mark.parametrize(
    ("error_type", "message", "expected"),
    [
        (
            "UnsupportedH100Path",
            "EXP-0018 fullgraph routing does not accept a cache; rejection occurs at "
            "mask construction before Cache.update",
            "exp0018_mask_cache_rejection",
        ),
        (
            "Unsupported",
            "Observed exception: UserDefinedExceptionObjectVariable(UnsupportedH100Path)",
            "exp0018_mask_cache_rejection_dynamo_wrapper",
        ),
    ],
)
def test_real_cache_rejection_classifier_accepts_only_declared_boundary(
    error_type, message, expected
):
    assert PROBE._classify_cache_object_rejection(error_type, message) == expected


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        ("ConstraintViolationError", "unrelated shape"),
        ("RuntimeError", "Inductor code cache write failed"),
        ("UnsupportedH100Path", "EXP-0018 unrelated compiler assertion"),
        ("Unsupported", "cache_position is unsupported"),
        ("Unsupported", "Observed exception: RuntimeError"),
    ],
)
def test_real_cache_rejection_classifier_rejects_unrelated_errors(error_type, message):
    assert PROBE._classify_cache_object_rejection(error_type, message) is None


def test_full_layer_comparison_requires_and_records_bitwise_equality():
    eager = PROBE.torch.zeros(4, dtype=PROBE.torch.float32)
    compiled = eager.clone()

    comparison = PROBE._full_layer_comparison(compiled, eager, label="unit")

    assert comparison == {
        "exact": True,
        "bitwise": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
    }


def test_full_layer_comparison_rejects_any_nonbitwise_drift():
    eager = PROBE.torch.zeros(1, dtype=PROBE.torch.float32)
    compiled = PROBE.torch.tensor([0.0001], dtype=PROBE.torch.float32)

    with pytest.raises(AssertionError, match="EXP-0021 bitwise whole-layer equality"):
        PROBE._full_layer_comparison(compiled, eager, label="unit")


def test_raw_tensor_comparison_records_drift_without_applying_a_tolerance():
    expected = PROBE.torch.zeros(4, dtype=PROBE.torch.bfloat16)
    candidate = expected.clone()
    candidate[0] = 0.0703125

    comparison = PROBE._tensor_comparison(candidate, expected)

    assert comparison == {
        "bitwise": False,
        "max_abs": 0.0703125,
        "mean_abs": 0.017578125,
        "shape": [4],
        "dtype": "torch.bfloat16",
    }


def test_cache_snapshot_records_logical_storage_content_and_allocation():
    class Layer:
        is_initialized = True

        def __init__(self):
            self.cumulative_length = PROBE.torch.tensor(1)
            self.keys = PROBE.torch.tensor([[1.0, 2.0]])
            self.values = PROBE.torch.tensor([[3.0, 4.0]])

    class Cache:
        def __init__(self):
            self.layers = [Layer()]

        def get_seq_length(self, _layer_idx):
            return 1

    cache = Cache()
    before = PROBE._cache_state_snapshot(cache, 0)
    cache.layers[0].keys[0, 0] = 9.0
    after = PROBE._cache_state_snapshot(cache, 0)

    assert before["logical_lengths"] == [1]
    assert before["target_layer"]["storage_identity"] == after["target_layer"]["storage_identity"]
    assert before["target_layer"]["content_sha256"] != after["target_layer"]["content_sha256"]
    assert before["target_layer"]["allocation"]["unique_storage_count"] == 3


def test_report_schema_accepts_the_complete_machine_readable_evidence():
    PROBE._validate_report(_report())


def test_report_schema_accepts_only_confirmed_outer_drift_localization():
    report = _localization_report()
    PROBE._validate_report(report)

    report["localization"]["prepared_opaque"]["output"]["bitwise"] = False
    with pytest.raises(ValueError, match="localization"):
        PROBE._validate_report(report)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("localization", "hypothesis_supported"), False),
        (("localization", "outer_drift_observed"), False),
        (("localization", "exp0018_failure_reproduced"), False),
        (("localization", "prepared_opaque", "reference", "output", "passed"), False),
        (("localization", "whole_layer", "output", "within_exp0018_frozen_tolerance"), True),
        (("localization", "qkv", "graph_break_count"), 1),
        (("localization", "qkv", "graph_count"), 3),
    ],
)
def test_report_schema_rejects_weakened_outer_drift_evidence(path, value):
    report = _localization_report()
    cursor = report
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(ValueError, match="localization"):
        PROBE._validate_report(report)


def test_report_schema_accepts_only_complete_unmutated_cache_matrix():
    report = _negative_cache_report()
    PROBE._validate_report(report)

    assert len(report["results"]) == 16
    report["results"][0]["entry_counters_after"]["layer_entry"] = 1
    with pytest.raises(ValueError, match="pre-entry immutable rejection"):
        PROBE._validate_report(report)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("families", "local", "hidden_size"), 64),
        (("families", "local", "fake_tensor", "real_body_entered"), True),
        (("families", "local", "opcheck", "status"), "failed"),
        (
            ("families", "local", "fa4_forward_application_keys", "new_class_count"),
            2,
        ),
        (
            ("families", "local", "backends", "eager", "public_dynamic_default", "graph_count"),
            1,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "shape_policy",
                "is_pytorch_default",
            ),
            True,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "graph_count",
            ),
            2,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "graph_break_count",
            ),
            1,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "custom_op_node",
            ),
            False,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "cases",
                0,
                "exact",
            ),
            False,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "public_dynamic_default",
                "cases",
                0,
                "bitwise",
            ),
            False,
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "reset_positions",
                "status",
            ),
            "admitted",
        ),
        (
            (
                "families",
                "local",
                "backends",
                "eager",
                "scoped_backed_size_oblivious",
                "cache",
                "status",
            ),
            "admitted",
        ),
    ],
)
def test_report_schema_rejects_missing_or_weakened_evidence(path, value):
    report = _report()
    cursor = report
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(ValueError):
        PROBE._validate_report(report)


def test_cli_writes_and_prints_the_same_valid_json(monkeypatch, tmp_path, capsys):
    report = _report(lengths=(1,), families=("local",), backends=("eager",))
    monkeypatch.setattr(PROBE, "_run_probe", lambda _args: copy.deepcopy(report))
    output = tmp_path / "exp0018.json"

    result = PROBE.main(
        [
            "--family",
            "local",
            "--backend",
            "eager",
            "--lengths",
            "1",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written == report


def test_cli_fails_closed_off_h100_without_importing_transformers(monkeypatch, capsys):
    monkeypatch.setattr(PROBE, "PINNED_TORCH_VERSION", PROBE.torch.__version__)
    monkeypatch.setattr(PROBE.h100_torch_ops, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(PROBE.torch.cuda, "is_available", lambda: False)

    result = PROBE.main(["--family", "local", "--backend", "eager", "--lengths", "1"])

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == PROBE.SCHEMA_VERSION
    assert payload["experiment"] == PROBE.EXPERIMENT
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert "H100" in payload["error"]


def test_failure_json_is_written_when_runtime_probe_raises(monkeypatch, tmp_path, capsys):
    def fail(_args):
        raise AssertionError("one-graph requirement unmet")

    monkeypatch.setattr(PROBE, "_run_probe", fail)
    output = tmp_path / "failure.json"

    result = PROBE.main(["--output", str(output)])

    assert result == 1
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == printed
    assert printed["status"] == "failed"
    assert printed["error_type"] == "AssertionError"
    assert printed["request"]["lengths"] == list(PROBE.DEFAULT_LENGTHS)


def test_dynamic_cache_admission_emits_failed_machine_json(monkeypatch, capsys):
    def admit(_args):
        raise AssertionError(
            "local/eager/DynamicCache/empty admitted a compiled cache request: "
            "output_shape=(1, 1, 5376), cache_unchanged=False"
        )

    monkeypatch.setattr(PROBE, "_run_probe", admit)

    result = PROBE.main(["--negative-cache-object"])

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["request"]["negative_cache_object"] is True
    assert "DynamicCache/empty admitted a compiled cache request" in payload["error"]
    assert "cache_unchanged=False" in payload["error"]
