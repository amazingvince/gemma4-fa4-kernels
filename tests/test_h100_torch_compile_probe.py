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
        "graph_nodes": [
            {
                "nodes": [
                    {
                        "op": "call_function",
                        "target": f"gemma4_fa4.h100_{family}_fwd.default",
                    }
                ]
            }
        ],
        "cases": [
            {
                "seqlen": length,
                "bitwise_eager_compiled": True,
                "output_shape": [1, length, 32, 256 if family == "local" else 512],
                "output_dtype": "torch.bfloat16",
            }
            for length in lengths
        ],
        "nondefault_stream_bitwise": True,
        "reset_positions": {"status": "rejected"},
        "cache": {"status": "rejected"},
    }


def _public_dynamic_result(lengths: tuple[int, ...], family: str) -> dict:
    expected_graphs = 2 if 1 in lengths and any(length > 1 for length in lengths) else 1
    return {
        "status": "observed",
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
        "one_graph_requirement_met": expected_graphs == 1,
        "graph_break_count": 0,
        "custom_op_node": True,
        "graph_nodes": [
            {
                "nodes": [
                    {
                        "op": "call_function",
                        "target": f"gemma4_fa4.h100_{family}_fwd.default",
                    }
                ]
            }
        ],
        "cases": [{"seqlen": length, "bitwise_eager_compiled": True} for length in lengths],
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
        "direct_reference": [{"seqlen": length} for length in lengths],
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


def _negative_cache_report() -> dict:
    return {
        "schema_version": PROBE.SCHEMA_VERSION,
        "experiment": PROBE.EXPERIMENT,
        "status": "passed",
        "request": {"mode": "negative_cache_object", "seed": 17},
        "environment": {
            "torch": PROBE.PINNED_TORCH_VERSION,
            "device_name": "NVIDIA H100 80GB HBM3",
            "capability": [9, 0],
            "caches": {"fa4": {"path": "/tmp/fa4", "file_count": 0, "total_bytes": 0}},
        },
        "result": {
            "status": "rejected",
            "layer_idx": 0,
            "cache_type": "DynamicCache",
            "cache_length_before": 0,
            "cache_length_after": 0,
            "error_type": "Unsupported",
            "error": "EXP-0017 does not accept a cache",
        },
    }


def test_declared_matrix_and_actual_layer_indices_are_locked():
    assert PROBE.DEFAULT_LENGTHS == (1, 32, 33, 1023, 1024)
    assert PROBE.FAMILY_LAYERS == {"local": 0, "global": 5}
    assert PROBE.BACKENDS == ("eager", "inductor")
    assert PROBE.GEMMA4_31B.hidden_size == 5376


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

    output = PROBE._cache_object_callable(runtime, cache)("hidden", "cos", "sin", "positions")

    assert output == "output"
    assert calls == [("mask", cache), ("layer", cache, "mask-plan")]


@pytest.mark.parametrize("family", ["local", "global"])
def test_custom_op_node_detection_requires_project_namespace_and_family(family):
    fragment = f"h100_{family}_fwd"
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
        "EXP-0017 FakeTensor/torch.compile does not accept a cache",
        (
            "Observed exception: developer context "
            "UserDefinedExceptionObjectVariable(UnsupportedH100Path)"
        ),
    ],
)
def test_cache_rejection_classifier_accepts_direct_and_dynamo_wrapped_failures(message):
    assert PROBE._is_expected_cache_rejection(message)


@pytest.mark.parametrize(
    "message",
    [
        "ConstraintViolationError: unrelated shape",
        "Inductor code cache write failed",
        "EXP-0017 unrelated compiler assertion",
        "cache_position is unsupported",
    ],
)
def test_cache_rejection_classifier_rejects_unrelated_compiler_errors(message):
    assert not PROBE._is_expected_cache_rejection(message)


def test_report_schema_accepts_the_complete_machine_readable_evidence():
    PROBE._validate_report(_report())


def test_report_schema_accepts_only_unmutated_dynamic_cache_rejection():
    report = _negative_cache_report()
    PROBE._validate_report(report)

    report["result"]["cache_length_after"] = 1
    with pytest.raises(ValueError, match="DynamicCache"):
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
                "bitwise_eager_compiled",
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
    output = tmp_path / "exp0017.json"

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
            "actual pinned layer 0 admitted an unsupported DynamicCache object under fullgraph: "
            "output_shape=(1, 1, 5376), cache_length_before=0, cache_length_after=1"
        )

    monkeypatch.setattr(PROBE, "_run_probe", admit)

    result = PROBE.main(["--negative-cache-object"])

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["request"]["negative_cache_object"] is True
    assert "admitted an unsupported DynamicCache" in payload["error"]
    assert "cache_length_after=1" in payload["error"]
