#!/usr/bin/env python3
"""EXP-0023 H100 probe for the explicit guarded compile facade.

This is a correctness and compiler-boundary probe, not a benchmark. It does
not claim support for raw ``torch.compile(layer)`` or compiled cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import probe_h100_torch_compile as base
import torch

from gemma4_fa4.h100 import UnsupportedH100Path
from gemma4_fa4.transformers_integration import compile_gemma4_fa4_h100_layer

EXPERIMENT = "EXP-0023"
SCHEMA_VERSION = 1
DEFAULT_LENGTHS = base.DEFAULT_LENGTHS
FAMILY_LAYERS = base.FAMILY_LAYERS
BACKENDS = base.BACKENDS


class _CapturingBackend:
    """Capture every user-backend attempt before delegating to a stock backend."""

    def __init__(self, name: str) -> None:
        from torch._dynamo.backends.registry import lookup_backend

        self.name = name
        self._delegate = lookup_backend(name)
        self.graphs: list[dict[str, Any]] = []

    def __call__(self, graph_module, example_inputs):
        nodes = []
        for node in graph_module.graph.nodes:
            example_value = node.meta.get("example_value")
            nodes.append(
                {
                    "op": node.op,
                    "target": str(node.target),
                    "example_value_type": (
                        type(example_value).__name__ if example_value is not None else None
                    ),
                    "requires_grad": (
                        bool(example_value.requires_grad)
                        if isinstance(example_value, torch.Tensor)
                        else None
                    ),
                }
            )
        record = {
            "nodes": nodes,
            "example_input_types": [type(value).__name__ for value in example_inputs],
            "example_input_requires_grad": [
                bool(value.requires_grad)
                for value in example_inputs
                if isinstance(value, torch.Tensor)
            ],
            "backend_grad_enabled": torch.is_grad_enabled(),
            "backend_inference_mode": torch.is_inference_mode_enabled(),
        }
        self.graphs.append(record)
        try:
            compiled = self._delegate(graph_module, example_inputs)
        except Exception as exc:
            record["delegate_status"] = "raised"
            record["delegate_error_type"] = type(exc).__name__
            record["delegate_error"] = str(exc)
            raise
        record["delegate_status"] = "returned"
        return compiled


class _ScalarField(NamedTuple):
    name: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None]
    mutated: Any


def _scalar_fields(runtime) -> tuple[_ScalarField, ...]:
    config = runtime.config
    module = runtime.layer
    sliding_rope = config.rope_parameters["sliding_attention"]
    full_rope = config.rope_parameters["full_attention"]
    return (
        _ScalarField(
            "config.sliding_rope_theta",
            lambda: sliding_rope["rope_theta"],
            lambda value: sliding_rope.__setitem__("rope_theta", value),
            10_001.0,
        ),
        _ScalarField(
            "config.full_partial_rotary_factor",
            lambda: full_rope["partial_rotary_factor"],
            lambda value: full_rope.__setitem__("partial_rotary_factor", value),
            0.5,
        ),
        _ScalarField(
            "config.full_rope_theta",
            lambda: full_rope["rope_theta"],
            lambda value: full_rope.__setitem__("rope_theta", value),
            1_000_001.0,
        ),
        _ScalarField(
            "config.attention_dropout",
            lambda: config.attention_dropout,
            lambda value: setattr(config, "attention_dropout", value),
            0.125,
        ),
        _ScalarField(
            "config.final_logit_softcapping",
            lambda: config.final_logit_softcapping,
            lambda value: setattr(config, "final_logit_softcapping", value),
            31.0,
        ),
        _ScalarField(
            "config.rms_norm_eps",
            lambda: config.rms_norm_eps,
            lambda value: setattr(config, "rms_norm_eps", value),
            1e-5,
        ),
        _ScalarField(
            "module.scaling",
            lambda: module.scaling,
            lambda value: setattr(module, "scaling", value),
            2.0,
        ),
        _ScalarField(
            "module.attention_dropout",
            lambda: module.attention_dropout,
            lambda value: setattr(module, "attention_dropout", value),
            0.125,
        ),
        _ScalarField(
            "module.q_norm.eps",
            lambda: module.q_norm.eps,
            lambda value: setattr(module.q_norm, "eps", value),
            1e-5,
        ),
        _ScalarField(
            "module.k_norm.eps",
            lambda: module.k_norm.eps,
            lambda value: setattr(module.k_norm, "eps", value),
            1e-5,
        ),
        _ScalarField(
            "module.v_norm.eps",
            lambda: module.v_norm.eps,
            lambda value: setattr(module.v_norm, "eps", value),
            1e-5,
        ),
    )


def _forbidden_inner_sources(graphs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inventory Python scalar/object sources forbidden in the tensor-only frame."""

    forbidden_fragments = (
        "scalar_tensor",
        "_local_scalar_dense",
        "stack of type object",
    )
    inventory = []
    for graph_index, graph in enumerate(graphs):
        for node_index, node in enumerate(graph["nodes"]):
            target = node["target"]
            forbidden = (
                (node["op"] == "placeholder" and node.get("example_value_type") == "SymFloat")
                or target == "item"
                or any(fragment in target for fragment in forbidden_fragments)
                or (
                    node["op"] == "placeholder"
                    and any(fragment in target.lower() for fragment in ("config", "module"))
                )
            )
            if forbidden:
                inventory.append(
                    {
                        "graph_index": graph_index,
                        "node_index": node_index,
                        **node,
                    }
                )
    return inventory


def _inductor_cache_inventory(backend: str) -> dict[str, Any] | None:
    if backend != "inductor":
        return None
    raw = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not raw:
        raise RuntimeError("EXP-0023 requires an isolated TORCHINDUCTOR_CACHE_DIR")
    return base._cache_inventory(Path(raw).resolve())


def _assert_unchanged(
    *,
    label: str,
    capture: _CapturingBackend,
    graph_count: int,
    facade,
    compiled_entries: int,
    application_keys: tuple[str, ...],
    inductor_cache: dict[str, Any] | None,
) -> None:
    torch.cuda.synchronize()
    if len(capture.graphs) != graph_count:
        raise AssertionError(f"{label} reached the backend")
    if facade.compiled_entry_count != compiled_entries:
        raise AssertionError(f"{label} reached the compiled tensor-only function")
    if base._forward_application_snapshot() != application_keys:
        raise AssertionError(f"{label} changed the FA4 application-key set")
    if _inductor_cache_inventory(capture.name) != inductor_cache:
        raise AssertionError(f"{label} changed the isolated Inductor cache")


def _expect_scalar_rejection(
    facade,
    capture: _CapturingBackend,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected_output: torch.Tensor,
    field: _ScalarField,
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    hidden, cos, sin, positions = inputs
    original = field.getter()
    graph_count = len(capture.graphs)
    compiled_entries = facade.compiled_entry_count
    application_keys = base._forward_application_snapshot()
    inductor_cache = _inductor_cache_inventory(capture.name)
    field.setter(value)
    try:
        with torch.inference_mode():
            facade(hidden, (cos, sin), position_ids=positions)
    except UnsupportedH100Path as exc:
        rejection = {"error_type": type(exc).__name__, "error": str(exc)}
    else:
        raise AssertionError(f"{label} was admitted")
    finally:
        field.setter(original)
    _assert_unchanged(
        label=label,
        capture=capture,
        graph_count=graph_count,
        facade=facade,
        compiled_entries=compiled_entries,
        application_keys=application_keys,
        inductor_cache=inductor_cache,
    )

    with torch.inference_mode():
        restored, _weights = facade(hidden, (cos, sin), position_ids=positions)
    comparison = base._full_layer_comparison(
        restored,
        expected_output,
        label=f"{label}/restored",
    )
    torch.cuda.synchronize()
    if len(capture.graphs) != graph_count:
        raise AssertionError(f"{label} restoration added a backend graph")
    if facade.compiled_entry_count != compiled_entries + 1:
        raise AssertionError(f"{label} restoration did not reuse the compiled function")
    if base._forward_application_snapshot() != application_keys:
        raise AssertionError(f"{label} restoration changed the FA4 application-key set")
    if _inductor_cache_inventory(capture.name) != inductor_cache:
        raise AssertionError(f"{label} restoration changed the isolated Inductor cache")
    return {
        "name": label,
        "mutated_repr": repr(value),
        "rejected_before_compiled_entry": True,
        "rejection": rejection,
        "restored": comparison,
        "backend_graph_count_unchanged": True,
        "inductor_cache_unchanged": True,
        "fa4_application_keys_unchanged": True,
    }


def _mutation_sweep(
    runtime,
    facade,
    capture: _CapturingBackend,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected_output: torch.Tensor,
) -> dict[str, Any]:
    fields = _scalar_fields(runtime)
    exact = [
        _expect_scalar_rejection(
            facade,
            capture,
            inputs,
            expected_output,
            field,
            field.mutated,
            label=field.name,
        )
        for field in fields
    ]
    scaling = next(field for field in fields if field.name == "module.scaling")
    malformed = [
        _expect_scalar_rejection(
            facade,
            capture,
            inputs,
            expected_output,
            scaling,
            value,
            label=f"module.scaling/{name}",
        )
        for name, value in (
            ("equal-int", 1),
            ("nan", float("nan")),
            ("positive-inf", float("inf")),
            ("negative-inf", float("-inf")),
            ("negative-finite", -1.0),
        )
    ]
    return {
        "status": "passed",
        "exact_field_count": len(exact),
        "malformed_count": len(malformed),
        "exact_fields": exact,
        "malformed": malformed,
    }


def _run_family(
    family: str,
    backend: str,
    lengths: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    runtime = base._make_family_runtime(family, seed=seed)
    capture = _CapturingBackend(backend)
    eager_call = base._layer_callable(runtime)
    application_before = base._forward_application_snapshot()

    cases = []
    inputs_by_length = {}
    expected_by_length = {}
    references = []
    with torch.inference_mode():
        for index, seqlen in enumerate(lengths):
            inputs = base._make_inputs(runtime, seqlen, seed=seed + 100 + index)
            expected = eager_call(*inputs)
            inputs_by_length[seqlen] = inputs
            expected_by_length[seqlen] = expected
            references.append(base._direct_reference_case(runtime, inputs, expected))
    application_warmed = base._forward_application_snapshot()
    added = sorted(set(application_warmed) - set(application_before))
    if not set(application_before).issubset(application_warmed) or len(added) > 1:
        raise AssertionError(
            f"{family}/{backend} facade probe exceeded one bounded FA4 key class: "
            f"before={application_before}, warmed={application_warmed}, added={added}"
        )

    facade = compile_gemma4_fa4_h100_layer(runtime.layer, backend=capture)
    with torch.inference_mode():
        for seqlen in lengths:
            hidden, cos, sin, positions = inputs_by_length[seqlen]
            output, attention_weights = facade(
                hidden,
                (cos, sin),
                position_ids=positions,
            )
            if attention_weights is not None:
                raise AssertionError("guarded facade unexpectedly returned attention weights")
            cases.append(
                {
                    "seqlen": seqlen,
                    **base._full_layer_comparison(
                        output,
                        expected_by_length[seqlen],
                        label=f"{family}/{backend}/facade/S{seqlen}",
                    ),
                }
            )
    torch.cuda.synchronize()

    expected_graph_count = 2 if 1 in lengths and any(length > 1 for length in lengths) else 1
    graph_count = len(capture.graphs)
    if graph_count != expected_graph_count:
        raise AssertionError(
            f"{family}/{backend} facade produced {graph_count} backend attempts; "
            f"expected {expected_graph_count}; guard_failures={dict(torch._dynamo.guard_failures)}; "
            f"captured={capture.graphs}"
        )
    graph_break_count = base._graph_break_count()
    if graph_break_count != 0:
        raise AssertionError(f"{family}/{backend} facade produced {graph_break_count} graph breaks")
    if any(graph.get("delegate_status") != "returned" for graph in capture.graphs):
        raise AssertionError(f"{family}/{backend} facade had a rejected backend delegate attempt")
    op_counts = [
        sum(
            runtime.layer_custom_op_fragment in node["target"] and "gemma4_fa4" in node["target"]
            for node in graph["nodes"]
            if node["op"] == "call_function"
        )
        for graph in capture.graphs
    ]
    if any(count != 1 for count in op_counts):
        raise AssertionError(
            f"{family}/{backend} facade graphs require exactly one whole-layer op: {op_counts}"
        )
    if base._contains_custom_op(capture.graphs, "weight_snapshot"):
        raise AssertionError(f"{family}/{backend} facade graph retained a weight snapshot")
    allowed_input_types = {"Tensor", "Parameter", "SymInt", "int"}
    unexpected_input_types = sorted(
        {
            input_type
            for graph in capture.graphs
            for input_type in graph["example_input_types"]
            if input_type not in allowed_input_types
        }
    )
    if unexpected_input_types:
        raise AssertionError(
            f"{family}/{backend} facade received non-tensor/non-shape inputs: "
            f"{unexpected_input_types}"
        )
    forbidden_sources = _forbidden_inner_sources(capture.graphs)
    if forbidden_sources:
        raise AssertionError(
            f"{family}/{backend} facade graph retained forbidden sources: {forbidden_sources}"
        )

    mutation_length = 1 if 1 in inputs_by_length else lengths[0]
    mutations = _mutation_sweep(
        runtime,
        facade,
        capture,
        inputs_by_length[mutation_length],
        expected_by_length[mutation_length],
    )
    application_after = base._forward_application_snapshot()
    if application_after != application_warmed:
        raise AssertionError(f"{family} facade matrix changed the warmed FA4 application keys")
    return {
        "status": "passed",
        "layer_idx": runtime.layer_idx,
        "layer_type": runtime.spec.kind,
        "backend": backend,
        "lengths": list(lengths),
        "graph_count": graph_count,
        "expected_graph_count": expected_graph_count,
        "graph_break_count": graph_break_count,
        "compiled_entry_count": facade.compiled_entry_count,
        "whole_layer_custom_op_counts": op_counts,
        "unexpected_input_types": unexpected_input_types,
        "forbidden_inner_sources": forbidden_sources,
        "graphs": capture.graphs,
        "cases": cases,
        "direct_reference": references,
        "mutations": mutations,
        "fa4_application_keys": {
            "before": list(application_before),
            "warmed": list(application_warmed),
            "after": list(application_after),
            "added": added,
            "bounded": True,
        },
    }


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    base._require_h100()
    families = tuple(FAMILY_LAYERS) if args.family == "all" else (args.family,)
    backends = BACKENDS if args.backend == "all" else (args.backend,)
    cache_dirs = base._prepare_fresh_cache_dirs(backends)
    results = {}
    for family_index, family in enumerate(families):
        application_before = base._forward_application_snapshot()
        backend_results = {}
        for backend_index, backend in enumerate(backends):
            backend_results[backend] = _run_family(
                family,
                backend,
                args.lengths,
                seed=args.seed + family_index * 10_000 + backend_index * 1_000,
            )
        application_after = base._forward_application_snapshot()
        added = sorted(set(application_after) - set(application_before))
        if len(added) != 1 or not set(application_before).issubset(application_after):
            raise AssertionError(
                f"{family} facade backend matrix must add exactly one bounded FA4 key class: "
                f"before={application_before}, after={application_after}, added={added}"
            )
        results[family] = {
            "status": "passed",
            "fa4_application_keys": {
                "before": list(application_before),
                "after": list(application_after),
                "added_family_class": added,
                "new_class_count": 1,
                "reused_across_backend_matrix": True,
            },
            "backends": backend_results,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "boundary": "explicit_guarded_facade_not_raw_torch_compile_layer",
        "request": {
            "families": list(families),
            "backends": list(backends),
            "lengths": list(args.lengths),
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
            "caches": {name: base._cache_inventory(path) for name, path in cache_dirs.items()},
        },
        "families": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILY_LAYERS, "all"), default="all")
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    parser.add_argument(
        "--lengths",
        type=base._parse_lengths,
        default=DEFAULT_LENGTHS,
        help="comma-separated lengths in 1..1024",
    )
    parser.add_argument("--seed", type=int, default=23023)
    parser.add_argument("--output", type=Path)
    return parser


def _emit(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = _run_probe(args)
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "failed",
            "request": {
                "family": args.family,
                "backend": args.backend,
                "lengths": list(args.lengths),
                "seed": args.seed,
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _emit(report, args.output)
        return 1
    _emit(report, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
