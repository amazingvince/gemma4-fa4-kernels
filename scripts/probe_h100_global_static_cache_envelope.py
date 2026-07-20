#!/usr/bin/env python3
"""EXP-0026 global guarded-StaticCache correctness envelope probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import probe_h100_guarded_compile_facade as guarded
import probe_h100_static_cache_compile as cache_probe
import probe_h100_torch_compile as base
import torch

from gemma4_fa4.h100 import (
    UnsupportedH100Path,
    fa4_global_forward_only,
    fa4_global_text_forward,
)
from gemma4_fa4.reference import reference_attention
from gemma4_fa4.transformers_integration import (
    compile_gemma4_fa4_h100_static_cache_decode,
)

EXPERIMENT = "EXP-0026"
SCHEMA_VERSION = 1
BACKENDS = ("eager", "inductor")


def _graph_signature(graph: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            node["op"],
            "<SymInt>"
            if node["op"] == "placeholder" and node.get("example_value_type") == "SymInt"
            else node["target"],
        )
        for node in graph["nodes"]
    )


def _changed_slots_exact(before: torch.Tensor, after: torch.Tensor) -> tuple[int, ...]:
    changed = (before.view(torch.int16) != after.view(torch.int16)).any(
        dim=(0, 1, 3)
    )
    return tuple(int(index) for index in torch.nonzero(changed).flatten().cpu().tolist())


def _under_inference(call):
    with torch.inference_mode():
        return call()


def _audit_graph(capture: guarded._CapturingBackend, *, label: str) -> dict[str, Any]:
    if len(capture.graphs) != 1:
        raise AssertionError(f"{label} requires exactly one backend graph")
    graph = capture.graphs[0]
    if graph.get("delegate_status") != "returned":
        raise AssertionError(f"{label} backend delegate did not return")
    cache_ops = [
        node
        for node in graph["nodes"]
        if node["op"] == "call_function"
        and "gemma4_fa4" in node["target"]
        and "h100_global_static_cache_decode_fwd" in node["target"]
    ]
    cache_placeholders = [
        node
        for node in graph["nodes"]
        if node["op"] == "placeholder"
        and any(name in node["target"] for name in ("cache_k", "cache_v", "cache_length"))
    ]
    cache_getattrs = [
        node
        for node in graph["nodes"]
        if node["op"] == "get_attr" and "cache" in node["target"]
    ]
    forbidden = guarded._forbidden_inner_sources(capture.graphs)
    if len(cache_ops) != 1 or len(cache_placeholders) != 3 or cache_getattrs or forbidden:
        raise AssertionError(
            f"{label} violated the explicit cache graph ABI: "
            f"ops={cache_ops}, placeholders={cache_placeholders}, "
            f"getattrs={cache_getattrs}, forbidden={forbidden}"
        )
    unexpected_types = sorted(
        set(graph["example_input_types"]).difference({"Tensor", "Parameter", "SymInt", "int"})
    )
    if unexpected_types:
        raise AssertionError(f"{label} retained object graph inputs: {unexpected_types}")
    return {
        "cache_op_count": len(cache_ops),
        "cache_placeholder_count": len(cache_placeholders),
        "cache_getattr_count": len(cache_getattrs),
        "forbidden_source_count": len(forbidden),
        "signature": [list(item) for item in _graph_signature(graph)],
        "graph": graph,
    }


@torch.inference_mode()
def _prepared_reference(
    runtime,
    inputs: tuple[torch.Tensor, ...],
    cache: Any,
    active_length: int,
    candidate_output: torch.Tensor,
    candidate_lse: torch.Tensor,
) -> dict[str, Any]:
    hidden, cos, sin, _positions = inputs
    q, _new_k, _new_v = base._prepare_qkv(runtime, hidden, cos, sin)
    layer = cache.layers[cache_probe.GLOBAL_LAYER]
    active_k = layer.keys[:, :, :active_length, :]
    active_v = layer.values[:, :, :active_length, :]
    reference_output, reference_lse = reference_attention(
        q,
        active_k,
        active_v,
        softmax_scale=1.0,
        sliding_window=None,
        allow_vision_bidirectional=False,
        upcast=torch.float32,
        return_lse=True,
    )
    q_bshd = q.transpose(1, 2)
    k_bshd = active_k.transpose(1, 2)
    v_bshd = active_v.transpose(1, 2)
    if active_length <= 1024:
        q_bshd = torch.cat(
            (
                torch.zeros(
                    (1, active_length - 1, 32, 512),
                    dtype=q.dtype,
                    device=q.device,
                ),
                q_bshd,
            ),
            dim=1,
        )
        prepared_output, prepared_lse = fa4_global_text_forward(
            q_bshd,
            k_bshd,
            v_bshd,
        )
        prepared_output = prepared_output[:, -1:, :, :]
        prepared_lse = prepared_lse[:, :, -1:].contiguous()
    else:
        prepared_output, prepared_lse = fa4_global_forward_only(
            q_bshd,
            k_bshd,
            v_bshd,
        )
    projected = runtime.layer.o_proj(prepared_output.reshape(1, 1, -1).contiguous())
    if not torch.equal(projected, candidate_output):
        raise AssertionError("global cache projection transport is not bitwise exact")
    if not torch.equal(prepared_lse, candidate_lse):
        raise AssertionError("global cache compiled/prepared FP32 LSE is not bitwise exact")
    return {
        "active_length": active_length,
        "prepared_output": base._assert_close(
            prepared_output.transpose(1, 2),
            reference_output,
            atol=base.GLOBAL_OUTPUT_ATOL,
            rtol=base.GLOBAL_OUTPUT_RTOL,
        ),
        "prepared_lse": base._assert_close(
            prepared_lse,
            reference_lse,
            atol=base.GLOBAL_LSE_ATOL,
            rtol=0.0,
        ),
        "projection_transport_bitwise": True,
        "compiled_lse_bitwise": True,
    }


def _weight_identical_eager_layer(runtime):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    layer = Gemma4TextAttention(runtime.config, layer_idx=cache_probe.GLOBAL_LAYER).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer.load_state_dict(runtime.layer.state_dict())
    layer.eval()
    return layer


def _run_case(
    *,
    backend: str,
    prompt_length: int,
    capacity: int,
    decode_steps: int,
    seed: int,
    hostile_tail: bool,
    nondefault_stream: bool,
    inductor_cache: Path | None,
) -> dict[str, Any]:
    label = f"{backend}/K{prompt_length + 1}"
    runtime = base._make_family_runtime("global", seed=seed)
    eager_layer = _weight_identical_eager_layer(runtime)
    with torch.inference_mode():
        candidate_cache = cache_probe._early_static_cache(runtime, capacity)
        eager_cache = cache_probe._early_static_cache(runtime, capacity)
        prefill = base._make_inputs(runtime, prompt_length, seed=seed + 1)
        candidate_prefill = cache_probe._eager_layer_call(
            runtime,
            runtime.layer,
            candidate_cache,
            prefill,
        )
        eager_prefill = cache_probe._eager_layer_call(
            runtime,
            eager_layer,
            eager_cache,
            prefill,
        )
        if not torch.equal(candidate_prefill, eager_prefill):
            raise AssertionError(f"{label} eager prefill twins diverged")

        hostile_digests = None
        if hostile_tail:
            candidate_layer = candidate_cache.layers[cache_probe.GLOBAL_LAYER]
            first_tail = prompt_length + decode_steps
            if first_tail >= capacity:
                raise AssertionError("hostile-tail case has no unwritten physical slot")
            candidate_layer.keys[:, :, first_tail:, :].fill_(float("nan"))
            candidate_layer.values[:, :, first_tail:, :].fill_(-47)
            hostile_digests = {
                "keys": cache_probe._cache_digest(
                    candidate_layer.keys[:, :, first_tail:, :]
                ),
                "values": cache_probe._cache_digest(
                    candidate_layer.values[:, :, first_tail:, :]
                ),
            }

    capture = guarded._CapturingBackend(backend)
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        candidate_cache,
        backend=capture,
    )
    application_before = base._forward_application_snapshot()
    step_records: list[dict[str, Any]] = []
    first_inductor_inventory = None
    stream = torch.cuda.Stream() if nondefault_stream else None
    for step in range(decode_steps):
        position = prompt_length + step
        inputs = base._make_inputs(
            runtime,
            1,
            seed=seed + 2 + step,
            position_start=position,
        )
        candidate_before = cache_probe._cache_snapshot(candidate_cache)
        eager_before = cache_probe._cache_snapshot(eager_cache)
        candidate_context = torch.cuda.stream(stream) if stream is not None else torch.cuda.device(0)
        with candidate_context, torch.inference_mode():
            candidate_output, candidate_weights = facade(
                inputs[0],
                (inputs[1], inputs[2]),
                position_ids=inputs[3],
            )
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()
        application_after_candidate = base._forward_application_snapshot()
        with torch.inference_mode():
            eager_output = cache_probe._eager_layer_call(
                runtime,
                eager_layer,
                eager_cache,
                inputs,
            )
        torch.cuda.synchronize()
        if candidate_weights is not None or facade.last_lse is None:
            raise AssertionError(f"{label} returned an invalid layer/LSE contract")
        if not torch.equal(candidate_output, eager_output):
            maximum, mean = base._finite_error(candidate_output, eager_output)
            raise AssertionError(
                f"{label} step {step} is not bitwise eager-equal "
                f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
            )
        candidate_after = cache_probe._cache_snapshot(candidate_cache)
        eager_after = cache_probe._cache_snapshot(eager_cache)
        active_length = position + 1
        candidate_layer = candidate_cache.layers[cache_probe.GLOBAL_LAYER]
        eager_cache_layer = eager_cache.layers[cache_probe.GLOBAL_LAYER]
        for operand, candidate_tensor, eager_tensor in (
            ("K", candidate_layer.keys, eager_cache_layer.keys),
            ("V", candidate_layer.values, eager_cache_layer.values),
        ):
            if not torch.equal(
                candidate_tensor[:, :, :active_length, :],
                eager_tensor[:, :, :active_length, :],
            ):
                raise AssertionError(f"{label} active {operand} differs from eager")
        changed_k = _changed_slots_exact(
            candidate_before["keys"],
            candidate_after["keys"],
        )
        changed_v = _changed_slots_exact(
            candidate_before["values"],
            candidate_after["values"],
        )
        if changed_k != (position,) or changed_v != (position,):
            raise AssertionError(
                f"{label} step {step} changed wrong slots: K={changed_k}, V={changed_v}"
            )
        if (
            candidate_after["length"] != active_length
            or eager_after["length"] != active_length
            or candidate_after["pointers"] != candidate_before["pointers"]
            or eager_after["pointers"] != eager_before["pointers"]
        ):
            raise AssertionError(f"{label} counter or address contract changed")
        if base._forward_application_snapshot() != application_after_candidate:
            raise AssertionError(f"{label} eager twin added a different FA4 key")
        reference = _prepared_reference(
            runtime,
            inputs,
            candidate_cache,
            active_length,
            candidate_output,
            facade.last_lse,
        )
        step_records.append(
            {
                "position": position,
                "active_length": active_length,
                "changed_slots": {"k": list(changed_k), "v": list(changed_v)},
                "whole_layer_bitwise": True,
                "active_cache_matches_eager": True,
                "addresses_stable": True,
                "reference": reference,
            }
        )
        if backend == "inductor" and inductor_cache is not None:
            inventory = base._cache_inventory(inductor_cache)
            if first_inductor_inventory is None:
                first_inductor_inventory = inventory
            elif inventory != first_inductor_inventory:
                raise AssertionError(f"{label} second decode changed the Inductor cache")

    graph_record = _audit_graph(capture, label=label)
    if facade.compiled_entry_count != decode_steps:
        raise AssertionError(f"{label} compiled-entry count is wrong")
    added_applications = sorted(
        set(base._forward_application_snapshot()).difference(application_before)
    )
    if len(added_applications) > 1:
        raise AssertionError(f"{label} added too many global FA4 application classes")
    if hostile_tail:
        candidate_layer = candidate_cache.layers[cache_probe.GLOBAL_LAYER]
        first_tail = prompt_length + decode_steps
        current = {
            "keys": cache_probe._cache_digest(candidate_layer.keys[:, :, first_tail:, :]),
            "values": cache_probe._cache_digest(candidate_layer.values[:, :, first_tail:, :]),
        }
        if current != hostile_digests:
            raise AssertionError(f"{label} touched the hostile unwritten tail")

    return {
        "label": label,
        "backend": backend,
        "prompt_length": prompt_length,
        "capacity": capacity,
        "decode_steps": decode_steps,
        "hostile_tail": hostile_tail,
        "nondefault_stream": nondefault_stream,
        "backend_attempts": len(capture.graphs),
        "compiled_entries": facade.compiled_entry_count,
        "graph": graph_record,
        "steps": step_records,
        "fa4_application_keys_added": added_applications,
        "hostile_tail_unchanged": hostile_tail,
    }


def _expect_preentry_rejection(
    facade,
    capture,
    cache,
    call,
    *,
    label: str,
) -> str:
    snapshot = cache_probe._cache_snapshot(cache)
    graphs = len(capture.graphs)
    entries = facade.compiled_entry_count
    applications = base._forward_application_snapshot()
    try:
        call()
    except UnsupportedH100Path as exc:
        error = str(exc)
    else:
        raise AssertionError(f"{label} was admitted")
    cache_probe._assert_cache_unchanged(cache, snapshot, label=label)
    if (
        len(capture.graphs) != graphs
        or facade.compiled_entry_count != entries
        or base._forward_application_snapshot() != applications
    ):
        raise AssertionError(f"{label} reached the compiled boundary")
    return error


def _negative_matrix(seed: int) -> dict[str, str]:
    runtime = base._make_family_runtime("global", seed=seed)
    with torch.inference_mode():
        cache = cache_probe._early_static_cache(runtime, 65)
        prefill = base._make_inputs(runtime, 32, seed=seed + 1)
        cache_probe._eager_layer_call(runtime, runtime.layer, cache, prefill)
    capture = guarded._CapturingBackend("eager")
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        cache,
        backend=capture,
    )
    valid = base._make_inputs(runtime, 1, seed=seed + 2, position_start=32)
    batch_hidden = valid[0].expand(2, -1, -1).contiguous()
    q2_hidden = valid[0].expand(-1, 2, -1).contiguous()
    q2_cos = valid[1].expand(-1, 2, -1).contiguous()
    q2_sin = valid[2].expand(-1, 2, -1).contiguous()
    q2_positions = torch.tensor([[32, 33]], device="cuda", dtype=torch.int64)
    errors = {
        "batch2": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: _under_inference(
                lambda: facade(
                    batch_hidden,
                    (valid[1], valid[2]),
                    position_ids=valid[3],
                )
            ),
            label="B2",
        ),
        "query2": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: _under_inference(
                lambda: facade(
                    q2_hidden,
                    (q2_cos, q2_sin),
                    position_ids=q2_positions,
                )
            ),
            label="Q2",
        ),
        "metadata": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: _under_inference(
                lambda: facade(
                    valid[0],
                    (valid[1], valid[2]),
                    position_ids=valid[3],
                    document_ids=torch.zeros_like(valid[3]),
                )
            ),
            label="metadata",
        ),
        "active_grad": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="active grad mode",
        ),
    }
    with torch.inference_mode():
        lazy_cache = cache_probe._early_static_cache(runtime, 65)
        lazy_cache.layers[cache_probe.GLOBAL_LAYER].is_initialized = False
        try:
            compile_gemma4_fa4_h100_static_cache_decode(runtime.layer, lazy_cache)
        except UnsupportedH100Path as exc:
            errors["lazy_cache"] = str(exc)
        else:
            raise AssertionError("lazy cache was admitted at construction")

        offloaded_cache = cache_probe._early_static_cache(runtime, 65)
        offloaded_cache.offloading = True
        try:
            compile_gemma4_fa4_h100_static_cache_decode(runtime.layer, offloaded_cache)
        except UnsupportedH100Path as exc:
            errors["offloaded_cache"] = str(exc)
        else:
            raise AssertionError("offloaded cache was admitted at construction")

        exhausted_cache = cache_probe._early_static_cache(runtime, 65)
        prefill64 = base._make_inputs(runtime, 64, seed=seed + 3)
        cache_probe._eager_layer_call(runtime, runtime.layer, exhausted_cache, prefill64)
        try:
            compile_gemma4_fa4_h100_static_cache_decode(runtime.layer, exhausted_cache)
        except UnsupportedH100Path as exc:
            errors["capacity_exhaustion"] = str(exc)
        else:
            raise AssertionError("cache without a spare tail slot was admitted")
    return errors


def _run_matrix(*, seed: int, reverse_order: bool = False) -> dict[str, Any]:
    base._require_h100()
    cache_dirs = base._prepare_fresh_cache_dirs(BACKENDS)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    case_specs = [
        ("inductor", 32, 65, 2, False, False),
        ("eager", 32, 65, 2, False, False),
        ("inductor", 1024, 1026, 1, True, True),
        ("eager", 1024, 1026, 1, True, False),
    ]
    if reverse_order:
        case_specs.reverse()
    cases = [
        _run_case(
            backend=backend,
            prompt_length=prompt,
            capacity=capacity,
            decode_steps=steps,
            seed=seed + index * 100,
            hostile_tail=hostile,
            nondefault_stream=stream,
            inductor_cache=cache_dirs.get("inductor"),
        )
        for index, (backend, prompt, capacity, steps, hostile, stream) in enumerate(
            case_specs
        )
    ]
    shape_signatures = {
        tuple(tuple(item) for item in case["graph"]["signature"])
        for case in cases
    }
    semantic_signatures = {
        tuple(item for item in signature if item != ("placeholder", "<SymInt>"))
        for signature in shape_signatures
    }
    if len(semantic_signatures) != 1:
        raise AssertionError(
            "global runtime cases produced different semantic graph structures: "
            + json.dumps(
                {case["label"]: case["graph"]["signature"] for case in cases},
                sort_keys=True,
            )
        )
    graph_breaks = base._graph_break_count()
    if graph_breaks != 0:
        raise AssertionError(f"global StaticCache matrix produced {graph_breaks} graph breaks")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "case_order": "reverse" if reverse_order else "default",
        "cases": cases,
        "negative_matrix": _negative_matrix(seed + 10_000),
        "graph_breaks": graph_breaks,
        "semantic_graph_classes": len(semantic_signatures),
        "runtime_shape_signatures": len(shape_signatures),
        "cache_dirs": {
            name: base._cache_inventory(path) for name, path in cache_dirs.items()
        },
        "claims": {
            "local_cache": False,
            "compiled_prefill": False,
            "raw_torch_compile_layer": False,
            "performance": False,
            "b300": False,
        },
    }


def _run_sanitizer_case(*, seed: int) -> dict[str, Any]:
    """Run one exact global K1025 cache-decode call for Compute Sanitizer."""

    base._require_h100()
    cache_dirs = base._prepare_fresh_cache_dirs(("inductor",))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    case = _run_case(
        backend="inductor",
        prompt_length=1024,
        capacity=1026,
        decode_steps=1,
        seed=seed,
        hostile_tail=True,
        nondefault_stream=False,
        inductor_cache=cache_dirs["inductor"],
    )
    graph_breaks = base._graph_break_count()
    if graph_breaks != 0:
        raise AssertionError(
            f"global StaticCache sanitizer case produced {graph_breaks} graph breaks"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "mode": "sanitizer_case",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "case": case,
        "graph_breaks": graph_breaks,
        "cache_dirs": {
            name: base._cache_inventory(path) for name, path in cache_dirs.items()
        },
        "claims": {
            "local_cache": False,
            "compiled_prefill": False,
            "raw_torch_compile_layer": False,
            "performance": False,
            "b300": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=26_001)
    parser.add_argument(
        "--sanitizer-case",
        action="store_true",
        help="run only the global Inductor Q1/K1025 case",
    )
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="run the correctness matrix in reverse case order",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = (
        _run_sanitizer_case(seed=args.seed)
        if args.sanitizer_case
        else _run_matrix(seed=args.seed, reverse_order=args.reverse_order)
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        print(
            json.dumps(
                {
                    "experiment": report["experiment"],
                    "mode": report.get("mode", "matrix"),
                    "output": str(args.output),
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
