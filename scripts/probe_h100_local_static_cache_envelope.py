#!/usr/bin/env python3
"""EXP-0028 local guarded-StaticSlidingWindow correctness envelope."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import probe_h100_guarded_compile_facade as guarded
import probe_h100_local_static_cache_compile as first
import probe_h100_static_cache_compile as cache_probe
import probe_h100_torch_compile as base
import torch

from gemma4_fa4.h100 import UnsupportedH100Path
from gemma4_fa4.transformers_integration import (
    compile_gemma4_fa4_h100_static_cache_decode,
)

EXPERIMENT = "EXP-0028"
SCHEMA_VERSION = 1
BACKENDS = ("eager", "inductor")
ACCEPTED_NATIVE_VARLEN_APPLICATION = (
    "e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176"
)


def _graph_signature(graph: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            node["op"],
            "<SymInt>"
            if node["op"] == "placeholder"
            and node.get("example_value_type") == "SymInt"
            else node["target"],
        )
        for node in graph["nodes"]
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    return first._cache_digest(tensor)


def _weight_identical_eager_layer(runtime):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    layer = Gemma4TextAttention(runtime.config, layer_idx=first.LOCAL_LAYER).to(
        device="cuda", dtype=torch.bfloat16
    )
    layer.load_state_dict(runtime.layer.state_dict())
    layer.eval()
    return layer


def _run_case(
    *,
    backend: str,
    prompt_length: int,
    decode_steps: int,
    seed: int,
    hostile_tail: bool,
    nondefault_stream: bool,
    inductor_cache: Path | None,
) -> dict[str, Any]:
    label = f"{backend}/K{prompt_length + 1}"
    if hostile_tail:
        label += "/hostile"
    if nondefault_stream:
        label += "/stream"
    runtime = base._make_family_runtime("local", seed=seed)
    eager_layer = _weight_identical_eager_layer(runtime)
    candidate_cache = cache_probe._early_static_cache(runtime, first.CAPACITY)
    eager_cache = cache_probe._early_static_cache(runtime, first.CAPACITY)
    with torch.inference_mode():
        prefill = base._make_inputs(runtime, prompt_length, seed=seed + 1)
        candidate_prefill = first._eager_layer_call(
            runtime, runtime.layer, candidate_cache, prefill
        )
        eager_prefill = first._eager_layer_call(
            runtime, eager_layer, eager_cache, prefill
        )
        if not torch.equal(candidate_prefill, eager_prefill):
            raise AssertionError(f"{label} eager prefill twins diverged")
        candidate_layer = candidate_cache.layers[first.LOCAL_LAYER]
        hostile_start = prompt_length + decode_steps
        hostile_digests = None
        if hostile_tail:
            if hostile_start >= first.CAPACITY:
                raise AssertionError("hostile-tail case has no unwritten tail")
            candidate_layer.keys[:, :, hostile_start:, :].fill_(float("nan"))
            candidate_layer.values[:, :, hostile_start:, :].fill_(-47)
            candidate_layer.values[:, :, hostile_start::2, :].fill_(float("nan"))
            hostile_digests = {
                "keys": _tensor_digest(candidate_layer.keys[:, :, hostile_start:, :]),
                "values": _tensor_digest(candidate_layer.values[:, :, hostile_start:, :]),
            }

    capture = guarded._CapturingBackend(backend)
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        candidate_cache,
        backend=capture,
    )
    application_before = base._forward_application_snapshot()
    stream = torch.cuda.Stream() if nondefault_stream else None
    first_inductor_inventory = None
    steps: list[dict[str, Any]] = []
    output_digests: list[str] = []
    for step in range(decode_steps):
        position = prompt_length + step
        inputs = base._make_inputs(
            runtime,
            1,
            seed=seed + 2 + step,
            position_start=position,
        )
        candidate_before = first._cache_snapshot(candidate_cache, clone=True)
        eager_before = first._cache_snapshot(eager_cache, clone=True)
        candidate_context = (
            torch.cuda.stream(stream) if stream is not None else torch.cuda.device(0)
        )
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
            eager_output = first._eager_layer_call(
                runtime, eager_layer, eager_cache, inputs
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
        if base._forward_application_snapshot() != application_after_candidate:
            raise AssertionError(f"{label} eager twin added a different FA4 key")
        candidate_after = first._cache_snapshot(candidate_cache, clone=True)
        eager_after = first._cache_snapshot(eager_cache, clone=True)
        active_length = min(position + 1, first.CAPACITY)
        candidate_layer = candidate_cache.layers[first.LOCAL_LAYER]
        eager_cache_layer = eager_cache.layers[first.LOCAL_LAYER]
        for operand, candidate_tensor, eager_tensor in (
            ("K", candidate_layer.keys, eager_cache_layer.keys),
            ("V", candidate_layer.values, eager_cache_layer.values),
        ):
            if not torch.equal(
                candidate_tensor[:, :, :active_length, :],
                eager_tensor[:, :, :active_length, :],
            ):
                raise AssertionError(f"{label} active {operand} differs from eager")
        candidate_mutation = first._assert_step_mutation(
            candidate_before, candidate_after, position=position
        )
        eager_mutation = first._assert_step_mutation(
            eager_before, eager_after, position=position
        )
        reference = first._prepared_reference(
            runtime,
            inputs,
            candidate_cache,
            candidate_output,
            facade.last_lse,
        )
        output_digests.append(_tensor_digest(candidate_output))
        steps.append(
            {
                "position": position,
                "active_length": active_length,
                "candidate_mutation": candidate_mutation,
                "eager_mutation": eager_mutation,
                "whole_layer_bitwise": True,
                "active_cache_matches_eager": True,
                "reference": reference,
            }
        )
        if backend == "inductor" and inductor_cache is not None:
            inventory = base._cache_inventory(inductor_cache)
            if first_inductor_inventory is None:
                first_inductor_inventory = inventory
            elif inventory != first_inductor_inventory:
                raise AssertionError(f"{label} later decode changed the Inductor cache")

    graph = first._graph_record(capture)
    if facade.compiled_entry_count != decode_steps:
        raise AssertionError(f"{label} compiled-entry count is wrong")
    added_applications = sorted(
        set(base._forward_application_snapshot()).difference(application_before)
    )
    if len(added_applications) > 1:
        raise AssertionError(f"{label} added too many local FA4 application classes")
    if hostile_tail:
        candidate_layer = candidate_cache.layers[first.LOCAL_LAYER]
        current = {
            "keys": _tensor_digest(candidate_layer.keys[:, :, hostile_start:, :]),
            "values": _tensor_digest(candidate_layer.values[:, :, hostile_start:, :]),
        }
        if current != hostile_digests:
            raise AssertionError(f"{label} touched the hostile unwritten tail")
    return {
        "label": label,
        "backend": backend,
        "prompt_length": prompt_length,
        "decode_steps": decode_steps,
        "hostile_tail": hostile_tail,
        "hostile_tail_unchanged": hostile_tail,
        "nondefault_stream": nondefault_stream,
        "backend_attempts": len(capture.graphs),
        "compiled_entries": facade.compiled_entry_count,
        "graph": graph,
        "graph_signature": [list(item) for item in _graph_signature(graph["graph"])],
        "steps": steps,
        "output_digests": output_digests,
        "fa4_application_keys_added": added_applications,
    }


def _under_inference(call):
    with torch.inference_mode():
        return call()


def _expect_preentry_rejection(
    facade,
    capture,
    cache,
    call,
    *,
    label: str,
) -> str:
    snapshot = first._cache_snapshot(cache, clone=False)
    graphs = len(capture.graphs)
    entries = facade.compiled_entry_count
    applications = base._forward_application_snapshot()
    try:
        call()
    except UnsupportedH100Path as exc:
        error = str(exc)
    else:
        raise AssertionError(f"{label} was admitted")
    current = first._cache_snapshot(cache, clone=False)
    state_fields = (
        "digests",
        "pointers",
        "tensor_length",
        "absolute_length",
        "versions",
    )
    if any(current[field] != snapshot[field] for field in state_fields):
        raise AssertionError(f"{label} mutated cache state")
    if (
        len(capture.graphs) != graphs
        or facade.compiled_entry_count != entries
        or base._forward_application_snapshot() != applications
    ):
        raise AssertionError(f"{label} reached the compiled boundary")
    return error


def _negative_matrix(seed: int) -> dict[str, str]:
    runtime = base._make_family_runtime("local", seed=seed)
    cache = cache_probe._early_static_cache(runtime, first.CAPACITY)
    with torch.inference_mode():
        prefill = base._make_inputs(runtime, 32, seed=seed + 1)
        first._eager_layer_call(runtime, runtime.layer, cache, prefill)
    capture = guarded._CapturingBackend("eager")
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer, cache, backend=capture
    )
    valid = base._make_inputs(runtime, 1, seed=seed + 2, position_start=32)
    batch_hidden = valid[0].expand(2, -1, -1).contiguous()
    q2_hidden = valid[0].expand(-1, 2, -1).contiguous()
    q2_cos = valid[1].expand(-1, 2, -1).contiguous()
    q2_sin = valid[2].expand(-1, 2, -1).contiguous()
    q2_positions = torch.tensor([[32, 33]], device="cuda", dtype=torch.int64)
    wrong_position = torch.tensor([[31]], device="cuda", dtype=torch.int64)
    requires_grad_hidden = valid[0].detach().requires_grad_(True)
    errors = {
        "batch2": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: _under_inference(
                lambda: facade(
                    batch_hidden, (valid[1], valid[2]), position_ids=valid[3]
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
                    q2_hidden, (q2_cos, q2_sin), position_ids=q2_positions
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
        "wrong_position": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: _under_inference(
                lambda: facade(
                    valid[0], (valid[1], valid[2]), position_ids=wrong_position
                )
            ),
            label="wrong position",
        ),
        "active_grad": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="active grad mode",
        ),
        "requires_grad_input": _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: _under_inference(
                lambda: facade(
                    requires_grad_hidden,
                    (valid[1], valid[2]),
                    position_ids=valid[3],
                )
            ),
            label="requires-grad input",
        ),
    }
    layer = cache.layers[first.LOCAL_LAYER]
    with torch.inference_mode():
        layer.cumulative_length.add_(1)
        errors["counter_mismatch"] = _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="counter mismatch",
        )
        layer.cumulative_length.sub_(1)
        facade._cache_versions = first._cache_snapshot(cache, clone=False)["versions"]

        layer.cumulative_length_int += 1
        errors["python_counter_mismatch"] = _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="Python counter mismatch",
        )
        layer.cumulative_length_int -= 1

        original_keys = layer.keys
        with torch.inference_mode(False):
            rebound_keys = original_keys.clone()
        layer.keys = rebound_keys
        errors["root_rebind"] = _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="root rebind",
        )
        layer.keys = original_keys

        original_compiled_keys = facade._binding.compiled_keys
        object.__setattr__(
            facade._binding,
            "compiled_keys",
            original_compiled_keys.clone(),
        )
        errors["forged_compiled_view"] = _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="forged compiled view",
        )
        object.__setattr__(
            facade._binding,
            "compiled_keys",
            original_compiled_keys,
        )

        foreign_cache = cache_probe._early_static_cache(runtime, first.CAPACITY)
        original_cache = facade._binding.cache
        object.__setattr__(facade._binding, "cache", foreign_cache)
        errors["foreign_cache"] = _expect_preentry_rejection(
            facade,
            capture,
            cache,
            lambda: facade(valid[0], (valid[1], valid[2]), position_ids=valid[3]),
            label="foreign cache",
        )
        object.__setattr__(facade._binding, "cache", original_cache)

    for name, mutate in (
        ("lazy_cache", lambda candidate: setattr(candidate.layers[0], "is_initialized", False)),
        ("offloaded_cache", lambda candidate: setattr(candidate, "offloading", True)),
    ):
        candidate = cache_probe._early_static_cache(runtime, first.CAPACITY)
        with torch.inference_mode():
            mutate(candidate)
        try:
            compile_gemma4_fa4_h100_static_cache_decode(runtime.layer, candidate)
        except UnsupportedH100Path as exc:
            errors[name] = str(exc)
        else:
            raise AssertionError(f"{name} was admitted at construction")

    wrong_capacity = cache_probe._early_static_cache(runtime, 65)
    try:
        compile_gemma4_fa4_h100_static_cache_decode(runtime.layer, wrong_capacity)
    except UnsupportedH100Path as exc:
        errors["wrong_capacity"] = str(exc)
    else:
        raise AssertionError("wrong local capacity was admitted")

    wrong_class = cache_probe._early_static_cache(runtime, first.CAPACITY)
    with torch.inference_mode():
        wrong_class.layers[0] = wrong_class.layers[5]
    try:
        compile_gemma4_fa4_h100_static_cache_decode(runtime.layer, wrong_class)
    except UnsupportedH100Path as exc:
        errors["wrong_layer_class"] = str(exc)
    else:
        raise AssertionError("wrong local layer class was admitted")

    reset_cache = cache_probe._early_static_cache(runtime, first.CAPACITY)
    with torch.inference_mode():
        first._eager_layer_call(runtime, runtime.layer, reset_cache, prefill)
    reset_capture = guarded._CapturingBackend("eager")
    reset_facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        reset_cache,
        backend=reset_capture,
    )
    with torch.inference_mode():
        reset_cache.reset()
    errors["reset_cache"] = _expect_preentry_rejection(
        reset_facade,
        reset_capture,
        reset_cache,
        lambda: _under_inference(
            lambda: reset_facade(
                valid[0], (valid[1], valid[2]), position_ids=valid[3]
            )
        ),
        label="reset cache",
    )
    return errors


def _run_matrix(*, seed: int, reverse_order: bool) -> dict[str, Any]:
    base._require_h100()
    cache_dirs = base._prepare_fresh_cache_dirs(BACKENDS)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    case_specs = [
        ("inductor", 32, 2, False, False, 0),
        ("eager", 32, 2, False, False, 0),
        ("inductor", 32, 2, True, True, 0),
        ("eager", 32, 2, True, False, 0),
        ("inductor", 1023, 3, False, False, 100),
        ("eager", 1023, 3, False, False, 100),
    ]
    if reverse_order:
        case_specs.reverse()
    applications_before = base._forward_application_snapshot()
    cases = [
        _run_case(
            backend=backend,
            prompt_length=prompt,
            decode_steps=steps,
            seed=seed + seed_offset,
            hostile_tail=hostile,
            nondefault_stream=stream,
            inductor_cache=cache_dirs.get("inductor"),
        )
        for backend, prompt, steps, hostile, stream, seed_offset in case_specs
    ]
    clean_outputs = {
        (case["backend"], case["prompt_length"]): case["output_digests"]
        for case in cases
        if not case["hostile_tail"]
    }
    for case in cases:
        if case["hostile_tail"] and case["output_digests"] != clean_outputs[
            (case["backend"], case["prompt_length"])
        ]:
            raise AssertionError(f"{case['label']} hostile tail changed output bytes")
    shape_signatures = {
        tuple(tuple(item) for item in case["graph_signature"]) for case in cases
    }
    semantic_signatures = {
        tuple(item for item in signature if item != ("placeholder", "<SymInt>"))
        for signature in shape_signatures
    }
    if len(semantic_signatures) != 1:
        raise AssertionError("local cases produced different semantic graph structures")
    graph_breaks = base._graph_break_count()
    if graph_breaks:
        raise AssertionError(f"local cache matrix produced {graph_breaks} graph breaks")
    application_delta = sorted(
        set(base._forward_application_snapshot()).difference(applications_before)
    )
    decode_application_delta = sorted(
        {
            key
            for case in cases
            for key in case["fa4_application_keys_added"]
        }
    )
    if decode_application_delta != [ACCEPTED_NATIVE_VARLEN_APPLICATION]:
        raise AssertionError(
            "local underfill/full-window decodes must share the accepted native "
            f"application class; found {decode_application_delta}; per-case additions="
            f"{[(case['label'], case['fa4_application_keys_added']) for case in cases]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "pass",
        "mode": "matrix",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "seed": seed,
        "case_order": "reverse" if reverse_order else "default",
        "cases": cases,
        "negative_matrix": _negative_matrix(seed + 10_000),
        "graph_breaks": graph_breaks,
        "semantic_graph_classes": len(semantic_signatures),
        "runtime_shape_signatures": len(shape_signatures),
        "all_prefill_and_decode_application_delta": application_delta,
        "decode_application_delta": decode_application_delta,
        "cache_dirs": {
            name: base._cache_inventory(path) for name, path in cache_dirs.items()
        },
        "claims": {
            "exact_local_layer_0_q1_text_decode": True,
            "compiled_prefill": False,
            "vision_or_document_cache": False,
            "training": False,
            "performance": False,
            "b300": False,
        },
    }


def _run_sanitizer_case(*, seed: int) -> dict[str, Any]:
    base._require_h100()
    cache_dirs = base._prepare_fresh_cache_dirs(("inductor",))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    case = _run_case(
        backend="inductor",
        prompt_length=1024,
        decode_steps=2,
        seed=seed,
        hostile_tail=False,
        nondefault_stream=False,
        inductor_cache=cache_dirs["inductor"],
    )
    graph_breaks = base._graph_break_count()
    if graph_breaks:
        raise AssertionError(
            f"local sanitizer case produced {graph_breaks} graph breaks"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "pass",
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
            "exact_local_layer_0_q1_text_decode": True,
            "performance": False,
            "b300": False,
        },
    }


def _run_negative_only(*, seed: int) -> dict[str, Any]:
    base._require_h100()
    cache_dirs = base._prepare_fresh_cache_dirs(("eager",))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    errors = _negative_matrix(seed)
    graph_breaks = base._graph_break_count()
    if graph_breaks:
        raise AssertionError(f"negative-only matrix produced {graph_breaks} graph breaks")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "pass",
        "mode": "negative_only",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "negative_matrix": errors,
        "graph_breaks": graph_breaks,
        "cache_dirs": {
            name: base._cache_inventory(path) for name, path in cache_dirs.items()
        },
        "claims": {
            "preentry_rejection_only": True,
            "performance": False,
            "b300": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=28_001)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--sanitizer-case", action="store_true")
    parser.add_argument("--negative-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.sanitizer_case and args.negative_only:
        raise ValueError("--sanitizer-case and --negative-only are mutually exclusive")
    if args.sanitizer_case:
        report = _run_sanitizer_case(seed=args.seed)
    elif args.negative_only:
        report = _run_negative_only(seed=args.seed)
    else:
        report = _run_matrix(seed=args.seed, reverse_order=args.reverse_order)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        print(
            json.dumps(
                {
                    "experiment": report["experiment"],
                    "mode": report["mode"],
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
