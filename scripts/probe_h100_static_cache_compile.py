#!/usr/bin/env python3
"""EXP-0025 first discriminator for an explicit StaticCache view ABI.

This is a correctness/compiler-boundary probe for actual pinned global layer 5.
It does not claim compiled prefill, local-cache support, raw layer compilation,
performance, or B300 compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import probe_h100_guarded_compile_facade as guarded
import probe_h100_torch_compile as base
import torch

from gemma4_fa4.h100 import UnsupportedH100Path, fa4_global_text_forward
from gemma4_fa4.reference import reference_attention
from gemma4_fa4.transformers_integration import (
    compile_gemma4_fa4_h100_static_cache_decode,
)

EXPERIMENT = "EXP-0025"
SCHEMA_VERSION = 1
GLOBAL_LAYER = 5
PROMPT_LENGTH = 32
CAPACITY = 65


def _cache_digest(tensor: torch.Tensor) -> str:
    raw = bytes(
        tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
    )
    return hashlib.sha256(raw).hexdigest()


def _cache_snapshot(cache: Any) -> dict[str, Any]:
    layer = cache.layers[GLOBAL_LAYER]
    tensors = (layer.keys, layer.values, layer.cumulative_length)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise AssertionError("global StaticCache layer is not initialized")
    return {
        "keys": layer.keys.detach().clone(),
        "values": layer.values.detach().clone(),
        "length_tensor": layer.cumulative_length.detach().clone(),
        "length": int(layer.cumulative_length.detach().item()),
        "pointers": tuple(tensor.untyped_storage().data_ptr() for tensor in tensors),
        "digests": {
            "keys": _cache_digest(layer.keys),
            "values": _cache_digest(layer.values),
            "length": _cache_digest(layer.cumulative_length),
        },
    }


def _assert_cache_unchanged(cache: Any, snapshot: dict[str, Any], *, label: str) -> None:
    current = _cache_snapshot(cache)
    if current["pointers"] != snapshot["pointers"]:
        raise AssertionError(f"{label} changed cache storage addresses")
    if current["digests"] != snapshot["digests"]:
        raise AssertionError(f"{label} mutated cache bytes")


def _early_static_cache(runtime, capacity: int):
    from transformers import StaticCache

    cache = StaticCache(config=runtime.config, max_cache_len=capacity)
    num_heads = [
        base.GEMMA4_31B.spec_for_layer(index).num_kv_heads
        for index in range(base.GEMMA4_31B.num_hidden_layers)
    ]
    head_dims = [
        base.GEMMA4_31B.spec_for_layer(index).head_dim_qk
        for index in range(base.GEMMA4_31B.num_hidden_layers)
    ]
    cache.early_initialization(
        batch_size=1,
        num_heads=num_heads,
        head_dim=head_dims,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    return cache


def _eager_layer_call(runtime, layer, cache, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    from transformers.masking_utils import create_causal_mask

    hidden, cos, sin, positions = inputs
    attention_mask = create_causal_mask(
        runtime.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=cache,
        position_ids=positions,
        layer_idx=GLOBAL_LAYER,
    )
    output, weights = layer(
        hidden,
        (cos, sin),
        attention_mask,
        {},
        past_key_values=cache,
        position_ids=positions,
        allow_flex_fallback=False,
    )
    if weights is not None:
        raise AssertionError("pinned global layer unexpectedly returned attention weights")
    return output


def _changed_slots(before: torch.Tensor, after: torch.Tensor) -> tuple[int, ...]:
    changed = (before != after).any(dim=(0, 1, 3))
    return tuple(int(index) for index in torch.nonzero(changed).flatten().cpu().tolist())


@torch.inference_mode()
def _reference_record(
    runtime,
    inputs: tuple[torch.Tensor, ...],
    cache: Any,
    candidate_output: torch.Tensor,
    candidate_lse: torch.Tensor,
) -> dict[str, Any]:
    hidden, cos, sin, _positions = inputs
    q, _new_k, _new_v = base._prepare_qkv(runtime, hidden, cos, sin)
    layer = cache.layers[GLOBAL_LAYER]
    active_k = layer.keys[:, :, : PROMPT_LENGTH + 1, :]
    active_v = layer.values[:, :, : PROMPT_LENGTH + 1, :]
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
    padded_q = torch.cat(
        (
            torch.zeros(
                (1, 32, 32, 512),
                dtype=q.dtype,
                device=q.device,
            ),
            q.transpose(1, 2),
        ),
        dim=1,
    )
    prepared_output, prepared_lse = fa4_global_text_forward(
        padded_q,
        active_k.transpose(1, 2),
        active_v.transpose(1, 2),
    )
    prepared_output = prepared_output[:, -1:, :, :]
    prepared_lse = prepared_lse[:, :, -1:]
    projected = runtime.layer.o_proj(prepared_output.reshape(1, 1, -1).contiguous())
    if not torch.equal(projected, candidate_output):
        raise AssertionError("compiled layer output differs from the prepared FA4 projection")
    if not torch.equal(prepared_lse, candidate_lse):
        raise AssertionError("compiled FP32 LSE differs from the prepared FA4 result")
    return {
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


def _run_first_discriminator(*, seed: int, backend: str) -> dict[str, Any]:
    base._require_h100()
    if backend != "inductor":
        raise ValueError("the EXP-0024 first discriminator is Inductor-only")
    cache_dirs = base._prepare_fresh_cache_dirs((backend,))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    runtime = base._make_family_runtime("global", seed=seed)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    eager_layer = Gemma4TextAttention(runtime.config, layer_idx=GLOBAL_LAYER).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    eager_layer.load_state_dict(runtime.layer.state_dict())
    eager_layer.eval()

    with torch.inference_mode():
        candidate_cache = _early_static_cache(runtime, CAPACITY)
        eager_cache = _early_static_cache(runtime, CAPACITY)
        foreign_cache = _early_static_cache(runtime, CAPACITY)
        prefill = base._make_inputs(runtime, PROMPT_LENGTH, seed=seed + 1)
        candidate_prefill = _eager_layer_call(
            runtime,
            runtime.layer,
            candidate_cache,
            prefill,
        )
        eager_prefill = _eager_layer_call(runtime, eager_layer, eager_cache, prefill)
        if not torch.equal(candidate_prefill, eager_prefill):
            raise AssertionError("weight-identical eager prefill twins diverged")

    candidate_before = _cache_snapshot(candidate_cache)
    eager_before = _cache_snapshot(eager_cache)
    if candidate_before["digests"] != eager_before["digests"]:
        raise AssertionError("weight-identical prefill caches differ before compiled decode")

    capture = guarded._CapturingBackend(backend)
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        candidate_cache,
        backend=capture,
    )
    decode = base._make_inputs(
        runtime,
        1,
        seed=seed + 2,
        position_start=PROMPT_LENGTH,
    )
    application_before = base._forward_application_snapshot()
    inductor_before = base._cache_inventory(cache_dirs["inductor"])
    with torch.inference_mode():
        candidate_output, candidate_weights = facade(
            decode[0],
            (decode[1], decode[2]),
            position_ids=decode[3],
        )
    torch.cuda.synchronize()
    if candidate_weights is not None or facade.last_lse is None:
        raise AssertionError("compiled cache facade returned an invalid layer/LSE contract")
    application_after_compiled = base._forward_application_snapshot()
    with torch.inference_mode():
        eager_output = _eager_layer_call(runtime, eager_layer, eager_cache, decode)
    torch.cuda.synchronize()
    application_after_eager = base._forward_application_snapshot()

    if not torch.equal(candidate_output, eager_output):
        maximum, mean = base._finite_error(candidate_output, eager_output)
        raise AssertionError(
            "compiled StaticCache output is not bitwise equal to eager twin "
            f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
        )
    candidate_after = _cache_snapshot(candidate_cache)
    eager_after = _cache_snapshot(eager_cache)
    if candidate_after["digests"] != eager_after["digests"]:
        raise AssertionError("compiled cache mutation differs byte-for-byte from eager twin")
    if candidate_after["pointers"] != candidate_before["pointers"]:
        raise AssertionError("compiled cache storage addresses changed")
    if eager_after["pointers"] != eager_before["pointers"]:
        raise AssertionError("eager cache storage addresses changed")
    if candidate_before["pointers"][0] == candidate_before["pointers"][1]:
        raise AssertionError("compiled cache K/V storage aliases")
    if candidate_after["length"] != PROMPT_LENGTH + 1:
        raise AssertionError("compiled cache counter did not advance exactly once")
    changed_k = _changed_slots(candidate_before["keys"], candidate_after["keys"])
    changed_v = _changed_slots(candidate_before["values"], candidate_after["values"])
    if changed_k != (PROMPT_LENGTH,) or changed_v != (PROMPT_LENGTH,):
        raise AssertionError(
            f"compiled cache changed wrong slots: K={changed_k}, V={changed_v}"
        )

    reference = _reference_record(
        runtime,
        decode,
        candidate_cache,
        candidate_output,
        facade.last_lse,
    )
    graph_break_count = base._graph_break_count()
    if len(capture.graphs) != 1 or graph_break_count != 0:
        raise AssertionError(
            "compiled StaticCache discriminator requires one backend attempt and zero graph breaks"
        )
    graph = capture.graphs[0]
    op_count = sum(
        "h100_global_static_cache_decode_fwd" in node["target"]
        and "gemma4_fa4" in node["target"]
        for node in graph["nodes"]
        if node["op"] == "call_function"
    )
    if op_count != 1:
        raise AssertionError(f"compiled cache graph requires exactly one cache op, got {op_count}")
    lifted_cache_sources = [
        node
        for node in graph["nodes"]
        if node["op"] == "get_attr"
        and node["target"] in {"cache_k", "cache_v", "cache_length"}
    ]
    if lifted_cache_sources:
        raise AssertionError(
            "compiled cache tensors must remain explicit inputs, not FX get_attr sources: "
            f"{lifted_cache_sources}"
        )
    cache_placeholders = [
        node
        for node in graph["nodes"]
        if node["op"] == "placeholder"
        and any(
            name in node["target"]
            for name in ("cache_k", "cache_v", "cache_length")
        )
    ]
    if len(cache_placeholders) != 3:
        raise AssertionError(
            "compiled cache graph requires three explicit cache placeholders, got "
            f"{cache_placeholders}"
        )
    unexpected_types = sorted(
        set(graph["example_input_types"]).difference({"Tensor", "Parameter", "SymInt", "int"})
    )
    if unexpected_types:
        raise AssertionError(f"compiled cache graph retained object inputs: {unexpected_types}")
    forbidden_sources = guarded._forbidden_inner_sources(capture.graphs)
    if forbidden_sources:
        raise AssertionError(f"compiled cache graph retained Python sources: {forbidden_sources}")
    added_applications = sorted(
        set(application_after_compiled).difference(application_before)
    )
    if len(added_applications) > 1:
        raise AssertionError("compiled Q1/K33 added more than one FA4 application key")
    if application_after_eager != application_after_compiled:
        raise AssertionError("eager twin added a different FA4 application key")

    rejection_inputs = base._make_inputs(
        runtime,
        1,
        seed=seed + 3,
        position_start=PROMPT_LENGTH + 1,
    )
    rejection_before = _cache_snapshot(candidate_cache)
    graph_count = len(capture.graphs)
    entry_count = facade.compiled_entry_count
    application_before_rejections = base._forward_application_snapshot()
    inductor_before_rejections = base._cache_inventory(cache_dirs["inductor"])
    bad_position = decode[3]
    try:
        with torch.inference_mode():
            facade(
                rejection_inputs[0],
                (rejection_inputs[1], rejection_inputs[2]),
                position_ids=bad_position,
            )
    except UnsupportedH100Path as exc:
        altered_position_error = str(exc)
    else:
        raise AssertionError("altered decode position was admitted")
    _assert_cache_unchanged(candidate_cache, rejection_before, label="altered position")
    try:
        with torch.inference_mode():
            facade(
                rejection_inputs[0],
                (rejection_inputs[1], rejection_inputs[2]),
                position_ids=rejection_inputs[3],
                past_key_values=foreign_cache,
            )
    except UnsupportedH100Path as exc:
        foreign_cache_error = str(exc)
    else:
        raise AssertionError("foreign StaticCache argument was admitted")
    _assert_cache_unchanged(candidate_cache, rejection_before, label="foreign cache")

    bound_layer = candidate_cache.layers[GLOBAL_LAYER]
    original_keys = bound_layer.keys
    bound_layer.keys = original_keys.clone()
    try:
        try:
            with torch.inference_mode():
                facade(
                    rejection_inputs[0],
                    (rejection_inputs[1], rejection_inputs[2]),
                    position_ids=rejection_inputs[3],
                )
        except UnsupportedH100Path as exc:
            rebound_root_error = str(exc)
        else:
            raise AssertionError("rebound StaticCache root was admitted")
    finally:
        bound_layer.keys = original_keys
    _assert_cache_unchanged(candidate_cache, rejection_before, label="rebound root")

    binding = facade._binding
    original_compiled_keys = binding.compiled_keys
    object.__setattr__(binding, "compiled_keys", original_compiled_keys.clone())
    try:
        try:
            with torch.inference_mode():
                facade(
                    rejection_inputs[0],
                    (rejection_inputs[1], rejection_inputs[2]),
                    position_ids=rejection_inputs[3],
                )
        except UnsupportedH100Path as exc:
            forged_view_error = str(exc)
        else:
            raise AssertionError("forged cache transport view was admitted")
    finally:
        object.__setattr__(binding, "compiled_keys", original_compiled_keys)
    _assert_cache_unchanged(candidate_cache, rejection_before, label="forged view")
    if (
        len(capture.graphs) != graph_count
        or facade.compiled_entry_count != entry_count
        or base._forward_application_snapshot() != application_before_rejections
        or base._cache_inventory(cache_dirs["inductor"]) != inductor_before_rejections
    ):
        raise AssertionError("a rejected request reached or changed the compiled boundary")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "scope": "global_layer5_q1_k33_first_discriminator",
        "backend": backend,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "prompt_length": PROMPT_LENGTH,
        "active_length": PROMPT_LENGTH + 1,
        "physical_capacity": CAPACITY,
        "backend_attempts": len(capture.graphs),
        "graph_breaks": graph_break_count,
        "cache_op_count": op_count,
        "compiled_entries": facade.compiled_entry_count,
        "whole_layer_bitwise": True,
        "cache_bytes_match_eager": True,
        "cache_changed_slots": {"k": list(changed_k), "v": list(changed_v)},
        "cache_addresses_stable": True,
        "cache_k_v_distinct": True,
        "reference": reference,
        "fa4_application_keys": {
            "before_count": len(application_before),
            "after_compiled_count": len(application_after_compiled),
            "added": added_applications,
            "eager_reused_compiled_key_set": True,
        },
        "inductor_cache": {
            "before": inductor_before,
            "after": base._cache_inventory(cache_dirs["inductor"]),
        },
        "graph": graph,
        "rejections": {
            "altered_position": altered_position_error,
            "foreign_cache": foreign_cache_error,
            "rebound_root": rebound_root_error,
            "forged_view": forged_view_error,
            "before_compiled_entry": True,
            "cache_bytes_unchanged": True,
        },
        "claims": {
            "compiled_prefill": False,
            "local_cache": False,
            "raw_torch_compile_layer": False,
            "performance": False,
            "b300": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24_033)
    parser.add_argument("--backend", choices=("inductor",), default="inductor")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = _run_first_discriminator(seed=args.seed, backend=args.backend)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
