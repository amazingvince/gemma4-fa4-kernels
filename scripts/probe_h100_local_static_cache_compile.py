#!/usr/bin/env python3
"""EXP-0027 first discriminator for compiled local sliding-cache decode.

This is a correctness and compiler-boundary probe for the exact pinned local
layer 0, B1/Q1, text-only inference envelope.  It makes no performance,
compiled-prefill, multimodal-cache, training, B300, or full-model claim.
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
import probe_h100_static_cache_compile as cache_probe
import probe_h100_torch_compile as base
import torch

from gemma4_fa4.h100 import fa4_local_varlen_forward
from gemma4_fa4.reference import reference_attention
from gemma4_fa4.transformers_integration import (
    Gemma4H100CompiledLocalStaticCacheDecodeFacade,
    compile_gemma4_fa4_h100_static_cache_decode,
)

EXPERIMENT = "EXP-0027"
SCHEMA_VERSION = 1
LOCAL_LAYER = 0
CAPACITY = 1024
PROMPT_LENGTH = 1023
DECODE_STEPS = 3
LOCAL_OUTPUT_ATOL = 0.03125
LOCAL_OUTPUT_RTOL = 0.02
LOCAL_LSE_ATOL = 0.125


def _cache_digest(tensor: torch.Tensor) -> str:
    raw = bytes(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
    return hashlib.sha256(raw).hexdigest()


def _cache_snapshot(cache: Any, *, clone: bool) -> dict[str, Any]:
    layer = cache.layers[LOCAL_LAYER]
    tensors = (layer.keys, layer.values, layer.cumulative_length)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise AssertionError("local StaticSlidingWindowLayer is not initialized")
    return {
        "keys": layer.keys.detach().clone() if clone else layer.keys,
        "values": layer.values.detach().clone() if clone else layer.values,
        "tensor_length": int(layer.cumulative_length.detach().item()),
        "absolute_length": layer.cumulative_length_int,
        "versions": tuple(tensor._version for tensor in tensors),
        "pointers": tuple(tensor.untyped_storage().data_ptr() for tensor in tensors),
        "digests": {
            "keys": _cache_digest(layer.keys),
            "values": _cache_digest(layer.values),
            "length": _cache_digest(layer.cumulative_length),
        },
    }


def _eager_layer_call(
    runtime,
    layer,
    cache: Any,
    inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    hidden, cos, sin, positions = inputs
    attention_mask = runtime.mask_builder(
        runtime.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=cache,
        position_ids=positions,
        layer_idx=LOCAL_LAYER,
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
        raise AssertionError("pinned local layer unexpectedly returned attention weights")
    return output


@torch.inference_mode()
def _prepared_reference(
    runtime,
    inputs: tuple[torch.Tensor, ...],
    cache: Any,
    candidate_output: torch.Tensor,
    candidate_lse: torch.Tensor,
) -> dict[str, Any]:
    hidden, cos, sin, _positions = inputs
    query, _new_key, _new_value = base._prepare_qkv(runtime, hidden, cos, sin)
    cache_layer = cache.layers[LOCAL_LAYER]
    active_length = min(cache_layer.cumulative_length_int, CAPACITY)
    packed_q = query.transpose(1, 2).reshape(1, 32, 256)
    packed_k = cache_layer.keys[:, :, :active_length, :].transpose(1, 2).reshape(
        active_length, 16, 256
    )
    packed_v = cache_layer.values[:, :, :active_length, :].transpose(1, 2).reshape(
        active_length, 16, 256
    )
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, active_length], dtype=torch.int32, device="cuda")
    prepared_output, prepared_lse = fa4_local_varlen_forward(
        packed_q,
        packed_k,
        packed_v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=active_length,
    )
    projected = runtime.layer.o_proj(prepared_output.reshape(1, 1, -1).contiguous())
    shaped_lse = prepared_lse.unsqueeze(0)
    if not torch.equal(projected, candidate_output):
        raise AssertionError("compiled local layer output differs from prepared FA4 replay")
    if not torch.equal(shaped_lse, candidate_lse):
        raise AssertionError("compiled local FP32 LSE differs from prepared FA4 replay")

    reference_output, reference_lse = reference_attention(
        query,
        cache_layer.keys[:, :, :active_length, :],
        cache_layer.values[:, :, :active_length, :],
        softmax_scale=1.0,
        sliding_window=1024,
        allow_vision_bidirectional=False,
        q_start=active_length - 1,
        upcast=torch.float32,
        return_lse=True,
    )
    prepared_bhsd = prepared_output.reshape(1, 1, 32, 256).transpose(1, 2)
    return {
        "active_k": active_length,
        "prepared_output": base._assert_close(
            prepared_bhsd,
            reference_output,
            atol=LOCAL_OUTPUT_ATOL,
            rtol=LOCAL_OUTPUT_RTOL,
        ),
        "prepared_lse": base._assert_close(
            shaped_lse,
            reference_lse,
            atol=LOCAL_LSE_ATOL,
            rtol=0.0,
        ),
        "projection_transport_bitwise": True,
        "compiled_lse_bitwise": True,
    }


def _assert_step_mutation(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    position: int,
) -> dict[str, Any]:
    if before["pointers"] != after["pointers"]:
        raise AssertionError("local cache root addresses changed")
    if position < CAPACITY:
        if after["tensor_length"] != before["tensor_length"] + 1:
            raise AssertionError("underfill/boundary CUDA counter did not advance")
        for name in ("keys", "values"):
            if not torch.equal(
                before[name][:, :, :position, :],
                after[name][:, :, :position, :],
            ):
                raise AssertionError(f"boundary update changed the retained {name} prefix")
        transition = "boundary-fill"
    else:
        if after["tensor_length"] != CAPACITY:
            raise AssertionError("rollover CUDA counter did not remain saturated")
        for name in ("keys", "values"):
            if not torch.equal(after[name][:, :, :-1, :], before[name][:, :, 1:, :]):
                raise AssertionError(f"rollover did not shift {name} left exactly")
        transition = "roll-left"
    if after["absolute_length"] != before["absolute_length"] + 1:
        raise AssertionError("Python absolute count did not advance exactly once")
    if after["versions"][0] == before["versions"][0] or after["versions"][1] == before["versions"][1]:
        raise AssertionError("declared local K/V mutation did not advance tensor versions")
    counter_changed = after["versions"][2] != before["versions"][2]
    if counter_changed is not (position < CAPACITY):
        raise AssertionError("CUDA counter version disagrees with saturation branch")
    return {
        "transition": transition,
        "absolute_before": before["absolute_length"],
        "absolute_after": after["absolute_length"],
        "tensor_before": before["tensor_length"],
        "tensor_after": after["tensor_length"],
        "versions_before": list(before["versions"]),
        "versions_after": list(after["versions"]),
    }


def _graph_record(capture: guarded._CapturingBackend) -> dict[str, Any]:
    if len(capture.graphs) != 1:
        raise AssertionError(f"expected one compiled graph, found {len(capture.graphs)}")
    graph = capture.graphs[0]
    cache_ops = [
        node
        for node in graph["nodes"]
        if "h100_local_static_cache_decode_fwd" in node["target"]
    ]
    if len(cache_ops) != 1:
        raise AssertionError(f"expected one local cache op, found {len(cache_ops)}")
    get_attrs = [node for node in graph["nodes"] if node["op"] == "get_attr"]
    if get_attrs:
        raise AssertionError("compiled local cache graph captured module/cache state")
    forbidden = guarded._forbidden_inner_sources(capture.graphs)
    if forbidden:
        raise AssertionError(f"compiled local cache graph contains forbidden sources: {forbidden}")
    placeholders = [node for node in graph["nodes"] if node["op"] == "placeholder"]
    if len(placeholders) != 13:
        raise AssertionError(
            f"local cache graph must expose all 13 tensor arguments, found {len(placeholders)}"
        )
    return {
        "attempts": len(capture.graphs),
        "cache_op_count": len(cache_ops),
        "get_attr_count": len(get_attrs),
        "placeholder_count": len(placeholders),
        "forbidden_sources": forbidden,
        "graph": graph,
    }


def _run_first_discriminator(*, seed: int) -> dict[str, Any]:
    base._require_h100()
    cache_dirs = base._prepare_fresh_cache_dirs(("inductor",))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    runtime = base._make_family_runtime("local", seed=seed)

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    eager_layer = Gemma4TextAttention(runtime.config, layer_idx=LOCAL_LAYER).to(
        device="cuda", dtype=torch.bfloat16
    )
    eager_layer.load_state_dict(runtime.layer.state_dict())
    eager_layer.eval()
    candidate_cache = cache_probe._early_static_cache(runtime, CAPACITY)
    eager_cache = cache_probe._early_static_cache(runtime, CAPACITY)

    with torch.inference_mode():
        prefill = base._make_inputs(runtime, PROMPT_LENGTH, seed=seed + 1)
        candidate_prefill = _eager_layer_call(
            runtime, runtime.layer, candidate_cache, prefill
        )
        eager_prefill = _eager_layer_call(runtime, eager_layer, eager_cache, prefill)
    if not torch.equal(candidate_prefill, eager_prefill):
        raise AssertionError("weight-identical local eager prefills diverged")
    candidate_prefill_state = _cache_snapshot(candidate_cache, clone=False)
    eager_prefill_state = _cache_snapshot(eager_cache, clone=False)
    if candidate_prefill_state["digests"] != eager_prefill_state["digests"]:
        raise AssertionError("weight-identical local prefill cache bytes diverged")

    capture = guarded._CapturingBackend("inductor")
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        candidate_cache,
        backend=capture,
    )
    if not isinstance(facade, Gemma4H100CompiledLocalStaticCacheDecodeFacade):
        raise AssertionError("local constructor returned the wrong guarded facade type")
    binding = facade._binding
    root_pointers = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (binding.keys, binding.values, binding.cumulative_length)
    )
    view_pointers = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (binding.compiled_keys, binding.compiled_values, binding.compiled_length)
    )
    if root_pointers != view_pointers:
        raise AssertionError("compiled local cache views do not cover the exact roots")
    if any(
        getattr(view, "_dynamo_static_input_type", None) is not None
        for view in (binding.compiled_keys, binding.compiled_values, binding.compiled_length)
    ):
        raise AssertionError("compiled local cache views inherited a static-input marker")

    application_before = base._forward_application_snapshot()
    steps = []
    for index in range(DECODE_STEPS):
        position = PROMPT_LENGTH + index
        inputs = base._make_inputs(
            runtime,
            1,
            seed=seed + 2 + index,
            position_start=position,
        )
        candidate_before = _cache_snapshot(candidate_cache, clone=True)
        eager_before = _cache_snapshot(eager_cache, clone=True)
        with torch.inference_mode():
            candidate_output, candidate_weights = facade(
                inputs[0],
                (inputs[1], inputs[2]),
                position_ids=inputs[3],
            )
        if candidate_weights is not None or facade.last_lse is None:
            raise AssertionError("compiled local facade returned an invalid output/LSE contract")
        with torch.inference_mode():
            eager_output = _eager_layer_call(runtime, eager_layer, eager_cache, inputs)
        torch.cuda.synchronize()
        if not torch.equal(candidate_output, eager_output):
            maximum, mean = base._finite_error(candidate_output, eager_output)
            raise AssertionError(
                "compiled local output is not bitwise equal to eager twin "
                f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
            )
        candidate_after = _cache_snapshot(candidate_cache, clone=True)
        eager_after = _cache_snapshot(eager_cache, clone=True)
        if candidate_after["digests"] != eager_after["digests"]:
            raise AssertionError("compiled local cache differs byte-for-byte from eager twin")
        candidate_mutation = _assert_step_mutation(
            candidate_before, candidate_after, position=position
        )
        eager_mutation = _assert_step_mutation(eager_before, eager_after, position=position)
        reference = _prepared_reference(
            runtime,
            inputs,
            candidate_cache,
            candidate_output,
            facade.last_lse,
        )
        steps.append(
            {
                "position": position,
                "candidate_mutation": candidate_mutation,
                "eager_mutation": eager_mutation,
                "cache_bitwise": True,
                "whole_layer_bitwise": True,
                "reference": reference,
            }
        )

    graph_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    if graph_breaks:
        raise AssertionError(f"local cache discriminator produced {graph_breaks} graph breaks")
    graph = _graph_record(capture)
    application_after = base._forward_application_snapshot()
    application_delta = sorted(set(application_after).difference(application_before))
    if len(application_delta) != 1:
        raise AssertionError(
            "three full-window local decodes must add exactly one FA4 application class"
        )
    if facade.compiled_entry_count != DECODE_STEPS:
        raise AssertionError("facade compiled-entry counter disagrees with decode calls")
    return {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "mode": "first_discriminator",
        "seed": seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "prompt_length": PROMPT_LENGTH,
        "steps": steps,
        "graph_breaks": graph_breaks,
        "graph": graph,
        "fa4_application_delta": application_delta,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=27_001)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = _run_first_discriminator(seed=args.seed)
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
