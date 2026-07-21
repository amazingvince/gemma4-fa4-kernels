#!/usr/bin/env python3
"""EXP-0043 all-layer guarded compiled-StaticCache probe for H100."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import probe_h100_guarded_compile_facade as guarded
import probe_h100_static_cache_compile as cache_probe
import probe_h100_torch_compile as base
import torch

from gemma4_fa4.h100 import (
    UnsupportedH100Path,
    fa4_global_text_forward,
    fa4_local_varlen_forward,
)
from gemma4_fa4.model_spec import GEMMA4_31B
from gemma4_fa4.reference import reference_attention
from gemma4_fa4.transformers_integration import (
    compile_gemma4_fa4_h100_static_cache_decode,
)

EXPERIMENT = "EXP-0043"
BACKENDS = ("eager", "inductor")
PROMPT_LENGTH = 32
CAPACITY = 1026


def _family(layer_idx: int) -> str:
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    return "global" if spec.kind == "full_attention" else "local"


def _layer_seed(seed: int, layer_idx: int) -> int:
    return seed + layer_idx * 10


def _cache_snapshot(cache: Any, layer_idx: int, *, clone: bool) -> dict[str, Any]:
    layer = cache.layers[layer_idx]
    keys = layer.keys.detach().clone() if clone else layer.keys
    values = layer.values.detach().clone() if clone else layer.values
    length = layer.cumulative_length.detach().clone() if clone else layer.cumulative_length
    return {
        "keys": keys,
        "values": values,
        "length": length,
        "logical_length": int(layer.cumulative_length.detach().item()),
        "absolute_length": getattr(layer, "cumulative_length_int", None),
        "pointers": tuple(
            tensor.untyped_storage().data_ptr()
            for tensor in (layer.keys, layer.values, layer.cumulative_length)
        ),
        "versions": tuple(
            _tensor_version(tensor)
            for tensor in (layer.keys, layer.values, layer.cumulative_length)
        ),
    }


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None


def _weight_identical_eager_layer(runtime, layer_idx: int):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    layer = Gemma4TextAttention(runtime.config, layer_idx=layer_idx).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer.load_state_dict(runtime.layer.state_dict())
    layer.eval()
    return layer


def _eager_layer_call(
    runtime,
    layer,
    cache: Any,
    inputs: tuple[torch.Tensor, ...],
    *,
    layer_idx: int,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    hidden, cos, sin, positions = inputs
    attention_mask = runtime.mask_builder(
        runtime.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=cache,
        position_ids=positions,
        layer_idx=layer_idx,
    )
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    output, weights = layer(
        hidden,
        (cos, sin),
        attention_mask,
        shared_kv_states,
        past_key_values=cache,
        position_ids=positions,
        allow_flex_fallback=False,
    )
    if weights is not None:
        raise AssertionError(f"layer {layer_idx} unexpectedly returned attention weights")
    expected_shared = bool(layer.store_full_length_kv)
    if bool(shared_kv_states) is not expected_shared:
        raise AssertionError(f"layer {layer_idx} terminal shared-KV marker was not exact")
    if shared_kv_states and set(shared_kv_states) != {layer.layer_type}:
        raise AssertionError(f"layer {layer_idx} wrote the wrong shared-KV family")
    return output, shared_kv_states


def _changed_slots(before: torch.Tensor, after: torch.Tensor) -> tuple[int, ...]:
    changed = (before.view(torch.int16) != after.view(torch.int16)).any(dim=(0, 1, 3))
    return tuple(int(index) for index in torch.nonzero(changed).flatten().cpu().tolist())


@torch.inference_mode()
def _prepared_reference(
    runtime,
    inputs: tuple[torch.Tensor, ...],
    cache: Any,
    candidate_output: torch.Tensor,
    candidate_lse: torch.Tensor,
    *,
    layer_idx: int,
) -> dict[str, Any]:
    hidden, cos, sin, _positions = inputs
    query, _new_key, _new_value = base._prepare_qkv(runtime, hidden, cos, sin)
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    cache_layer = cache.layers[layer_idx]
    active_length = int(cache_layer.cumulative_length.detach().item())
    active_k = cache_layer.keys[:, :, :active_length, :]
    active_v = cache_layer.values[:, :, :active_length, :]
    reference_output, reference_lse = reference_attention(
        query,
        active_k,
        active_v,
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
        allow_vision_bidirectional=False,
        q_start=active_length - 1,
        upcast=torch.float32,
        return_lse=True,
    )
    if spec.kind == "full_attention":
        padded_q = torch.cat(
            (
                torch.zeros(
                    (1, active_length - 1, spec.num_q_heads, spec.head_dim_qk),
                    dtype=query.dtype,
                    device=query.device,
                ),
                query.transpose(1, 2),
            ),
            dim=1,
        )
        prepared_output, prepared_lse = fa4_global_text_forward(
            padded_q,
            active_k.transpose(1, 2),
            active_v.transpose(1, 2),
        )
        prepared_output = prepared_output[:, -1:, :, :]
        prepared_lse = prepared_lse[:, :, -1:].contiguous()
        output_atol = base.GLOBAL_OUTPUT_ATOL
        output_rtol = base.GLOBAL_OUTPUT_RTOL
        lse_atol = base.GLOBAL_LSE_ATOL
    else:
        packed_q = query.transpose(1, 2).reshape(1, spec.num_q_heads, spec.head_dim_qk)
        packed_k = active_k.transpose(1, 2).reshape(
            active_length,
            spec.num_kv_heads,
            spec.head_dim_qk,
        )
        packed_v = active_v.transpose(1, 2).reshape(
            active_length,
            spec.num_kv_heads,
            spec.head_dim_v,
        )
        cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, active_length], dtype=torch.int32, device="cuda")
        prepared_output, raw_lse = fa4_local_varlen_forward(
            packed_q,
            packed_k,
            packed_v,
            cu_q,
            cu_k,
            max_seqlen_q=1,
            max_seqlen_k=active_length,
        )
        prepared_output = prepared_output.reshape(1, 1, spec.num_q_heads, spec.head_dim_v)
        prepared_lse = raw_lse.unsqueeze(0)
        output_atol = base.LOCAL_OUTPUT_ATOL
        output_rtol = base.LOCAL_OUTPUT_RTOL
        lse_atol = base.LOCAL_LSE_ATOL
    projected = runtime.layer.o_proj(prepared_output.reshape(1, 1, -1).contiguous())
    if not torch.equal(projected, candidate_output):
        raise AssertionError(f"layer {layer_idx} projection transport is not bitwise")
    if not torch.equal(prepared_lse, candidate_lse):
        raise AssertionError(f"layer {layer_idx} compiled/prepared LSE is not bitwise")
    return {
        "active_length": active_length,
        "prepared_output": base._assert_close(
            prepared_output.transpose(1, 2),
            reference_output,
            atol=output_atol,
            rtol=output_rtol,
        ),
        "prepared_lse": base._assert_close(
            prepared_lse,
            reference_lse,
            atol=lse_atol,
            rtol=0.0,
        ),
        "projection_transport_bitwise": True,
        "compiled_lse_bitwise": True,
    }


def _prefill_all_layers(
    candidate_cache: Any, eager_cache: Any, *, seed: int
) -> list[dict[str, Any]]:
    records = []
    for layer_idx in range(GEMMA4_31B.num_hidden_layers):
        family = _family(layer_idx)
        runtime = base._make_family_runtime(
            family,
            seed=_layer_seed(seed, layer_idx),
            layer_idx=layer_idx,
        )
        eager_layer = _weight_identical_eager_layer(runtime, layer_idx)
        inputs = base._make_inputs(runtime, PROMPT_LENGTH, seed=seed + 1 + layer_idx)
        with torch.inference_mode():
            candidate_output, candidate_shared = _eager_layer_call(
                runtime,
                runtime.layer,
                candidate_cache,
                inputs,
                layer_idx=layer_idx,
            )
            eager_output, eager_shared = _eager_layer_call(
                runtime,
                eager_layer,
                eager_cache,
                inputs,
                layer_idx=layer_idx,
            )
        if not torch.equal(candidate_output, eager_output):
            raise AssertionError(f"layer {layer_idx} eager-prefill twins diverged")
        if bool(candidate_shared) is not bool(eager_shared):
            raise AssertionError(f"layer {layer_idx} eager-prefill shared markers diverged")
        candidate_snapshot = _cache_snapshot(candidate_cache, layer_idx, clone=False)
        eager_snapshot = _cache_snapshot(eager_cache, layer_idx, clone=False)
        if (
            candidate_snapshot["logical_length"] != PROMPT_LENGTH
            or candidate_snapshot["absolute_length"]
            != (PROMPT_LENGTH if family == "local" else None)
            or not torch.equal(candidate_snapshot["keys"], eager_snapshot["keys"])
            or not torch.equal(candidate_snapshot["values"], eager_snapshot["values"])
        ):
            raise AssertionError(f"layer {layer_idx} eager-prefill cache state diverged")
        records.append(
            {
                "layer_idx": layer_idx,
                "family": family,
                "bitwise": True,
                "terminal_shared_kv_marker": bool(candidate_shared),
            }
        )
        del candidate_output, eager_output, eager_layer, inputs, runtime
        gc.collect()
    torch.cuda.synchronize()
    return records


def _run_decode_layer(
    layer_idx: int,
    *,
    candidate_cache: Any,
    eager_cache: Any,
    capture: guarded._CapturingBackend,
    seed: int,
) -> dict[str, Any]:
    family = _family(layer_idx)
    runtime = base._make_family_runtime(
        family,
        seed=_layer_seed(seed, layer_idx),
        layer_idx=layer_idx,
    )
    eager_layer = _weight_identical_eager_layer(runtime, layer_idx)
    inputs = base._make_inputs(
        runtime,
        1,
        seed=seed + 2 + layer_idx,
        position_start=PROMPT_LENGTH,
    )
    candidate_before = _cache_snapshot(candidate_cache, layer_idx, clone=True)
    eager_before = _cache_snapshot(eager_cache, layer_idx, clone=True)
    applications_before = base._forward_application_snapshot()
    facade = compile_gemma4_fa4_h100_static_cache_decode(
        runtime.layer,
        candidate_cache,
        backend=capture,
    )
    with torch.inference_mode():
        candidate_output, candidate_weights = facade(
            inputs[0],
            (inputs[1], inputs[2]),
            position_ids=inputs[3],
        )
    torch.cuda.synchronize()
    applications_after = base._forward_application_snapshot()
    with torch.inference_mode():
        eager_output, eager_shared = _eager_layer_call(
            runtime,
            eager_layer,
            eager_cache,
            inputs,
            layer_idx=layer_idx,
        )
    torch.cuda.synchronize()
    if candidate_weights is not None or facade.last_lse is None or facade.compiled_entry_count != 1:
        raise AssertionError(f"layer {layer_idx} violated the compiled facade output contract")
    if not torch.equal(candidate_output, eager_output):
        maximum, mean = base._finite_error(candidate_output, eager_output)
        raise AssertionError(
            f"layer {layer_idx} is not bitwise eager-equal "
            f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
        )
    candidate_after = _cache_snapshot(candidate_cache, layer_idx, clone=False)
    eager_after = _cache_snapshot(eager_cache, layer_idx, clone=False)
    if candidate_before["pointers"] != candidate_after["pointers"]:
        raise AssertionError(f"layer {layer_idx} rebound cache storage")
    for name in ("keys", "values"):
        if _changed_slots(candidate_before[name], candidate_after[name]) != (PROMPT_LENGTH,):
            raise AssertionError(f"layer {layer_idx} mutated unexpected {name} slots")
        if not torch.equal(candidate_after[name], eager_after[name]):
            raise AssertionError(f"layer {layer_idx} {name} cache differs from eager")
    if (
        candidate_after["logical_length"] != PROMPT_LENGTH + 1
        or candidate_after["logical_length"] != eager_after["logical_length"]
        or candidate_after["absolute_length"] != eager_after["absolute_length"]
        or not torch.equal(candidate_after["length"], eager_after["length"])
    ):
        raise AssertionError(f"layer {layer_idx} cache counters differ from eager")
    reference = _prepared_reference(
        runtime,
        inputs,
        candidate_cache,
        candidate_output,
        facade.last_lse,
        layer_idx=layer_idx,
    )
    record = {
        "layer_idx": layer_idx,
        "family": family,
        "whole_layer_bitwise": True,
        "cache_bitwise": True,
        "mutated_slot": PROMPT_LENGTH,
        "logical_length": candidate_after["logical_length"],
        "absolute_length": candidate_after["absolute_length"],
        "terminal_shared_kv_marker": bool(eager_shared),
        "fa4_application_keys_added": sorted(
            set(applications_after).difference(applications_before)
        ),
        "reference": reference,
    }
    del candidate_output, eager_output, eager_layer, facade, inputs, runtime
    del candidate_before, eager_before
    gc.collect()
    return record


def _audit_graphs(capture: guarded._CapturingBackend) -> dict[str, Any]:
    if base._graph_break_count() != 0:
        raise AssertionError(f"{capture.name} all-layer cache sweep produced graph breaks")
    if len(capture.graphs) != 2:
        raise AssertionError(
            f"{capture.name} requires exactly two family cache graphs, got {len(capture.graphs)}"
        )
    op_targets = []
    for graph in capture.graphs:
        targets = [
            node["target"]
            for node in graph["nodes"]
            if node["op"] == "call_function"
            and "gemma4_fa4" in node["target"]
            and "static_cache_decode_fwd" in node["target"]
        ]
        if len(targets) != 1:
            raise AssertionError(f"cache graph requires one project cache op: {targets}")
        if any(node["op"] == "get_attr" for node in graph["nodes"]):
            raise AssertionError("cache graph captured module/cache state through get_attr")
        op_targets.extend(targets)
    forbidden = guarded._forbidden_inner_sources(capture.graphs)
    if forbidden:
        raise AssertionError(f"cache graphs captured forbidden sources: {forbidden}")
    if not any("local" in target for target in op_targets) or not any(
        "global" in target for target in op_targets
    ):
        raise AssertionError(f"cache graph families are incomplete: {op_targets}")
    return {
        "graph_count": len(capture.graphs),
        "graph_break_count": 0,
        "cache_op_targets": op_targets,
        "forbidden_inner_sources": forbidden,
        "graphs": capture.graphs,
    }


def _mutation_rejections(
    candidate_cache: Any,
    *,
    seed: int,
) -> dict[str, str]:
    errors = {}
    for layer_idx, replacement in ((1, 2), (11, 17)):
        family = _family(layer_idx)
        runtime = base._make_family_runtime(
            family,
            seed=_layer_seed(seed, layer_idx),
            layer_idx=layer_idx,
        )
        capture = guarded._CapturingBackend("eager")
        facade = compile_gemma4_fa4_h100_static_cache_decode(
            runtime.layer,
            candidate_cache,
            backend=capture,
        )
        inputs = base._make_inputs(
            runtime,
            1,
            seed=seed + 3 + layer_idx,
            position_start=PROMPT_LENGTH + 1,
        )
        snapshot = _cache_snapshot(candidate_cache, layer_idx, clone=True)
        applications = base._forward_application_snapshot()
        runtime.layer.layer_idx = replacement
        try:
            with torch.inference_mode():
                facade(inputs[0], (inputs[1], inputs[2]), position_ids=inputs[3])
        except UnsupportedH100Path as exc:
            errors[family] = str(exc)
        else:
            raise AssertionError(f"{family} same-family layer mutation was admitted")
        current = _cache_snapshot(candidate_cache, layer_idx, clone=False)
        if (
            facade.compiled_entry_count != 0
            or capture.graphs
            or base._forward_application_snapshot() != applications
            or current["pointers"] != snapshot["pointers"]
            or not torch.equal(current["keys"], snapshot["keys"])
            or not torch.equal(current["values"], snapshot["values"])
            or not torch.equal(current["length"], snapshot["length"])
            or current["absolute_length"] != snapshot["absolute_length"]
        ):
            raise AssertionError(f"{family} mutation rejection reached or changed cache state")
        del facade, inputs, runtime, snapshot
        gc.collect()
    return errors


def _run_backend(name: str, *, seed: int) -> dict[str, Any]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    bootstrap = base._make_family_runtime("local", seed=seed, layer_idx=0)
    candidate_cache = cache_probe._early_static_cache(bootstrap, CAPACITY)
    eager_cache = cache_probe._early_static_cache(bootstrap, CAPACITY)
    del bootstrap
    prefill = _prefill_all_layers(candidate_cache, eager_cache, seed=seed)
    applications_before = base._forward_application_snapshot()
    capture = guarded._CapturingBackend(name)
    layers = [
        _run_decode_layer(
            layer_idx,
            candidate_cache=candidate_cache,
            eager_cache=eager_cache,
            capture=capture,
            seed=seed,
        )
        for layer_idx in range(GEMMA4_31B.num_hidden_layers)
    ]
    applications_after = base._forward_application_snapshot()
    family_additions: dict[str, set[str]] = {"local": set(), "global": set()}
    for record in layers:
        family_additions[record["family"]].update(record["fa4_application_keys_added"])
    if any(len(keys) > 1 for keys in family_additions.values()):
        raise AssertionError(f"layer index added cache application classes: {family_additions}")
    graph_audit = _audit_graphs(capture)
    mutations = _mutation_rejections(candidate_cache, seed=seed)
    report = {
        "status": "passed",
        "backend": name,
        "prefill": prefill,
        "layers": layers,
        "graph_audit": graph_audit,
        "fa4_application_keys": {
            "before": list(applications_before),
            "after": list(applications_after),
            "added_by_family": {family: sorted(keys) for family, keys in family_additions.items()},
        },
        "same_family_mutation_rejections": mutations,
    }
    del candidate_cache, eager_cache
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=43000)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    base._require_h100()
    selected = BACKENDS if args.backend == "all" else (args.backend,)
    cache_dirs = base._prepare_fresh_cache_dirs(selected)
    backends = {
        name: _run_backend(name, seed=args.seed + index * 100_000)
        for index, name in enumerate(selected)
    }
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "passed",
        "boundary": "guarded_static_cache_q1_all_60_layers",
        "request": {
            "backends": list(selected),
            "prompt_length": PROMPT_LENGTH,
            "decode_length": PROMPT_LENGTH + 1,
            "capacity": CAPACITY,
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
            "caches": {name: base._cache_inventory(path) for name, path in cache_dirs.items()},
        },
        "layer_count": GEMMA4_31B.num_hidden_layers,
        "local_layer_count": sum(_family(index) == "local" for index in range(60)),
        "global_layer_count": sum(_family(index) == "global" for index in range(60)),
        "backends": backends,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        raise RuntimeError("EXP-0043 requires an isolated TORCHINDUCTOR_CACHE_DIR")
    main()
