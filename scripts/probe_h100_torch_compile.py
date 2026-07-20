#!/usr/bin/env python3
"""EXP-0022 H100 fullgraph probe for the pinned Gemma 4 attention layers.

This is a correctness and compiler-boundary probe, not a benchmark.  It keeps
the exact locked model width and runs actual pinned ``Gemma4TextAttention``
layers 0 and 5 through the registered no-cache FA4 custom operators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import torch

from gemma4_fa4 import h100_torch_ops
from gemma4_fa4.model_spec import GEMMA4_31B, AttentionLayerSpec
from gemma4_fa4.reference import reference_attention
from gemma4_fa4.transformers_integration import (
    BACKEND_NAME,
    register_gemma4_fa4_h100,
)

EXPERIMENT = "EXP-0022"
SCHEMA_VERSION = 1
PINNED_TORCH_VERSION = "2.8.0+cu128"
DEFAULT_LENGTHS = (1, 32, 33, 1023, 1024)
FAMILY_LAYERS = {"local": 0, "global": 5}
BACKENDS = ("eager", "inductor")
FULL_LAYER_OUTPUT_ATOL = 0.0625
FULL_LAYER_OUTPUT_RTOL = 0.02

LOCAL_OUTPUT_ATOL = 0.03125
LOCAL_OUTPUT_RTOL = 0.02
LOCAL_LSE_ATOL = 0.125
GLOBAL_OUTPUT_ATOL = 0.0625
GLOBAL_OUTPUT_RTOL = 0.03
GLOBAL_LSE_ATOL = 0.25

ROOT = Path(__file__).resolve().parents[1]


class _FamilyRuntime(NamedTuple):
    name: str
    layer_idx: int
    spec: AttentionLayerSpec
    config: Any
    layer: torch.nn.Module
    rotary: torch.nn.Module
    mask_builder: Callable[..., Any]
    apply_rotary_pos_emb: Callable[..., torch.Tensor]
    custom_op: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    custom_op_fragment: str
    layer_custom_op: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    layer_opcheck_target: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    layer_custom_op_fragment: str


class _CapturingBackend:
    """Capture Dynamo graphs while delegating to the named stock backend."""

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
            "backend_grad_enabled": torch.is_grad_enabled(),
            "backend_inference_mode": torch.is_inference_mode_enabled(),
            "example_input_requires_grad": [
                bool(value.requires_grad)
                for value in example_inputs
                if isinstance(value, torch.Tensor)
            ],
            "example_input_types": [type(value).__name__ for value in example_inputs],
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


def _parse_lengths(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one sequence length is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("sequence lengths must not repeat")
    if any(value < 1 or value > 1024 for value in values):
        raise argparse.ArgumentTypeError("sequence lengths must be in the closed range 1..1024")
    return values


def _require_h100() -> None:
    if torch.__version__ != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"{EXPERIMENT} requires PyTorch {PINNED_TORCH_VERSION}, found {torch.__version__}"
        )
    if not h100_torch_ops.CUSTOM_OPS_AVAILABLE:
        raise RuntimeError(f"{EXPERIMENT} requires torch.library custom_op/register_fake")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{EXPERIMENT} requires CUDA on an NVIDIA H100 (SM90)")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if capability != (9, 0) or "H100" not in name.upper():
        raise RuntimeError(
            f"{EXPERIMENT} requires an H100 (SM90), found {name!r} with {capability}"
        )


def _prepare_fresh_cache_dirs(backends: Sequence[str]) -> dict[str, Path]:
    if os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED") != "1":
        raise RuntimeError("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 is required")
    raw = {"fa4": os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")}
    if "inductor" in backends:
        raw["inductor"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    paths: dict[str, Path] = {}
    for name, value in raw.items():
        if not value:
            raise RuntimeError(f"an explicit isolated {name} cache directory is required")
        path = Path(value).expanduser().resolve()
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"the {name} cache directory must be fresh and empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def _cache_inventory(path: Path) -> dict[str, Any]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "path": str(path),
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def _application_key_digests(keys: Sequence[object]) -> tuple[str, ...]:
    digests = tuple(sorted(hashlib.sha256(pickle.dumps(key)).hexdigest() for key in keys))
    if len(set(digests)) != len(digests):
        raise AssertionError("distinct FA4 forward application keys produced duplicate digests")
    return digests


def _forward_application_snapshot() -> tuple[str, ...]:
    from flash_attn.cute.interface import _flash_attn_fwd

    backing = getattr(_flash_attn_fwd.compile_cache, "cache", None)
    if not isinstance(backing, dict):
        raise AssertionError("the pinned FA4 forward cache no longer exposes its key map")
    return _application_key_digests(tuple(backing))


def _locked_text_config():
    try:
        from transformers import Gemma4TextConfig
    except Exception as exc:
        raise RuntimeError("the pinned Transformers checkout is required") from exc

    payload = json.loads((ROOT / "configs/model/gemma4-31b.lock.json").read_text())
    config = Gemma4TextConfig(**payload["text_config"])
    config._attn_implementation = BACKEND_NAME
    if config.hidden_size != GEMMA4_31B.hidden_size:
        raise AssertionError("compile probe must retain the locked hidden_size=5376")
    return config


def _make_family_runtime(family: str, *, seed: int) -> _FamilyRuntime:
    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextAttention,
            Gemma4TextRotaryEmbedding,
            apply_rotary_pos_emb,
        )
    except Exception as exc:
        raise RuntimeError("the pinned Transformers Gemma 4 implementation is required") from exc

    if family not in FAMILY_LAYERS:
        raise ValueError(f"unknown layer family: {family}")
    register_gemma4_fa4_h100()
    config = _locked_text_config()
    layer_idx = FAMILY_LAYERS[family]
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    torch.manual_seed(seed)
    layer = Gemma4TextAttention(config, layer_idx=layer_idx).to(device="cuda", dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(config, device="cuda").to(device="cuda")
    layer.eval()
    rotary.eval()

    if family == "local":
        mask_builder = create_sliding_window_causal_mask
        custom_op = h100_torch_ops.h100_local_fwd
        custom_op_fragment = "h100_local_fwd"
        layer_custom_op = h100_torch_ops.h100_local_layer_fwd
        layer_opcheck_target = h100_torch_ops.h100_local_layer_op
        layer_custom_op_fragment = "h100_local_layer_fwd"
    else:
        mask_builder = create_causal_mask
        custom_op = h100_torch_ops.h100_global_fwd
        custom_op_fragment = "h100_global_fwd"
        layer_custom_op = h100_torch_ops.h100_global_layer_fwd
        layer_opcheck_target = h100_torch_ops.h100_global_layer_op
        layer_custom_op_fragment = "h100_global_layer_fwd"
    return _FamilyRuntime(
        name=family,
        layer_idx=layer_idx,
        spec=spec,
        config=config,
        layer=layer,
        rotary=rotary,
        mask_builder=mask_builder,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        custom_op=custom_op,
        custom_op_fragment=custom_op_fragment,
        layer_custom_op=layer_custom_op,
        layer_opcheck_target=layer_opcheck_target,
        layer_custom_op_fragment=layer_custom_op_fragment,
    )


def _make_inputs(
    runtime: _FamilyRuntime,
    seqlen: int,
    *,
    seed: int,
    position_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        (1, seqlen, runtime.config.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(
        position_start,
        position_start + seqlen,
        dtype=torch.int64,
        device="cuda",
    ).unsqueeze(0)
    cos, sin = runtime.rotary(hidden, positions, layer_type=runtime.spec.kind)
    return hidden, cos, sin, positions


def _mark_sequence_dynamic(inputs: Sequence[torch.Tensor]) -> None:
    mark_dynamic = getattr(getattr(torch, "_dynamo", None), "mark_dynamic", None)
    if not callable(mark_dynamic):
        raise RuntimeError(f"{EXPERIMENT} requires torch._dynamo.mark_dynamic")
    for tensor in inputs:
        mark_dynamic(tensor, 1, min=1, max=1024)


def _layer_callable(runtime: _FamilyRuntime):
    def call(
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = runtime.mask_builder(
            runtime.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
            layer_idx=runtime.layer_idx,
        )
        output, weights = runtime.layer(
            hidden,
            (cos, sin),
            attention_mask,
            {},
            position_ids=position_ids,
            allow_flex_fallback=False,
        )
        if weights is not None:
            raise AssertionError("Gemma4TextAttention unexpectedly returned attention weights")
        return output

    return call


def _cache_marker_callable(runtime: _FamilyRuntime):
    def call(
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = runtime.mask_builder(
            runtime.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
            layer_idx=runtime.layer_idx,
        )
        output, _weights = runtime.layer(
            hidden,
            (cos, sin),
            attention_mask,
            {},
            position_ids=position_ids,
            cache_position=position_ids,
            allow_flex_fallback=False,
        )
        return output

    return call


def _cache_object_callable(
    runtime: _FamilyRuntime,
    cache: Any,
    entry_counters: dict[str, int] | None = None,
):
    """Build both the registered mask and actual layer call around one cache object."""

    def call(
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = runtime.mask_builder(
            runtime.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=position_ids,
            layer_idx=runtime.layer_idx,
        )
        if entry_counters is not None:
            entry_counters["layer_entry"] += 1
        output, _weights = runtime.layer(
            hidden,
            (cos, sin),
            attention_mask,
            {},
            past_key_values=cache,
            position_ids=position_ids,
            allow_flex_fallback=False,
        )
        return output

    return call


def _prepare_qkv_math(
    runtime: _FamilyRuntime,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = runtime.layer
    hidden_shape = (*hidden.shape[:-1], -1, layer.head_dim)
    q = layer.q_proj(hidden).view(hidden_shape)
    q = layer.q_norm(q)
    q = runtime.apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    projected_k = layer.k_proj(hidden).view(hidden_shape)
    projected_v = (
        layer.v_proj(hidden).view(hidden_shape) if layer.v_proj is not None else projected_k
    )
    k = layer.k_norm(projected_k)
    k = runtime.apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    v = layer.v_norm(projected_v).transpose(1, 2)
    return q, k, v


def _prepare_qkv(
    runtime: _FamilyRuntime,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = _prepare_qkv_math(runtime, hidden, cos, sin)
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise AssertionError("prepared K and V must use distinct storage")
    return q, k, v


def _whole_layer_explicit_inputs(
    runtime: _FamilyRuntime,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    detach_weights: bool = True,
) -> tuple[torch.Tensor, ...]:
    hidden, cos, sin, positions = inputs
    layer = runtime.layer
    packed = torch.zeros_like(positions)

    def select(weight: torch.Tensor) -> torch.Tensor:
        return weight.detach() if detach_weights else weight

    weights = [select(layer.q_proj.weight), select(layer.k_proj.weight)]
    if layer.v_proj is not None:
        weights.append(select(layer.v_proj.weight))
    weights.extend(
        [
            select(layer.o_proj.weight),
            select(layer.q_norm.weight),
            select(layer.k_norm.weight),
        ]
    )
    return hidden, cos, sin, positions, packed, *weights


def _finite_error(candidate: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    if candidate.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: candidate={tuple(candidate.shape)} expected={tuple(expected.shape)}"
        )
    candidate_finite = torch.isfinite(candidate)
    expected_finite = torch.isfinite(expected)
    if not torch.equal(candidate_finite, expected_finite):
        raise AssertionError("candidate/reference finite masks differ")
    if not torch.equal(candidate[~candidate_finite], expected[~expected_finite]):
        raise AssertionError("candidate/reference non-finite sentinels differ")
    if not bool(candidate_finite.any().item()):
        return 0.0, 0.0
    error = (candidate[candidate_finite].float() - expected[expected_finite].float()).abs()
    return float(error.max().item()), float(error.mean().item())


def _assert_close(
    candidate: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    maximum, mean = _finite_error(candidate, expected)
    torch.testing.assert_close(candidate, expected, atol=atol, rtol=rtol)
    return {"max_abs": maximum, "mean_abs": mean}


def _tensor_comparison(candidate: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Record raw stage drift without introducing a new acceptance tolerance."""

    maximum, mean = _finite_error(candidate, expected)
    return {
        "bitwise": torch.equal(candidate, expected),
        "max_abs": maximum,
        "mean_abs": mean,
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
    }


def _full_layer_comparison(
    compiled: torch.Tensor,
    eager: torch.Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    """Apply EXP-0022's predeclared bitwise whole-layer gate."""

    maximum, mean = _finite_error(compiled, eager)
    bitwise = torch.equal(compiled, eager)
    if not bitwise:
        raise AssertionError(
            f"{label} violates EXP-0022 bitwise whole-layer equality "
            f"(max_abs={maximum:.9g}, mean_abs={mean:.9g}); tolerances may not be substituted"
        )
    return {
        "exact": True,
        "bitwise": True,
        "max_abs": maximum,
        "mean_abs": mean,
    }


def _direct_reference_case(
    runtime: _FamilyRuntime,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    eager_layer_output: torch.Tensor,
) -> dict[str, Any]:
    hidden, cos, sin, positions = inputs
    q, k, v = _prepare_qkv(runtime, hidden, cos, sin)
    packed = torch.zeros_like(positions)
    output, lse = runtime.custom_op(q, k, v, positions, packed)
    whole_output, whole_lse = runtime.layer_custom_op(
        *_whole_layer_explicit_inputs(runtime, inputs)
    )
    live_weight_output, live_weight_lse = runtime.layer_custom_op(
        *_whole_layer_explicit_inputs(runtime, inputs, detach_weights=False)
    )
    reference_output, reference_lse = reference_attention(
        q,
        k,
        v,
        softmax_scale=1.0,
        sliding_window=runtime.spec.sliding_window,
        allow_vision_bidirectional=False,
        upcast=torch.float32,
        return_lse=True,
    )
    reference_output = reference_output.transpose(1, 2)
    if output.shape != (1, hidden.shape[1], runtime.spec.num_q_heads, runtime.spec.head_dim_v):
        raise AssertionError("custom op returned an invalid BSHD output shape")
    if output.dtype != torch.bfloat16:
        raise AssertionError("custom op output must be BF16")
    if lse.shape != (1, runtime.spec.num_q_heads, hidden.shape[1]) or lse.dtype != torch.float32:
        raise AssertionError("custom op returned an invalid FP32 LSE contract")

    pointers = {tensor.untyped_storage().data_ptr() for tensor in (q, k, v, output, lse)}
    if len(pointers) != 5:
        raise AssertionError("custom-op inputs and outputs must all use distinct storage")
    projected = runtime.layer.o_proj(output.reshape(1, hidden.shape[1], -1).contiguous())
    if not torch.equal(projected, eager_layer_output):
        raise AssertionError("direct prepared custom-op path differs from eager full-layer output")
    if not torch.equal(whole_output, eager_layer_output):
        maximum, mean = _finite_error(whole_output, eager_layer_output)
        raise AssertionError(
            "direct whole-layer custom op differs from the pinned eager layer "
            f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
        )
    if not torch.equal(whole_lse, lse):
        raise AssertionError("direct whole-layer FP32 LSE differs from the prepared custom op")
    if not torch.equal(live_weight_output, whole_output) or not torch.equal(
        live_weight_lse, whole_lse
    ):
        raise AssertionError(
            "dormant requires-grad module weights changed the inference-only whole-layer result"
        )

    if runtime.name == "local":
        output_atol, output_rtol, lse_atol = (
            LOCAL_OUTPUT_ATOL,
            LOCAL_OUTPUT_RTOL,
            LOCAL_LSE_ATOL,
        )
    else:
        output_atol, output_rtol, lse_atol = (
            GLOBAL_OUTPUT_ATOL,
            GLOBAL_OUTPUT_RTOL,
            GLOBAL_LSE_ATOL,
        )
    return {
        "seqlen": int(hidden.shape[1]),
        "output": _assert_close(output, reference_output, atol=output_atol, rtol=output_rtol),
        "lse": _assert_close(lse, reference_lse, atol=lse_atol, rtol=0.0),
        "output_dtype": str(output.dtype),
        "lse_dtype": str(lse.dtype),
        "distinct_storage": True,
        "projection_transport_bitwise": True,
        "whole_layer_output_bitwise": True,
        "whole_layer_lse_bitwise": True,
        "dormant_weight_output_bitwise": True,
        "dormant_weight_lse_bitwise": True,
    }


def _fake_shape_proof(runtime: _FamilyRuntime, *, seqlen: int = 33) -> dict[str, Any]:
    from torch._subclasses.fake_tensor import FakeTensorMode

    def forbidden(*_args, **_kwargs):
        raise AssertionError("FakeTensor custom-op execution entered a real FA4 body")

    original_local = h100_torch_ops.fa4_local_text_forward
    original_global = h100_torch_ops.fa4_global_text_forward
    h100_torch_ops.fa4_local_text_forward = forbidden
    h100_torch_ops.fa4_global_text_forward = forbidden
    try:
        with FakeTensorMode(), torch.inference_mode():
            hidden = torch.empty(
                (1, seqlen, GEMMA4_31B.hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            cos = torch.empty(
                (1, seqlen, runtime.spec.head_dim_qk),
                dtype=torch.bfloat16,
                device="cuda",
            )
            sin = torch.empty_like(cos)
            positions = torch.arange(seqlen, dtype=torch.int64, device="cuda").unsqueeze(0)
            packed = torch.zeros((1, seqlen), dtype=torch.int64, device="cuda")
            head_dim = runtime.spec.head_dim_qk
            weight_shapes = [
                (runtime.spec.num_q_heads * head_dim, GEMMA4_31B.hidden_size),
                (runtime.spec.num_kv_heads * head_dim, GEMMA4_31B.hidden_size),
            ]
            if runtime.name == "local":
                weight_shapes.append((runtime.spec.num_kv_heads * head_dim, GEMMA4_31B.hidden_size))
            weight_shapes.extend(
                [
                    (GEMMA4_31B.hidden_size, runtime.spec.num_q_heads * head_dim),
                    (head_dim,),
                    (head_dim,),
                ]
            )
            weights = tuple(
                torch.empty(shape, dtype=torch.bfloat16, device="cuda") for shape in weight_shapes
            )
            explicit_inputs = (hidden, cos, sin, positions, packed, *weights)
            output, lse = runtime.layer_custom_op(*explicit_inputs)
            if output.shape != (1, seqlen, GEMMA4_31B.hidden_size):
                raise AssertionError("fake whole-layer custom op returned an invalid BSH shape")
            if lse.shape != (1, runtime.spec.num_q_heads, seqlen):
                raise AssertionError("fake whole-layer custom op returned an invalid LSE shape")
            if output.dtype != torch.bfloat16 or lse.dtype != torch.float32:
                raise AssertionError("fake whole-layer custom op returned invalid output dtypes")
            if any(output is tensor or lse is tensor for tensor in explicit_inputs):
                raise AssertionError("fake whole-layer outputs must use fresh symbolic tensors")
    finally:
        h100_torch_ops.fa4_local_text_forward = original_local
        h100_torch_ops.fa4_global_text_forward = original_global
    return {
        "status": "passed",
        "seqlen": seqlen,
        "output_shape": [1, seqlen, GEMMA4_31B.hidden_size],
        "lse_shape": [1, runtime.spec.num_q_heads, seqlen],
        "real_body_entered": False,
    }


def _opcheck(runtime: _FamilyRuntime, *, seed: int) -> dict[str, Any]:
    inputs = _make_inputs(runtime, 33, seed=seed)
    explicit_inputs = _whole_layer_explicit_inputs(runtime, inputs)
    layer_result = torch.library.opcheck(
        runtime.layer_opcheck_target,
        explicit_inputs,
        raise_exception=False,
    )
    serialized = {name: str(value) for name, value in layer_result.items()}
    failures = {name: value for name, value in serialized.items() if value != "SUCCESS"}
    if failures:
        raise AssertionError(f"torch.library.opcheck failures: {failures}")
    return {"status": "passed", "tests": serialized}


def _graph_break_count() -> int:
    counters = getattr(getattr(torch, "_dynamo", None), "utils", None)
    values = getattr(counters, "counters", {}).get("graph_break", {})
    return sum(int(value) for value in values.values())


def _contains_custom_op(graphs: Sequence[dict[str, Any]], fragment: str) -> bool:
    return any(
        fragment in node["target"] and "gemma4_fa4" in node["target"]
        for graph in graphs
        for node in graph["nodes"]
        if node["op"] == "call_function"
    )


def _scalar_graph_inventory(graphs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return graph artifacts forbidden by EXP-0022's static-scalar gate."""

    forbidden_targets = {"item"}
    forbidden_fragments = ("scalar_tensor", "_local_scalar_dense", "stack of type object")
    inventory = []
    for graph_index, graph in enumerate(graphs):
        for node_index, node in enumerate(graph["nodes"]):
            target = node["target"]
            symbolic_float = node["op"] == "placeholder" and node.get("example_value_type") in {
                "SymFloat",
                "float",
            }
            scalar_node = target in forbidden_targets or any(
                fragment in target for fragment in forbidden_fragments
            )
            if symbolic_float or scalar_node:
                inventory.append(
                    {
                        "graph_index": graph_index,
                        "node_index": node_index,
                        "op": node["op"],
                        "target": target,
                        "example_value_type": node.get("example_value_type"),
                    }
                )
    return inventory


def _is_expected_cache_marker_rejection(message: str) -> bool:
    lowered = message.lower()
    return "faketensor/torch.compile does not accept a cache" in lowered or (
        "observed exception" in lowered and "UnsupportedH100Path" in message
    )


def _classify_cache_object_rejection(error_type: str, message: str) -> str | None:
    direct = (
        "EXP-0018 fullgraph routing does not accept a cache; "
        "rejection occurs at mask construction before Cache.update"
    )
    if error_type == "UnsupportedH100Path" and direct in message:
        return "exp0018_mask_cache_rejection"
    lowered = message.lower()
    if (
        error_type == "Unsupported"
        and "observed exception" in lowered
        and "UserDefinedExceptionObjectVariable(UnsupportedH100Path)" in message
    ):
        return "exp0018_mask_cache_rejection_dynamo_wrapper"
    return None


def _expect_cache_rejection(
    runtime: _FamilyRuntime,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, str]:
    compiled = torch.compile(
        _cache_marker_callable(runtime),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    try:
        compiled(*inputs)
    except Exception as exc:
        message = str(exc)
        if not _is_expected_cache_marker_rejection(message):
            raise AssertionError(f"cache path failed for an unrelated reason: {message}") from exc
        return {"status": "rejected", "error_type": type(exc).__name__, "error": message}
    raise AssertionError("compiled cache marker was admitted unexpectedly")


def _run_public_dynamic_diagnostic(
    runtime: _FamilyRuntime,
    backend: str,
    lengths: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Record stock ``dynamic=True`` behavior without the internal shape policy."""

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    capture = _CapturingBackend(backend)
    eager_call = _layer_callable(runtime)
    compiled_call = torch.compile(
        eager_call,
        backend=capture,
        fullgraph=True,
        dynamic=True,
    )
    cases = []
    with torch.inference_mode():
        for index, seqlen in enumerate(lengths):
            inputs = _make_inputs(runtime, seqlen, seed=seed + index)
            eager_output = eager_call(*inputs)
            compiled_output = compiled_call(*inputs)
            comparison = _full_layer_comparison(
                compiled_output,
                eager_output,
                label=f"{runtime.name}/{backend}/public/S{seqlen}",
            )
            cases.append({"seqlen": seqlen, **comparison})

    graph_count = len(capture.graphs)
    graph_break_count = _graph_break_count()
    expected_graph_count = 2 if 1 in lengths and any(length > 1 for length in lengths) else 1
    custom_op_node = _contains_custom_op(capture.graphs, runtime.layer_custom_op_fragment)
    weight_snapshot_node = _contains_custom_op(capture.graphs, "weight_snapshot")
    scalar_graph_inventory = _scalar_graph_inventory(capture.graphs)
    if graph_count != expected_graph_count:
        guard_failures = {
            getattr(code, "co_name", repr(code)): list(reasons)
            for code, reasons in torch._dynamo.guard_failures.items()
        }
        raise AssertionError(
            f"{runtime.name}/{backend} public dynamic diagnostic produced {graph_count} graphs; "
            f"expected the pinned {expected_graph_count}-class S1 specialization outcome; "
            f"guard_failures={guard_failures}; captured={capture.graphs}"
        )
    if graph_break_count != 0:
        raise AssertionError(
            f"{runtime.name}/{backend} public dynamic diagnostic produced "
            f"{graph_break_count} graph breaks"
        )
    if not custom_op_node:
        raise AssertionError(
            f"{runtime.name}/{backend} public dynamic graphs lack the project custom-op node"
        )
    if weight_snapshot_node:
        raise AssertionError(f"{runtime.name}/{backend} public graph retained a weight snapshot")
    if scalar_graph_inventory:
        raise AssertionError(
            f"{runtime.name}/{backend} public graph retained forbidden scalar artifacts: "
            f"{scalar_graph_inventory}"
        )
    graph_classes = []
    singleton_lengths = [length for length in lengths if length == 1]
    nonsingleton_lengths = [length for length in lengths if length > 1]
    if singleton_lengths:
        graph_classes.append({"class": "S1", "lengths": singleton_lengths})
    if nonsingleton_lengths:
        graph_classes.append({"class": "S>1", "lengths": nonsingleton_lengths})
    return {
        "status": "passed",
        "fullgraph": True,
        "backend_delegate": f"stock_{backend}",
        "shape_policy": {
            "torch_compile_dynamic": True,
            "mark_dynamic_sequence_dim": False,
            "backed_size_oblivious": False,
            "is_pytorch_default": True,
        },
        "graph_count": graph_count,
        "expected_graph_count": expected_graph_count,
        "graph_classes": graph_classes,
        "bounded_exactly_s1_or_gt1": True,
        "one_graph_requirement_met": graph_count == 1,
        "graph_break_count": graph_break_count,
        "custom_op_node": custom_op_node,
        "weight_snapshot_node": weight_snapshot_node,
        "scalar_graph_inventory": scalar_graph_inventory,
        "graph_nodes": capture.graphs,
        "cases": cases,
    }


def _run_scoped_backend_matrix(
    runtime: _FamilyRuntime,
    backend: str,
    lengths: Sequence[int],
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the one-graph matrix under the explicit S1-capable shape policy."""

    try:
        from torch.fx.experimental import _config as fx_config
    except Exception as exc:
        raise RuntimeError(f"{EXPERIMENT} requires torch.fx.experimental._config") from exc
    if not hasattr(fx_config, "backed_size_oblivious"):
        raise RuntimeError(f"{EXPERIMENT} requires the backed_size_oblivious shape policy")
    with fx_config.patch(backed_size_oblivious=True):
        return _run_scoped_backend_matrix_inner(
            runtime,
            backend,
            lengths,
            seed=seed,
        )


def _run_scoped_backend_matrix_inner(
    runtime: _FamilyRuntime,
    backend: str,
    lengths: Sequence[int],
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    capture = _CapturingBackend(backend)
    eager_call = _layer_callable(runtime)
    compiled_call = torch.compile(
        eager_call,
        backend=capture,
        fullgraph=True,
        dynamic=True,
    )
    cases: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    stream_inputs = None
    stream_expected = None
    with torch.inference_mode():
        for index, seqlen in enumerate(lengths):
            inputs = _make_inputs(runtime, seqlen, seed=seed + index)
            _mark_sequence_dynamic(inputs)
            eager_output = eager_call(*inputs)
            compiled_output = compiled_call(*inputs)
            comparison = _full_layer_comparison(
                compiled_output,
                eager_output,
                label=f"{runtime.name}/{backend}/scoped/S{seqlen}",
            )
            references.append(_direct_reference_case(runtime, inputs, eager_output))
            cases.append(
                {
                    "seqlen": seqlen,
                    **comparison,
                    "output_shape": list(compiled_output.shape),
                    "output_dtype": str(compiled_output.dtype),
                }
            )
            if seqlen == 33 or (stream_inputs is None and index == len(lengths) - 1):
                stream_inputs = inputs
                stream_expected = compiled_output.clone()

        assert stream_inputs is not None and stream_expected is not None
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            stream_output = compiled_call(*stream_inputs)
        stream.synchronize()
        stream_comparison = _full_layer_comparison(
            stream_output,
            stream_expected,
            label=f"{runtime.name}/{backend}/nondefault-stream",
        )

        reset_positions = torch.ones_like(stream_inputs[3])
        reset_inputs = (*stream_inputs[:3], reset_positions)
        try:
            compiled_call(*reset_inputs)
        except Exception as exc:
            reset_rejection = {
                "status": "rejected",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:
            raise AssertionError("reset position_ids were admitted unexpectedly")

    graph_count = len(capture.graphs)
    graph_break_count = _graph_break_count()
    custom_op_node = _contains_custom_op(capture.graphs, runtime.layer_custom_op_fragment)
    weight_snapshot_node = _contains_custom_op(capture.graphs, "weight_snapshot")
    scalar_graph_inventory = _scalar_graph_inventory(capture.graphs)
    if graph_count != 1:
        raise AssertionError(
            f"{runtime.name}/{backend} produced {graph_count} graphs for lengths {tuple(lengths)}; "
            "the public dynamic=True one-graph requirement is unmet"
        )
    if graph_break_count != 0:
        raise AssertionError(f"{runtime.name}/{backend} produced {graph_break_count} graph breaks")
    if not custom_op_node:
        raise AssertionError(f"{runtime.name}/{backend} graph lacks the project custom-op node")
    if weight_snapshot_node:
        raise AssertionError(f"{runtime.name}/{backend} graph retained a weight snapshot")
    if scalar_graph_inventory:
        raise AssertionError(
            f"{runtime.name}/{backend} graph retained forbidden scalar artifacts: "
            f"{scalar_graph_inventory}"
        )

    cache_rejection = _expect_cache_rejection(runtime, stream_inputs)
    return (
        {
            "status": "passed",
            "fullgraph": True,
            "backend_delegate": f"stock_{backend}",
            "shape_policy": {
                "torch_compile_dynamic": True,
                "mark_dynamic_sequence_dim": True,
                "dynamic_sequence_min": 1,
                "dynamic_sequence_max": 1024,
                "backed_size_oblivious": True,
                "is_pytorch_default": False,
            },
            "graph_count": graph_count,
            "graph_break_count": graph_break_count,
            "custom_op_node": custom_op_node,
            "weight_snapshot_node": weight_snapshot_node,
            "scalar_graph_inventory": scalar_graph_inventory,
            "graph_nodes": capture.graphs,
            "cases": cases,
            "nondefault_stream": stream_comparison,
            "reset_positions": reset_rejection,
            "cache": cache_rejection,
        },
        references,
    )


def _run_family(
    family: str,
    backends: Sequence[str],
    lengths: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    runtime = _make_family_runtime(family, seed=seed)
    application_before = _forward_application_snapshot()
    fake = _fake_shape_proof(runtime)
    with torch.no_grad():
        opcheck = _opcheck(runtime, seed=seed + 10)
    application_warmed = _forward_application_snapshot()
    added = sorted(set(application_warmed) - set(application_before))
    if len(added) != 1 or not set(application_before).issubset(application_warmed):
        raise AssertionError(
            f"{family} must add exactly one bounded FA4 forward application-key class"
        )
    backend_results = {}
    reference_by_length: dict[int, dict[str, Any]] = {}
    for index, backend in enumerate(backends):
        public_dynamic = _run_public_dynamic_diagnostic(
            runtime,
            backend,
            lengths,
            seed=seed + 1000 * (index + 1),
        )
        scoped_result, references = _run_scoped_backend_matrix(
            runtime,
            backend,
            lengths,
            seed=seed + 10_000 + 1000 * (index + 1),
        )
        backend_results[backend] = {
            "public_dynamic_default": public_dynamic,
            "scoped_backed_size_oblivious": scoped_result,
        }
        for reference in references:
            reference_by_length.setdefault(reference["seqlen"], reference)
    application_after = _forward_application_snapshot()
    if application_after != application_warmed:
        raise AssertionError(
            f"{family} compiler matrices changed the warmed FA4 forward application keys"
        )
    return {
        "status": "passed",
        "layer_idx": runtime.layer_idx,
        "layer_type": runtime.spec.kind,
        "hidden_size": runtime.config.hidden_size,
        "geometry": {
            "q_heads": runtime.spec.num_q_heads,
            "kv_heads": runtime.spec.num_kv_heads,
            "head_dim": runtime.spec.head_dim_qk,
            "scale": runtime.spec.softmax_scale,
        },
        "fake_tensor": fake,
        "opcheck": opcheck,
        "fa4_forward_application_keys": {
            "before": list(application_before),
            "warmed": list(application_warmed),
            "after": list(application_after),
            "added_family_class": added,
            "new_class_count": 1,
            "reused_across_compiler_matrix": True,
        },
        "direct_reference": [reference_by_length[length] for length in lengths],
        "backends": backend_results,
    }


def _compile_localization_stage(
    function: Callable,
    inputs: tuple[torch.Tensor, ...],
    *,
    label: str,
    custom_op_fragment: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compile one diagnostic stage with stock Inductor and capture its graph."""

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    capture = _CapturingBackend("inductor")
    compiled = torch.compile(
        function,
        backend=capture,
        fullgraph=True,
        dynamic=True,
    )
    with torch.inference_mode():
        output = compiled(*inputs)
    torch.cuda.synchronize()
    graph_count = len(capture.graphs)
    graph_break_count = _graph_break_count()
    if not 1 <= graph_count <= 2 or graph_break_count != 0:
        raise AssertionError(
            f"outer-drift localization stage {label!r} requires 1..2 full Inductor "
            f"captures and zero breaks; observed graphs={graph_count}, "
            f"breaks={graph_break_count}"
        )
    result = {
        "stage": label,
        "graph_count": graph_count,
        "max_expected_graph_count": 2,
        "bounded_graph_count": True,
        "graph_break_count": graph_break_count,
        "graph_nodes": capture.graphs,
    }
    if custom_op_fragment is not None:
        result["custom_op_node"] = _contains_custom_op(
            capture.graphs,
            custom_op_fragment,
        )
        if result["custom_op_node"] is not True:
            raise AssertionError(
                f"localization graph lacks project custom op {custom_op_fragment!r}"
            )
    return output, result


def _qkv_localization_callable(runtime: _FamilyRuntime):
    def call(
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _prepare_qkv_math(runtime, hidden, cos, sin)

    return call


def _prepared_localization_callable(runtime: _FamilyRuntime):
    def call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        packed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return runtime.custom_op(q, k, v, positions, packed)

    return call


def _projection_localization_callable(runtime: _FamilyRuntime):
    def call(attention_output: torch.Tensor) -> torch.Tensor:
        flattened = attention_output.reshape(
            attention_output.shape[0],
            attention_output.shape[1],
            -1,
        ).contiguous()
        return runtime.layer.o_proj(flattened)

    return call


def _reference_policy(
    runtime: _FamilyRuntime,
    output: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, Any]:
    reference_output, reference_lse = reference_attention(
        q,
        k,
        v,
        softmax_scale=1.0,
        sliding_window=runtime.spec.sliding_window,
        allow_vision_bidirectional=False,
        upcast=torch.float32,
        return_lse=True,
    )
    reference_output = reference_output.transpose(1, 2)
    if runtime.name == "local":
        output_atol, output_rtol, lse_atol = (
            LOCAL_OUTPUT_ATOL,
            LOCAL_OUTPUT_RTOL,
            LOCAL_LSE_ATOL,
        )
    else:
        output_atol, output_rtol, lse_atol = (
            GLOBAL_OUTPUT_ATOL,
            GLOBAL_OUTPUT_RTOL,
            GLOBAL_LSE_ATOL,
        )
    output_error = _assert_close(
        output,
        reference_output,
        atol=output_atol,
        rtol=output_rtol,
    )
    lse_error = _assert_close(lse, reference_lse, atol=lse_atol, rtol=0.0)
    return {
        "output": {
            "passed": True,
            **output_error,
            "atol": output_atol,
            "rtol": output_rtol,
        },
        "lse": {
            "passed": True,
            **lse_error,
            "atol": lse_atol,
            "rtol": 0.0,
        },
    }


def _within_exp0018_frozen_tolerance(
    candidate: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    try:
        torch.testing.assert_close(
            candidate,
            expected,
            atol=FULL_LAYER_OUTPUT_ATOL,
            rtol=FULL_LAYER_OUTPUT_RTOL,
        )
    except AssertionError:
        return False
    return True


def _run_outer_drift_localization(args: argparse.Namespace) -> dict[str, Any]:
    """Localize EXP-0018's S1023 failure before widening the opaque boundary."""

    if (
        args.family != "all"
        or args.backend != "all"
        or tuple(args.lengths) != DEFAULT_LENGTHS
        or args.seed != 17017
    ):
        raise ValueError(
            "--localize-outer-drift has a fixed local/Inductor/S1023/seed-17017 "
            "coordinate; do not combine it with family/backend/length/seed overrides"
        )
    _require_h100()
    cache_dirs = _prepare_fresh_cache_dirs(("inductor",))
    runtime_seed = args.seed
    input_seed = args.seed + 2003
    runtime = _make_family_runtime("local", seed=runtime_seed)
    inputs = _make_inputs(runtime, 1023, seed=input_seed)
    hidden, cos, sin, positions = inputs
    packed = torch.zeros_like(positions)
    application_before = _forward_application_snapshot()

    eager_layer = _layer_callable(runtime)
    with torch.inference_mode():
        eager_whole = eager_layer(*inputs)
        eager_qkv = _prepare_qkv(runtime, hidden, cos, sin)

    compiled_qkv, qkv_graph = _compile_localization_stage(
        _qkv_localization_callable(runtime),
        (hidden, cos, sin),
        label="qkv",
    )
    qkv_components = {
        name: _tensor_comparison(candidate, expected)
        for name, candidate, expected in zip(
            ("q", "k", "v"),
            compiled_qkv,
            eager_qkv,
            strict=True,
        )
    }
    qkv_pointers = {tensor.untyped_storage().data_ptr() for tensor in compiled_qkv}
    qkv_graph.update(
        {
            "components": qkv_components,
            "distinct_storage": len(qkv_pointers) == 3,
        }
    )
    if qkv_graph["distinct_storage"] is not True:
        raise AssertionError("compiled localization Q/K/V must use distinct storage")

    with torch.inference_mode():
        eager_attention, eager_lse = runtime.custom_op(
            *eager_qkv,
            positions,
            packed,
        )
    compiled_prepared, prepared_graph = _compile_localization_stage(
        _prepared_localization_callable(runtime),
        (*eager_qkv, positions, packed),
        label="prepared_opaque",
        custom_op_fragment=runtime.custom_op_fragment,
    )
    compiled_attention, compiled_lse = compiled_prepared
    prepared_graph.update(
        {
            "output": _tensor_comparison(compiled_attention, eager_attention),
            "lse": _tensor_comparison(compiled_lse, eager_lse),
            "reference": _reference_policy(
                runtime,
                eager_attention,
                eager_lse,
                *eager_qkv,
            ),
        }
    )

    projection_callable = _projection_localization_callable(runtime)
    with torch.inference_mode():
        eager_projection = projection_callable(eager_attention)
    compiled_projection, projection_graph = _compile_localization_stage(
        projection_callable,
        (eager_attention,),
        label="output_projection",
    )
    projection_graph["output"] = _tensor_comparison(
        compiled_projection,
        eager_projection,
    )
    if not torch.equal(eager_projection, eager_whole):
        raise AssertionError(
            "the direct eager prepared-attention path must be bitwise equal to the pinned layer"
        )

    with torch.inference_mode():
        hybrid_attention, _hybrid_lse = runtime.custom_op(
            *compiled_qkv,
            positions,
            packed,
        )
        hybrid_output = projection_callable(hybrid_attention)
    hybrid = {"output": _tensor_comparison(hybrid_output, eager_whole)}

    compiled_whole, whole_graph = _compile_localization_stage(
        eager_layer,
        inputs,
        label="whole_layer",
        custom_op_fragment=runtime.custom_op_fragment,
    )
    whole_comparison = _tensor_comparison(compiled_whole, eager_whole)
    whole_comparison["within_exp0018_frozen_tolerance"] = _within_exp0018_frozen_tolerance(
        compiled_whole, eager_whole
    )
    whole_graph["output"] = whole_comparison

    application_after = _forward_application_snapshot()
    added_application_keys = sorted(set(application_after) - set(application_before))
    if len(added_application_keys) != 1:
        raise AssertionError(
            "outer-drift localization must add exactly one local FA4 application-key class"
        )

    prepared_bitwise = bool(
        prepared_graph["output"]["bitwise"] and prepared_graph["lse"]["bitwise"]
    )
    outer_comparisons = (
        *qkv_components.values(),
        projection_graph["output"],
        hybrid["output"],
    )
    outer_drift = any(item["bitwise"] is False for item in outer_comparisons)
    exp0018_failure_reproduced = bool(
        whole_comparison["bitwise"] is False
        and whole_comparison["within_exp0018_frozen_tolerance"] is False
    )
    hypothesis_supported = bool(prepared_bitwise and outer_drift and exp0018_failure_reproduced)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "request": {
            "mode": "localize_outer_drift",
            "family": "local",
            "backend": "inductor",
            "seqlen": 1023,
            "runtime_seed": runtime_seed,
            "input_seed": input_seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "caches": {name: _cache_inventory(path) for name, path in cache_dirs.items()},
        },
        "fa4_forward_application_keys": {
            "before": list(application_before),
            "after": list(application_after),
            "added": added_application_keys,
            "new_class_count": len(added_application_keys),
        },
        "localization": {
            "qkv": qkv_graph,
            "prepared_opaque": prepared_graph,
            "output_projection": projection_graph,
            "hybrid_compiled_qkv": hybrid,
            "whole_layer": whole_graph,
            "prepared_opaque_bitwise": prepared_bitwise,
            "outer_drift_observed": outer_drift,
            "exp0018_failure_reproduced": exp0018_failure_reproduced,
            "hypothesis_supported": hypothesis_supported,
        },
    }


def _selected(value: str, all_values: Sequence[str]) -> tuple[str, ...]:
    return tuple(all_values) if value == "all" else (value,)


def _cache_seq_length(cache: Any, layer_idx: int) -> int:
    value = cache.get_seq_length(layer_idx)
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_state(tensor: torch.Tensor) -> dict[str, Any]:
    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "storage_pointer": storage.data_ptr(),
        "storage_nbytes": storage.nbytes(),
        "storage_offset": tensor.storage_offset(),
        "content_sha256": _tensor_content_digest(tensor),
    }


def _cache_state_snapshot(cache: Any, layer_idx: int) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    layers = cache.layers
    layer = layers[layer_idx]
    tensors = {
        name: _tensor_state(value)
        for name, value in sorted(vars(layer).items())
        if isinstance(value, torch.Tensor)
    }
    counters: dict[str, int | float | bool] = {}
    for name in ("cumulative_length", "cumulative_length_int", "_seen_tokens"):
        if not hasattr(layer, name):
            continue
        value = getattr(layer, name)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            counters[name] = value.item()
        elif isinstance(value, (bool, int, float)):
            counters[name] = value

    unique_storages = {
        (item["device"], item["storage_pointer"], item["storage_nbytes"])
        for item in tensors.values()
    }
    content_payload = {
        "counters": counters,
        "tensor_digests": {name: item["content_sha256"] for name, item in tensors.items()},
    }
    return {
        "cache_type": type(cache).__name__,
        "cache_identity": id(cache),
        "layer_idx": layer_idx,
        "layer_count": len(layers),
        "logical_lengths": [_cache_seq_length(cache, index) for index in range(len(layers))],
        "initialized_layers": [
            index for index, candidate in enumerate(layers) if candidate.is_initialized
        ],
        "target_layer": {
            "type": type(layer).__name__,
            "is_initialized": bool(layer.is_initialized),
            "logical_length": _cache_seq_length(cache, layer_idx),
            "counters": counters,
            "tensors": tensors,
            "storage_identity": {name: item["storage_pointer"] for name, item in tensors.items()},
            "content_sha256": hashlib.sha256(
                json.dumps(content_payload, sort_keys=True).encode()
            ).hexdigest(),
            "allocation": {
                "allocated_tensor_names": [
                    name for name, item in tensors.items() if item["numel"] > 0
                ],
                "unique_storage_count": len(unique_storages),
                "storage_nbytes": sum(item[2] for item in unique_storages),
            },
        },
    }


def _make_real_cache(cache_type: str, runtime: _FamilyRuntime):
    try:
        from transformers import DynamicCache, StaticCache
    except Exception as exc:
        raise RuntimeError("the pinned Transformers cache classes are required") from exc
    if cache_type == "DynamicCache":
        return DynamicCache(config=runtime.config)
    if cache_type == "StaticCache":
        return StaticCache(config=runtime.config, max_cache_len=8)
    raise ValueError(f"unknown cache type: {cache_type}")


def _populate_cache(
    cache: Any,
    runtime: _FamilyRuntime,
    *,
    seed: int,
) -> None:
    inputs = _make_inputs(runtime, 1, seed=seed)
    with torch.inference_mode():
        q, k, v = _prepare_qkv(runtime, *inputs[:3])
        del q
        cache.update(k, v, runtime.layer_idx)
    if _cache_seq_length(cache, runtime.layer_idx) != 1:
        raise AssertionError("nonempty cache setup did not establish one valid prior token")


@contextmanager
def _instrument_cache_boundaries(
    runtime: _FamilyRuntime,
    cache: Any,
    counters: dict[str, int],
):
    from gemma4_fa4 import transformers_integration as integration

    cache_class = type(cache)
    original_update = cache_class.update
    cache_class_owned_update = "update" in cache_class.__dict__
    op_names = (
        ("h100_local_fwd", "h100_local_layer_fwd")
        if runtime.name == "local"
        else ("h100_global_fwd", "h100_global_layer_fwd")
    )
    original_ops = {name: getattr(integration, name) for name in op_names}

    def counted_update(self, *args, **kwargs):
        if self is cache:
            counters["cache_update"] += 1
        return original_update(self, *args, **kwargs)

    cache_class.update = counted_update
    for name, original_op in original_ops.items():

        def counted_op(*args, _original_op=original_op, **kwargs):
            counters["custom_op_entry"] += 1
            return _original_op(*args, **kwargs)

        setattr(integration, name, counted_op)
    try:
        yield
    finally:
        if cache_class_owned_update:
            cache_class.update = original_update
        else:
            delattr(cache_class, "update")
        for name, original_op in original_ops.items():
            setattr(integration, name, original_op)


def _run_cache_rejection_case(
    runtime: _FamilyRuntime,
    *,
    backend: str,
    cache_type: str,
    cache_state: str,
    seed: int,
) -> dict[str, Any]:
    cache = _make_real_cache(cache_type, runtime)
    if cache_state == "nonempty":
        _populate_cache(cache, runtime, seed=seed)
    elif cache_state != "empty":
        raise ValueError(f"unknown cache state: {cache_state}")

    logical_length = _cache_seq_length(cache, runtime.layer_idx)
    expected_length = 1 if cache_state == "nonempty" else 0
    if logical_length != expected_length:
        raise AssertionError(
            f"{cache_type}/{cache_state} setup length={logical_length}, expected={expected_length}"
        )
    inputs = _make_inputs(
        runtime,
        1,
        seed=seed + 1,
        position_start=logical_length,
    )
    before = _cache_state_snapshot(cache, runtime.layer_idx)
    entry_counters = {"layer_entry": 0, "cache_update": 0, "custom_op_entry": 0}
    counters_before = dict(entry_counters)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    capture = _CapturingBackend(backend)
    compiled = torch.compile(
        _cache_object_callable(runtime, cache, entry_counters),
        backend=capture,
        fullgraph=True,
        dynamic=True,
    )
    error: Exception | None = None
    output: torch.Tensor | None = None
    with _instrument_cache_boundaries(runtime, cache, entry_counters):
        try:
            with torch.inference_mode():
                output = compiled(*inputs)
        except Exception as exc:
            error = exc

    after = _cache_state_snapshot(cache, runtime.layer_idx)
    counters_after = dict(entry_counters)
    if error is None:
        assert output is not None
        raise AssertionError(
            f"{runtime.name}/{backend}/{cache_type}/{cache_state} admitted a compiled cache "
            f"request: output_shape={tuple(output.shape)}, output_dtype={output.dtype}, "
            f"entry_counters={counters_after}, cache_unchanged={before == after}"
        )
    error_type = type(error).__name__
    message = str(error)
    classification = _classify_cache_object_rejection(error_type, message)
    if classification is None:
        raise AssertionError(
            f"{runtime.name}/{backend}/{cache_type}/{cache_state} failed for an unrelated "
            f"reason ({error_type}): {message}"
        ) from error
    if counters_after != counters_before:
        raise AssertionError(
            f"{runtime.name}/{backend}/{cache_type}/{cache_state} rejection occurred after an "
            f"entry boundary advanced: before={counters_before}, after={counters_after}"
        ) from error
    if before != after:
        raise AssertionError(
            f"{runtime.name}/{backend}/{cache_type}/{cache_state} rejection mutated cache state"
        ) from error
    if capture.graphs:
        raise AssertionError(
            f"{runtime.name}/{backend}/{cache_type}/{cache_state} reached the compiler backend "
            "before cache rejection"
        ) from error

    return {
        "status": "rejected",
        "family": runtime.name,
        "layer_idx": runtime.layer_idx,
        "backend": backend,
        "cache_type": cache_type,
        "cache_state": cache_state,
        "rejection_class": classification,
        "error_type": error_type,
        "error": message,
        "cache_unchanged": True,
        "cache_before": before,
        "cache_after": after,
        "entry_counters_before": counters_before,
        "entry_counters_after": counters_after,
        "compiler_backend_graph_count": len(capture.graphs),
        "dynamo_graph_break_count": _graph_break_count(),
    }


def _run_negative_cache_object_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run EXP-0018's real cache-object fail-closed Cartesian matrix."""

    _require_h100()
    families = _selected(args.family, tuple(FAMILY_LAYERS))
    backends = _selected(args.backend, BACKENDS)
    cache_types = ("DynamicCache", "StaticCache")
    cache_states = ("empty", "nonempty")
    cache_dirs = _prepare_fresh_cache_dirs(backends)
    results = []
    for family_index, family in enumerate(families):
        runtime = _make_family_runtime(family, seed=args.seed + family_index * 100_000)
        for backend_index, backend in enumerate(backends):
            for type_index, cache_type in enumerate(cache_types):
                for state_index, cache_state in enumerate(cache_states):
                    results.append(
                        _run_cache_rejection_case(
                            runtime,
                            backend=backend,
                            cache_type=cache_type,
                            cache_state=cache_state,
                            seed=(
                                args.seed
                                + family_index * 100_000
                                + backend_index * 10_000
                                + type_index * 1000
                                + state_index * 100
                            ),
                        )
                    )
        torch._dynamo.reset()
        del runtime
        torch.cuda.empty_cache()

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "request": {
            "mode": "negative_cache_object",
            "families": list(families),
            "backends": list(backends),
            "cache_types": list(cache_types),
            "cache_states": list(cache_states),
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "caches": {name: _cache_inventory(path) for name, path in cache_dirs.items()},
        },
        "summary": {
            "case_count": len(results),
            "rejected_before_entry_count": len(results),
        },
        "results": results,
    }


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.localize_outer_drift:
        return _run_outer_drift_localization(args)
    if args.negative_cache_object:
        return _run_negative_cache_object_probe(args)
    _require_h100()
    families = _selected(args.family, tuple(FAMILY_LAYERS))
    backends = _selected(args.backend, BACKENDS)
    cache_dirs = _prepare_fresh_cache_dirs(backends)
    results = {}
    for index, family in enumerate(families):
        results[family] = _run_family(
            family,
            backends,
            args.lengths,
            seed=args.seed + index * 100_000,
        )
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "passed",
        "request": {
            "families": list(families),
            "backends": list(backends),
            "lengths": list(args.lengths),
            "declared_matrix_complete": tuple(args.lengths) == DEFAULT_LENGTHS,
            "shape_policies": [
                "pytorch_default_dynamic",
                "scoped_backed_size_oblivious",
            ],
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "caches": {name: _cache_inventory(path) for name, path in cache_dirs.items()},
        },
        "families": results,
    }


def _valid_full_layer_comparison(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    maximum = value.get("max_abs")
    mean = value.get("mean_abs")
    return (
        value.get("exact") is True
        and value.get("bitwise") is True
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(maximum)
        and maximum >= 0
        and isinstance(mean, (int, float))
        and not isinstance(mean, bool)
        and math.isfinite(mean)
        and mean >= 0
    )


def _valid_tensor_comparison(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    maximum = value.get("max_abs")
    mean = value.get("mean_abs")
    return (
        type(value.get("bitwise")) is bool
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(maximum)
        and maximum >= 0
        and isinstance(mean, (int, float))
        and not isinstance(mean, bool)
        and math.isfinite(mean)
        and mean >= 0
        and isinstance(value.get("shape"), list)
        and all(type(dimension) is int and dimension >= 0 for dimension in value["shape"])
        and isinstance(value.get("dtype"), str)
    )


def _valid_reference_result(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("passed") is not True:
        return False
    maximum = value.get("max_abs")
    mean = value.get("mean_abs")
    return (
        isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(maximum)
        and maximum >= 0
        and isinstance(mean, (int, float))
        and not isinstance(mean, bool)
        and math.isfinite(mean)
        and mean >= 0
    )


def _validate_outer_drift_localization(report: dict[str, Any]) -> None:
    request = report.get("request")
    localization = report.get("localization")
    caches = report.get("environment", {}).get("caches")
    application_keys = report.get("fa4_forward_application_keys")
    expected_request = {
        "mode": "localize_outer_drift",
        "family": "local",
        "backend": "inductor",
        "seqlen": 1023,
        "runtime_seed": 17017,
        "input_seed": 19020,
    }
    if request != expected_request:
        raise ValueError("outer-drift localization request is not the predeclared coordinate")
    if not isinstance(caches, dict) or set(caches) != {"fa4", "inductor"}:
        raise ValueError("outer-drift localization lacks isolated FA4/Inductor caches")
    if any(
        not isinstance(item.get("path"), str)
        or not isinstance(item.get("file_count"), int)
        or item.get("file_count") < 0
        or not isinstance(item.get("total_bytes"), int)
        or item.get("total_bytes") < 0
        for item in caches.values()
    ):
        raise ValueError("outer-drift localization has an invalid cache inventory")
    if (
        not isinstance(application_keys, dict)
        or application_keys.get("new_class_count") != 1
        or len(application_keys.get("added", ())) != 1
        or not isinstance(application_keys.get("before"), list)
        or not isinstance(application_keys.get("after"), list)
    ):
        raise ValueError("outer-drift localization lacks one bounded FA4 application class")
    if not isinstance(localization, dict):
        raise ValueError("outer-drift localization payload is missing")

    qkv = localization.get("qkv")
    prepared = localization.get("prepared_opaque")
    projection = localization.get("output_projection")
    hybrid = localization.get("hybrid_compiled_qkv")
    whole = localization.get("whole_layer")
    stages = (qkv, prepared, projection, whole)
    if any(
        not isinstance(stage, dict)
        or not 1 <= stage.get("graph_count", 0) <= 2
        or stage.get("graph_break_count") != 0
        for stage in stages
    ):
        raise ValueError("outer-drift localization stage graph evidence is incomplete")
    components = qkv.get("components")
    if (
        qkv.get("distinct_storage") is not True
        or not isinstance(components, dict)
        or set(components) != {"q", "k", "v"}
        or not all(_valid_tensor_comparison(value) for value in components.values())
    ):
        raise ValueError("outer-drift localization Q/K/V evidence is incomplete")
    reference = prepared.get("reference")
    if (
        prepared.get("custom_op_node") is not True
        or not _valid_tensor_comparison(prepared.get("output"))
        or not _valid_tensor_comparison(prepared.get("lse"))
        or prepared["output"].get("bitwise") is not True
        or prepared["lse"].get("bitwise") is not True
        or not isinstance(reference, dict)
        or not _valid_reference_result(reference.get("output"))
        or not _valid_reference_result(reference.get("lse"))
    ):
        raise ValueError("outer-drift localization lost prepared FA4 opacity/reference evidence")
    if (
        not _valid_tensor_comparison(projection.get("output"))
        or not isinstance(hybrid, dict)
        or not _valid_tensor_comparison(hybrid.get("output"))
        or whole.get("custom_op_node") is not True
        or not _valid_tensor_comparison(whole.get("output"))
        or whole["output"].get("bitwise") is not False
        or whole["output"].get("within_exp0018_frozen_tolerance") is not False
    ):
        raise ValueError("outer-drift localization did not reproduce the whole-layer falsifier")
    observed_outer_comparisons = (
        *components.values(),
        projection["output"],
        hybrid["output"],
    )
    if not any(value.get("bitwise") is False for value in observed_outer_comparisons):
        raise ValueError("outer-drift localization did not identify drift outside prepared FA4")
    if (
        localization.get("prepared_opaque_bitwise") is not True
        or localization.get("outer_drift_observed") is not True
        or localization.get("exp0018_failure_reproduced") is not True
        or localization.get("hypothesis_supported") is not True
    ):
        raise ValueError("outer-drift localization does not support the predeclared hypothesis")


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("experiment") != EXPERIMENT:
        raise ValueError("compile report has an invalid schema version or experiment")
    if report.get("status") != "passed":
        raise ValueError("only passed compile reports satisfy the success schema")
    request = report.get("request")
    if isinstance(request, dict) and request.get("mode") == "localize_outer_drift":
        _validate_outer_drift_localization(report)
        return
    if isinstance(request, dict) and request.get("mode") == "negative_cache_object":
        results = report.get("results")
        summary = report.get("summary")
        caches = report.get("environment", {}).get("caches")
        families = request.get("families", ())
        backends = request.get("backends", ())
        cache_types = request.get("cache_types", ())
        cache_states = request.get("cache_states", ())
        expected_coordinates = {
            (family, backend, cache_type, cache_state)
            for family in families
            for backend in backends
            for cache_type in cache_types
            for cache_state in cache_states
        }
        expected_caches = {"fa4"}
        if "inductor" in backends:
            expected_caches.add("inductor")
        if not isinstance(caches, dict) or set(caches) != expected_caches:
            raise ValueError("cache-negative report lacks its isolated cache inventory")
        if not isinstance(results, list) or len(results) != len(expected_coordinates):
            raise ValueError("cache-negative report has an incomplete Cartesian matrix")
        observed_coordinates = {
            (
                result.get("family"),
                result.get("backend"),
                result.get("cache_type"),
                result.get("cache_state"),
            )
            for result in results
        }
        if observed_coordinates != expected_coordinates:
            raise ValueError("cache-negative report coordinates disagree with the request")
        zero_entries = {"layer_entry": 0, "cache_update": 0, "custom_op_entry": 0}
        for result in results:
            before = result.get("cache_before")
            after = result.get("cache_after")
            target = before.get("target_layer", {}) if isinstance(before, dict) else {}
            expected_length = 1 if result.get("cache_state") == "nonempty" else 0
            if (
                result.get("status") != "rejected"
                or not str(result.get("rejection_class", "")).startswith("exp0018_")
                or result.get("cache_unchanged") is not True
                or before != after
                or result.get("entry_counters_before") != zero_entries
                or result.get("entry_counters_after") != zero_entries
                or result.get("compiler_backend_graph_count") != 0
                or target.get("logical_length") != expected_length
                or not isinstance(target.get("storage_identity"), dict)
                or not isinstance(target.get("content_sha256"), str)
                or not isinstance(target.get("allocation"), dict)
            ):
                raise ValueError("cache-negative report lacks a pre-entry immutable rejection")
        if (
            not isinstance(summary, dict)
            or summary.get("case_count") != len(expected_coordinates)
            or summary.get("rejected_before_entry_count") != len(expected_coordinates)
        ):
            raise ValueError("cache-negative report summary is inconsistent")
        return
    families = report.get("families")
    if not isinstance(request, dict) or not isinstance(families, dict):
        raise ValueError("compile report is missing request/family objects")
    caches = report.get("environment", {}).get("caches")
    expected_caches = {"fa4"}
    if "inductor" in request.get("backends", ()):
        expected_caches.add("inductor")
    if not isinstance(caches, dict) or set(caches) != expected_caches:
        raise ValueError("compile report is missing the isolated cache inventory")
    if any(
        not isinstance(item.get("path"), str)
        or not isinstance(item.get("file_count"), int)
        or item.get("file_count") < 0
        or not isinstance(item.get("total_bytes"), int)
        or item.get("total_bytes") < 0
        for item in caches.values()
    ):
        raise ValueError("compile report contains an invalid cache inventory")
    if set(families) != set(request.get("families", ())):
        raise ValueError("compile report family keys disagree with the request")
    requested_backends = set(request.get("backends", ()))
    requested_lengths = list(request.get("lengths", ()))
    for family, result in families.items():
        if family not in FAMILY_LAYERS or result.get("status") != "passed":
            raise ValueError("compile report contains an invalid family result")
        if result.get("hidden_size") != GEMMA4_31B.hidden_size:
            raise ValueError("compile report did not use the locked hidden size")
        if result.get("fake_tensor", {}).get("real_body_entered") is not False:
            raise ValueError("compile report lacks the fake shape-only proof")
        if result.get("opcheck", {}).get("status") != "passed":
            raise ValueError("compile report lacks a passing opcheck")
        application_keys = result.get("fa4_forward_application_keys", {})
        if (
            application_keys.get("new_class_count") != 1
            or application_keys.get("reused_across_compiler_matrix") is not True
            or len(application_keys.get("added_family_class", ())) != 1
            or application_keys.get("warmed") != application_keys.get("after")
        ):
            raise ValueError("compile report lacks bounded FA4 application-key evidence")
        references = result.get("direct_reference")
        if (
            not isinstance(references, list)
            or [item.get("seqlen") for item in references] != requested_lengths
            or not all(
                item.get("projection_transport_bitwise") is True
                and item.get("whole_layer_output_bitwise") is True
                and item.get("whole_layer_lse_bitwise") is True
                and item.get("dormant_weight_output_bitwise") is True
                and item.get("dormant_weight_lse_bitwise") is True
                for item in references
            )
        ):
            raise ValueError("compile report reference matrix disagrees with requested lengths")
        backend_results = result.get("backends")
        if not isinstance(backend_results, dict) or set(backend_results) != requested_backends:
            raise ValueError("compile report backend keys disagree with the request")
        for policies in backend_results.values():
            if not isinstance(policies, dict) or set(policies) != {
                "public_dynamic_default",
                "scoped_backed_size_oblivious",
            }:
                raise ValueError("compile report does not preserve both shape-policy results")
            public = policies["public_dynamic_default"]
            expected_public_graphs = (
                2
                if 1 in requested_lengths and any(length > 1 for length in requested_lengths)
                else 1
            )
            expected_graph_classes = []
            if 1 in requested_lengths:
                expected_graph_classes.append({"class": "S1", "lengths": [1]})
            nonsingleton_lengths = [length for length in requested_lengths if length > 1]
            if nonsingleton_lengths:
                expected_graph_classes.append({"class": "S>1", "lengths": nonsingleton_lengths})
            public_cases = public.get("cases")
            if (
                public.get("status") != "passed"
                or public.get("shape_policy", {}).get("backed_size_oblivious") is not False
                or public.get("shape_policy", {}).get("is_pytorch_default") is not True
                or public.get("graph_count") != expected_public_graphs
                or public.get("expected_graph_count") != expected_public_graphs
                or public.get("graph_classes") != expected_graph_classes
                or public.get("bounded_exactly_s1_or_gt1") is not True
                or public.get("one_graph_requirement_met") is not (expected_public_graphs == 1)
                or public.get("graph_break_count") != 0
                or public.get("custom_op_node") is not True
                or public.get("weight_snapshot_node") is not False
                or not isinstance(public_cases, list)
                or [item.get("seqlen") for item in public_cases] != requested_lengths
                or not all(_valid_full_layer_comparison(item) for item in public_cases)
            ):
                raise ValueError("compile report lost the public dynamic-shape diagnostic")
            backend_result = policies["scoped_backed_size_oblivious"]
            cases = backend_result.get("cases")
            if (
                backend_result.get("status") != "passed"
                or backend_result.get("shape_policy", {}).get("backed_size_oblivious") is not True
                or backend_result.get("shape_policy", {}).get("is_pytorch_default") is not False
                or backend_result.get("graph_count") != 1
                or backend_result.get("graph_break_count") != 0
                or backend_result.get("custom_op_node") is not True
                or backend_result.get("weight_snapshot_node") is not False
                or not _valid_full_layer_comparison(backend_result.get("nondefault_stream"))
                or backend_result.get("reset_positions", {}).get("status") != "rejected"
                or backend_result.get("cache", {}).get("status") != "rejected"
                or not isinstance(cases, list)
                or [item.get("seqlen") for item in cases] != requested_lengths
                or not all(_valid_full_layer_comparison(item) for item in cases)
            ):
                raise ValueError("compile report contains incomplete backend evidence")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILY_LAYERS, "all"), default="all")
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    parser.add_argument(
        "--lengths",
        type=_parse_lengths,
        default=DEFAULT_LENGTHS,
        help="comma-separated lengths in 1..1024 (default: 1,32,33,1023,1024)",
    )
    parser.add_argument("--seed", type=int, default=17017)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--negative-cache-object",
        action="store_true",
        help=(
            "run only real empty/nonempty DynamicCache and StaticCache fail-closed cases "
            "for the selected families/backends; any entry or mutation is a probe failure"
        ),
    )
    modes.add_argument(
        "--localize-outer-drift",
        action="store_true",
        help=(
            "run only EXP-0019's fixed local/Inductor/S1023 QKV, prepared-FA4, "
            "projection, hybrid, and whole-layer drift localization"
        ),
    )
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    return parser


def _emit(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = _run_probe(args)
        _validate_report(report)
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
                "negative_cache_object": args.negative_cache_object,
                "localize_outer_drift": args.localize_outer_drift,
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
