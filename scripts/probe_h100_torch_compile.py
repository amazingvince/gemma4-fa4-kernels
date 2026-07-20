#!/usr/bin/env python3
"""EXP-0017 H100 fullgraph probe for the pinned Gemma 4 attention layers.

This is a correctness and compiler-boundary probe, not a benchmark.  It keeps
the exact locked model width and runs actual pinned ``Gemma4TextAttention``
layers 0 and 5 through the registered no-cache FA4 custom operators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections.abc import Callable, Sequence
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

EXPERIMENT = "EXP-0017"
SCHEMA_VERSION = 1
PINNED_TORCH_VERSION = "2.8.0+cu128"
DEFAULT_LENGTHS = (1, 32, 33, 1023, 1024)
FAMILY_LAYERS = {"local": 0, "global": 5}
BACKENDS = ("eager", "inductor")

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


class _CapturingBackend:
    """Capture Dynamo graphs while delegating to the named stock backend."""

    def __init__(self, name: str) -> None:
        from torch._dynamo.backends.registry import lookup_backend

        self.name = name
        self._delegate = lookup_backend(name)
        self.graphs: list[dict[str, Any]] = []

    def __call__(self, graph_module, example_inputs):
        nodes = [{"op": node.op, "target": str(node.target)} for node in graph_module.graph.nodes]
        self.graphs.append({"nodes": nodes})
        return self._delegate(graph_module, example_inputs)


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
    else:
        mask_builder = create_causal_mask
        custom_op = h100_torch_ops.h100_global_fwd
        custom_op_fragment = "h100_global_fwd"
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
    )


def _make_inputs(
    runtime: _FamilyRuntime,
    seqlen: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        (1, seqlen, runtime.config.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(seqlen, dtype=torch.int64, device="cuda").unsqueeze(0)
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


def _cache_object_callable(runtime: _FamilyRuntime, cache: Any):
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


def _prepare_qkv(
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
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise AssertionError("prepared K and V must use distinct storage")
    return q, k, v


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


def _direct_reference_case(
    runtime: _FamilyRuntime,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    eager_layer_output: torch.Tensor,
) -> dict[str, Any]:
    hidden, cos, sin, positions = inputs
    q, k, v = _prepare_qkv(runtime, hidden, cos, sin)
    packed = torch.zeros_like(positions)
    output, lse = runtime.custom_op(q, k, v, positions, packed)
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
        with FakeTensorMode():
            q = torch.empty(
                (1, runtime.spec.num_q_heads, seqlen, runtime.spec.head_dim_qk),
                dtype=torch.bfloat16,
                device="cuda",
            )
            k = torch.empty(
                (1, runtime.spec.num_kv_heads, seqlen, runtime.spec.head_dim_qk),
                dtype=torch.bfloat16,
                device="cuda",
            )
            v = torch.empty(
                (1, runtime.spec.num_kv_heads, seqlen, runtime.spec.head_dim_v),
                dtype=torch.bfloat16,
                device="cuda",
            )
            positions = torch.arange(seqlen, dtype=torch.int64, device="cuda").unsqueeze(0)
            packed = torch.zeros((1, seqlen), dtype=torch.int64, device="cuda")
            output, lse = runtime.custom_op(q, k, v, positions, packed)
            if output.shape != (1, seqlen, runtime.spec.num_q_heads, runtime.spec.head_dim_v):
                raise AssertionError("fake custom op returned an invalid BSHD shape")
            if lse.shape != (1, runtime.spec.num_q_heads, seqlen):
                raise AssertionError("fake custom op returned an invalid LSE shape")
            if output.dtype != torch.bfloat16 or lse.dtype != torch.float32:
                raise AssertionError("fake custom op returned invalid output dtypes")
    finally:
        h100_torch_ops.fa4_local_text_forward = original_local
        h100_torch_ops.fa4_global_text_forward = original_global
    return {
        "status": "passed",
        "seqlen": seqlen,
        "output_shape": [1, seqlen, runtime.spec.num_q_heads, runtime.spec.head_dim_v],
        "lse_shape": [1, runtime.spec.num_q_heads, seqlen],
        "real_body_entered": False,
    }


def _opcheck(runtime: _FamilyRuntime, *, seed: int) -> dict[str, Any]:
    inputs = _make_inputs(runtime, 33, seed=seed)
    q, k, v = _prepare_qkv(runtime, *inputs[:3])
    positions = inputs[3]
    packed = torch.zeros_like(positions)
    result = torch.library.opcheck(
        runtime.custom_op,
        (q, k, v, positions, packed),
        raise_exception=False,
    )
    serialized = {name: str(value) for name, value in result.items()}
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


def _is_expected_cache_rejection(message: str) -> bool:
    lowered = message.lower()
    return (
        "cache" in lowered
        or EXPERIMENT in message
        or ("observed exception" in lowered and "UnsupportedH100Path" in message)
    )


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
        if not _is_expected_cache_rejection(message):
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
            if not torch.equal(eager_output, compiled_output):
                maximum, mean = _finite_error(compiled_output, eager_output)
                raise AssertionError(
                    f"{runtime.name}/{backend}/public/S{seqlen} is not bitwise eager-equivalent "
                    f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
                )
            cases.append({"seqlen": seqlen, "bitwise_eager_compiled": True})

    graph_count = len(capture.graphs)
    graph_break_count = _graph_break_count()
    expected_graph_count = 2 if 1 in lengths and any(length > 1 for length in lengths) else 1
    custom_op_node = _contains_custom_op(capture.graphs, runtime.custom_op_fragment)
    if graph_count != expected_graph_count:
        raise AssertionError(
            f"{runtime.name}/{backend} public dynamic diagnostic produced {graph_count} graphs; "
            f"expected the pinned {expected_graph_count}-class S1 specialization outcome"
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
    return {
        "status": "observed",
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
        "one_graph_requirement_met": graph_count == 1,
        "graph_break_count": graph_break_count,
        "custom_op_node": custom_op_node,
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
            bitwise = torch.equal(eager_output, compiled_output)
            if not bitwise:
                maximum, mean = _finite_error(compiled_output, eager_output)
                raise AssertionError(
                    f"{runtime.name}/{backend}/S{seqlen} is not bitwise eager-equivalent "
                    f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
                )
            references.append(_direct_reference_case(runtime, inputs, eager_output))
            cases.append(
                {
                    "seqlen": seqlen,
                    "bitwise_eager_compiled": True,
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
        if not torch.equal(stream_output, stream_expected):
            raise AssertionError(f"{runtime.name}/{backend} changed on a nondefault CUDA stream")

        reset_positions = torch.zeros_like(stream_inputs[3])
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
    custom_op_node = _contains_custom_op(capture.graphs, runtime.custom_op_fragment)
    if graph_count != 1:
        raise AssertionError(
            f"{runtime.name}/{backend} produced {graph_count} graphs for lengths {tuple(lengths)}; "
            "the public dynamic=True one-graph requirement is unmet"
        )
    if graph_break_count != 0:
        raise AssertionError(f"{runtime.name}/{backend} produced {graph_break_count} graph breaks")
    if not custom_op_node:
        raise AssertionError(f"{runtime.name}/{backend} graph lacks the project custom-op node")

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
            "graph_nodes": capture.graphs,
            "cases": cases,
            "nondefault_stream_bitwise": True,
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


def _selected(value: str, all_values: Sequence[str]) -> tuple[str, ...]:
    return tuple(all_values) if value == "all" else (value,)


def _cache_seq_length(cache: Any, layer_idx: int) -> int:
    value = cache.get_seq_length(layer_idx)
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _run_negative_cache_object_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Require an actual pinned-layer DynamicCache request to fail closed."""

    _require_h100()
    cache_dirs = _prepare_fresh_cache_dirs(("eager",))
    try:
        from transformers import DynamicCache
    except Exception as exc:
        raise RuntimeError("the pinned Transformers DynamicCache is required") from exc

    runtime = _make_family_runtime("local", seed=args.seed)
    cache = DynamicCache(config=runtime.config)
    inputs = _make_inputs(runtime, 1, seed=args.seed + 1)
    before = _cache_seq_length(cache, runtime.layer_idx)
    compiled = torch.compile(
        _cache_object_callable(runtime, cache),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    try:
        with torch.inference_mode():
            output = compiled(*inputs)
    except Exception as exc:
        after = _cache_seq_length(cache, runtime.layer_idx)
        message = str(exc)
        if not _is_expected_cache_rejection(message):
            raise AssertionError(
                f"DynamicCache request failed for an unrelated reason: {message}"
            ) from exc
        if after != before:
            raise AssertionError(
                "DynamicCache request raised only after mutating cache state: "
                f"before={before}, after={after}"
            ) from exc
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "passed",
            "request": {"mode": "negative_cache_object", "seed": args.seed},
            "environment": {
                "torch": torch.__version__,
                "device_name": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "caches": {name: _cache_inventory(path) for name, path in cache_dirs.items()},
            },
            "result": {
                "status": "rejected",
                "layer_idx": runtime.layer_idx,
                "cache_type": type(cache).__name__,
                "cache_length_before": before,
                "cache_length_after": after,
                "error_type": type(exc).__name__,
                "error": message,
            },
        }

    after = _cache_seq_length(cache, runtime.layer_idx)
    raise AssertionError(
        "actual pinned layer 0 admitted an unsupported DynamicCache object under fullgraph: "
        f"output_shape={tuple(output.shape)}, output_dtype={output.dtype}, "
        f"cache_length_before={before}, cache_length_after={after}"
    )


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
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


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("experiment") != EXPERIMENT:
        raise ValueError("compile report has an invalid schema version or experiment")
    if report.get("status") != "passed":
        raise ValueError("only passed compile reports satisfy the success schema")
    request = report.get("request")
    if isinstance(request, dict) and request.get("mode") == "negative_cache_object":
        result = report.get("result")
        caches = report.get("environment", {}).get("caches")
        if (
            not isinstance(result, dict)
            or result.get("status") != "rejected"
            or result.get("layer_idx") != 0
            or result.get("cache_type") != "DynamicCache"
            or result.get("cache_length_before") != result.get("cache_length_after")
            or not isinstance(caches, dict)
            or set(caches) != {"fa4"}
        ):
            raise ValueError("compile report lacks the DynamicCache fail-closed proof")
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
            if (
                public.get("status") != "observed"
                or public.get("shape_policy", {}).get("backed_size_oblivious") is not False
                or public.get("shape_policy", {}).get("is_pytorch_default") is not True
                or public.get("graph_count") != expected_public_graphs
                or public.get("expected_graph_count") != expected_public_graphs
                or public.get("one_graph_requirement_met") is not (expected_public_graphs == 1)
                or public.get("graph_break_count") != 0
                or public.get("custom_op_node") is not True
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
                or backend_result.get("nondefault_stream_bitwise") is not True
                or backend_result.get("reset_positions", {}).get("status") != "rejected"
                or backend_result.get("cache", {}).get("status") != "rejected"
                or not isinstance(cases, list)
                or [item.get("seqlen") for item in cases] != requested_lengths
                or not all(item.get("bitwise_eager_compiled") is True for item in cases)
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
    parser.add_argument(
        "--negative-cache-object",
        action="store_true",
        help=(
            "run only the actual layer-0 DynamicCache fail-closed sentinel; "
            "admission is a probe failure"
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
