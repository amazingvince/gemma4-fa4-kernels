"""Native-shape Gemma 4 text attention through the public FA4 API.

This integration module is deliberately separate from the model-agnostic
FlashAttention implementation.  It validates Gemma's prepared Q/K/V contract,
then forwards the original heads and dimensions without padding or KV repeats.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .model_spec import AttentionLayerSpec
from .transformers_integration import (
    BACKEND_NAME,
    Gemma4MaskPlan,
    UnsupportedH100Path,
    _as_mask_plan,
    _fa4_layout_is_legal,
    _mask_plan_matches_native_contract,
    _normalize_position_ids,
    _pack_local_inputs,
    _packed_sequence_ids,
    _padding_mask,
    _unpack_local_result,
    register_gemma4_fa4_h100,
)

BACKEND_NAME_NATIVE = "gemma4_fa4_h100_native"
BACKEND_NAME_GLOBAL_NATIVE = "gemma4_fa4_h100_global_native"


@dataclass(frozen=True)
class NativeGemmaGeometry:
    model: str
    layer_count: int
    local_q_heads: int
    local_kv_heads: int
    global_q_heads: int
    global_kv_heads: int


GEMMA4_NATIVE_GEOMETRIES = (
    NativeGemmaGeometry("12b", 48, 16, 8, 16, 1),
    NativeGemmaGeometry("31b", 60, 32, 16, 32, 4),
)
_ORACLE_FAMILIES: set[tuple[str, str]] = set()


def _load_flash_attn_func() -> Callable:
    from flash_attn.cute import flash_attn_func

    return flash_attn_func


def _load_flash_attn_varlen_func() -> Callable:
    from flash_attn.cute import flash_attn_varlen_func

    return flash_attn_varlen_func


def _load_flash_attn2_func() -> Callable:
    from flash_attn import flash_attn_func

    return flash_attn_func


def _load_flash_attn2_varlen_func() -> Callable:
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func


def _deterministic_requested(value: Any = None) -> bool:
    """Match FlashAttention's opt-in deterministic policy."""

    if value is not None:
        if not isinstance(value, bool):
            raise ValueError("deterministic must be bool when provided")
        return value
    raw = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0")
    if raw not in {"0", "1"}:
        raise ValueError("FLASH_ATTENTION_DETERMINISTIC must be 0 or 1")
    return raw == "1"


def _positions_are_packed(
    position_ids: torch.Tensor | None,
    *,
    padding: torch.Tensor | None,
    q_length: int,
    kv_length: int,
) -> bool:
    """Detect position resets while avoiding one device sync per layer.

    Transformers passes the same position tensor to every decoder layer. Cache
    only the unpadded result on that exact tensor and mutation version; padded
    inputs still inspect their layer-specific valid-token mask directly.
    """

    if position_ids is None or q_length != kv_length:
        return False
    if padding is None:
        return _packed_sequence_ids(position_ids) is not None

    for batch_idx in range(position_ids.shape[0]):
        values = position_ids[batch_idx, padding[batch_idx]]
        if values.numel() > 1 and bool((torch.diff(values) != 1).any().detach().item()):
            return True
    return False


def _layer_index(module: Any) -> int:
    layer_idx = getattr(module, "layer_idx", None)
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise ValueError("Gemma 4 attention module must expose an integer layer_idx")
    return layer_idx


def _prepared_geometry(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[NativeGemmaGeometry, AttentionLayerSpec]:
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("prepared Q/K/V must be rank-4 BHSD tensors")
    if not (query.device == key.device == value.device) or query.device.type != "cuda":
        raise ValueError("prepared Q/K/V must share one CUDA device")
    if not (query.dtype == key.dtype == value.dtype == torch.bfloat16):
        raise ValueError("Gemma 4 native integration requires BF16 prepared Q/K/V")
    if any(tensor.stride(-1) != 1 for tensor in (query, key, value)):
        raise ValueError("prepared Q/K/V must have unit head-dimension stride")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("prepared Q/K/V batch sizes must match")
    if key.shape[2] != value.shape[2]:
        raise ValueError("prepared K/V sequence lengths must match")
    if key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
        raise ValueError("prepared K and V must be distinct operands")

    q_heads, q_dim = query.shape[1], query.shape[-1]
    kv_heads, k_dim, v_dim = key.shape[1], key.shape[-1], value.shape[-1]
    if q_dim != k_dim or k_dim != v_dim or q_dim not in (256, 512):
        raise ValueError("Gemma 4 native integration requires symmetric d256 or d512")
    is_global = q_dim == 512
    candidates = [
        geometry
        for geometry in GEMMA4_NATIVE_GEOMETRIES
        if (
            (geometry.global_q_heads, geometry.global_kv_heads)
            if is_global
            else (geometry.local_q_heads, geometry.local_kv_heads)
        )
        == (q_heads, kv_heads)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"prepared head geometry Hq={q_heads}, Hkv={kv_heads}, d={q_dim} "
            "is not a native Gemma 4 12B/31B shape"
        )
    geometry = candidates[0]
    layer_idx = _layer_index(module)
    if not 0 <= layer_idx < geometry.layer_count:
        raise ValueError(
            f"Gemma 4 {geometry.model} layer_idx must be in 0..{geometry.layer_count - 1}"
        )
    expected_global = (layer_idx + 1) % 6 == 0
    if expected_global != is_global:
        raise ValueError("prepared head dimension conflicts with the layer family")
    spec = AttentionLayerSpec(
        kind="full_attention" if is_global else "sliding_attention",
        head_dim_qk=q_dim,
        head_dim_v=v_dim,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        softmax_scale=1.0,
        is_causal=True,
        vision_bidirectional_within_block=not is_global,
        sliding_window=None if is_global else 1024,
        rope_type="proportional" if is_global else "default",
        rope_theta=1_000_000.0 if is_global else 10_000.0,
        partial_rotary_factor=0.25 if is_global else 1.0,
        projection_source_shared_between_k_and_v=is_global,
    )
    if getattr(module, "head_dim", q_dim) != q_dim:
        raise ValueError("module head_dim conflicts with prepared Q/K/V")
    if getattr(module, "num_key_value_groups", spec.qhead_per_kvhead) != spec.qhead_per_kvhead:
        raise ValueError("module GQA ratio conflicts with prepared Q/K/V")
    if getattr(module, "scaling", 1.0) != 1.0:
        raise ValueError("Gemma 4 attention scale must be exactly 1.0")
    return geometry, spec


def _record_route(
    module: Any,
    *,
    route: str,
    geometry: NativeGemmaGeometry,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segments: list[int],
) -> None:
    module._gemma4_fa4_last_path = route
    counts = getattr(module, "_gemma4_fa4_route_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        module._gemma4_fa4_route_counts = counts
    counts[route] = int(counts.get(route, 0)) + 1
    module._gemma4_fa4_native_geometry = {
        "model": geometry.model,
        "q": tuple(query.shape),
        "k": tuple(key.shape),
        "v": tuple(value.shape),
        "k_v_distinct": key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr(),
        "segments": segments,
    }


def _prepared_backward_oracle(
    spec: AttentionLayerSpec,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> dict[str, Any] | None:
    if (
        spec.kind != "full_attention"
        or os.environ.get("GEMMA4_FA4_PREPARED_BACKWARD_ORACLE") != "1"
    ):
        return None

    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        q_candidate = query.detach().transpose(1, 2).requires_grad_(True)
        k_candidate = key.detach().transpose(1, 2).requires_grad_(True)
        v_candidate = value.detach().transpose(1, 2).requires_grad_(True)
        candidate_output, _ = _load_flash_attn_func()(
            q_candidate,
            k_candidate,
            v_candidate,
            causal=True,
            softmax_scale=1.0,
            num_splits=1,
            pack_gqa=False,
            deterministic=True,
            return_lse=True,
        )

        q_baseline = query.detach().requires_grad_(True)
        k_baseline = key.detach().requires_grad_(True)
        v_baseline = value.detach().requires_grad_(True)
        baseline_output = torch.nn.functional.scaled_dot_product_attention(
            q_baseline,
            k_baseline,
            v_baseline,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0,
            enable_gqa=True,
        ).transpose(1, 2)

        q_reference = query.detach().transpose(1, 2).float().requires_grad_(True)
        k_reference = key.detach().transpose(1, 2).float().requires_grad_(True)
        v_reference = value.detach().transpose(1, 2).float().requires_grad_(True)
        repeat = spec.qhead_per_kvhead
        k_expanded = k_reference.repeat_interleave(repeat, dim=2)
        v_expanded = v_reference.repeat_interleave(repeat, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q_reference, k_expanded)
        q_idx = torch.arange(scores.shape[-2], device=scores.device)[:, None]
        k_idx = torch.arange(scores.shape[-1], device=scores.device)[None, :]
        scores.masked_fill_(~(k_idx <= q_idx)[None, None], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        reference_output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v_expanded)

        generator = torch.Generator(device=query.device).manual_seed(0xFA4512)
        dout = torch.randn(
            candidate_output.shape,
            dtype=candidate_output.dtype,
            device=candidate_output.device,
            generator=generator,
        )
        candidate_grads = torch.autograd.grad(
            candidate_output,
            (q_candidate, k_candidate, v_candidate),
            dout,
        )
        baseline_grads = torch.autograd.grad(
            baseline_output,
            (q_baseline, k_baseline, v_baseline),
            dout,
        )
        reference_grads = torch.autograd.grad(
            reference_output,
            (q_reference, k_reference, v_reference),
            dout.float(),
        )

    metrics: dict[str, Any] = {}
    for name, candidate_grad, baseline_grad, reference_grad in zip(
        ("dq", "dk", "dv"),
        candidate_grads,
        baseline_grads,
        reference_grads,
        strict=True,
    ):
        baseline_grad = baseline_grad.transpose(1, 2)
        candidate_difference = (candidate_grad.float() - reference_grad).abs()
        baseline_difference = (baseline_grad.float() - reference_grad).abs()
        baseline_max_abs = float(baseline_difference.max().item())
        quantization_atol = 2.0 * float(
            (reference_grad.to(torch.bfloat16).float() - reference_grad).abs().max().item()
        )
        limit = 2.0 * baseline_max_abs + quantization_atol
        max_abs = float(candidate_difference.max().item())
        if max_abs > limit:
            raise RuntimeError(
                f"prepared-QKV FA4 backward oracle failed for {name}: "
                f"max_abs={max_abs}, baseline_max_abs={baseline_max_abs}, limit={limit}"
            )
        metrics[name] = {
            "max_abs": max_abs,
            "mean_abs": float(candidate_difference.mean().item()),
            "baseline_max_abs": baseline_max_abs,
            "baseline_mean_abs": float(baseline_difference.mean().item()),
            "limit": limit,
            "passed": True,
        }
    return metrics


def _maybe_check_prepared_oracle(
    module: Any,
    *,
    geometry: NativeGemmaGeometry,
    spec: AttentionLayerSpec,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if os.environ.get("GEMMA4_FA4_PREPARED_ORACLE") != "1":
        return
    family = (geometry.model, spec.kind)
    if family in _ORACLE_FAMILIES:
        return
    with torch.no_grad():
        q = query.detach().transpose(1, 2).float()
        k = key.detach().transpose(1, 2).float().repeat_interleave(spec.qhead_per_kvhead, dim=2)
        v = value.detach().transpose(1, 2).float().repeat_interleave(spec.qhead_per_kvhead, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k)
        q_idx = torch.arange(q.shape[1], device=q.device)[:, None]
        k_idx = torch.arange(k.shape[1], device=q.device)[None, :]
        allowed = k_idx <= q_idx
        if spec.sliding_window is not None:
            allowed = allowed & (k_idx > q_idx - spec.sliding_window)
        scores.masked_fill_(~allowed[None, None, :, :], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        reference = torch.einsum("bhqk,bkhd->bqhd", probabilities, v)
        candidate = output.detach().float()
        difference = (candidate - reference).abs()
        baseline = (
            torch.nn.functional.scaled_dot_product_attention(
                query.detach(),
                key.detach(),
                value.detach(),
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
                enable_gqa=True,
            )
            .transpose(1, 2)
            .float()
        )
        baseline_max_abs = float((baseline - reference).abs().max().item())
        quantization_atol = 2.0 * float(
            (reference.to(torch.bfloat16).float() - reference).abs().max().item()
        )
        limit = 2.0 * baseline_max_abs + quantization_atol
        max_abs = float(difference.max().item())
        if max_abs > limit:
            dump_path = os.environ.get("GEMMA4_FA4_ORACLE_DUMP")
            if dump_path:
                torch.save(
                    {
                        "geometry": geometry.model,
                        "kind": spec.kind,
                        "query": query.detach().cpu(),
                        "key": key.detach().cpu(),
                        "value": value.detach().cpu(),
                        "output": output.detach().cpu(),
                        "reference": reference.cpu(),
                    },
                    dump_path,
                )
            raise RuntimeError(
                f"prepared-QKV FA4 oracle failed for {geometry.model}/{spec.kind}: "
                f"max_abs={max_abs}, baseline_max_abs={baseline_max_abs}, limit={limit}"
            )
        backward = _prepared_backward_oracle(spec, query, key, value)
        module._gemma4_fa4_prepared_oracle = {
            "model": geometry.model,
            "kind": spec.kind,
            "max_abs": max_abs,
            "mean_abs": float(difference.mean().item()),
            "baseline_max_abs": baseline_max_abs,
            "limit": limit,
            "passed": True,
        }
        if backward is not None:
            module._gemma4_fa4_prepared_oracle["backward"] = backward
    _ORACLE_FAMILIES.add(family)


def gemma4_native_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run text-only Gemma 4 12B/31B attention at its native head geometry."""

    geometry, spec = _prepared_geometry(module, query, key, value)
    dropout = float(kwargs.get("dropout", 0.0))
    scaling = float(kwargs.get("scaling", 1.0))
    sliding_window = kwargs.get("sliding_window", spec.sliding_window)
    if dropout != 0.0 or scaling != 1.0 or sliding_window != spec.sliding_window:
        raise ValueError("native Gemma 4 attention requires dropout=0, scale=1, and exact window")
    if kwargs.get("output_attentions", False):
        raise UnsupportedH100Path("native FA4 does not return materialized attention weights")
    if kwargs.get("vision_block_ids") is not None:
        raise UnsupportedH100Path("native Axolotl validation is text-only")

    batch_size, _, q_length, _ = query.shape
    kv_length = key.shape[2]
    position_ids = _normalize_position_ids(
        kwargs.get("position_ids"),
        batch_size=batch_size,
        q_length=q_length,
        device=query.device,
    )
    plan = _as_mask_plan(
        attention_mask,
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        spec=spec,
        vision_block_ids=None,
        document_ids=kwargs.get("document_ids"),
    )
    if not _mask_plan_matches_native_contract(
        plan,
        spec,
        position_ids=position_ids,
        vision_block_ids=None,
        document_ids=kwargs.get("document_ids"),
    ):
        raise UnsupportedH100Path("attention mask is not the exact text-only Gemma predicate")

    q_bshd = query.transpose(1, 2)
    k_bshd = key.transpose(1, 2)
    v_bshd = value.transpose(1, 2)
    if not _fa4_layout_is_legal(q_bshd, k_bshd, v_bshd):
        raise UnsupportedH100Path("prepared BHSD-to-BSHD view violates the FA4 layout contract")

    padding = _padding_mask(plan, query.device)
    has_padding = padding is not None and not bool(padding.all().detach().item())
    packed_positions = _positions_are_packed(
        position_ids,
        padding=padding,
        q_length=q_length,
        kv_length=kv_length,
    )
    explicit_cu = kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seq_lens_k") is not None
    use_varlen = (
        has_padding or packed_positions or explicit_cu or batch_size != 1 or q_length != kv_length
    )
    window_size = (None, None) if spec.sliding_window is None else (spec.fa_window_size_left, 0)
    segments = [q_length]
    deterministic = _deterministic_requested(kwargs.get("deterministic"))
    pack_gqa = False if spec.kind == "full_attention" else None

    if not use_varlen:
        output, _lse = _load_flash_attn_func()(
            q_bshd,
            k_bshd,
            v_bshd,
            causal=True,
            window_size=window_size,
            softmax_scale=1.0,
            num_splits=1,
            pack_gqa=pack_gqa,
            deterministic=deterministic,
            return_lse=True,
        )
        suffix = "global_fixed" if spec.kind == "full_attention" else "local_fixed"
    else:
        packed = _pack_local_inputs(
            q_bshd,
            k_bshd,
            v_bshd,
            plan,
            position_ids=position_ids,
            vision_block_ids=None,
            document_ids=kwargs.get("document_ids"),
            cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
            cu_seq_lens_k=kwargs.get("cu_seq_lens_k"),
            max_length_q=kwargs.get("max_length_q"),
            max_length_k=kwargs.get("max_length_k"),
        )
        packed_output, lse = _load_flash_attn_varlen_func()(
            packed.q,
            packed.k,
            packed.v,
            cu_seqlens_q=packed.cu_q,
            cu_seqlens_k=packed.cu_k,
            max_seqlen_q=packed.max_q,
            max_seqlen_k=packed.max_k,
            causal=True,
            window_size=window_size,
            softmax_scale=1.0,
            num_splits=1,
            pack_gqa=pack_gqa,
            deterministic=deterministic,
            return_lse=True,
        )
        output, _lse = _unpack_local_result(packed_output, lse, packed)
        cumulative = [int(item) for item in packed.cu_q.detach().cpu().tolist()]
        segments = [end - start for start, end in zip(cumulative[:-1], cumulative[1:], strict=True)]
        suffix = "global_varlen" if spec.kind == "full_attention" else "local_varlen"

    if output.shape != (batch_size, q_length, query.shape[1], value.shape[-1]):
        raise RuntimeError("public FA4 returned an invalid Gemma output shape")
    if not use_varlen:
        _maybe_check_prepared_oracle(
            module,
            geometry=geometry,
            spec=spec,
            query=query,
            key=key,
            value=value,
            output=output,
        )
    route = f"fa4_native/{suffix}"
    _record_route(
        module,
        route=route,
        geometry=geometry,
        query=query,
        key=key,
        value=value,
        segments=segments,
    )
    return output, None


def _unpack_fa2_output(output: torch.Tensor, packed: Any) -> torch.Tensor:
    batch_size, q_length, q_heads, q_dim = packed.original_q_shape
    if packed.q_flat_indices is None:
        return output.reshape(batch_size, q_length, q_heads, q_dim)
    output_flat = torch.zeros(
        batch_size * q_length,
        q_heads,
        q_dim,
        dtype=output.dtype,
        device=output.device,
    )
    return output_flat.index_copy(0, packed.q_flat_indices, output).view(
        batch_size, q_length, q_heads, q_dim
    )


def _gemma4_fa2_local_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    geometry, spec = _prepared_geometry(module, query, key, value)
    if spec.kind != "sliding_attention":
        raise ValueError("the FA2 local delegate accepts only sliding Gemma layers")
    dropout = float(kwargs.get("dropout", 0.0))
    scaling = float(kwargs.get("scaling", 1.0))
    sliding_window = kwargs.get("sliding_window", spec.sliding_window)
    if dropout != 0.0 or scaling != 1.0 or sliding_window != spec.sliding_window:
        raise ValueError("Gemma 4 local attention requires dropout=0, scale=1, and exact window")
    if kwargs.get("output_attentions", False):
        raise UnsupportedH100Path("the FA2 local delegate does not return attention weights")
    if kwargs.get("vision_block_ids") is not None:
        raise UnsupportedH100Path("native Axolotl validation is text-only")

    batch_size, _, q_length, _ = query.shape
    kv_length = key.shape[2]
    position_ids = _normalize_position_ids(
        kwargs.get("position_ids"),
        batch_size=batch_size,
        q_length=q_length,
        device=query.device,
    )
    plan = _as_mask_plan(
        attention_mask,
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        spec=spec,
        vision_block_ids=None,
        document_ids=kwargs.get("document_ids"),
    )
    if not _mask_plan_matches_native_contract(
        plan,
        spec,
        position_ids=position_ids,
        vision_block_ids=None,
        document_ids=kwargs.get("document_ids"),
    ):
        raise UnsupportedH100Path("attention mask is not the exact text-only Gemma predicate")

    q_bshd = query.transpose(1, 2)
    k_bshd = key.transpose(1, 2)
    v_bshd = value.transpose(1, 2)
    padding = _padding_mask(plan, query.device)
    has_padding = padding is not None and not bool(padding.all().detach().item())
    packed_positions = _positions_are_packed(
        position_ids,
        padding=padding,
        q_length=q_length,
        kv_length=kv_length,
    )
    explicit_cu = kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seq_lens_k") is not None
    use_varlen = (
        has_padding or packed_positions or explicit_cu or batch_size != 1 or q_length != kv_length
    )
    window_size = (-1, -1) if kv_length <= spec.sliding_window else (spec.fa_window_size_left, 0)
    deterministic = _deterministic_requested(kwargs.get("deterministic"))
    segments = [q_length]

    if not use_varlen:
        output = _load_flash_attn2_func()(
            q_bshd,
            k_bshd,
            v_bshd,
            dropout_p=0.0,
            softmax_scale=1.0,
            causal=True,
            window_size=window_size,
            deterministic=deterministic,
        )
        suffix = "fixed"
    else:
        packed = _pack_local_inputs(
            q_bshd,
            k_bshd,
            v_bshd,
            plan,
            position_ids=position_ids,
            vision_block_ids=None,
            document_ids=kwargs.get("document_ids"),
            cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
            cu_seq_lens_k=kwargs.get("cu_seq_lens_k"),
            max_length_q=kwargs.get("max_length_q"),
            max_length_k=kwargs.get("max_length_k"),
        )
        packed_output = _load_flash_attn2_varlen_func()(
            packed.q,
            packed.k,
            packed.v,
            cu_seqlens_q=packed.cu_q,
            cu_seqlens_k=packed.cu_k,
            max_seqlen_q=packed.max_q,
            max_seqlen_k=packed.max_k,
            dropout_p=0.0,
            softmax_scale=1.0,
            causal=True,
            window_size=window_size,
            deterministic=deterministic,
        )
        output = _unpack_fa2_output(packed_output, packed)
        cumulative = [int(item) for item in packed.cu_q.detach().cpu().tolist()]
        segments = [end - start for start, end in zip(cumulative[:-1], cumulative[1:], strict=True)]
        suffix = "varlen"

    if not use_varlen:
        _maybe_check_prepared_oracle(
            module,
            geometry=geometry,
            spec=spec,
            query=query,
            key=key,
            value=value,
            output=output,
        )
    _record_route(
        module,
        route=f"fa2_local/{suffix}",
        geometry=geometry,
        query=query,
        key=key,
        value=value,
        segments=segments,
    )
    return output, None


def gemma4_global_native_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Use FA2 for local d256 and the fork's native FA4 path for global d512."""

    if query.shape[-1] == 512:
        return gemma4_native_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )
    return _gemma4_fa2_local_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )


def _registered_value(registry: Any, key: str) -> Any | None:
    try:
        return registry[key]
    except (KeyError, TypeError):
        return None


def register_gemma4_fa4_h100_native() -> str:
    """Register the native-shape attention route and exact project mask."""

    register_gemma4_fa4_h100()
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:  # pragma: no cover - optional integration dependency
        raise RuntimeError("Transformers is required for native Gemma registration") from exc
    base_mask = _registered_value(ALL_MASK_ATTENTION_FUNCTIONS, BACKEND_NAME)
    if base_mask is None:
        raise RuntimeError("the base Gemma 4 FA4 mask backend was not registered")
    targets = (
        (ALL_ATTENTION_FUNCTIONS, gemma4_native_attention_forward, "attention"),
        (ALL_MASK_ATTENTION_FUNCTIONS, base_mask, "mask"),
    )
    for registry, expected, label in targets:
        existing = _registered_value(registry, BACKEND_NAME_NATIVE)
        if existing is not None and existing is not expected:
            raise RuntimeError(
                f"Transformers {label} backend {BACKEND_NAME_NATIVE!r} is already registered"
            )
    for registry, expected, _label in targets:
        if _registered_value(registry, BACKEND_NAME_NATIVE) is None:
            registry.register(BACKEND_NAME_NATIVE, expected)
    return BACKEND_NAME_NATIVE


def register_gemma4_fa4_h100_global_native() -> str:
    """Register FA2 local attention with native-shape FA4 global attention."""

    register_gemma4_fa4_h100()
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:  # pragma: no cover - optional integration dependency
        raise RuntimeError("Transformers is required for native Gemma registration") from exc
    base_mask = _registered_value(ALL_MASK_ATTENTION_FUNCTIONS, BACKEND_NAME)
    if base_mask is None:
        raise RuntimeError("the base Gemma 4 FA4 mask backend was not registered")
    targets = (
        (ALL_ATTENTION_FUNCTIONS, gemma4_global_native_attention_forward, "attention"),
        (ALL_MASK_ATTENTION_FUNCTIONS, base_mask, "mask"),
    )
    for registry, expected, label in targets:
        existing = _registered_value(registry, BACKEND_NAME_GLOBAL_NATIVE)
        if existing is not None and existing is not expected:
            raise RuntimeError(
                f"Transformers {label} backend {BACKEND_NAME_GLOBAL_NATIVE!r} is already registered"
            )
    for registry, expected, _label in targets:
        if _registered_value(registry, BACKEND_NAME_GLOBAL_NATIVE) is None:
            registry.register(BACKEND_NAME_GLOBAL_NATIVE, expected)
    return BACKEND_NAME_GLOBAL_NATIVE


__all__ = [
    "BACKEND_NAME_GLOBAL_NATIVE",
    "BACKEND_NAME_NATIVE",
    "GEMMA4_NATIVE_GEOMETRIES",
    "NativeGemmaGeometry",
    "gemma4_global_native_attention_forward",
    "gemma4_native_attention_forward",
    "register_gemma4_fa4_h100_global_native",
    "register_gemma4_fa4_h100_native",
]
