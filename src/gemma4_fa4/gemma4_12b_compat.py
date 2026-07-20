"""Explicit Gemma 4 12B compatibility route for the validated 31B H100 kernels.

This module does not claim a native 12B specialization. It preserves 12B GQA
semantics by embedding prepared Q/K/V heads into the 31B kernel geometry and
slicing the first 16 query heads from the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from .model_spec import AttentionLayerSpec, Gemma4ModelSpec
from .transformers_integration import (
    BACKEND_NAME,
    gemma4_fa4_prepared,
    register_gemma4_fa4_h100,
)

BACKEND_NAME_12B_COMPAT = "gemma4_fa4_h100_12b_compat"
GEMMA4_12B_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

SLIDING_ATTENTION_12B = AttentionLayerSpec(
    kind="sliding_attention",
    head_dim_qk=256,
    head_dim_v=256,
    num_q_heads=16,
    num_kv_heads=8,
    softmax_scale=1.0,
    is_causal=True,
    vision_bidirectional_within_block=True,
    sliding_window=1024,
    rope_type="default",
    rope_theta=10_000.0,
    partial_rotary_factor=1.0,
    projection_source_shared_between_k_and_v=False,
)

GLOBAL_ATTENTION_12B = AttentionLayerSpec(
    kind="full_attention",
    head_dim_qk=512,
    head_dim_v=512,
    num_q_heads=16,
    num_kv_heads=1,
    softmax_scale=1.0,
    is_causal=True,
    vision_bidirectional_within_block=False,
    sliding_window=None,
    rope_type="proportional",
    rope_theta=1_000_000.0,
    partial_rotary_factor=0.25,
    projection_source_shared_between_k_and_v=True,
)

GEMMA4_12B_HARNESS = Gemma4ModelSpec(
    model_id="google/gemma-4-12B-it",
    revision=GEMMA4_12B_REVISION,
    hidden_size=3840,
    num_hidden_layers=48,
    max_position_embeddings=262_144,
    num_kv_shared_layers=0,
    use_bidirectional_attention="vision",
    sliding=SLIDING_ATTENTION_12B,
    full=GLOBAL_ATTENTION_12B,
)


@dataclass(frozen=True)
class AdaptedPrepared:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    surrogate_module: Any
    output_heads: int


def _layer_index(module: Any) -> int:
    layer_idx = getattr(module, "layer_idx", None)
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise ValueError("Gemma 4 12B attention module must expose an integer layer_idx")
    if not 0 <= layer_idx < GEMMA4_12B_HARNESS.num_hidden_layers:
        raise ValueError("Gemma 4 12B layer_idx must be in 0..47")
    return layer_idx


def _validate_prepared_geometry(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, AttentionLayerSpec]:
    layer_idx = _layer_index(module)
    spec = GEMMA4_12B_HARNESS.spec_for_layer(layer_idx)
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Gemma 4 12B prepared Q/K/V must be rank-4 BHSD tensors")
    if not (query.device == key.device == value.device):
        raise ValueError("Gemma 4 12B prepared Q/K/V must share one device")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("Gemma 4 12B prepared Q/K/V must share one dtype")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Gemma 4 12B prepared Q/K/V batch sizes must match")
    if key.shape[2] != value.shape[2]:
        raise ValueError("Gemma 4 12B prepared K/V sequence lengths must match")
    if query.shape[1] != 16:
        raise ValueError("Gemma 4 12B compatibility route requires 16 query heads")
    if key.shape[1] != spec.num_kv_heads or value.shape[1] != spec.num_kv_heads:
        raise ValueError(
            f"Gemma 4 12B {spec.kind} compatibility route requires "
            f"{spec.num_kv_heads} KV heads"
        )
    if query.shape[-1] != spec.head_dim_qk or key.shape[-1] != spec.head_dim_qk:
        raise ValueError(f"Gemma 4 12B {spec.kind} Q/K dimension must be {spec.head_dim_qk}")
    if value.shape[-1] != spec.head_dim_v:
        raise ValueError(f"Gemma 4 12B {spec.kind} V dimension must be {spec.head_dim_v}")
    module_head_dim = getattr(module, "head_dim", spec.head_dim_qk)
    if module_head_dim != spec.head_dim_qk:
        raise ValueError("Gemma 4 12B module head_dim conflicts with its layer family")
    module_groups = getattr(module, "num_key_value_groups", spec.qhead_per_kvhead)
    if module_groups != spec.qhead_per_kvhead:
        raise ValueError("Gemma 4 12B module GQA ratio conflicts with its layer family")
    if getattr(module, "scaling", 1.0) != 1.0:
        raise ValueError("Gemma 4 12B attention scaling must be exactly 1.0")
    return layer_idx, spec


def _zero_heads(tensor: torch.Tensor, count: int) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[1] = count
    return tensor.new_zeros(shape)


def adapt_12b_prepared(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> AdaptedPrepared:
    """Map 12B BHSD tensors to the validated 31B prepared-head geometry."""

    layer_idx, spec = _validate_prepared_geometry(module, query, key, value)
    query_32 = torch.cat((query, _zero_heads(query, 16)), dim=1)
    if spec.kind == "sliding_attention":
        key_16 = torch.cat((key, _zero_heads(key, 8)), dim=1)
        value_16 = torch.cat((value, _zero_heads(value, 8)), dim=1)
        target_groups = 2
    else:
        # Q0..7 reads KV0 and Q8..15 reads KV1 in the target GQA-8 layout.
        key_16 = torch.cat((key, key, _zero_heads(key, 2)), dim=1)
        value_16 = torch.cat((value, value, _zero_heads(value, 2)), dim=1)
        target_groups = 8
    surrogate = SimpleNamespace(
        layer_idx=layer_idx,
        layer_type=spec.kind,
        is_sliding=spec.kind == "sliding_attention",
        head_dim=spec.head_dim_qk,
        num_key_value_groups=target_groups,
        scaling=1.0,
        sliding_window=spec.sliding_window,
        training=getattr(module, "training", True),
    )
    return AdaptedPrepared(query_32, key_16, value_16, surrogate, output_heads=16)


def gemma4_12b_compat_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Transformers attention-interface entry point for the 12B compatibility path."""

    adapted = adapt_12b_prepared(module, query, key, value)
    spec = GEMMA4_12B_HARNESS.spec_for_layer(adapted.surrogate_module.layer_idx)
    if kwargs.get("scaling", 1.0) != 1.0:
        raise ValueError("Gemma 4 12B attention scaling must be exactly 1.0")
    if kwargs.get("dropout", 0.0) != 0.0:
        raise ValueError("Gemma 4 12B compatibility attention requires zero dropout")
    if kwargs.get("sliding_window", spec.sliding_window) != spec.sliding_window:
        raise ValueError("Gemma 4 12B sliding-window argument conflicts with the layer family")
    kwargs["scaling"] = 1.0
    kwargs["dropout"] = 0.0
    kwargs["sliding_window"] = spec.sliding_window
    kwargs["allow_flex_fallback"] = False
    result = gemma4_fa4_prepared(
        adapted.surrogate_module,
        adapted.query,
        adapted.key,
        adapted.value,
        attention_mask,
        **kwargs,
    )
    output = result.output[:, :, : adapted.output_heads, :]
    route = f"fa4_12b_compat/{result.path}"
    module._gemma4_fa4_last_path = route
    counts = getattr(module, "_gemma4_fa4_route_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        module._gemma4_fa4_route_counts = counts
    counts[route] = int(counts.get(route, 0)) + 1
    return output, None


def _registered_value(registry: Any, key: str) -> Any | None:
    try:
        return registry[key]
    except (KeyError, TypeError):
        return None


def register_gemma4_fa4_h100_12b_compat() -> str:
    """Register the separately named 12B adapter and the exact project mask."""

    register_gemma4_fa4_h100()
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:  # pragma: no cover - optional integration dependency
        raise RuntimeError("Transformers is required for 12B compatibility registration") from exc

    base_mask = _registered_value(ALL_MASK_ATTENTION_FUNCTIONS, BACKEND_NAME)
    if base_mask is None:
        raise RuntimeError("the base Gemma 4 FA4 mask backend was not registered")
    targets = (
        (ALL_ATTENTION_FUNCTIONS, gemma4_12b_compat_attention_forward, "attention"),
        (ALL_MASK_ATTENTION_FUNCTIONS, base_mask, "mask"),
    )
    for registry, expected, label in targets:
        existing = _registered_value(registry, BACKEND_NAME_12B_COMPAT)
        if existing is not None and existing is not expected:
            raise RuntimeError(
                f"Transformers {label} backend {BACKEND_NAME_12B_COMPAT!r} is already registered"
            )
    for registry, expected, _label in targets:
        if _registered_value(registry, BACKEND_NAME_12B_COMPAT) is None:
            registry.register(BACKEND_NAME_12B_COMPAT, expected)
    return BACKEND_NAME_12B_COMPAT


__all__ = [
    "BACKEND_NAME_12B_COMPAT",
    "GEMMA4_12B_HARNESS",
    "GEMMA4_12B_REVISION",
    "GLOBAL_ATTENTION_12B",
    "SLIDING_ATTENTION_12B",
    "AdaptedPrepared",
    "adapt_12b_prepared",
    "gemma4_12b_compat_attention_forward",
    "register_gemma4_fa4_h100_12b_compat",
]
