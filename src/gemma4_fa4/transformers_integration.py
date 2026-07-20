"""Pinned Transformers integration for the Gemma 4 H100 FA4 paths.

The pinned Gemma implementation presents prepared Q/K/V as ``(B, H, S, D)``.
This module owns the model-specific conversion, packed-sequence construction,
and exact fallback policy.  It deliberately registers under a unique backend
name instead of changing Transformers' generic ``flash_attention_4`` entry.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import CodeType
from typing import Any

import torch

from .h100 import (
    GlobalBackwardBudgetExceeded,
    UnsupportedH100Path,
    _has_proven_nonoverlap,
    fa4_global_forward_only,
    fa4_global_text_forward,
    fa4_global_varlen_forward,
    fa4_global_varlen_forward_only,
    fa4_local_forward,
    fa4_local_varlen_forward,
)
from .h100_torch_ops import (
    CUSTOM_OPS_AVAILABLE,
    h100_global_fwd,
    h100_global_layer_fwd,
    h100_global_static_cache_decode_fwd,
    h100_local_fwd,
    h100_local_layer_fwd,
    h100_local_static_cache_decode_fwd,
)
from .model_spec import GEMMA4_31B, AttentionLayerSpec

BACKEND_NAME = "gemma4_fa4_h100"
PINNED_TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
_FLEX_H100_D512_KERNEL_OPTIONS = {
    "fwd_BLOCK_M": 32,
    "fwd_BLOCK_N": 16,
    "fwd_num_warps": 4,
    "fwd_num_stages": 1,
    "bwd_BLOCK_M1": 16,
    "bwd_BLOCK_N1": 16,
    "bwd_BLOCK_M2": 16,
    "bwd_BLOCK_N2": 16,
    "bwd_num_warps": 4,
    "bwd_num_stages": 1,
}
_GLOBAL_COMPOSED_BACKWARD_MAX_SEQLEN = 2048
_PINNED_LAYER_TYPES = GEMMA4_31B.layer_types()
_REGISTERED_COMPILE_MASK_CALLBACK = object()
_COMPILE_LOCAL_MASK_ORIGIN = object()
_COMPILE_GLOBAL_MASK_ORIGIN = object()
_PINNED_GEMMA4_TEXT_CONFIG_CLASS: type | None = None
_PINNED_GEMMA4_TEXT_ATTENTION_CLASS: type | None = None
_PINNED_STATIC_CACHE_CLASS: type | None = None
_PINNED_STATIC_LAYER_CLASS: type | None = None
_PINNED_STATIC_SLIDING_WINDOW_LAYER_CLASS: type | None = None
_PINNED_MASKING_UTILS_MODULE: Any | None = None
_TORCH_IS_COMPILING = getattr(
    getattr(torch, "compiler", None),
    "is_compiling",
    lambda: False,
)


@dataclass(frozen=True)
class Gemma4MaskPlan:
    """The exact mask information retained by the companion HF mask adapter."""

    batch_size: int
    q_length: int
    kv_length: int
    q_offset: int | torch.Tensor
    kv_offset: int | torch.Tensor
    mask_function: Callable | None
    attention_mask: torch.Tensor | None


@dataclass(frozen=True)
class Gemma4DispatchResult:
    """Prepared-attention result with an observable implementation path."""

    output: torch.Tensor
    lse: torch.Tensor | None
    path: str


@dataclass(frozen=True)
class _PackedLocalInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    q_flat_indices: torch.Tensor | None
    original_q_shape: tuple[int, int, int, int]
    vision_block_ids: torch.Tensor | None
    document_ids: torch.Tensor | None


_INTERNAL_MASK_PLAN = object()


def _mark_internal_plan(plan: Gemma4MaskPlan) -> Gemma4MaskPlan:
    object.__setattr__(plan, "_gemma4_fa4_mask_origin", _INTERNAL_MASK_PLAN)
    return plan


def gemma4_fa4_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int | torch.Tensor = 0,
    kv_offset: int | torch.Tensor = 0,
    mask_function: Callable | None = None,
    attention_mask: torch.Tensor | None = None,
    config: Any | None = None,
    use_vmap: bool | None = None,
    local_size: int | None = None,
    past_key_values: Any | None = None,
    _registered_callback: object | None = None,
    _upstream_plain_origin: object | None = None,
    _upstream_mask_recipient: Callable | None = None,
    **_kwargs,
) -> Gemma4MaskPlan:
    """Preserve the composed pinned-Transformers mask for FA4 or Flex routing."""

    plan = Gemma4MaskPlan(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=_snapshot_real_offset(q_offset),
        kv_offset=_snapshot_real_offset(kv_offset),
        mask_function=mask_function,
        attention_mask=attention_mask,
    )
    compile_mask = _pinned_compile_mask_origin(
        registered_callback=_registered_callback,
        config=config,
        attention_mask=attention_mask,
        mask_function=mask_function,
        use_vmap=use_vmap,
        local_size=local_size,
        past_key_values=past_key_values,
        upstream_plain_origin=_upstream_plain_origin,
        upstream_mask_recipient=_upstream_mask_recipient,
    )
    if compile_mask is not None:
        object.__setattr__(plan, "_gemma4_fa4_compile_origin", compile_mask)
        object.__setattr__(plan, "_gemma4_fa4_compile_config", config)
    return plan


def _registered_gemma4_fa4_mask(
    *,
    past_key_values: Any | None = None,
    _gemma4_fa4_plain_mask_origin: object | None = None,
    _gemma4_fa4_mask_recipient: Callable | None = None,
    **kwargs: Any,
) -> Gemma4MaskPlan:
    """Registry-only entry point that supplies the compiler-origin capability."""

    if bool(_TORCH_IS_COMPILING()) and past_key_values is not None:
        raise UnsupportedH100Path(
            "EXP-0018 fullgraph routing does not accept a cache; "
            "rejection occurs at mask construction before Cache.update"
        )
    return gemma4_fa4_mask(
        past_key_values=past_key_values,
        _registered_callback=_REGISTERED_COMPILE_MASK_CALLBACK,
        _upstream_plain_origin=_gemma4_fa4_plain_mask_origin,
        _upstream_mask_recipient=_gemma4_fa4_mask_recipient,
        **kwargs,
    )


# Preserve the diagnostic name used by the existing H100 registration probe.
_registered_gemma4_fa4_mask.__name__ = "gemma4_fa4_mask"


def _callable_closure_values(function: Callable) -> tuple[Any, ...]:
    return tuple(cell.cell_contents for cell in (function.__closure__ or ()))


def _nested_code_objects(code: CodeType) -> tuple[CodeType, ...]:
    nested: list[CodeType] = []
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            nested.append(constant)
            nested.extend(_nested_code_objects(constant))
    return tuple(nested)


def _is_pinned_mask_callable(function: Callable, qualname: str) -> bool:
    """Match executable code/globals, not forgeable module/name metadata alone."""

    masking_utils = _PINNED_MASKING_UTILS_MODULE
    if masking_utils is None:
        try:
            import transformers.masking_utils as masking_utils
        except Exception:
            return False
    if getattr(function, "__globals__", None) is not vars(masking_utils):
        return False
    if qualname == "causal_mask_function":
        return function is masking_utils.causal_mask_function
    factories = {
        "sliding_window_overlay.<locals>.inner_mask": ("sliding_window_overlay", "inner_mask"),
        "blockwise_overlay.<locals>.inner_mask": ("blockwise_overlay", "inner_mask"),
        "packed_sequence_mask_function.<locals>.inner_mask": (
            "packed_sequence_mask_function",
            "inner_mask",
        ),
        "and_masks.<locals>.and_mask": ("and_masks", "and_mask"),
        "or_masks.<locals>.or_mask": ("or_masks", "or_mask"),
    }
    factory_spec = factories.get(qualname)
    if factory_spec is None:
        return False
    factory_name, code_name = factory_spec
    factory = getattr(masking_utils, factory_name, None)
    function_code = getattr(function, "__code__", None)
    return callable(factory) and any(
        candidate is function_code and candidate.co_name == code_name
        for candidate in _nested_code_objects(factory.__code__)
    )


def _is_pinned_gemma4_text_config(config: Any) -> bool:
    """Recognize the exact registered config class and locked 31B contract."""

    rope_parameters = getattr(config, "rope_parameters", None)
    if type(rope_parameters) is not dict or len(rope_parameters) != 2:
        return False
    sliding_rope = rope_parameters.get("sliding_attention")
    full_rope = rope_parameters.get("full_attention")
    if (
        type(sliding_rope) is not dict
        or len(sliding_rope) != 2
        or sliding_rope.get("rope_type") != "default"
        or sliding_rope.get("rope_theta") != 10_000.0
        or type(full_rope) is not dict
        or len(full_rope) != 3
        or full_rope.get("rope_type") != "proportional"
        or full_rope.get("partial_rotary_factor") != 0.25
        or full_rope.get("rope_theta") != 1_000_000.0
    ):
        return False
    if (
        _PINNED_GEMMA4_TEXT_CONFIG_CLASS is None
        or type(config) is not _PINNED_GEMMA4_TEXT_CONFIG_CLASS
    ):
        return False
    if (
        getattr(config, "_attn_implementation", None) != BACKEND_NAME
        or getattr(config, "hidden_size", None) != GEMMA4_31B.hidden_size
        or getattr(config, "intermediate_size", None) != 21_504
        or getattr(config, "num_hidden_layers", None) != GEMMA4_31B.num_hidden_layers
        or getattr(config, "num_attention_heads", None) != GEMMA4_31B.sliding.num_q_heads
        or getattr(config, "num_key_value_heads", None) != GEMMA4_31B.sliding.num_kv_heads
        or getattr(config, "num_global_key_value_heads", None) != GEMMA4_31B.full.num_kv_heads
        or getattr(config, "head_dim", None) != GEMMA4_31B.sliding.head_dim_qk
        or getattr(config, "global_head_dim", None) != GEMMA4_31B.full.head_dim_qk
        or getattr(config, "sliding_window", None) != GEMMA4_31B.sliding.sliding_window
        or getattr(config, "max_position_embeddings", None) != GEMMA4_31B.max_position_embeddings
        or getattr(config, "attention_dropout", None) != 0.0
        or getattr(config, "attention_bias", None) is not False
        or getattr(config, "attention_k_eq_v", None) is not True
        or getattr(config, "num_kv_shared_layers", None) != GEMMA4_31B.num_kv_shared_layers
        or getattr(config, "use_bidirectional_attention", None)
        != GEMMA4_31B.use_bidirectional_attention
        or getattr(config, "hidden_size_per_layer_input", None) != 0
        or getattr(config, "final_logit_softcapping", None) != 30.0
        or getattr(config, "rms_norm_eps", None) != 1e-6
        or getattr(config, "is_causal", True) is not True
    ):
        return False
    layer_types = getattr(config, "layer_types", None)
    if type(layer_types) is not list or len(layer_types) != len(_PINNED_LAYER_TYPES):
        return False
    observed_layer_types = (
        layer_types[0],
        layer_types[1],
        layer_types[2],
        layer_types[3],
        layer_types[4],
        layer_types[5],
        layer_types[6],
        layer_types[7],
        layer_types[8],
        layer_types[9],
        layer_types[10],
        layer_types[11],
        layer_types[12],
        layer_types[13],
        layer_types[14],
        layer_types[15],
        layer_types[16],
        layer_types[17],
        layer_types[18],
        layer_types[19],
        layer_types[20],
        layer_types[21],
        layer_types[22],
        layer_types[23],
        layer_types[24],
        layer_types[25],
        layer_types[26],
        layer_types[27],
        layer_types[28],
        layer_types[29],
        layer_types[30],
        layer_types[31],
        layer_types[32],
        layer_types[33],
        layer_types[34],
        layer_types[35],
        layer_types[36],
        layer_types[37],
        layer_types[38],
        layer_types[39],
        layer_types[40],
        layer_types[41],
        layer_types[42],
        layer_types[43],
        layer_types[44],
        layer_types[45],
        layer_types[46],
        layer_types[47],
        layer_types[48],
        layer_types[49],
        layer_types[50],
        layer_types[51],
        layer_types[52],
        layer_types[53],
        layer_types[54],
        layer_types[55],
        layer_types[56],
        layer_types[57],
        layer_types[58],
        layer_types[59],
    )
    return observed_layer_types == _PINNED_LAYER_TYPES


def _pinned_compile_mask_origin(
    *,
    registered_callback: object | None,
    config: Any,
    attention_mask: torch.Tensor | None,
    mask_function: Callable | None,
    use_vmap: bool | None,
    local_size: int | None,
    past_key_values: Any | None,
    upstream_plain_origin: object | None,
    upstream_mask_recipient: Callable | None,
) -> object | None:
    """Stamp only EXP-0018's exact upstream plain no-cache mask families."""

    if (
        registered_callback is not _REGISTERED_COMPILE_MASK_CALLBACK
        or not _is_pinned_gemma4_text_config(config)
        or past_key_values is not None
        or upstream_mask_recipient is not _registered_gemma4_fa4_mask
        or attention_mask is not None
        or use_vmap is not False
        or mask_function is None
    ):
        return None

    expected_origin: object | None = None
    pinned_masking = _PINNED_MASKING_UTILS_MODULE
    if pinned_masking is None:
        return None
    if local_size is None and upstream_plain_origin is getattr(
        pinned_masking, "_GEMMA4_FA4_PLAIN_CAUSAL_MASK_ORIGIN", None
    ):
        expected_origin = _COMPILE_GLOBAL_MASK_ORIGIN
    elif (
        type(local_size) is int
        and local_size == GEMMA4_31B.sliding.sliding_window
        and upstream_plain_origin
        is getattr(pinned_masking, "_GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN", None)
    ):
        expected_origin = _COMPILE_LOCAL_MASK_ORIGIN
    if expected_origin is None:
        return None

    # Dynamo represents the freshly constructed pinned mask closure as a
    # NestedUserFunctionVariable and deliberately exposes neither __module__
    # nor __qualname__.  The identity-bound upstream origin and recipient,
    # registry-only capability, exact pinned config, no-cache state,
    # use_vmap=False, no explicit mask, layer family, and attention-side
    # runtime guards are therefore the compiler proof. Non-Dynamo FakeTensor
    # calls still receive ordinary Python functions and retain the stronger
    # executable-code/closure proof below.
    if bool(_TORCH_IS_COMPILING()):
        return expected_origin

    parsed = _parse_pinned_mask_function(mask_function)
    if parsed is None:
        return None
    expression, metadata = parsed
    if metadata["vision"] or len(metadata["packed"]) > 1:
        return None
    expression = _normalize_boolean_expression(expression)
    packed_suffix = (("packed",),) if metadata["packed"] else ()
    expected_global = _and_expression(("causal",), *packed_suffix) if packed_suffix else ("causal",)
    if expected_origin is _COMPILE_GLOBAL_MASK_ORIGIN and expression == expected_global:
        return expected_origin
    expected_local = _and_expression(
        ("causal",),
        ("sliding", GEMMA4_31B.sliding.sliding_window),
        *packed_suffix,
    )
    if expected_origin is _COMPILE_LOCAL_MASK_ORIGIN and expression == expected_local:
        return expected_origin
    return None


def _parse_pinned_mask_function(
    function: Callable,
) -> tuple[tuple[Any, ...], dict[str, list[torch.Tensor]]] | None:
    """Parse only the boolean combinators shipped by the pinned Transformers SHA."""

    module = getattr(function, "__module__", "")
    name = getattr(function, "__qualname__", "")
    if module != "transformers.masking_utils" or not _is_pinned_mask_callable(function, name):
        return None
    values = _callable_closure_values(function)
    metadata: dict[str, list[torch.Tensor]] = {"vision": [], "packed": []}
    if name == "causal_mask_function":
        return ("causal",), metadata
    if name == "sliding_window_overlay.<locals>.inner_mask":
        windows = [value for value in values if isinstance(value, int)]
        if len(windows) != 1:
            return None
        return ("sliding", windows[0]), metadata
    if name in {
        "blockwise_overlay.<locals>.inner_mask",
        "packed_sequence_mask_function.<locals>.inner_mask",
    }:
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if len(tensors) != 1:
            return None
        kind = "vision" if name.startswith("blockwise_overlay") else "packed"
        metadata[kind].append(tensors[0])
        return (kind,), metadata
    combiners = {
        "and_masks.<locals>.and_mask": "and",
        "or_masks.<locals>.or_mask": "or",
    }
    if name not in combiners:
        return None
    children = next(
        (
            value
            for value in values
            if isinstance(value, (tuple, list)) and all(callable(item) for item in value)
        ),
        None,
    )
    if children is None or len(children) < 2:
        return None
    parsed_children: list[tuple[Any, ...]] = []
    for child in children:
        parsed = _parse_pinned_mask_function(child)
        if parsed is None:
            return None
        expression, child_metadata = parsed
        parsed_children.append(expression)
        for kind in metadata:
            metadata[kind].extend(child_metadata[kind])
    return (combiners[name], *parsed_children), metadata


def _normalize_boolean_expression(expression: tuple[Any, ...]) -> tuple[Any, ...]:
    operator = expression[0]
    if operator not in {"and", "or"}:
        return expression
    children: list[tuple[Any, ...]] = []
    for raw_child in expression[1:]:
        child = _normalize_boolean_expression(raw_child)
        if child[0] == operator:
            children.extend(child[1:])
        else:
            children.append(child)
    return (operator, *sorted(children, key=repr))


def _and_expression(*children: tuple[Any, ...]) -> tuple[Any, ...]:
    return _normalize_boolean_expression(("and", *children))


def _packed_sequence_ids(position_ids: torch.Tensor) -> torch.Tensor | None:
    first = position_ids[:, :1] - 1
    groups = (torch.diff(position_ids, prepend=first, dim=-1) != 1).cumsum(-1)
    if not _is_fake_tensor(groups) and bool((groups[:, -1] == 0).all().item()):
        return None
    return groups


def _metadata_slice(
    values: torch.Tensor,
    *,
    batch_size: int,
    kv_length: int,
    kv_offset: int,
) -> torch.Tensor | None:
    if values.ndim != 2 or values.shape[0] not in (1, batch_size):
        return None
    if values.shape[0] == 1 and batch_size != 1:
        values = values.expand(batch_size, -1)
    end = kv_offset + kv_length
    if values.shape[1] < end:
        return None
    return values[:, kv_offset:end]


def _mask_plan_matches_native_contract(
    plan: Gemma4MaskPlan,
    spec: AttentionLayerSpec,
    *,
    position_ids: torch.Tensor | None,
    vision_block_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
) -> bool:
    """Prove that a pinned callable contains exactly the native FA4 predicate."""

    if plan.attention_mask is not None and plan.attention_mask.ndim != 2:
        return False
    if getattr(plan, "_gemma4_fa4_mask_origin", None) is _INTERNAL_MASK_PLAN:
        return True
    if plan.mask_function is None:
        return True
    parsed = _parse_pinned_mask_function(plan.mask_function)
    if parsed is None:
        return False
    expression, metadata = parsed
    expression = _normalize_boolean_expression(expression)
    causal = ("causal",)
    packed = ("packed",)
    expected_packed = None
    if position_ids is not None and plan.q_length == plan.kv_length:
        expected_packed = _packed_sequence_ids(position_ids)
    if document_ids is not None and plan.q_length == plan.kv_length:
        document_boundaries = torch.nn.functional.pad(
            torch.diff(document_ids, dim=-1) != 0,
            (1, 0),
            value=False,
        )
        document_groups = document_boundaries.to(torch.int64).cumsum(-1)
        if not _is_fake_tensor(document_groups) and bool(
            (document_groups[:, -1] == 0).all().item()
        ):
            document_groups = None
        if expected_packed is None:
            expected_packed = document_groups
        elif document_groups is not None:
            expected_packed = (
                torch.nn.functional.pad(
                    (torch.diff(position_ids, dim=-1) != 1)
                    | (torch.diff(document_ids, dim=-1) != 0),
                    (1, 0),
                    value=False,
                )
                .to(torch.int64)
                .cumsum(-1)
            )

    if spec.kind == "full_attention":
        expected = causal if expected_packed is None else _and_expression(causal, packed)
        if metadata["vision"]:
            return False
    else:
        sliding = ("sliding", spec.sliding_window)
        active_vision = bool(
            vision_block_ids is not None
            and not _is_fake_tensor(vision_block_ids)
            and (vision_block_ids >= 0).any().item()
        )
        if metadata["vision"]:
            if len(metadata["vision"]) != 1 or vision_block_ids is None:
                return False
            try:
                kv_offset = _python_int(plan.kv_offset, name="kv_offset")
            except TypeError:
                return False
            encoded_vision = _metadata_slice(
                metadata["vision"][0],
                batch_size=plan.batch_size,
                kv_length=plan.kv_length,
                kv_offset=kv_offset,
            )
            if encoded_vision is None or not torch.equal(
                encoded_vision.to(
                    device=vision_block_ids.device,
                    dtype=torch.int64,
                ),
                vision_block_ids.to(torch.int64),
            ):
                return False
            base = _and_expression(("or", causal, ("vision",)), sliding)
        elif active_vision:
            return False
        else:
            base = _and_expression(causal, sliding)
        expected = base if expected_packed is None else _and_expression(base, packed)

    if expected_packed is None:
        if metadata["packed"]:
            return False
    else:
        if len(metadata["packed"]) != 1:
            return False
        encoded = metadata["packed"][0]
        if encoded.shape[0] == 1 and plan.batch_size != 1:
            encoded = encoded.expand(plan.batch_size, -1)
        if encoded.shape != expected_packed.shape or not torch.equal(
            encoded.to(device=expected_packed.device, dtype=torch.int64),
            expected_packed.to(torch.int64),
        ):
            return False
    return expression == expected


def _python_int(value: int | torch.Tensor, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1 and not value.is_meta:
        if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{name} scalar tensor must use an integer, non-bool dtype")
        return int(value.detach().item())
    raise TypeError(f"{name} must be a Python integer or scalar integer tensor")


def _spec_for_module(module: Any) -> AttentionLayerSpec:
    layer_idx = getattr(module, "layer_idx", None)
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise ValueError("Gemma 4 attention module must expose an integer layer_idx")
    try:
        return GEMMA4_31B.spec_for_layer(layer_idx)
    except IndexError as exc:
        raise IndexError(f"Gemma 4 layer index {layer_idx} is outside 0..59") from exc


def _is_fake_tensor(tensor: torch.Tensor) -> bool:
    try:
        from torch._subclasses.fake_tensor import FakeTensor
    except ImportError:  # pragma: no cover - pinned PyTorch provides it
        return False
    return isinstance(tensor, FakeTensor)


def _snapshot_real_offset(value: int | torch.Tensor) -> int | torch.Tensor:
    """Detach a real cache counter from later in-place StaticCache updates."""

    if isinstance(value, torch.Tensor) and not _is_fake_tensor(value):
        return value.detach().clone()
    return value


def _validate_prepared_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
) -> tuple[int, int, int]:
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Transformers prepared q, k, and v must be rank-4 BHSD tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("prepared q, k, and v must share one device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("the Gemma 4 H100 integration accepts BF16 q, k, and v only")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("prepared q, k, and v must have unit last-dimension stride")
    if any(stride <= 0 for t in (q, k, v) for stride in t.stride()):
        raise ValueError("prepared q, k, and v must use positive, non-broadcast layouts")

    batch, q_heads, q_length, q_dim = q.shape
    k_batch, kv_heads, kv_length, k_dim = k.shape
    if batch <= 0 or q_length <= 0 or kv_length <= 0:
        raise ValueError("batch and sequence lengths must be positive")
    if k_batch != batch or v.shape[0] != batch:
        raise ValueError("prepared q, k, and v batch dimensions must match")
    if (q_heads, q_dim) != (spec.num_q_heads, spec.head_dim_qk):
        raise ValueError("prepared q shape does not match the locked layer geometry")
    if (kv_heads, k_dim) != (spec.num_kv_heads, spec.head_dim_qk):
        raise ValueError("prepared k shape does not match the locked layer geometry")
    if v.shape != (batch, spec.num_kv_heads, kv_length, spec.head_dim_v):
        raise ValueError("prepared v shape does not match the locked layer geometry")

    fake_inputs = [_is_fake_tensor(t) for t in (q, k, v)]
    if any(fake_inputs) and not all(fake_inputs):
        raise ValueError("prepared q, k, and v must all be real or all be fake tensors")
    if not all(fake_inputs):
        if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
            raise ValueError("prepared K and V must be distinct, non-aliasing operands")
    return batch, q_length, kv_length


def _validate_semantics(
    module: Any,
    spec: AttentionLayerSpec,
    *,
    dropout: float,
    scaling: float,
    sliding_window: int | None,
    output_attentions: bool,
) -> None:
    if scaling != 1.0:
        raise ValueError("Gemma 4 attention scaling must be exactly 1.0")
    if dropout != 0.0:
        raise ValueError("Gemma 4 H100 attention requires zero dropout")
    if output_attentions:
        raise UnsupportedH100Path(
            "output_attentions/attention weights are not exposed by the H100 FA4 integration"
        )
    if sliding_window != spec.sliding_window:
        raise ValueError(
            f"layer {module.layer_idx} requires sliding_window={spec.sliding_window}, "
            f"got {sliding_window}"
        )
    module_scaling = getattr(module, "scaling", 1.0)
    if module_scaling != 1.0:
        raise ValueError("the pinned Gemma attention module must have scaling=1.0")
    if getattr(module, "head_dim", spec.head_dim_qk) != spec.head_dim_qk:
        raise ValueError("module head_dim conflicts with the locked layer index")
    if getattr(module, "num_key_value_groups", spec.qhead_per_kvhead) != spec.qhead_per_kvhead:
        raise ValueError("module GQA ratio conflicts with the locked layer index")


def _is_compiler_or_fake(*tensors: torch.Tensor) -> bool:
    return bool(_TORCH_IS_COMPILING()) or any(_is_fake_tensor(tensor) for tensor in tensors)


def _compiler_origin_for_spec(spec: AttentionLayerSpec) -> object:
    return (
        _COMPILE_GLOBAL_MASK_ORIGIN if spec.kind == "full_attention" else _COMPILE_LOCAL_MASK_ORIGIN
    )


def _run_compiler_fixed_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: Gemma4MaskPlan,
    spec: AttentionLayerSpec,
    *,
    position_ids: torch.Tensor | None,
    vision_block_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | torch.Tensor | None,
    max_length_k: int | torch.Tensor | None,
    extra_kwargs: dict[str, Any],
) -> Gemma4DispatchResult:
    """Admit only EXP-0017's opaque no-cache fixed-forward compiler ABI."""

    expected_layer_idx = 5 if spec.kind == "full_attention" else 0
    module_config = getattr(module, "config", None)
    if (
        _PINNED_GEMMA4_TEXT_ATTENTION_CLASS is None
        or type(module) is not _PINNED_GEMMA4_TEXT_ATTENTION_CLASS
        or getattr(module, "layer_idx", None) != expected_layer_idx
        or getattr(module, "training", None) is not False
        or not _is_pinned_gemma4_text_config(module_config)
        or getattr(plan, "_gemma4_fa4_compile_config", None) is not module_config
    ):
        raise UnsupportedH100Path(
            "EXP-0017 compiler routing requires the exact pinned eval-mode "
            f"Gemma4TextAttention layer {expected_layer_idx} and its own mask config"
        )
    if not isinstance(plan, Gemma4MaskPlan) or (
        getattr(plan, "_gemma4_fa4_compile_origin", None) is not _compiler_origin_for_spec(spec)
    ):
        raise UnsupportedH100Path(
            "Transformers FakeTensor/torch.compile requires the pinned Gemma mask origin "
            "for the selected layer family"
        )
    if plan.attention_mask is not None:
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile does not accept padding or explicit masks"
        )
    if type(plan.q_offset) is not int or plan.q_offset != 0:
        raise UnsupportedH100Path("EXP-0017 FakeTensor/torch.compile does not accept Q offsets")
    if type(plan.kv_offset) is not int or plan.kv_offset != 0:
        raise UnsupportedH100Path("EXP-0017 FakeTensor/torch.compile does not accept K/V offsets")
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile requires rank-4 prepared BHSD Q/K/V"
        )
    if not (query.device == key.device == value.device) or query.device.type != "cuda":
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile requires Q/K/V on one CUDA device"
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise UnsupportedH100Path("EXP-0017 FakeTensor/torch.compile requires BF16 prepared Q/K/V")
    if not bool(_TORCH_IS_COMPILING()):
        fake_inputs = tuple(_is_fake_tensor(tensor) for tensor in (query, key, value))
        if any(fake_inputs) and not all(fake_inputs):
            raise UnsupportedH100Path(
                "EXP-0017 FakeTensor/torch.compile requires Q/K/V to share one tensor mode"
            )
    if any(tensor.requires_grad for tensor in (query, key, value)):
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile is forward-only and rejects requires_grad inputs"
        )

    batch_size, q_heads, q_length, q_dim = query.shape
    k_batch, kv_heads, kv_length, k_dim = key.shape
    if batch_size != 1 or k_batch != 1 or value.shape[0] != 1:
        raise UnsupportedH100Path("EXP-0017 FakeTensor/torch.compile accepts B1 only")
    if q_length != kv_length or value.shape[2] != kv_length:
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile requires equal Q/K/V sequence lengths"
        )
    if q_length < 1 or q_length > 1024:
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile requires a sequence length in 1..1024"
        )
    if (q_heads, q_dim) != (spec.num_q_heads, spec.head_dim_qk):
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile Q geometry conflicts with the locked layer"
        )
    if (kv_heads, k_dim) != (spec.num_kv_heads, spec.head_dim_qk) or value.shape[1:] != (
        spec.num_kv_heads,
        kv_length,
        spec.head_dim_v,
    ):
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile K/V geometry conflicts with the locked layer"
        )
    if plan.batch_size != batch_size or plan.q_length != q_length or plan.kv_length != kv_length:
        raise UnsupportedH100Path("EXP-0017 pinned mask dimensions do not match prepared Q/K/V")
    if position_ids is None or not isinstance(position_ids, torch.Tensor):
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile requires explicit position_ids"
        )
    if (
        position_ids.ndim != 2
        or position_ids.shape != (1, q_length)
        or position_ids.device != query.device
        or position_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise UnsupportedH100Path(
            "EXP-0017 position_ids must be CUDA INT32/INT64 with shape (1, S)"
        )
    first_position = position_ids[:, :1] - 1
    packed_sequence_ids = (torch.diff(position_ids, prepend=first_position, dim=-1) != 1).cumsum(-1)
    if vision_block_ids is not None or document_ids is not None:
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile accepts text-only requests without metadata"
        )
    if any(
        value is not None
        for value in (
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
        )
    ):
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile does not accept packed sequence metadata"
        )
    if any(
        extra_kwargs.get(name) is not None
        for name in ("past_key_values", "cache", "cache_position")
    ):
        raise UnsupportedH100Path("EXP-0017 FakeTensor/torch.compile does not accept a cache")

    if not CUSTOM_OPS_AVAILABLE:
        raise UnsupportedH100Path(
            "EXP-0017 FakeTensor/torch.compile requires PyTorch custom_op and register_fake APIs"
        )
    if spec.kind == "full_attention":
        output, lse = h100_global_fwd(
            query,
            key,
            value,
            position_ids,
            packed_sequence_ids,
        )
        path = "fa4_global_compiled_op"
    else:
        output, lse = h100_local_fwd(
            query,
            key,
            value,
            position_ids,
            packed_sequence_ids,
        )
        path = "fa4_local_compiled_op"
    return Gemma4DispatchResult(output=output, lse=lse, path=path)


def _fa4_layout_is_legal(*tensors: torch.Tensor) -> bool:
    """Host-side form of the pinned CuTe 128-bit outer-stride contract."""

    for tensor in tensors:
        if tensor.stride(-1) != 1:
            return False
        if any(stride <= 0 or stride % 8 for stride in tensor.stride()[:-1]):
            return False
        if not _has_proven_nonoverlap(tensor):
            return False
        if not _is_fake_tensor(tensor) and tensor.data_ptr() % 16:
            return False
    return True


def _as_mask_plan(
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    *,
    batch_size: int,
    q_length: int,
    kv_length: int,
    spec: AttentionLayerSpec,
    vision_block_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
) -> Gemma4MaskPlan:
    if isinstance(attention_mask, Gemma4MaskPlan):
        if (
            attention_mask.batch_size != batch_size
            or attention_mask.q_length != q_length
            or attention_mask.kv_length != kv_length
        ):
            raise ValueError("Gemma4MaskPlan dimensions do not match prepared q/k/v")
        return attention_mask
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a Gemma4MaskPlan, tensor, or None")

    q_offset = kv_length - q_length

    def default_mask(batch_idx, _head_idx, q_idx, kv_idx):
        # ``flex_attention_mask`` applies q/kv offsets before invoking this
        # absolute-position predicate.
        q_absolute = q_idx
        allowed = kv_idx <= q_absolute
        if document_ids is not None:
            allowed = allowed & (
                document_ids[batch_idx, q_absolute] == document_ids[batch_idx, kv_idx]
            )
        if spec.sliding_window is not None:
            allowed = allowed & (kv_idx > q_absolute - spec.sliding_window)
            if vision_block_ids is not None:
                q_vision = vision_block_ids[batch_idx, q_absolute]
                k_vision = vision_block_ids[batch_idx, kv_idx]
                same_vision = (q_vision >= 0) & (q_vision == k_vision)
                if document_ids is not None:
                    same_vision = same_vision & (
                        document_ids[batch_idx, q_absolute] == document_ids[batch_idx, kv_idx]
                    )
                allowed = (kv_idx > q_absolute - spec.sliding_window) & (allowed | same_vision)
        return allowed

    return _mark_internal_plan(
        Gemma4MaskPlan(
            batch_size=batch_size,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=0,
            mask_function=default_mask,
            attention_mask=attention_mask,
        )
    )


def _padding_mask(plan: Gemma4MaskPlan, device: torch.device) -> torch.Tensor | None:
    mask = plan.attention_mask
    if mask is None:
        return None
    if mask.ndim != 2:
        return None
    if mask.shape[0] not in (1, plan.batch_size):
        raise ValueError("2D attention mask batch dimension must be 1 or match prepared q/k/v")
    if mask.shape[0] == 1 and plan.batch_size != 1:
        mask = mask.expand(plan.batch_size, -1)
    if mask.device != device:
        mask = mask.to(device=device)
    if not _is_fake_tensor(mask) and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("2D attention mask must use 0/1 padding values")
    kv_offset = _python_int(plan.kv_offset, name="kv_offset")
    if kv_offset < 0:
        raise ValueError("kv_offset must be nonnegative")
    target_length = kv_offset + plan.kv_length
    if mask.shape[1] < target_length:
        mask = torch.nn.functional.pad(mask, (0, target_length - mask.shape[1]), value=0)
    return mask[:, kv_offset:target_length].to(dtype=torch.bool)


def _normalize_eager_static_cache_prefix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: Gemma4MaskPlan,
    *,
    vision_block_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    requires_backward: bool,
    has_explicit_cu: bool,
    has_explicit_max: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Gemma4MaskPlan,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Expose a proven active prefix of an underfilled eager StaticCache.

    Pinned ``StaticLayer`` instances return their complete physical K/V backing
    after each update.  Their mask is built before that update, so the saved
    query offset and the returned K/V offsets identify the logical prefix as
    ``q_offset + Sq - kv_offset``.  This helper never left-trims or reorders a
    cache and deliberately accepts only the EXP-0016 B1 text/no-grad contract.
    """

    physical_k = key.shape[2]
    try:
        q_offset = _python_int(plan.q_offset, name="q_offset")
        kv_offset = _python_int(plan.kv_offset, name="kv_offset")
    except TypeError as exc:
        raise UnsupportedH100Path(
            "eager StaticCache active-prefix offsets must be real scalar integers"
        ) from exc

    logical_k = q_offset + query.shape[2] - kv_offset
    if logical_k == physical_k:
        return query, key, value, plan, vision_block_ids, document_ids
    if logical_k <= 0 or logical_k > physical_k:
        raise UnsupportedH100Path(
            "eager StaticCache active K length is outside the physical K/V backing"
        )
    if q_offset < 0 or kv_offset < 0 or q_offset < kv_offset:
        raise UnsupportedH100Path(
            "eager StaticCache offsets must describe a nonnegative lower-right prefix"
        )
    if kv_offset != 0:
        raise UnsupportedH100Path(
            "an underfilled eager StaticCache prefix must start at physical K offset zero"
        )
    if query.shape[0] != 1:
        raise UnsupportedH100Path("EXP-0016 eager StaticCache normalization accepts B1 only")
    if requires_backward:
        raise UnsupportedH100Path("EXP-0016 eager StaticCache normalization is inference-only")
    if has_explicit_cu or has_explicit_max:
        raise UnsupportedH100Path(
            "EXP-0016 eager StaticCache normalization does not accept explicit packed lengths"
        )
    if document_ids is not None:
        raise UnsupportedH100Path(
            "EXP-0016 eager StaticCache normalization does not accept document metadata"
        )
    if vision_block_ids is not None:
        if bool((vision_block_ids >= 0).any().detach().item()):
            raise UnsupportedH100Path(
                "EXP-0016 eager StaticCache normalization accepts text-only vision metadata"
            )
        vision_block_ids = None

    padding = _padding_mask(plan, query.device)
    if padding is not None and (
        padding.shape != (1, physical_k) or not bool(padding[0, :logical_k].all().item())
    ):
        raise UnsupportedH100Path("eager StaticCache requires one contiguous valid K prefix")

    native_attention_mask = plan.attention_mask
    if native_attention_mask is not None:
        native_attention_mask = native_attention_mask[:, :logical_k]

    return (
        query,
        key[:, :, :logical_k, :],
        value[:, :, :logical_k, :],
        replace(plan, kv_length=logical_k, attention_mask=native_attention_mask),
        vision_block_ids,
        document_ids,
    )


def _offset_matches_lower_right(plan: Gemma4MaskPlan, q_length: int, kv_length: int) -> bool:
    try:
        q_offset = _python_int(plan.q_offset, name="q_offset")
        kv_offset = _python_int(plan.kv_offset, name="kv_offset")
    except TypeError:
        return False
    return q_offset - kv_offset == kv_length - q_length


def _mask_plan_has_future(plan: Gemma4MaskPlan, device: torch.device) -> bool:
    if plan.mask_function is None or plan.q_length != plan.kv_length or plan.q_length < 2:
        return False
    if not _offset_matches_lower_right(plan, plan.q_length, plan.kv_length):
        return True
    try:
        q_offset = _python_int(plan.q_offset, name="q_offset")
        kv_offset = _python_int(plan.kv_offset, name="kv_offset")
        batch = torch.arange(plan.batch_size, device=device)[:, None]
        query = torch.arange(plan.q_length - 1, device=device)[None, :] + q_offset
        key = torch.arange(plan.q_length - 1, device=device)[None, :] + kv_offset + 1
        head = torch.zeros_like(query)
        allowed = plan.mask_function(batch, head, query, key)
        if not isinstance(allowed, torch.Tensor):
            return bool(allowed)
        return bool(allowed.any().detach().item())
    except Exception:
        # A mask that cannot be classified cheaply must stay on the exact fallback.
        return True


def _position_segments(
    position_ids: torch.Tensor | None,
    valid: torch.Tensor,
) -> tuple[list[int], torch.Tensor]:
    batch_size, seqlen = valid.shape
    if position_ids is not None:
        if position_ids.shape != (batch_size, seqlen):
            raise ValueError("position_ids must match the prepared Q sequence")
        if position_ids.device != valid.device:
            raise ValueError("position_ids must share the prepared tensor device")

    lengths: list[int] = []
    flat_indices: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        indices = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        if indices.numel() == 0:
            lengths.append(0)
            flat_indices.append(indices + batch_idx * seqlen)
            continue
        first = int(indices[0].item())
        last = int(indices[-1].item())
        if indices.numel() != last - first + 1:
            raise UnsupportedH100Path("holey/static padding requires exact FlexAttention fallback")
        flat_indices.append(indices + batch_idx * seqlen)
        if position_ids is None:
            lengths.append(indices.numel())
            continue
        values = position_ids[batch_idx, indices].detach().cpu()
        boundaries = torch.nonzero(torch.diff(values) != 1, as_tuple=False).flatten() + 1
        starts = [0, *[int(value) for value in boundaries.tolist()]]
        ends = [*[int(value) for value in boundaries.tolist()], int(values.numel())]
        lengths.extend(end - start for start, end in zip(starts, ends, strict=True))
    return lengths, torch.cat(flat_indices)


def _normalize_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    q_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.ndim != 2 or position_ids.shape[1] != q_length:
        raise ValueError("position_ids must have shape (1 or B, prepared Q length)")
    if position_ids.shape[0] not in (1, batch_size):
        raise ValueError("position_ids batch dimension must be 1 or match prepared q/k/v")
    if position_ids.device != device:
        raise ValueError("position_ids must share the prepared tensor device")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("position_ids must use INT32 or INT64")
    return position_ids.expand(batch_size, -1) if position_ids.shape[0] == 1 else position_ids


def _cumulative(lengths: list[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        total = values[-1] + length
        if total > torch.iinfo(torch.int32).max:
            raise UnsupportedH100Path("packed cumulative totals must fit signed INT32")
        values.append(total)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _validate_explicit_cumulative(
    values: torch.Tensor,
    *,
    name: str,
    total: int,
    batch_boundaries: tuple[int, ...],
) -> list[int]:
    if values.ndim != 1 or values.numel() < 2 or values.dtype != torch.int32:
        raise ValueError(f"{name} must be a rank-1 INT32 B+1 tensor")
    raw = [int(value) for value in values.detach().cpu().tolist()]
    if raw[0] != 0 or raw[-1] != total:
        raise ValueError(f"{name} must start at zero and end at the packed total")
    if any(end < start for start, end in zip(raw[:-1], raw[1:], strict=True)):
        raise ValueError(f"{name} must describe nondecreasing segments")
    if any(boundary not in raw for boundary in batch_boundaries):
        raise ValueError(f"{name} must preserve every batch-row boundary")
    return [end - start for start, end in zip(raw[:-1], raw[1:], strict=True)]


def _normalize_metadata(
    values: torch.Tensor | None,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
    pad_value: int | None = None,
) -> torch.Tensor | None:
    if values is None:
        return None
    if values.ndim != 2 or values.shape[0] not in (1, shape[0]):
        raise ValueError(f"{name} must have shape (1 or {shape[0]}, <= {shape[1]})")
    if values.shape[0] == 1 and shape[0] != 1:
        values = values.expand(shape[0], -1)
    if values.shape[1] != shape[1]:
        if pad_value is None or values.shape[1] > shape[1]:
            raise ValueError(f"{name} must have shape {shape}")
        values = torch.nn.functional.pad(values, (0, shape[1] - values.shape[1]), value=pad_value)
    if values.device != device:
        raise ValueError(f"{name} must share the prepared tensor device")
    if values.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use INT32 or INT64")
    return values


def _pack_local_inputs(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    plan: Gemma4MaskPlan,
    *,
    position_ids: torch.Tensor | None,
    vision_block_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | torch.Tensor | None,
    max_length_k: int | torch.Tensor | None,
) -> _PackedLocalInputs:
    batch_size, q_length, q_heads, q_dim = q_bshd.shape
    kv_length = k_bshd.shape[1]
    device = q_bshd.device
    vision = _normalize_metadata(
        vision_block_ids,
        name="vision_block_ids",
        shape=(batch_size, kv_length),
        device=device,
        pad_value=-1,
    )
    documents = _normalize_metadata(
        document_ids,
        name="document_ids",
        shape=(batch_size, kv_length),
        device=device,
        pad_value=-1,
    )
    if (cu_seq_lens_q is None) != (cu_seq_lens_k is None):
        raise ValueError("cu_seq_lens_q and cu_seq_lens_k must be provided together")

    padding = _padding_mask(plan, device)
    if q_length != kv_length:
        if padding is not None and not bool(padding.all().detach().item()):
            raise UnsupportedH100Path(
                "padded lower-right attention requires exact FlexAttention fallback"
            )
        if not _offset_matches_lower_right(plan, q_length, kv_length):
            raise UnsupportedH100Path("cache/query offset is not the FA4 lower-right alignment")
        if not _fa4_layout_is_legal(q_bshd, k_bshd, v_bshd):
            raise UnsupportedH100Path("lower-right prepared layout is outside the FA4 contract")
        if batch_size == 1:
            packed_q, packed_k, packed_v = q_bshd[0], k_bshd[0], v_bshd[0]
        else:
            packed_q = q_bshd.reshape(-1, q_heads, q_dim)
            packed_k = k_bshd.reshape(-1, k_bshd.shape[2], k_bshd.shape[3])
            packed_v = v_bshd.reshape(-1, v_bshd.shape[2], v_bshd.shape[3])
        if cu_seq_lens_q is None:
            q_lengths = [q_length] * batch_size
            k_lengths = [kv_length] * batch_size
            cu_q = _cumulative(q_lengths, device)
            cu_k = _cumulative(k_lengths, device)
        else:
            if cu_seq_lens_q.device != device or cu_seq_lens_k.device != device:
                raise ValueError("cumulative arrays must share the prepared tensor device")
            q_lengths = _validate_explicit_cumulative(
                cu_seq_lens_q,
                name="cu_seq_lens_q",
                total=batch_size * q_length,
                batch_boundaries=tuple(i * q_length for i in range(1, batch_size)),
            )
            k_lengths = _validate_explicit_cumulative(
                cu_seq_lens_k,
                name="cu_seq_lens_k",
                total=batch_size * kv_length,
                batch_boundaries=tuple(i * kv_length for i in range(1, batch_size)),
            )
            cu_q, cu_k = cu_seq_lens_q, cu_seq_lens_k
        flat_vision = None if vision is None else vision.reshape(-1)
        flat_documents = None if documents is None else documents.reshape(-1)
        q_indices = None
    else:
        valid = (
            torch.ones((batch_size, q_length), dtype=torch.bool, device=device)
            if padding is None
            else padding
        )
        q_lengths, flat_indices = _position_segments(position_ids, valid)
        derived_cu = _cumulative(q_lengths, device)
        if cu_seq_lens_q is not None:
            if cu_seq_lens_q.device != device or cu_seq_lens_k.device != device:
                raise ValueError("cumulative arrays must share the prepared tensor device")
            explicit_q = _validate_explicit_cumulative(
                cu_seq_lens_q,
                name="cu_seq_lens_q",
                total=int(valid.sum().item()),
                batch_boundaries=tuple(
                    int(value) for value in torch.cumsum(valid.sum(dim=1), dim=0)[:-1].tolist()
                ),
            )
            explicit_k = _validate_explicit_cumulative(
                cu_seq_lens_k,
                name="cu_seq_lens_k",
                total=int(valid.sum().item()),
                batch_boundaries=tuple(
                    int(value) for value in torch.cumsum(valid.sum(dim=1), dim=0)[:-1].tolist()
                ),
            )
            has_derived_boundaries = position_ids is not None or not bool(valid.all().item())
            if has_derived_boundaries and (explicit_q != q_lengths or explicit_k != q_lengths):
                raise ValueError("explicit cumulative arrays conflict with padding/position_ids")
            q_lengths = explicit_q
            k_lengths = explicit_k
            cu_q, cu_k = cu_seq_lens_q, cu_seq_lens_k
        else:
            cu_q = derived_cu
            cu_k = derived_cu.clone()

        all_valid = bool(valid.all().detach().item())
        if all_valid and batch_size == 1:
            packed_q, packed_k, packed_v = q_bshd[0], k_bshd[0], v_bshd[0]
            q_indices = None
        elif all_valid:
            packed_q = q_bshd.reshape(-1, q_heads, q_dim)
            packed_k = k_bshd.reshape(-1, k_bshd.shape[2], k_bshd.shape[3])
            packed_v = v_bshd.reshape(-1, v_bshd.shape[2], v_bshd.shape[3])
            q_indices = None
        else:
            packed_q = q_bshd[valid]
            packed_k = k_bshd[valid]
            packed_v = v_bshd[valid]
            q_indices = flat_indices
        flat_vision = None if vision is None else vision[valid]
        flat_documents = None if documents is None else documents[valid]
        if cu_seq_lens_q is None:
            k_lengths = q_lengths

    if sum(q_lengths) <= 0 or sum(k_lengths) <= 0:
        raise UnsupportedH100Path("packed physical Q/K totals must both be positive")
    derived_max_q = max(q_lengths)
    derived_max_k = max(k_lengths)
    if derived_max_q <= 0 or derived_max_k <= 0:
        raise UnsupportedH100Path("packed maximum Q/K lengths must both be positive")
    requested_max_q = (
        derived_max_q if max_length_q is None else _python_int(max_length_q, name="max_length_q")
    )
    requested_max_k = (
        derived_max_k if max_length_k is None else _python_int(max_length_k, name="max_length_k")
    )
    if (requested_max_q, requested_max_k) != (derived_max_q, derived_max_k):
        raise ValueError("max_length_q/max_length_k must equal the cumulative maxima")

    if flat_vision is not None and not bool((flat_vision >= 0).any().detach().item()):
        flat_vision = None
    if flat_vision is not None and flat_documents is None:
        flat_documents = torch.repeat_interleave(
            torch.arange(len(k_lengths), dtype=torch.int32, device=device),
            torch.tensor(k_lengths, dtype=torch.int64, device=device),
        )

    return _PackedLocalInputs(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        cu_q=cu_q,
        cu_k=cu_k,
        max_q=derived_max_q,
        max_k=derived_max_k,
        q_flat_indices=q_indices,
        original_q_shape=(batch_size, q_length, q_heads, q_dim),
        vision_block_ids=flat_vision,
        document_ids=flat_documents,
    )


def _unpack_local_result(
    output: torch.Tensor,
    lse: torch.Tensor,
    packed: _PackedLocalInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, q_length, q_heads, q_dim = packed.original_q_shape
    if packed.q_flat_indices is None:
        unpacked_output = output.reshape(batch_size, q_length, q_heads, q_dim)
        unpacked_lse = lse.reshape(q_heads, batch_size, q_length).permute(1, 0, 2)
        return unpacked_output, unpacked_lse

    output_flat = torch.zeros(
        batch_size * q_length,
        q_heads,
        q_dim,
        dtype=output.dtype,
        device=output.device,
    )
    output_flat = output_flat.index_copy(0, packed.q_flat_indices, output)
    lse_flat = torch.full(
        (batch_size * q_length, q_heads),
        -torch.inf,
        dtype=lse.dtype,
        device=lse.device,
    )
    lse_flat = lse_flat.index_copy(0, packed.q_flat_indices, lse.transpose(0, 1))
    return (
        output_flat.view(batch_size, q_length, q_heads, q_dim),
        lse_flat.view(batch_size, q_length, q_heads).permute(0, 2, 1),
    )


def _cumulative_lengths(values: torch.Tensor) -> list[int]:
    raw = [int(value) for value in values.detach().cpu().tolist()]
    return [end - start for start, end in zip(raw[:-1], raw[1:], strict=True)]


def _split_equal_lengths_on_documents(
    lengths: list[int],
    document_ids: torch.Tensor | None,
) -> list[int]:
    if document_ids is None:
        return lengths
    split_lengths: list[int] = []
    start = 0
    for length in lengths:
        segment = document_ids[start : start + length]
        raw_ids = [int(value) for value in segment.detach().cpu().tolist()]
        seen: set[int] = set()
        previous: int | None = None
        for document_id in raw_ids:
            if document_id != previous:
                if document_id in seen:
                    raise UnsupportedH100Path(
                        "global composition cannot preserve noncontiguous repeated document IDs"
                    )
                seen.add(document_id)
                previous = document_id
        boundaries = torch.nonzero(torch.diff(segment) != 0, as_tuple=False).flatten() + 1
        points = [0, *[int(value) for value in boundaries.detach().cpu().tolist()], length]
        split_lengths.extend(
            end - begin for begin, end in zip(points[:-1], points[1:], strict=True)
        )
        start += length
    return split_lengths


def _global_segment_lengths(packed: _PackedLocalInputs) -> tuple[list[int], list[int]]:
    q_lengths = _cumulative_lengths(packed.cu_q)
    k_lengths = _cumulative_lengths(packed.cu_k)
    if len(q_lengths) != len(k_lengths):
        raise ValueError("global packed Q/K cumulative arrays must have one-to-one segments")
    if packed.document_ids is not None:
        if q_lengths != k_lengths:
            documents = torch.split(packed.document_ids, k_lengths)
            if any(
                q_length > 0 and bool((torch.diff(values) != 0).any().item())
                for q_length, values in zip(q_lengths, documents, strict=True)
            ):
                raise UnsupportedH100Path(
                    "document-split lower-right global attention is not yet composable"
                )
        else:
            q_lengths = _split_equal_lengths_on_documents(
                q_lengths,
                packed.document_ids,
            )
            k_lengths = list(q_lengths)
    if any(
        q_length < 0 or k_length < 0 or q_length > k_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    ):
        raise UnsupportedH100Path("global composition requires 0 <= Sq <= Sk per segment")
    if sum(q_lengths) <= 0 or sum(k_lengths) <= 0:
        raise UnsupportedH100Path("global packed physical Q/K totals must both be positive")
    return q_lengths, k_lengths


def _run_global_composed(
    packed: _PackedLocalInputs,
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lengths, k_lengths = _global_segment_lengths(packed)
    if any(
        q_length > 0 and k_length > _GLOBAL_COMPOSED_BACKWARD_MAX_SEQLEN
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    ):
        raise UnsupportedH100Path(
            "EXP-0012 composed global backward requires every K segment <= 2048"
        )
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for q_segment, k_segment, v_segment, q_length, k_length in zip(
        torch.split(packed.q, q_lengths, dim=0),
        torch.split(packed.k, k_lengths, dim=0),
        torch.split(packed.v, k_lengths, dim=0),
        q_lengths,
        k_lengths,
        strict=True,
    ):
        if q_length == 0:
            continue
        if q_length != k_length:
            prefix = torch.zeros(
                (k_length - q_length, *q_segment.shape[1:]),
                dtype=q_segment.dtype,
                device=q_segment.device,
            )
            q_segment = torch.cat((prefix, q_segment), dim=0)
        q_fixed = q_segment.unsqueeze(0)
        k_fixed = k_segment.unsqueeze(0)
        v_fixed = v_segment.unsqueeze(0)
        if not _fa4_layout_is_legal(q_fixed, k_fixed, v_fixed):
            q_fixed, k_fixed, v_fixed = (
                tensor.contiguous() for tensor in (q_fixed, k_fixed, v_fixed)
            )
        output, lse = fa4_global_text_forward(q_fixed, k_fixed, v_fixed, spec=spec)
        outputs.append(output[0, -q_length:])
        lses.append(lse[0, :, -q_length:])

    return _unpack_local_result(
        torch.cat(outputs, dim=0),
        torch.cat(lses, dim=1),
        packed,
    )


def _run_global_varlen_forward_only(
    packed: _PackedLocalInputs,
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lengths, k_lengths = _global_segment_lengths(packed)
    cu_q = _cumulative(q_lengths, packed.q.device)
    cu_k = _cumulative(k_lengths, packed.q.device)
    output, lse = fa4_global_varlen_forward_only(
        packed.q,
        packed.k,
        packed.v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        spec=spec,
    )
    return _unpack_local_result(output, lse, packed)


def _run_global_native_varlen(
    packed: _PackedLocalInputs,
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lengths, k_lengths = _global_segment_lengths(packed)
    if any(k_length > GEMMA4_31B.max_position_embeddings for k_length in k_lengths):
        raise UnsupportedH100Path("native global backward requires every K segment <= 262144")
    cu_q = _cumulative(q_lengths, packed.q.device)
    cu_k = _cumulative(k_lengths, packed.q.device)
    output, lse = fa4_global_varlen_forward(
        packed.q,
        packed.k,
        packed.v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        spec=spec,
    )
    return _unpack_local_result(output, lse, packed)


def _run_flex_fallback(
    module: Any,
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    plan: Gemma4MaskPlan,
    *,
    scaling: float,
) -> torch.Tensor:
    try:
        from transformers.integrations.flex_attention import flex_attention_forward
        from transformers.masking_utils import causal_mask_function, flex_attention_mask
    except Exception as exc:  # pragma: no cover - depends on optional pinned HF
        raise UnsupportedH100Path(
            "exact FlexAttention fallback requires the pinned Transformers checkout"
        ) from exc

    if plan.attention_mask is not None and plan.attention_mask.ndim == 4:
        exact_mask = plan.attention_mask
        if exact_mask.dtype == torch.bool or not exact_mask.is_floating_point():
            raise UnsupportedH100Path(
                "boolean/integer 4D masks have ambiguous semantics for the additive "
                "FlexAttention score-mask interface"
            )
        expected_prefix = (plan.batch_size, 1, plan.q_length, plan.kv_length)
        if (
            exact_mask.shape[0] not in (1, plan.batch_size)
            or exact_mask.shape[1:] != expected_prefix[1:]
        ):
            raise UnsupportedH100Path(
                "additive 4D FlexAttention masks must have shape "
                f"(1 or B, 1, Sq, Sk)={expected_prefix}"
            )
        if exact_mask.device != q_bhsd.device:
            exact_mask = exact_mask.to(device=q_bhsd.device)
        if exact_mask.shape[0] == 1 and plan.batch_size != 1:
            exact_mask = exact_mask.expand(plan.batch_size, -1, -1, -1)
    else:
        exact_mask = flex_attention_mask(
            batch_size=plan.batch_size,
            q_length=plan.q_length,
            kv_length=plan.kv_length,
            q_offset=plan.q_offset,
            kv_offset=plan.kv_offset,
            mask_function=(
                causal_mask_function if plan.mask_function is None else plan.mask_function
            ),
            attention_mask=plan.attention_mask,
            device=q_bhsd.device,
        )
    output, _fallback_lse = flex_attention_forward(
        module,
        q_bhsd,
        k_bhsd,
        v_bhsd,
        exact_mask,
        dropout=0.0,
        scaling=scaling,
        kernel_options=(_FLEX_H100_D512_KERNEL_OPTIONS if q_bhsd.shape[-1] == 512 else None),
    )
    return output


def _fallback_result(
    reason: str,
    *,
    module: Any,
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    plan: Gemma4MaskPlan,
    scaling: float,
    allow_flex_fallback: bool,
) -> Gemma4DispatchResult:
    if not allow_flex_fallback:
        raise UnsupportedH100Path(f"{reason}; exact FlexAttention fallback is disabled")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q_bhsd, k_bhsd, v_bhsd)):
        raise UnsupportedH100Path(
            f"{reason}; FlexAttention fallback backward is not accepted on H100"
        )
    warnings.warn(f"Gemma 4 H100 FA4 fallback: {reason}", RuntimeWarning, stacklevel=3)
    output = _run_flex_fallback(
        module,
        q_bhsd,
        k_bhsd,
        v_bhsd,
        plan,
        scaling=scaling,
    )
    expected = (q_bhsd.shape[0], q_bhsd.shape[2], q_bhsd.shape[1], q_bhsd.shape[3])
    if output.shape != expected or output.dtype != q_bhsd.dtype:
        raise RuntimeError("FlexAttention fallback returned an invalid BSHD output contract")
    return Gemma4DispatchResult(output=output, lse=None, path="flex_attention")


def gemma4_fa4_prepared(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float = 1.0,
    sliding_window: int | None = None,
    position_ids: torch.Tensor | None = None,
    vision_block_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | torch.Tensor | None = None,
    max_length_k: int | torch.Tensor | None = None,
    allow_flex_fallback: bool = True,
    output_attentions: bool = False,
    **_kwargs,
) -> Gemma4DispatchResult:
    """Route pinned Transformers prepared Q/K/V to exact H100 implementations."""

    spec = _spec_for_module(module)
    _validate_semantics(
        module,
        spec,
        dropout=dropout,
        scaling=scaling,
        sliding_window=sliding_window,
        output_attentions=output_attentions,
    )
    if _is_compiler_or_fake(query, key, value):
        if not isinstance(attention_mask, Gemma4MaskPlan):
            raise UnsupportedH100Path(
                "Transformers FakeTensor/torch.compile requires a pinned-origin Gemma4MaskPlan"
            )
        return _run_compiler_fixed_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            spec,
            position_ids=position_ids,
            vision_block_ids=vision_block_ids,
            document_ids=document_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            extra_kwargs=_kwargs,
        )
    batch_size, q_length, kv_length = _validate_prepared_inputs(query, key, value, spec)
    if _is_fake_tensor(query):
        raise UnsupportedH100Path(
            "Transformers FakeTensor/torch.compile tracing is not accepted for this adapter"
        )
    position_ids = _normalize_position_ids(
        position_ids,
        batch_size=batch_size,
        q_length=q_length,
        device=query.device,
    )
    vision = _normalize_metadata(
        vision_block_ids,
        name="vision_block_ids",
        shape=(batch_size, kv_length),
        device=query.device,
        pad_value=-1,
    )
    documents = _normalize_metadata(
        document_ids,
        name="document_ids",
        shape=(batch_size, kv_length),
        device=query.device,
        pad_value=-1,
    )
    plan = _as_mask_plan(
        attention_mask,
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        spec=spec,
        vision_block_ids=vision,
        document_ids=documents,
    )
    requires_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )

    if not _mask_plan_matches_native_contract(
        plan,
        spec,
        position_ids=position_ids,
        vision_block_ids=vision,
        document_ids=documents,
    ):
        return _fallback_result(
            "the exact mask is not the proven Gemma 4 native predicate",
            module=module,
            q_bhsd=query,
            k_bhsd=key,
            v_bhsd=value,
            plan=plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )

    physical_query, physical_key, physical_value = query, key, value
    physical_plan = plan
    has_explicit_cu = cu_seq_lens_q is not None or cu_seq_lens_k is not None
    has_explicit_max = max_length_q is not None or max_length_k is not None
    try:
        query, key, value, plan, vision, documents = _normalize_eager_static_cache_prefix(
            query,
            key,
            value,
            plan,
            vision_block_ids=vision,
            document_ids=documents,
            requires_backward=requires_backward,
            has_explicit_cu=has_explicit_cu,
            has_explicit_max=has_explicit_max,
        )
    except UnsupportedH100Path as exc:
        return _fallback_result(
            str(exc),
            module=module,
            q_bhsd=physical_query,
            k_bhsd=physical_key,
            v_bhsd=physical_value,
            plan=physical_plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )
    kv_length = key.shape[2]
    static_prefix_trimmed = kv_length != physical_key.shape[2]
    # Pinned Transformers deliberately skips position-ID packed-sequence
    # inference whenever a cache is present.  Preserve the already-proven
    # physical mask predicate after exposing a StaticCache prefix instead of
    # reinterpreting rotary positions as document boundaries.
    native_position_ids = None if static_prefix_trimmed else position_ids

    q_bshd = query.transpose(1, 2)
    k_bshd = key.transpose(1, 2)
    v_bshd = value.transpose(1, 2)
    if not _fa4_layout_is_legal(q_bshd, k_bshd, v_bshd):
        return _fallback_result(
            "prepared BHSD-to-BSHD view has an unsupported FA4 layout",
            module=module,
            q_bhsd=physical_query,
            k_bhsd=physical_key,
            v_bhsd=physical_value,
            plan=physical_plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )

    padding = _padding_mask(plan, query.device)
    has_padding = padding is not None and not bool(padding.all().detach().item())
    packed_positions = False
    if native_position_ids is not None and q_length == kv_length:
        if native_position_ids.shape != (batch_size, q_length):
            raise ValueError("position_ids must match the prepared Q sequence")
        for batch_idx in range(batch_size):
            valid = (
                torch.ones(q_length, dtype=torch.bool, device=query.device)
                if padding is None
                else padding[batch_idx]
            )
            values = native_position_ids[batch_idx, valid]
            if values.numel() > 1 and bool((torch.diff(values) != 1).any().detach().item()):
                packed_positions = True
                break
    has_documents = documents is not None and bool(
        (documents != documents[:, :1]).any().detach().item()
    )

    if spec.kind == "full_attention":
        fixed_global = (
            batch_size == 1
            and q_length == kv_length
            and q_length <= (2048 if requires_backward else 1024)
            and not has_padding
            and not packed_positions
            and not has_explicit_cu
            and not has_documents
            and _offset_matches_lower_right(plan, q_length, kv_length)
        )
        if fixed_global:
            try:
                output, lse = fa4_global_text_forward(q_bshd, k_bshd, v_bshd, spec=spec)
            except UnsupportedH100Path as exc:
                return _fallback_result(
                    str(exc),
                    module=module,
                    q_bhsd=physical_query,
                    k_bhsd=physical_key,
                    v_bhsd=physical_value,
                    plan=physical_plan,
                    scaling=scaling,
                    allow_flex_fallback=allow_flex_fallback,
                )
            return Gemma4DispatchResult(output=output, lse=lse, path="fa4_global_fixed")
        direct_forward_only = (
            not requires_backward
            and kv_length > 1024
            and not has_padding
            and not packed_positions
            and not has_explicit_cu
            and not has_documents
            and _offset_matches_lower_right(plan, q_length, kv_length)
        )
        if direct_forward_only:
            try:
                output, lse = fa4_global_forward_only(q_bshd, k_bshd, v_bshd, spec=spec)
            except UnsupportedH100Path as exc:
                return _fallback_result(
                    str(exc),
                    module=module,
                    q_bhsd=physical_query,
                    k_bhsd=physical_key,
                    v_bhsd=physical_value,
                    plan=physical_plan,
                    scaling=scaling,
                    allow_flex_fallback=allow_flex_fallback,
                )
            return Gemma4DispatchResult(
                output=output,
                lse=lse,
                path="fa4_global_forward_only",
            )
        try:
            packed = _pack_local_inputs(
                q_bshd,
                k_bshd,
                v_bshd,
                plan,
                position_ids=native_position_ids,
                vision_block_ids=vision,
                document_ids=documents,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
            )
            if requires_backward:
                try:
                    output, lse = _run_global_native_varlen(packed, spec)
                    path = "fa4_global_varlen_native"
                except GlobalBackwardBudgetExceeded:
                    q_lengths, k_lengths = _global_segment_lengths(packed)
                    if any(
                        q_length > 0 and k_length > _GLOBAL_COMPOSED_BACKWARD_MAX_SEQLEN
                        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
                    ):
                        raise
                    output, lse = _run_global_composed(packed, spec)
                    path = "fa4_global_varlen_composed_budget_fallback"
            else:
                q_lengths, k_lengths = _global_segment_lengths(packed)
                if all(
                    q_length == 0 or k_length <= 1024
                    for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
                ):
                    output, lse = _run_global_composed(packed, spec)
                    path = "fa4_global_varlen"
                else:
                    output, lse = _run_global_varlen_forward_only(packed, spec)
                    path = "fa4_global_varlen_forward_only"
        except GlobalBackwardBudgetExceeded:
            raise
        except UnsupportedH100Path as exc:
            return _fallback_result(
                str(exc),
                module=module,
                q_bhsd=physical_query,
                k_bhsd=physical_key,
                v_bhsd=physical_value,
                plan=physical_plan,
                scaling=scaling,
                allow_flex_fallback=allow_flex_fallback,
            )
        return Gemma4DispatchResult(output=output, lse=lse, path=path)

    missing_vision_metadata = vision is None and _mask_plan_has_future(plan, query.device)
    fixed_local = (
        batch_size == 1
        and q_length == kv_length
        and q_length <= 1025
        and not has_padding
        and not packed_positions
        and not has_explicit_cu
        and not has_documents
        and not missing_vision_metadata
        and _offset_matches_lower_right(plan, q_length, kv_length)
    )
    if fixed_local:
        active_vision = vision
        if active_vision is not None and not bool((active_vision >= 0).any().detach().item()):
            active_vision = None
        try:
            output, lse = fa4_local_forward(
                q_bshd,
                k_bshd,
                v_bshd,
                vision_block_ids=active_vision,
                spec=spec,
            )
        except UnsupportedH100Path as exc:
            return _fallback_result(
                str(exc),
                module=module,
                q_bhsd=physical_query,
                k_bhsd=physical_key,
                v_bhsd=physical_value,
                plan=physical_plan,
                scaling=scaling,
                allow_flex_fallback=allow_flex_fallback,
            )
        return Gemma4DispatchResult(output=output, lse=lse, path="fa4_local_fixed")

    if missing_vision_metadata:
        return _fallback_result(
            "the exact mask admits future vision tokens but compact vision metadata is missing",
            module=module,
            q_bhsd=physical_query,
            k_bhsd=physical_key,
            v_bhsd=physical_value,
            plan=physical_plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )

    try:
        packed = _pack_local_inputs(
            q_bshd,
            k_bshd,
            v_bshd,
            plan,
            position_ids=native_position_ids,
            vision_block_ids=vision,
            document_ids=documents,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        output, lse = fa4_local_varlen_forward(
            packed.q,
            packed.k,
            packed.v,
            packed.cu_q,
            packed.cu_k,
            max_seqlen_q=packed.max_q,
            max_seqlen_k=packed.max_k,
            vision_block_ids=packed.vision_block_ids,
            document_ids=packed.document_ids,
            spec=spec,
        )
        output, lse = _unpack_local_result(output, lse, packed)
    except UnsupportedH100Path as exc:
        return _fallback_result(
            str(exc),
            module=module,
            q_bhsd=physical_query,
            k_bhsd=physical_key,
            v_bhsd=physical_value,
            plan=physical_plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )
    return Gemma4DispatchResult(output=output, lse=lse, path="fa4_local_varlen")


def gemma4_fa4_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Pinned Transformers attention-interface entry point."""

    result = gemma4_fa4_prepared(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )
    if not _is_compiler_or_fake(query, key, value):
        module._gemma4_fa4_last_path = result.path
    return result.output, None


def _whole_layer_projection_weight(
    module: Any,
    name: str,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    projection = getattr(module, name, None)
    if type(projection) is not torch.nn.Linear or projection.bias is not None:
        raise UnsupportedH100Path(f"EXP-0020 requires the pinned bias-free {name} projection")
    weight = projection.weight
    if (
        not isinstance(weight, torch.nn.Parameter)
        or weight.requires_grad is not True
        or weight.shape != expected_shape
        or weight.dtype != torch.bfloat16
        or weight.device.type != "cuda"
        or not weight.is_contiguous()
    ):
        raise UnsupportedH100Path(f"EXP-0020 {name} weight conflicts with the tensor-explicit ABI")
    return weight


def _whole_layer_norm_weight(
    module: Any,
    name: str,
    head_dim: int,
) -> torch.Tensor:
    norm = getattr(module, name, None)
    weight = getattr(norm, "weight", None)
    if (
        norm is None
        or getattr(norm, "eps", None) != 1e-6
        or getattr(norm, "with_scale", None) is not True
        or not isinstance(weight, torch.nn.Parameter)
        or weight.requires_grad is not True
        or weight.shape != (head_dim,)
        or weight.dtype != torch.bfloat16
        or weight.device.type != "cuda"
        or not weight.is_contiguous()
    ):
        raise UnsupportedH100Path(f"EXP-0020 {name} conflicts with the tensor-explicit RMS ABI")
    return weight


def _validate_whole_layer_module(
    module: Any,
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, ...]:
    expected_layer_idx = 5 if spec.kind == "full_attention" else 0
    config = getattr(module, "config", None)
    if (
        _PINNED_GEMMA4_TEXT_ATTENTION_CLASS is None
        or type(module) is not _PINNED_GEMMA4_TEXT_ATTENTION_CLASS
        or getattr(module, "layer_idx", None) != expected_layer_idx
        or getattr(module, "training", None) is not False
        or not _is_pinned_gemma4_text_config(config)
        or getattr(module, "is_sliding", None) is not (spec.kind == "sliding_attention")
        or getattr(module, "is_kv_shared_layer", None) is not False
        or getattr(module, "store_full_length_kv", None) is not False
        or getattr(module, "head_dim", None) != spec.head_dim_qk
        or getattr(module, "num_key_value_groups", None) != spec.qhead_per_kvhead
        or getattr(module, "scaling", None) != 1.0
        or getattr(module, "attention_dropout", None) != 0.0
    ):
        raise UnsupportedH100Path(
            "EXP-0020 whole-layer routing requires the exact pinned eval-mode "
            f"Gemma4TextAttention layer {expected_layer_idx}"
        )

    use_alternative = spec.kind == "full_attention"
    if getattr(module, "use_alternative_attention", None) is not use_alternative:
        raise UnsupportedH100Path("EXP-0020 module K/V projection mode conflicts with its family")
    v_norm = getattr(module, "v_norm", None)
    if (
        v_norm is None
        or getattr(v_norm, "eps", None) != 1e-6
        or getattr(v_norm, "with_scale", None) is not False
        or hasattr(v_norm, "weight")
    ):
        raise UnsupportedH100Path("EXP-0020 requires the pinned scale-free V RMSNorm")

    hidden_size = GEMMA4_31B.hidden_size
    q_width = spec.num_q_heads * spec.head_dim_qk
    kv_width = spec.num_kv_heads * spec.head_dim_qk
    q_weight = _whole_layer_projection_weight(module, "q_proj", (q_width, hidden_size))
    k_weight = _whole_layer_projection_weight(module, "k_proj", (kv_width, hidden_size))
    o_weight = _whole_layer_projection_weight(module, "o_proj", (hidden_size, q_width))
    q_norm_weight = _whole_layer_norm_weight(module, "q_norm", spec.head_dim_qk)
    k_norm_weight = _whole_layer_norm_weight(module, "k_norm", spec.head_dim_qk)
    source_parameters = [
        module.q_proj.weight,
        module.k_proj.weight,
        module.o_proj.weight,
        module.q_norm.weight,
        module.k_norm.weight,
    ]
    if use_alternative:
        if getattr(module, "v_proj", object()) is not None:
            raise UnsupportedH100Path(
                "EXP-0020 global routing requires one shared K-projection source"
            )
        if len({id(parameter) for parameter in source_parameters}) != len(source_parameters):
            raise UnsupportedH100Path("EXP-0020 weights conflict with the tensor-explicit ABI")
        return q_weight, k_weight, o_weight, q_norm_weight, k_norm_weight

    v_weight = _whole_layer_projection_weight(module, "v_proj", (kv_width, hidden_size))
    source_parameters.append(module.v_proj.weight)
    if len({id(parameter) for parameter in source_parameters}) != len(source_parameters):
        raise UnsupportedH100Path("EXP-0020 weights conflict with the tensor-explicit ABI")
    return q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight


def gemma4_fa4_compile_layer(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """EXP-0020's exact, inference-only compiled whole-attention-layer boundary."""

    if not bool(_TORCH_IS_COMPILING()):
        raise UnsupportedH100Path("EXP-0020 whole-layer routing is compile-only")
    if not CUSTOM_OPS_AVAILABLE:
        raise UnsupportedH100Path(
            "EXP-0020 whole-layer routing requires PyTorch custom_op and register_fake APIs"
        )
    if past_key_values is not None:
        raise UnsupportedH100Path("EXP-0020 whole-layer routing rejects every cache")
    if type(shared_kv_states) is not dict or shared_kv_states:
        raise UnsupportedH100Path("EXP-0020 whole-layer routing rejects shared prepared KV")
    if type(position_embeddings) is not tuple or len(position_embeddings) != 2:
        raise UnsupportedH100Path("EXP-0020 requires the pinned cosine/sine tuple")

    allowed_kwargs = {"position_ids", "allow_flex_fallback"}
    unexpected_kwargs = set(kwargs).difference(allowed_kwargs)
    if unexpected_kwargs:
        raise UnsupportedH100Path(
            "EXP-0020 whole-layer routing rejects metadata outside its declared scope: "
            + ", ".join(sorted(unexpected_kwargs))
        )
    if kwargs.get("allow_flex_fallback", False) is not False:
        raise UnsupportedH100Path("EXP-0020 whole-layer routing does not accept fallback")
    position_ids = kwargs.get("position_ids")
    if not isinstance(position_ids, torch.Tensor):
        raise UnsupportedH100Path("EXP-0020 requires explicit position_ids")

    spec = _spec_for_module(module)
    weights = _validate_whole_layer_module(module, spec)
    expected_origin = _compiler_origin_for_spec(spec)
    config = getattr(module, "config", None)
    if (
        not isinstance(attention_mask, Gemma4MaskPlan)
        or getattr(attention_mask, "_gemma4_fa4_compile_origin", None) is not expected_origin
        or getattr(attention_mask, "_gemma4_fa4_compile_config", None) is not config
        or attention_mask.attention_mask is not None
        or type(attention_mask.q_offset) is not int
        or attention_mask.q_offset != 0
        or type(attention_mask.kv_offset) is not int
        or attention_mask.kv_offset != 0
    ):
        raise UnsupportedH100Path(
            "EXP-0020 requires the exact pinned no-cache mask origin for its layer family"
        )

    cos, sin = position_embeddings
    if hidden_states.ndim != 3:
        raise UnsupportedH100Path("EXP-0020 hidden_states must be rank-3 BSH")
    batch_size, seqlen, hidden_size = hidden_states.shape
    if batch_size != 1 or seqlen < 1 or seqlen > 1024 or hidden_size != GEMMA4_31B.hidden_size:
        raise UnsupportedH100Path("EXP-0020 requires B1, hidden size 5376, and 1 <= S <= 1024")
    expected_rotary_shape = (1, seqlen, spec.head_dim_qk)
    if (
        cos.shape != expected_rotary_shape
        or sin.shape != expected_rotary_shape
        or position_ids.shape != (1, seqlen)
        or attention_mask.batch_size != 1
        or attention_mask.q_length != seqlen
        or attention_mask.kv_length != seqlen
    ):
        raise UnsupportedH100Path("EXP-0020 input and mask shapes conflict with the locked layer")
    explicit_inputs = (hidden_states, cos, sin, position_ids, *weights)
    if any(tensor.device != hidden_states.device for tensor in explicit_inputs):
        raise UnsupportedH100Path("EXP-0020 tensors must share one CUDA device")
    if hidden_states.device.type != "cuda":
        raise UnsupportedH100Path("EXP-0020 whole-layer routing requires CUDA")
    if any(tensor.dtype != torch.bfloat16 for tensor in (hidden_states, cos, sin, *weights)):
        raise UnsupportedH100Path("EXP-0020 activations and source weights must use BF16")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise UnsupportedH100Path("EXP-0020 position_ids must use INT32 or INT64")
    if any(tensor.requires_grad for tensor in (hidden_states, cos, sin)):
        raise UnsupportedH100Path("EXP-0020 whole-layer activation ABI rejects requires_grad")
    if torch.is_grad_enabled():
        raise UnsupportedH100Path("EXP-0020 whole-layer routing requires inference mode")

    first_position = position_ids[:, :1] - 1
    packed_sequence_ids = (torch.diff(position_ids, prepend=first_position, dim=-1) != 1).cumsum(-1)
    if spec.kind == "full_attention":
        output, _lse = h100_global_layer_fwd(
            hidden_states,
            cos,
            sin,
            position_ids,
            packed_sequence_ids,
            *weights,
        )
    else:
        output, _lse = h100_local_layer_fwd(
            hidden_states,
            cos,
            sin,
            position_ids,
            packed_sequence_ids,
            *weights,
        )
    return output, None


def _guarded_local_tensor_only(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """EXP-0023 local inner frame: tensors and integer shape policy only."""

    packed_sequence_ids = torch.zeros_like(position_ids)
    output, _lse = h100_local_layer_fwd(
        hidden_states,
        cos,
        sin,
        position_ids,
        packed_sequence_ids,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )
    return output


def _guarded_global_tensor_only(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """EXP-0023 global inner frame: tensors and integer shape policy only."""

    packed_sequence_ids = torch.zeros_like(position_ids)
    output, _lse = h100_global_layer_fwd(
        hidden_states,
        cos,
        sin,
        position_ids,
        packed_sequence_ids,
        q_proj_weight,
        k_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )
    return output


def _validate_guarded_facade_inputs(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate EXP-0023 before entering its compiled tensor-only function."""

    if torch.is_grad_enabled():
        raise UnsupportedH100Path("EXP-0023 guarded facade requires inference/no-grad mode")
    if type(position_embeddings) is not tuple or len(position_embeddings) != 2:
        raise UnsupportedH100Path("EXP-0023 requires the pinned cosine/sine tuple")
    cos, sin = position_embeddings
    if hidden_states.ndim != 3:
        raise UnsupportedH100Path("EXP-0023 hidden_states must be rank-3 BSH")
    batch_size, seqlen, hidden_size = hidden_states.shape
    if batch_size != 1 or seqlen < 1 or seqlen > 1024 or hidden_size != GEMMA4_31B.hidden_size:
        raise UnsupportedH100Path("EXP-0023 requires B1, hidden size 5376, and 1 <= S <= 1024")
    expected_rotary_shape = (1, seqlen, spec.head_dim_qk)
    if cos.shape != expected_rotary_shape or sin.shape != expected_rotary_shape:
        raise UnsupportedH100Path("EXP-0023 rotary tensors conflict with the locked layer")
    if position_ids.shape != (1, seqlen):
        raise UnsupportedH100Path("EXP-0023 position_ids shape conflicts with hidden_states")
    explicit_tensors = (hidden_states, cos, sin, position_ids, *weights)
    if any(tensor.device != hidden_states.device for tensor in explicit_tensors):
        raise UnsupportedH100Path("EXP-0023 tensors must share one CUDA device")
    if hidden_states.device.type != "cuda":
        raise UnsupportedH100Path("EXP-0023 guarded facade requires CUDA")
    if any(tensor.dtype != torch.bfloat16 for tensor in (hidden_states, cos, sin, *weights)):
        raise UnsupportedH100Path("EXP-0023 activations and source weights must use BF16")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise UnsupportedH100Path("EXP-0023 position_ids must use INT32 or INT64")
    if any(tensor.requires_grad for tensor in (hidden_states, cos, sin)):
        raise UnsupportedH100Path("EXP-0023 activation tensors must not require grad")
    if any(not tensor.is_contiguous() for tensor in explicit_tensors):
        raise UnsupportedH100Path("EXP-0023 facade tensors must be contiguous")
    expected_positions = torch.arange(
        seqlen,
        dtype=position_ids.dtype,
        device=position_ids.device,
    ).unsqueeze(0)
    if not torch.equal(position_ids, expected_positions):
        raise UnsupportedH100Path("EXP-0023 requires exact zero-based position_ids")
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_tensors}
    if len(storage_pointers) != len(explicit_tensors):
        raise UnsupportedH100Path("EXP-0023 facade tensors must use distinct storage")
    return cos, sin


_GUARDED_FACADE_SCALAR_CONTRACT = (
    ("config.sliding_rope_theta", 10_000.0),
    ("config.full_partial_rotary_factor", 0.25),
    ("config.full_rope_theta", 1_000_000.0),
    ("config.attention_dropout", 0.0),
    ("config.final_logit_softcapping", 30.0),
    ("config.rms_norm_eps", 1e-6),
    ("module.scaling", 1.0),
    ("module.attention_dropout", 0.0),
    ("module.q_norm.eps", 1e-6),
    ("module.k_norm.eps", 1e-6),
    ("module.v_norm.eps", 1e-6),
)


def _guarded_facade_scalar_values(module: Any) -> tuple[Any, ...]:
    config = module.config
    sliding_rope = config.rope_parameters["sliding_attention"]
    full_rope = config.rope_parameters["full_attention"]
    return (
        sliding_rope["rope_theta"],
        full_rope["partial_rotary_factor"],
        full_rope["rope_theta"],
        config.attention_dropout,
        config.final_logit_softcapping,
        config.rms_norm_eps,
        module.scaling,
        module.attention_dropout,
        module.q_norm.eps,
        module.k_norm.eps,
        module.v_norm.eps,
    )


def _validate_guarded_facade_scalars(module: Any) -> None:
    """Require exact float types and values outside Dynamo on every call."""

    values = _guarded_facade_scalar_values(module)
    for (name, expected), value in zip(
        _GUARDED_FACADE_SCALAR_CONTRACT,
        values,
        strict=True,
    ):
        if type(value) is not float or value != expected:
            raise UnsupportedH100Path(
                f"EXP-0023 guarded facade requires {name}={expected!r} as an exact float"
            )


class Gemma4H100CompiledLayerFacade:
    """Explicit guarded facade; not raw ``torch.compile(layer)`` support."""

    def __init__(
        self,
        module: Any,
        spec: AttentionLayerSpec,
        compiled_call: Callable[..., torch.Tensor],
    ) -> None:
        self._module = module
        self._spec = spec
        self._compiled_call = compiled_call
        self._compiled_entry_count = 0

    @property
    def compiled_entry_count(self) -> int:
        return self._compiled_entry_count

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        weights = _validate_whole_layer_module(self._module, self._spec)
        _validate_guarded_facade_scalars(self._module)
        cos, sin = _validate_guarded_facade_inputs(
            hidden_states,
            position_embeddings,
            position_ids,
            weights,
            self._spec,
        )
        self._compiled_entry_count += 1
        output = self._compiled_call(hidden_states, cos, sin, position_ids, *weights)
        return output, None


def compile_gemma4_fa4_h100_layer(
    module: Any,
    *,
    backend: str | Callable = "inductor",
) -> Gemma4H100CompiledLayerFacade:
    """Build EXP-0023's per-call guarded, tensor-only compiled layer facade."""

    register_gemma4_fa4_h100()
    if not CUSTOM_OPS_AVAILABLE:
        raise UnsupportedH100Path("EXP-0023 requires PyTorch custom_op and register_fake APIs")
    compile_api = getattr(torch, "compile", None)
    if not callable(compile_api):
        raise UnsupportedH100Path("EXP-0023 requires torch.compile")
    if not (backend in {"eager", "inductor"} if isinstance(backend, str) else callable(backend)):
        raise ValueError("EXP-0023 backend must be eager, inductor, or a callable delegate wrapper")
    spec = _spec_for_module(module)
    _validate_whole_layer_module(module, spec)
    _validate_guarded_facade_scalars(module)
    inner = (
        _guarded_global_tensor_only if spec.kind == "full_attention" else _guarded_local_tensor_only
    )
    compiled_call = compile_api(
        inner,
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    return Gemma4H100CompiledLayerFacade(module, spec, compiled_call)


def _tensor_version_or_none(tensor: torch.Tensor) -> int | None:
    """Read a mutation version when the tensor is not an inference tensor."""

    try:
        return tensor._version
    except RuntimeError:
        return None


@dataclass(frozen=True)
class _GuardedGlobalStaticCacheBinding:
    cache: Any
    layer: Any
    layer_ids: tuple[int, ...]
    keys: torch.Tensor
    values: torch.Tensor
    cumulative_length: torch.Tensor
    compiled_keys: torch.Tensor
    compiled_values: torch.Tensor
    compiled_length: torch.Tensor
    tensor_ids: tuple[int, int, int]
    compiled_tensor_ids: tuple[int, int, int]
    storage_pointers: tuple[int, int, int]
    max_cache_len: int


@dataclass(frozen=True)
class _GuardedLocalStaticCacheBinding:
    cache: Any
    layer: Any
    layer_ids: tuple[int, ...]
    keys: torch.Tensor
    values: torch.Tensor
    cumulative_length: torch.Tensor
    compiled_keys: torch.Tensor
    compiled_values: torch.Tensor
    compiled_length: torch.Tensor
    tensor_ids: tuple[int, int, int]
    compiled_tensor_ids: tuple[int, int, int]
    storage_pointers: tuple[int, int, int]
    max_cache_len: int


def _static_cache_tensor_versions(
    binding: _GuardedGlobalStaticCacheBinding | _GuardedLocalStaticCacheBinding,
) -> tuple[int | None, int | None, int | None]:
    return tuple(
        _tensor_version_or_none(tensor)
        for tensor in (binding.keys, binding.values, binding.cumulative_length)
    )


def _validate_static_cache_layer_classes(cache: Any) -> tuple[Any, ...]:
    if (
        _PINNED_STATIC_CACHE_CLASS is None
        or _PINNED_STATIC_LAYER_CLASS is None
        or _PINNED_STATIC_SLIDING_WINDOW_LAYER_CLASS is None
        or type(cache) is not _PINNED_STATIC_CACHE_CLASS
        or getattr(cache, "offloading", None) is not False
        or getattr(cache, "layer_class_to_replicate", object()) is not None
        or type(getattr(cache, "layers", None)) is not list
        or len(cache.layers) != GEMMA4_31B.num_hidden_layers
    ):
        raise UnsupportedH100Path(
            "EXP-0024 requires the exact pinned, non-offloaded StaticCache container"
        )
    for layer_idx, layer in enumerate(cache.layers):
        expected_class = (
            _PINNED_STATIC_LAYER_CLASS
            if GEMMA4_31B.spec_for_layer(layer_idx).kind == "full_attention"
            else _PINNED_STATIC_SLIDING_WINDOW_LAYER_CLASS
        )
        if type(layer) is not expected_class:
            raise UnsupportedH100Path(
                "EXP-0024 StaticCache layer classes conflict with the locked model"
            )
    return tuple(cache.layers)


def _bind_global_static_cache(
    module: Any,
    cache: Any,
    spec: AttentionLayerSpec,
) -> tuple[_GuardedGlobalStaticCacheBinding, int]:
    """Freeze the actual layer-5 StaticCache tensor identities before Dynamo."""

    if spec.kind != "full_attention" or getattr(module, "layer_idx", None) != 5:
        raise UnsupportedH100Path(
            "EXP-0024's first candidate accepts only pinned global layer 5"
        )
    layers = _validate_static_cache_layer_classes(cache)
    layer = layers[5]
    if (
        getattr(layer, "is_initialized", None) is not True
        or getattr(layer, "is_sliding", None) is not False
        or type(getattr(layer, "max_cache_len", None)) is not int
    ):
        raise UnsupportedH100Path(
            "EXP-0024 global decode requires an early-initialized pinned StaticLayer"
        )
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    cumulative_length = getattr(layer, "cumulative_length", None)
    if not all(isinstance(tensor, torch.Tensor) for tensor in (keys, values, cumulative_length)):
        raise UnsupportedH100Path("EXP-0024 global cache tensors are not initialized")
    compiled_keys = keys.view_as(keys)
    compiled_values = values.view_as(values)
    compiled_length = cumulative_length.reshape(())
    binding = _GuardedGlobalStaticCacheBinding(
        cache=cache,
        layer=layer,
        layer_ids=tuple(id(item) for item in layers),
        keys=keys,
        values=values,
        cumulative_length=cumulative_length,
        compiled_keys=compiled_keys,
        compiled_values=compiled_values,
        compiled_length=compiled_length,
        tensor_ids=(id(keys), id(values), id(cumulative_length)),
        compiled_tensor_ids=(
            id(compiled_keys),
            id(compiled_values),
            id(compiled_length),
        ),
        storage_pointers=tuple(
            tensor.untyped_storage().data_ptr()
            for tensor in (keys, values, cumulative_length)
        ),
        max_cache_len=layer.max_cache_len,
    )
    logical_length, _versions = _validate_global_static_cache_binding(binding)
    return binding, logical_length


def _bind_local_static_cache(
    module: Any,
    cache: Any,
    spec: AttentionLayerSpec,
) -> tuple[_GuardedLocalStaticCacheBinding, int]:
    """Freeze the actual layer-0 sliding-cache tensor identities before Dynamo."""

    if spec.kind != "sliding_attention" or getattr(module, "layer_idx", None) != 0:
        raise UnsupportedH100Path(
            "EXP-0028's first candidate accepts only pinned local layer 0"
        )
    layers = _validate_static_cache_layer_classes(cache)
    layer = layers[0]
    if (
        getattr(layer, "is_initialized", None) is not True
        or getattr(layer, "is_sliding", None) is not True
        or type(getattr(layer, "max_cache_len", None)) is not int
        or layer.max_cache_len != GEMMA4_31B.spec_for_layer(0).sliding_window
        or type(getattr(layer, "cumulative_length_int", None)) is not int
    ):
        raise UnsupportedH100Path(
            "EXP-0028 local decode requires an early-initialized pinned "
            "StaticSlidingWindowLayer"
        )
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    cumulative_length = getattr(layer, "cumulative_length", None)
    if not all(isinstance(tensor, torch.Tensor) for tensor in (keys, values, cumulative_length)):
        raise UnsupportedH100Path("EXP-0028 local sliding-cache tensors are not initialized")
    compiled_keys = keys.view_as(keys)
    compiled_values = values.view_as(values)
    compiled_length = cumulative_length.reshape(())
    binding = _GuardedLocalStaticCacheBinding(
        cache=cache,
        layer=layer,
        layer_ids=tuple(id(item) for item in layers),
        keys=keys,
        values=values,
        cumulative_length=cumulative_length,
        compiled_keys=compiled_keys,
        compiled_values=compiled_values,
        compiled_length=compiled_length,
        tensor_ids=(id(keys), id(values), id(cumulative_length)),
        compiled_tensor_ids=(
            id(compiled_keys),
            id(compiled_values),
            id(compiled_length),
        ),
        storage_pointers=tuple(
            tensor.untyped_storage().data_ptr()
            for tensor in (keys, values, cumulative_length)
        ),
        max_cache_len=layer.max_cache_len,
    )
    absolute_length, _tensor_length, _versions = _validate_local_static_cache_binding(
        binding
    )
    return binding, absolute_length


def _validate_global_static_cache_binding(
    binding: _GuardedGlobalStaticCacheBinding,
    *,
    expected_length: int | None = None,
    expected_versions: tuple[int | None, int | None, int | None] | None = None,
) -> tuple[int, tuple[int | None, int | None, int | None]]:
    """Revalidate container, tensors, addresses, geometry, and logical state."""

    layers = _validate_static_cache_layer_classes(binding.cache)
    if (
        tuple(id(item) for item in layers) != binding.layer_ids
        or layers[5] is not binding.layer
        or getattr(binding.layer, "is_initialized", None) is not True
        or getattr(binding.layer, "max_cache_len", None) != binding.max_cache_len
    ):
        raise UnsupportedH100Path("EXP-0024 StaticCache identity or capacity changed")
    live_tensors = (
        getattr(binding.layer, "keys", None),
        getattr(binding.layer, "values", None),
        getattr(binding.layer, "cumulative_length", None),
    )
    if (
        not all(isinstance(tensor, torch.Tensor) for tensor in live_tensors)
        or tuple(id(tensor) for tensor in live_tensors) != binding.tensor_ids
        or tuple(tensor.untyped_storage().data_ptr() for tensor in live_tensors)
        != binding.storage_pointers
    ):
        raise UnsupportedH100Path("EXP-0024 StaticCache storage was rebound or aliased")
    keys, values, cumulative_length = live_tensors
    compiled_tensors = (
        binding.compiled_keys,
        binding.compiled_values,
        binding.compiled_length,
    )
    if (
        tuple(id(tensor) for tensor in compiled_tensors)
        != binding.compiled_tensor_ids
        or any(
            getattr(root, "_dynamo_static_input_type", None) != "guarded"
            for root in live_tensors
        )
        or any(
            getattr(view, "_dynamo_static_input_type", None) is not None
            for view in compiled_tensors
        )
        or any(
            view.shape != root.shape
            or view.stride() != root.stride()
            or view.storage_offset() != root.storage_offset()
            or view.untyped_storage().data_ptr()
            != root.untyped_storage().data_ptr()
            for view, root in zip(compiled_tensors, live_tensors, strict=True)
        )
    ):
        raise UnsupportedH100Path(
            "EXP-0025 compiled cache views no longer cover their exact pinned roots"
        )
    if (
        keys.device.type != "cuda"
        or values.device != keys.device
        or cumulative_length.device != keys.device
        or keys.dtype != torch.bfloat16
        or values.dtype != torch.bfloat16
        or cumulative_length.dtype not in (torch.int32, torch.int64)
        or keys.shape != (1, 4, binding.max_cache_len, 512)
        or values.shape != keys.shape
        or cumulative_length.ndim != 0
        or not keys.is_contiguous()
        or not values.is_contiguous()
        or not cumulative_length.is_contiguous()
        or keys.requires_grad
        or values.requires_grad
        or cumulative_length.requires_grad
        or keys.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
        or getattr(binding.layer, "device", None) != keys.device
        or getattr(binding.layer, "dtype", None) != torch.bfloat16
        or getattr(binding.layer, "batch_size", None) != 1
        or getattr(binding.layer, "num_heads", None) != 4
        or getattr(binding.layer, "k_head_dim", None) != 512
        or getattr(binding.layer, "v_head_dim", None) != 512
    ):
        raise UnsupportedH100Path(
            "EXP-0024 StaticCache tensor geometry/device/dtype/layout is invalid"
        )
    logical_length = int(cumulative_length.detach().item())
    if (
        logical_length < 1
        or logical_length >= binding.max_cache_len
        or binding.max_cache_len > GEMMA4_31B.max_position_embeddings
    ):
        raise UnsupportedH100Path(
            "EXP-0024 global decode requires a nonempty prefix and spare cache capacity"
        )
    if expected_length is not None and logical_length != expected_length:
        raise UnsupportedH100Path(
            "EXP-0024 StaticCache logical length changed outside the guarded facade"
        )
    versions = _static_cache_tensor_versions(binding)
    if expected_versions is not None and any(
        expected is not None and current != expected
        for current, expected in zip(versions, expected_versions, strict=True)
    ):
        raise UnsupportedH100Path(
            "EXP-0024 StaticCache tensors were mutated outside the guarded facade"
        )
    return logical_length, versions


def _validate_local_static_cache_binding(
    binding: _GuardedLocalStaticCacheBinding,
    *,
    expected_absolute_length: int | None = None,
    expected_versions: tuple[int | None, int | None, int | None] | None = None,
) -> tuple[int, int, tuple[int | None, int | None, int | None]]:
    """Revalidate local sliding-cache identity and dual-counter state."""

    layers = _validate_static_cache_layer_classes(binding.cache)
    if (
        tuple(id(item) for item in layers) != binding.layer_ids
        or layers[0] is not binding.layer
        or getattr(binding.layer, "is_initialized", None) is not True
        or getattr(binding.layer, "is_sliding", None) is not True
        or getattr(binding.layer, "max_cache_len", None) != binding.max_cache_len
        or binding.max_cache_len != 1024
    ):
        raise UnsupportedH100Path(
            "EXP-0028 StaticSlidingWindow identity or capacity changed"
        )
    live_tensors = (
        getattr(binding.layer, "keys", None),
        getattr(binding.layer, "values", None),
        getattr(binding.layer, "cumulative_length", None),
    )
    if (
        not all(isinstance(tensor, torch.Tensor) for tensor in live_tensors)
        or tuple(id(tensor) for tensor in live_tensors) != binding.tensor_ids
        or tuple(tensor.untyped_storage().data_ptr() for tensor in live_tensors)
        != binding.storage_pointers
    ):
        raise UnsupportedH100Path(
            "EXP-0028 StaticSlidingWindow storage was rebound or aliased"
        )
    keys, values, cumulative_length = live_tensors
    compiled_tensors = (
        binding.compiled_keys,
        binding.compiled_values,
        binding.compiled_length,
    )
    if (
        tuple(id(tensor) for tensor in compiled_tensors)
        != binding.compiled_tensor_ids
        or any(
            getattr(root, "_dynamo_static_input_type", None) != "guarded"
            for root in live_tensors
        )
        or any(
            getattr(view, "_dynamo_static_input_type", None) is not None
            for view in compiled_tensors
        )
        or any(
            view.shape != root.shape
            or view.stride() != root.stride()
            or view.storage_offset() != root.storage_offset()
            or view.untyped_storage().data_ptr()
            != root.untyped_storage().data_ptr()
            for view, root in zip(compiled_tensors, live_tensors, strict=True)
        )
    ):
        raise UnsupportedH100Path(
            "EXP-0028 compiled cache views no longer cover their exact pinned roots"
        )
    absolute_length = getattr(binding.layer, "cumulative_length_int", None)
    if (
        type(absolute_length) is not int
        or absolute_length < 1
        or absolute_length >= GEMMA4_31B.max_position_embeddings
    ):
        raise UnsupportedH100Path(
            "EXP-0028 local decode requires a nonempty in-range Python absolute length"
        )
    if (
        keys.device.type != "cuda"
        or values.device != keys.device
        or cumulative_length.device != keys.device
        or keys.dtype != torch.bfloat16
        or values.dtype != torch.bfloat16
        or cumulative_length.dtype not in (torch.int32, torch.int64)
        or keys.shape != (1, 16, binding.max_cache_len, 256)
        or values.shape != keys.shape
        or cumulative_length.ndim != 0
        or not keys.is_contiguous()
        or not values.is_contiguous()
        or not cumulative_length.is_contiguous()
        or keys.requires_grad
        or values.requires_grad
        or cumulative_length.requires_grad
        or keys.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
        or getattr(binding.layer, "device", None) != keys.device
        or getattr(binding.layer, "dtype", None) != torch.bfloat16
        or getattr(binding.layer, "batch_size", None) != 1
        or getattr(binding.layer, "num_heads", None) != 16
        or getattr(binding.layer, "k_head_dim", None) != 256
        or getattr(binding.layer, "v_head_dim", None) != 256
    ):
        raise UnsupportedH100Path(
            "EXP-0028 StaticSlidingWindow tensor geometry/device/dtype/layout is invalid"
        )
    tensor_length = int(cumulative_length.detach().item())
    if tensor_length != min(absolute_length, binding.max_cache_len):
        raise UnsupportedH100Path(
            "EXP-0028 Python absolute length and saturated CUDA length disagree"
        )
    if expected_absolute_length is not None and absolute_length != expected_absolute_length:
        raise UnsupportedH100Path(
            "EXP-0028 local absolute length changed outside the guarded facade"
        )
    versions = _static_cache_tensor_versions(binding)
    if expected_versions is not None and any(
        expected is not None and current != expected
        for current, expected in zip(versions, expected_versions, strict=True)
    ):
        raise UnsupportedH100Path(
            "EXP-0028 local cache tensors were mutated outside the guarded facade"
        )
    return absolute_length, tensor_length, versions


def _validate_local_static_cache_post_op(
    binding: _GuardedLocalStaticCacheBinding,
    *,
    absolute_before: int,
    versions_before: tuple[int | None, int | None, int | None],
) -> tuple[int | None, int | None, int | None]:
    """Prove the K/V-only op left both eager-owned counters untouched."""

    if getattr(binding.layer, "cumulative_length_int", None) != absolute_before:
        raise RuntimeError("EXP-0028 compiled call mutated the Python absolute length")
    live_tensors = (
        getattr(binding.layer, "keys", None),
        getattr(binding.layer, "values", None),
        getattr(binding.layer, "cumulative_length", None),
    )
    if (
        not all(isinstance(tensor, torch.Tensor) for tensor in live_tensors)
        or tuple(id(tensor) for tensor in live_tensors) != binding.tensor_ids
        or tuple(tensor.untyped_storage().data_ptr() for tensor in live_tensors)
        != binding.storage_pointers
    ):
        raise RuntimeError("EXP-0028 compiled call rebound local cache storage")
    tensor_length = int(binding.cumulative_length.detach().item())
    expected_tensor_length = min(absolute_before, binding.max_cache_len)
    if tensor_length != expected_tensor_length:
        raise RuntimeError(
            "EXP-0028 K/V cache op mutated the eager-owned CUDA counter bytes"
        )
    versions_after = _static_cache_tensor_versions(binding)
    for name, before, after in zip(
        ("K", "V"),
        versions_before[:2],
        versions_after[:2],
        strict=True,
    ):
        if before is not None and after == before:
            raise RuntimeError(f"EXP-0028 cache op did not mutate declared {name}")
    before_counter = versions_before[2]
    after_counter = versions_after[2]
    if before_counter is not None and after_counter != before_counter:
        raise RuntimeError(
            "EXP-0028 K/V cache op changed the eager-owned CUDA counter version"
        )
    return versions_after


def _advance_local_static_cache_cuda_counter(
    binding: _GuardedLocalStaticCacheBinding,
    *,
    absolute_before: int,
    versions_before: tuple[int | None, int | None, int | None],
) -> tuple[int | None, int | None, int | None]:
    """Apply and prove the pinned underfill-only CUDA counter transition."""

    tensor_length_before = int(binding.cumulative_length.detach().item())
    if tensor_length_before != min(absolute_before, binding.max_cache_len):
        raise RuntimeError("EXP-0028 CUDA counter changed before the eager transaction")
    if absolute_before < binding.max_cache_len:
        binding.cumulative_length.add_(1)
    tensor_length_after = int(binding.cumulative_length.detach().item())
    if tensor_length_after != min(absolute_before + 1, binding.max_cache_len):
        raise RuntimeError("EXP-0028 eager CUDA counter transaction produced wrong bytes")
    versions_after = _static_cache_tensor_versions(binding)
    if versions_after[:2] != versions_before[:2]:
        raise RuntimeError("EXP-0028 eager counter transaction changed K/V versions")
    counter_before = versions_before[2]
    counter_after = versions_after[2]
    if counter_before is not None:
        counter_changed = counter_after != counter_before
        if counter_changed is not (absolute_before < binding.max_cache_len):
            raise RuntimeError(
                "EXP-0028 eager CUDA counter version disagrees with saturation state"
            )
    return versions_after


def _validate_static_cache_decode_weights(
    current: tuple[torch.Tensor, ...],
    frozen: tuple[torch.Tensor, ...],
    frozen_versions: tuple[int | None, ...],
) -> None:
    if len(current) != len(frozen) or any(
        live is not expected
        or live.untyped_storage().data_ptr()
        != expected.untyped_storage().data_ptr()
        or (
            version is not None
            and _tensor_version_or_none(live) != version
        )
        for live, expected, version in zip(
            current,
            frozen,
            frozen_versions,
            strict=True,
        )
    ):
        raise UnsupportedH100Path(
            "EXP-0024 module weights changed after facade construction"
        )


def _validate_global_static_cache_decode_inputs(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    binding: _GuardedGlobalStaticCacheBinding,
    logical_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail closed on the complete public Q1 boundary before compiled entry."""

    if torch.is_grad_enabled():
        raise UnsupportedH100Path("EXP-0024 guarded cache decode requires inference/no-grad")
    if type(position_embeddings) is not tuple or len(position_embeddings) != 2:
        raise UnsupportedH100Path("EXP-0024 requires the pinned cosine/sine tuple")
    cos, sin = position_embeddings
    if hidden_states.shape != (1, 1, GEMMA4_31B.hidden_size):
        raise UnsupportedH100Path("EXP-0024 global cache decode accepts only B1/Q1")
    if cos.shape != (1, 1, 512) or sin.shape != (1, 1, 512):
        raise UnsupportedH100Path("EXP-0024 global rotary tensors have the wrong shape")
    if position_ids.shape != (1, 1) or position_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise UnsupportedH100Path("EXP-0024 requires one explicit integer position")
    explicit_tensors = (
        hidden_states,
        cos,
        sin,
        position_ids,
        binding.compiled_keys,
        binding.compiled_values,
        binding.compiled_length,
        *weights,
    )
    if hidden_states.device.type != "cuda" or any(
        tensor.device != hidden_states.device for tensor in explicit_tensors
    ):
        raise UnsupportedH100Path("EXP-0024 tensors must share one CUDA device")
    if torch.cuda.get_device_capability(hidden_states.device) != (9, 0):
        raise UnsupportedH100Path("EXP-0024 requires an H100/SM90 device")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (
            hidden_states,
            cos,
            sin,
            binding.compiled_keys,
            binding.compiled_values,
            *weights,
        )
    ):
        raise UnsupportedH100Path("EXP-0024 activations, cache, and weights must use BF16")
    if any(tensor.requires_grad for tensor in (hidden_states, cos, sin)):
        raise UnsupportedH100Path("EXP-0024 activation tensors must not require grad")
    if any(not tensor.is_contiguous() for tensor in explicit_tensors):
        raise UnsupportedH100Path("EXP-0024 facade tensors must be contiguous")
    if int(position_ids.detach().item()) != logical_length:
        raise UnsupportedH100Path(
            "EXP-0024 position must equal the guarded StaticCache logical length"
        )
    if logical_length + 1 >= binding.max_cache_len:
        raise UnsupportedH100Path(
            "EXP-0024 global decode requires spare unwritten cache capacity"
        )
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_tensors}
    if len(storage_pointers) != len(explicit_tensors):
        raise UnsupportedH100Path("EXP-0024 facade tensors must use distinct storage")
    return cos, sin


def _guarded_global_static_cache_decode_tensor_only(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_length: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EXP-0024 inner frame: one cache-aware op and tensor arguments only."""

    return h100_global_static_cache_decode_fwd(
        hidden_states,
        cos,
        sin,
        position_ids,
        cache_k,
        cache_v,
        cache_length,
        q_proj_weight,
        k_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )


def _validate_local_static_cache_decode_inputs(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    binding: _GuardedLocalStaticCacheBinding,
    absolute_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail closed on EXP-0028's complete local Q1 boundary."""

    if torch.is_grad_enabled():
        raise UnsupportedH100Path("EXP-0028 guarded cache decode requires inference/no-grad")
    if type(position_embeddings) is not tuple or len(position_embeddings) != 2:
        raise UnsupportedH100Path("EXP-0028 requires the pinned cosine/sine tuple")
    cos, sin = position_embeddings
    if hidden_states.shape != (1, 1, GEMMA4_31B.hidden_size):
        raise UnsupportedH100Path("EXP-0028 local cache decode accepts only B1/Q1")
    if cos.shape != (1, 1, 256) or sin.shape != (1, 1, 256):
        raise UnsupportedH100Path("EXP-0028 local rotary tensors have the wrong shape")
    if position_ids.shape != (1, 1) or position_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise UnsupportedH100Path("EXP-0028 requires one explicit integer position")
    explicit_tensors = (
        hidden_states,
        cos,
        sin,
        position_ids,
        binding.compiled_keys,
        binding.compiled_values,
        binding.compiled_length,
        *weights,
    )
    if hidden_states.device.type != "cuda" or any(
        tensor.device != hidden_states.device for tensor in explicit_tensors
    ):
        raise UnsupportedH100Path("EXP-0028 tensors must share one CUDA device")
    if torch.cuda.get_device_capability(hidden_states.device) != (9, 0):
        raise UnsupportedH100Path("EXP-0028 requires an H100/SM90 device")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (
            hidden_states,
            cos,
            sin,
            binding.compiled_keys,
            binding.compiled_values,
            *weights,
        )
    ):
        raise UnsupportedH100Path("EXP-0028 activations, cache, and weights must use BF16")
    if any(tensor.requires_grad for tensor in (hidden_states, cos, sin)):
        raise UnsupportedH100Path("EXP-0028 activation tensors must not require grad")
    if any(not tensor.is_contiguous() for tensor in explicit_tensors):
        raise UnsupportedH100Path("EXP-0028 facade tensors must be contiguous")
    if int(position_ids.detach().item()) != absolute_length:
        raise UnsupportedH100Path(
            "EXP-0028 position must equal the guarded Python absolute length"
        )
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_tensors}
    if len(storage_pointers) != len(explicit_tensors):
        raise UnsupportedH100Path("EXP-0028 facade tensors must use distinct storage")
    return cos, sin


def _guarded_local_static_cache_decode_tensor_only(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_length: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EXP-0028 inner frame: one local cache op and tensor arguments only."""

    return h100_local_static_cache_decode_fwd(
        hidden_states,
        cos,
        sin,
        position_ids,
        cache_k,
        cache_v,
        cache_length,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )


class Gemma4H100CompiledStaticCacheDecodeFacade:
    """Guarded EXP-0024 decode facade; eager prefill remains outside Dynamo."""

    def __init__(
        self,
        module: Any,
        spec: AttentionLayerSpec,
        binding: _GuardedGlobalStaticCacheBinding,
        weights: tuple[torch.Tensor, ...],
        compiled_call: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        logical_length: int,
    ) -> None:
        self._module = module
        self._spec = spec
        self._binding = binding
        self._weights = weights
        self._weight_versions = tuple(_tensor_version_or_none(weight) for weight in weights)
        self._cache_versions = _static_cache_tensor_versions(binding)
        self._compiled_call = compiled_call
        self._expected_length = logical_length
        self._compiled_entry_count = 0
        self._last_lse: torch.Tensor | None = None

    @property
    def compiled_entry_count(self) -> int:
        return self._compiled_entry_count

    @property
    def last_lse(self) -> torch.Tensor | None:
        return self._last_lse

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if kwargs:
            raise UnsupportedH100Path(
                "EXP-0024 rejects unsupported decode metadata: "
                + ", ".join(sorted(kwargs))
            )
        weights = _validate_whole_layer_module(self._module, self._spec)
        _validate_static_cache_decode_weights(
            weights,
            self._weights,
            self._weight_versions,
        )
        _validate_guarded_facade_scalars(self._module)
        logical_length, versions = _validate_global_static_cache_binding(
            self._binding,
            expected_length=self._expected_length,
            expected_versions=self._cache_versions,
        )
        cos, sin = _validate_global_static_cache_decode_inputs(
            hidden_states,
            position_embeddings,
            position_ids,
            weights,
            self._binding,
            logical_length,
        )
        self._compiled_entry_count += 1
        output, lse = self._compiled_call(
            hidden_states,
            cos,
            sin,
            position_ids,
            self._binding.compiled_keys,
            self._binding.compiled_values,
            self._binding.compiled_length,
            *weights,
        )
        next_length, next_versions = _validate_global_static_cache_binding(
            self._binding,
            expected_length=logical_length + 1,
        )
        if any(
            before is not None and after == before
            for before, after in zip(versions, next_versions, strict=True)
        ):
            raise RuntimeError("EXP-0024 cache op did not mutate every declared tensor")
        self._expected_length = next_length
        self._cache_versions = next_versions
        self._last_lse = lse
        return output, None


class Gemma4H100CompiledLocalStaticCacheDecodeFacade:
    """Guarded EXP-0028 local decode facade with eager-owned Python state."""

    def __init__(
        self,
        module: Any,
        spec: AttentionLayerSpec,
        binding: _GuardedLocalStaticCacheBinding,
        weights: tuple[torch.Tensor, ...],
        compiled_call: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        absolute_length: int,
    ) -> None:
        self._module = module
        self._spec = spec
        self._binding = binding
        self._weights = weights
        self._weight_versions = tuple(_tensor_version_or_none(weight) for weight in weights)
        self._cache_versions = _static_cache_tensor_versions(binding)
        self._compiled_call = compiled_call
        self._expected_absolute_length = absolute_length
        self._compiled_entry_count = 0
        self._last_lse: torch.Tensor | None = None

    @property
    def compiled_entry_count(self) -> int:
        return self._compiled_entry_count

    @property
    def last_lse(self) -> torch.Tensor | None:
        return self._last_lse

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if kwargs:
            raise UnsupportedH100Path(
                "EXP-0028 rejects unsupported decode metadata: "
                + ", ".join(sorted(kwargs))
            )
        weights = _validate_whole_layer_module(self._module, self._spec)
        _validate_static_cache_decode_weights(
            weights,
            self._weights,
            self._weight_versions,
        )
        _validate_guarded_facade_scalars(self._module)
        absolute_length, _tensor_length, versions = _validate_local_static_cache_binding(
            self._binding,
            expected_absolute_length=self._expected_absolute_length,
            expected_versions=self._cache_versions,
        )
        cos, sin = _validate_local_static_cache_decode_inputs(
            hidden_states,
            position_embeddings,
            position_ids,
            weights,
            self._binding,
            absolute_length,
        )
        self._compiled_entry_count += 1
        output, lse = self._compiled_call(
            hidden_states,
            cos,
            sin,
            position_ids,
            self._binding.compiled_keys,
            self._binding.compiled_values,
            self._binding.compiled_length,
            *weights,
        )
        next_versions = _validate_local_static_cache_post_op(
            self._binding,
            absolute_before=absolute_length,
            versions_before=versions,
        )
        next_versions = _advance_local_static_cache_cuda_counter(
            self._binding,
            absolute_before=absolute_length,
            versions_before=next_versions,
        )
        self._binding.layer.cumulative_length_int = absolute_length + 1
        next_absolute_length, _next_tensor_length, validated_versions = (
            _validate_local_static_cache_binding(
                self._binding,
                expected_absolute_length=absolute_length + 1,
                expected_versions=next_versions,
            )
        )
        self._expected_absolute_length = next_absolute_length
        self._cache_versions = validated_versions
        self._last_lse = lse
        return output, None


def compile_gemma4_fa4_h100_static_cache_decode(
    module: Any,
    cache: Any,
    *,
    backend: str | Callable = "inductor",
) -> (
    Gemma4H100CompiledStaticCacheDecodeFacade
    | Gemma4H100CompiledLocalStaticCacheDecodeFacade
):
    """Build the guarded global or local pinned StaticCache decode facade."""

    register_gemma4_fa4_h100()
    if not CUSTOM_OPS_AVAILABLE:
        raise UnsupportedH100Path("EXP-0024 requires PyTorch custom_op and register_fake APIs")
    compile_api = getattr(torch, "compile", None)
    if not callable(compile_api):
        raise UnsupportedH100Path("EXP-0024 requires torch.compile")
    if not (backend in {"eager", "inductor"} if isinstance(backend, str) else callable(backend)):
        raise ValueError("EXP-0024 backend must be eager, inductor, or a callable delegate wrapper")
    spec = _spec_for_module(module)
    weights = _validate_whole_layer_module(module, spec)
    _validate_guarded_facade_scalars(module)
    if spec.kind == "full_attention":
        binding, logical_length = _bind_global_static_cache(module, cache, spec)
        if logical_length + 1 >= binding.max_cache_len:
            raise UnsupportedH100Path(
                "EXP-0024 global decode requires spare unwritten cache capacity"
            )
        inner = _guarded_global_static_cache_decode_tensor_only
        facade_type = Gemma4H100CompiledStaticCacheDecodeFacade
    else:
        binding, logical_length = _bind_local_static_cache(module, cache, spec)
        inner = _guarded_local_static_cache_decode_tensor_only
        facade_type = Gemma4H100CompiledLocalStaticCacheDecodeFacade
    if torch.cuda.get_device_capability(binding.keys.device) != (9, 0):
        raise UnsupportedH100Path("EXP-0024 requires an H100/SM90 device")
    explicit_storage = {
        tensor.untyped_storage().data_ptr()
        for tensor in (
            *weights,
            binding.compiled_keys,
            binding.compiled_values,
            binding.compiled_length,
        )
    }
    if len(explicit_storage) != len(weights) + 3:
        raise UnsupportedH100Path(
            "EXP-0024 module weights and cache tensors must use distinct storage"
        )
    compiled_call = compile_api(
        inner,
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    return facade_type(
        module,
        spec,
        binding,
        weights,
        compiled_call,
        logical_length,
    )


# The pinned Transformers patch discovers this hook only on the selected
# project attention interface.  Other registered attention functions and every
# eager invocation retain their original path.
gemma4_fa4_attention_forward._gemma4_fa4_compile_layer = gemma4_fa4_compile_layer


def _registered_value(registry: Any, key: str):
    try:
        return registry[key]
    except (KeyError, TypeError):
        return None


def register_gemma4_fa4_h100() -> str:
    """Register the Gemma-specific attention and mask entries idempotently."""

    global _PINNED_GEMMA4_TEXT_ATTENTION_CLASS
    global _PINNED_GEMMA4_TEXT_CONFIG_CLASS, _PINNED_MASKING_UTILS_MODULE
    global _PINNED_STATIC_CACHE_CLASS, _PINNED_STATIC_LAYER_CLASS
    global _PINNED_STATIC_SLIDING_WINDOW_LAYER_CLASS
    try:
        import transformers.masking_utils as masking_utils
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the pinned Transformers checkout is required for backend registration"
        ) from exc
    _PINNED_MASKING_UTILS_MODULE = masking_utils
    try:
        from transformers.cache_utils import (
            StaticCache,
            StaticLayer,
            StaticSlidingWindowLayer,
        )
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
    except Exception:
        # Lightweight registry doubles and older unsupported Transformers builds
        # can still exercise registration, but cannot mint a compiler origin.
        pass
    else:
        _PINNED_GEMMA4_TEXT_CONFIG_CLASS = Gemma4TextConfig
        _PINNED_GEMMA4_TEXT_ATTENTION_CLASS = Gemma4TextAttention
        _PINNED_STATIC_CACHE_CLASS = StaticCache
        _PINNED_STATIC_LAYER_CLASS = StaticLayer
        _PINNED_STATIC_SLIDING_WINDOW_LAYER_CLASS = StaticSlidingWindowLayer

    targets = (
        (ALL_ATTENTION_FUNCTIONS, gemma4_fa4_attention_forward, "attention"),
        (ALL_MASK_ATTENTION_FUNCTIONS, _registered_gemma4_fa4_mask, "mask"),
    )
    for registry, expected, label in targets:
        existing = _registered_value(registry, BACKEND_NAME)
        if existing is not None and existing is not expected:
            raise RuntimeError(
                f"Transformers {label} backend {BACKEND_NAME!r} is already registered"
            )
    for registry, expected, _label in targets:
        if _registered_value(registry, BACKEND_NAME) is None:
            registry.register(BACKEND_NAME, expected)
    return BACKEND_NAME


__all__ = [
    "BACKEND_NAME",
    "PINNED_TRANSFORMERS_REVISION",
    "Gemma4H100CompiledLayerFacade",
    "Gemma4H100CompiledLocalStaticCacheDecodeFacade",
    "Gemma4H100CompiledStaticCacheDecodeFacade",
    "Gemma4DispatchResult",
    "Gemma4MaskPlan",
    "gemma4_fa4_attention_forward",
    "gemma4_fa4_compile_layer",
    "gemma4_fa4_mask",
    "gemma4_fa4_prepared",
    "compile_gemma4_fa4_h100_layer",
    "compile_gemma4_fa4_h100_static_cache_decode",
    "register_gemma4_fa4_h100",
]
