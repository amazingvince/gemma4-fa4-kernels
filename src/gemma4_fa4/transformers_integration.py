"""Pinned Transformers integration for the Gemma 4 H100 FA4 paths.

The pinned Gemma implementation presents prepared Q/K/V as ``(B, H, S, D)``.
This module owns the model-specific conversion, packed-sequence construction,
and exact fallback policy.  It deliberately registers under a unique backend
name instead of changing Transformers' generic ``flash_attention_4`` entry.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from types import CodeType
from typing import Any

import torch

from .h100 import (
    GlobalBackwardBudgetExceeded,
    UnsupportedH100Path,
    fa4_global_forward_only,
    fa4_global_text_forward,
    fa4_global_varlen_forward,
    fa4_global_varlen_forward_only,
    fa4_local_forward,
    fa4_local_varlen_forward,
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
    **_kwargs,
) -> Gemma4MaskPlan:
    """Preserve the composed pinned-Transformers mask for FA4 or Flex routing."""

    return Gemma4MaskPlan(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
    )


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


def _fa4_layout_is_legal(*tensors: torch.Tensor) -> bool:
    """Host-side form of the pinned CuTe 128-bit outer-stride contract."""

    for tensor in tensors:
        if tensor.stride(-1) != 1:
            return False
        if any(stride <= 0 or stride % 8 for stride in tensor.stride()[:-1]):
            return False
        if torch._debug_has_internal_overlap(tensor) != 0:
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
            raise UnsupportedH100Path("empty padded sequences require exact FlexAttention fallback")
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
        values.append(values[-1] + length)
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
    if any(end <= start for start, end in zip(raw[:-1], raw[1:], strict=True)):
        raise ValueError(f"{name} must describe nonempty, increasing segments")
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

    derived_max_q = max(q_lengths)
    derived_max_k = max(k_lengths)
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
            if any(bool((torch.diff(values) != 0).any().item()) for values in documents):
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
        q_length <= 0 or q_length > k_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    ):
        raise UnsupportedH100Path("global composition requires 1 <= Sq <= Sk per segment")
    return q_lengths, k_lengths


def _run_global_composed(
    packed: _PackedLocalInputs,
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lengths, k_lengths = _global_segment_lengths(packed)
    if any(k_length > 2048 for k_length in k_lengths):
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
    if any(k_length > 2048 for k_length in k_lengths):
        raise UnsupportedH100Path(
            "EXP-0013 native global backward requires every K segment <= 2048"
        )
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

    q_bshd = query.transpose(1, 2)
    k_bshd = key.transpose(1, 2)
    v_bshd = value.transpose(1, 2)
    requires_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not _fa4_layout_is_legal(q_bshd, k_bshd, v_bshd):
        return _fallback_result(
            "prepared BHSD-to-BSHD view has an unsupported FA4 layout",
            module=module,
            q_bhsd=query,
            k_bhsd=key,
            v_bhsd=value,
            plan=plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )

    padding = _padding_mask(plan, query.device)
    has_padding = padding is not None and not bool(padding.all().detach().item())
    packed_positions = False
    if position_ids is not None and q_length == kv_length:
        if position_ids.shape != (batch_size, q_length):
            raise ValueError("position_ids must match the prepared Q sequence")
        for batch_idx in range(batch_size):
            valid = (
                torch.ones(q_length, dtype=torch.bool, device=query.device)
                if padding is None
                else padding[batch_idx]
            )
            values = position_ids[batch_idx, valid]
            if values.numel() > 1 and bool((torch.diff(values) != 1).any().detach().item()):
                packed_positions = True
                break
    has_explicit_cu = cu_seq_lens_q is not None or cu_seq_lens_k is not None
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
                    q_bhsd=query,
                    k_bhsd=key,
                    v_bhsd=value,
                    plan=plan,
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
                    q_bhsd=query,
                    k_bhsd=key,
                    v_bhsd=value,
                    plan=plan,
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
                position_ids=position_ids,
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
                    output, lse = _run_global_composed(packed, spec)
                    path = "fa4_global_varlen_composed_budget_fallback"
            else:
                q_lengths, k_lengths = _global_segment_lengths(packed)
                if all(k_length <= 1024 for k_length in k_lengths):
                    output, lse = _run_global_composed(packed, spec)
                    path = "fa4_global_varlen"
                else:
                    output, lse = _run_global_varlen_forward_only(packed, spec)
                    path = "fa4_global_varlen_forward_only"
        except UnsupportedH100Path as exc:
            return _fallback_result(
                str(exc),
                module=module,
                q_bhsd=query,
                k_bhsd=key,
                v_bhsd=value,
                plan=plan,
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
                q_bhsd=query,
                k_bhsd=key,
                v_bhsd=value,
                plan=plan,
                scaling=scaling,
                allow_flex_fallback=allow_flex_fallback,
            )
        return Gemma4DispatchResult(output=output, lse=lse, path="fa4_local_fixed")

    if missing_vision_metadata:
        return _fallback_result(
            "the exact mask admits future vision tokens but compact vision metadata is missing",
            module=module,
            q_bhsd=query,
            k_bhsd=key,
            v_bhsd=value,
            plan=plan,
            scaling=scaling,
            allow_flex_fallback=allow_flex_fallback,
        )

    try:
        packed = _pack_local_inputs(
            q_bshd,
            k_bshd,
            v_bshd,
            plan,
            position_ids=position_ids,
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
            q_bhsd=query,
            k_bhsd=key,
            v_bhsd=value,
            plan=plan,
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
    module._gemma4_fa4_last_path = result.path
    return result.output, None


def _registered_value(registry: Any, key: str):
    try:
        return registry[key]
    except (KeyError, TypeError):
        return None


def register_gemma4_fa4_h100() -> str:
    """Register the Gemma-specific attention and mask entries idempotently."""

    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the pinned Transformers checkout is required for backend registration"
        ) from exc

    targets = (
        (ALL_ATTENTION_FUNCTIONS, gemma4_fa4_attention_forward, "attention"),
        (ALL_MASK_ATTENTION_FUNCTIONS, gemma4_fa4_mask, "mask"),
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
    "Gemma4DispatchResult",
    "Gemma4MaskPlan",
    "gemma4_fa4_attention_forward",
    "gemma4_fa4_mask",
    "gemma4_fa4_prepared",
    "register_gemma4_fa4_h100",
]
