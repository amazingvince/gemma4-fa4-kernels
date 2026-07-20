#!/usr/bin/env python3
"""H100 correctness probe for the pinned Gemma 4 Transformers integration."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from gemma4_fa4.h100 import fa4_global_varlen_forward
from gemma4_fa4.masks import gemma4_attention_mask
from gemma4_fa4.model_spec import GLOBAL_ATTENTION, SLIDING_ATTENTION, AttentionLayerSpec
from gemma4_fa4.reference import reference_attention
from gemma4_fa4.transformers_integration import (
    BACKEND_NAME,
    Gemma4DispatchResult,
    Gemma4MaskPlan,
    gemma4_fa4_mask,
    gemma4_fa4_prepared,
    register_gemma4_fa4_h100,
)

LOCAL_OUTPUT_ATOL = 0.03125
LOCAL_OUTPUT_RTOL = 0.02
LOCAL_LSE_ATOL = 0.125
GLOBAL_OUTPUT_ATOL = 0.0625
GLOBAL_OUTPUT_RTOL = 0.03
GLOBAL_LSE_ATOL = 0.25

# Frozen EXP-0004 gradient policy. The BF16 baseline and quantization term are
# established independently of the candidate, so this is not an observed-error tolerance.
UPSTREAM_ERROR_MULTIPLIER = 2.0
QUANTIZATION_PERTURBATION = 0.3

STATIC_CACHE_CASES = (
    "static-cache-local-small",
    "static-cache-local-boundary",
    "static-cache-local-first-roll",
    "static-cache-global-small",
    "static-cache-global-k1025",
)

CASES = (
    "local-fixed-strided",
    "local-packed-padding",
    "local-lower-right",
    "global-fixed-strided",
    "global-varlen-batch",
    "global-packed-empty-row",
    "global-document-split-backward",
    "global-lower-right-long-backward",
    "global-packed-long-backward",
    "global-native-varlen-noncontiguous-gradients",
    "global-forward-only-long",
    "global-varlen-forward-only-long",
    "hf-mask-transport",
    *STATIC_CACHE_CASES,
)
MAX_CONTEXT_CASE = "global-forward-only-max-context"
PROBE_BACKEND_NAME = f"{BACKEND_NAME}_capture"
EAGER_PROBE_BACKEND_NAME = f"{BACKEND_NAME}_eager_capture"

ReferenceBuilder = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


class _AttentionModule(torch.nn.Module):
    """Minimal pinned-Transformers attention-interface owner."""

    def __init__(self, layer_idx: int, spec: AttentionLayerSpec) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = spec.kind
        self.is_sliding = spec.kind == "sliding_attention"
        self.head_dim = spec.head_dim_qk
        self.num_key_value_groups = spec.qhead_per_kvhead
        self.scaling = spec.softmax_scale
        self.sliding_window = spec.sliding_window


class _CaptureLanguageModel(torch.nn.Module):
    """Stop a tiny Gemma model immediately after its patched transport boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.received: dict[str, Any] | None = None

    def forward(self, **kwargs):
        self.received = kwargs
        hidden = kwargs["inputs_embeds"]
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            shared_kv_states=None,
        )


def _require_h100() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this probe case requires CUDA on an NVIDIA H100 (SM90)")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"this probe case requires SM90, found compute capability {capability}")


def _make_inputs(
    spec: AttentionLayerSpec,
    *,
    batch: int,
    q_length: int,
    kv_length: int | None = None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_length = q_length if kv_length is None else kv_length
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = (
        (batch, spec.num_q_heads, q_length, spec.head_dim_qk),
        (batch, spec.num_kv_heads, kv_length, spec.head_dim_qk),
        (batch, spec.num_kv_heads, kv_length, spec.head_dim_v),
    )
    values = []
    for shape in shapes:
        tensor = torch.randn(
            shape,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        values.append(tensor.requires_grad_(True))
    return values[0], values[1], values[2]


def _module_call(
    module: _AttentionModule,
    spec: AttentionLayerSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None = None,
    **kwargs,
) -> Gemma4DispatchResult:
    return gemma4_fa4_prepared(
        module,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0,
        scaling=1.0,
        sliding_window=spec.sliding_window,
        **kwargs,
    )


def _assert_transformers_view_is_zero_copy(tensor: torch.Tensor) -> list[int]:
    view = tensor.transpose(1, 2)
    if view.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
        raise AssertionError("BHSD-to-BSHD transpose allocated new storage")
    if view.storage_offset() != tensor.storage_offset():
        raise AssertionError("BHSD-to-BSHD transpose changed the storage offset")
    expected_stride = (
        tensor.stride(0),
        tensor.stride(2),
        tensor.stride(1),
        tensor.stride(3),
    )
    if view.stride() != expected_stride:
        raise AssertionError("BHSD-to-BSHD transpose produced unexpected strides")
    if view.is_contiguous():
        raise AssertionError("the canonical Transformers BSHD view unexpectedly became contiguous")
    if view.stride(-1) != 1 or any(stride % 8 for stride in view.stride()[:-1]):
        raise AssertionError("the zero-copy BSHD view violates the pinned FA4 stride contract")
    return list(view.stride())


def _assert_distinct_operands_and_gradients(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    q, k, v = inputs
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise AssertionError("prepared K and V must not alias")
    for name, gradient, source in zip(("dQ", "dK", "dV"), gradients, inputs, strict=True):
        if gradient.shape != source.shape or gradient.dtype != torch.bfloat16:
            raise AssertionError(f"{name} returned an invalid shape or dtype")
        if not torch.isfinite(gradient).all():
            raise AssertionError(f"{name} contains non-finite values")
    pointers = [gradient.untyped_storage().data_ptr() for gradient in gradients]
    if len(set(pointers)) != 3:
        raise AssertionError("dQ, dK, and dV must use distinct storage")


def _fixed_builder(
    spec: AttentionLayerSpec,
    *,
    q_start: int | None = None,
    upcast: torch.dtype,
) -> ReferenceBuilder:
    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        out, lse = reference_attention(
            q,
            k,
            v,
            softmax_scale=1.0,
            sliding_window=spec.sliding_window,
            allow_vision_bidirectional=False,
            q_start=q_start,
            upcast=upcast,
            return_lse=True,
        )
        return out.transpose(1, 2), lse

    return build


def _padded_builder(
    lengths: tuple[int, ...],
    spec: AttentionLayerSpec,
    *,
    upcast: torch.dtype,
) -> ReferenceBuilder:
    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        padded_length = q.shape[2]
        outputs = []
        lses = []
        for batch_idx, length in enumerate(lengths):
            if length == 0:
                outputs.append(q.new_zeros((1, padded_length, q.shape[1], v.shape[-1])))
                lses.append(
                    torch.full(
                        (1, q.shape[1], padded_length),
                        -torch.inf,
                        dtype=torch.float32,
                        device=q.device,
                    )
                )
                continue
            out_i, lse_i = reference_attention(
                q[batch_idx : batch_idx + 1, :, :length],
                k[batch_idx : batch_idx + 1, :, :length],
                v[batch_idx : batch_idx + 1, :, :length],
                softmax_scale=1.0,
                sliding_window=spec.sliding_window,
                allow_vision_bidirectional=False,
                upcast=upcast,
                return_lse=True,
            )
            out_i = out_i.transpose(1, 2)
            outputs.append(F.pad(out_i, (0, 0, 0, 0, 0, padded_length - length)))
            lses.append(F.pad(lse_i, (0, padded_length - length), value=float("-inf")))
        return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)

    return build


def _upstream_style_builder(
    spec: AttentionLayerSpec,
    *,
    q_start: int | None = None,
) -> ReferenceBuilder:
    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        k_expanded = torch.repeat_interleave(k, spec.qhead_per_kvhead, dim=1)
        v_expanded = torch.repeat_interleave(v, spec.qhead_per_kvhead, dim=1)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_expanded * 1.0)
        allowed = gemma4_attention_mask(
            batch_size=q.shape[0],
            q_len=q.shape[2],
            kv_len=k.shape[2],
            device=q.device,
            sliding_window=spec.sliding_window,
            q_start=q_start,
            allow_vision_bidirectional=False,
        )
        scores.masked_fill_(~allowed, float("-inf"))
        lse = torch.logsumexp(scores.float(), dim=-1)
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.einsum("bhqk,bhkd->bhqd", probabilities, v_expanded)
        return out.transpose(1, 2), lse

    return build


def _packed_rectangular_builder(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    spec: AttentionLayerSpec,
    *,
    upcast: torch.dtype | None,
) -> ReferenceBuilder:
    if len(q_lengths) != len(k_lengths):
        raise ValueError("packed Q/K lengths must have one-to-one segments")

    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        outputs = []
        lses = []
        q_segments = torch.split(q, q_lengths, dim=2)
        k_segments = torch.split(k, k_lengths, dim=2)
        v_segments = torch.split(v, k_lengths, dim=2)
        for q_segment, k_segment, v_segment, q_length, k_length in zip(
            q_segments,
            k_segments,
            v_segments,
            q_lengths,
            k_lengths,
            strict=True,
        ):
            builder = (
                _upstream_style_builder(spec, q_start=k_length - q_length)
                if upcast is None
                else _fixed_builder(
                    spec,
                    q_start=k_length - q_length,
                    upcast=upcast,
                )
            )
            output, lse = builder(q_segment, k_segment, v_segment)
            outputs.append(output)
            lses.append(lse)
        return torch.cat(outputs, dim=1), torch.cat(lses, dim=2)

    return build


def _packed_thd_builder(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    spec: AttentionLayerSpec,
    *,
    upcast: torch.dtype | None,
) -> ReferenceBuilder:
    """Adapt the independent packed reference to native THD tensors."""

    packed_builder = _packed_rectangular_builder(
        q_lengths,
        k_lengths,
        spec,
        upcast=upcast,
    )

    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        output, lse = packed_builder(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )
        return output.squeeze(0), lse.squeeze(0)

    return build


def _odd_padded_gradient_outputs(
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make unit-inner-stride gradient views with one odd padded column."""

    output_generator = torch.Generator(device=output.device).manual_seed(seed)
    lse_generator = torch.Generator(device=lse.device).manual_seed(seed + 1)
    dout_storage = torch.randn(
        (*output.shape[:-1], output.shape[-1] + 1),
        dtype=output.dtype,
        device=output.device,
        generator=output_generator,
    )
    dlse_storage = torch.randn(
        (*lse.shape[:-1], lse.shape[-1] + 1),
        dtype=lse.dtype,
        device=lse.device,
        generator=lse_generator,
    )
    dout = dout_storage[..., :-1]
    dlse = dlse_storage[..., :-1]
    if dout.is_contiguous() or dlse.is_contiguous():
        raise AssertionError("odd-padded gradient outputs must be noncontiguous views")
    if dout.stride(-1) != 1 or dlse.stride(-1) != 1:
        raise AssertionError("odd-padded gradient outputs must retain unit inner stride")
    return dout, dlse


def _padded_upstream_style_builder(
    lengths: tuple[int, ...],
    spec: AttentionLayerSpec,
) -> ReferenceBuilder:
    fixed = _upstream_style_builder(spec)

    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        padded_length = q.shape[2]
        outputs = []
        lses = []
        for batch_idx, length in enumerate(lengths):
            if length == 0:
                outputs.append(q.new_zeros((1, padded_length, q.shape[1], v.shape[-1])))
                lses.append(
                    torch.full(
                        (1, q.shape[1], padded_length),
                        -torch.inf,
                        dtype=torch.float32,
                        device=q.device,
                    )
                )
                continue
            out_i, lse_i = fixed(
                q[batch_idx : batch_idx + 1, :, :length],
                k[batch_idx : batch_idx + 1, :, :length],
                v[batch_idx : batch_idx + 1, :, :length],
            )
            outputs.append(F.pad(out_i, (0, 0, 0, 0, 0, padded_length - length)))
            lses.append(F.pad(lse_i, (0, padded_length - length), value=float("-inf")))
        return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)

    return build


def _differentiate(
    out: torch.Tensor,
    lse: torch.Tensor | None,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dout: torch.Tensor,
    dlse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if lse is None:
        return torch.autograd.grad(out, inputs, dout)
    return torch.autograd.grad((out, lse), inputs, (dout, dlse))


def _reference_run(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    builder: ReferenceBuilder,
    dout: torch.Tensor,
    dlse: torch.Tensor,
    *,
    include_lse: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    q, k, v = (tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    out, lse = builder(q, k, v)
    gradients = _differentiate(out, lse if include_lse else None, (q, k, v), dout, dlse)
    return out, lse, gradients


def _finite_error(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"shape mismatch: candidate={tuple(candidate.shape)} reference={tuple(reference.shape)}"
        )
    candidate_finite = torch.isfinite(candidate)
    reference_finite = torch.isfinite(reference)
    if not torch.equal(candidate_finite, reference_finite):
        raise AssertionError("candidate and reference have different finite-value masks")
    if not torch.equal(candidate[~candidate_finite], reference[~reference_finite]):
        raise AssertionError("candidate and reference have different non-finite sentinels")
    if not bool(candidate_finite.any()):
        return 0.0, 0.0
    error = (candidate[candidate_finite].float() - reference[reference_finite].float()).abs()
    return error.max().item(), error.mean().item()


def _assert_close(
    name: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    maximum, mean = _finite_error(candidate, reference)
    torch.testing.assert_close(
        candidate,
        reference,
        atol=atol,
        rtol=rtol,
        msg=(
            f"{name} exceeds the frozen numerical envelope "
            f"(max_abs={maximum:.9g}, mean_abs={mean:.9g}, atol={atol:.9g}, rtol={rtol:.9g})"
        ),
    )
    return {"max_abs": maximum, "mean_abs": mean}


def _quantization_atol(reference: torch.Tensor) -> float:
    roundtrip = (
        (reference + QUANTIZATION_PERTURBATION - QUANTIZATION_PERTURBATION) - reference
    ).abs()
    return 2.0 * roundtrip.max().item()


def _assert_gradients(
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bf16_baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, float]]:
    results = {}
    for name, actual, expected, baseline in zip(
        ("dQ", "dK", "dV"), candidate, reference, bf16_baseline, strict=True
    ):
        if not torch.isfinite(actual).all():
            raise AssertionError(f"{name} contains non-finite values")
        error = (actual.float() - expected.float()).abs()
        baseline_error = (baseline.float() - expected.float()).abs()
        maximum = error.max().item()
        baseline_maximum = baseline_error.max().item()
        quantization_atol = _quantization_atol(expected)
        limit = UPSTREAM_ERROR_MULTIPLIER * baseline_maximum + quantization_atol
        if maximum > limit:
            raise AssertionError(
                f"{name} max_abs={maximum:.8g} exceeds frozen EXP-0004 limit={limit:.8g}"
            )
        results[name] = {
            "max_abs": maximum,
            "mean_abs": error.mean().item(),
            "baseline_max_abs": baseline_maximum,
            "limit": limit,
        }
    return results


def _validate_kernel_result(
    *,
    case: str,
    result: Gemma4DispatchResult,
    expected_path: str,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference_builder: ReferenceBuilder,
    bf16_builder: ReferenceBuilder,
    seed: int,
    output_atol: float,
    output_rtol: float,
    lse_atol: float | None,
) -> dict[str, Any]:
    if result.path != expected_path:
        raise AssertionError(f"{case} routed to {result.path!r}, expected {expected_path!r}")
    if result.output.dtype != torch.bfloat16:
        raise AssertionError("attention output must be BF16")
    if lse_atol is None:
        if result.lse is not None:
            raise AssertionError("FlexAttention fallback must expose lse=None")
    else:
        expected_lse_shape = (inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2])
        if (
            result.lse is None
            or result.lse.dtype != torch.float32
            or result.lse.shape != expected_lse_shape
        ):
            raise AssertionError("FA4 path returned an invalid FP32 LSE contract")

    output_generator = torch.Generator(device="cuda").manual_seed(seed + 10_001)
    lse_generator = torch.Generator(device="cuda").manual_seed(seed + 20_003)
    dout = torch.randn(
        result.output.shape,
        dtype=torch.bfloat16,
        device="cuda",
        generator=output_generator,
    )
    dlse = torch.randn(
        (inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]),
        dtype=torch.float32,
        device="cuda",
        generator=lse_generator,
    )
    if dout.untyped_storage().data_ptr() == dlse.untyped_storage().data_ptr():
        raise AssertionError("dout and dlse must be independent allocations")

    gradients = _differentiate(result.output, result.lse, inputs, dout, dlse)
    _assert_distinct_operands_and_gradients(inputs, gradients)
    include_lse = result.lse is not None
    reference_out, reference_lse, reference_gradients = _reference_run(
        inputs,
        reference_builder,
        dout,
        dlse,
        include_lse=include_lse,
    )
    _, _, bf16_gradients = _reference_run(
        inputs,
        bf16_builder,
        dout,
        dlse,
        include_lse=include_lse,
    )

    record: dict[str, Any] = {
        "case": case,
        "path": result.path,
        "output": _assert_close(
            "O",
            result.output,
            reference_out,
            atol=output_atol,
            rtol=output_rtol,
        ),
        "gradients": _assert_gradients(gradients, reference_gradients, bf16_gradients),
        "gradient_sources": "independent_dout+dlse" if include_lse else "dout_only",
    }
    if lse_atol is not None:
        assert result.lse is not None
        record["lse"] = _assert_close(
            "LSE",
            result.lse,
            reference_lse,
            atol=lse_atol,
            rtol=0.0,
        )
        record["lse_dtype"] = str(result.lse.dtype)
    else:
        record["lse"] = None
    return record


def _probe_offset(value: int | torch.Tensor, *, name: str) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise AssertionError(f"{name} must be a real scalar integer")
    return int(value.detach().item())


def _capturing_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Gemma4MaskPlan | torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run the real adapter while retaining prepared O/LSE for the probe oracle."""

    if not isinstance(attention_mask, Gemma4MaskPlan):
        raise AssertionError("StaticCache probe requires the registered Gemma4MaskPlan")
    q_offset = _probe_offset(attention_mask.q_offset, name="q_offset")
    kv_offset = _probe_offset(attention_mask.kv_offset, name="kv_offset")
    active_k = q_offset + query.shape[2] - kv_offset
    if active_k <= 0 or active_k > key.shape[2]:
        raise AssertionError("StaticCache probe observed an invalid active K interval")

    result = gemma4_fa4_prepared(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )
    if result.lse is None:
        raise AssertionError("StaticCache FA4 probe path must expose FP32 LSE")
    module._gemma4_fa4_last_path = result.path
    module._gemma4_fa4_probe_capture = {
        "active_k": active_k,
        "grad_enabled": torch.is_grad_enabled(),
        "k": key[:, :, :active_k, :].detach().clone(),
        "kv_offset": kv_offset,
        "lse": result.lse.detach().clone(),
        "output": result.output.detach().clone(),
        "physical_k": key.shape[2],
        "q": query.detach().clone(),
        "q_offset": q_offset,
        "v": value[:, :, :active_k, :].detach().clone(),
    }
    return result.output, None


def _capturing_eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the pinned eager implementation and retain its prepared boundary."""

    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward

    output, weights = eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )
    module._gemma4_fa4_eager_probe_capture = {
        "grad_enabled": torch.is_grad_enabled(),
        "k": key.detach().clone(),
        "output": output.detach().clone(),
        "q": query.detach().clone(),
        "v": value.detach().clone(),
    }
    return output, weights


def _register_static_cache_probe_backend() -> None:
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:
        raise RuntimeError(
            "StaticCache cases require the pinned, patched Transformers checkout"
        ) from exc

    register_gemma4_fa4_h100()
    targets = (
        (
            ALL_ATTENTION_FUNCTIONS,
            _capturing_attention_forward,
            "attention",
            PROBE_BACKEND_NAME,
        ),
        (ALL_MASK_ATTENTION_FUNCTIONS, gemma4_fa4_mask, "mask", PROBE_BACKEND_NAME),
        (
            ALL_ATTENTION_FUNCTIONS,
            _capturing_eager_attention_forward,
            "eager attention",
            EAGER_PROBE_BACKEND_NAME,
        ),
        (
            ALL_MASK_ATTENTION_FUNCTIONS,
            eager_mask,
            "eager mask",
            EAGER_PROBE_BACKEND_NAME,
        ),
    )
    for registry, expected, label, backend_name in targets:
        try:
            existing = registry[backend_name]
        except (KeyError, TypeError):
            existing = None
        if existing is None:
            registry.register(backend_name, expected)
        elif existing is not expected:
            raise RuntimeError(
                f"Transformers {label} backend {backend_name!r} is already registered"
            )


def _locked_static_cache_config(implementation: str):
    try:
        from transformers import Gemma4TextConfig
    except Exception as exc:
        raise RuntimeError(
            "StaticCache cases require the pinned, patched Transformers checkout"
        ) from exc

    lock_path = Path(__file__).resolve().parents[1] / "configs/model/gemma4-31b.lock.json"
    locked_text = dict(json.loads(lock_path.read_text())["text_config"])
    locked_text["hidden_size"] = 64
    locked_text["intermediate_size"] = 128
    config = Gemma4TextConfig(**locked_text)
    config._attn_implementation = implementation
    return config


def _initialize_static_cache_layer(cache: Any, layer_idx: int, spec: AttentionLayerSpec):
    layer = cache.layers[layer_idx]
    if layer.is_initialized:
        raise AssertionError("StaticCache probe layer was initialized unexpectedly")
    empty_k = torch.empty(
        (1, spec.num_kv_heads, 0, spec.head_dim_qk),
        dtype=torch.bfloat16,
        device="cuda",
    )
    empty_v = torch.empty(
        (1, spec.num_kv_heads, 0, spec.head_dim_v),
        dtype=torch.bfloat16,
        device="cuda",
    )
    layer.lazy_initialization(empty_k, empty_v)
    return layer


def _fill_hostile_cache(layer: Any) -> None:
    """Mix finite and NaN sentinels so either K- or V-tail reads are observable."""

    layer.keys.fill_(61)
    layer.values.fill_(-47)
    layer.keys[:, :, ::2, :].fill_(float("nan"))
    layer.values[:, :, 1::2, :].fill_(float("nan"))


def _assert_exact_with_nan(actual: torch.Tensor, expected: torch.Tensor, *, label: str) -> None:
    if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
        raise AssertionError(f"{label} changed the NaN sentinel locations")
    actual_finite = torch.nan_to_num(actual.float())
    expected_finite = torch.nan_to_num(expected.float())
    if not torch.equal(actual_finite, expected_finite):
        raise AssertionError(f"{label} changed values outside the intended cache interval")


def _cache_pointers(layer: Any, *, label: str) -> tuple[int, int]:
    pointers = (
        layer.keys.untyped_storage().data_ptr(),
        layer.values.untyped_storage().data_ptr(),
    )
    if pointers[0] == pointers[1]:
        raise AssertionError(f"{label} K/V cache storage must remain distinct")
    return pointers


def _assert_static_cache_mutation(
    before: tuple[torch.Tensor, torch.Tensor],
    layer: Any,
    *,
    previous_active: int,
    active_k: int,
    rolled: bool,
    label: str,
) -> None:
    after = (layer.keys, layer.values)
    for operand, old, new in zip(("K", "V"), before, after, strict=True):
        if rolled:
            _assert_exact_with_nan(
                new[:, :, :-1, :],
                old[:, :, 1:, :],
                label=f"{label} rolled {operand}",
            )
            continue
        _assert_exact_with_nan(
            new[:, :, :previous_active, :],
            old[:, :, :previous_active, :],
            label=f"{label} retained {operand} prefix",
        )
        _assert_exact_with_nan(
            new[:, :, active_k:, :],
            old[:, :, active_k:, :],
            label=f"{label} unwritten {operand} tail",
        )


def _assert_static_matches_dynamic(
    static_layer: Any,
    dynamic_layer: Any,
    *,
    active_k: int,
    label: str,
) -> None:
    for operand, static, dynamic in (
        ("K", static_layer.keys, dynamic_layer.keys),
        ("V", static_layer.values, dynamic_layer.values),
    ):
        dynamic_length = dynamic.shape[2]
        if dynamic_length <= 0 or dynamic_length > active_k:
            raise AssertionError(f"{label} eager oracle returned an invalid {operand} length")
        expected = static[:, :, active_k - dynamic_length : active_k, :]
        torch.testing.assert_close(dynamic, expected, atol=0.0, rtol=0.0)


def _validate_static_cache_capture(
    module: Any,
    spec: AttentionLayerSpec,
    *,
    expected_path: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | int | bool]]:
    capture = getattr(module, "_gemma4_fa4_probe_capture", None)
    if not isinstance(capture, dict):
        raise AssertionError(f"{label} did not retain the prepared FA4 result")
    if getattr(module, "_gemma4_fa4_last_path", None) != expected_path:
        raise AssertionError(
            f"{label} routed to {getattr(module, '_gemma4_fa4_last_path', None)!r}, "
            f"expected {expected_path!r}"
        )
    if capture["grad_enabled"]:
        raise AssertionError(f"{label} ran outside torch.inference_mode()")

    q = capture["q"]
    k = capture["k"]
    v = capture["v"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    reference_output, reference_lse = _fixed_builder(
        spec,
        q_start=k.shape[2] - q.shape[2],
        upcast=torch.float32,
    )(q, k, v)
    output_atol = LOCAL_OUTPUT_ATOL if spec.kind == "sliding_attention" else GLOBAL_OUTPUT_ATOL
    output_rtol = LOCAL_OUTPUT_RTOL if spec.kind == "sliding_attention" else GLOBAL_OUTPUT_RTOL
    lse_atol = LOCAL_LSE_ATOL if spec.kind == "sliding_attention" else GLOBAL_LSE_ATOL
    prepared_output = capture["output"]
    prepared_lse = capture["lse"]
    assert isinstance(prepared_output, torch.Tensor)
    assert isinstance(prepared_lse, torch.Tensor)
    record = {
        "output": _assert_close(
            f"{label} prepared O",
            prepared_output,
            reference_output,
            atol=output_atol,
            rtol=output_rtol,
        ),
        "lse": _assert_close(
            f"{label} prepared LSE",
            prepared_lse,
            reference_lse,
            atol=lse_atol,
            rtol=0.0,
        ),
    }
    return record, capture


def _assert_projection_transport(
    module: Any,
    prepared_output: torch.Tensor,
    layer_output: torch.Tensor,
    *,
    label: str,
) -> None:
    replay = module.o_proj(
        prepared_output.reshape(prepared_output.shape[0], prepared_output.shape[1], -1).contiguous()
    )
    if not torch.equal(replay, layer_output):
        maximum, mean = _finite_error(replay, layer_output)
        raise AssertionError(
            f"{label} did not preserve the captured prepared O through o_proj "
            f"(max_abs={maximum:.9g}, mean_abs={mean:.9g})"
        )


def _validate_eager_cache_capture(
    module: Any,
    spec: AttentionLayerSpec,
    layer_output: torch.Tensor,
    expected_prepared: dict[str, torch.Tensor | int | bool],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | bool]]:
    capture = getattr(module, "_gemma4_fa4_eager_probe_capture", None)
    if not isinstance(capture, dict):
        raise AssertionError(f"{label} did not retain the pinned eager prepared result")
    if capture["grad_enabled"]:
        raise AssertionError(f"{label} ran outside torch.inference_mode()")
    for operand in ("q", "k", "v"):
        actual = capture[operand]
        expected = expected_prepared[operand]
        assert isinstance(actual, torch.Tensor)
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    q = capture["q"]
    k = capture["k"]
    v = capture["v"]
    prepared_output = capture["output"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(prepared_output, torch.Tensor)
    reference_output, _reference_lse = _fixed_builder(
        spec,
        q_start=k.shape[2] - q.shape[2],
        upcast=torch.float32,
    )(q, k, v)
    reference_maximum, reference_mean = _finite_error(prepared_output, reference_output)
    _assert_projection_transport(
        module,
        prepared_output,
        layer_output,
        label=f"{label} eager layer",
    )
    return (
        {
            "output_vs_project_reference": {
                "max_abs": reference_maximum,
                "mean_abs": reference_mean,
            },
            "projection_transport_exact": True,
        },
        capture,
    )


def _run_static_cache_sequence(
    *,
    case: str,
    spec: AttentionLayerSpec,
    layer_idx: int,
    capacity: int,
    prompt_length: int,
    step_names: tuple[str, ...],
    expected_paths: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    _require_h100()
    if len(step_names) != len(expected_paths) or not step_names:
        raise ValueError("StaticCache step names and paths must be nonempty and aligned")
    if len(step_names) > 1 and any(name == "prefill" for name in step_names[1:]):
        raise ValueError("StaticCache probe accepts exactly one leading prefill")

    try:
        from transformers import DynamicCache, StaticCache
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextAttention,
            Gemma4TextRotaryEmbedding,
        )
    except Exception as exc:
        raise RuntimeError(
            "StaticCache cases require the pinned, patched Transformers checkout"
        ) from exc

    _register_static_cache_probe_backend()
    candidate_config = _locked_static_cache_config(PROBE_BACKEND_NAME)
    oracle_config = _locked_static_cache_config(EAGER_PROBE_BACKEND_NAME)
    torch.manual_seed(seed)
    candidate = Gemma4TextAttention(candidate_config, layer_idx=layer_idx).to(
        device="cuda", dtype=torch.bfloat16
    )
    hostile_candidate = Gemma4TextAttention(candidate_config, layer_idx=layer_idx).to(
        device="cuda", dtype=torch.bfloat16
    )
    oracle = Gemma4TextAttention(oracle_config, layer_idx=layer_idx).to(
        device="cuda", dtype=torch.bfloat16
    )
    rotary = Gemma4TextRotaryEmbedding(candidate_config, device="cuda").to(device="cuda")
    hostile_candidate.load_state_dict(candidate.state_dict())
    oracle.load_state_dict(candidate.state_dict())
    candidate.eval()
    hostile_candidate.eval()
    oracle.eval()

    candidate_cache = StaticCache(config=candidate_config, max_cache_len=capacity)
    hostile_cache = StaticCache(config=candidate_config, max_cache_len=capacity)
    oracle_cache = DynamicCache(config=oracle_config)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    mask_builder = (
        create_sliding_window_causal_mask
        if spec.kind == "sliding_attention"
        else create_causal_mask
    )

    with torch.inference_mode():
        candidate_layer = _initialize_static_cache_layer(candidate_cache, layer_idx, spec)
        hostile_layer = _initialize_static_cache_layer(hostile_cache, layer_idx, spec)
        _fill_hostile_cache(hostile_layer)
        candidate_pointers = _cache_pointers(candidate_layer, label="candidate")
        hostile_pointers = _cache_pointers(hostile_layer, label="hostile candidate")
        physical_k = candidate_layer.keys.shape[2]
        if physical_k != hostile_layer.keys.shape[2]:
            raise AssertionError("clean and hostile StaticCache capacities differ")

        step_records: list[dict[str, Any]] = []
        seen_tokens = 0
        lengths = (prompt_length, *(1 for _ in step_names[1:]))
        for step_name, q_length, expected_path in zip(
            step_names, lengths, expected_paths, strict=True
        ):
            positions = torch.arange(
                seen_tokens,
                seen_tokens + q_length,
                dtype=torch.long,
                device="cuda",
            ).unsqueeze(0)
            hidden = torch.randn(
                (1, q_length, candidate_config.hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            cos, sin = rotary(hidden, positions, layer_type=spec.kind)
            candidate_mask = mask_builder(
                candidate_config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=candidate_cache,
                position_ids=positions,
                layer_idx=layer_idx,
            )
            hostile_mask = mask_builder(
                candidate_config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=hostile_cache,
                position_ids=positions,
                layer_idx=layer_idx,
            )
            oracle_mask = mask_builder(
                oracle_config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=oracle_cache,
                position_ids=positions,
                layer_idx=layer_idx,
            )
            if not isinstance(candidate_mask, Gemma4MaskPlan) or not isinstance(
                hostile_mask, Gemma4MaskPlan
            ):
                raise AssertionError("registered StaticCache mask backend did not return a plan")

            candidate_before = (candidate_layer.keys.clone(), candidate_layer.values.clone())
            hostile_before = (hostile_layer.keys.clone(), hostile_layer.values.clone())
            candidate_output, candidate_weights = candidate(
                hidden,
                (cos, sin),
                candidate_mask,
                {},
                past_key_values=candidate_cache,
                position_ids=positions,
                allow_flex_fallback=False,
            )
            hostile_output, hostile_weights = hostile_candidate(
                hidden,
                (cos, sin),
                hostile_mask,
                {},
                past_key_values=hostile_cache,
                position_ids=positions,
                allow_flex_fallback=False,
            )
            oracle_output, _oracle_weights = oracle(
                hidden,
                (cos, sin),
                oracle_mask,
                {},
                past_key_values=oracle_cache,
                position_ids=positions,
            )
            if candidate_weights is not None or hostile_weights is not None:
                raise AssertionError(
                    "registered FA4 attention backend unexpectedly returned weights"
                )
            if candidate_output.shape != hidden.shape or hostile_output.shape != hidden.shape:
                raise AssertionError(
                    "StaticCache Gemma4TextAttention returned an invalid output shape"
                )

            candidate_reference, candidate_capture = _validate_static_cache_capture(
                candidate,
                spec,
                expected_path=expected_path,
                label=f"{case}/{step_name}",
            )
            hostile_reference, hostile_capture = _validate_static_cache_capture(
                hostile_candidate,
                spec,
                expected_path=expected_path,
                label=f"{case}/{step_name}/hostile",
            )
            _assert_projection_transport(
                candidate,
                candidate_capture["output"],
                candidate_output,
                label=f"{case}/{step_name} FA4 layer",
            )
            _assert_projection_transport(
                hostile_candidate,
                hostile_capture["output"],
                hostile_output,
                label=f"{case}/{step_name} hostile FA4 layer",
            )
            eager_reference, eager_capture = _validate_eager_cache_capture(
                oracle,
                spec,
                oracle_output,
                candidate_capture,
                label=f"{case}/{step_name}",
            )
            eager_prepared_output = eager_capture["output"]
            candidate_prepared_output = candidate_capture["output"]
            assert isinstance(eager_prepared_output, torch.Tensor)
            assert isinstance(candidate_prepared_output, torch.Tensor)
            prepared_cross_maximum, prepared_cross_mean = _finite_error(
                candidate_prepared_output,
                eager_prepared_output,
            )
            active_k = min(seen_tokens + q_length, physical_k)
            if candidate_capture["active_k"] != active_k:
                raise AssertionError("captured active K length disagrees with cache progression")
            if hostile_capture["active_k"] != active_k:
                raise AssertionError("hostile active K length disagrees with cache progression")
            if candidate_capture["physical_k"] != physical_k:
                raise AssertionError(
                    "attention backend did not receive the physical StaticCache backing"
                )
            expected_kv_offset = (
                max(seen_tokens - physical_k + 1, 0) if spec.kind == "sliding_attention" else 0
            )
            if candidate_capture["q_offset"] != seen_tokens:
                raise AssertionError("StaticCache query offset was not snapshotted before update")
            if candidate_capture["kv_offset"] != expected_kv_offset:
                raise AssertionError("StaticCache K offset disagrees with the pinned cache order")
            for operand in ("q", "k", "v", "output", "lse"):
                clean = candidate_capture[operand]
                hostile = hostile_capture[operand]
                assert isinstance(clean, torch.Tensor)
                assert isinstance(hostile, torch.Tensor)
                if not torch.equal(clean, hostile):
                    raise AssertionError(
                        f"hostile unwritten cache tail changed prepared {operand} at {step_name}"
                    )
            if not torch.equal(candidate_output, hostile_output):
                raise AssertionError(f"hostile unwritten cache tail changed layer O at {step_name}")

            rolled = spec.kind == "sliding_attention" and seen_tokens >= physical_k
            _assert_static_cache_mutation(
                candidate_before,
                candidate_layer,
                previous_active=min(seen_tokens, physical_k),
                active_k=active_k,
                rolled=rolled,
                label=f"{case}/{step_name}/clean",
            )
            _assert_static_cache_mutation(
                hostile_before,
                hostile_layer,
                previous_active=min(seen_tokens, physical_k),
                active_k=active_k,
                rolled=rolled,
                label=f"{case}/{step_name}/hostile",
            )
            if _cache_pointers(candidate_layer, label="candidate") != candidate_pointers:
                raise AssertionError("candidate StaticCache K/V addresses changed")
            if _cache_pointers(hostile_layer, label="hostile candidate") != hostile_pointers:
                raise AssertionError("hostile StaticCache K/V addresses changed")
            for operand, clean, hostile in (
                ("K", candidate_layer.keys, hostile_layer.keys),
                ("V", candidate_layer.values, hostile_layer.values),
            ):
                if not torch.equal(clean[:, :, :active_k, :], hostile[:, :, :active_k, :]):
                    raise AssertionError(f"hostile tail changed the active {operand} cache prefix")
            _assert_static_matches_dynamic(
                candidate_layer,
                oracle_cache.layers[layer_idx],
                active_k=active_k,
                label=f"{case}/{step_name}",
            )

            layer_maximum, layer_mean = _finite_error(candidate_output, oracle_output)
            step_records.append(
                {
                    "active_k": active_k,
                    "hostile_prepared_reference": hostile_reference,
                    "hostile_tail_slots": physical_k - active_k,
                    "eager_prepared_reference": eager_reference,
                    "prepared_candidate_vs_eager": {
                        "max_abs": prepared_cross_maximum,
                        "mean_abs": prepared_cross_mean,
                    },
                    "layer_output": {
                        "candidate_vs_eager_max_abs": layer_maximum,
                        "candidate_vs_eager_mean_abs": layer_mean,
                        "projection_transport_exact": True,
                    },
                    "path": expected_path,
                    "physical_k": physical_k,
                    "prepared_reference": candidate_reference,
                    "q_length": q_length,
                    "q_offset": candidate_capture["q_offset"],
                    "kv_offset": candidate_capture["kv_offset"],
                    "rolled": rolled,
                    "step": step_name,
                }
            )
            seen_tokens += q_length

    return {
        "actual_gemma4_text_attention": True,
        "cache_addresses_stable": True,
        "cache_kv_storage_distinct": True,
        "case": case,
        "hostile_finite_and_nan_tail_isolation": True,
        "inference_mode": True,
        "oracle": "pinned-eager-dynamic-cache+project-reference",
        "paths": [step["path"] for step in step_records],
        "pinned_position_embeddings": True,
        "steps": step_records,
    }


def _run_static_cache_local_boundary(seed: int) -> dict[str, Any]:
    return _run_static_cache_sequence(
        case="static-cache-local-boundary",
        spec=SLIDING_ATTENTION,
        layer_idx=0,
        capacity=1024,
        prompt_length=1023,
        step_names=("prefill", "boundary_decode"),
        expected_paths=("fa4_local_fixed", "fa4_local_varlen"),
        seed=seed,
    )


def _run_static_cache_local_small(seed: int) -> dict[str, Any]:
    return _run_static_cache_sequence(
        case="static-cache-local-small",
        spec=SLIDING_ATTENTION,
        layer_idx=0,
        capacity=1024,
        prompt_length=32,
        step_names=("prefill", "decode"),
        expected_paths=("fa4_local_fixed", "fa4_local_varlen"),
        seed=seed,
    )


def _run_static_cache_local_first_roll(seed: int) -> dict[str, Any]:
    return _run_static_cache_sequence(
        case="static-cache-local-first-roll",
        spec=SLIDING_ATTENTION,
        layer_idx=0,
        capacity=1024,
        prompt_length=1023,
        step_names=("prefill", "boundary_decode", "first_roll"),
        expected_paths=("fa4_local_fixed", "fa4_local_varlen", "fa4_local_varlen"),
        seed=seed,
    )


def _run_static_cache_global_small(seed: int) -> dict[str, Any]:
    return _run_static_cache_sequence(
        case="static-cache-global-small",
        spec=GLOBAL_ATTENTION,
        layer_idx=5,
        capacity=65,
        prompt_length=32,
        step_names=("prefill", "decode"),
        expected_paths=("fa4_global_fixed", "fa4_global_varlen"),
        seed=seed,
    )


def _run_static_cache_global_k1025(seed: int) -> dict[str, Any]:
    return _run_static_cache_sequence(
        case="static-cache-global-k1025",
        spec=GLOBAL_ATTENTION,
        layer_idx=5,
        capacity=1026,
        prompt_length=1024,
        step_names=("prefill", "decode"),
        expected_paths=("fa4_global_fixed", "fa4_global_forward_only"),
        seed=seed,
    )


def _run_local_fixed_strided(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = SLIDING_ATTENTION
    inputs = _make_inputs(spec, batch=1, q_length=65, seed=seed)
    strides = [_assert_transformers_view_is_zero_copy(tensor) for tensor in inputs]
    result = _module_call(
        _AttentionModule(0, spec),
        spec,
        *inputs,
        allow_flex_fallback=False,
    )
    record = _validate_kernel_result(
        case="local-fixed-strided",
        result=result,
        expected_path="fa4_local_fixed",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, upcast=torch.float32),
        bf16_builder=_upstream_style_builder(spec),
        seed=seed,
        output_atol=LOCAL_OUTPUT_ATOL,
        output_rtol=LOCAL_OUTPUT_RTOL,
        lse_atol=LOCAL_LSE_ATOL,
    )
    record["bshd_view_strides"] = strides
    record["zero_copy"] = True
    return record


def _run_local_packed_padding(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = SLIDING_ATTENTION
    lengths = (6, 3)
    padded_length = 7
    inputs = _make_inputs(spec, batch=2, q_length=padded_length, seed=seed)
    padding = torch.zeros((2, padded_length), dtype=torch.int32, device="cuda")
    for batch_idx, length in enumerate(lengths):
        padding[batch_idx, :length] = 1
    plan = Gemma4MaskPlan(
        batch_size=2,
        q_length=padded_length,
        kv_length=padded_length,
        q_offset=0,
        kv_offset=0,
        mask_function=None,
        attention_mask=padding,
    )
    result = _module_call(
        _AttentionModule(0, spec),
        spec,
        *inputs,
        attention_mask=plan,
        allow_flex_fallback=False,
    )
    record = _validate_kernel_result(
        case="local-packed-padding",
        result=result,
        expected_path="fa4_local_varlen",
        inputs=inputs,
        reference_builder=_padded_builder(lengths, spec, upcast=torch.float32),
        bf16_builder=_padded_upstream_style_builder(lengths, spec),
        seed=seed,
        output_atol=LOCAL_OUTPUT_ATOL,
        output_rtol=LOCAL_OUTPUT_RTOL,
        lse_atol=LOCAL_LSE_ATOL,
    )
    if torch.count_nonzero(result.output[0, lengths[0] :]).item() != 0:
        raise AssertionError("first padded output tail must be exactly zero")
    if torch.count_nonzero(result.output[1, lengths[1] :]).item() != 0:
        raise AssertionError("second padded output tail must be exactly zero")
    assert result.lse is not None
    if not torch.isneginf(result.lse[0, :, lengths[0] :]).all():
        raise AssertionError("first padded LSE tail must be -inf")
    if not torch.isneginf(result.lse[1, :, lengths[1] :]).all():
        raise AssertionError("second padded LSE tail must be -inf")
    record["packed_lengths"] = list(lengths)
    record["padding_sentinels"] = "zero_O/-inf_LSE"
    return record


def _run_local_lower_right(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = SLIDING_ATTENTION
    q_length, kv_length = 3, 9
    q_start = kv_length - q_length
    inputs = _make_inputs(
        spec,
        batch=1,
        q_length=q_length,
        kv_length=kv_length,
        seed=seed,
    )
    strides = [_assert_transformers_view_is_zero_copy(tensor) for tensor in inputs]
    cu_q = torch.tensor([0, q_length], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, kv_length], dtype=torch.int32, device="cuda")
    plan = Gemma4MaskPlan(
        batch_size=1,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_start,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
    )
    result = _module_call(
        _AttentionModule(0, spec),
        spec,
        *inputs,
        attention_mask=plan,
        position_ids=torch.arange(q_start, kv_length, device="cuda").unsqueeze(0),
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=q_length,
        max_length_k=kv_length,
        allow_flex_fallback=False,
    )
    record = _validate_kernel_result(
        case="local-lower-right",
        result=result,
        expected_path="fa4_local_varlen",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, q_start=q_start, upcast=torch.float32),
        bf16_builder=_upstream_style_builder(spec, q_start=q_start),
        seed=seed,
        output_atol=LOCAL_OUTPUT_ATOL,
        output_rtol=LOCAL_OUTPUT_RTOL,
        lse_atol=LOCAL_LSE_ATOL,
    )
    record["q_offset"] = q_start
    record["bshd_view_strides"] = strides
    record["zero_copy"] = True
    return record


def _run_global_fixed_strided(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    inputs = _make_inputs(spec, batch=1, q_length=33, seed=seed)
    strides = [_assert_transformers_view_is_zero_copy(tensor) for tensor in inputs]
    result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *inputs,
        allow_flex_fallback=False,
    )
    record = _validate_kernel_result(
        case="global-fixed-strided",
        result=result,
        expected_path="fa4_global_fixed",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, upcast=torch.float32),
        bf16_builder=_upstream_style_builder(spec),
        seed=seed,
        output_atol=GLOBAL_OUTPUT_ATOL,
        output_rtol=GLOBAL_OUTPUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )
    record["bshd_view_strides"] = strides
    record["zero_copy"] = True
    return record


def _run_global_varlen_batch(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    inputs = _make_inputs(spec, batch=2, q_length=5, seed=seed)
    module = _AttentionModule(5, spec)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Gemma 4 H100 FA4 fallback:*",
            category=RuntimeWarning,
        )
        result = _module_call(module, spec, *inputs, allow_flex_fallback=True)
    record = _validate_kernel_result(
        case="global-varlen-batch",
        result=result,
        expected_path="fa4_global_varlen_native",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, upcast=torch.float32),
        bf16_builder=_upstream_style_builder(spec),
        seed=seed,
        output_atol=GLOBAL_OUTPUT_ATOL,
        output_rtol=GLOBAL_OUTPUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )
    return record


def _run_global_packed_empty_row(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    lengths = (0, 65)
    padded_length = max(lengths)
    inputs = _make_inputs(spec, batch=2, q_length=padded_length, seed=seed)
    padding = torch.zeros((2, padded_length), dtype=torch.int32, device=inputs[0].device)
    padding[1, : lengths[1]] = 1
    plan = Gemma4MaskPlan(
        batch_size=2,
        q_length=padded_length,
        kv_length=padded_length,
        q_offset=0,
        kv_offset=0,
        mask_function=None,
        attention_mask=padding,
    )
    call_kwargs = {"attention_mask": plan, "allow_flex_fallback": False}
    result = _module_call(_AttentionModule(5, spec), spec, *inputs, **call_kwargs)
    record = _validate_kernel_result(
        case="global-packed-empty-row",
        result=result,
        expected_path="fa4_global_varlen_native",
        inputs=inputs,
        reference_builder=_padded_builder(lengths, spec, upcast=torch.float32),
        bf16_builder=_padded_upstream_style_builder(lengths, spec),
        seed=seed,
        output_atol=GLOBAL_OUTPUT_ATOL,
        output_rtol=GLOBAL_OUTPUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )
    if torch.count_nonzero(result.output[0]).item() != 0:
        raise AssertionError("fully padded global output row must be exactly zero")
    assert result.lse is not None
    if not torch.isneginf(result.lse[0]).all():
        raise AssertionError("fully padded global LSE row must be exactly -inf")

    mutated_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    with torch.no_grad():
        mutated_inputs[0][0].fill_(17)
        mutated_inputs[1][0].fill_(-31)
        mutated_inputs[2][0].fill_(47)
    mutated_result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *mutated_inputs,
        **call_kwargs,
    )
    if not torch.equal(result.output[1], mutated_result.output[1]):
        raise AssertionError("empty-row mutation leaked into neighboring global output")
    assert mutated_result.lse is not None
    if not torch.equal(result.lse[1], mutated_result.lse[1]):
        raise AssertionError("empty-row mutation leaked into neighboring global LSE")

    isolation_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    isolation_result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *isolation_inputs,
        **call_kwargs,
    )
    assert isolation_result.lse is not None
    gradients = _differentiate(
        isolation_result.output,
        isolation_result.lse,
        isolation_inputs,
        torch.ones_like(isolation_result.output),
        torch.zeros_like(isolation_result.lse),
    )
    for name, gradient in zip(("dQ", "dK", "dV"), gradients, strict=True):
        if torch.count_nonzero(gradient[0]).item() != 0:
            raise AssertionError(f"fully padded row produced nonzero {name}")
        if torch.count_nonzero(gradient[1]).item() == 0:
            raise AssertionError(f"neighboring nonempty row produced zero {name}")

    actual_gemma4_layer_empty_row = False
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
        from transformers import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        register_gemma4_fa4_h100()
        lock_path = Path(__file__).resolve().parents[1] / "configs/model/gemma4-31b.lock.json"
        locked_text = dict(json.loads(lock_path.read_text())["text_config"])
        locked_text["hidden_size"] = 64
        locked_text["intermediate_size"] = 128
        locked_config = Gemma4TextConfig(**locked_text)
        locked_config._attn_implementation = BACKEND_NAME
        attention = Gemma4TextAttention(locked_config, layer_idx=5).to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        hidden = torch.randn(
            (2, padded_length, 64),
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed + 1),
            requires_grad=True,
        )
        positions = torch.arange(padded_length, device="cuda").expand(2, -1)
        cos = torch.ones(
            (2, padded_length, spec.head_dim_qk),
            dtype=torch.bfloat16,
            device="cuda",
        )
        sin = torch.zeros_like(cos)
        actual_output, actual_weights = attention(
            hidden,
            (cos, sin),
            plan,
            {},
            position_ids=positions,
            allow_flex_fallback=False,
        )
        if actual_weights is not None or actual_output.shape != hidden.shape:
            raise AssertionError(
                "real mixed-empty Gemma4TextAttention returned an invalid contract"
            )
        if getattr(attention, "_gemma4_fa4_last_path", None) != "fa4_global_varlen_native":
            raise AssertionError("real mixed-empty Gemma4TextAttention did not select native THD")
        if torch.count_nonzero(actual_output[0]).item() != 0:
            raise AssertionError("real Gemma4TextAttention did not restore the empty output row")
        (hidden_gradient,) = torch.autograd.grad(actual_output[1].float().square().mean(), hidden)
        if torch.count_nonzero(hidden_gradient[0]).item() != 0:
            raise AssertionError("real Gemma4TextAttention produced an empty-row hidden gradient")
        if (
            not torch.isfinite(hidden_gradient[1]).all()
            or torch.count_nonzero(hidden_gradient[1]).item() == 0
        ):
            raise AssertionError("real Gemma4TextAttention produced an invalid neighbor gradient")
        actual_gemma4_layer_empty_row = True

    record["packed_lengths"] = list(lengths)
    record["cu_seqlens"] = [0, 0, lengths[1]]
    record["padding_sentinels"] = "zero_O/-inf_LSE"
    record["hostile_empty_row_isolation"] = True
    record["empty_row_gradients_exact_zero"] = True
    record["actual_gemma4_layer_empty_row"] = actual_gemma4_layer_empty_row
    return record


def _run_global_document_split_backward(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    document_lengths = (31, 33, 65)
    total_length = sum(document_lengths)
    inputs = _make_inputs(spec, batch=1, q_length=total_length, seed=seed)
    document_ids = torch.repeat_interleave(
        torch.arange(len(document_lengths), dtype=torch.int32, device=inputs[0].device),
        torch.tensor(document_lengths, dtype=torch.int64, device=inputs[0].device),
    ).unsqueeze(0)
    call_kwargs = {
        "document_ids": document_ids,
        "allow_flex_fallback": False,
    }
    result = _module_call(_AttentionModule(5, spec), spec, *inputs, **call_kwargs)
    reference_builder = _packed_rectangular_builder(
        document_lengths,
        document_lengths,
        spec,
        upcast=torch.float32,
    )
    bf16_builder = _packed_rectangular_builder(
        document_lengths,
        document_lengths,
        spec,
        upcast=None,
    )
    record = _validate_kernel_result(
        case="global-document-split-backward",
        result=result,
        expected_path="fa4_global_varlen_native",
        inputs=inputs,
        reference_builder=reference_builder,
        bf16_builder=bf16_builder,
        seed=seed,
        output_atol=GLOBAL_OUTPUT_ATOL,
        output_rtol=GLOBAL_OUTPUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )

    first_length = document_lengths[0]
    mutated_q = inputs[0].detach().clone().requires_grad_(True)
    mutated_k = inputs[1].detach().clone()
    mutated_v = inputs[2].detach().clone()
    mutated_k[:, :, first_length:] = 0
    mutated_v[:, :, first_length:] = 32
    mutated_inputs = (
        mutated_q,
        mutated_k.requires_grad_(True),
        mutated_v.requires_grad_(True),
    )
    mutated_result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *mutated_inputs,
        **call_kwargs,
    )
    if mutated_result.path != "fa4_global_varlen_native":
        raise AssertionError("document mutation left the native global varlen route")
    if not torch.equal(result.output[:, :first_length], mutated_result.output[:, :first_length]):
        raise AssertionError("later-document K/V leaked into the first-document output")
    assert result.lse is not None and mutated_result.lse is not None
    if not torch.equal(result.lse[:, :, :first_length], mutated_result.lse[:, :, :first_length]):
        raise AssertionError("later-document K leaked into the first-document LSE")
    if torch.equal(result.output[:, first_length:], mutated_result.output[:, first_length:]):
        raise AssertionError("hostile later-document mutation was not discriminating")

    isolation_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    isolation_result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *isolation_inputs,
        **call_kwargs,
    )
    if isolation_result.lse is None:
        raise AssertionError("native document-split path must expose FP32 LSE")
    dout = torch.zeros_like(isolation_result.output)
    dout[:, :first_length] = 1
    dlse = torch.zeros_like(isolation_result.lse)
    dlse[:, :, :first_length] = 1
    gradients = _differentiate(
        isolation_result.output,
        isolation_result.lse,
        isolation_inputs,
        dout,
        dlse,
    )
    for name, gradient in zip(("dQ", "dK", "dV"), gradients, strict=True):
        if torch.count_nonzero(gradient[:, :, first_length:]).item() != 0:
            raise AssertionError(f"first-document losses leaked into later-document {name}")
    _, _, reference_gradients = _reference_run(
        isolation_inputs,
        reference_builder,
        dout,
        dlse,
        include_lse=True,
    )
    _, _, bf16_gradients = _reference_run(
        isolation_inputs,
        bf16_builder,
        dout,
        dlse,
        include_lse=True,
    )
    record["isolation_gradient_errors"] = _assert_gradients(
        gradients,
        reference_gradients,
        bf16_gradients,
    )
    record["document_lengths"] = list(document_lengths)
    record["rebuilt_cu_seqlens"] = [0, 31, 64, 129]
    record["hostile_forward_isolation"] = True
    record["structured_gradient_isolation"] = True
    return record


def _run_global_lower_right_long_backward(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    q_length, kv_length = 33, 2049
    q_start = kv_length - q_length
    inputs = _make_inputs(
        spec,
        batch=1,
        q_length=q_length,
        kv_length=kv_length,
        seed=seed,
    )
    plan = Gemma4MaskPlan(
        batch_size=1,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_start,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
    )
    result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *inputs,
        attention_mask=plan,
        allow_flex_fallback=False,
    )
    record = _validate_kernel_result(
        case="global-lower-right-long-backward",
        result=result,
        expected_path="fa4_global_varlen_native",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, q_start=q_start, upcast=torch.float32),
        bf16_builder=_upstream_style_builder(spec, q_start=q_start),
        seed=seed,
        output_atol=GLOBAL_OUTPUT_ATOL,
        output_rtol=GLOBAL_OUTPUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )
    record.update(q_length=q_length, kv_length=kv_length, q_offset=q_start)
    return record


def _run_global_packed_long_backward(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    q_lengths = (33, 65)
    k_lengths = (2049, 4097)
    q_total = sum(q_lengths)
    k_total = sum(k_lengths)
    inputs = _make_inputs(
        spec,
        batch=1,
        q_length=q_total,
        kv_length=k_total,
        seed=seed,
    )
    plan = Gemma4MaskPlan(
        batch_size=1,
        q_length=q_total,
        kv_length=k_total,
        q_offset=k_total - q_total,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
    )
    cu_q = torch.tensor((0, q_lengths[0], q_total), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, k_lengths[0], k_total), dtype=torch.int32, device="cuda")
    call_kwargs = {
        "attention_mask": plan,
        "cu_seq_lens_q": cu_q,
        "cu_seq_lens_k": cu_k,
        "max_length_q": max(q_lengths),
        "max_length_k": max(k_lengths),
        "allow_flex_fallback": False,
    }
    result = _module_call(_AttentionModule(5, spec), spec, *inputs, **call_kwargs)
    record = _validate_kernel_result(
        case="global-packed-long-backward",
        result=result,
        expected_path="fa4_global_varlen_native",
        inputs=inputs,
        reference_builder=_packed_rectangular_builder(
            q_lengths,
            k_lengths,
            spec,
            upcast=torch.float32,
        ),
        bf16_builder=_packed_rectangular_builder(q_lengths, k_lengths, spec, upcast=None),
        seed=seed,
        output_atol=GLOBAL_OUTPUT_ATOL,
        output_rtol=GLOBAL_OUTPUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )

    isolated_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)
    isolated_result = _module_call(
        _AttentionModule(5, spec),
        spec,
        *isolated_inputs,
        **call_kwargs,
    )
    dout = torch.zeros_like(isolated_result.output)
    dout[:, : q_lengths[0]] = 1
    assert isolated_result.lse is not None
    dlse = torch.zeros_like(isolated_result.lse)
    dlse[:, :, : q_lengths[0]] = 1
    gradients = _differentiate(
        isolated_result.output,
        isolated_result.lse,
        isolated_inputs,
        dout,
        dlse,
    )
    if torch.count_nonzero(gradients[0][:, :, q_lengths[0] :]).item() != 0:
        raise AssertionError("first packed segment leaked into second-segment dQ")
    for name, gradient in zip(("dK", "dV"), gradients[1:], strict=True):
        if torch.count_nonzero(gradient[:, :, k_lengths[0] :]).item() != 0:
            raise AssertionError(f"first packed segment leaked into second-segment {name}")

    record["q_lengths"] = list(q_lengths)
    record["k_lengths"] = list(k_lengths)
    record["segment_gradient_isolation"] = True
    return record


def _run_global_native_varlen_noncontiguous_gradients(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    q_lengths = (31, 65)
    k_lengths = (33, 129)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = (
        (sum(q_lengths), spec.num_q_heads, spec.head_dim_qk),
        (sum(k_lengths), spec.num_kv_heads, spec.head_dim_qk),
        (sum(k_lengths), spec.num_kv_heads, spec.head_dim_v),
    )
    inputs = tuple(
        torch.randn(
            shape,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).requires_grad_(True)
        for shape in shapes
    )
    cu_q = torch.tensor((0, q_lengths[0], sum(q_lengths)), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, k_lengths[0], sum(k_lengths)), dtype=torch.int32, device="cuda")
    output, lse = fa4_global_varlen_forward(
        *inputs,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        spec=spec,
    )
    if output.shape != inputs[0].shape or output.dtype != torch.bfloat16:
        raise AssertionError("direct native THD path returned an invalid output contract")
    if lse.shape != (spec.num_q_heads, sum(q_lengths)) or lse.dtype != torch.float32:
        raise AssertionError("direct native THD path returned an invalid FP32 LSE contract")

    dout, dlse = _odd_padded_gradient_outputs(output, lse, seed=seed + 30_007)
    gradients = _differentiate(output, lse, inputs, dout, dlse)
    _assert_distinct_operands_and_gradients(inputs, gradients)
    reference_builder = _packed_thd_builder(
        q_lengths,
        k_lengths,
        spec,
        upcast=torch.float32,
    )
    bf16_builder = _packed_thd_builder(q_lengths, k_lengths, spec, upcast=None)
    reference_output, reference_lse, reference_gradients = _reference_run(
        inputs,
        reference_builder,
        dout,
        dlse,
        include_lse=True,
    )
    _, _, bf16_gradients = _reference_run(
        inputs,
        bf16_builder,
        dout,
        dlse,
        include_lse=True,
    )
    return {
        "case": "global-native-varlen-noncontiguous-gradients",
        "path": "fa4_global_varlen_native_direct",
        "q_lengths": list(q_lengths),
        "k_lengths": list(k_lengths),
        "geometry": "32Q/4KV/d512/GQA8/causal/scale1",
        "output": _assert_close(
            "O",
            output,
            reference_output,
            atol=GLOBAL_OUTPUT_ATOL,
            rtol=GLOBAL_OUTPUT_RTOL,
        ),
        "lse": _assert_close(
            "LSE",
            lse,
            reference_lse,
            atol=GLOBAL_LSE_ATOL,
            rtol=0.0,
        ),
        "gradients": _assert_gradients(
            gradients,
            reference_gradients,
            bf16_gradients,
        ),
        "gradient_sources": "odd_padded_noncontiguous_dout+dlse",
        "dout_stride": list(dout.stride()),
        "dlse_stride": list(dlse.stride()),
        "dout_contiguous": dout.is_contiguous(),
        "dlse_contiguous": dlse.is_contiguous(),
    }


def _validate_forward_only_result(
    *,
    case: str,
    result: Gemma4DispatchResult,
    expected_path: str,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference_builder: ReferenceBuilder,
) -> dict[str, Any]:
    if result.path != expected_path:
        raise AssertionError(f"{case} routed to {result.path!r}, expected {expected_path!r}")
    if result.output.requires_grad or result.output.grad_fn is not None:
        raise AssertionError("forward-only result unexpectedly retained an autograd graph")
    if result.lse is None or result.lse.dtype != torch.float32:
        raise AssertionError("forward-only FA4 must expose FP32 LSE")
    with torch.no_grad():
        reference_output, reference_lse = reference_builder(*(tensor.detach() for tensor in inputs))
    return {
        "case": case,
        "path": result.path,
        "output": _assert_close(
            "O",
            result.output,
            reference_output,
            atol=GLOBAL_OUTPUT_ATOL,
            rtol=GLOBAL_OUTPUT_RTOL,
        ),
        "lse": _assert_close(
            "LSE",
            result.lse,
            reference_lse,
            atol=GLOBAL_LSE_ATOL,
            rtol=0.0,
        ),
        "lse_dtype": str(result.lse.dtype),
        "autograd": "disabled",
    }


def _run_global_forward_only_long(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    q_length, kv_length = 1, 2048
    q_start = kv_length - q_length
    inputs = _make_inputs(
        spec,
        batch=1,
        q_length=q_length,
        kv_length=kv_length,
        seed=seed,
    )
    plan = Gemma4MaskPlan(
        batch_size=1,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_start,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
    )
    with torch.no_grad():
        result = _module_call(
            _AttentionModule(5, spec),
            spec,
            *inputs,
            attention_mask=plan,
            allow_flex_fallback=False,
        )
    record = _validate_forward_only_result(
        case="global-forward-only-long",
        result=result,
        expected_path="fa4_global_forward_only",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, q_start=q_start, upcast=torch.float32),
    )
    record["q_length"] = q_length
    record["kv_length"] = kv_length
    return record


def _run_global_varlen_forward_only_long(seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    q_length, kv_length = 33, 2048
    q_start = kv_length - q_length
    inputs = _make_inputs(
        spec,
        batch=1,
        q_length=q_length,
        kv_length=kv_length,
        seed=seed,
    )
    plan = Gemma4MaskPlan(
        batch_size=1,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_start,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
    )
    cu_q = torch.tensor([0, q_length], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, kv_length], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        result = _module_call(
            _AttentionModule(5, spec),
            spec,
            *inputs,
            attention_mask=plan,
            cu_seq_lens_q=cu_q,
            cu_seq_lens_k=cu_k,
            max_length_q=q_length,
            max_length_k=kv_length,
            allow_flex_fallback=False,
        )
    record = _validate_forward_only_result(
        case="global-varlen-forward-only-long",
        result=result,
        expected_path="fa4_global_varlen_forward_only",
        inputs=inputs,
        reference_builder=_fixed_builder(spec, q_start=q_start, upcast=torch.float32),
    )
    record["q_length"] = q_length
    record["kv_length"] = kv_length
    return record


def _run_global_forward_only_max_context(_seed: int) -> dict[str, Any]:
    _require_h100()
    spec = GLOBAL_ATTENTION
    kv_length = 262_144
    q = torch.zeros(
        (1, spec.num_q_heads, 1, spec.head_dim_qk),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.zeros(
        (1, spec.num_kv_heads, kv_length, spec.head_dim_qk),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.zeros(
        (1, spec.num_kv_heads, kv_length, spec.head_dim_v),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v[:, :, 0] = 64
    plan = Gemma4MaskPlan(
        batch_size=1,
        q_length=1,
        kv_length=kv_length,
        q_offset=kv_length - 1,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
    )
    with torch.no_grad():
        result = _module_call(
            _AttentionModule(5, spec),
            spec,
            q,
            k,
            v,
            attention_mask=plan,
            allow_flex_fallback=False,
        )
    if result.path != "fa4_global_forward_only" or result.lse is None:
        raise AssertionError("maximum-context global decode did not use forward-only FA4")
    expected_output = torch.full_like(result.output, 64 / kv_length)
    expected_lse = torch.full_like(result.lse, float(torch.log(torch.tensor(kv_length))))
    torch.testing.assert_close(result.output, expected_output, atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.lse, expected_lse, atol=1e-4, rtol=0.0)
    return {
        "case": MAX_CONTEXT_CASE,
        "path": result.path,
        "q_length": 1,
        "kv_length": kv_length,
        "output_value": result.output[0, 0, 0, 0].item(),
        "expected_output_value": 64 / kv_length,
        "lse_value": result.lse[0, 0, 0].item(),
        "expected_lse_value": float(torch.log(torch.tensor(kv_length))),
        "autograd": "disabled",
    }


def _run_hf_mask_transport(seed: int) -> dict[str, Any]:
    try:
        from transformers import Gemma4Config, Gemma4TextConfig
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4Model,
            Gemma4TextAttention,
            create_masks_for_vision_model,
        )
        from transformers.utils import logging as transformers_logging
    except Exception as exc:
        raise RuntimeError(
            "hf-mask-transport requires the pinned, patched Transformers checkout"
        ) from exc

    transformers_logging.set_verbosity_error()

    registered_name = register_gemma4_fa4_h100()
    if registered_name != BACKEND_NAME:
        raise AssertionError("registration returned the wrong backend name")
    if ALL_ATTENTION_FUNCTIONS[BACKEND_NAME].__name__ != "gemma4_fa4_attention_forward":
        raise AssertionError("the project attention backend was not registered")
    if ALL_MASK_ATTENTION_FUNCTIONS[BACKEND_NAME].__name__ != "gemma4_fa4_mask":
        raise AssertionError("the project mask backend was not registered")

    text_config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        use_cache=False,
        tie_word_embeddings=False,
        sliding_window=1024,
        layer_types=["sliding_attention"],
        use_bidirectional_attention="vision",
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=0,
        num_global_key_value_heads=1,
        global_head_dim=8,
    )
    text_config._attn_implementation = BACKEND_NAME
    config = Gemma4Config(
        text_config=text_config,
        vision_config=None,
        audio_config=None,
        tie_word_embeddings=False,
    )
    config.get_text_config()._attn_implementation = BACKEND_NAME
    model = Gemma4Model(config)
    capture = _CaptureLanguageModel()
    model.language_model = capture

    def placeholder_mask(_self, input_ids, inputs_embeds):
        del input_ids
        shape = inputs_embeds.shape[:2]
        zeros = torch.zeros(shape, dtype=torch.bool, device=inputs_embeds.device)
        return zeros, zeros.clone(), zeros.clone()

    model.get_placeholder_mask = MethodType(placeholder_mask, model)
    mm_token_type_ids = torch.tensor([[0, 1, 1, 0, 2, 2, 0]], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs_embeds = torch.randn((1, 7, 16), generator=generator)
    attention_mask = torch.ones((1, 7), dtype=torch.long)
    model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
    )
    if capture.received is None:
        raise AssertionError("tiny Gemma model did not call its language-model boundary")
    expected_ids = torch.tensor([[-1, 0, 0, -1, 1, 1, -1]], dtype=torch.int32)
    transported_ids = capture.received.get("vision_block_ids")
    if not isinstance(transported_ids, torch.Tensor):
        raise AssertionError("patched Gemma forward did not transport vision_block_ids")
    torch.testing.assert_close(transported_ids.to(torch.int32), expected_ids)
    mask_mapping = capture.received.get("attention_mask")
    if not isinstance(mask_mapping, dict):
        raise AssertionError("create_masks_for_vision_model did not return the per-layer mapping")
    if set(mask_mapping) != {"full_attention", "sliding_attention"}:
        raise AssertionError("Gemma mask mapping has unexpected layer-family keys")
    if not all(isinstance(value, Gemma4MaskPlan) for value in mask_mapping.values()):
        raise AssertionError("the registered project mask adapter did not preserve mask plans")

    scalar = torch.tensor(0)
    final_key = torch.tensor(6)
    if bool(mask_mapping["sliding_attention"].mask_function(scalar, scalar, scalar, final_key)):
        raise AssertionError("derived text tokens unexpectedly became one vision block")

    override = torch.full_like(expected_ids, 17)
    model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        vision_block_ids=override,
        use_cache=False,
    )
    assert capture.received is not None
    if capture.received.get("vision_block_ids") is not override:
        raise AssertionError("Gemma forward overwrote an explicit vision_block_ids value")
    override_mapping = capture.received.get("attention_mask")
    if not isinstance(override_mapping, dict):
        raise AssertionError("explicit vision IDs did not produce a mask mapping")
    if not bool(
        override_mapping["sliding_attention"].mask_function(
            scalar,
            scalar,
            scalar,
            final_key,
        )
    ):
        raise AssertionError("explicit vision IDs were not authoritative for mask construction")
    if bool(
        override_mapping["full_attention"].mask_function(
            scalar,
            scalar,
            scalar,
            final_key,
        )
    ):
        raise AssertionError("explicit vision IDs leaked into global attention")

    model(
        inputs_embeds=inputs_embeds,
        attention_mask=mask_mapping,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
    )
    assert capture.received is not None
    prebuilt_ids = capture.received.get("vision_block_ids")
    if not isinstance(prebuilt_ids, torch.Tensor):
        raise AssertionError("prebuilt generation masks lost compact vision IDs")
    torch.testing.assert_close(prebuilt_ids.to(torch.int32), expected_ids)

    backend_path = None
    actual_module_gradient_finite = False
    actual_global_path = None
    actual_global_module_gradient_finite = False
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
        spec = SLIDING_ATTENTION
        q, k, v = _make_inputs(spec, batch=1, q_length=7, seed=seed + 1)
        attention_module = _AttentionModule(0, spec)
        backend_output, backend_weights = ALL_ATTENTION_FUNCTIONS[BACKEND_NAME](
            attention_module,
            q,
            k,
            v,
            mask_mapping["sliding_attention"],
            dropout=0.0,
            scaling=1.0,
            sliding_window=1024,
            vision_block_ids=transported_ids.to(device="cuda", dtype=torch.int32),
            allow_flex_fallback=False,
        )
        if backend_weights is not None or backend_output.shape != (1, 7, 32, 256):
            raise AssertionError(
                "registered backend returned an invalid attention-interface result"
            )
        backend_path = getattr(attention_module, "_gemma4_fa4_last_path", None)
        if backend_path != "fa4_local_fixed":
            raise AssertionError("transported vision IDs did not reach the registered FA4 backend")

        lock_path = Path(__file__).resolve().parents[1] / "configs/model/gemma4-31b.lock.json"
        locked_text = dict(json.loads(lock_path.read_text())["text_config"])
        locked_text["hidden_size"] = 64
        locked_text["intermediate_size"] = 128
        locked_config = Gemma4TextConfig(**locked_text)
        locked_config._attn_implementation = BACKEND_NAME
        actual_attention = Gemma4TextAttention(locked_config, layer_idx=0).to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        hidden = torch.randn(
            (1, 7, 64),
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed + 2),
            requires_grad=True,
        )
        positions = torch.arange(7, device="cuda").unsqueeze(0)
        vision_cuda = transported_ids.to(device="cuda", dtype=torch.int32)
        actual_masks = create_masks_for_vision_model(
            locked_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
            block_sequence_ids=vision_cuda,
        )
        cos = torch.ones((1, 7, 256), dtype=torch.bfloat16, device="cuda")
        sin = torch.zeros_like(cos)
        actual_output, actual_weights = actual_attention(
            hidden,
            (cos, sin),
            actual_masks["sliding_attention"],
            {},
            position_ids=positions,
            vision_block_ids=vision_cuda,
            allow_flex_fallback=False,
        )
        if actual_weights is not None or actual_output.shape != hidden.shape:
            raise AssertionError("real Gemma4TextAttention returned an invalid output contract")
        if getattr(actual_attention, "_gemma4_fa4_last_path", None) != "fa4_local_fixed":
            raise AssertionError("real Gemma4TextAttention did not select the local FA4 path")
        (hidden_gradient,) = torch.autograd.grad(actual_output.float().square().mean(), hidden)
        actual_module_gradient_finite = bool(torch.isfinite(hidden_gradient).all().item())
        if not actual_module_gradient_finite:
            raise AssertionError("real Gemma4TextAttention produced a non-finite input gradient")

        global_length = 2049
        actual_global_attention = Gemma4TextAttention(locked_config, layer_idx=5).to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        global_hidden = torch.randn(
            (1, global_length, 64),
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed + 3),
            requires_grad=True,
        )
        global_positions = torch.arange(global_length, device="cuda").unsqueeze(0)
        global_cos = torch.ones(
            (1, global_length, GLOBAL_ATTENTION.head_dim_qk),
            dtype=torch.bfloat16,
            device="cuda",
        )
        global_sin = torch.zeros_like(global_cos)
        global_mask = Gemma4MaskPlan(
            batch_size=1,
            q_length=global_length,
            kv_length=global_length,
            q_offset=0,
            kv_offset=0,
            mask_function=None,
            attention_mask=None,
        )
        global_output, global_weights = actual_global_attention(
            global_hidden,
            (global_cos, global_sin),
            global_mask,
            {},
            position_ids=global_positions,
            allow_flex_fallback=False,
        )
        if global_weights is not None or global_output.shape != global_hidden.shape:
            raise AssertionError("real global Gemma4TextAttention returned an invalid contract")
        actual_global_path = getattr(actual_global_attention, "_gemma4_fa4_last_path", None)
        if actual_global_path != "fa4_global_varlen_native":
            raise AssertionError("real global Gemma4TextAttention did not select native THD")
        (global_hidden_gradient,) = torch.autograd.grad(
            global_output.float().square().mean(),
            global_hidden,
        )
        actual_global_module_gradient_finite = bool(
            torch.isfinite(global_hidden_gradient).all().item()
        )
        if not actual_global_module_gradient_finite:
            raise AssertionError(
                "real global Gemma4TextAttention produced a non-finite input gradient"
            )

    return {
        "case": "hf-mask-transport",
        "path": backend_path,
        "registered_backend": BACKEND_NAME,
        "vision_block_ids": expected_ids.tolist(),
        "mask_plan_keys": sorted(mask_mapping),
        "explicit_override_authoritative": True,
        "prebuilt_mask_transport": True,
        "backend_executed": backend_path is not None,
        "actual_text_attention_executed": actual_module_gradient_finite,
        "actual_global_text_attention_path": actual_global_path,
        "actual_global_text_attention_s2049_backward": (actual_global_module_gradient_finite),
    }


RUNNERS: dict[str, Callable[[int], dict[str, Any]]] = {
    "local-fixed-strided": _run_local_fixed_strided,
    "local-packed-padding": _run_local_packed_padding,
    "local-lower-right": _run_local_lower_right,
    "global-fixed-strided": _run_global_fixed_strided,
    "global-varlen-batch": _run_global_varlen_batch,
    "global-packed-empty-row": _run_global_packed_empty_row,
    "global-document-split-backward": _run_global_document_split_backward,
    "global-lower-right-long-backward": _run_global_lower_right_long_backward,
    "global-packed-long-backward": _run_global_packed_long_backward,
    "global-native-varlen-noncontiguous-gradients": (
        _run_global_native_varlen_noncontiguous_gradients
    ),
    "global-forward-only-long": _run_global_forward_only_long,
    "global-varlen-forward-only-long": _run_global_varlen_forward_only_long,
    MAX_CONTEXT_CASE: _run_global_forward_only_max_context,
    "hf-mask-transport": _run_hf_mask_transport,
    "static-cache-local-small": _run_static_cache_local_small,
    "static-cache-local-boundary": _run_static_cache_local_boundary,
    "static-cache-local-first-roll": _run_static_cache_local_first_roll,
    "static-cache-global-small": _run_static_cache_global_small,
    "static-cache-global-k1025": _run_static_cache_global_k1025,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, MAX_CONTEXT_CASE, "all"), default="all")
    parser.add_argument("--seed", type=int, default=11011)
    args = parser.parse_args()

    selected = CASES if args.case == "all" else (args.case,)
    try:
        results = [RUNNERS[name](args.seed + index * 100) for index, name in enumerate(selected)]
    except Exception as exc:
        print(
            json.dumps(
                {
                    "case": args.case,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "status": "failed",
                },
                sort_keys=True,
            )
        )
        return 1

    device = None
    if torch.cuda.is_available():
        device = {
            "capability": list(torch.cuda.get_device_capability()),
            "name": torch.cuda.get_device_name(),
        }
    print(
        json.dumps(
            {
                "device": device,
                "requested_case": args.case,
                "results": results,
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
