"""Pure contracts shared by the Axolotl plugin and report-comparison CLI."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

from .gemma4_12b_compat import BACKEND_NAME_12B_COMPAT

HARNESS_SCHEMA_VERSION = 2
HARNESS_BACKENDS = ("project_12b_compat", "hybrid", "sdpa")
LORA_INITIALIZATION_SCHEME = "name-seeded-kaiming-uniform-a-zero-b-v1"
WORKLOAD_KEYS = (
    "model_id",
    "model_revision",
    "sequence_len",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "warmup_steps",
    "seed",
    "dataset_sha256",
)
LOSS_ATOL = 5e-3
LOSS_RTOL = 2e-3
GRADIENT_MIN_COSINE = 0.999
GRADIENT_MAX_REL_L2 = 1e-2


class HarnessComparisonError(ValueError):
    """Raised when two reports are not semantically comparable or correct."""


def initialize_lora_parameters(model: Any, *, seed: int) -> dict[str, Any]:
    """Reset trainable LoRA parameters without depending on process-global RNG state."""

    parameters = [
        (name, parameter)
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Axolotl harness requires trainable LoRA parameters")
    digest = hashlib.sha256()
    value_count = 0
    with torch.no_grad():
        for name, parameter in parameters:
            is_a = ".lora_A." in f".{name}"
            is_b = ".lora_B." in f".{name}"
            if is_a == is_b:
                raise ValueError(
                    "Axolotl harness accepts only trainable lora_A/lora_B parameters; "
                    f"got {name}"
                )
            if is_a:
                if parameter.ndim < 2 or parameter.shape[-1] <= 0:
                    raise ValueError(f"LoRA A parameter {name} has invalid geometry")
                name_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8],
                    "little",
                ) & ((1 << 63) - 1)
                generator = torch.Generator(device="cpu").manual_seed(name_seed)
                bound = 1.0 / math.sqrt(parameter.shape[-1])
                initialized = torch.empty(parameter.shape, dtype=torch.float32).uniform_(
                    -bound,
                    bound,
                    generator=generator,
                )
                parameter.copy_(
                    initialized.to(device=parameter.device, dtype=parameter.dtype)
                )
            else:
                parameter.zero_()
            raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            digest.update(len(name).to_bytes(4, "little"))
            digest.update(name.encode())
            digest.update(raw)
            value_count += parameter.numel()
    return {
        "scheme": LORA_INITIALIZATION_SCHEME,
        "seed": int(seed),
        "sha256": digest.hexdigest(),
        "parameter_count": len(parameters),
        "value_count": value_count,
    }


def _require_equal_or_set(cfg: MutableMapping[str, Any], key: str, expected: Any) -> None:
    if key in cfg and cfg[key] is not None and cfg[key] != expected:
        raise ValueError(f"Axolotl harness requires {key}={expected!r}, got {cfg[key]!r}")
    cfg[key] = expected


def mutate_axolotl_config(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Select one attention backend and freeze the short zero-update workload."""

    backend = cfg.get("fa4_harness_backend", "project_12b_compat")
    if backend not in HARNESS_BACKENDS:
        raise ValueError(f"fa4_harness_backend must be one of {HARNESS_BACKENDS}, got {backend!r}")
    for key, expected in (
        ("learning_rate", 0.0),
        ("weight_decay", 0.0),
        ("lora_dropout", 0.0),
        ("sample_packing", False),
        ("torch_compile", False),
        ("gradient_accumulation_steps", 1),
        ("micro_batch_size", 1),
        ("sequence_len", 1024),
        ("max_steps", 8),
        ("seed", 3600),
    ):
        _require_equal_or_set(cfg, key, expected)
    if backend == "project_12b_compat":
        cfg["attn_implementation"] = BACKEND_NAME_12B_COMPAT
        cfg["gemma4_hybrid_attn_impl"] = False
    elif backend == "hybrid":
        cfg["attn_implementation"] = "flash_attention_2"
        cfg["gemma4_hybrid_attn_impl"] = True
    else:
        cfg["attn_implementation"] = "sdpa"
        cfg["gemma4_hybrid_attn_impl"] = False
    return cfg


def _linear_percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one timing is required")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_timings(values: Sequence[float]) -> dict[str, int | float]:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timings must be a nonempty sequence of positive finite milliseconds")
    p25 = _linear_percentile(values, 0.25)
    p75 = _linear_percentile(values, 0.75)
    return {
        "count": len(values),
        "median_ms": _linear_percentile(values, 0.5),
        "p25_ms": p25,
        "p75_ms": p75,
        "iqr_ms": p75 - p25,
    }


def _checked_workload(report: Mapping[str, Any]) -> dict[str, Any]:
    workload = report.get("workload")
    if not isinstance(workload, Mapping):
        raise HarnessComparisonError("report is missing its workload contract")
    missing = [key for key in WORKLOAD_KEYS if key not in workload]
    if missing:
        raise HarnessComparisonError(f"workload is missing keys: {', '.join(missing)}")
    return {key: workload[key] for key in WORKLOAD_KEYS}


def _checked_lora_initialization(
    report: Mapping[str, Any],
    *,
    workload_seed: int,
) -> dict[str, Any]:
    initialization = report.get("lora_initialization")
    if not isinstance(initialization, Mapping):
        raise HarnessComparisonError("report is missing its LoRA initialization fingerprint")
    required = ("scheme", "seed", "sha256", "parameter_count", "value_count")
    missing = [key for key in required if key not in initialization]
    if missing:
        raise HarnessComparisonError(
            f"LoRA initialization is missing keys: {', '.join(missing)}"
        )
    sha256 = initialization["sha256"]
    if (
        initialization["scheme"] != LORA_INITIALIZATION_SCHEME
        or initialization["seed"] != workload_seed
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(initialization["parameter_count"], bool)
        or not isinstance(initialization["parameter_count"], int)
        or initialization["parameter_count"] <= 0
        or isinstance(initialization["value_count"], bool)
        or not isinstance(initialization["value_count"], int)
        or initialization["value_count"] <= 0
    ):
        raise HarnessComparisonError("LoRA initialization fingerprint is invalid")
    return {key: initialization[key] for key in required}


def _compare_losses(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_count: int,
) -> dict[str, float]:
    try:
        left = [float(value) for value in baseline.get("losses", ())]
        right = [float(value) for value in candidate.get("losses", ())]
    except (TypeError, ValueError) as exc:
        raise HarnessComparisonError("loss sequences must contain finite numbers") from exc
    if len(left) != expected_count or len(right) != expected_count:
        raise HarnessComparisonError(
            f"loss sequences must each contain exactly {expected_count} measured steps"
        )
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise HarnessComparisonError("loss sequences must contain only finite numbers")
    deltas = [abs(a - b) for a, b in zip(left, right, strict=True)]
    tolerances = [LOSS_ATOL + LOSS_RTOL * abs(a) for a in left]
    if any(delta > tolerance for delta, tolerance in zip(deltas, tolerances, strict=True)):
        raise HarnessComparisonError(
            f"loss drift exceeds atol={LOSS_ATOL} and rtol={LOSS_RTOL}: max_abs={max(deltas)}"
        )
    return {"max_abs": max(deltas), "max_rel": max(d / max(abs(a), 1e-12) for a, d in zip(left, deltas, strict=True))}


def _compare_gradient_probe(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    left = baseline.get("gradient_probe")
    right = candidate.get("gradient_probe")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise HarnessComparisonError("gradient probe is missing")
    if left.get("parameter_names") != right.get("parameter_names"):
        raise HarnessComparisonError("gradient probe parameter names differ")
    try:
        a = [float(value) for value in left.get("values", ())]
        b = [float(value) for value in right.get("values", ())]
    except (TypeError, ValueError) as exc:
        raise HarnessComparisonError("gradient probe values must be finite numbers") from exc
    if not a or len(a) != len(b):
        raise HarnessComparisonError("gradient probe values are empty or have different lengths")
    if any(not math.isfinite(value) for value in (*a, *b)):
        raise HarnessComparisonError("gradient probe values must be finite")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if any(not math.isfinite(value) for value in (dot, norm_a, norm_b)):
        raise HarnessComparisonError("gradient probe metrics must be finite")
    if norm_a == 0 or norm_b == 0:
        raise HarnessComparisonError("gradient probe has a zero norm")
    cosine = dot / (norm_a * norm_b)
    rel_l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True))) / norm_a
    if not math.isfinite(cosine) or not math.isfinite(rel_l2):
        raise HarnessComparisonError("gradient cosine and relative L2 must be finite")
    if cosine < GRADIENT_MIN_COSINE or rel_l2 > GRADIENT_MAX_REL_L2:
        raise HarnessComparisonError(
            "gradient drift exceeds policy: "
            f"cosine={cosine}, rel_l2={rel_l2}, required cosine>={GRADIENT_MIN_COSINE}, "
            f"rel_l2<={GRADIENT_MAX_REL_L2}"
        )
    return {"cosine": cosine, "rel_l2": rel_l2}


def _validate_project_routes(report: Mapping[str, Any], *, expected_steps: int) -> None:
    routes = report.get("routes")
    layer_routes = report.get("layer_routes")
    if not isinstance(routes, Mapping) or not routes:
        raise HarnessComparisonError("project route totals are missing")
    if not isinstance(layer_routes, Mapping) or set(layer_routes) != {str(i) for i in range(48)}:
        raise HarnessComparisonError("project layer route coverage must contain exactly layers 0..47")
    forbidden = ("fallback", "flex", "sdpa", "eager")
    for route, count in routes.items():
        lowered = str(route).lower()
        if any(token in lowered for token in forbidden):
            raise HarnessComparisonError(f"project fallback route observed: {route}")
        if int(count) <= 0:
            raise HarnessComparisonError(f"project route {route} has a nonpositive count")
    expected_totals: dict[str, int] = {}
    for layer_idx in range(48):
        observed = layer_routes[str(layer_idx)]
        if not isinstance(observed, Mapping) or len(observed) != 1:
            raise HarnessComparisonError(f"layer {layer_idx} must have exactly one project route")
        expected_suffix = "fa4_global_fixed" if (layer_idx + 1) % 6 == 0 else "fa4_local_fixed"
        expected_route = f"fa4_12b_compat/{expected_suffix}"
        if set(observed) != {expected_route} or int(observed[expected_route]) != expected_steps:
            raise HarnessComparisonError(
                f"layer {layer_idx} must record {expected_steps} calls to {expected_route}"
            )
        expected_totals[expected_route] = expected_totals.get(expected_route, 0) + int(
            observed[expected_route]
        )
    if {str(key): int(value) for key, value in routes.items()} != expected_totals:
        raise HarnessComparisonError("project route totals do not match per-layer route evidence")


def _checked_timings(
    report: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> dict[str, int | float]:
    try:
        values = tuple(float(value) for value in report.get("measured_step_ms", ()))
    except (TypeError, ValueError) as exc:
        raise HarnessComparisonError(f"{label} timings must be finite numbers") from exc
    if len(values) != expected_count:
        raise HarnessComparisonError(
            f"{label} timings must contain exactly {expected_count} measured steps"
        )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise HarnessComparisonError(
            f"{label} timings must contain only positive finite milliseconds"
        )
    return summarize_timings(values)


def compare_reports(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate semantics/correctness first, then compute candidate speedup."""

    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("schema_version") != HARNESS_SCHEMA_VERSION:
            raise HarnessComparisonError(f"{label} report has an unsupported schema version")
        if report.get("status") != "complete":
            raise HarnessComparisonError(f"{label} report is not complete")
    left_workload = _checked_workload(baseline)
    right_workload = _checked_workload(candidate)
    if left_workload != right_workload:
        raise HarnessComparisonError("workload contracts differ between reports")
    try:
        max_steps = int(left_workload["max_steps"])
        warmup_steps = int(left_workload["warmup_steps"])
    except (TypeError, ValueError) as exc:
        raise HarnessComparisonError("workload step counts must be integers") from exc
    expected_measured = max_steps - warmup_steps
    if max_steps <= 0 or warmup_steps < 0 or expected_measured <= 0:
        raise HarnessComparisonError("workload step counts are invalid")
    left_initialization = _checked_lora_initialization(
        baseline,
        workload_seed=int(left_workload["seed"]),
    )
    right_initialization = _checked_lora_initialization(
        candidate,
        workload_seed=int(right_workload["seed"]),
    )
    if left_initialization != right_initialization:
        raise HarnessComparisonError("LoRA initialization fingerprints differ")
    if candidate.get("backend") != "project_12b_compat":
        raise HarnessComparisonError("candidate report is not the project_12b_compat backend")
    _validate_project_routes(candidate, expected_steps=max_steps)
    loss = _compare_losses(baseline, candidate, expected_count=expected_measured)
    gradient = _compare_gradient_probe(baseline, candidate)
    baseline_timing = _checked_timings(
        baseline,
        expected_count=expected_measured,
        label="baseline",
    )
    candidate_timing = _checked_timings(
        candidate,
        expected_count=expected_measured,
        label="candidate",
    )
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "passed": True,
        "baseline_backend": baseline.get("backend"),
        "candidate_backend": candidate.get("backend"),
        "workload": left_workload,
        "loss": loss,
        "gradient": gradient,
        "baseline_timing": baseline_timing,
        "candidate_timing": candidate_timing,
        "speedup": float(baseline_timing["median_ms"]) / float(candidate_timing["median_ms"]),
    }


__all__ = [
    "GRADIENT_MAX_REL_L2",
    "GRADIENT_MIN_COSINE",
    "HARNESS_BACKENDS",
    "HARNESS_SCHEMA_VERSION",
    "LORA_INITIALIZATION_SCHEME",
    "HarnessComparisonError",
    "LOSS_ATOL",
    "LOSS_RTOL",
    "compare_reports",
    "initialize_lora_parameters",
    "mutate_axolotl_config",
    "summarize_timings",
]
