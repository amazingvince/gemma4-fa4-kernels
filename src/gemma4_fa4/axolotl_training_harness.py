"""Contracts and comparison policy for real Axolotl full-parameter training."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

from .gemma4_native import BACKEND_NAME_GLOBAL_NATIVE, BACKEND_NAME_NATIVE

TRAINING_SCHEMA_VERSION = 1
TRAINING_MATRIX_SCHEMA_VERSION = 1
TRAINING_BACKENDS = ("native", "global_native", "sdpa")
PROSPECTIVE_MATRIX_SEEDS = (1729, 31415, 65537)
PROSPECTIVE_MATRIX_THRESHOLDS = {
    "per_seed_loss_mean_relative_delta_max": 0.03,
    "median_loss_mean_relative_delta_max": 0.02,
    "per_seed_paired_loss_nmae_max": 0.15,
    "median_paired_loss_nmae_max": 0.10,
    "per_seed_final20_relative_delta_max": 0.20,
    "median_final20_relative_delta_max": 0.15,
    "grad_norm_median_ratio_min": 0.75,
    "grad_norm_median_ratio_max": 1.33,
    "grad_norm_p95_ratio_min": 0.50,
    "grad_norm_p95_ratio_max": 2.00,
    "per_seed_median_step_speedup_min": 1.05,
    "median_step_speedup_min": 1.05,
    "peak_memory_ratio_max": 1.01,
}
EXPECTED_MODEL_ID = "google/gemma-4-12B-it"
EXPECTED_MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


class TrainingComparisonError(ValueError):
    """Raised when full-training reports are incomplete, unstable, or incomparable."""


def align_trainable_parameters_to_bf16(model: Any) -> dict[str, Any]:
    """Align full-training parameter storage before Trainer creates its optimizer.

    Axolotl intentionally leaves embedding and normalization modules in FP32 for
    eager/SDPA, but casts them to the compute dtype for flash-attention backends.
    A backend comparison therefore needs an explicit common storage contract.
    """

    converted: list[dict[str, Any]] = []
    converted_elements = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if not parameter.is_floating_point():
                raise TypeError(f"trainable parameter {name!r} is not floating point")
            if parameter.grad is not None:
                raise RuntimeError("BF16 alignment must run before gradients are created")
            if parameter.dtype == torch.bfloat16:
                continue
            source_dtype = str(parameter.dtype)
            elements = parameter.numel()
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
            converted.append(
                {
                    "name": name,
                    "elements": elements,
                    "source_dtype": source_dtype,
                }
            )
            converted_elements += elements
    evidence = {
        "target_dtype": "torch.bfloat16",
        "converted_elements": converted_elements,
        "converted_parameters": converted,
    }
    model._gemma4_fa4_precision_alignment = evidence
    return evidence


def mutate_full_training_config(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Freeze the memory-safe full-BF16 training contract and select one backend."""

    backend = cfg.get("fa4_training_backend", "native")
    if backend not in TRAINING_BACKENDS:
        raise ValueError(f"fa4_training_backend must be one of {TRAINING_BACKENDS}")
    if cfg.get("adapter") not in (None, ""):
        raise ValueError("full-parameter training must not configure an adapter")

    required = {
        "base_model": EXPECTED_MODEL_ID,
        "revision_of_model": EXPECTED_MODEL_REVISION,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adafactor",
        "torch_compile": False,
        "sample_packing": False,
        "skip_prepare_dataset": True,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    }
    for key, expected in required.items():
        actual = cfg.get(key, expected)
        if actual != expected:
            raise ValueError(f"full-BF16 training requires {key}={expected!r}, got {actual!r}")
        cfg[key] = expected

    sequence_len = cfg.setdefault("sequence_len", 512)
    if isinstance(sequence_len, bool) or sequence_len not in {256, 512, 1024, 2048}:
        raise ValueError("full-BF16 training sequence_len must be 256, 512, 1024, or 2048")
    max_steps = cfg.setdefault("max_steps", 100)
    expected_steps = cfg.setdefault("fa4_training_expected_steps", max_steps)
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= 100
        or expected_steps != max_steps
    ):
        raise ValueError("max_steps must equal fa4_training_expected_steps and be in 1..100")
    warmup = cfg.setdefault("fa4_training_timing_warmup_steps", min(5, max_steps - 1))
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < max_steps:
        raise ValueError("timing warmup must be an integer in 0..max_steps-1")
    seed = cfg.setdefault("seed", 4721)
    data_seed = cfg.setdefault("data_seed", seed)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
        or data_seed != seed
    ):
        raise ValueError("seed and data_seed must be the same uint32 value")

    if backend == "native":
        cfg["attn_implementation"] = BACKEND_NAME_NATIVE
    elif backend == "global_native":
        cfg["attn_implementation"] = BACKEND_NAME_GLOBAL_NATIVE
    else:
        cfg["attn_implementation"] = "sdpa"
    cfg["gemma4_hybrid_attn_impl"] = False
    return cfg


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise TrainingComparisonError("metric sequence is empty")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metrics(report: Mapping[str, Any]) -> list[dict[str, float | int]]:
    raw = report.get("training_metrics")
    if not isinstance(raw, list):
        raise TrainingComparisonError("report is missing training_metrics")
    expected_steps = int(report.get("workload", {}).get("max_steps", -1))
    if len(raw) != expected_steps:
        raise TrainingComparisonError(
            f"training_metrics must contain exactly {expected_steps} update steps"
        )
    normalized = []
    for expected_step, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping) or item.get("step") != expected_step:
            raise TrainingComparisonError("training_metrics steps must be contiguous from one")
        try:
            loss = float(item["loss"])
            grad_norm = float(item["grad_norm"])
            learning_rate = float(item["learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingComparisonError("training metrics are malformed") from exc
        if not all(math.isfinite(value) for value in (loss, grad_norm, learning_rate)):
            raise TrainingComparisonError("loss, grad norm, and learning rate must be finite")
        if loss < 0 or grad_norm <= 0 or learning_rate < 0:
            raise TrainingComparisonError("training metrics contain an invalid sign")
        normalized.append(
            {
                "step": expected_step,
                "loss": loss,
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
            }
        )
    return normalized


def _validate_report(report: Mapping[str, Any], *, backend: str) -> list[dict[str, float | int]]:
    if report.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise TrainingComparisonError("report has an unsupported schema version")
    if report.get("status") != "complete" or report.get("errors"):
        raise TrainingComparisonError("report is not complete")
    if report.get("backend") != backend:
        raise TrainingComparisonError(f"expected backend {backend!r}")
    if backend == "sdpa" and any(
        report.get(field) != {} for field in ("routes", "layer_routes", "native_geometries")
    ):
        raise TrainingComparisonError("the SDPA control contains project attention evidence")
    parameters = report.get("parameters")
    if not isinstance(parameters, Mapping) or int(parameters.get("trainable", 0)) <= 0:
        raise TrainingComparisonError("report is missing full-parameter evidence")
    if parameters.get("adapter_parameters") != 0:
        raise TrainingComparisonError("adapter parameters were observed in a full training run")
    if parameters.get("trainable_dtypes") != {"torch.bfloat16": parameters.get("trainable")}:
        raise TrainingComparisonError("not every trainable parameter is BF16")
    return _metrics(report)


def _curve_summary(metrics: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    losses = [float(item["loss"]) for item in metrics]
    norms = [float(item["grad_norm"]) for item in metrics]
    window = min(10, len(losses))
    initial_loss = statistics.median(losses[:window])
    final_loss = statistics.median(losses[-window:])
    median_norm = statistics.median(norms)
    p95_norm = _percentile(norms, 0.95)
    if final_loss > initial_loss * 1.10:
        raise TrainingComparisonError(
            f"loss curve diverged: initial={initial_loss}, final={final_loss}"
        )
    if max(norms) > 10.0 * median_norm:
        raise TrainingComparisonError("gradient norm has a greater-than-10x median spike")
    return {
        "initial_loss_median": initial_loss,
        "final_loss_median": final_loss,
        "loss_mean": statistics.fmean(losses),
        "grad_norm_median": median_norm,
        "grad_norm_p95": p95_norm,
        "grad_norm_max": max(norms),
    }


def compare_full_training_reports(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare pure-SDPA and all-FA4 full-training trajectories."""

    control_metrics = _validate_report(control, backend="sdpa")
    candidate_metrics = _validate_report(candidate, backend="native")
    for key in ("workload", "dataset", "parameters"):
        if control.get(key) != candidate.get(key):
            raise TrainingComparisonError(f"{key} differs between training reports")

    control_summary = _curve_summary(control_metrics)
    candidate_summary = _curve_summary(candidate_metrics)
    loss_mean_relative_delta = abs(
        candidate_summary["loss_mean"] - control_summary["loss_mean"]
    ) / max(control_summary["loss_mean"], 1e-12)
    final_loss_abs_delta = abs(
        candidate_summary["final_loss_median"] - control_summary["final_loss_median"]
    )
    median_norm_ratio = candidate_summary["grad_norm_median"] / control_summary["grad_norm_median"]
    p95_norm_ratio = candidate_summary["grad_norm_p95"] / control_summary["grad_norm_p95"]
    if loss_mean_relative_delta > 0.05 or final_loss_abs_delta > 0.15:
        raise TrainingComparisonError(
            "candidate loss trajectory differs materially from control: "
            f"mean_relative_delta={loss_mean_relative_delta}, "
            f"final_abs_delta={final_loss_abs_delta}"
        )
    if not 0.5 <= median_norm_ratio <= 2.0 or not 1 / 3 <= p95_norm_ratio <= 3.0:
        raise TrainingComparisonError(
            "candidate gradient-norm distribution differs materially from control: "
            f"median_ratio={median_norm_ratio}, p95_ratio={p95_norm_ratio}"
        )

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "passed": True,
        "control": control_summary,
        "candidate": candidate_summary,
        "loss_mean_relative_delta": loss_mean_relative_delta,
        "final_loss_abs_delta": final_loss_abs_delta,
        "grad_norm_median_ratio": median_norm_ratio,
        "grad_norm_p95_ratio": p95_norm_ratio,
    }


def _matrix_curve_summary(metrics: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    losses = [float(item["loss"]) for item in metrics]
    norms = [float(item["grad_norm"]) for item in metrics]
    window = min(20, len(losses))
    return {
        "initial20_loss_median": statistics.median(losses[:window]),
        "final20_loss_median": statistics.median(losses[-window:]),
        "loss_mean": statistics.fmean(losses),
        "grad_norm_median": statistics.median(norms),
        "grad_norm_p95": _percentile(norms, 0.95),
        "grad_norm_max": max(norms),
    }


def _matrix_timing_median(report: Mapping[str, Any]) -> float:
    raw = report.get("step_times_ms")
    expected_steps = int(report.get("workload", {}).get("max_steps", -1))
    if not isinstance(raw, list) or len(raw) != expected_steps:
        raise TrainingComparisonError("step timing evidence is incomplete")
    values: list[float] = []
    for expected_step, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping) or item.get("step") != expected_step:
            raise TrainingComparisonError("step timings must be contiguous from one")
        milliseconds = float(item.get("milliseconds", float("nan")))
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise TrainingComparisonError("step timings must be finite and positive")
        values.append(milliseconds)
    warmup = report.get("timing_warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < len(values):
        raise TrainingComparisonError("timing warmup is invalid")
    return statistics.median(values[warmup:])


def _matrix_peak_memory(report: Mapping[str, Any]) -> tuple[int, int]:
    memory = report.get("peak_memory")
    if not isinstance(memory, Mapping):
        raise TrainingComparisonError("peak memory evidence is missing")
    allocated = int(memory.get("allocated_bytes", 0))
    reserved = int(memory.get("reserved_bytes", 0))
    if allocated <= 0 or reserved < allocated:
        raise TrainingComparisonError("peak memory evidence is invalid")
    return allocated, reserved


def compare_full_training_matrix(
    reports: Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the predeclared robust gate to three fresh SDPA/FA4 seed pairs."""

    if set(reports) != set(PROSPECTIVE_MATRIX_SEEDS):
        raise TrainingComparisonError(f"matrix seeds must be exactly {PROSPECTIVE_MATRIX_SEEDS}")
    thresholds = PROSPECTIVE_MATRIX_THRESHOLDS
    seed_results: dict[str, Any] = {}
    for seed in PROSPECTIVE_MATRIX_SEEDS:
        control, candidate = reports[seed]
        control_metrics = _validate_report(control, backend="sdpa")
        candidate_metrics = _validate_report(candidate, backend="native")
        for key in ("workload", "dataset", "parameters"):
            if control.get(key) != candidate.get(key):
                raise TrainingComparisonError(f"seed {seed}: {key} differs between reports")
        if control.get("workload", {}).get("seed") != seed:
            raise TrainingComparisonError(f"seed {seed}: workload seed is incorrect")

        control_summary = _matrix_curve_summary(control_metrics)
        candidate_summary = _matrix_curve_summary(candidate_metrics)
        control_losses = [float(item["loss"]) for item in control_metrics]
        candidate_losses = [float(item["loss"]) for item in candidate_metrics]
        loss_mean_delta = abs(candidate_summary["loss_mean"] - control_summary["loss_mean"]) / max(
            abs(control_summary["loss_mean"]), 1e-12
        )
        paired_loss_nmae = statistics.fmean(
            abs(candidate - control)
            for control, candidate in zip(control_losses, candidate_losses, strict=True)
        ) / max(abs(control_summary["loss_mean"]), 1e-12)
        final20_delta = abs(
            candidate_summary["final20_loss_median"] - control_summary["final20_loss_median"]
        ) / max(abs(control_summary["final20_loss_median"]), 1e-12)
        grad_median_ratio = (
            candidate_summary["grad_norm_median"] / control_summary["grad_norm_median"]
        )
        grad_p95_ratio = candidate_summary["grad_norm_p95"] / control_summary["grad_norm_p95"]
        control_step_ms = _matrix_timing_median(control)
        candidate_step_ms = _matrix_timing_median(candidate)
        speedup = control_step_ms / candidate_step_ms
        control_allocated, control_reserved = _matrix_peak_memory(control)
        candidate_allocated, candidate_reserved = _matrix_peak_memory(candidate)
        allocated_ratio = candidate_allocated / control_allocated
        reserved_ratio = candidate_reserved / control_reserved

        reasons: list[str] = []
        if control_summary["final20_loss_median"] >= control_summary["initial20_loss_median"]:
            reasons.append("SDPA loss did not improve from the initial to final window")
        if candidate_summary["final20_loss_median"] >= candidate_summary["initial20_loss_median"]:
            reasons.append("FA4 loss did not improve from the initial to final window")
        if loss_mean_delta > thresholds["per_seed_loss_mean_relative_delta_max"]:
            reasons.append("mean-loss relative delta exceeded the per-seed limit")
        if paired_loss_nmae > thresholds["per_seed_paired_loss_nmae_max"]:
            reasons.append("paired loss NMAE exceeded the per-seed limit")
        if final20_delta > thresholds["per_seed_final20_relative_delta_max"]:
            reasons.append("final-20 loss relative delta exceeded the per-seed limit")
        if not (
            thresholds["grad_norm_median_ratio_min"]
            <= grad_median_ratio
            <= thresholds["grad_norm_median_ratio_max"]
        ):
            reasons.append("gradient median ratio is outside the allowed interval")
        if not (
            thresholds["grad_norm_p95_ratio_min"]
            <= grad_p95_ratio
            <= thresholds["grad_norm_p95_ratio_max"]
        ):
            reasons.append("gradient P95 ratio is outside the allowed interval")
        if speedup < thresholds["per_seed_median_step_speedup_min"]:
            reasons.append("median step speedup missed the per-seed minimum")
        if max(allocated_ratio, reserved_ratio) > thresholds["peak_memory_ratio_max"]:
            reasons.append("candidate peak memory exceeded the per-seed limit")

        seed_results[str(seed)] = {
            "passed": not reasons,
            "reasons": reasons,
            "control": control_summary,
            "candidate": candidate_summary,
            "loss_mean_relative_delta": loss_mean_delta,
            "paired_loss_nmae": paired_loss_nmae,
            "final20_loss_relative_delta": final20_delta,
            "grad_norm_median_ratio": grad_median_ratio,
            "grad_norm_p95_ratio": grad_p95_ratio,
            "control_step_median_ms": control_step_ms,
            "candidate_step_median_ms": candidate_step_ms,
            "median_step_speedup": speedup,
            "peak_allocated_ratio": allocated_ratio,
            "peak_reserved_ratio": reserved_ratio,
        }

    def seed_median(field: str) -> float:
        return statistics.median(float(item[field]) for item in seed_results.values())

    aggregate = {
        "loss_mean_relative_delta_median": seed_median("loss_mean_relative_delta"),
        "paired_loss_nmae_median": seed_median("paired_loss_nmae"),
        "final20_loss_relative_delta_median": seed_median("final20_loss_relative_delta"),
        "median_step_speedup": seed_median("median_step_speedup"),
    }
    aggregate_reasons: list[str] = []
    if any(not item["passed"] for item in seed_results.values()):
        aggregate_reasons.append("one or more seed pairs failed a per-seed gate")
    if (
        aggregate["loss_mean_relative_delta_median"]
        > thresholds["median_loss_mean_relative_delta_max"]
    ):
        aggregate_reasons.append("median mean-loss relative delta exceeded its limit")
    if aggregate["paired_loss_nmae_median"] > thresholds["median_paired_loss_nmae_max"]:
        aggregate_reasons.append("median paired loss NMAE exceeded its limit")
    if (
        aggregate["final20_loss_relative_delta_median"]
        > thresholds["median_final20_relative_delta_max"]
    ):
        aggregate_reasons.append("median final-20 loss relative delta exceeded its limit")
    if aggregate["median_step_speedup"] < thresholds["median_step_speedup_min"]:
        aggregate_reasons.append("median seed-level speedup missed its limit")

    return {
        "schema_version": TRAINING_MATRIX_SCHEMA_VERSION,
        "passed": not aggregate_reasons,
        "reasons": aggregate_reasons,
        "seeds": list(PROSPECTIVE_MATRIX_SEEDS),
        "thresholds": dict(thresholds),
        "seed_results": seed_results,
        "aggregate": aggregate,
    }


__all__ = [
    "EXPECTED_MODEL_ID",
    "EXPECTED_MODEL_REVISION",
    "PROSPECTIVE_MATRIX_SEEDS",
    "PROSPECTIVE_MATRIX_THRESHOLDS",
    "TRAINING_BACKENDS",
    "TRAINING_MATRIX_SCHEMA_VERSION",
    "TRAINING_SCHEMA_VERSION",
    "TrainingComparisonError",
    "compare_full_training_matrix",
    "compare_full_training_reports",
    "mutate_full_training_config",
]
