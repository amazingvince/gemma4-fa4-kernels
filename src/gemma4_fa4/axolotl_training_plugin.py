"""Axolotl plugin for matched, real-update Gemma 4 full-BF16 training runs."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import torch
from axolotl.integrations.base import BasePlugin
from pydantic import BaseModel, Field
from transformers import TrainerCallback

from .axolotl_plugin import _collect_native_evidence, _collect_routes, _validate_real_model
from .axolotl_training_harness import (
    TRAINING_SCHEMA_VERSION,
    align_trainable_parameters_to_bf16,
    mutate_full_training_config,
)
from .gemma4_native import (
    register_gemma4_fa4_h100_global_native,
    register_gemma4_fa4_h100_native,
)


class Fa4FullTrainingArgs(BaseModel):
    fa4_training_backend: Literal["native", "global_native", "sdpa"] = "native"
    fa4_training_report_path: str = "agent_space/axolotl-full-training/report.json"
    fa4_training_dataset_path: str = "agent_space/axolotl-full-training/alpaca.jsonl"
    fa4_training_expected_steps: int = Field(default=100, ge=1, le=100)
    fa4_training_timing_warmup_steps: int = Field(default=5, ge=0, le=99)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    value = getattr(cfg, key, default)
    return default if value is None else value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parameter_evidence(model: Any) -> dict[str, Any]:
    trainable = 0
    frozen = 0
    adapter_parameters = 0
    dtype_counts: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
            dtype_counts[str(parameter.dtype)] += count
            if "lora_" in name.lower() or "adapter" in name.lower():
                adapter_parameters += count
        else:
            frozen += count
    return {
        "trainable": trainable,
        "frozen": frozen,
        "adapter_parameters": adapter_parameters,
        "trainable_dtypes": dict(sorted(dtype_counts.items())),
    }


def _validate_routes(
    backend: str,
    routes: dict[str, int],
    layer_routes: dict[str, dict[str, int]],
    *,
    expected_steps: int,
) -> list[str]:
    errors: list[str] = []
    if backend == "sdpa":
        if routes or layer_routes:
            errors.append("the SDPA control recorded a project FA4/FA2 route")
        return errors
    if set(layer_routes) != {str(index) for index in range(48)}:
        return ["route evidence does not cover exactly Gemma 4 12B layers 0..47"]
    rebuilt: Counter[str] = Counter()
    for layer_idx in range(48):
        observed = layer_routes[str(layer_idx)]
        family = "global" if (layer_idx + 1) % 6 == 0 else "local"
        expected_route = f"fa4_native/{family}_fixed"
        if backend == "global_native" and family == "local":
            expected_route = "fa2_local/fixed"
        if set(observed) != {expected_route}:
            errors.append(f"layer {layer_idx} used {sorted(observed)} instead of {expected_route}")
            continue
        count = int(observed[expected_route])
        if count < expected_steps or count % expected_steps != 0:
            errors.append(
                f"layer {layer_idx} recorded {count} calls; expected a positive integer "
                f"multiple of {expected_steps} with checkpoint recomputation"
            )
        rebuilt[expected_route] += count
    if dict(rebuilt) != {str(key): int(value) for key, value in routes.items()}:
        errors.append("route totals do not equal the per-layer evidence")
    return errors


class Fa4FullTrainingCallback(TrainerCallback):
    def __init__(self, cfg: Any, model: Any):
        self.cfg = cfg
        self.model = model
        self.backend = str(_cfg_get(cfg, "fa4_training_backend"))
        self.expected_steps = int(_cfg_get(cfg, "fa4_training_expected_steps"))
        self.timing_warmup = int(_cfg_get(cfg, "fa4_training_timing_warmup_steps", 5))
        self.report_path = Path(str(_cfg_get(cfg, "fa4_training_report_path")))
        self.dataset_path = Path(str(_cfg_get(cfg, "fa4_training_dataset_path")))
        self.precision_alignment = getattr(
            model,
            "_gemma4_fa4_precision_alignment",
            {
                "target_dtype": "torch.bfloat16",
                "converted_elements": 0,
                "converted_parameters": [],
            },
        )
        self.parameters = _parameter_evidence(model)
        self._active_step: int | None = None
        self._start_event: torch.cuda.Event | None = None
        self._times: dict[int, float] = {}
        self._metrics: dict[int, dict[str, float | int]] = {}
        self._peak_allocated = 0
        self._peak_reserved = 0

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._active_step = int(state.global_step) + 1
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._start_event is None or self._active_step is None:
            raise RuntimeError("full-training step ended without an active CUDA timing event")
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        step = int(state.global_step)
        if step != self._active_step:
            raise RuntimeError(f"Axolotl step accounting changed: {self._active_step} -> {step}")
        self._times[step] = float(self._start_event.elapsed_time(end_event))
        self._peak_allocated = max(self._peak_allocated, int(torch.cuda.max_memory_allocated()))
        self._peak_reserved = max(self._peak_reserved, int(torch.cuda.max_memory_reserved()))
        self._active_step = None
        self._start_event = None
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control
        step = int(state.global_step)
        if step <= 0:
            return control
        required = ("loss", "grad_norm", "learning_rate")
        missing = [key for key in required if key not in logs]
        if missing:
            raise RuntimeError(f"training log at step {step} is missing {missing}")
        values = {key: float(logs[key]) for key in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"training log at step {step} contains a nonfinite metric")
        self._metrics[step] = {"step": step, **values}
        return control

    def on_train_end(self, args, state, control, **kwargs):
        routes, layer_routes = _collect_routes(self.model)
        native_geometries, _prepared_oracles = _collect_native_evidence(self.model)
        errors = _validate_routes(
            self.backend,
            routes,
            layer_routes,
            expected_steps=self.expected_steps,
        )
        expected = set(range(1, self.expected_steps + 1))
        if set(self._metrics) != expected:
            errors.append("loss/gradient metrics do not cover every requested update step")
        if set(self._times) != expected:
            errors.append("CUDA timings do not cover every requested update step")
        if self.backend == "sdpa":
            if native_geometries:
                errors.append("the SDPA control recorded native FA4 geometry evidence")
        else:
            if set(native_geometries) != {str(index) for index in range(48)}:
                errors.append("native geometry evidence does not cover exactly 48 layers")
            if any(not item.get("k_v_distinct", False) for item in native_geometries.values()):
                errors.append("prepared K/V storage aliasing was observed")
        sequence_len = int(_cfg_get(self.cfg, "sequence_len"))
        for item in native_geometries.values() if self.backend != "sdpa" else ():
            q_length = item.get("q", [None, None, None])[2]
            k_length = item.get("k", [None, None, None])[2]
            v_length = item.get("v", [None, None, None])[2]
            if (
                not isinstance(q_length, int)
                or not 1 <= q_length <= sequence_len
                or q_length != k_length
                or k_length != v_length
            ):
                errors.append("prepared Q/K/V lengths must be equal and within 1..sequence_len")
                break
        if self.parameters["adapter_parameters"] != 0:
            errors.append("adapter parameters were trainable during full-parameter training")
        if self.parameters["trainable_dtypes"] != {"torch.bfloat16": self.parameters["trainable"]}:
            errors.append("not every trainable model parameter was BF16")

        provenance_path = self.dataset_path.with_suffix(
            self.dataset_path.suffix + ".provenance.json"
        )
        if provenance_path.is_file():
            dataset = json.loads(provenance_path.read_text(encoding="utf-8"))
        else:
            dataset = {"error": f"missing dataset provenance {provenance_path}"}
            errors.append("dataset provenance is missing")

        properties = torch.cuda.get_device_properties(0)
        report = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "status": "complete" if not errors else "incomplete",
            "errors": errors,
            "backend": self.backend,
            "workload": {
                "model_id": _cfg_get(self.cfg, "base_model"),
                "model_revision": _cfg_get(self.cfg, "revision_of_model"),
                "sequence_len": sequence_len,
                "micro_batch_size": int(_cfg_get(self.cfg, "micro_batch_size")),
                "gradient_accumulation_steps": int(
                    _cfg_get(self.cfg, "gradient_accumulation_steps")
                ),
                "max_steps": self.expected_steps,
                "seed": int(_cfg_get(self.cfg, "seed")),
                "optimizer": _cfg_get(self.cfg, "optimizer"),
                "learning_rate": float(_cfg_get(self.cfg, "learning_rate")),
                "gradient_checkpointing": bool(_cfg_get(self.cfg, "gradient_checkpointing")),
            },
            "dataset": dataset,
            "precision_alignment": self.precision_alignment,
            "parameters": self.parameters,
            "training_metrics": [self._metrics[step] for step in sorted(self._metrics)],
            "step_times_ms": [
                {"step": step, "milliseconds": self._times[step]} for step in sorted(self._times)
            ],
            "timing_warmup_steps": self.timing_warmup,
            "routes": routes,
            "layer_routes": layer_routes,
            "native_geometries": native_geometries,
            "peak_memory": {
                "allocated_bytes": self._peak_allocated,
                "reserved_bytes": self._peak_reserved,
            },
            "environment": {
                "python": os.sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": _package_version("transformers"),
                "axolotl": _package_version("axolotl"),
                "flash_attn_cute": _package_version("flash-attn-cute"),
                "cuda_runtime": torch.version.cuda,
                "device_name": properties.name,
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
                "fa4_cache_dir": os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"),
                "project_revision": os.getenv("GEMMA4_FA4_SOURCE_REVISION"),
                "flash_attention_revision": os.getenv("FLASH_ATTENTION_SOURCE_REVISION"),
            },
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.report_path)
        if errors:
            raise RuntimeError(f"full-training report is incomplete: {errors}")
        return control


class Fa4FullTrainingPlugin(BasePlugin):
    def get_input_args(self) -> str:
        return "gemma4_fa4.axolotl_training_plugin.Fa4FullTrainingArgs"

    def register(self, cfg: dict):
        environment_overrides = {
            "fa4_training_backend": os.getenv("GEMMA4_FA4_TRAINING_BACKEND"),
            "fa4_training_report_path": os.getenv("GEMMA4_FA4_TRAINING_REPORT_PATH"),
            "fa4_training_dataset_path": os.getenv("GEMMA4_FA4_TRAINING_DATASET_PATH"),
            "output_dir": os.getenv("GEMMA4_FA4_OUTPUT_DIR"),
        }
        for key, value in environment_overrides.items():
            if value:
                cfg[key] = value
        if os.getenv("GEMMA4_FA4_SEQUENCE_LEN"):
            cfg["sequence_len"] = int(os.environ["GEMMA4_FA4_SEQUENCE_LEN"])
        if os.getenv("GEMMA4_FA4_MAX_STEPS"):
            steps = int(os.environ["GEMMA4_FA4_MAX_STEPS"])
            cfg["max_steps"] = steps
            cfg["fa4_training_expected_steps"] = steps
            cfg["fa4_training_timing_warmup_steps"] = min(5, max(0, steps - 1))
        dataset_path = cfg.get("fa4_training_dataset_path")
        if dataset_path and cfg.get("datasets"):
            cfg["datasets"][0]["path"] = dataset_path
            cfg["dataset_prepared_path"] = str(Path(dataset_path).parent / "prepared")
        mutate_full_training_config(cfg)

        from axolotl.utils.schemas import config as axolotl_schema_config
        from axolotl.utils.schemas import enums as axolotl_schema_enums

        if cfg["fa4_training_backend"] != "sdpa":
            extended = frozenset(
                (*axolotl_schema_config.CANONICAL_ATTN_IMPLS, cfg["attn_implementation"])
            )
            axolotl_schema_config.CANONICAL_ATTN_IMPLS = extended
            axolotl_schema_enums.CANONICAL_ATTN_IMPLS = extended
            if cfg["fa4_training_backend"] == "native":
                register_gemma4_fa4_h100_native()
            else:
                register_gemma4_fa4_h100_global_native()

    def post_model_load(self, cfg, model):
        _validate_real_model(model, cfg)
        align_trainable_parameters_to_bf16(model)

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [Fa4FullTrainingCallback(cfg, trainer.model)]


__all__ = [
    "Fa4FullTrainingArgs",
    "Fa4FullTrainingCallback",
    "Fa4FullTrainingPlugin",
]
