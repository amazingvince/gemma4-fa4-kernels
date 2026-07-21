"""Axolotl plugin and Trainer callback for the EXP-0036 real-model harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Literal

import torch
from axolotl.integrations.base import BasePlugin
from pydantic import BaseModel, Field
from transformers import TrainerCallback

from .axolotl_harness import (
    HARNESS_SCHEMA_VERSION,
    initialize_lora_parameters,
    mutate_axolotl_config,
)
from .gemma4_12b_compat import (
    GEMMA4_12B_HARNESS,
    GEMMA4_12B_REVISION,
    register_gemma4_fa4_h100_12b_compat,
)
from .gemma4_native import (
    register_gemma4_fa4_h100_global_native,
    register_gemma4_fa4_h100_native,
)


class Fa4HarnessArgs(BaseModel):
    fa4_harness_backend: Literal[
        "native", "global_native", "project_12b_compat", "hybrid", "sdpa"
    ] = "project_12b_compat"
    fa4_harness_report_path: str = "agent_space/axolotl-exp0036/report.json"
    fa4_harness_dataset_path: str = "agent_space/axolotl-exp0036/gemma4-12b-smoke.jsonl"
    fa4_harness_warmup_steps: int = Field(default=3, ge=1)
    fa4_harness_gradient_values: int = Field(default=4096, ge=256, le=16384)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    value = getattr(cfg, key, default)
    return default if value is None else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_real_model(model: Any, cfg: Any) -> None:
    if _cfg_get(cfg, "base_model") != GEMMA4_12B_HARNESS.model_id:
        raise ValueError(f"EXP-0036 requires base_model={GEMMA4_12B_HARNESS.model_id}")
    if _cfg_get(cfg, "revision_of_model") != GEMMA4_12B_REVISION:
        raise ValueError(f"EXP-0036 requires revision_of_model={GEMMA4_12B_REVISION}")
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None)
    if text is None and hasattr(getattr(model, "base_model", None), "config"):
        config = model.base_model.config
        text = getattr(config, "text_config", config)
    if text is None:
        raise ValueError("EXP-0036 could not locate the Gemma 4 text configuration")
    expected = {
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "max_position_embeddings": 262_144,
        "num_kv_shared_layers": 0,
    }
    mismatches = {
        name: (getattr(text, name, None), value)
        for name, value in expected.items()
        if getattr(text, name, None) != value
    }
    layer_types = tuple(getattr(text, "layer_types", ()))
    if layer_types != GEMMA4_12B_HARNESS.layer_types():
        mismatches["layer_types"] = (layer_types, GEMMA4_12B_HARNESS.layer_types())
    if mismatches:
        raise ValueError(f"EXP-0036 model configuration mismatches the lock: {mismatches}")


def _gradient_sketch(model: Any, limit: int) -> dict[str, Any]:
    candidates = [
        (name, parameter.grad.detach())
        for name, parameter in sorted(model.named_parameters())
        if "lora_" in name and parameter.grad is not None and parameter.grad.numel() > 0
    ]
    if not candidates:
        raise RuntimeError("EXP-0036 did not observe any LoRA gradients")
    per_parameter = max(1, limit // len(candidates))
    names: list[str] = []
    values: list[float] = []
    for name, gradient in candidates:
        flat = gradient.float().reshape(-1)
        count = min(per_parameter, flat.numel(), limit - len(values))
        if count <= 0:
            break
        if count == flat.numel():
            selected = flat
        else:
            indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
            selected = flat.index_select(0, indices)
        names.append(name)
        values.extend(float(value) for value in selected.cpu().tolist())
    if not values or not any(value != 0.0 for value in values):
        raise RuntimeError("EXP-0036 LoRA gradient sketch is empty or all-zero")
    return {"parameter_names": names, "values": values}


def _collect_routes(model: Any) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    totals: dict[str, int] = {}
    layers: dict[str, dict[str, int]] = {}
    for module in model.modules():
        counts = getattr(module, "_gemma4_fa4_route_counts", None)
        layer_idx = getattr(module, "layer_idx", None)
        if not isinstance(counts, dict) or not isinstance(layer_idx, int):
            continue
        normalized = {str(route): int(count) for route, count in counts.items()}
        layers[str(layer_idx)] = normalized
        for route, count in normalized.items():
            totals[route] = totals.get(route, 0) + count
    return totals, layers


def _collect_native_evidence(model: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometries: dict[str, Any] = {}
    oracles: list[dict[str, Any]] = []
    for module in model.modules():
        layer_idx = getattr(module, "layer_idx", None)
        geometry = getattr(module, "_gemma4_fa4_native_geometry", None)
        oracle = getattr(module, "_gemma4_fa4_prepared_oracle", None)
        if isinstance(layer_idx, int) and isinstance(geometry, dict):
            geometries[str(layer_idx)] = geometry
        if isinstance(oracle, dict):
            oracles.append(oracle)
    return geometries, oracles


class Fa4HarnessCallback(TrainerCallback):
    def __init__(self, cfg: Any, model: Any):
        self.cfg = cfg
        self.model = model
        self.backend = str(_cfg_get(cfg, "fa4_harness_backend"))
        self.warmup_steps = int(_cfg_get(cfg, "fa4_harness_warmup_steps", 3))
        self.gradient_limit = int(_cfg_get(cfg, "fa4_harness_gradient_values", 4096))
        self.report_path = Path(str(_cfg_get(cfg, "fa4_harness_report_path")))
        self.dataset_path = Path(str(_cfg_get(cfg, "fa4_harness_dataset_path")))
        self.lora_initialization = initialize_lora_parameters(
            model,
            seed=int(_cfg_get(cfg, "seed")),
        )
        self._start_event: torch.cuda.Event | None = None
        self.measured_step_ms: list[float] = []
        self.losses: list[float] = []
        self.gradient_probe: dict[str, Any] | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._start_event is None:
            raise RuntimeError("EXP-0036 step ended without a CUDA start event")
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        if int(state.global_step) > self.warmup_steps:
            self.measured_step_ms.append(float(self._start_event.elapsed_time(end_event)))
        self._start_event = None
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and int(state.global_step) > self.warmup_steps:
            self.losses.append(float(logs["loss"]))
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.gradient_probe is None:
            self.gradient_probe = _gradient_sketch(
                kwargs.get("model", self.model), self.gradient_limit
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        routes, layer_routes = _collect_routes(self.model)
        native_geometries, prepared_oracles = _collect_native_evidence(self.model)
        expected_measured = int(_cfg_get(self.cfg, "max_steps")) - self.warmup_steps
        errors = []
        if len(self.measured_step_ms) != expected_measured:
            errors.append(
                f"expected {expected_measured} measured steps, got {len(self.measured_step_ms)}"
            )
        if len(self.losses) != expected_measured:
            errors.append(f"expected {expected_measured} measured losses, got {len(self.losses)}")
        if self.gradient_probe is None:
            errors.append("gradient probe was not captured")
        if self.backend in {"native", "global_native", "project_12b_compat"} and set(
            layer_routes
        ) != {str(index) for index in range(48)}:
            errors.append("project route evidence does not cover exactly 48 layers")
        if self.backend in {"native", "global_native"}:
            if set(native_geometries) != {str(index) for index in range(48)}:
                errors.append("native geometry evidence does not cover exactly 48 layers")
            if any(not item.get("k_v_distinct", False) for item in native_geometries.values()):
                errors.append("native geometry evidence observed aliased K/V")
            expected_sequence = int(_cfg_get(self.cfg, "sequence_len"))
            if any(
                item.get("q", [None, None, None])[2] != expected_sequence
                or item.get("k", [None, None, None])[2] != expected_sequence
                or item.get("v", [None, None, None])[2] != expected_sequence
                for item in native_geometries.values()
            ):
                errors.append("native prepared Q/K/V length does not match sequence_len")
            if bool(_cfg_get(self.cfg, "sample_packing", False)) and any(
                len(item.get("segments", [])) < 2 for item in native_geometries.values()
            ):
                errors.append("native packed route did not preserve multiple document segments")
            oracle_kinds = {item.get("kind") for item in prepared_oracles if item.get("passed")}
            if os.environ.get("GEMMA4_FA4_PREPARED_ORACLE") == "1" and oracle_kinds != {
                "sliding_attention",
                "full_attention",
            }:
                errors.append("prepared-QKV oracle did not cover local and global attention")
        dataset_hash = _sha256(self.dataset_path)
        properties = torch.cuda.get_device_properties(0)
        report = {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "status": "complete" if not errors else "incomplete",
            "errors": errors,
            "backend": self.backend,
            "workload": {
                "model_id": _cfg_get(self.cfg, "base_model"),
                "model_revision": _cfg_get(self.cfg, "revision_of_model"),
                "sequence_len": int(_cfg_get(self.cfg, "sequence_len")),
                "micro_batch_size": int(_cfg_get(self.cfg, "micro_batch_size")),
                "gradient_accumulation_steps": int(
                    _cfg_get(self.cfg, "gradient_accumulation_steps")
                ),
                "max_steps": int(_cfg_get(self.cfg, "max_steps")),
                "warmup_steps": self.warmup_steps,
                "seed": int(_cfg_get(self.cfg, "seed")),
                "dataset_sha256": dataset_hash,
            },
            "measured_step_ms": self.measured_step_ms,
            "losses": self.losses,
            "gradient_probe": self.gradient_probe,
            "lora_initialization": self.lora_initialization,
            "routes": routes,
            "layer_routes": layer_routes,
            "native_geometries": native_geometries,
            "prepared_qkv_oracles": prepared_oracles,
            "peak_memory": {
                "allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "reserved_bytes": int(torch.cuda.max_memory_reserved()),
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
            },
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.report_path)
        if errors:
            raise RuntimeError(f"EXP-0036 report is incomplete: {errors}")
        return control


class Fa4AxolotlHarnessPlugin(BasePlugin):
    def get_input_args(self) -> str:
        return "gemma4_fa4.axolotl_plugin.Fa4HarnessArgs"

    def register(self, cfg: dict):
        environment_overrides = {
            "fa4_harness_backend": os.getenv("GEMMA4_FA4_HARNESS_BACKEND"),
            "fa4_harness_report_path": os.getenv("GEMMA4_FA4_REPORT_PATH"),
            "fa4_harness_dataset_path": os.getenv("GEMMA4_FA4_DATASET_PATH"),
            "output_dir": os.getenv("GEMMA4_FA4_OUTPUT_DIR"),
        }
        for key, value in environment_overrides.items():
            if value:
                cfg[key] = value
        if os.getenv("GEMMA4_FA4_SEQUENCE_LEN"):
            cfg["sequence_len"] = int(os.environ["GEMMA4_FA4_SEQUENCE_LEN"])
        if os.getenv("GEMMA4_FA4_SAMPLE_PACKING"):
            raw_packing = os.environ["GEMMA4_FA4_SAMPLE_PACKING"].lower()
            if raw_packing not in {"0", "1", "false", "true"}:
                raise ValueError("GEMMA4_FA4_SAMPLE_PACKING must be true/false or 1/0")
            cfg["sample_packing"] = raw_packing in {"1", "true"}
        dataset_path = cfg.get("fa4_harness_dataset_path")
        if dataset_path and cfg.get("datasets"):
            cfg["datasets"][0]["path"] = dataset_path
            cfg["dataset_prepared_path"] = str(Path(dataset_path).parent / "prepared")
        mutate_axolotl_config(cfg)
        warmup = int(cfg.get("fa4_harness_warmup_steps", 3))
        if not 0 < warmup < int(cfg["max_steps"]):
            raise ValueError("fa4_harness_warmup_steps must be between zero and max_steps")
        if cfg["fa4_harness_backend"] in {"native", "global_native", "project_12b_compat"}:
            # Axolotl validates canonical attention names after plugin.register().
            # Extend that process-local allowlist explicitly for this registered
            # Transformers backend; no short alias or hub-kernel escape hatch.
            from axolotl.utils.schemas import config as axolotl_schema_config
            from axolotl.utils.schemas import enums as axolotl_schema_enums

            extended = frozenset(
                (*axolotl_schema_config.CANONICAL_ATTN_IMPLS, cfg["attn_implementation"])
            )
            axolotl_schema_config.CANONICAL_ATTN_IMPLS = extended
            axolotl_schema_enums.CANONICAL_ATTN_IMPLS = extended
            if cfg["fa4_harness_backend"] in {"native", "global_native"}:
                if cfg.get("sample_packing", False):
                    cfg["skip_prepare_dataset"] = False
                    cfg["dataset_num_proc"] = 1
                    packing = frozenset(
                        (
                            *axolotl_schema_config.ATTN_IMPLS_SUPPORTING_PACKING,
                            cfg["attn_implementation"],
                        )
                    )
                    axolotl_schema_config.ATTN_IMPLS_SUPPORTING_PACKING = packing
                    axolotl_schema_enums.ATTN_IMPLS_SUPPORTING_PACKING = packing
                if cfg["fa4_harness_backend"] == "global_native":
                    register_gemma4_fa4_h100_global_native()
                else:
                    register_gemma4_fa4_h100_native()
            else:
                register_gemma4_fa4_h100_12b_compat()

    def post_model_load(self, cfg, model):
        _validate_real_model(model, cfg)

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [Fa4HarnessCallback(cfg, trainer.model)]


__all__ = [
    "Fa4AxolotlHarnessPlugin",
    "Fa4HarnessArgs",
    "Fa4HarnessCallback",
]
