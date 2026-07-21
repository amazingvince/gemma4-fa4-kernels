#!/usr/bin/env python3
"""EXP-0042 all-layer guarded compiled-dispatch probe for H100."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import probe_h100_guarded_compile_facade as guarded
import probe_h100_torch_compile as base
import torch

from gemma4_fa4.model_spec import GEMMA4_31B
from gemma4_fa4.transformers_integration import compile_gemma4_fa4_h100_layer

EXPERIMENT = "EXP-0042"
BACKENDS = ("eager", "inductor")


def _family(layer_idx: int) -> str:
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    return "global" if spec.kind == "full_attention" else "local"


def _run_layer(
    layer_idx: int,
    *,
    backend: guarded._CapturingBackend,
    seqlen: int,
    seed: int,
) -> dict[str, Any]:
    family = _family(layer_idx)
    runtime = base._make_family_runtime(
        family,
        seed=seed,
        layer_idx=layer_idx,
    )
    inputs = base._make_inputs(runtime, seqlen, seed=seed + 1)
    eager_call = base._layer_callable(runtime)
    with torch.inference_mode():
        expected = eager_call(*inputs)
        facade = compile_gemma4_fa4_h100_layer(runtime.layer, backend=backend)
        hidden, cos, sin, positions = inputs
        output, attention_weights = facade(
            hidden,
            (cos, sin),
            position_ids=positions,
        )
    torch.cuda.synchronize()
    if attention_weights is not None or facade.compiled_entry_count != 1:
        raise AssertionError(f"layer {layer_idx} violated the guarded facade output contract")
    comparison = base._full_layer_comparison(
        output,
        expected,
        label=f"{backend.name}/layer-{layer_idx}/S{seqlen}",
    )
    del output, expected, facade, inputs, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "layer_idx": layer_idx,
        "family": family,
        "seqlen": seqlen,
        "comparison": comparison,
    }


def _run_backend(name: str, *, seqlen: int, seed: int) -> dict[str, Any]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    capture = guarded._CapturingBackend(name)
    results = [
        _run_layer(
            layer_idx,
            backend=capture,
            seqlen=seqlen,
            seed=seed + layer_idx * 10,
        )
        for layer_idx in range(GEMMA4_31B.num_hidden_layers)
    ]
    if base._graph_break_count() != 0:
        raise AssertionError(f"{name} all-layer sweep produced graph breaks")
    if len(capture.graphs) != 2:
        raise AssertionError(
            f"{name} all-layer sweep requires exactly two family graphs, got {len(capture.graphs)}"
        )
    op_counts = []
    for graph in capture.graphs:
        count = sum(
            node["op"] == "call_function"
            and "gemma4_fa4" in node["target"]
            and "layer_fwd" in node["target"]
            for node in graph["nodes"]
        )
        op_counts.append(count)
    if op_counts != [1, 1]:
        raise AssertionError(f"{name} graphs require one whole-layer op each: {op_counts}")
    forbidden = guarded._forbidden_inner_sources(capture.graphs)
    if forbidden:
        raise AssertionError(f"{name} graphs captured forbidden Python/scalar sources: {forbidden}")
    return {
        "status": "passed",
        "backend": name,
        "graph_count": len(capture.graphs),
        "graph_break_count": 0,
        "whole_layer_custom_op_counts": op_counts,
        "forbidden_inner_sources": forbidden,
        "layers": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    parser.add_argument("--seqlen", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.seqlen < 1 or args.seqlen > 1024:
        raise ValueError("--seqlen must be in 1..1024")
    base._require_h100()
    selected = BACKENDS if args.backend == "all" else (args.backend,)
    cache_dirs = base._prepare_fresh_cache_dirs(selected)
    application_before = base._forward_application_snapshot()
    backends = {
        name: _run_backend(name, seqlen=args.seqlen, seed=args.seed + index * 100_000)
        for index, name in enumerate(selected)
    }
    application_after = base._forward_application_snapshot()
    added = sorted(set(application_after) - set(application_before))
    if len(added) != 2 or not set(application_before).issubset(application_after):
        raise AssertionError(
            "all-layer sweep must add exactly one local and one global FA4 application class"
        )
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "passed",
        "boundary": "guarded_no_cache_tensor_only_all_60_layers",
        "request": {
            "backends": list(selected),
            "seqlen": args.seqlen,
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
            "caches": {name: base._cache_inventory(path) for name, path in cache_dirs.items()},
        },
        "layer_count": GEMMA4_31B.num_hidden_layers,
        "local_layer_count": sum(_family(index) == "local" for index in range(60)),
        "global_layer_count": sum(_family(index) == "global" for index in range(60)),
        "fa4_application_keys": {
            "before": list(application_before),
            "after": list(application_after),
            "added_family_classes": added,
            "new_class_count": 2,
        },
        "backends": backends,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        raise RuntimeError("EXP-0042 requires an isolated TORCHINDUCTOR_CACHE_DIR")
    main()
