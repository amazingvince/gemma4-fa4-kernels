---
name: gemma4-kernel-project
description: Use when working in this repository on Gemma 4 attention semantics, FA4 integration, GPU environments, benchmarks, experiments, or kernel roadmaps.
---

# Gemma 4 FA4 project router

## Required order

1. `AGENTS.md`
2. `docs/model-contract.md`
3. `docs/status.md`
4. the relevant section of `docs/design.md`
5. `docs/experimental-plan.md`
6. **REQUIRED SUB-SKILL for kernel work:** `writing-cute-dsl-kernels`

## Semantic stop signs

Stop and correct the task if it assumes any of the following:

- full-attention Q heads are 8 rather than 32;
- attention scale is `1/sqrt(d)` rather than 1.0;
- `attention_k_eq_v` makes prepared K and V identical;
- an attention backward may return one dKV without applying distinct K/V
  preparation adjoints;
- vision bidirectionality applies to global layers;
- a 1024 window includes key index `q-1024`;
- cross-layer KV reuse is assumed active; the checkpoint sets `num_kv_shared_layers=0`.

## Task routing

| Work | Read / run |
|---|---|
| Model fact | `configs/model/gemma4-31b.lock.json`, `docs/model-contract.md`, `scripts/verify_model_contract.py` |
| Remote host | `docs/remote-gpus.md`, `remote/*.env.example`, `scripts/remote/` |
| Environment refresh | `docs/environment.md`, `upstream.lock.json`; treat as an experiment boundary |
| Kernel design | `skills/writing-cute-dsl-kernels/templates/kernel-design-brief.md` |
| Kernel review | `skills/writing-cute-dsl-kernels/templates/review-checklist.md` |
| Correctness | `src/gemma4_fa4/`, `tests/`, reference § in the CuTe skill |
| Benchmark | `benchmarks/bench_attention.py`; use equal semantics and distinct modes |
| Experiment | `experiments/TEMPLATE.md`, then `scripts/record_result.py` |

## Definition of a valid iteration

Report the pinned environment, exemplar revision, invariant changed, compile
result, correctness matrix, sanitizer evidence, generated-code observation,
benchmark method, result, cache/variant impact, and retain/revert/refine
decision. Unsupported or unrun evidence is stated explicitly.
