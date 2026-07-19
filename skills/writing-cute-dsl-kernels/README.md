# Writing CuTe DSL Kernels — Coding-Agent Skill

A research-backed skill package for designing, implementing, reviewing, debugging, and tuning NVIDIA CuTe DSL GPU kernels.

## Install

Copy the entire `writing-cute-dsl-kernels` directory into the skill directory used by the coding agent. The entry point is `SKILL.md`.

Do not copy only `SKILL.md`: it routes detailed tasks to the reference and template files.

## Contents

```text
writing-cute-dsl-kernels/
├── SKILL.md
├── README.md
├── manifest.json
├── MERGE_NOTES.md
├── references/
│   ├── 01-execution-model.md
│   ├── 02-layouts-tensors-partitioning.md
│   ├── 03-kernel-family-patterns.md
│   ├── 04-architecture-playbooks.md
│   ├── 05-memory-pipelines-synchronization.md
│   ├── 06-jit-integration-versioning.md
│   ├── 07-correctness-debugging-profiling.md
│   ├── 08-source-map.md
│   ├── 09-agent-assisted-development.md
│   └── 10-production-exemplars-and-learning-map.md
├── templates/
│   ├── kernel-design-brief.md
│   ├── implementation-shape.md
│   ├── review-checklist.md
│   └── benchmark-report.md
├── tests/
│   ├── evaluation-prompts.md
│   └── static-checks.md
└── scripts/
    └── validate_skill.py
```

## Validate

```bash
python writing-cute-dsl-kernels/scripts/validate_skill.py
```

This performs structural/static checks only. It cannot establish that a generated kernel compiles, is race-free, or is fast.

## Runtime Verification

For deployment, run the evaluation prompts in fresh agent contexts, then compile agent-generated kernels against the pinned CuTe DSL revision on the target NVIDIA GPU. Run correctness tests, compute-sanitizer, generated-code inspection, and reproducible benchmarks.

## Design Choices

The main skill is a decision router rather than a giant API dump. It forces:

- environment/version identification;
- kernel-family and architecture classification;
- explicit layout and ownership proof;
- architecture-specific memory and MMA dataflow;
- complete async pipeline state machines;
- coordinate-tensor predication;
- correctness and sanitizer gates before performance;
- measured, reproducible tuning.

Detailed material is split into references so an agent can load only the relevant context.

## Scope

Included:

- CuTe DSL execution and JIT mental model;
- static/dynamic layouts and specialization control;
- tensors, layout algebra, tiling, TiledCopy/TiledMMA, predication;
- elementwise, transform, reduction, GEMV, GEMM, attention, persistent/grouped, mixed/block-scaled patterns;
- Ampere/Ada warp MMA, Hopper WGMMA, Blackwell SM100/103/110 tcgen05/TMEM, and SM120/121 warp MMA;
- `cp.async`, TMA, barriers, multistage pipelines, and Blackwell CLC scheduling;
- guarded NVVM/LLVM/inline-PTX extension paths;
- DLPack/framework integration, caching, `cute.compile_to`, debug/profiling/autotuning;
- exemplar-first and agent-assisted experimental workflows;
- production exemplar and learning map;
- source map, templates, and 17 evaluation prompts.

Not included:

- a promise that API spellings remain valid across revisions;
- a universal optimal tile;
- compiled binaries;
- runtime performance data;
- replacement for matching official examples and installed-source inspection.

## Research Baseline

Compiled from official NVIDIA CUTLASS documentation and repository material accessed on 2026-07-18. See `references/08-source-map.md`. The skill intentionally requires a version gate because the DSL and helpers continue to evolve.


## Merged Provenance

Version 1.1 folds in the user-supplied `cutedsl-kernel-writing.skill` package, especially its production-exemplar, SM103/SM120, fake-tensor workflow, debugging, learning-resource, and AI-assisted-development insights. Project-specific claims were either qualified or omitted. See `MERGE_NOTES.md`.
