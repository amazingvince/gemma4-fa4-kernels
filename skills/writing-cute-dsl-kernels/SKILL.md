---
name: writing-cute-dsl-kernels
description: Use when designing, implementing, porting, reviewing, debugging, profiling, or tuning NVIDIA CuTe DSL or CuTeDSL GPU kernels and agentic kernel workflows, especially work involving @cute.jit, @cute.kernel, layouts, TensorSSA, tiled copy or MMA, cp.async, TMA, WGMMA, tcgen05, TMEM, pipelines, clusters, predication, JIT caching, framework integration, FlashAttention-style kernels, KernelBench, or AI-assisted GPU kernel generation.
---

# Writing CuTe DSL GPU Kernels

## Overview

A CuTe DSL kernel is not ordinary Python and should not be designed as a line-by-line CUDA translation. Treat it as the composition of four contracts:

1. **Logical contract** — shapes, dtypes, strides, broadcasting, reduction semantics, numerics, and aliasing.
2. **Layout contract** — how coordinates map to memory and how tiles are partitioned across CTAs, warps or warpgroups, threads, and values.
3. **Hardware dataflow contract** — the architecture-specific path through GMEM, SMEM, RMEM, and, where applicable, TMEM.
4. **Synchronization contract** — ownership and lifecycle of every asynchronous copy, MMA group, barrier, and pipeline stage.

**Core principle:** derive code from those contracts, prove the layout and synchronization invariants, establish correctness, and only then tune performance.

The CuTe DSL is evolving. Never assume an API spelling, helper, or example is stable across installed versions.

## Mandatory Start Gate

Before writing kernel code, record all of the following in a design brief:

| Field | Required decision |
|---|---|
| Environment | Installed `nvidia-cutlass-dsl` or CUTLASS revision, CUDA toolkit, Python, GPU model, compute capability |
| Operation | Mathematical formula, accumulation type, output conversion, epilogue, side effects |
| Input contract | Shapes, strides, alignment, dtype, device, stream, aliasing, batch/group semantics |
| Shape regime | Fixed, bounded dynamic, fully dynamic; representative and adversarial sizes |
| Target | Exact SM architecture and whether portability or a specialized fast path is required |
| Kernel family | Elementwise/copy, reduction, GEMV, GEMM, attention, grouped/persistent, block-scaled, other |
| Baseline | Trusted reference implementation and tolerance policy |
| Measurement | Warmup, synchronization, timing method, workload set, comparison target |

If any field is unknown, preserve it as an explicit assumption in the implementation and tests. Do not silently choose a GPU architecture or tensor layout.

## Version Gate

Perform this before copying any API from documentation or an example:

1. Inspect the installed package version or repository revision.
2. Check the matching changelog and limitations page.
3. Find the closest official example in that same revision.
4. Verify imported symbols and signatures against the installed source or generated API docs.
5. Pin the version in tests, benchmark reports, and generated code comments.
6. Prefer a small adapter around volatile helpers rather than spreading unstable APIs through the kernel.

Use `references/06-jit-integration-versioning.md` and `references/08-source-map.md`.

## Exemplar-First Gate

Do not begin an advanced CuTe DSL kernel from a blank file when a close, verified exemplar exists.

1. Classify the exact target architecture, kernel family, dtype path, and shape regime.
2. Find the closest **official** example in the matching CUTLASS/CuTe DSL revision. This is the API and architecture baseline.
3. When useful, inspect a production exemplar such as FlashAttention-4 or QuACK for repository structure, testing, scheduling, or tuning patterns. Treat it as implementation evidence, not API truth.
4. Record the source path, revision, target SM, copied invariants, and every adaptation in the design brief.
5. Preserve the exemplar's tests while reducing it to the smallest version that still proves the intended dataflow.
6. Start from a blank implementation only when no close exemplar exists; then build the simplest synchronous or single-stage correct form before adding overlap.

Never transplant a helper, barrier count, stage policy, cache key, or architecture branch without restating why its assumptions hold in the new kernel. Use `references/10-production-exemplars-and-learning-map.md`.

## Architecture Router

Never mix these dataflows merely because their operations have similar names.

| Target | Compute pattern | Operand path | Accumulator | Characteristic synchronization |
|---|---|---|---|---|
| SM80/SM86/SM89-class warp MMA | Warp-level `mma.sync` selected through CuTe tiled MMA | GMEM → SMEM, then SMEM → RMEM fragments, commonly with `ldmatrix`-style tiled copies | RMEM | CTA synchronization around shared-memory reuse; `cp.async` group commit/wait when used |
| SM90 Hopper WGMMA | Warpgroup MMA | GMEM → SMEM, then MMA consumes SMEM descriptors for A/B as supported | RMEM | TMA/mbarrier lifecycle plus warpgroup fence, commit-group, and wait-group ordering |
| SM100/SM103/SM110 Blackwell tcgen05 family | tcgen05 MMA, one-CTA or two-CTA variants only where the exact op supports the target | GMEM → SMEM; MMA operands/descriptors and scale factors follow the selected atom | TMEM for accumulators, then TMEM → RMEM for epilogue | TMEM allocation/deallocation, TMA barriers, tcgen05 MMA group ordering, CTA-group coordination |
| SM120/SM121 Blackwell warp-MMA family | Warp-level `mma.sync`, including supported block-scaled forms; **not** the SM100 tcgen05/TMEM path | GMEM → SMEM → RMEM fragments and scale factors according to the selected warp atom | RMEM | Warp convergence plus copy/barrier ordering; no tcgen05 allocation or MMA groups |
| Other or newer SM | Current official architecture example only | Do not extrapolate | Verify | Verify against current docs and generated code |

When the requested target is absent or ambiguous, design a portable fallback and isolate the architecture-specific fast path.

**Cluster Launch Control note:** on supported Blackwell targets, CLC can dynamically schedule persistent cluster work, but it does not change the selected MMA/mainloop contract. Derive its producer, consumer, 16-byte response, cross-proxy fence, coordinate mapping, and coordinated shutdown from the current official scheduler/pipeline model.

## Kernel-Family Router

| Family | Begin with | Main risk |
|---|---|---|
| Elementwise, conversion, fused pointwise | Direct tiled GMEM access; vectorize only after proving alignment and tails | Out-of-bounds vector lanes, poor coalescing, excess recompilation |
| Copy, transpose, layout transform | TiledCopy and explicit source/destination partitions; add SMEM only when it repairs access order or enables reuse | Incorrect partition congruence, bank conflicts, alignment assumptions |
| Reduction, norm, softmax | Hierarchical thread → warp → CTA reduction; stable accumulation; explicit all-masked and empty behavior | Numerical instability, barrier divergence, excessive SMEM |
| GEMV or small-K matmul | Match parallel axis and reduction width to the shape regime; compare SIMT and MMA routes | Underutilization and launch overhead |
| GEMM-like | Architecture playbook, tiled MMA, staged mainloop, separate epilogue | K-tail handling, synchronization, register/SMEM pressure |
| Attention or multi-MMA fusion | Compose proven tiled MMA and reduction pieces; state online-softmax invariants | Live-range explosion, numerical error, pipeline dependency errors |
| Grouped, persistent, MoE | Scheduler and work-queue contract first; validate per-problem metadata | Deadlock, load imbalance, stale descriptor/state reuse |
| Mixed-input or block-scaled | Official architecture example for dtype and scale-factor layout | Scale-factor addressing, conversion semantics, unsupported atom combinations |

Load `references/03-kernel-family-patterns.md` after choosing a family.

## Required Design Workflow

### 1. Specify the logical operation

Write the exact index formula. State accumulation precision, NaN/Inf behavior, rounding/conversion, masked values, and whether output may alias input. Define behavior for zero extents and partial tiles.

### 2. Choose static versus dynamic information

Specialize only values that materially alter generated code: dtype, architecture, tile shape, stage count, MMA atom, transpose mode, or epilogue kind. Keep ordinary dimensions and strides dynamic when one compiled kernel should cover them.

Estimate the specialization count before implementation:

`variants = product(number of values for every constexpr dimension)`

If the expected variant count is large or unbounded, redesign the interface.

### 3. Draw the dataflow

For each operand and output, write its route and owner:

`logical tensor → CTA tile → thread/warp partition → memory space → compute fragment → epilogue → store`

Include the producer and consumer agent for every stage.

### 4. Prove layouts before compute

For every tensor produced by tiling or partitioning, record:

- logical modes and mode order;
- shape and stride, including static versus dynamic components;
- memory space and alignment;
- owner mapping: CTA, warp/warpgroup, thread, and value;
- congruence with the copy or MMA partition;
- in-bounds predicate or padding invariant.

Use compile-time assertions where possible. Print or visualize small layouts during development. Do not proceed because a layout “looks plausible.”

### 5. Select movement primitives

Choose the simplest mechanism that meets the dataflow:

- direct vector/scalar GMEM access for one-pass, naturally coalesced traffic;
- tiled synchronous copy for explicit ownership and vectorization;
- `cp.async`-family movement when the target and pipeline benefit;
- TMA for supported multidimensional asynchronous transfers, multicast, or architecture playbooks that require it;
- SMEM swizzling only to solve a demonstrated bank-conflict or instruction-layout requirement.

State alignment, transaction shape, zero-fill or predication behavior, and synchronization.

### 6. Select the compute atom

Choose an instruction atom supported by the exact architecture, dtypes, operand major modes, accumulation type, and tile shape. Build the tiled MMA around the atom. Derive A/B/C partitions and fragments from the tiled MMA; do not hand-invent lane maps.

### 7. Write the synchronization state machine

For every asynchronous operation, name these transitions:

`acquire → issue → commit/arrive → wait → consume → release`

Specify who executes each transition, which stage index and phase it uses, whether all required threads participate, and how the pipeline drains. A missing tail or release is a correctness bug, not a tuning issue.

### 8. Design boundaries and epilogue

Apply identical tiling/partitioning to a coordinate or identity tensor to derive predicates. Mask invalid loads and stores; choose neutral fill values for invalid reduction/MMA inputs. Keep accumulator conversion and output predication separate from the mainloop.

### 9. Implement in layers

Use this order:

1. host/JIT argument validation and launch;
2. kernel indexing and CTA tile selection;
3. layouts, storage, and partitions;
4. one-stage copy;
5. compute without overlap;
6. epilogue;
7. correctness tests;
8. asynchronous/multistage pipeline;
9. profiling and tuning.

Never introduce a multistage pipeline before a synchronous or single-stage version is correct unless the instruction itself requires asynchronous machinery.

### 10. Verify, inspect, and benchmark

Pass the correctness matrix, run race and memory checking, inspect generated artifacts when performance is unexplained, then benchmark with stable methodology. A faster result that fails an edge case is not a valid candidate.

## Agentic Development Gate

When a coding agent writes or tunes the kernel, require an observable loop rather than a single speculative generation:

1. snapshot package/revision, CUDA, GPU, framework, commands, and environment variables;
2. anchor the change to an official example and, when useful, a named production exemplar;
3. establish a reference test and a measured baseline before optimization;
4. change one conceptual variable at a time—layout, tile, stage count, role split, epilogue, or cache policy;
5. compile and run the smallest discriminating correctness case;
6. run sanitizer or synchronization diagnostics after pipeline, barrier, cluster, TMEM, or role changes;
7. inspect IR/PTX/SASS and profiler evidence before explaining a performance change;
8. record the hypothesis, diff, results, and retain/revert decision in project-local scratch notes;
9. require human review at pipeline/barrier design, TMEM lifetime, warp-role partitioning, scale-factor layout, and persistent scheduler boundaries.

The agent must not promote scratch code merely because it compiles or wins one friendly benchmark. Use `references/09-agent-assisted-development.md` and the evaluation prompts.

## Hard Rules

- Do not invent CuTe DSL APIs. Verify against the installed version.
- Do not generate an advanced asynchronous or tensor-core kernel from a blank page when a close verified exemplar exists.
- Do not use architecture-specific instructions without an explicit target and capability guard.
- Do not use a low-level NVVM/LLVM or inline-PTX escape hatch until public CuTe/architecture operations have been checked; isolate and verify every retained helper.
- Do not derive lane ownership with ad hoc arithmetic when TiledCopy/TiledMMA can derive it.
- Do not reuse SMEM/TMEM until every prior consumer has completed.
- Do not place a barrier in control flow that can diverge across its participating agent set.
- Do not assume dimensions divide tile sizes; prove divisibility or predicate.
- Do not mark dimensions/layouts static merely to make compilation succeed.
- Do not claim a performance win without device, clocks/environment, shapes, warmup, synchronization, statistics, and a baseline.
- Do not treat example kernels as benchmark results; measure the actual workload.
- Do not explain a performance change from intuition alone when generated-code or profiler evidence is available.
- Do not optimize only one friendly shape. Include tail-heavy, small, large, and layout-variant cases.

## Deliverable Contract for a Coding Agent

A completed kernel task should include:

1. **Version note** — package/revision, CUDA, target SM, and volatile APIs verified.
2. **Exemplar provenance** — official source path/revision, optional production exemplar, inherited assumptions, and adaptations.
3. **Design brief** — completed `templates/kernel-design-brief.md`.
4. **Implementation** — host/JIT wrapper, kernel, explicit launch configuration, and architecture guard/fallback.
5. **Invariant notes** — layout shapes/strides, ownership, pipeline lifecycle, alignment, predication, and resource assumptions.
6. **Tests** — reference comparison across the declared matrix plus failure tests for invalid contracts.
7. **Debug evidence** — sanitizer/race-check result for representative cases.
8. **Benchmark report** — completed `templates/benchmark-report.md`; no unsupported conclusions.
9. **Iteration record** — hypotheses, one-variable changes, tool outputs, and retain/revert decisions for agent-assisted work.
10. **Review record** — completed `templates/review-checklist.md`.

## Reference Loading Map

Load only what the task needs:

| Situation | Read |
|---|---|
| Confusion about Python/JIT/device behavior, control flow, constexpr | `references/01-execution-model.md` |
| Layout, tensor, tiling, partitioning, predication, naming | `references/02-layouts-tensors-partitioning.md` |
| Choosing an algorithmic kernel structure | `references/03-kernel-family-patterns.md` |
| Warp MMA, WGMMA, tcgen05/TMEM, Blackwell Cluster Launch Control | `references/04-architecture-playbooks.md` |
| `cp.async`, TMA, barriers, stages, pipeline deadlocks | `references/05-memory-pipelines-synchronization.md` |
| Package compatibility, caching, DLPack/framework calls | `references/06-jit-integration-versioning.md` |
| Wrong answers, crashes, sanitizer, IR/PTX/SASS, profiling, tuning | `references/07-correctness-debugging-profiling.md` |
| Source authority, official docs, production repositories, and evidence tiers | `references/08-source-map.md` |
| AI/LLM-assisted kernel development, iteration loops, tool contracts, and evaluation | `references/09-agent-assisted-development.md` |
| FlashAttention/QuACK patterns, exemplar selection, and task-specific learning routes | `references/10-production-exemplars-and-learning-map.md` |

Use the templates before coding and the evaluation prompts when testing an agent that consumes this skill.
