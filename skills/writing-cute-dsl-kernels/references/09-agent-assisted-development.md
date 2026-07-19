# Agent-Assisted CuTe DSL Kernel Development

## Purpose

Use this reference when a coding agent or LLM is generating, porting, debugging, or tuning a CuTe DSL kernel. It defines an evidence-driven interaction loop. It does not lower the correctness bar or authorize the agent to invent APIs.

**Core principle:** the agent should operate as a measured experimentalist around verified abstractions, not as a one-shot code generator.

## Capability Calibration

A useful difficulty ladder is:

| Work class | Typical agent reliability | Required response |
|---|---|---|
| Elementwise fusion, simple conversion, direct tiled copy | Often tractable with a clear contract and tests | Generate a scalar/tiled baseline, prove tails, then vectorize |
| Reductions, norms, softmax, layout transforms | Tractable only with explicit numerical and ownership invariants | Start from a matching reduction exemplar; stress ragged and extreme cases |
| Architecture GEMM adaptation | High risk without a close official example | Preserve the example's dataflow, atom, partitions, and synchronization; change one dimension at a time |
| WGMMA/tcgen05/TMEM, multi-stage attention, persistent clusters, 2-CTA | Unreliable as blank-page generation | Require exemplar anchoring, tiny state-machine proofs, sanitizer, generated-code inspection, and human review |

This is a risk model, not a permanent judgment about model capability. Recalibrate from fresh evaluation runs. A 2025 presentation by Tri Dao showed strong results for simple kernels and much weaker correctness for advanced memory/tensor-core cases; treat that as historical evidence motivating the gates, not as a timeless benchmark result.

## Task Routing Heuristic

Before choosing CuTe DSL, state why the task needs it.

- Let a framework compiler handle ordinary fusion when it already produces adequate code.
- Consider a higher-level GPU DSL for reductions and memory-bound kernels when its abstraction can express the needed access pattern.
- Choose CuTe DSL when exact layout algebra, hardware atoms, architecture-specific movement, Tensor Core orchestration, or production integration justifies the control.

This heuristic prevents spending agent effort on unnecessary low-level machinery. It is not a ban on any tool.

## Agent Preflight Contract

The agent must create or update a short experiment record containing:

### Environment snapshot

```text
nvidia-cutlass-dsl / CUTLASS revision:
CUDA toolkit and driver:
Python and framework:
GPU model and exact SM target:
compiler/debug environment variables:
launch stream and integration path:
```

### Operation contract

Record the formula, shapes/strides/dtypes, accumulation and conversion, aliasing, masks, zero extents, alignment, shape distribution, and tolerated error.

### Exemplar provenance

```text
official source path + revision:
production exemplar path + revision, if used:
matching properties:
inherited assumptions:
adaptations:
unverified differences:
```

The official source governs API and architecture semantics. Production code supplies implementation and repository patterns only after re-verification.

### Evaluation harness

Create before optimization:

- trusted framework/reference implementation;
- deterministic small cases with coordinate-coded data;
- random and adversarial shapes;
- numerical tolerance policy;
- invalid-contract tests;
- warmup and CUDA-event benchmark harness;
- sanitizer/generated-code commands appropriate to the target.

## The One-Change Experimental Loop

Run this loop for every meaningful change:

1. **Hypothesis** — state the expected correctness or performance effect and the specific mechanism.
2. **One conceptual change** — alter one of layout, tile, copy, stages, roles, atom, epilogue, scheduler, cache policy, or compiler option. Mechanical edits needed for that one concept may travel together.
3. **Compile check** — verify imports/signatures against the installed revision and compile the smallest relevant specialization.
4. **Discriminating correctness** — run the smallest case likely to expose the changed invariant, then the declared matrix.
5. **Safety check** — after synchronization, pipeline, cluster, TMEM, descriptor, or role changes, run sanitizer or a targeted state trace before timing.
6. **Artifact check** — inspect IR/PTX/SASS when the hypothesis depends on instruction selection, vectorization, spills, or barriers.
7. **Benchmark** — measure steady state on representative shapes; include variability and resource information.
8. **Decision** — retain, revert, or refine based on evidence.
9. **Log** — save command, diff, result, and next hypothesis in project-local scratch space.

Combining five optimizations in one edit destroys attribution and makes wrong-code bisection harder. Break the loop only for a mechanically inseparable transformation, and say why.

## Recommended Repository Shape

```text
project/
├── AGENTS.md or CLAUDE.md       # commands, versions, target GPUs, invariants
├── kernels/                     # product kernel code
├── references/                  # pinned source notes and architecture links
├── tests/
│   ├── correctness/
│   ├── invalid_contracts/
│   └── compile_matrix/
├── benchmarks/
├── tools/                       # PTX/SASS, profiler, cache and repro helpers
└── agent_space/                 # disposable notes, traces, experiments
```

The exact names are optional. The separation is not: scratch experiments must not masquerade as reviewed product code.

The repository instruction file should state:

- supported wheel/revision and target SMs;
- setup, test, sanitizer, benchmark, and artifact commands;
- architecture dispatch rules;
- canonical exemplars;
- cache directory and invalidation policy;
- prohibited claims and release gates;
- locations the agent may edit.

## Work Decomposition for Advanced Kernels

Divide a complex kernel into independently reviewable contracts:

1. public wrapper and dispatch;
2. logical index/reference implementation;
3. CTA tile scheduler;
4. GMEM tensor/layout creation;
5. SMEM/TMEM storage plan;
6. copy partitions and tail fill;
7. compute atom and fragments;
8. single-stage state machine;
9. multistage/role overlap;
10. epilogue and stores;
11. cache/FFI integration;
12. tests and benchmark policy.

Ask the agent to prove each interface before composing the next. For attention, separately validate the GEMM fragments, online-softmax recurrence, masks, rescaling, output accumulation, and scheduler shutdown.

## Tool Contract

### Compiler and source

- inspect installed signatures and matching examples;
- retain compile logs and exact target flags;
- treat source/changelog mismatches as version problems, not invitations to guess.

### Correctness

- compare outputs and, where applicable, gradients;
- report maximum absolute/relative error and failing coordinates;
- use coordinate-coded and per-scale-block unique inputs to reveal layout mistakes;
- never convert a failure into a looser tolerance without numerical analysis.

### Sanitizer and tracing

- run memory, race, and synchronization checks appropriate to the environment;
- reduce hangs to one stage, one tile, and explicit state transitions;
- guard `cute.printf` to a minimal agent subset and cap output;
- distinguish tool limitations from proven correctness.

### Generated code

- confirm intended architecture and instruction family;
- check registers, spills, local memory, shared memory, TMEM use, barriers, and vector widths;
- correlate source/IR/PTX/SASS before asserting why performance moved.

### Profiling

- use CUDA events for device time and profile only warmed executors;
- identify whether the limiter is memory, compute, dependency stalls, occupancy, launch/host overhead, or epilogue;
- inspect a small representative kernel set rather than profiling every failed candidate.

## Tuning Boundary

Agents are well suited to bounded searches over objectively measured variables:

- tile shapes;
- stage counts;
- warp/warpgroup arrangements;
- cluster or CTA-group policies;
- vector widths;
- epilogue subtiles;
- persistent scheduler parameters.

Before search, filter candidates by instruction support, alignment/divisibility, thread count, SMEM/TMEM, register pressure estimates, and launch constraints. Compile each unique static configuration once. Keep a compact policy table rather than a specialization per exact runtime shape.

Do not let the agent mutate architecture semantics, barrier participants, or scale layouts as unconstrained tuning knobs.

## Human-Review Boundaries

Require explicit human review when a change affects:

- barrier participant masks/counts or divergent control flow;
- pipeline prologue, phase/index progression, drain, or stage release;
- WGMMA/tcgen05 fence, commit, wait, or visibility;
- TMEM allocation, ownership, epilogue reads, or deallocation;
- 2-CTA/cluster role coordination and shutdown;
- persistent work queues and inactive-role behavior;
- scale-factor logical/physical layouts;
- custom inline PTX/NVVM;
- cache keys or ABI constraints that can silently select stale code;
- numerical algorithms or tolerance changes.

The reviewer should receive the invariant, source anchor, state table, smallest failing/passing test, sanitizer evidence, and generated-code/profiler evidence—not only the final diff.

## Agent Evaluation

Evaluate the skill in fresh contexts with both correctness and performance criteria.

KernelBench's `fast_p` family is a useful shape for aggregate metrics: a task counts only when it is correct and exceeds a chosen speed threshold. Adapt it rather than copying benchmark assumptions blindly.

Track at least:

- compile success rate;
- correctness rate over hidden and adversarial cases;
- sanitizer-clean rate;
- unsupported-API or wrong-architecture rate;
- median and tail speedup on correct candidates;
- compile time and number of variants;
- number of iterations/tool calls to a valid result;
- regression rate on previously passing cases.

A performance score must never hide wrong answers. Report `fast_0`-style correctness separately and retain per-task failures.

## Red Flags

Stop and reset the approach when the agent says or implies:

- “This helper probably exists.”
- “SM120 is Blackwell, so the SM100 tcgen05 path should work.”
- “It compiles, so the barrier protocol is correct.”
- “The random square test passed, so the layout is right.”
- “The profiler is unnecessary; this should reduce latency.”
- “I changed the tile, stages, swizzle, and epilogue together.”
- “The cache can key on the input tensor.”
- “The sanitizer warning is probably a false positive.”
- “This production trick is universal because FA4 uses it.”
- “One fast shape proves the kernel is better.”

Each statement requires source verification, decomposition, or direct evidence before continuing.

## Agent Output Contract

For each completed iteration, produce:

```text
Goal and target:
Pinned environment:
Source/exemplar provenance:
Invariant changed:
Hypothesis:
Files/diff:
Compile command and result:
Correctness matrix and error summary:
Sanitizer/trace result:
IR/PTX/SASS observation:
Benchmark method and result:
Cache/variant effect:
Decision: retain | revert | refine
Remaining risks and reviewer boundary:
```

This record lets another agent or engineer reproduce the result and prevents unmeasured intuition from becoming repository folklore.
