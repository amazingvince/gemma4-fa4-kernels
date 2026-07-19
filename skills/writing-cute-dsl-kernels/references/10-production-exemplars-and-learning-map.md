# Production Exemplars and Learning Map

## Evidence Hierarchy

Use sources in this order:

1. **Installed source and matching official NVIDIA example** — API behavior and architecture contract.
2. **Official NVIDIA documentation for the same revision** — conceptual and API guidance.
3. **Production CuTe DSL repository** — integration, decomposition, test, scheduling, and tuning patterns.
4. **Research paper, author tutorial, or benchmark repository** — algorithmic rationale and evaluation ideas.
5. **Community tutorial or blog** — intuition and learning sequence only; verify every API and architecture claim.

A lower tier may explain *why* a pattern works better, but it cannot override the installed implementation.

## Selecting an Exemplar

Score candidates on:

- exact target SM and instruction family;
- operation family and numerical recurrence;
- dtype, accumulator, scale mode, and operand major modes;
- tile/scheduler/cluster shape;
- dynamic shape and tail behavior;
- framework/FFI boundary;
- wheel/repository revision;
- test and benchmark coverage.

Choose the closest semantic and architecture match, not the most famous or fastest-looking file.

Record:

```text
Candidate:
Revision:
Target and toolchain:
Matches:
Differences:
Hidden assumptions found in tests/launch code:
Patterns to borrow:
Patterns explicitly rejected:
```

## Official CUTLASS/CuTe DSL Examples

The official repository is the safest skeleton source. Search the matching revision under the CuTe DSL examples for:

- Ampere/Ada warp-MMA dense GEMM and elementwise kernels;
- Hopper WGMMA GEMM, attention, norm, grouped, and TMA tutorials;
- Blackwell SM100/103/110 tcgen05 dense, mixed-input, block-scaled, grouped, MoE, reduce, norm, distributed, and attention kernels;
- Blackwell GeForce/SM120 warp-level and block-scaled examples;
- framework, DLPack, TVM FFI, fake tensor, inline PTX, dynamic SMEM, CUDA graph, and export examples.

Read the neighboring test, command-line flags, utility module, and launch code. Those files often hold the alignment, architecture target, shape, and synchronization assumptions missing from the core kernel.

## FlashAttention-4 as a Production Exemplar

FlashAttention-4 is valuable for studying a large CuTe DSL system: architecture-separated forward/backward kernels, online softmax, masks, paged KV, persistent scheduling, 2-CTA coordination, cache management, fake-tensor compilation, and low-level debugging.

High-value repository patterns include:

### Agent-ready repository instructions

Its repository guidance records active code locations, dependencies, tests, environment variables, core abstractions, architecture files, and debug documents. Adopt this *documentation pattern* in projects using agents.

### Disposable scratch space

It reserves `agent_space/` for notes, profiler output, minimal repros, and experiments. The important invariant is separation from product code; the exact directory name is optional.

### Two-pass compile/test workflow

The project documents a compile-only pass using framework fake tensors and persistent caching, followed by a real-GPU test pass that reuses artifacts. This is useful for a large parametrized matrix. It is project-specific and does not replace the official rule that CuTe's own fake-tensor ABI is tied to TVM FFI.

### Architecture-specific modules

Separate files and helpers make SM90 WGMMA and SM100-family tcgen05 assumptions visible. When porting, preserve this separation rather than adding opaque conditionals inside one mainloop.

### Explicit compile-cache keys

Production keys account for dtype, head dimension, masks/modifiers, architecture, block sizes, and other compile-time policy. Borrow the audit discipline, not a literal key tuple.

### Targeted artifact and sanitizer tooling

Repository hooks retain PTX/CUBIN/SASS, add line information, and document focused `cute.printf` and sanitizer workflows. Verify current variable names and wrappers in the checked revision.

### Low-level implementation patterns that require qualification

- named barrier enums start at 1 in current files because barrier 0 is reserved for `sync_threads()` there;
- a compact pipeline state uses power-of-two stages to turn division/modulo into bit operations;
- frozen pipeline dataclass instances are reclassified with `object.__setattr__` to extend factory results.

These are useful evidence, not general DSL laws. Copy only with a source revision, explanation, regression test, and API check.

## QuACK as a Smaller Exemplar

QuACK is useful when FlashAttention is too large. It provides focused memory-bound kernels and GEMM components, including norms, softmax/cross-entropy, reductions, epilogue fusion, grouped or specialized GEMMs, and framework bindings.

Use it to study:

- thread → warp → CTA/cluster reduction structure;
- TensorSSA and shuffle-based reductions;
- clean host/JIT/kernel decomposition;
- compile-cache organization;
- memory-bandwidth measurement;
- reusable epilogue/fusion composition.

Recheck APIs against the installed NVIDIA package and remeasure on the target GPU. A production result from another shape distribution is not a benchmark baseline for the new workload.

## KernelBench and Agent Harnesses

KernelBench is not a kernel API source. It is a model for an evaluation harness that couples correctness with measured performance and records metrics such as `fast_p`.

Borrow:

- isolated candidate execution;
- reference comparison before speed scoring;
- reproducible timing;
- aggregate metrics that require both correctness and a speed threshold;
- support for iterative generate/test/refine loops.

Add CuTe-specific checks for architecture, compile variants, sanitizer results, and API invention.

## Tutorials and Papers

Use author tutorials, Colfax material, GPU MODE lectures, and papers to understand:

- layout algebra and ownership;
- Hopper WGMMA pipelines and warp specialization;
- Blackwell TMEM/tcgen05 and cluster control;
- memory-bound reduction design;
- attention algorithm/kernel co-design;
- AI-assisted kernel-generation limits and opportunities.

Tutorial snippets age quickly. Translate their concept into the installed API through the version and exemplar gates.

## Task-Specific Reading Routes

### First CuTe DSL kernel

1. official introduction and control flow;
2. TensorSSA/tensor/layout examples;
3. elementwise official example;
4. layout algebra and predication tutorial;
5. framework integration and JIT caching;
6. one small production kernel.

### Memory-bound reduction or norm

1. official tensor/layout/copy concepts;
2. current reduction/norm official example;
3. QuACK memory-bound kernel and its tests;
4. profiler counters for bandwidth, transactions, and launch overhead;
5. ragged/extreme-value correctness matrix.

### Hopper GEMM or attention

1. WGMMA programming guide;
2. matching SM90 official GEMM/attention example;
3. pipeline API and TMA docs;
4. production SM90 module in FlashAttention or QuACK;
5. generated-code and sanitizer workflow.

### SM100/SM103 tcgen05 kernel

1. tcgen05 programming and API docs for the exact target;
2. matching dense/block-scaled/mixed-input official example;
3. TMEM allocation and epilogue utilities;
4. one- versus two-CTA example as required;
5. production SM100/103 module;
6. target-device retuning.

### SM120/SM121 block-scaled kernel

1. warp-level MMA guide and supported-op table;
2. `blackwell_geforce` official example;
3. scale-factor layout utilities;
4. verify that no SM100 tcgen05/TMEM assumption leaked in;
5. inspect target PTX/SASS and tune on the actual device.

### Agentic kernel project

1. `references/09-agent-assisted-development.md`;
2. official examples and source map;
3. production repository instruction file and scratch/test conventions;
4. KernelBench-style correctness/performance harness;
5. fresh-context skill evaluation prompts.

## Scheduling and Low-Level Extension Routes

For dynamic persistent Blackwell scheduling, read in this order:

1. official Blackwell CLC programming model;
2. installed `PipelineClcFetchAsync` and `ClcDynamicPersistentTileScheduler` signatures;
3. closest official CLC-enabled kernel in the same revision;
4. production scheduler code only for additional decomposition or tuning ideas.

For a missing instruction or math primitive, first search current `cute.arch`, NVIDIA GPU dialect, and official feature examples. Use a production inline-PTX/NVVM helper only as a pinned, target-guarded specimen. Re-derive operand constraints, proxy/fence requirements, side effects, and fallback from installed source and generated code.

## Borrowing Checklist

Before promoting any borrowed pattern:

- [ ] source URL/path and immutable revision recorded;
- [ ] license/project constraints checked;
- [ ] installed API and exact target support verified;
- [ ] layout, agent set, barrier count, and memory-space assumptions restated;
- [ ] cache key and ABI impact recorded;
- [ ] smallest source test reproduced;
- [ ] adaptation has a failing-then-passing regression case;
- [ ] sanitizer/generated-code checks run when applicable;
- [ ] representative benchmark rerun locally;
- [ ] project-specific tricks labeled in comments/docs.

## Search Discipline

Useful repository searches include:

```text
path:examples/python/CuTeDSL <sm target> <operation>
path:examples/python/CuTeDSL blackwell_geforce <dtype or scale format>
path:flash_attn/cute <pipeline or architecture symbol>
repo:Dao-AILab/quack <operation or reduction primitive>
repo:ScalingIntelligence/KernelBench cute correctness speedup
```

Search results are discovery aids. Open the exact file and revision before citing or copying it.
