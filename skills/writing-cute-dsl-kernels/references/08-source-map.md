# Source Map and Evidence Tiers

## Research Baseline

This skill was synthesized on **2026-07-18** from NVIDIA’s official CUTLASS repository and documentation. The repository’s current release observed during research was CUTLASS **4.6.1**. The docs and package are evolving; use the installed package/revision as the source of truth.

Stability labels:

- **Concept** — durable CuTe model likely to remain useful.
- **API** — verify exact spelling/signature in the installed revision.
- **Example** — strongest starting point when revision and target match.
- **Experimental** — expect changes and gate behind tests/version checks.

## Primary Documentation

| Topic | Stability | Official source |
|---|---|---|
| CuTe DSL introduction, decorators, call conventions | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html |
| End-to-end code generation, meta-time/runtime, preprocessing/tracing | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html |
| Control flow, `range`, `cutlass.range`, `range_constexpr` | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html |
| Static versus dynamic layouts | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html |
| JIT caching | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html |
| Framework and DLPack integration | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html |
| TVM FFI, fake tensors, compiled constraints, host-overhead guidance | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html |
| Debugging and artifact retention | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html |
| Autotuning guidance | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.html |
| Naming conventions | Concept | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/naming_conventions.html |
| Pipeline API, including `PipelineClcFetchAsync` | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html |
| Utility/scheduler API, including `ClcDynamicPersistentTileScheduler` | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html |
| Blackwell Cluster Launch Control programming model | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html |
| Warp-level MMA guide, including SM120 block-scaled path | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html |
| Warp MMA API and exact supported-architecture tables | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html |
| Hopper WGMMA guide | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wgmma_programming.html |
| Blackwell tcgen05 guide | Concept/API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html |
| tcgen05 API and exact supported-architecture tables | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html |
| Current limitations | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html |
| Current CUTLASS changelog, including experimental `cute.compile_to` status | API | https://docs.nvidia.com/cutlass/latest/CHANGELOG.html |
| FAQ and beta/version guidance | API | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html |
| CuTe layout fundamentals | Concept | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html |
| CuTe layout algebra | Concept | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html |
| CuTe GEMM tiling/partitioning tutorial | Concept | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html |
| General predication with coordinate tensors | Concept | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html |

The C++ CuTe conceptual tutorials are useful because the Python DSL intentionally mirrors the CuTe programming model, but exact Python APIs must still be checked in Python DSL docs/source.

## Repository

- Repository: https://github.com/NVIDIA/cutlass
- CuTe DSL example roots: search both the reorganized `examples/cute` tree and the older/compatibility `examples/python/CuTeDSL` tree in the pinned revision.
- Package/source tree: search the revision for `cutlass.cute`, `cutlass.pipeline`, `cutlass.utils`, and `cute_nvgpu`.
- Changelog: use both repository release notes and the DSL/API changelog in the matching docs.

## Example-First Navigation

Start from the closest target/family directory in the same revision:

```text
examples/
├── cute/                         # current/reorganized examples in recent releases
│   ├── ampere/
│   ├── hopper/
│   ├── blackwell/                # SM100/103/110 tcgen05 family
│   └── blackwell_geforce/        # SM120/121 warp-MMA family
└── python/CuTeDSL/               # older/compatibility organization in some revisions
    ├── cute/<architecture>/kernel/
    ├── cute/<architecture>/tutorial/
    ├── examples/
    └── notebooks/
```

Directory names can move. Search the repository revision rather than hard-coding imports from this map.

## Canonical Files Mentioned in Official Guides

These paths are high-value search anchors, not stable API contracts:

- the current dense/block-scaled/mixed-input examples under `examples/cute/blackwell/` or their `examples/python/CuTeDSL` equivalents;
- the SM120 block-scaled example under `examples/cute/blackwell_geforce/kernel/blockscaled_gemm/` referenced by the warp-MMA guide;
- the current Hopper attention and dense-GEMM examples;
- architecture-specific dense GEMM examples under the Ampere, Hopper, Blackwell, and Blackwell-GeForce directories;
- feature examples for DLPack/framework integration, JAX, TVM FFI, fake tensors, inline PTX, dynamic SMEM, and dependent launch where present

Always inspect neighboring tests and command-line parameters; they reveal shape, alignment, architecture, and launch assumptions omitted from the core function.

## Production Exemplars

Production repositories are secondary to installed/official sources for API truth, but they provide high-value integration evidence.

| Repository/file | Use it for | Guardrail |
|---|---|---|
| https://github.com/Dao-AILab/flash-attention | FlashAttention-4 architecture separation, attention pipelines, persistent scheduling, compile caches, test matrix, debug tooling | Pin a commit; verify active paths and supported targets in that revision |
| https://github.com/Dao-AILab/flash-attention/blob/main/CLAUDE.md | Agent scratch space, commands, two-pass compile/test flow, repository map | Project instructions can change; do not treat environment variables as CuTe APIs |
| https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/named_barrier.py | Example of named-barrier allocation | Barrier 0 rule is cited as an implementation convention; verify before reuse |
| https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/pipeline.py | Pipeline wrappers, compact state, version-sensitive extension patterns | Copy semantics only after matching pipeline API and tests |
| https://github.com/Dao-AILab/quack | Smaller memory-bound and GEMM exemplars | Recheck NVIDIA APIs and benchmark on the target workload |

## Agentic Kernel Development and Evaluation

These are primary sources for their own systems or author positions, not substitutes for NVIDIA API documentation.

| Source | Use it for |
|---|---|
| https://github.com/ScalingIntelligence/KernelBench | Correctness-plus-performance evaluation, `fast_p`, iterative candidate tooling |
| https://hc2025.hotchips.org/assets/program/tutorials/dsl_llm_kernels.pdf | Tri Dao's 2025 capability framing and tool/abstraction recommendations; treat as a dated snapshot |
| https://github.com/NVIDIA/cutlass | Current official examples, changelog, source, and release context |

## Curated Secondary Learning Material

Use tutorials for explanations and reading order, then verify against official/current sources:

- Colfax Research CuTe/CUTLASS tutorials for Hopper, Blackwell, TMA, CLC, block scaling, and layout algebra;
- GPU MODE lectures for practitioner explanations;
- Modal, author blogs, and paper walkthroughs for onboarding and source tours;
- FlashAttention papers/blogs for algorithm/kernel co-design;
- benchmark and contest starter kits for harness structure.

Do not copy an API from a tutorial without the Version Gate.

## Educational Sequence

For a developer new to CuTe DSL, follow this order in the current official notebooks/examples:

1. hello world and compilation;
2. printing/meta-time versus runtime;
3. native data types;
4. tensors and TensorSSA;
5. layout algebra;
6. elementwise add;
7. CUDA graphs/framework integration;
8. simple tiled copy;
9. architecture GEMM tutorial;
10. production kernel closest to the intended operation.

## How an Agent Should Cite a Choice

In code review/design notes, record decisions like:

```text
Target: SM90
Pattern source: official WGMMA programming guide, matching installed CUTLASS revision
Closest example: <repository path + commit>
Adaptations: dynamic M/N, custom epilogue, stage count 3
Reverified APIs: <symbols/signatures>
```

This is more useful than a generic “based on CUTLASS.”

## Search Queries for the Repository

Use scoped searches:

```text
path:examples/cute <architecture> <operation>
path:examples/python/CuTeDSL <architecture> <operation>
path:python/cutlass/cute <symbol>
path:python/cutlass/pipeline <pipeline class>
path:examples/python/CuTeDSL make_tiled_mma
path:examples/python/CuTeDSL tma <dtype>
path:examples/python/CuTeDSL tcgen05 <scale mode>
```

When GitHub paths and installed wheel source differ, prefer the installed source for API behavior and the matching repository tag for examples.

## Source-Use Rules

- Prefer installed source, NVIDIA docs, and the official repository over production repositories or tutorials for API truth.
- Use production repositories for implementation patterns and third-party material for intuition or performance observations, then verify independently.
- Treat “latest” docs as potentially newer than the installed wheel.
- Treat old blog snippets as untrusted until compiled and tested.
- Never infer support for a dtype/atom/SM from a neighboring architecture.
- Re-run source discovery after every package upgrade.


## Research Refresh Checklist

Before a material update to this skill:

- check the current PyPI package and CUTLASS changelog/release tag;
- compare `latest` docs with the installed/tagged revision;
- verify supported-architecture tables for every MMA atom mentioned;
- reopen canonical examples because paths and helpers move;
- inspect production exemplar instructions at a pinned commit;
- date any model-capability or benchmark conclusions;
- rerun static checks and fresh-context evaluation scenarios.
