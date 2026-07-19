# Merge Notes

Date: 2026-07-18

## Inputs

- Base package: `writing-cute-dsl-kernels` version 1.0.0.
- User-supplied skill: `cutedsl-kernel-writing.skill`.
- User-supplied skill SHA-256: `21b6c8065a978e27f4984cb7824c431176a188c707d2a14d6699ec1910c08073`.
- External refresh: current NVIDIA CuTe DSL documentation, NVIDIA CUTLASS repository/release material, FlashAttention source/instructions, QuACK repository context, KernelBench, and Tri Dao's 2025 tutorial slides.

## Integrated Directly

- exemplar-first kernel development;
- explicit production exemplars and task-specific reading routes;
- SM103 target-specific verification within the tcgen05 family;
- separate SM120/SM121 warp-level path rather than SM100 tcgen05/TMEM reuse;
- Blackwell Cluster Launch Control scheduling, participant, fence, and shutdown obligations;
- a guarded CuTe → architecture wrapper → NVVM/LLVM → inline-PTX escape-hatch hierarchy;
- custom compile-cache discipline, experimental `cute.compile_to` handling, and production two-pass compile/test workflow;
- agent scratch-space and one-concept-per-iteration practices;
- correctness/tool-feedback gates for AI-assisted kernel generation;
- FlashAttention/QuACK learning anchors;
- KernelBench-style correctness-plus-performance evaluation;
- additional evaluation scenarios covering architecture mismatch, agent pressure, ABI/cache misuse, portability, and source freshness.

## Integrated with Qualification

| Source idea | Merged treatment |
|---|---|
| Barrier 0 reserved for `sync_threads()` | Recorded as a current FlashAttention implementation convention; target semantics must be verified before reuse. |
| Power-of-two pipeline stage counts | Kept as a potential index/phase micro-optimization, not a correctness rule or universal optimum. |
| Frozen dataclass `__class__` reassignment | Kept as a version-sensitive production compatibility pattern; supported public APIs are preferred. |
| FA4 two-pass fake-tensor flow | Kept as a project pattern, while the official CuTe fake-tensor flow is stated as TVM-FFI-only. |
| SM120 is “Ampere-like” | Replaced with the precise current model: target-specific warp-level `mma.sync`, including block-scaled ops, with RMEM accumulators and no SM100 tcgen05/TMEM path. |
| Class-per-kernel structure | Presented as an optional production organization pattern, not a DSL requirement. |
| Production cache keys | Kept as an audit model; literal key fields must follow the target kernel's codegen contract. |

## Omitted or Reframed

- Unqualified “no performance compromise” and compile-time multiplier claims were not made part of the agent contract; generated code and local measurements govern.
- Beta-graduation timing was omitted from behavioral guidance because release status changes.
- Production repository environment variables were not promoted to standard CuTe DSL APIs.
- Tutorial and blog snippets were tiered below installed source and official examples.
- Broad architecture-family inheritance was replaced by exact supported-op and feature-target checks.

## New Files

- `references/09-agent-assisted-development.md`
- `references/10-production-exemplars-and-learning-map.md`
- `MERGE_NOTES.md`

## Validation Boundary

Static content checks can verify package structure, required routing rules, links, source tiers, and authoring hygiene. They cannot establish that generated kernels compile, are race-free, or are fast. Fresh agent evaluations and target-GPU runtime tests remain required.
