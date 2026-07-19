# CuTe DSL Kernel Design Brief

Complete this before implementation. Replace every `<...>` field.

## 1. Environment and Version

- Date:
- `nvidia-cutlass-dsl` version:
- CUTLASS tag/commit:
- CUDA toolkit / driver:
- Python:
- Framework:
- GPU model / compute capability:
- Official example and docs matched to this revision:
- APIs/signatures explicitly verified:

## 2. Operation Contract

- Operation name:
- Mathematical index formula:
- Input tensors, logical modes, dtypes:
- Output tensors, logical modes, dtypes:
- Accumulation dtype:
- Epilogue/conversion:
- Masking semantics:
- NaN/Inf policy:
- Zero-size / empty / all-masked behavior:
- Aliasing/in-place policy:
- Determinism requirement:

## 3. Shape and Layout Regime

- Supported ranks:
- Dynamic dimensions:
- Static dimensions/configuration:
- Supported stride/major modes:
- Alignment classes:
- Representative shapes:
- Adversarial/tail shapes:
- Rejected/fallback cases:
- Expected specialization count and cache key:

## 4. Target and Kernel Family

- Target architecture:
- Kernel family:
- Portable fallback:
- Selected official starting example:
- Selected compute atom and why:
- Unsupported atom/dtype combinations guarded:

## 5. Tile and Ownership Hierarchy

- Problem modes:
- CTA tile:
- Cluster/CTA group:
- Warp or warpgroup tile:
- Instruction tile:
- Thread/value layout:
- Grid/block/cluster:
- Persistent scheduler/work mapping, if any:

Attach a tensor table:

| Tensor | Logical modes | Shape/stride | Static/dynamic | Memspace | Owner/partition | Consumer |
|---|---|---|---|---|---|---|
| | | | | | | |

## 6. Data Movement

For each edge, record primitive, vector width, alignment, predicate, and synchronization.

| Edge | Primitive | Layout/partition | Alignment | Tail behavior | Completion signal |
|---|---|---|---|---|---|
| GMEM → SMEM | | | | | |
| SMEM → RMEM/descriptor | | | | | |
| MMA → accumulator | | | | | |
| accumulator → epilogue | | | | | |
| epilogue → GMEM | | | | | |

- SMEM layout/swizzle source:
- Storage span/alignment proof:
- TMA descriptor assumptions:
- TMEM allocation/lifetime, if any:

## 7. Pipeline State Machine

- Number of stages:
- Producer agents:
- Consumer agents:
- Named/memory barriers:
- Barrier participant counts:
- Stage state `(index, phase)` convention:
- Prologue:
- Steady-state transition:
- Drain/tail:
- Store completion:
- Persistent-loop reset:

| Transition | Agent | Stage/index/phase | Event | Matching transition |
|---|---|---|---|---|
| acquire | | | | |
| issue | | | | |
| commit/arrive | | | | |
| wait | | | | |
| consume | | | | |
| release | | | | |

## 8. Predication and Neutral Values

- Coordinate tensor(s):
- M/N/K or other bound predicates:
- Invalid load fill:
- Invalid store behavior:
- Predicate reuse/broadcast:
- Vector-tail strategy:
- All-masked/empty reduction behavior:

## 9. Resource Budget

- Threads/warps per CTA:
- CTAs per cluster:
- Dynamic/static SMEM:
- Barrier/descriptor storage:
- Expected registers:
- TMEM:
- Expected occupancy/residency:
- Resource constraints used to filter tuning candidates:

## 10. Correctness Plan

- Trusted reference:
- Tolerance/error policy:
- Coordinate-coded test:
- Boundary shapes:
- Value edge cases:
- Stream/concurrency cases:
- Sanitizer commands:
- Expected failure/rejection tests:

## 11. Benchmark Plan

- Baseline:
- Workload distribution:
- Warmup:
- Timing mechanism:
- Synchronization:
- Repetitions/statistic:
- Clock/power controls:
- Metrics:
- Search space:
- Cache key for tuning result:
- Acceptance threshold:

## 12. Assumptions and Risks

- Verified assumptions:
- Unverified assumptions:
- Version-sensitive helpers:
- Primary correctness risk:
- Primary performance risk:
- Rollback/simpler implementation:
