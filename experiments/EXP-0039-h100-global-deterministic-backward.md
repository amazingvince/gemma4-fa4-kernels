# EXP-0039: H100 global deterministic backward dispatch

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-backward
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Design brief: `docs/h100-global-deterministic-backward-design.md`

## Invariant changed

An explicit `deterministic=True` request selects ordered dQ/dK/dV reductions
with batch/tile semaphores. The default fast route and all model semantics are
unchanged.

## Hypothesis

The reviewed three-main-launch route produces bitwise-identical fixed and
packed dQ/dK/dV over five same-input repeats, remains sanitizer-clean and
bounded in code/cache/memory, and costs no more than 25% at global S8K backward
relative to the accepted EXP-0038 nondeterministic default.

## Single change

Expose and key deterministic global backward dispatch. Use two deterministic
V256 dKV launches plus one deterministic EXP-0038 full-D dQ launch, retaining
the explicit shared-epilogue release wait between its output slabs.

## Gates

- [x] design and ownership review
- [x] local API/cache/preflight tests
- [x] fixed and packed reference matrices
- [x] five-repeat bitwise O/LSE/dQ/dK/dV
- [x] dO-only, LSE-only, combined, GQA ownership, and empty-segment isolation
- [x] fixed and packed memcheck/synccheck/racecheck
- [x] generated resources and exactly three main launches
- [x] bounded compile-cache inventory
- [x] S8K performance characterization and S64K smoke
- [x] explicit fast-default rollback regression
- [x] strict environment, full local/H100 suites, and pinned HF oracle

## Correctness and synchronization evidence

Fixed S1/31/32/33/63/64/65/127/128/129 and native packed tiny, mixed,
reversed, mixed-empty, and fixed-parity cases pass the independent reference
policy. Five same-input repeats are bitwise identical for O, FP32 LSE, dQ,
dK, and dV. The matrix includes dO-only, LSE-only, combined gradients,
structured V-slab superposition, isolated GQA-head ownership, hostile packed
segment mutation, exact-zero empty-query ownership, and a nondefault stream.

Fixed S128 and packed mixed workloads report zero memcheck and synccheck
errors. Racecheck reports zero hazards, zero errors, and zero warnings for
both. Actual peak allocation is 33,654,272 bytes against a 37,847,424-byte
fixed bound and 47,323,648 bytes against a 66,889,696-byte packed bound.

## Generated code and launch evidence

Fresh fixed and packed caches each contain exactly two unique main objects:
one deterministic dKV object reused by both V256 launches and one
deterministic full-D dQ object. Nsight Systems records exactly three main
backward launches. All four objects use 168 registers and zero local memory.
Fixed objects have zero stack. Packed dQ retains the accepted scheduler's
24-byte stack with 17 LDL and 8 STL instructions; packed dKV has zero stack.
The dKV objects contain 72 HGMMA instructions and the dQ objects contain 68.

## Measurement

- Clock/power: unlocked; clock locking was not permitted.
- L2: hot.
- S8K: 10 warmups / 30 CUDA-event repetitions.
- S64K: short 2 warmup / 5 repetition smoke only.
- Baseline: unchanged EXP-0038 fast default in the same benchmark API.

| case | fast median/IQR | deterministic median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 42.722 / 0.059 ms | 51.635 / 1.554 ms | +20.9% |
| global S64K bwd hot smoke | 2578.563 / 0.824 ms | 5364.229 / 1.784 ms | +108.0% |

The S8K IQRs are disjoint and the cost remains below the declared 25% gate.
The S64K result is a smoke characterization, not an accepted throughput
claim: deterministic serialization is about 2.08x slower there. Raw timing
rows are retained in
`agent_space/remote-h100-exp0039/exp0039-benchmarks.jsonl`.

## Decision

ACCEPT

The opt-in route clears the declared correctness, bitwise-repeatability,
sanitizer, resource, cache, launch-count, memory, S8K-cost, S64K-smoke, strict
environment, full-suite, and pinned-oracle gates. EXP-0038 remains the fast
default and immediate rollback. Acceptance is limited to exact BF16 H100
global-causal backward and does not claim bounded-memory owner-computes,
deterministic local attention, B300 support, or a speedup.
