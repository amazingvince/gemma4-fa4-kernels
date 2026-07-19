# EXP-0013: H100 native packed global d512 backward

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512 native packed-varlen backward
- Architecture: sm_90
- Starting revision: `3d3d340da8e9a3d99506dac5f5e108df3260cde6`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `521a4e5eeff8c4750fa9ee20499c3bc2ed6597396fee58a585201aac02766abe`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: pinned FA4 native THD/cu-seqlens backward plus the accepted
  EXP-0006/0012 split dKV and D256-dQ ownership variants

## Invariant changed

No model invariant changes. Retain exact BF16, 32 query heads, 4 distinct K/V
heads, GQA-8, d512, lower-right causal masking, scale 1.0, FP32 accumulation,
and separate dQ/dK/dV.

Change only the global packed training execution class. Replace the
per-segment fixed-call composition and its zero-Q lower-right prefix with one
native THD launch family carrying explicit INT32 cumulative Q/K lengths. Keep
the EXP-0012 per-segment boundary `1 <= Sq <= Sk <= 2048`; K>2048 training is
not inferred from this experiment. Retain the exact EXP-0012 composer only as
a pre-launch HBM-budget fallback; validation, contract, assertion, and runtime
failures must propagate instead of silently changing routes.

## Hypothesis

The accepted split SM90 scheduler can consume native packed Q/K offsets while
preserving lower-right causal coordinates and cross-segment isolation. For
nonempty packed segments through K2048, native forward plus split backward will
stay inside the frozen O/FP32-LSE/dQ/dK/dV policy, emit one bounded varlen
family limited to the predeclared single/single, single/multi, and multi/multi
block classes independent of runtime totals and cumulative values, and pass
memory and synchronization tools without changing the accepted main-kernel
tile, stage, warp-group, or ownership decisions.

Falsification is any numerical-policy failure, nonzero cross-segment gradient,
incorrect lower-right coordinate, illegal access, barrier/race report,
length-keyed object growth, changed fixed-path object bytes, preflight
underestimate, or failure to keep K/V and dK/dV distinct.

## Single change

Thread `cu_seqlens_q`, `cu_seqlens_k`, and host maximum lengths through the
two-slab global autograd coordinator and the three split backward variants.
Use packed FP32 accumulators and the pinned varlen postprocess layouts. Route
accepted eager packed/lower-right global training to that native path. Do not
change a tile, stage count, warp-group role, fixed-path cache key, numerical
policy, or performance setting.

Aggregate native scratch scales with packed totals, whereas the accepted
composer reuses scratch per segment. A dedicated preflight-budget exception
may select the composer before either native forward slab launches; no other
exception may trigger that fallback.

## Planned correctness evidence

- [ ] CPU/fake validation for THD shapes, cumulative arrays, maxima, aliasing,
      hard K2048 guard, and fail-closed HBM admission
- [ ] native equal-length packed parity with the fixed route
- [ ] lower-right Q33/K1025 independent O/LSE/dQ/dK/dV reference
- [ ] hostile mixed packed Q=[33,65], K=[1025,2048] reference and exact
      cross-segment gradient isolation
- [ ] dO-only, true LSE-only, and combined dO+dLSE gradients
- [ ] repeated-run and nondefault-stream checks
- [ ] framework route assertion proving the native path is selected
- [ ] budget-only composition fallback and runtime-failure propagation
- [ ] full local and H100 suites

## Planned synchronization and generated-code evidence

- [ ] memcheck, synccheck, and racecheck at the first long lower-right tail
- [ ] memcheck, synccheck, and racecheck for hostile mixed packed segments
- [ ] fresh-cache fixed-versus-varlen inventory across max-length classes
      `[31,32]`, `[33,64]`, and `[65,129]`
- [ ] runtime total, segment order, and cumulative-value cache reuse
- [ ] fixed object hashes unchanged from EXP-0012
- [ ] native varlen PTX/SASS plus register, spill, static-SMEM, and
      dynamic-SMEM inventory
- [ ] measured peak no larger than the conservative packed preflight

## Measurement

No performance measurement is authorized. Timing output is not a benchmark or
speed claim.

Predeclare packed padded totals and conservative allocation bounds:

```text
Pq = round_up(Tq, 64) + 64*B
Pk = round_up(Tk, 32) + 32*B
W  = 131072*Tq + 16384*Tk + 66048*Pq + 16384*Pk
A  = W + 65792*Tq
```

`W` mirrors the packed backward allocations; project admission `A` also
reserves retained/anticipated O/LSE and dO/dLSE. Reject padded totals outside
signed INT32 and require both `W` and `A` to fit
`min(80% of free HBM, free HBM - 2 GiB)` at their respective checks.

## Decision

Pending real-H100 evidence. Do not accept native packed global backward,
K>2048 training, deterministic gradients, FakeTensor/`torch.compile`,
compiled/static-cache execution, performance, or B300 from compilation alone.
