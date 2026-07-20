# EXP-0014: H100 native packed global long backward

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512 native packed-varlen long backward
- Architecture: sm_90
- Starting revision: `b7aae21debf42e5dc37be49abcd3fe779cf4c8e2`
- Implementation revision: pending
- Result record: pending
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `c1f5be0ef864fcd716309ae1add48a4c71b8da28578a983083bbba91054a8ee0`
- CuTe DSL / CUDA / PyTorch: pending H100 verification
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `a58d0f6c47e0e8171db6d241df82b05ef6e485fa7dc06f088f406638bee404a8`

## Invariant changed

No model invariant changes. Retain exact BF16, 32 query heads, 4 distinct K/V
heads, GQA-8, d512, lower-right causal masking, scale 1.0, FP32 accumulation,
and separate dQ/dK/dV.

Change only the native packed-THD global-backward admission ceiling from
per-segment K2048 to the locked Gemma context maximum K262144. The fixed BSHD
path remains capped at B1/S2048. Nonempty segments must still satisfy
`1 <= Sq <= Sk`, exact host maxima, signed-INT32 cumulative and padded totals,
and both project and upstream guarded-HBM preflights.

The EXP-0012 composer remains legal only when every segment has K<=2048. A
native K>2048 budget rejection must occur before forward and propagate; it
must never enter that composer. Validation, assertion, compilation, launch,
and runtime failures remain fail-closed at every length.

## Hypothesis

Extending only native THD global-backward admission from K2048 through the
locked K262144 maximum will preserve exact lower-right O, FP32 LSE, and
separate dQ/dK/dV for every resource-admissible request, while reusing
EXP-0013's three scheduler classes and unchanged generated main-kernel
objects.

Falsification is any fixed-path admission above S2048, composer use for
K>2048, forward launch before a known budget rejection, numerical-policy
failure, nonzero cross-segment gradient, incorrect lower-right coordinate,
illegal access, barrier/race report, length-keyed object growth, changed main
kernel bytes/resources, signed-INT32 overflow, or understated memory demand.

## Single change

Give native packed global backward its own model-maximum admission constant
and thread it through the project adapter, pinned Transformers routing, the
managed FA4 varlen assertion, and dedicated probes. Do not change the fixed
cap, tile sizes, stage counts, warp-group roles, accumulation policy,
application cache key, numerical tolerances, or performance settings.

## Predeclared correctness gates

- [ ] CPU/fake validation accepts native K2049 and K262144 and rejects K262145
- [ ] fixed BSHD S2049 remains rejected
- [ ] nonempty `1 <= Sq <= Sk`, exact maxima, distinct K/V, contiguous INT32
      cumulative arrays, and signed-INT32 padded-total guards remain enforced
- [ ] native budget failure occurs before either forward slab
- [ ] composer fallback remains available for budget-only failures at K<=2048
- [ ] every K>2048 budget, validation, assertion, compile, or runtime failure
      propagates without composer or FlexAttention training fallback
- [ ] Q33/K2049 passes independent O/LSE/dQ/dK/dV reference policy
- [ ] packed Q=[33,65], K=[2049,4097] passes reference and exact isolation
- [ ] square S2049 passes independent reference policy
- [ ] dO-only, true LSE-only, and combined gradients pass; LSE-only dV is zero
- [ ] Q129/K4097 passes on three nondefault-stream runs
- [ ] zero, repeated, coordinate-sensitive, and exact finite-score-32 cases pass
- [ ] analytic Q1/K262144 zero-score case with Q=0/K=1 proves O,
      `LSE=log(K)`, dO-only zero dQ/dK, exact-one LSE dQ, zero LSE dK/dV,
      and structured dV without constructing a quadratic dense reference
- [ ] an analytic square S32768 zero-score case activates causal rows 0, 31,
      63, and S-1 and proves the expected four dV regions if guarded HBM
      preflight admits it
- [ ] an intentionally inadmissible long square fails before forward
- [ ] pinned Transformers routes equal S2049, Q33/K4097, and mixed long packed
      training to native THD; actual Gemma global-layer coverage runs if its
      isolated memory footprint is admissible

## Predeclared synchronization and generated-code gates

- [ ] memcheck, synccheck, and racecheck pass for Q=[64,65], K=[2048,2049]
- [ ] memcheck passes for Q33/K4097
- [ ] K2049, K4097, S32768, and K262144 replay add no scheduler/application key
- [ ] native cache remains exactly three classes times dKV/dQ-low/dQ-high
- [ ] fixed application keys and object contents remain unchanged
- [ ] native PTX/SASS hashes and register, stack/local, and shared-memory
      resources remain unchanged from EXP-0013
- [ ] measured peak is no larger than the conservative packed preflight

## Resource envelope

The unchanged EXP-0013 conservative estimator is:

```text
Pq = round_up(Tq, 64) + 64*B
Pk = round_up(Tk, 32) + 32*B
W  = 131072*Tq + 16384*Tk + 66048*Pq + 16384*Pk
A  = W + 65792*Tq
```

Project admission adds a fixed 16 MiB runtime/allocator reserve to `A`. The
reserve was introduced after the first Q33/K2049 measurement exceeded the
pure tensor-size estimate by 348,416 bytes; that underestimation falsified the
original preflight gate before any numerical result was accepted.

Admission requires the relevant estimate to fit
`min(80% of free HBM, free HBM - 2 GiB)`. Approximate predeclared project-side
additional allocations are 8.01 GiB for Q1/K262144, 9.03 GiB for square
S32768, 36.10 GiB for square S131072, and 72.19 GiB for square S262144. These
are resource estimates, not performance measurements.

## Measurement

No performance measurement is authorized. Timing output is not a benchmark or
speed claim.

## Decision

Pending H100 evidence. Do not infer acceptance from compilation or short-case
correctness.
