# EXP-0014: H100 native packed global long backward

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512 native packed-varlen long backward
- Architecture: sm_90
- Starting revision: `b7aae21debf42e5dc37be49abcd3fe779cf4c8e2`
- Implementation revision: `364ea6ab27513a42d1b3e9f7baf9213720c1a530`
- Result record: schema-valid `EXP-0014` entry in `experiments/results.jsonl`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `c1f5be0ef864fcd716309ae1add48a4c71b8da28578a983083bbba91054a8ee0`
- Accepted H100 patch SHA256:
  `97dd1dd7c9c8efb5f2b2fd06f601a1bcc8abbe9ebcf43e9ea86365769c2e2749`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `91827fafa60d1710abb22034581f4db6c95d654c7af9b1e7ba5a07a51b4d6e35`
- Strict environment artifact: `agent_space/h100-check-exp0014.json`, SHA256
  `cfdcfe48c1f65325fbaa144e2a44294682095a45390ec6246143cd37a740e49b`

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

## Correctness evidence

- [x] CPU/fake validation accepts native K2049 and K262144 and rejects K262145
- [x] fixed BSHD S2049 remains rejected
- [x] nonempty `1 <= Sq <= Sk`, exact maxima, distinct K/V, contiguous INT32
      cumulative arrays, and signed-INT32 padded-total guards remain enforced
- [x] native budget failure occurs before either forward slab
- [x] composer fallback remains available for budget-only failures at K<=2048
- [x] every K>2048 budget, validation, assertion, compile, or runtime failure
      propagates without composer or FlexAttention training fallback
- [x] Q33/K2049 passes independent O/LSE/dQ/dK/dV reference policy
- [x] packed Q=[33,65], K=[2049,4097] passes reference and exact isolation
- [x] square S2049 passes independent reference policy
- [x] dO-only, true LSE-only, and combined gradients pass; LSE-only dV is zero
- [x] Q129/K4097 passes on three nondefault-stream runs
- [x] zero, repeated, coordinate-sensitive, and exact finite-score-32 cases pass
- [x] analytic Q1/K262144 zero-score case with Q=0/K=1 proves O,
      `LSE=log(K)`, dO-only zero dQ/dK, the exact BF16-staged LSE-dQ
      profile, zero LSE dK/dV,
      and structured dV without constructing a quadratic dense reference
- [x] an analytic square S32768 zero-score case activates causal rows 0, 31,
      63, and S-1; guarded HBM admitted it and all four expected dV regions pass
- [x] an intentionally inadmissible long square fails before forward
- [x] pinned Transformers routes equal S2049, Q33/K4097, and mixed long packed
      training to native THD; the actual pinned Gemma global layer runs S2049
      backward with a finite hidden-state gradient

## Synchronization and generated code

- [x] memcheck, synccheck, and racecheck pass for Q=[64,65], K=[2048,2049]
- [x] memcheck passes for Q33/K4097
- [x] K2049, K4097, S32768, and K262144 replay add no scheduler/application key
- [x] native cache remains exactly three classes times dKV/dQ-low/dQ-high
- [x] fixed application keys and object contents remain unchanged
- [x] native PTX/SASS hashes and register, stack/local, and shared-memory
      resources remain unchanged from EXP-0013
- [x] measured peak is no larger than the conservative packed preflight

The independent dense-reference envelope was deliberately bounded at K4097
and at no more score elements than square S2049. Larger cases used analytic
or compile-only evidence, so no long test constructed a quadratic FP32/GQA
reference. Representative numerical results were:

| case | O max abs | LSE max abs | dQ max abs | dK max abs | dV max abs |
|---|---:|---:|---:|---:|---:|
| Q33/K2049 combined | 0.015625 | 0.00011444 | 0.5 | 0.5 | 0.03125 |
| Q33/K2049 dO only | 0.015625 | 0.00011444 | 0.5 | 0.375 | 0.03125 |
| Q33/K2049 LSE only | 0.015625 | 0.00011444 | 0.0625 | 0.0625 | 0.0 |
| packed Q=[33,65], K=[2049,4097] | 0.015625 | 0.00012970 | 0.5 | 0.5 | 0.03125 |
| square S2049 | 0.015625 | 0.00014496 | 1.0 | 1.0 | 0.125 |

The packed case retained exact-zero inactive dQ/dK/dV before and after a
hostile mutation of the other segment. Three Q129/K4097 nondefault-stream
runs independently passed; O/LSE/dK/dV were bitwise stable while dQ varied by
at most 0.03125, so this experiment makes no deterministic-gradient claim.

The nonquadratic oracle used Q=0, K=1, and constant V=0.25. At Q1/K262144,
O, LSE, dQ, dK, and dV all matched their analytic targets exactly for combined
and LSE-only differentiation. At square S32768, rows 0, 31, 63, and S-1
proved the four causal dV regions 32/24/16/8; O, dK, and dV were exact, LSE
max error was 9.5367432e-7, and all dQ elements exactly matched the backward
pipeline's independently reconstructed BF16 dS-staging profile. An additional
Q1/K4096 score-32 case had exact O/LSE/dQ/dK/dV errors of zero, preventing the
zero-score oracle from hiding score reconstruction mistakes.

The pinned Transformers integration routed Q33/K2049 and mixed
Q=[33,65]/K=[2049,4097] backward through `fa4_global_varlen_native` and kept
packed gradient isolation. The actual pinned `Gemma4TextAttention` global
layer at index 5 executed S2049 backward through the same native path with a
finite hidden-state gradient. No full-model training claim is inferred.

Memcheck, synccheck, and racecheck each reported zero issues for mixed
Q=[64,65], K=[2048,2049]; Q33/K4097 memcheck also reported zero errors. The
fresh cache source namespace changed, as expected from the managed-interface
source edit, to
`3e6db223212ba9119a3d164193682b94fe1f5afdcfb4d77583f472e9a69651ac`.
The final inventory remained 28 objects, 16 unique contents, and 1,956,400
bytes. Long K2049/K4097, square S32768, and model-max K262144 replay added no
object or application key. The nine native SS/SM/MM application-key suffixes
and the dKV/dQ-low/dQ-high object SHA256 values remained exactly those
recorded in EXP-0013:

| variant | unchanged object SHA256 |
|---|---|
| dKV | `1df46be3f4071fa3fcf0a7d5a40f7ddbc4c589131ab83155b7f7089a51a23771` |
| dQ low | `84ff8c1dddd65e9e1d5bb42c82b54aef0a68cb38543fc3b7f4764cbac208f775` |
| dQ high | `b32f6711d185366c4effbda8c01aec99877f70da0ea11d73103c48aa1ac9f6ec` |

Byte-identical main objects retain EXP-0013's PTX/SASS and resource evidence:
PTX 8.8 targeting `sm_90a`, 168 registers and 1 KiB static shared memory for
all variants, zero local memory, zero dKV stack, a 16-byte dQ stack, and the
same HGMMA/TMA/barrier instruction inventories. This is code-generation
evidence, not a performance claim.

## Resource envelope

The EXP-0013 workspace and retained-output terms remain unchanged. EXP-0014
strengthens only the final project admission estimate with a fixed reserve:

```text
Pq = round_up(Tq, 64) + 64*B
Pk = round_up(Tk, 32) + 32*B
W  = 131072*Tq + 16384*Tk + 66048*Pq + 16384*Pk
A_base    = W + 65792*Tq
A_EXP0014 = A_base + 16 MiB
```

The reserve was introduced after the first Q33/K2049 measurement exceeded the
pure tensor-size estimate by 348,416 bytes; that underestimation falsified the
original preflight gate before any numerical result was accepted.

Admission requires the relevant estimate to fit
`min(80% of free HBM, free HBM - 2 GiB)`. Approximate project-side additional
allocations are 8.02 GiB for Q1/K262144, 9.04 GiB for square
S32768, 36.11 GiB for square S131072, and 72.21 GiB for square S262144. The
exact first two estimates are 8,615,887,104 and 9,710,370,816 bytes. These
are resource estimates, not performance measurements.

Measured peak increments remained bounded:

| case | measured bytes | estimated bytes |
|---|---:|---:|
| Q33/K2049 | 83,472,896 | 99,901,696 |
| packed Q98/K6146, B2 | 238,047,744 | 255,910,400 |
| square S2049 | 547,865,088 | 632,045,824 |
| Q1/K262144 | 8,599,110,144 | 8,615,887,104 |
| square S32768 | 8,615,657,472 | 9,710,370,816 |

The full square S262144 was not allocated or launched. The reproducible
meta-tensor preflight computed 77,532,266,496 required bytes against
84,465,025,024 free and a 67,572,020,019-byte guarded budget, then raised the
dedicated budget exception before forward exactly as required.

## Measurement

No performance measurement is authorized. Timing output is not a benchmark or
speed claim.

## Decision

**ACCEPT.** Extend the exact H100 native packed THD/cu-seqlens global backward
to every nonempty segment satisfying `1 <= Sq <= Sk <= 262144`, subject to
exact maxima, signed-INT32 cumulative/padded totals, distinct K/V, and the
guarded-HBM preflight. Fixed BSHD and the exact composer remain capped at
S/K2048. For K>2048, budget rejection propagates before forward and may not
select the composer or FlexAttention.

This does not accept empty packed segments, deterministic gradients,
FakeTensor/`torch.compile`, compiled/static-cache model execution,
performance, B300, dropout, or any geometry outside exact Gemma 4 global
attention.

## Record

The schema-valid EXP-0014 record names implementation
`364ea6ab27513a42d1b3e9f7baf9213720c1a530`, environment policy
`91827fafa60d1710abb22034581f4db6c95d654c7af9b1e7ba5a07a51b4d6e35`,
and strict artifact `agent_space/h100-check-exp0014.json`.

Final local verification on the implementation tree was 220 passed, 81
skipped, and 9 warnings; compileall, Ruff, format, and diff checks passed. The
aggregate H100 suite was 291 passed, 14 skipped, and 1 expected xfail with 105
deprecation/runtime warnings in the first aggregate run; the final canonical
bundle run was 291 passed, 14 skipped, 1 expected xfail, and 9 warnings. The
strict H100 environment and exact managed
patch checks passed. Focused numerical, analytic, framework, sanitizer,
cache, and code-generation evidence above is the acceptance basis; no timing
was treated as a benchmark.
