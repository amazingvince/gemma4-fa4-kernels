# EXP-0013: H100 native packed global d512 backward

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512 native packed-varlen backward
- Architecture: sm_90
- Starting revision: `3d3d340da8e9a3d99506dac5f5e108df3260cde6`
- Implementation revision: `87ff75b1b40b55149ec5beea7480ed9ac14c9146`
- Result record: schema-valid `EXP-0013` entry in `experiments/results.jsonl`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `521a4e5eeff8c4750fa9ee20499c3bc2ed6597396fee58a585201aac02766abe`
- Accepted H100 patch SHA256:
  `c1f5be0ef864fcd716309ae1add48a4c71b8da28578a983083bbba91054a8ee0`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `a58d0f6c47e0e8171db6d241df82b05ef6e485fa7dc06f088f406638bee404a8`
- Strict environment artifact: `agent_space/h100-check-exp0013.json`, SHA256
  `0ef779d2f175e821bd6520852d555b8759ce78e22ec402f270324398fbae9476`
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

## Correctness evidence

- [x] CPU/fake validation for THD shapes, cumulative arrays, maxima, aliasing,
      hard K2048 guard, signed-INT32 padding, and fail-closed HBM admission
- [x] native equal-length B1/S65 parity with the fixed route
- [x] lower-right Q33/K1025 independent O/LSE/dQ/dK/dV reference
- [x] hostile mixed packed Q=[33,65], K=[1025,2048] reference and exact
      cross-segment gradient isolation
- [x] dO-only, true LSE-only, and combined dO+dLSE gradients
- [x] repeated-run and nondefault-stream checks
- [x] framework native routing and document splitting, plus direct-native odd
      noncontiguous dO/dLSE checks
- [x] budget-only composition fallback and runtime-failure propagation
- [x] full local and H100 suites

The unchanged EXP-0004 numerical policy passed independently for every native
run and for the fixed side of the parity check. Representative maximum
absolute errors were:

| case | O | LSE | dQ | dK | dV |
|---|---:|---:|---:|---:|---:|
| equal S65 native | 0.015625 | 0.000091552734 | 0.5 | 0.625 | 0.125 |
| mixed Q=[33,65], K=[1025,2048] | 0.015625 | 0.00012207031 | 0.5 | 1.0 | 0.03125 |
| square S2048 | 0.015625 | 0.00014495850 | 0.5625 | 1.0 | 0.125 |
| mixed short, LSE only | within forward policy | within forward policy | 0.0625 | 0.0625 | 0.0 |

For B1/S65, fixed and native O/LSE were bitwise identical. Direct native/fixed
gradient deltas were 0.0078125 dQ, 0.015625 dK, and 0.0625 dV, inside the
predeclared pairwise frozen-policy bounds; both sides separately passed the
FP32 and independent BF16 references.

The mixed long probe proved exact-zero dQ/dK/dV outside the selected packed
segment before and after hostile mutation of every inactive segment. Framework
document splitting rebuilt cumulative arrays `[0,31,64,129]`, preserved exact
forward isolation, and separately isolated dQ/dK/dV. Odd-padded upstream views
with dO strides `[16416,513,1]` and dLSE strides `[97,1]` remained
noncontiguous and passed. LSE-only dV was exactly zero.

Three reverse-order and nondefault-stream executions independently passed the
frozen oracle. O/LSE were bitwise stable; parallel FP32 reduction order may
change gradients, so no deterministic-gradient claim is made.

Representative commands:

```bash
python scripts/probe_h100_global_varlen_backward.py \
  --q-lengths 65 --k-lengths 65 --reference --fixed-parity
python scripts/probe_h100_global_varlen_backward.py \
  --q-lengths 33,65 --k-lengths 1025,2048 --reference --isolation \
  --record-memory
python scripts/probe_h100_global_varlen_backward.py \
  --q-lengths 2048 --k-lengths 2048 --reference --record-memory
python scripts/probe_h100_transformers_integration.py --case all
```

## Synchronization and generated code

- [x] memcheck, synccheck, and racecheck for the 33 one-token-segment case
- [x] memcheck, synccheck, and racecheck for hostile mixed packed segments
- [x] framework document-split memcheck and direct-native odd noncontiguous
      gradient memcheck
- [x] fresh-cache fixed-versus-varlen inventory across max-length classes
      `[31,32]`, `[33,64]`, and `[65,129]`
- [x] runtime total, segment order, stride, and cumulative-value cache reuse
- [x] fixed object hashes unchanged from EXP-0012
- [x] native varlen PTX/SASS plus register, stack/local, static-SMEM, and
      generated shared-storage inventory
- [x] measured peak no larger than the conservative packed preflight

All six core sanitizer runs passed. Memcheck and synccheck each reported zero
errors; racecheck reported zero hazards, errors, and warnings for both the
mixed long and 33-segment cases. The document-split framework memcheck and the
direct-native odd-stride memcheck also reported zero errors.

The fresh cache source fingerprint was
`99f53104f784e5d3d322457794f4b5e5901280e5d4aff6a741c5a0baa593014b`.
Fixed BSHD retained exactly nine application keys and the EXP-0012 main-object
contents. Native THD added exactly nine distinct application keys: three
scheduler classes times dKV, dQ-low, and dQ-high. Runtime totals, logical
batch, cumulative values/order, legal strides, and K1025/K2048 replay added no
specialization. The final inventory was 28 objects, 16 unique contents, and
1,956,400 bytes.

The nine native application-key suffixes are:

| class | dKV | dQ low | dQ high |
|---|---|---|---|
| SS | `70ae09261e19fc87a2366bedb36abff131929287436f5fa6ff59d46687e42110` | `06b26d38303c4a96ed8fc9dffa5588600c77471df6548208413c9c3754b841b9` | `54b5f47e6d2989022e1f7c54c8b115315ae295e55b33bd0a4925fa960b2e1ce3` |
| SM | `e057f6ff8cc7e2d721221e35c1065d8389010b901e4ed32935267d9925897424` | `263d7f21fbe882ebec1c5acb0857f78cb8ce49aeb54d45339a5603b9de81ee9b` | `c0d711cdb09e96b29512314a1f96b812bdeb5e89032f3ad81c8ffeef3c975052` |
| MM | `95e7a67b917a643c84b62aaedb8235155144a339c63704bcabb84406984dc20e` | `8fa531972dccd72eb123766393c6e09b32f26e700cb824dd978efb27677a7e31` | `891f13b14cb1a54fc2930549c90e98b86042ce922370f573391b5f166fb11872` |

Native main-object content was class-invariant by variant:

| variant | object SHA256 |
|---|---|
| dKV | `1df46be3f4071fa3fcf0a7d5a40f7ddbc4c589131ab83155b7f7089a51a23771` |
| dQ low | `84ff8c1dddd65e9e1d5bb42c82b54aef0a68cb38543fc3b7f4764cbac208f775` |
| dQ high | `b32f6711d185366c4effbda8c01aec99877f70da0ea11d73103c48aa1ac9f6ec` |

The retained representative dQ-high PTX is version 8.8 targeting `sm_90a`,
SHA256 `d04ff0c81d99429a8a764c47633a461092de6b040c18355e622f42de8339c3fe`;
its dumped cubin SHA256 is
`a5e8c1635b24bc6ce6f7e6155767cf49f1eec1f8d7a8ca558191abbd616013d1`.
The dump filename is reused by successive compilations, so this retained pair
is representative dQ-high evidence. dKV and dQ-low cubin/SASS evidence was
extracted from their embedded cache objects.

The inspected dKV/dQ-low/dQ-high SASS hashes were respectively
`499f046cb79c131723e603fda15950b3bf64cc817141f5a436135f96cb67c3d6`,
`31f072941632f63836a3595a9ea5cbb175fd0df93b04c81d68159b66e4f252f6`,
and `839087271773bc4a55f47f7668513dbc789778eff36bfc0f377b04923185251a`.
dKV contains 72 HGMMA, 36 `UTMALDG`, eight `UBLKRED`, and four
`WARPGROUP.DEPBAR` instructions. Each dQ variant contains 50 HGMMA, 36
`UTMALDG`, two `UBLKRED`, 50 `WARPGROUP.DEPBAR`, seven `LDL`, and four `STL`.
All use 168 registers and 1 KiB static shared memory; dKV reports zero
stack/local, while dQ reports a 16-byte stack and zero local memory. The
generated shared-storage configuration remains 222,208 bytes for dKV and
218,112 bytes for dQ. These are code/resource observations, not occupancy or
performance claims.

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

For Q=[33,65], K=[1025,2048], measured peak allocation was 135,782,912
bytes, below the 138,453,504-byte estimate. At square S2048, measured peak was
542,932,992 bytes, below the 610,304,000-byte estimate.

## Decision

**ACCEPT.** The exact H100 native packed THD/cu-seqlens global backward is
accepted for nonempty segments satisfying `1 <= Sq <= Sk <= 2048` under the
locked BF16, 32Q/4KV, GQA-8, d512, lower-right-causal, scale-1.0, distinct-K/V,
separate-gradient, and resource-preflight contract. Only
`GlobalBackwardBudgetExceeded` may select the retained exact composer; other
failures propagate.

This does not accept K>2048 training, empty packed segments, deterministic
gradients, FakeTensor/`torch.compile`, compiled/static-cache execution,
performance, B300, dropout, or any model geometry other than exact Gemma 4
global attention.

## Record

The schema-valid EXP-0013 record names implementation
`87ff75b1b40b55149ec5beea7480ed9ac14c9146`, environment policy
`a58d0f6c47e0e8171db6d241df82b05ef6e485fa7dc06f088f406638bee404a8`,
and strict artifact `agent_space/h100-check-exp0013.json`.

Final local verification on the implementation tree was `208 passed, 78
skipped, 9 warnings`; compileall, Ruff, format, and diff checks passed. The
strict H100 environment, exact managed-patch check, and pinned Transformers
model oracle passed. Aggregate H100 pytest is recorded in `docs/status.md`;
the focused numerical, sanitizer, cache, and codegen evidence above is the
acceptance basis rather than an aggregate suite alone.
