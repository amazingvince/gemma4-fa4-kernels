# H100 M1 status

**Status date:** 2026-07-19

**Ordered gate result:** advanced through exact sparse-scheduled packed local
d256 vision/document metadata at the locked model maximum, within its declared
resource envelope.
The H100 environment, fixed and packed local d256 paths, exact local
multimodal masking, and composed global d512 text forward/backward passed
their declared gates. EXP-0003's
fixed elementwise dQ/dK envelope remains rejected; EXP-0004 preserved that
result and accepted the unchanged local backward under a separately
predeclared upstream-relative BF16 oracle. EXP-0005 remains the historical
rejection of the unchanged direct asymmetric GQA-8 global backward. EXP-0006
accepts a structural dQ/dKV split under the exact B1, S<=1024, BF16,
32Q/4KV, GQA-8, d512, causal, scale-1.0, distinct-K/V contract. EXP-0007
accepts the exact fixed B1 local vision predicate; EXP-0008 accepts nonempty
packed local self-attention with `B>=1` and `1 <= Sq <= Sk <= 1025`, including
lower-right native text and custom vision/document masking. EXP-0009 extends
native packed text to `1 <= Sq <= Sk <= 262144`. EXP-0010 extends exact
vision/document metadata through the same maximum when the schedule fits the
`2^40` padded-score, 2 GiB metadata, and 10%-free-HBM ceilings. Framework
dispatch and context-offset integration are the active ordered H100
compatibility gate; all benchmarks remain unrun.

M0 remains the semantic contract: scale is exactly `1.0`; K/V are distinct
prepared operands; backward returns separate dQ, dK, and dV; and the local
multimodal predicate is
`k > q - 1024 AND (k <= q OR same nonnegative vision block)`.

## Environment and provenance: PASS

Observed H100 identity:

```text
NVIDIA H100 80GB HBM3
compute capability 9.0, 81559 MiB
driver 580.126.09
CUDA toolkit 12.8 (nvcc V12.8.93)
PyTorch 2.8.0+cu128, runtime 12.8
CuTe DSL 4.6.0.dev0
quack-kernels 0.5.3
```

The strict H100 policy uses FA4 `[dev]`, never the CUDA-13 `cu13` extra.
ShellCheck 0.9.0 and CUDA-12.8-matched Nsight Systems 2024.6.2 are installed.
The profiler-required environment check passes with every required tool found.

FlashAttention is base revision
`77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus exactly one H100 patch:

```text
patches/flash-attention/0002-sm90-gemma4-d512-forward-backward.patch
SHA256 df345b01e4fab6d077898f642ac3ba40effffc6f2291803bc93ae1b0e38ec294
```

`scripts/check_env.py` requires the pinned base revision, exact patch diff,
exact patch hash, no additional tracked or untracked checkout changes, and
that the imported FA4/Transformers modules resolve inside those checkouts.
Transformers remains clean at
`7ea2320c76117e6742364808a666ef6f2fb40a67`.

The retained strict reports `agent_space/h100-check-precommit.json` and
`agent_space/h100-check-exp0006.json` both have SHA256
`b41270f33f67dda21af8c8b45d7f76c5287daa2c5a10598f3887f9a56cddc9d4`.
EXP-0001 through EXP-0003 are machine-recorded against source revision
`5b9bfab072e8cc28a7e92c9e956608db591b246c`.
EXP-0004 is machine-recorded against its validated source revision
`49fbcad2e2b761d9de50312f03335e27236a8a13`.
EXP-0005 is machine-recorded against its rejected source revision
`d7ac7273aaed5c57923301afa6f052333e91c5b7`.
EXP-0006's accepted implementation source is
`185f11cbda15ae7bd4841968c3dd46f95b670282`.
EXP-0007's accepted implementation source is
`d1b7e4ad0b1ffff6e3190a4b4411603cd544afe4`.
EXP-0008's accepted implementation source is
`de6450a9cf5040a7432ed7641b230bb29f835248`.
EXP-0009's accepted implementation source is
`9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf`.
EXP-0010's accepted implementation source is
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0`.

The patch opens the exact `(Dqk,Dv)=(512,256)` SM90 forward specialization and
the reviewed split-backward ownership variants, but is not itself a mask-mode
guard. The project adapter admits only the locked B1/S<=1024/32Q/4KV
global-causal text contract.

## Forward gates: PASS

### Local d256 fixed-length text forward

`fa4_local_text_forward` fixes contiguous BSHD BF16, 32Q/16KV for the model,
d256, scale 1.0, inclusive FA window `(1023, 0)`, causal text semantics,
distinct K/V, 16-byte base alignment, B=1/S<=1025, one split, no pack-GQA,
BF16 O, and FP32 LSE.

H100 evidence:

- fake compile: `1 passed, 36 deselected`;
- real focused run: `18 passed, 19 deselected`;
- sequence lengths `1,63,64,65,127,128,129,1023,1024,1025`;
- GQA ratios 1, 2, 4, and 8 at S=33;
- unchanged O tolerance `atol=0.03125, rtol=0.02`;
- unchanged LSE tolerance `atol=0.125, rtol=0`;
- exact repeat on a nondefault stream.

This is not a varlen or vision-mask result. See
`experiments/EXP-0001-h100-local-d256-forward.md`.

### Global d512 fixed-length text forward

The accepted M1 path is an exact correctness composition, not a fused d512
kernel. It splits V into two contiguous d256 slabs, runs the same patched SM90
`(Dqk,Dv)=(512,256)` M128 x N32 specialization twice over identical Q/K,
requires identical FP32 LSE, and concatenates the two d256 outputs:

```text
concat(P @ V0, P @ V1) = P @ concat(V0, V1)
```

The adapter rejects anything outside B=1/S<=1024 and 16-byte-aligned,
contiguous BF16 BSHD input storage.

H100 evidence:

- fake compile: `1 passed, 36 deselected`;
- real focused run: `17 passed, 20 deselected`;
- exact 32Q/4KV, GQA 8, causal, scale 1.0, distinct K/V;
- S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`;
- frozen O envelope `atol=0.0625, rtol=0.03`, observed max abs `0.015625`;
- frozen LSE envelope `atol=0.25, rtol=0`, observed max abs
  `0.00015258789`;
- exact LSE agreement between slab launches and exact nondefault-stream repeat;
- filtered memcheck and synccheck: 0 errors each;
- filtered racecheck: 0 hazards, 0 errors, 0 warnings.

Both forward acceptances are scoped to the seeded-random matrices above.
Coordinate-coded, zero/repeated/large-logit, and adversarial-BF16 cases remain
explicit follow-up hardening before integration.

The raw sanitizer instrumentation reports 34 CuTe/cuda-python
`cuGetProcAddress_v2` API-probe errors. Runs with `--report-api-errors no`
isolate and pass the memory/synchronization tools; both facts are recorded in
`experiments/EXP-0002-h100-global-d512-forward.md`.

Generated-code evidence for the asymmetric specialization:

- 168 registers, zero stack, zero local memory, 1 KiB static shared memory;
- 96 `HGMMA.64x32x16.F32.BF16` and 6
  `HGMMA.64x256x16.F32.BF16` instructions;
- 32 `UTMALDG`, 4 `UTMASTG`, and 6 `WARPGROUP.DEPBAR` instructions;
- 224 KiB modeled dynamic core storage;
- exact dynamic shared-memory launch metric remains unresolved because Nsight
  Compute returns `ERR_NVGPUCTRPERM` on this pod.

The two launches duplicate QK/softmax work and materialize V slabs. No speed or
efficiency claim is made.

## Local backward gates: EXP-0003 REJECT preserved; EXP-0004 PASS

The unchanged pinned M64 x N64 Q1/dO1/PdS1 local backward uses exact
32Q/16KV GQA-2, BF16, scale 1.0, causal W1024, distinct K/V, and separate
dQ/dK/dV. Upstream still excludes SM90 backward above d192 from its broad
test, so this is a project-scoped hardware validation rather than an
upstream/general support claim.

EXP-0003's exact fake compile produced separate shapes:

```text
dQ=(1,128,32,256)
dK=(1,128,16,256)
dV=(1,128,16,256)
```

Its first real S=128 comparison failed the frozen
`atol=0.125, rtol=0.05` envelope:

- first dQ failure: 552 / 1,048,576 mismatches, max abs `0.2890625`, max
  relative `1112.0`;
- diagnostic rerun without changing tolerance: dQ max/mean abs `0.28857422` /
  `0.0097916815`, dK `0.375` / `0.013713409`, and dV `0.0625` /
  `0.00076462259`;
- dQ and dK failed; dV passed. This rejection and the probe's default
  `frozen` policy remain intact.

EXP-0004 predeclared the numerical rule used by the pinned upstream CuTe
tests, with an independent PyTorch BF16 attention path as the baseline:

```text
max_abs(g - g_ref) <= 2 * max_abs(g_pt - g_ref) + quantization_atol
quantization_atol = 2 * max_abs((g_ref + 0.3 - 0.3) - g_ref)
```

No kernel, tile, stage, mask, accumulation type, or public adapter changed.
The exact B1/S/32Q/16KV/d256 matrix passed at
S=`1,63,64,65,127,128,129,1023,1024,1025`. Every case returned finite BF16
dQ/dK/dV with exact input shapes and three distinct output allocations. Across
the matrix, candidate maximum absolute error ranges were:

- dQ: `0.00002277` to `0.5`;
- dK: `0.00002480` to `0.5`;
- dV: `0` to `0.0625`.

Every value was below its independently computed upstream-relative limit; the
complete candidate/baseline max/mean table is retained in EXP-0004. Three
same-input S128 repetitions were bitwise equal for dQ, dK, and dV, and the
nondefault-stream run passed the same numerical gate.

Memcheck and synccheck reported zero errors, and racecheck reported zero
hazards/errors/warnings, at both S128 and the S129 partial-tile boundary. Main
backward generated-code evidence:

- PTX 8.8 targeting `sm_90a`;
- 32 `HGMMA.64x32x16.F32.BF16` plus 12
  `HGMMA.64x128x16.F32.BF16` instructions;
- 24 `UTMALDG.4D` and 5 `WARPGROUP.DEPBAR` instructions;
- 168 registers, zero stack, zero local memory, 1 KiB static shared memory;
- 208 KiB modeled core dynamic storage.

Two controls explain why EXP-0003 and EXP-0004 can legitimately have
different decisions. The pinned backward intentionally rounds P to BF16 for
dV and dS to BF16 before dQ/dK; a reference mirroring those stage boundaries
matched the candidate at mean errors `0.0000103`, `0.0000188`, and
`0.00000647`. The upstream-supported d128 control also failed EXP-0003's fixed
dQ/dK envelope, with maxima `0.25` and `0.25`. This diagnoses a
reference-rounding-policy mismatch; it does not retroactively loosen
EXP-0003.

Acceptance is limited to fixed-length B1, exact model GQA-2, d256, scale 1.0,
causal W1024 text attention, and the tested S<=1025 matrix. No backward GQA
1/4/8, varlen, vision-mask, global, long-context, performance, or B300 claim
is made. See EXP-0003 and EXP-0004 for the immutable reject/accept records.

## Global backward gates: EXP-0005 REJECT preserved; EXP-0006 PASS

EXP-0005 remains an immutable rejection of the unchanged path. The exact
two-slab forward presents each direct FA4 backward as d512 Q/K, d256 V/output,
and GQA-8. The pinned constructor rejects it before main compilation:

```text
AssertionError: GQA backward requires head_dim == head_dim_v
```

Head expansion bypassed that assertion in a diagnostic, but the unchanged
monolithic launch requested 345,088 bytes against SM90a's 232,448-byte limit.
EXP-0006 does not revise either result; it changes work ownership.

For each V256 slab, the accepted composition runs one M64 x N32 dKV-only main
launch and two M64 x N32 dQ-only launches. Each dQ launch owns one D256 output
slice and uses the matching K slice after recomputing full-d512 scores. The
dKV launch reduces all eight query-head contributions into four-KV-head FP32
accumulators. The dQ launches reduce K-major tiles into separate FP32 D256
accumulators. Common FP32 accumulators preserve the two slab contributions
before the sole BF16 dQ/dK conversion, while the two BF16 dV slabs are
concatenated:

```text
dQ = concat(dQ00 + dQ10, dQ01 + dQ11)
dK = dK0 + dK1
dV = concat(dV0, dV1)
```

The exact B1/BF16/32Q/4KV/GQA-8/d512/causal/scale-1.0/distinct-K/V
matrix passed at
S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`. Every case returned
finite BF16 dQ/dK/dV with exact shapes and distinct storage and passed the
unchanged EXP-0004 upstream-relative numerical rule against independent FP32
and BF16 PyTorch references. Structured half-zero dO-slab superposition and
isolated-query-head GQA ownership checks also passed.

Three same-input S128 repeats produced bitwise-identical O and FP32 LSE.
Gradients were not bitwise identical because FP32 bulk/atomic reduction order
can vary, but every repeat independently passed the frozen numerical policy.
The nondefault-stream run passed the same contract; this is not a
deterministic-gradient claim.

Memcheck and synccheck reported zero errors, and racecheck reported zero
hazards/errors/warnings, at both S128 and the S129 partial-tile boundary.
Generated-code resource evidence for all three main variants is:

- dKV-only: 222,208 bytes dynamic shared memory;
- each dQ-only D256 variant: 218,112 bytes dynamic shared memory;
- all variants: 168 registers, 1 KiB static shared memory, zero stack, and
  zero local memory.

Acceptance is limited to the exact fixed-length text envelope above. The full
d512 backward uses six main launches, temporary FP32 accumulators, and atomic
GQA reduction. No speed, efficiency, deterministic-gradient, long-context,
multimodal, or B300 claim is made. See EXP-0005 and EXP-0006 for the immutable
reject/accept records.

## Local multimodal gate: EXP-0007 PASS

The exact fixed-length local predicate now has a dedicated SM90 custom-mask
path:

```text
k > q - 1024
AND (k <= q OR same nonnegative vision block)
```

It passes B1/BF16/32Q/16KV/GQA-2/d256/scale-1.0 forward O/FP32-LSE and
separate dQ/dK/dV at
S=`1,31,32,33,63,64,65,127,128,129,1023,1024,1025`. Coverage includes
all-text runtime IDs, ID zero, mixed and adjacent vision spans, adversarial
masked K/V sentinels, exact S1025 window edges, LSE-only and combined
gradients, and isolated q63/head9 transposed GQA ownership. Three repeats and
a nondefault stream passed.

Memcheck, synccheck, and racecheck are clean at S128 and S129; S1025 memcheck
is also clean. Generated code realizes M128 x N80 forward and M64 x N64
backward. Both main kernels use 168 registers and 1 KiB static shared memory;
backward has zero stack/local memory, while custom forward reports a 40-byte
stack frame and zero separate local allocation. EXP-0007 makes no performance
claim and does not promote a sparse schedule.

Acceptance is deliberately fixed-length and B1. The public `(1,S)` INT32 or
range-checked INT64 vision IDs become a private contiguous INT32 `(S,)`
auxiliary. `vision_block_ids=None` preserves the native text path. Packed
varlen uses the separate accepted EXP-0008 adapter below; the generic
Transformers 2D FA4 adapter remains xfailed.

## Packed local gate: EXP-0008 PASS

`fa4_local_varlen_forward` accepts packed THD Q `(Tq,32,256)` and distinct
K/V `(Tk,16,256)`, CUDA INT32 cumulative arrays, exact Python max lengths,
and optional packed K-stream vision/document IDs. Text-only calls retain
FA4's native lower-right causal window `(1023,0)`. Metadata calls disable the
native mask flags and use one complete predicate with
`q_abs = q + Sk - Sq`:

```text
same document
AND k > q_abs - 1024
AND (k <= q_abs OR same nonnegative vision block)
```

Forward O and FP32 LSE passed equal-length batches from S1 through S1025,
reordered segments, and native/custom lower-right matrices including
Q=`[1,31,64,129]`, K=`[33,64,128,1025]`. An independent q1/k1025 sentinel
proved the strict excluded-key-0/included-key-1 boundary. Backward passed
native and custom O-only, true `dout=None` LSE-only, and combined gradients
under EXP-0004's unchanged upstream-relative BF16 policy. LSE-only dV was
exactly zero; a transposed q1/k1025 sentinel proved exact dV ownership.

Structured packed GQA ownership, internal document blocking, repeated-ID
isolation, and hostile cross-sequence K/V mutation all passed. Three
nondefault-stream repeats were bitwise equal for O, LSE, dQ, dK, and dV.
Changing Tq/Tk totals, cumulative values, segment order, tensor contents, and
metadata contents did not create new native or custom forward/backward cache
objects.

Memcheck, synccheck, and racecheck are clean for custom packed single-block
`[63,64]/[64,64]` and multi-block `[64,65]/[64,65]` cases; q1/k1025 memcheck
is also clean. Generated code retains M128 x N80 forward and M64 x N64
backward. Both main kernels use 168 registers and 1 KiB static shared memory;
backward has zero stack/local memory, while packed custom forward reports a
104-byte stack frame and zero separately reported local allocation. See
EXP-0008 for all bounded cache keys and PTX/cubin/SASS hashes.

EXP-0008 acceptance is scoped to nonempty sequences with
`1 <= Sq <= Sk <= 1025` on SM90. It does not itself accept empty sequences,
production context above 1025, block sparsity, generic framework dispatch,
performance, or B300. EXP-0009 separately widens native text; EXP-0010 widens
metadata within its sparse resource envelope.

## Production-length native packed text gate: EXP-0009 PASS

The adapter now admits native packed text with `B>=1` and per-sequence
`1 <= Sq <= Sk <= 262144`, retaining exact BF16 32Q/16KV GQA-2 d256,
scale 1.0, distinct K/V, CUDA INT32 cumulative arrays, and lower-right causal
W1024 semantics. Metadata-bearing vision/document calls retain the S1025 guard.
No native kernel, tile, pipeline, barrier, mask callable, or generated-code
decision changed.

The bounded-work basis is the pinned scheduler itself. Forward calls
`BlockInfo.get_n_block_min_max` for each M tile to derive the causal/local
K-block interval from runtime sequence-local coordinates. Backward calls
`BlockInfo.get_m_block_min_max` for each N tile to derive the transposed bounded
Q-block interval. The adapter fixes `num_splits=1`, so long runtime maxima do
not engage the host split heuristic or create a new split specialization. This
claim does not rely on `seqlen_k_loaded`.

H100 evidence at implementation revision
`9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf` includes:

- S2048 numerical forward/`out_lse` backward against independent FP32 and BF16
  references, repeated three times on a nondefault stream. O, LSE, dK, and dV
  were bitwise equal; dQ had maximum pairwise drift `0.03125`, and every
  dQ/dK/dV repeat passed the unchanged upstream-relative BF16 policy;
- a finite exact-shape S32768 nondefault-stream `out_lse` smoke;
- a true Q=K=262144 `out_lse` forward/backward run after a corrected
  `45,231,374,336`-byte preflight with `84,465,025,024` bytes free;
- an analytic Q1/K262144 sentinel proving the strict excluded key at
  `q_abs-1024`, the first included key, exact `log(1024)` LSE, and dV ownership;
- hostile long-ragged isolation at Q=`[33,65]`, K=`[2049,4097]` against both
  references, without an observed-after-the-fact fixed dQ tolerance;
- zero memcheck/synccheck errors and zero racecheck hazards/errors/warnings at
  Q=`[64,65]`, K=`[2048,2049]`, plus zero memcheck errors for Q1/K262144.

Changing native long runtime totals and maxima added no cache objects. The
forward-only cache retained one object; a backward invocation retained four
objects total including forward, preprocess, main backward, and postprocess.
The native forward key remains
`e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176` and
the multi/multi main-backward key remains
`a3c7d28fb5372354d1d121353d7803b12e6ba713817d650b5a7288237c006ec7`.

Retained PTX is version 8.8 targeting `sm_90a`; forward and main backward keep
the accepted M128 x N80 and M64 x N64 code. Both main kernels use 168 registers,
1 KiB static shared memory, zero stack, and zero separately reported local
memory. Forward retains its HGMMA/TMA/dependency-barrier instruction path;
backward does likewise, with no LDL/STL spill traffic observed. Exact hashes
and instruction counts are in EXP-0009.

This EXP-0009 acceptance does not itself include metadata-bearing calls above
1025, empty segments, deterministic dQ, generic dispatch, performance, B300,
or another architecture. EXP-0010 separately accepts long metadata below.

## Production-length packed metadata gate: EXP-0010 PASS

Metadata-bearing local calls above S1025 now use an exact per-sequence fixed
block-sparse composition. The adapter builds Q128/K80 forward incidence and an
independent Q64/K64 backward incidence before transposing the latter into
K-block rows. A tile is present if and only if at least one in-range pair
satisfies the full document/W1024/causal-or-same-vision predicate; every
candidate remains partial and re-evaluates that predicate token by token.

The pinned SM90 sparse loader traces its empty-list branch even when runtime
mask counts are nonzero. Forward and backward therefore carry explicit
row-aligned zero-count/width-one full-list sentinels. They are compile plumbing,
not allowed tiles. Packed Q/K/V and both metadata tensors are split once with
`torch.split`; each sequence owns one fixed call, preserving separate
dQ/dK/dV without per-slice full-base scatter buffers.

Acceptance is bounded explicitly. Before host enumeration, rectangular sparse
storage is checked against 2 GiB and 10% of current free HBM. Construction
stops above `2^40` padded score slots, and final compact CUDA storage is checked
again. The maximum square rectangular bound is 94,027,776 bytes. A
semantically valid schedule above a ceiling is rejected, never approximated;
such schedules are outside EXP-0010 rather than silently claimed compatible.

H100 evidence at implementation revision
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0` includes:

- 14 exact schedule tests, including exhaustive small metadata, independent
  forward/backward tile incidence, strict W1024/document cases, K262144
  sentinels, the maximum storage bound, and incremental work exhaustion;
- public adapter tests for the 2 GiB, 10%-free-HBM, and translated work-limit
  rejection paths;
- Q=`[33,65]`, K=`[2049,4097]` forward O/FP32-LSE and O-only, true LSE-only,
  and combined separate dQ/dK/dV references under the unchanged EXP-0004
  BF16 policy;
- three nondefault-stream repeats: O/LSE/dK/dV bitwise equal, dQ maximum
  pairwise drift `0.015625`, every repeat numerically valid;
- hostile repeated-ID packed/document isolation, Q1/K262144 strict-window
  ownership, and Q2049/K262144 far-future vision/different-document/far-past
  exclusion with `log(1025)` LSE and exact dV ownership;
- zero memcheck/synccheck errors and zero racecheck hazards/errors/warnings at
  Q=`[64,65]`, K=`[2048,2049]`;
- unchanged five-object bounded cache reuse across runtime lengths, metadata,
  segment order, contents, and compact widths;
- a real aggregate suite of `196 passed, 8 skipped, 1 xfailed`, plus a passing
  explicit sparse forward/backward fake compile.

Retained PTX 8.8 targets `sm_90a`. Forward contains 100 HGMMA, 56 TMA-load,
4 TMA-store, and 18 warpgroup arrive/dependency-barrier instructions. Main
backward contains 88 HGMMA, 32 TMA-load, and 20 warpgroup
arrive/dependency-barrier instructions. Both use 168 registers and 1 KiB
static shared memory. Main backward has zero stack/local and no LDL/STL.
Forward reports `LOCAL=0` but a 144-byte stack with 51 LDL/38 STL, so it is not
described as stack-traffic-free. Exact hashes and cache keys are in EXP-0010.

EXP-0010 is scoped to nonempty SM90 BF16 32Q/16KV GQA-2 d256 local attention,
scale 1.0, distinct prepared K/V, per-sequence `1 <= Sq <= Sk <= 262144`, and
requests inside the declared resource envelope. Empty segments, over-budget
schedules, deterministic dQ, generic framework dispatch/context offsets,
performance, B300, and other architectures remain excluded.

## Gate table

| Gate | Status | Evidence / stop condition |
|---|---|---|
| H100 identity | **PASS** | H100 80GB, CC 9.0, driver/toolkit above |
| Pinned CUDA-12.8 FA4 environment | **PASS** | Strict check including exact patch stack and profilers |
| CPU/model contract on H100 | **PASS** | Oracle status OK; full H100 suite below |
| Local d256 text forward | **PASS** | O/LSE, boundaries, GQA 1/2/4/8, stream repeat |
| Global d512 text forward | **PASS (composed)** | O/LSE through S1024, sanitizer and SASS evidence |
| Local d256 backward | **PASS (scoped)** | EXP-0003 reject preserved; EXP-0004 matrix/oracle, stream/repeat, sanitizers, SASS |
| Global d512 backward | **PASS (composed)** | EXP-0005 direct-path reject preserved; EXP-0006 split six-main-launch matrix, stream/repeat, sanitizers, resources |
| Multimodal local fwd/bwd | **PASS (fixed B1)** | EXP-0007 O/LSE/gradients, ownership, stream/repeat, sanitizers, SASS |
| Packed varlen local fwd/bwd | **PASS (scoped)** | EXP-0008 native/custom through S1025; EXP-0009 native text and EXP-0010 metadata through S262144 |
| Long vision/document metadata >1025 | **PASS (resource-scoped)** | EXP-0010 exact sparse fwd/bwd, references, isolation, K262144 sentinels, sanitizers, cache, SASS |
| Framework/context-offset integration | **NOT RUN / NEXT** | Must prove per-layer local/global dispatch and preserve prepared K/V semantics |
| Benchmarks | **NOT RUN** | Correctness sequence incomplete; no performance claim |

## Exact verification commands and latest results

```bash
bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 \
  python scripts/check_env.py --profile h100 --expect-arch sm_90 \
    --strict --require-profilers --require-transformers
bash scripts/remote/run.sh h100 \
  python scripts/verify_model_contract.py --transformers
bash scripts/remote/run.sh h100 pytest -q
```

Final local full-suite result after the EXP-0010 implementation changes:

```text
128 passed, 75 skipped
```

The skips are optional Transformers/H100 gates. The aggregate H100 bundle
result at implementation revision
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0` is:

```text
196 passed, 8 skipped, 1 xfailed
```

The eight skips are fake-compile-only tests in real execution. The expected
failure is the pinned Transformers generic FA4 mask adapter, which cannot
encode Gemma's vision future-token exception. Local, global, and multimodal
hardware acceptances come from the explicit EXP-0004, EXP-0006, EXP-0007,
EXP-0008, EXP-0009, and EXP-0010 probe matrices and sanitizer runs above;
aggregate pytest is not presented as a substitute for that evidence.

Representative reproduction commands follow. Run the EXP-0005 command from
its recorded source revision `d7ac7273aaed5c57923301afa6f052333e91c5b7`;
the current adapter intentionally selects EXP-0006 instead.

```bash
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0003-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0004-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
    --comparison-policy upstream-relative

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0005-global-bwd-true-fake \
  python scripts/probe_h100_global_backward.py --seqlen 128

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0006-global-bwd-real \
  python scripts/probe_h100_global_backward.py --reference \
    --seqlen 128 --repeats 3 --nondefault-stream

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0006-global-bwd-real \
  python scripts/probe_h100_global_backward.py \
    --seqlen 33 --compile-only --structured

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0007-real \
  python scripts/probe_h100_local_backward.py --seqlen 129 \
    --vision-pattern adjacent --gradient-source out_lse --reference \
    --comparison-policy upstream-relative --structured-ownership

bash scripts/remote/run.sh h100 compute-sanitizer --tool memcheck \
  --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_backward.py --seqlen 129 \
    --vision-pattern adjacent

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0008-real \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 1,31,64,129 --k-lengths 33,64,128,1025 \
    --vision-pattern all --document-pattern split \
    --gradient-source out_lse --reference \
    --comparison-policy upstream-relative

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0008-cache-custom \
  python scripts/probe_h100_local_varlen_cache.py --custom --backward

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0008-real \
  compute-sanitizer --tool racecheck --report-api-errors no \
    --error-exitcode 99 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 64,65 --k-lengths 64,65 \
    --vision-pattern adjacent --document-pattern split \
    --gradient-source out_lse

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0009-real \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 2048 --k-lengths 2048 --gradient-source out_lse \
    --reference --comparison-policy upstream-relative --repeats 3 \
    --nondefault-stream --allow-nondeterministic-dq

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 33,65 --k-lengths 2049,4097 \
    --gradient-source out_lse --long-text-isolation

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 262144 --k-lengths 262144 --gradient-source out_lse

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0009-cache \
  python scripts/probe_h100_local_varlen_cache.py --long-text --backward

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 33,65 --k-lengths 2049,4097 \
    --vision-pattern adjacent --document-pattern split \
    --gradient-source out_lse --reference \
    --comparison-policy upstream-relative --repeats 3 \
    --nondefault-stream --allow-nondeterministic-dq \
    --long-metadata-isolation

bash scripts/remote/run.sh h100 compute-sanitizer --tool racecheck \
  --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 64,65 --k-lengths 2048,2049 \
    --vision-pattern adjacent --document-pattern split \
    --gradient-source out_lse

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0010-cache \
  python scripts/probe_h100_local_varlen_cache.py \
    --custom --long-text --backward

bash scripts/remote/run.sh h100 env FLASH_ATTENTION_FAKE_TENSOR=1 \
  pytest -q \
    tests/test_h100_fa4.py::test_h100_local_sparse_empty_full_sentinel_fake_compile
```

The next H100 experiment is per-layer framework dispatch and context-offset
integration. It must select the accepted local/global paths without changing
scale, prepared K/V, masks, lower-right coordinates, or separate gradient
ownership. Do not skip ahead to performance tuning or B300, or rewrite a prior
experiment's decision.

## Deferred scope

B300/SM103, over-budget sparse schedules, fused single-launch global d512,
deterministic local/global gradients, generic Transformers multimodal
dispatch/context offsets, backward GQA ratios beyond the exact validated model
ratios (local 2 and global 8), empty packed segments, and all performance work
remain deferred. Framework integration is the active next correctness gate.
No H100 result is generalized to B300.
