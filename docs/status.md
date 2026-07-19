# H100 M1 status

**Status date:** 2026-07-19

**Ordered gate result:** advanced through fixed-length local d256 backward.
The H100 environment, local d256 text forward/autograd backward, and composed
global d512 text forward passed their declared gates. EXP-0003's fixed
elementwise dQ/dK envelope remains rejected; EXP-0004 preserved that result
and accepted the unchanged local backward under a separately predeclared
upstream-relative BF16 oracle. EXP-0005 then rejected the unchanged global
d512 slab backward at its constructor because pinned SM90 GQA backward requires
equal QK/V dimensions. Head expansion alone would bypass the assertion but
still exceeds the pinned register/SMEM budgets; the next global-backward
experiment requires a structural dQ/dKV resource split. Multimodal masking
and all benchmarks remain unrun.

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
patches/flash-attention/0001-sm90-d512-v256-forward.patch
SHA256 8d3404ccc8bdb2b3fd8e6de09e1f100827d071de283885bf737932bcb41aca4f
```

`scripts/check_env.py` requires the pinned base revision, exact patch diff,
exact patch hash, no additional tracked or untracked checkout changes, and
that the imported FA4/Transformers modules resolve inside those checkouts.
Transformers remains clean at
`7ea2320c76117e6742364808a666ef6f2fb40a67`.

The retained strict report is `agent_space/h100-check-precommit.json`
(SHA256
`9fe9febc9333e481c810caa8736cfb5ca7c28fe1439805060caf5a5c935497a4`).
EXP-0001 through EXP-0003 are machine-recorded against source revision
`5b9bfab072e8cc28a7e92c9e956608db591b246c`.
EXP-0004 is machine-recorded against its validated source revision
`49fbcad2e2b761d9de50312f03335e27236a8a13`.

The patch opens the exact `(Dqk,Dv)=(512,256)` SM90 dimension/tile
specialization but is not itself a mask-mode guard. The project adapter admits
only the locked 32Q/4KV global-causal text contract.

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

## Global backward first compile gate: EXP-0005 REJECT / REFINE

The exact two-slab global forward presents each FA4 launch as d512 Q/K with a
d256 V/output slab and model-level GQA-8. Autograd reached the pinned
`FlashAttentionBackwardSm90` constructor, which rejects that combination:

```text
AssertionError: GQA backward requires head_dim == head_dim_v
```

The failure occurs before main backward compilation, so no main cache key,
cubin, real numerical comparison, sanitizer result, or performance result
exists for global backward. It rejects only direct GQA-8 use of the pinned
unequal-dimension backward. Internal 4-to-32 KV-head expansion would preserve
the model algebra and bypass the assertion, but static accounting rejects the
unchanged monolithic path at 320 modeled accumulator registers and 336 KiB
core shared storage. An explicit-FakeTensorMode head-expanded diagnostic
compiled with exact full gradient shapes, but real launch validation rejected
345,088 allocated bytes against the 232,448-byte SM90a limit. The next
experiment must pair exact head expansion with D-chunked dQ accumulation or
separate Q-major dQ and K-major dK/dV ownership.

## Gate table

| Gate | Status | Evidence / stop condition |
|---|---|---|
| H100 identity | **PASS** | H100 80GB, CC 9.0, driver/toolkit above |
| Pinned CUDA-12.8 FA4 environment | **PASS** | Strict check including exact patch stack and profilers |
| CPU/model contract on H100 | **PASS** | Oracle status OK; full H100 suite below |
| Local d256 text forward | **PASS** | O/LSE, boundaries, GQA 1/2/4/8, stream repeat |
| Global d512 text forward | **PASS (composed)** | O/LSE through S1024, sanitizer and SASS evidence |
| Local d256 backward | **PASS (scoped)** | EXP-0003 reject preserved; EXP-0004 matrix/oracle, stream/repeat, sanitizers, SASS |
| Global d512 backward | **REJECT / REFINE** | EXP-0005 direct GQA-8 slab path hits unequal-dimension constructor assertion; no real run |
| Multimodal local fwd/bwd | **NOT RUN** | Follows global backward in the ordered gate |
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

Latest full H100 pytest result after applying the exact patch stack and the
EXP-0004 probe-policy tests:

```text
101 passed, 2 skipped, 1 xfailed
```

The two skips are fake-compile-only tests in real execution. The expected
failure is the pinned Transformers generic FA4 mask adapter, which cannot
encode Gemma's vision future-token exception. The dedicated multimodal kernel
gate did not run. Local backward hardware acceptance comes from the explicit
EXP-0004 probe matrix and sanitizer runs above; the aggregate pytest result is
not presented as a substitute for that evidence.

Reproduce both immutable numerical decisions exactly:

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
```

The next experiment must test an exact internal head-expansion global d512
backward with a structural dQ/dKV resource split. Do not skip ahead to
multimodal work or performance tuning, and do not rewrite any prior
experiment's decision.

## Deferred scope

B300/SM103, varlen, long-context production lengths, fused single-launch
global d512, real global backward, multimodal masking, backward GQA ratios
beyond the exact validated local model ratio 2, and all performance work
remain deferred. No H100 result is generalized to B300.
