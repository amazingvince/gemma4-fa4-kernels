# H100 M1 status

**Status date:** 2026-07-19

**Ordered gate result:** stopped at local d256 backward numerical correctness.
The H100 environment, local d256 text forward, and composed global d512 text
forward passed. Local backward compiled and ran but dQ and dK exceeded the
tolerance fixed before execution. Global backward, multimodal masking, and all
benchmarks were therefore not run.

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

## Backward gate: FAIL / STOP

The exact local d256 fake compile succeeded and produced separate shapes:

```text
dQ=(1,128,32,256)
dK=(1,128,16,256)
dV=(1,128,16,256)
```

The first real S=128 comparison then failed the frozen
`atol=0.125, rtol=0.05` envelope:

- first dQ failure: 552 / 1,048,576 mismatches, max abs `0.2890625`, max
  relative `1112.0`;
- diagnostic rerun without changing tolerance: dQ max/mean abs `0.28857422` /
  `0.0097916815`, dK `0.375` / `0.013713409`, and dV `0.0625` /
  `0.00076462259`;
- dQ and dK failed; dV passed.

Per the ordered gate, no longer local matrix, global backward, multimodal
forward/backward, or benchmark was run. See
`experiments/EXP-0003-h100-local-d256-backward.md`.

## Gate table

| Gate | Status | Evidence / stop condition |
|---|---|---|
| H100 identity | **PASS** | H100 80GB, CC 9.0, driver/toolkit above |
| Pinned CUDA-12.8 FA4 environment | **PASS** | Strict check including exact patch stack and profilers |
| CPU/model contract on H100 | **PASS** | Oracle status OK; full H100 suite below |
| Local d256 text forward | **PASS** | O/LSE, boundaries, GQA 1/2/4/8, stream repeat |
| Global d512 text forward | **PASS (composed)** | O/LSE through S1024, sanitizer and SASS evidence |
| Local d256 backward | **FAIL / STOP** | dQ and dK exceed frozen envelope at first S128 run |
| Global d512 backward | **NOT RUN** | Blocked by ordered prior gate |
| Multimodal local fwd/bwd | **NOT RUN** | Blocked by ordered prior gate |
| Benchmarks | **NOT RUN** | Blocked by correctness; no performance claim |

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

Latest full H100 pytest result after applying the exact patch stack:

```text
96 passed, 2 skipped, 1 xfailed
```

The two skips are fake-compile-only tests in real execution. The expected
failure is the pinned Transformers generic FA4 mask adapter, which cannot
encode Gemma's vision future-token exception. The dedicated multimodal kernel
gate did not run.

Reproduce the stopping failure exactly:

```bash
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0003-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference
```

The next session must open a new experiment with one hypothesis about the
SM90 d256 backward accumulation/configuration path. Do not resume with global
backward, multimodal work, or performance tuning, and do not loosen EXP-0003's
tolerance after observing the failure.

## Deferred scope

B300/SM103, varlen, long-context production lengths, fused single-launch
global d512, all backward beyond the rejected local baseline, multimodal
masking, and all performance work remain deferred. No H100 result is
generalized to B300.
