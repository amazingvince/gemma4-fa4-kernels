# EXP-0002: H100 global d512 forward composition

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512-fwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / 2.8.0+cu128
- QuACK runtime helpers: `quack-kernels==0.5.3`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `1e1767ac7a87ba5503a97aa38bbefcccef5fa88e9793f330bdd9b3658fcceb75`
- Strict environment artifact: `agent_space/h100-check-precommit.json`
  (SHA256
  `9fe9febc9333e481c810caa8736cfb5ca7c28fe1439805060caf5a5c935497a4`)
- Exemplar path and revision: pinned
  `flash_attn/cute/flash_fwd_sm90.py` and `interface.py`

## Invariant changed

Keep one mathematical d512 QK score/softmax distribution, but partition the
independent V/O dimension into two contiguous d256 slabs. Each slab launch
recomputes that same distribution from identical Q and K and must return the
same FP32 LSE; concatenating the two O slabs is algebraically identical to one
d512 PV product. K and V remain distinct prepared operands.

## Hypothesis

At the pinned revision, the SM90 asymmetric `(head_dim_qk=512,
head_dim_v=256)` M128 x N32, two-stage specialization compiles, both slab
launches return bitwise-identical LSE, and their concatenated BF16 O matches
the locked FP32 reference with `atol=0.0625, rtol=0.03`; LSE matches with
`atol=0.25, rtol=0`.

## Single change

Use the pinned SM90 kernel twice with one shared Q/K pair and two non-aliasing
V256 slabs. Select the static M128 x N32 configuration found by the pinned
`sm90_config_search.py`. Do not change score scaling, causal masking, softmax,
or output arithmetic, and do not add backward, multimodal masking, or tuning.

The initial isolated probe relaxed the upstream dimension guard only for exact
SM90 `(512, 256)` and called the private tile-selectable entrypoint. After it
passed, the durable path replaced that temporary override with the hash-locked
interface patch
`patches/flash-attention/0001-sm90-d512-v256-forward.patch`. The patch opens
only `(512, 256)` dimensions on SM90 and makes M128 x N32 the public tile
selection; the project adapter separately restricts use to global-causal text.
It uses the normal public `flash_attn_func` without global monkeypatching or
private APIs. Patch SHA256:
`8d3404ccc8bdb2b3fd8e6de09e1f100827d071de283885bf737932bcb41aca4f`.

## Correctness evidence

- [x] locked contract and pinned Transformers oracle
- [x] fake-tensor compile of the exact asymmetric specialization
- [x] O/LSE reference matrix at tile/tail and meaningful global lengths
- [x] exact 32Q/4KV GQA-8 geometry and distinct K/V
- [x] identical LSE from both V-slab launches
- [x] invalid layout/dtype/alias/fallback cases
- [x] repeated-run and nondefault-stream check
- [ ] coordinate-coded and adversarial-value matrix (deferred hardening)

The declared sequence matrix is `1, 31, 32, 33, 63, 64, 65, 127, 128,
129, 511, 512, 513, 1024`. Tolerances above are frozen before the first real
H100 run.

The durable adapter commands and complete focused results were:

```bash
env FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0002-production \
  pytest -q tests/test_h100_fa4.py -k global_d512_forward_fake_compile
# 1 passed, 36 deselected in 2.95s

env FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0002-production \
  pytest -q tests/test_h100_fa4.py -k 'global and not fake_compile'
# 17 passed, 20 deselected in 3.27s
```

The real run comprises two CPU contract tests, the 14 declared H100 sequence
cases, and one nondefault-stream exact-repeat case. The maximum observed O
absolute error was `0.015625`; the maximum FP32 LSE absolute error was
`0.00015258789`. Both are below the frozen envelopes. The adapter compares the
two slab LSE tensors exactly on every real call; no mismatch occurred.

## Synchronization and generated code

- [x] memcheck
- [x] synccheck
- [x] racecheck
- [x] IR/PTX/SASS observation
- [ ] complete registers/spills/dynamic-SMEM record

The pinned static estimator reports 224 KiB modeled shared memory and 152
modeled registers for the M128 x N32, two-warp-group, RS, two-stage
specialization. The filtered sanitizer commands were identical except for
`--tool memcheck`, `synccheck`, and `racecheck`:

```bash
compute-sanitizer --tool <tool> --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_global_d512.py --seqlen 33
```

Memcheck and synccheck each reported `ERROR SUMMARY: 0 errors`; racecheck
reported `0 hazards displayed (0 errors, 0 warnings)`. Without
`--report-api-errors no`, the first memcheck run completed the kernel but
reported 34 `CUDA_ERROR_INVALID_VALUE` failures from CuTe/cuda-python
`cuGetProcAddress_v2` capability probes. That raw run is not represented as
clean; the filter isolates device-memory/synchronization analysis from the
reproduced driver-API instrumentation incompatibility.

The production JIT fingerprint is
`4b2d00f9c628b1f1c6c10e891a241118d95e8d5f77fb3d40863788e1fb39fbc9`
with compile key
`9d458b84f3527bebfcb7ee487274eb29d88742098ee88282c56264cae78ef0f3`.
The host-wrapper object SHA256 is
`3fe53c5e9ded56603c53b5aedda602868c565ac3d983ce80ad422d3c96707018`.

Artifact capture produced PTX SHA256
`3005de409caa06f800e45e5649df1a67ba8c7f721820d0c6fa5ce4e8bd55a83f`
and cubin SHA256
`c67c606ab8dcb08a49302a7389b935e6a22301f19f8be2c53730c5c2534a0561`.
The SASS contains 96 `HGMMA.64x32x16.F32.BF16` instructions, 6
`HGMMA.64x256x16.F32.BF16` instructions, 32 `UTMALDG`, 4 `UTMASTG`, and 6
`WARPGROUP.DEPBAR` instructions. `cuobjdump --dump-resource-usage` reports
168 registers, zero stack, zero local memory, and 1 KiB static shared memory;
there is no local-memory spill evidence. Exact dynamic shared memory remains
unrecorded: the static estimator says 224 KiB core, while Nsight Compute could
not read launch metrics on the pod (`ERR_NVGPUCTRPERM`). The CUDA-12.8 system
`ptxas` also rejected emitted PTX 8.8 because it supports PTX through 8.7, so
the pinned DSL's embedded assembler produced the inspected cubin.

## Measurement

No performance measurement is authorized in this experiment.

## Decision

**ACCEPT**, scoped to fixed-length, contiguous-BSHD, BF16 global causal text
forward on this SM90 environment. The accepted implementation is the pinned,
hash-patched asymmetric kernel composed as two sequential V256 launches. This
does not accept a fused single-launch d512 kernel, varlen, backward,
multimodal masking, exact dynamic-SMEM accounting, performance, or any B300
claim. Coordinate-coded and special-value coverage is also still open.
Duplicated QK/softmax work makes this a correctness-first path only.

## Record

The ledger record names immutable source revision
`5b9bfab072e8cc28a7e92c9e956608db591b246c`:

```bash
python scripts/record_result.py EXP-0002 \
  --kernel h100-global-d512-forward --arch sm_90 --decision accept \
  --git-sha 5b9bfab072e8cc28a7e92c9e956608db591b246c \
  --hypothesis 'two pinned SM90 d512x256 launches compose exact global d512 forward'
```
