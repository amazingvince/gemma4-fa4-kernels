# EXP-0008: H100 local packed variable length

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-varlen-fwd / local-d256-varlen-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Design brief: `docs/h100-m1-local-varlen-design.md`

## Invariant changed

Add packed Tq/Tk inputs, CUDA INT32 cumulative lengths, lower-right query
coordinates, and globally indexed K-stream vision/document auxiliaries to the
accepted local d256 family. No model mask, scale, prepared-Q/K/V, WGMMA, TMA,
barrier, accumulation, or gradient-separation invariant changes.

## Hypothesis

The pinned public SM90 varlen autograd path will apply Gemma's exact local
predicate independently within every packed sequence in forward and
transposed backward, including Sq less than Sk, without keying compilation on
runtime cumulative values or metadata contents.

Falsification is a compile/launch error, cross-sequence/document leakage,
lower-right or W1024 boundary mismatch, O/LSE/gradient failure, runtime-value
cache growth, sanitizer finding, or unrecorded resource/protocol change.

## Single change

One validated Gemma self-attention packed adapter and one global-index custom
predicate.
Text-only varlen uses FA4's native causal/local path; fixed paths remain
unchanged.

## Correctness evidence

- [x] CPU packed reference and validation suite
- [x] exact text/custom fake compilation
- [x] equal and lower-right length matrices
- [x] O and FP32 LSE
- [x] separate dQ, dK, and dV
- [x] true `dout=None` LSE-only and combined gradients
- [x] vision/document/packed-boundary isolation
- [x] structured ownership
- [x] repeats and nondefault stream

The implementation revision is
`de6450a9cf5040a7432ed7641b230bb29f835248`. The final local suite reported
`105 passed, 67 skipped`; the skips are optional Transformers and H100 gates.
All three packed fake-tensor specializations compiled: native forward, custom
forward/backward, and LSE-only backward with no incoming O gradient.

Forward O and FP32 LSE passed the frozen `atol=0.03125, rtol=0.02` and
`atol=0.125, rtol=0` envelopes. Equal-length batches covered:

```text
[1]
[31,32,33]
[63,64,65,127,128,129]
[1023,1024,1025]
[129,1,65,33]
[33,65,1,129]
```

Native and custom lower-right paths covered Q/K matrices including
`[1,31,64,129] / [33,64,128,1025]`. Mixed, adjacent, and all-vision patterns,
internal document splits, repeated IDs across packed boundaries, and reordered
segments passed. An independent q1/k1025 zero-score sentinel proved that key 0
is excluded by the strict `k > q_abs - 1024` bound while key 1 contributes
exactly.

Backward native text and custom vision/document paths passed O-only,
LSE-only, and combined gradients under EXP-0004's unchanged predeclared
upstream-relative BF16 policy. All outputs were finite BF16 with distinct
dQ/dK/dV storage; observed maximum absolute errors remained at or below
`0.5`, `0.5`, and `0.0625` respectively. LSE-only dV was exactly zero. The
q1/k1025 transposed sentinel gave exactly zero dV at excluded key 0, nonzero
dV at key 1, and zero on every inactive KV head.

An isolated dO at packed sequence 1, q-local 0 / q-absolute 2 / query head 9
proved exact segment ownership, KV-head-4 GQA ownership, same-block future-key
admission, and document blocking. Mutating every K/V element in another
sequence to alternating large values left the first sequence's O, LSE, dQ,
dK, and dV bitwise unchanged. Three same-input runs on a nondefault stream
were bitwise equal for O, LSE, dQ, dK, and dV.

At the implementation revision, strict environment validation and the pinned
Transformers model oracle passed. Aggregate real-H100 pytest reported:

```text
167 passed, 6 skipped, 1 xfailed
```

The skips are fake-only tests in real mode. The expected failure remains the
pinned generic Transformers 2D mask adapter, which cannot carry Gemma's
future-token vision exception.

## Synchronization and generated code

- [x] memcheck
- [x] synccheck
- [x] racecheck
- [x] PTX/cubin/SASS observation
- [x] registers/spills/static SMEM and cache keys recorded

Custom packed single-block `[63,64] / [64,64]` and multi-block
`[64,65] / [64,65]` combined-gradient runs each reported zero memcheck and
synccheck errors and zero racecheck hazards/errors/warnings. The q1/k1025
custom lower-right tail also passed memcheck.

The isolated cache fingerprint is
`6167b82729b6e91949486c7a469474318b0906e396849651633c18d6f82aef92`.
Forward keys are:

```text
native  e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176
custom  4d1d2a598d6dec355c20a97deace1dedd1a883786a303b01703bcaaa9ccccaf3
```

Backward has the declared bounded `(single_q_block,single_k_block)` variants:

```text
native  single/single  94f70ecb48cc211c4f794de100b5d67e18f68e3634cf40fdfafa18f4895bdc4f
native  single/multi   c280928d15a213fea45200cdf8dbb3d1f78c8ddc65f208a104bab52ba14d73ad
native  multi/multi    a3c7d28fb5372354d1d121353d7803b12e6ba713817d650b5a7288237c006ec7
custom  single/single  f383eabe9784058f299c0c874537b45610634c389dea3c0d7f563791c5c9959d
custom  single/multi   89f80d21210df56786c51ed52a2618b90e6d0addc7d20101621288444e289fe6
custom  multi/multi    5dfd282f2c45c19935622b1ba93f13b5fc541622f6becc3c5358aea4bd335269
```

The three native main objects share SHA256
`c98de0355c9a49c69734f945f67939ace70d7d6d0eff9703e68d790f3c56742a`;
the three custom objects share
`ed497af90df240bced141d943136154afc471f775e6ab32edbeaad4530ba9424`.
Changed Tq/Tk totals, cumulative values, tensor values, segment order, and
metadata contents produced no additional native or custom forward/backward
objects.

Retained custom multi-block forward PTX/cubin SHA256 values are
`5bc4d43ee077e82fed1382f05358a39e63290090c061070d752e7a990d9dad36` /
`e05789014b7e144c61f6569df2d4046054fc8bee51b31f1fe52a7e2c2dae06ea`;
backward values are
`092ea5cf88991a4c4fd518db360977a85a81ed7f9f45db35b9e3f64745e90619` /
`a349dcff99ed60a905559069354ff7ac8336f09a08f05b989d12f65629f3dd5c`.
Both PTX files are version 8.8 targeting `sm_90a`.

SASS realizes the expected M128 x N80 forward (32
`HGMMA.64x80x16.F32.BF16`, 10 `HGMMA.64x256x16.F32.BF16`, 20 `UTMALDG`,
four `UTMASTG`, four `WARPGROUP.DEPBAR`) and M64 x N64 backward (32
`HGMMA.64x32x16.F32.BF16`, 12 `HGMMA.64x128x16.F32.BF16`, 24 `UTMALDG`,
five `WARPGROUP.DEPBAR`). Both main kernels use 168 registers and 1 KiB
static shared memory with zero separately reported local memory. Backward has
zero stack; packed custom forward has a 104-byte stack frame and scalar
LDL/STL traffic. The unchanged backward configuration models 208 KiB of core
dynamic storage. Forward dynamic shared memory remains unmeasured, so no
occupancy or performance claim is made.

## Measurement

No performance measurement or speed claim is authorized for EXP-0008.

## Decision

**ACCEPT**, scoped to nonempty packed local self-attention on H100 SM90 with
`B>=1`, per-sequence `1 <= Sq <= Sk <= 1025`, BF16, exact 32Q/16KV GQA-2
d256, scale 1.0, distinct K/V, CUDA INT32 cumulative arrays, and optional
K-stream vision/document IDs. Text uses native lower-right causal W1024;
metadata uses the complete dense custom predicate.

This decision does not accept empty sequences, context above 1025, a
block-sparse schedule, generic cross-attention/framework dispatch,
deterministic gradients, performance, B300, or any architecture other than
SM90.

## Record

The schema-validated EXP-0008 acceptance is appended to
`experiments/results.jsonl` against implementation revision
`de6450a9cf5040a7432ed7641b230bb29f835248`.
