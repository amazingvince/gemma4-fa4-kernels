# EXP-0007: H100 exact local multimodal mask

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-fwd / local-d256-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar path and revision: pinned `flash_attn/cute/mask.py`,
  `flash_fwd_sm90.py`, `flash_bwd_sm90.py`, `interface.py`, and
  `tests/cute/mask_mod_definitions.py`
- Design brief: `docs/h100-m1-local-multimodal-design.md`

## Invariant changed

Add one static, hash-stable SM90 mask callable and one runtime contiguous
INT32 `(S,)` vision-ID auxiliary tensor to the accepted local d256 path. The
public B=1 `(1,S)` IDs are normalized to that private layout. The mask callable, rather
than FA4's native causal/local flags, owns the complete Gemma predicate:

```text
k > q - 1024
AND
(k <= q OR same nonnegative vision block)
```

No WGMMA atom, tile ownership, TMA pipeline, barrier, stage, accumulation
type, epilogue, scale, or prepared-Q/K/V invariant changes.

## Hypothesis

The pinned SM90 custom-mask path, called with `causal=False` and no native
window, will apply Gemma's exact local vision overlay in both forward and
transposed backward and pass the fixed-length B1/32Q/16KV/d256 matrix without
changing the accepted text path.

The hypothesis is falsified by a compile/launch error, a mask boundary or
gradient mismatch, a cache variant keyed by runtime IDs/length, a sanitizer
report, or an unexpected pipeline/resource change.

## Single change

One mask/auxiliary dispatch path for fixed-length local multimodal attention.
Text dispatch remains byte-for-byte semantically equivalent to EXP-0001 and
EXP-0004.

## Correctness evidence

- [x] locked contract and pinned Transformers oracle
- [x] exact fake-tensor forward/backward compile
- [x] boundary/tail/vision-pattern matrix
- [x] O and FP32 LSE
- [x] separate dQ, dK, and dV under the EXP-0004 numerical policy
- [x] LSE-only and combined O/LSE gradients
- [x] invalid/fallback cases
- [x] repeated-run and nondefault-stream checks

The first real S1 backward attempt rejected the public `(1,1)` auxiliary:
CuTe could not choose a leading dimension because both dimensions had stride
one and neither had size greater than one. The public contract remains
`(1,S)`, while the adapter now validates and flattens it to a private `(S,)`
INT32 auxiliary. S1 then passed real forward and backward. INT64 input is
accepted only after an exact INT32 range check, preventing identity-changing
wraparound.

Real forward passed at S=`1,31,32,33,63,64,65,127,128,129,1023,1024,1025`
with all-text IDs, ID zero, mixed text/two-vision-block inputs, adjacent block
boundaries, and an all-vision S1025 edge. O stayed within
`atol=0.03125, rtol=0.02`; FP32 LSE stayed within `atol=0.125, rtol=0`.
Adversarial cases proved that a very large masked K/V token cannot change the
selected row, a same-block future V token does change it, and at S1025:

```text
(q=0, k=1024)    allowed
(q=1024, k=0)    rejected
(q=1024, k=1)    allowed
```

The same 13-length matrix passed separate dQ/dK/dV checks against FP32 and
independent BF16 references. Across the O-only matrix, candidate maximum
absolute error ranges were dQ=`2.2769e-5..0.5`, dK=`2.4796e-5..0.5`, and
dV=`0..0.0625`; every cell passed the frozen upstream-relative rule. S65
LSE-only gradients passed (dQ/dK max `0.0625`, dV exactly zero), and S129
combined O/LSE gradients passed. An isolated `dO[q=63,qhead=9]` proved that
only KV head 4 receives dK/dV, same-block future key 64 receives both, key 65
in a different block receives exactly zero, and inactive dQ ownership remains
exactly zero.

Three same-input S33 runs were bitwise equal for all three gradients. The
same case passed with producer, kernel, and consumer work on a nondefault CUDA
stream. The aggregate H100 bundle at source revision
`d1b7e4ad0b1ffff6e3190a4b4411603cd544afe4` reported:

```text
128 passed, 3 skipped, 1 xfailed
```

The skips are fake-only compile tests during real execution. The xfail is the
pre-existing generic Transformers 2D FA4 adapter, which cannot carry Gemma's
future-token vision exception; this dedicated adapter is the accepted path.

## Synchronization and generated code

- [x] memcheck
- [x] synccheck
- [x] racecheck
- [x] IR/PTX/SASS observation
- [x] registers/spills/static SMEM recorded

Memcheck and synccheck each reported `ERROR SUMMARY: 0 errors`, and racecheck
reported `0 hazards displayed (0 errors, 0 warnings)` at S128 and the S129
partial-tile boundary. S1025 all-vision memcheck also reported zero errors.

An isolated artifact cache established fingerprint
`6167b82729b6e91949486c7a469474318b0906e396849651633c18d6f82aef92`.
The value/length-independent forward key is
`4e2c1a85688dc3d0339e1477b4812214a6fdfd3ddf71b394a55390763b8c9bcf`.
Backward produced the declared bounded one-block and multi-block keys
`68819486f27454dd2016a731551962c5e5d9831710344697b1e8f633e4d1a20e`
and `cea1f69494a34c52ffd23d3c3ad430db01d797b0a7d1008fa3abaaa2784e5030`;
both cached main objects had SHA256
`a3807e1d43bc4026ed416bd18c95e3cfa451da626d94751bdac22fe300d563ff`.

Retained forward PTX/cubin SHA256 values are
`38375a2fc124ed2511009c01dbbd0f65433e127c84d19bb820eaf189e64bc3ba` /
`dcb7e009bc657b524bd224e515dd1358d0b62d6d8c8ab3eacd3449d5bd462398`;
backward PTX/cubin values are
`9e5ddf4438444035b2bc091e76c28de0c88fe8fffacc35e73643f3966899b6ec` /
`511e6c325c0625aa98d1053805ae8182e2862b006e5371e427b796c2e27f2b78`.
SASS confirms
the realized custom forward M128 x N80 path (32
`HGMMA.64x80x16.F32.BF16`, 10 `HGMMA.64x256x16.F32.BF16`, 20 `UTMALDG`,
four `UTMASTG`, four `WARPGROUP.DEPBAR`) and the M64 x N64 backward (32
`HGMMA.64x32x16.F32.BF16`, 12 `HGMMA.64x128x16.F32.BF16`, 24 `UTMALDG`,
five `WARPGROUP.DEPBAR`). Both use 168 registers and 1 KiB static shared
memory. Backward has zero stack/local memory; forward has a 40-byte stack
frame with scalar `LDL`/`STL` traffic and zero separately reported local
allocation. Dynamic shared memory remains unmeasured, so no occupancy or
performance claim is made.

Pinned upstream key-discipline debt: `num_stages_PdS` reaches the SM90
backward constructor but is absent from its cache key. It is invariant at one
for this d256 specialization, so EXP-0007 has no collision; any later tuning
of that stage count must first repair the key.

## Measurement

No performance measurement is authorized for this correctness experiment.

## Decision

**ACCEPT**, scoped to fixed-length contiguous BSHD, B1,
S=`1..1025`, BF16, 32Q/16KV GQA-2 d256, exact scale 1.0, distinct K/V, and
the exact runtime vision-block predicate on this pinned SM90 environment.
Text inputs with `vision_block_ids=None` retain the earlier native local path.

This decision does not accept packed varlen, B>1, document/padding metadata,
production long context, a block-sparse schedule, the generic Transformers
FA4 adapter, performance, B300, or any architecture other than SM90.

## Record

The H100 record was appended through the schema-validating writer against the
immutable implementation revision:

```bash
python scripts/record_result.py EXP-0007 \
  --kernel h100-local-d256-multimodal --arch sm_90 --decision accept \
  --git-sha d1b7e4ad0b1ffff6e3190a4b4411603cd544afe4 \
  --hypothesis \
    'exact custom mask satisfies Gemma local vision forward and transposed backward'
```
