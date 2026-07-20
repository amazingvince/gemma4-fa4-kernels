# EXP-0009: H100 packed local text at production lengths

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-fwd and local-d256-bwd integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar path and revision: pinned
  `flash_attn/cute/{interface,flash_fwd_sm90,flash_bwd_sm90,block_info}.py`
- Design brief: `docs/h100-m1-local-long-context-design.md`

## Invariant changed

The adapter's native packed-text admission bound changes from the EXP-0008
evidence ceiling of K1025 to the locked Gemma maximum K262144. The native
lower-right causal W1024 scheduler, mask, accumulation, tensor geometry,
pipeline, and generated kernels do not change. Metadata-bearing calls retain
the S1025 ceiling until an exact long-context sparse schedule is separately
proved.

## Hypothesis

The pinned SM90 native packed local forward/backward path uses runtime
sequence-local INT32 coordinates and a bounded W1024 tile schedule, so widening
only text admission through K262144 preserves exact lower-right O/LSE/dQ/dK/dV
semantics without making sequence values part of the compile cache key.

Falsification is any far-offset or strict-window mismatch, packed-sequence
leakage, failed gradient oracle, runtime-length cache growth, sanitizer finding,
or changed main-kernel code/resource signature.

## Single change

Admit nonempty native packed text with per-sequence
`1 <= Sq <= Sk <= 262144`; preserve the custom metadata `Sk <= 1025` guard.
No kernel, tile, stage, barrier, or mask callable changes in this experiment.
The implementation revision is
`9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf`.

## Correctness evidence

- [x] locked contract and optional HF oracle
- [x] CPU admission/rejection tests
- [x] long fake forward/backward/LSE-only compilation
- [x] tractable >1025 equal/lower-right reference cases
- [x] exact Q1/K262144 far-offset O/LSE/dV sentinel
- [x] long ragged packed-boundary isolation
- [x] repeated-run and nondefault-stream checks
- [x] memory preflight and full-model-maximum execution

The final local suite reported `110 passed, 71 skipped`. The skips are optional
Transformers and H100 gates. The fake-only H100 test file reported
`30 passed, 63 skipped`, including a max-context forward/autograd-backward
compile. Strict environment validation and the pinned Transformers oracle
passed on the target. Aggregate real-H100 pytest reported:

```text
175 passed, 7 skipped, 1 xfailed
```

The seven skips are fake-only cases in real mode. The expected failure remains
the pinned generic Transformers 2D mask adapter, which cannot represent
Gemma's future-token vision exception.

At S2048, a three-repeat numerical run on a nondefault stream compared forward
and `out_lse` backward against independent FP32 and BF16 references. O, LSE,
dK, and dV were bitwise equal across candidate repeats. dQ was non-bitwise,
with maximum pairwise difference `0.03125`, but each repeat's dQ, dK, and dV
passed EXP-0004's unchanged upstream-relative BF16 policy. This is evidence for
numerical validity, not deterministic dQ.

A nondefault-stream S32768 `out_lse` run returned finite O, FP32 LSE, and
separate finite BF16 dQ/dK/dV with exact shapes. The true full model maximum,
Q=K=262144, also completed `out_lse` forward/backward. Its corrected preflight
estimated `45,231,374,336` live bytes with `84,465,025,024` bytes free before
allocation. This is an execution/correctness smoke, not a performance result.

An analytic Q1/K262144 zero-score sentinel set
`q_abs=262143`, the strict excluded key to 261119, and the first included key
to 261120. The excluded V contribution and dV were exactly zero, the included
value contributed exactly 2.0, LSE equaled `log(1024)`, and inactive KV heads
had exactly zero dV. This proves far-offset lower-right arithmetic and the
strict `k > q_abs - 1024` boundary at the locked maximum.

Long ragged Q=`[33,65]`, K=`[2049,4097]` hostile-isolation runs compared both
base and mutated batches with independent FP32/BF16 references. Mutating the
other packed sequence left the protected sequence's reference dQ/dK/dV exact;
candidate O/LSE/dK/dV were exact across the control, and both candidate
gradient triplets passed the upstream-relative oracle. The expected dQ
reduction-order drift was measured separately instead of being hidden behind a
fixed absolute tolerance.

## Bounded-work argument

For each forward M tile, the pinned
`BlockInfo.get_n_block_min_max` computes the lower-right causal/local K-block
range from runtime `seqlen_q`, `seqlen_k`, and W1024. For each backward N tile,
`BlockInfo.get_m_block_min_max` computes the transposed bounded Q-block range.
The adapter fixes `num_splits=1`, so the host split heuristic cannot turn the
long runtime K maximum into a new structural specialization. These scheduler
ranges, plus the maximum-context execution and unchanged cache/code evidence,
are the basis for the bounded native-text claim.

No claim is based on `seqlen_k_loaded`: it is not the proof of bounded work for
this path.

## Synchronization and generated code

- [x] memcheck
- [x] synccheck
- [x] racecheck
- [x] cache-key boundedness across long runtime lengths
- [x] PTX/cubin/SASS and registers/spills/SMEM recorded

Native packed Q=`[64,65]`, K=`[2048,2049]` `out_lse` runs reported zero
memcheck and synccheck errors and zero racecheck hazards/errors/warnings. The
Q1/K262144 boundary sentinel also reported zero memcheck errors.

Changing long runtime totals and maxima from Q=`[64,65]`, K=`[2048,4097]` to
Q=`[129,33]`, K=`[8193,2049]` created no additional objects. A forward-only
cache contained one object, with the same native forward key recorded by
EXP-0008:

```text
e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176
```

The backward cache contained four objects total: that forward object, the
unchanged native multi/multi main-backward key
`a3c7d28fb5372354d1d121353d7803b12e6ba713817d650b5a7288237c006ec7`,
and the bounded preprocess and postprocess objects. Thus runtime lengths and
cumulative values did not enter the compile key.

Retained native forward PTX/cubin SHA256 values are
`4387297bdea5f445674c6321a3d41b1b731cc3b1cc39abfe5b573e121e9a6c6a` /
`1179e35cf31a480779595f089c8f7de9b065d264324d2a85ac2526c938db3773`;
main-backward values are
`72ff49bcd21d0a5742d1d269d61acc3d63dcc74caacd1162a9879e939f38d0c9` /
`8a1e7c5efc0d27e46ae4593fc21ba5872307cb1e623845b025fff7904ae47ea5`.
Both PTX files are version 8.8 targeting `sm_90a`.

Generated code remains the accepted native M128 x N80 forward and M64 x N64
backward. Forward contains 64 `HGMMA.64x64x16.F32.BF16`, 16
`HGMMA.64x256x16.F32.BF16`, 20 `UTMALDG`, four `UTMASTG`, and eight
`WARPGROUP.DEPBAR` instructions. Backward contains 32
`HGMMA.64x32x16.F32.BF16`, 12 `HGMMA.64x128x16.F32.BF16`, 24 `UTMALDG`, and
five `WARPGROUP.DEPBAR` instructions. Both main kernels use 168 registers,
1 KiB static shared memory, zero stack, and zero separately reported local
memory; no scalar LDL/STL spill traffic was observed. No resource, occupancy,
or performance improvement is claimed.

## Measurement

No performance measurement or speed claim is authorized for EXP-0009.

## Decision

**ACCEPT**, scoped to native nonempty packed local text on H100 SM90 with
`B>=1`, per-sequence `1 <= Sq <= Sk <= 262144`, BF16, exact 32Q/16KV GQA-2
d256, scale 1.0, distinct K/V, CUDA INT32 cumulative arrays, and lower-right
causal W1024 semantics.

This decision does not accept metadata-bearing vision/document calls above
1025, empty segments, deterministic dQ, performance, generic framework
dispatch, B300, or any architecture other than SM90. Long metadata is the next
H100 gate and requires an exact block-sparse schedule; it must not be admitted
by silently using dense custom-mask work.

## Record

The schema-validated EXP-0009 acceptance is appended separately through
`scripts/record_result.py` against implementation revision
`9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf`.
