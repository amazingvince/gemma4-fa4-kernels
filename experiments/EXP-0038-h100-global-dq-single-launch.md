# EXP-0038: H100 global dQ full-D single launch

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dq
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: pinned FA4 `flash_attn/cute/flash_bwd_sm90.py`
- Design brief: `docs/h100-global-dq-d512-single-launch-design.md`

## Invariant changed

One dQ-only main launch forms the accepted full-D score/dP/dS once, then
sequentially produces D256 low and high dQ accumulation slabs through the
same register fragment, shared epilogue arena, and empty/full barriers.

## Hypothesis

Replacing the two EXP-0035 dQ launches with one sequential full-D launch
reduces exact global S8K backward median by at least 10% relative to EXP-0037,
with non-overlapping IQRs and no correctness, sanitizer, spill, memory, or
S64K regression.

## Single change

Add one guarded `dq_d512_stream` specialization and adapter route. The
accepted full-D dKV kernel, forward kernels, preprocessing math, postprocess
conversion, mask, scale, dtype, tolerance, and public API do not change.

## Correctness evidence

- [x] reviewed ownership and barrier lifecycle
- [x] locked contract and optional HF oracle
- [x] fixed and packed reference matrices
- [x] O, FP32 LSE, separate dQ/dK/dV, ownership, and isolation
- [x] invalid/candidate-off fallback cases
- [x] repeated and nondefault-stream cases

The fixed matrix passed at S1/31/32/33/63/64/65/127/128/129. Native packed
tiny, mixed, reversed, and mixed-empty cases passed, including O+LSE
gradients, hostile inactive-slice mutation, structured GQA ownership, and
exact-zero empty-query ownership. The S128 nondefault-stream case passed
three independent repeats. The explicit flag-0 rollback passed the same
reference gate. As expected for the retained FP32 reduction protocol,
gradients are numerically conformant but not bitwise deterministic.

## Synchronization and generated code

- [x] memcheck
- [x] synccheck
- [x] racecheck
- [x] PTX/SASS instruction and launch-count inspection
- [x] registers, stack/local, and dynamic SMEM recorded
- [x] bounded fixed/packed compile-cache inventory

Fixed S128 and native packed mixed-empty workloads report zero memcheck
errors, synccheck errors/warnings, and racecheck hazards. Nsight Systems
records exactly two backward main launches: the unchanged full-D dKV kernel
and the new one-launch dQ kernel. The dQ object uses 384 threads, 201,728
dynamic shared bytes, 168 registers, zero local memory, 68 HGMMA and 56 UTMA
instructions. Its fixed form has zero stack bytes; the packed scheduler has
the same 24-byte stack and 17 LDL / 8 STL instructions as the accepted packed
dQ route, with no spill growth. Fresh fixed and packed caches contain exactly
one dKV and one dQ main object per ABI.

Observed PyTorch peak allocation was 33,652,736 bytes against a 37,847,040
byte fixed bound and 47,141,376 bytes against a 66,888,448 byte packed bound.

## Measurement

- Clock/power: unlocked; clock locking was not permitted
- L2: S8K hot first gate; hot/cold confirmation after acceptance
- Warmup/repetitions: 10/30 CUDA events, median/p25/p75/IQR
- Baseline: accepted EXP-0037 exact BF16 path

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 51.318 / 0.274 ms | 43.040 / 0.222 ms | -16.1% |
| global S8K fwd_bwd hot | 57.113 / 0.370 ms | 48.659 / 0.184 ms | -14.8% |
| global S8K bwd cold | 51.265 / 0.205 ms | 42.945 / 0.050 ms | -16.2% |
| global S8K fwd_bwd cold | 57.131 / 0.072 ms | 48.803 / 0.114 ms | -14.6% |
| global S64K bwd hot | 3054.023 / 2.193 ms | 2579.334 / 1.273 ms | -15.5% |
| global S64K fwd_bwd hot | 3417.313 / 6.462 ms | 2935.302 / 2.353 ms | -14.1% |

All six baseline/candidate IQRs are disjoint. Each row uses 10 warmups and 30
CUDA-event repetitions on the same H100 with the experiment flag as the only
route change.

## Decision

ACCEPT

The candidate clears the declared correctness, synchronization, generated
code, memory, S8K improvement, and S64K no-regression gates. Promote
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D512_SINGLE_LAUNCH=1` as the default.
Flag `0` remains the immediate rollback to EXP-0037's two dQ launches. This
acceptance applies only to exact BF16 H100 global-causal backward; it does not
claim a deterministic, local-attention, B300, or end-to-end model kernel.

## Record

```bash
python scripts/record_result.py EXP-0038 \
  --kernel global-d512-dq-single-launch --arch sm_90 \
  --decision '<accept|reject>' --hypothesis '<measured result>' --bench <jsonl>
```

The accepted H100 record is appended to `experiments/results.jsonl` against
implementation `1dce18eb5e53163942ebdf1bdd974910ca72e5be`, with the twelve
candidate/rollback rows from the retained benchmark artifact.
