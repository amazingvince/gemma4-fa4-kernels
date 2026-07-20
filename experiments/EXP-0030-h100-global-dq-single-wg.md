# EXP-0030: test one dQ warpgroup on H100

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dq
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: CUTLASS v4.6.0 `fmha.py` at `e6233cba...` and pinned FA4
  `flash_attn/cute/flash_bwd_sm90.py`
- Design brief: `docs/h100-global-dq-single-wg-design.md`

## Invariant changed

For each global d512 dQ-only launch, WG0 owns the complete M64 x D256 dQ
WGMMA and accumulation partition while WG1 retains its accepted QK, dP, dS,
pipeline, and barrier participation but does not issue dQ WGMMA.

## Hypothesis

On H100, one warpgroup already saturates the dQ WGMMA issue path, so assigning
dQ to WG0 alone and redistributing registers reduces global S8K backward median
time by at least 3% without spills or a correctness/synchronization regression.

## Single change

Set the existing `dQ_single_wg` constexpr to `True` only for the two global
d512 dQ-only specializations and include the bool in their compile-cache key.

## Correctness evidence

- [x] locked contract retained; no contract or tolerance change
- [x] targeted S128 reference on a nondefault stream
- [x] O, LSE, dQ, dK, and dV passed for three repetitions
- [x] compile key and project routing/lock tests passed
- [x] repeated-run evidence recorded; O/LSE were exact, while the accepted
  nondeterministic FP32 gradient accumulation remained non-bitwise-exact

## Synchronization and generated code

- [ ] memcheck (not run after the performance screen rejected the candidate)
- [ ] synccheck (not run after the performance screen rejected the candidate)
- [ ] racecheck (not run after the performance screen rejected the candidate)
- [ ] IR/PTX/SASS observation (not retained for the rejected candidate)
- [ ] registers/spills/SMEM/TMEM recorded (baseline only)

## Measurement

- Clock/power state: unlocked and labeled; capture pre/post samples
- Hot/cold L2: hot screen, then hot/cold confirmation only if accepted
- Warmup/repetitions/statistic: 10/30; CUDA events; median/p25/p75/IQR
- Semantically equivalent baseline: EXP-0029 exact project FA4 adapter

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 98.239 / 0.379 ms | 98.056 / 1.389 ms | -0.19% |
| global S8K fwd_bwd hot | 104.533 / 0.647 ms | 104.092 / 1.962 ms | -0.42% |

## Decision

REJECT

The candidate missed the predeclared 3% threshold by a wide margin and both
candidate IQRs overlap their baselines. The apparent sub-percent changes are
not distinguishable from unlocked-clock noise. The exact patch, environment
hash, and remote checkout were restored to the accepted two-warpgroup source
and strict `check_env.py` passed with no warnings or errors. No sanitizer or
S64K confirmation time was spent on a candidate already rejected by its first
falsifiable performance gate.

The next hypothesis is structural: compute both V256 slabs' dP contribution in
one dQ specialization so QK/probability and dQ work are not repeated once per
slab. That change requires its own reviewed pipeline/ownership design.

## Record

```bash
python scripts/record_result.py EXP-0030 \
  --kernel global-d512-dq-single-wg --arch sm_90 --decision <accepted|rejected> \
  --hypothesis '<measured result>' --bench <jsonl>
```
