# EXP-0032: one warpgroup owns full-V N16 dQ

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dq
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: CUTLASS v4.6.0 Hopper `fmha.py` at `e6233cba...` and pinned FA4
  `flash_attn/cute/flash_bwd_sm90.py`
- Design brief: `docs/h100-global-dq-full-v-n16-one-wg-design.md`

## Invariant changed

For dQ-only full-V512 M64xN16 specializations, one 128-thread MMA warpgroup
owns QK, dP, dS, and the full D256 dQ output tile. Pipeline/PdS consumer counts
become 128 and the sole dQ handoff has 160 participants including the store
warp. dKV remains the accepted two-WG N32 slab path.

## Hypothesis

One WG owning N16 avoids EXP-0031's invalid independent N8 fragments while the
halved QK/dQ arithmetic improves global S8K backward median by at least 10%,
without SMEM overflow, spills, or a synchronization/numerical regression.

## Single change

Introduce the scoped full-V512 N16 one-WG dQ variants only, including their
derived 256-thread block, 128-thread pipeline/PdS counts, 160-thread dQ
handoff, 240-register WG budget, exact cache key, and full dPsum buffer.

## Correctness evidence

- [x] human review of pipeline/barrier participant design (approved 2026-07-20)
- [x] locked contract retained; no contract or tolerance change
- [x] fake compile and targeted S128 reference on a nondefault stream
- [x] O, LSE, dQ, dK, and dV passed for three repetitions
- [x] repeated-run evidence recorded; O/LSE were exact, while the accepted
  nondeterministic FP32 gradient accumulation remained non-bitwise-exact
- [ ] invalid/fallback matrix (not run after the performance gate rejected it)

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck or documented, reproduced tool false positive
- [ ] IR/PTX/SASS observation
- [x] memory preflight: 33,652,736-byte peak delta was below the
  37,863,424-byte estimate
- [ ] registers/spills/SMEM/TMEM from generated code (not retained after reject)

## Measurement

- Clock/power state: unlocked and labeled; pre/post capture
- Hot/cold L2: hot screen; hot/cold confirmation if accepted
- Warmup/repetitions/statistic: 10/30 CUDA events; median/p25/p75/IQR
- Semantically equivalent baseline: EXP-0029 exact project FA4 adapter

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 98.239 / 0.379 ms | 102.102 / 1.345 ms | +3.93% |
| global S8K fwd_bwd hot | 104.533 / 0.647 ms | not run | rejected at bwd gate |
| global S64K bwd hot | 6043.744 / 3.456 ms | not run | rejected at S8K gate |

## Decision

REJECT

The candidate was 3.93% slower than the accepted baseline, with a wider and
non-overlapping IQR, and missed the predeclared requirement for at least a 10%
improvement. Per the staged gate, fwd_bwd, S64K, sanitizer, and generated-code
work were not run after the decisive S8K bwd rejection.

The final candidate patch SHA256 was
`fbb86c150457d0b5cf43f162716af1215d37600a740d35ba0ea6233f7035ce7b`.
Both the local and remote FlashAttention trees were restored to the accepted
patch SHA256
`eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`,
and the strict H100 environment check completed with no warnings or errors.

This result rejects the one-warpgroup ownership choice, not the broader goal
of combining both V256 slabs in one dQ CTA. Keeping two MMA warpgroups and
time-sharing the accepted V/dO SMEM slots is the preferred exact-BF16 follow-up.

## Record

```bash
python scripts/record_result.py EXP-0032 \
  --kernel global-d512-dq-full-v-n16-one-wg --arch sm_90 \
  --decision reject --hypothesis '<measured result>' --bench <jsonl>
```
