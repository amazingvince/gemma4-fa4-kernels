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

- [ ] human review of pipeline/barrier participant design
- [ ] locked contract and optional HF oracle
- [ ] targeted reference matrix including boundary/tail/adversarial case
- [ ] O, LSE, dQ, dK, dV as applicable
- [ ] invalid/fallback cases and repeated-run check

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck or documented, reproduced tool false positive
- [ ] IR/PTX/SASS observation
- [ ] registers/spills/SMEM/TMEM recorded

## Measurement

- Clock/power state: unlocked and labeled; pre/post capture
- Hot/cold L2: hot screen; hot/cold confirmation if accepted
- Warmup/repetitions/statistic: 10/30 CUDA events; median/p25/p75/IQR
- Semantically equivalent baseline: EXP-0029 exact project FA4 adapter

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 98.239 / 0.379 ms | pending | pending |
| global S8K fwd_bwd hot | 104.533 / 0.647 ms | pending | pending |
| global S64K bwd hot | 6043.744 / 3.456 ms | pending | pending |

## Decision

REFINE

Blocked only at the mandatory human review boundary for changed pipeline and
barrier participant counts. No EXP-0032 kernel code has been written.

## Record

```bash
python scripts/record_result.py EXP-0032 \
  --kernel global-d512-dq-full-v-n16-one-wg --arch sm_90 \
  --decision <accepted|rejected> --hypothesis '<measured result>' --bench <jsonl>
```
