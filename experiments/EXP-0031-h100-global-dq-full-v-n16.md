# EXP-0031: combine V slabs in an H100 N16 dQ kernel

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dq
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: CUTLASS v4.6.0 Hopper `fmha.py` at `e6233cba...` and pinned FA4
  `flash_attn/cute/flash_bwd_sm90.py`
- Design brief: `docs/h100-global-dq-full-v-n16-design.md`

## Invariant changed

Each dQ-only specialization owns the full V512/dO512 contribution to dP and
dPsum before forming dS, using M64xN16 so the accepted single-stage shared
storage remains at the H100 per-block ceiling. dKV ownership stays slabbed and
unchanged.

## Hypothesis

Combining the two V256 contributions before dS cuts global dQ main launches
from four to two and halves duplicated QK/dQ arithmetic, improving global S8K
backward median by at least 10% without violating the 232,448-byte SMEM ceiling
or the locked gradient contract.

## Single change

Replace only the two-slab N32 dQ specializations with two full-V512 N16 dQ
specializations (D256 offsets 0/256). Retain both accepted N32/Dv256 dKV
launches, existing pipeline/barriers, and all postprocess paths.

## Correctness evidence

- [ ] locked contract and optional HF oracle
- [ ] targeted reference matrix including boundary/tail/adversarial case
- [ ] O, LSE, dQ, dK, dV as applicable
- [ ] invalid/fallback cases
- [ ] repeated-run/determinism check

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck or documented, reproduced tool false positive
- [x] compile-stage layout discriminator recorded
- [ ] registers/spills/SMEM/TMEM recorded (compilation stopped first)

## Measurement

- Clock/power state: unlocked and labeled; capture pre/post samples
- Hot/cold L2: hot screen, hot/cold confirmation on acceptance
- Warmup/repetitions/statistic: 10/30; CUDA events; median/p25/p75/IQR
- Semantically equivalent baseline: EXP-0029 exact project FA4 adapter

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 98.239 / 0.379 ms | pending | pending |
| global S8K fwd_bwd hot | 104.533 / 0.647 ms | pending | pending |
| global S64K bwd hot | 6043.744 / 3.456 ms | pending | pending |

## Decision

REJECT

The two-warpgroup N16 specialization fails during CuTe compilation before a
kernel or resource report is emitted. `reshape_acc_to_frgA(acc_dP)` receives
an N8 accumulator partition per warpgroup; QuACK's SM90 back-to-back GEMM
conversion cannot form the required K16 dS operand from that isolated half and
rejects the odd accumulator-atom count. No kernel launched, so no correctness,
sanitizer, or timing claim exists. The accepted EXP-0029 patch was restored.

The coherent next variant is one MMA warpgroup owning the full N16 score/dP/dS
tile. That changes pipeline consumer and named-barrier participant counts from
256 to 128 and therefore requires a separate reviewed design before code.

## Record

```bash
python scripts/record_result.py EXP-0031 \
  --kernel global-d512-dq-full-v-n16 --arch sm_90 \
  --decision <accepted|rejected> --hypothesis '<measured result>' --bench <jsonl>
```
