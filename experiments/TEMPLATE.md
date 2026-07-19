# EXP-NNNN: one invariant, one hypothesis, one change

- Date / author:
- Kernel family: local-d256-fwd | local-d256-bwd | global-d512-fwd | global-d512-dq | global-d512-dk | global-d512-dv | integration
- Architecture: sm_90 | sm_103
- Upstream FA4 revision:
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env` for sm_90 or
  `configs/env/latest-compatible.env` for sm_103
- Model-contract lock hash:
- Exemplar path and revision:

## Invariant changed

State the ownership, layout, pipeline, masking, or scheduling invariant. Do not
write only a code diff.

## Hypothesis

One falsifiable sentence tied to a profiler counter or correctness risk.

## Single change

Exact constexpr, layout, stage transition, scheduler rule, or code path.

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
- [ ] IR/PTX/SASS observation
- [ ] registers/spills/SMEM/TMEM recorded

## Measurement

- Clock/power state:
- Hot/cold L2:
- Warmup/repetitions/statistic:
- Semantically equivalent baseline:

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|

## Decision

ACCEPT | REJECT | REFINE

Why, remaining risks, and next single hypothesis:

## Record

```bash
python scripts/record_result.py EXP-NNNN   --kernel <name> --arch <sm_90|sm_103> --decision <...>   --hypothesis '<sentence>' --bench <jsonl>
```
