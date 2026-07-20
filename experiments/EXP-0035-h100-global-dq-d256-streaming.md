# EXP-0035: H100 global dQ D256 streaming

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dq
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: pinned FA4 `flash_attn/cute/flash_bwd_sm90.py`
- Design brief: `docs/h100-global-dq-d256-streaming-design.md`

## Invariant changed

For the two dQ-only D256-output specializations, keep full K512/V512 resident
and reuse one Q256/dO256 slot across two lockstep pipeline generations. Both
MMA warpgroups accumulate the two QK and dP partial products in FP32 before
forming P and dS once. The accepted dKV kernels do not change.

## Hypothesis

Reducing four dQ launches to two, while halving repeated QK and dQ arithmetic,
improves exact global S8K backward median by at least 10% with a non-overlapping
IQR and no numerical, sanitizer, spill, or cache-key regression.

## Single change

Add only the scoped `dq_lo_stream` and `dq_hi_stream` variants described in the
reviewed design. No mask, scale, dtype, tolerance, dKV path, or fallback change.

## Correctness evidence

- [ ] human approval of the two-generation pipeline and barrier lifecycle
- [ ] locked contract retained; no tolerance change
- [ ] fake compile and targeted fixed/packed references
- [ ] O, LSE, separate dQ/dK/dV
- [ ] invalid/fallback and repeated nondefault-stream cases

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck or documented, reproduced tool false positive
- [ ] WGMMA SASS and two-generation issue count
- [ ] registers/spills and exact dynamic SMEM

## Measurement

- Clock/power: unlocked and labeled unless locking becomes available
- L2: S8K hot first gate, then hot/cold confirmation
- Warmup/repetitions: 10/30 CUDA events, median/p25/p75/IQR
- Baseline: accepted exact FA4 composition

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 96.928 / 1.308 ms | pending | require <=87.236 ms |

## Decision

HOLD FOR HUMAN REVIEW

The mathematical decomposition and estimated 201,728-byte SMEM budget fit the
H100 envelope. Implementation is deliberately paused at the mandatory
pipeline/barrier review gate.

## Record

```bash
python scripts/record_result.py EXP-0035 \
  --kernel global-d512-dq-d256-streaming --arch sm_90 \
  --decision refine --hypothesis '<measured result>' --bench <jsonl>
```
