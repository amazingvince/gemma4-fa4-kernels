# EXP-0000: example baseline capture (do not record as a real result)

- Date / author: example only
- Kernel family: integration
- Architecture: sm_103
- Upstream FA4 revision: see `upstream.lock.json`
- CuTe DSL / CUDA / PyTorch: see `configs/env/latest-compatible.env` for this
  sm_103 example (`configs/env/h100-compatible.env` is the sm_90 policy)
- Model-contract lock hash: capture with `sha256sum configs/model/gemma4-31b.lock.json`
- Exemplar path and revision: current upstream public interface

## Invariant changed

None. This record demonstrates the evidence shape for an M0 baseline.

## Hypothesis

The fixed canary and clock policy keep repeated median latency within the
measured noise threshold during one session.

## Single change

No code change; baseline measurement only.

## Correctness evidence

- [ ] locked contract and pinned HF oracle
- [ ] targeted O/LSE/dQ/dK/dV matrix
- [ ] invalid/fallback cases

## Synchronization and generated code

Not applicable for an unchanged upstream baseline; record compiled variants,
PTX/SASS spot check, and sanitizer coverage actually run.

## Measurement

- Clock/power state:
- Hot/cold L2:
- Warmup/repetitions/statistic:
- Semantically equivalent baseline:

| case | median | IQR | status |
|---|---:|---:|---|

## Decision

BASELINE — no accept/reject claim.

## Record

```bash
python scripts/record_result.py EXP-0000 \
  --kernel upstream-baseline --arch sm_103 --decision baseline \
  --hypothesis 'canary drift stays below the measured noise threshold' \
  --bench agent_space/baseline.jsonl
```
