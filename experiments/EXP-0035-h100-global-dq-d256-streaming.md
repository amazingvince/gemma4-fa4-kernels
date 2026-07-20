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

- [x] human approval of the two-generation pipeline and barrier lifecycle
- [x] locked contract retained; no tolerance change
- [x] fake compile and fixed S1/31/32/33/63/64/65/127/128/129 references
- [x] packed tiny/mixed/reversed/empty references through K2048
- [x] O, LSE, separate dQ/dK/dV, structured ownership, and segment isolation
- [x] repeated nondefault-stream cases and bounded fixed/packed memory

## Synchronization and generated code

- [x] fixed and packed memcheck: 0 errors
- [x] fixed and packed synccheck: 0 errors
- [x] fixed and packed racecheck: 0 hazards after the statistic-load rendezvous
- [x] SASS: 66 HGMMA instructions in each stream dQ specialization
- [x] resources: 168 registers, 0 local bytes, 201,728 dynamic shared bytes
- [x] retained dKV object byte-identical, SHA-256 `1df46be3f4071fa3fcf0a7d5a40f7ddbc4c589131ab83155b7f7089a51a23771`

The strengthened packed O+LSE racecheck exposed a producer overwrite against
the two consumer warpgroups' statistic loads. The accepted implementation adds
one two-warpgroup rendezvous after both statistic loads and WGMMA drain, before
either consumer releases the one-stage slot. Post-fix fixed and packed
memcheck/synccheck/racecheck are all clean. The rendezvous changes S8K backward
from 57.456 ms to 57.683 ms (0.4%).

## Measurement

- Clock/power: unlocked and labeled unless locking becomes available
- L2: S8K hot first gate, then hot/cold confirmation
- Warmup/repetitions: 10/30 CUDA events, median/p25/p75/IQR
- Baseline: accepted exact FA4 composition

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 96.928 / 1.308 ms | 57.683 / 1.054 ms | -40.5%, 1.68x |
| global S8K fwd_bwd hot | 103.098 / 1.224 ms | 63.507 / 0.158 ms | -38.4%, 1.62x |
| global S8K bwd cold | 96.839 / 1.101 ms | 57.530 / 0.369 ms | -40.6%, 1.68x |
| global S8K fwd_bwd cold | 103.333 / 1.209 ms | 63.056 / 0.088 ms | -39.0%, 1.64x |
| global S64K bwd hot, 10/30 | 6039.759 / 23.567 ms | 3435.331 / 2.474 ms | -43.1%, 1.76x |
| global S64K fwd_bwd hot, 10/30 | 6400.255 / 2.418 ms | 3803.218 / 2.149 ms | -40.6%, 1.68x |

The S8K first-gate baseline was also rerun in the same live session at
97.130 / 1.329 ms. Its IQR is disjoint from the candidate's. All measurements
are unlocked-clock H100 results and include the full-D dPsum add and all four
main backward launches.

## Decision

ACCEPT

The exact-BF16 D256-streaming dQ path passes compile, fixed and packed
correctness, memory, generated-code, sanitizer, hot/cold S8K, and full S64K
gates. It becomes the default H100 global d512 backward path. Set
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D256_STREAM=0` to select the retained
accepted slab dQ fallback without changing dKV.

## Record

```bash
python scripts/record_result.py EXP-0035 \
  --kernel global-d512-dq-d256-streaming --arch sm_90 \
  --decision accept --hypothesis '<measured result>' --bench <jsonl>
```

Raw post-fix benchmark rows, fixed/packed probes, and sanitizer logs are under
`agent_space/remote-h100-exp0035/`.
