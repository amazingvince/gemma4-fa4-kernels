# EXP-0036: Axolotl Gemma 4 12B real-model harness

- Date / author: 2026-07-20 / Codex
- Kernel family: integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Axolotl revision: `2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417`
- Gemma 4 12B-it revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- CuTe DSL / CUDA / PyTorch: experimental Axolotl environment; the accepted
  `configs/env/h100-compatible.env` is not widened by this experiment
- Gemma 4 12B harness lock hash:
  `371c558900c662328c7fff91a6cb586dfd6d99997e7cc4f78e832707b9963bb9`
- Gemma 4 31B kernel-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar path and revision: Axolotl
  `examples/gemma4-unified/12b-text-lora.yaml` at the pinned revision above

## Invariant changed

No accepted 31B kernel invariant changes. Add a separately named integration-only
adapter that embeds Gemma 4 12B prepared heads into the validated 31B attention
geometry and slices the output back to 12B geometry. The benchmark must prove that
all 48 real-model layers enter an FA4 route and must reject every fallback.

## Hypothesis

For a fixed B1/S1024, BF16, eight-step, zero-learning-rate LoRA workload on one
H100, the 12B compatibility adapter matches the hybrid baseline's losses and a
bounded LoRA-gradient sketch within the frozen harness policy, with 40 local and
8 global FA4 layers observed on every project step; measured median/IQR determines
whether compatibility overcompute is faster or slower than the hybrid default.

## Single change

Add the 12B head-geometry compatibility adapter, Axolotl plugin, deterministic
smoke workload, report comparator, and orchestration scripts. The integration
proof may simplify only a pinned all-negative Unified vision closure to boolean
false; any active vision closure remains rejected. Do not modify kernel code,
mask/scale semantics, tolerances, or the accepted 31B backend name.

## Correctness evidence

- [x] locked 12B configuration and pinned-model identity
- [x] CPU forward equivalence for local and global prepared attention
- [x] CPU dQ/dK/dV equivalence, including duplicated global KV reduction
- [ ] real-model loss and bounded gradient-sketch comparison: rejected by the
  frozen loss and gradient gates
- [x] invalid geometry, active-vision, and fallback cases
- [x] deterministic dataset plus identical name-seeded LoRA fingerprint across
  the three independent processes

## Synchronization and generated code

- [x] no H100 kernel source or memory/barrier protocol changed; sanitizers are
  not applicable to this integration-only candidate
- [x] project route coverage recorded for all 48 model layers
- [x] 320 local and 64 global native calls recorded; no fallback route observed
- [x] isolated cache key
  `48655f5d8f5d62e671313936c4b4976815da48e17f73ca561d912923ea87a2d9`
  recorded under the timestamped evidence directory

## Measurement

- Clock/power state: unlocked/default application clocks; performance is
  rejection evidence only, not an accepted ruler
- Hot/cold L2: end-to-end training step, naturally hot after three excluded warmups
- Warmup/repetitions/statistic: 3 warmup / 5 measured; median/p25/p75/IQR
- Semantically equivalent baseline: Axolotl Gemma 4 hybrid attention on the identical
  zero-update LoRA workload; SDPA is a secondary correctness baseline

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| hybrid vs project | 250.489 / 0.750 ms | 280.720 / 2.529 ms | 0.892x (project slower) |
| SDPA vs project | 256.564 / 0.863 ms | 280.720 / 2.529 ms | 0.914x (project slower) |

## Decision

REJECT

REJECT the 12B compatibility route as a faster, correctness-accepted real-model
candidate. Final isolated-cache run `20260720T211644Z` used the locked dataset SHA256
`27c2f7f54f7ebb123bb7336b75ce73b6a7e532dc7e33683a9da249478ada839a`,
exactly 1024 tokens per B1 step, and the identical 368-parameter LoRA
initialization fingerprint
`219d5a46e7c638ed54cbc6df3a886b9b622a7b63092ed1155506fa877ee95f59`
in all three processes.

The project completed all eight steps and recorded exactly 320
`fa4_local_fixed` plus 64 `fa4_global_fixed` compatibility calls, covering all
40 local and 8 global layers on every step with no fallback. The real-model
oracle nevertheless rejected it:

- hybrid/project maximum measured-loss delta: `0.05737781524658203`;
- SDPA/project maximum measured-loss delta: `0.07951641082763672`;
- hybrid/project gradient cosine / relative L2: `0.1765651 / 1.0069687`;
- SDPA/project gradient cosine / relative L2: `0.4867264 / 0.9706694`.

All exceed the predeclared `loss atol=5e-3, rtol=2e-3`, gradient cosine
`>=0.999`, and relative-L2 `<=0.01` policy. Hybrid and SDPA also differ by
`0.07342815399169922` maximum loss and have gradient cosine `0.1919194`, so
this end-to-end BF16 oracle does not establish equivalence among the baselines
either; the policy is not loosened after observing the result.

Performance also rejects the overcomputing adapter: its 280.720 ms median is
12.1% slower than hybrid and 9.4% slower than SDPA under the unlocked-clock
screen. Peak active memory was 37.68 GiB for project versus 36.44 GiB for
hybrid. No 12B correctness, speedup, training-convergence, or native-12B-kernel
claim is accepted. The harness itself remains useful as fail-closed evidence
and now records deterministic LoRA fingerprints and rejected comparison JSON.

## Record

```bash
python scripts/record_result.py EXP-0036 \
  --kernel axolotl-gemma4-12b-compat --arch sm_90 \
  --decision reject --hypothesis '<measured result>' --bench <jsonl>
```
