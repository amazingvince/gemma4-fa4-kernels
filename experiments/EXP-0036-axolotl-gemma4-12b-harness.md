# EXP-0036: Axolotl Gemma 4 12B real-model harness

- Date / author: 2026-07-20 / Codex
- Kernel family: integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Axolotl revision: `2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417`
- Gemma 4 12B-it revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- CuTe DSL / CUDA / PyTorch: experimental Axolotl environment; the accepted
  `configs/env/h100-compatible.env` is not widened by this experiment
- Model-contract lock hash:
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

Add only the 12B head-geometry compatibility adapter, Axolotl plugin, deterministic
smoke workload, report comparator, and orchestration scripts. Do not modify kernel
code, mask/scale semantics, tolerances, or the accepted 31B backend name.

## Correctness evidence

- [ ] locked 12B configuration and pinned-model identity
- [ ] CPU forward equivalence for local and global prepared attention
- [ ] CPU dQ/dK/dV equivalence, including duplicated global KV reduction
- [ ] real-model loss and bounded gradient-sketch comparison
- [ ] invalid geometry and fallback cases
- [ ] repeated-run/determinism check

## Synchronization and generated code

- [ ] unchanged accepted objects verified by scope inspection
- [ ] project route coverage recorded for all 48 model layers
- [ ] no new memory or barrier protocol (sanitizers not applicable to adapter-only change)
- [ ] existing kernel cache keys recorded by the remote run

## Measurement

- Clock/power state: pending remote preflight
- Hot/cold L2: end-to-end training step, naturally hot after three excluded warmups
- Warmup/repetitions/statistic: 3 warmup / 5 measured; median/p25/p75/IQR
- Semantically equivalent baseline: Axolotl Gemma 4 hybrid attention on the identical
  zero-update LoRA workload; SDPA is a secondary correctness baseline

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| 12B B1/S1024 fwd+bwd step | pending | pending | pending |

## Decision

REFINE

REFINE. The implementation is locally complete: 454 tests pass (106 expected
GPU/optional skips), including exact CPU O/dQ/dK/dV equivalence for both 12B layer
families, invalid-geometry rejection, collision-safe registration, report-policy
gates, and deterministic assets. Ruff and `git diff --check` pass.

Remote H100 evidence reached the following boundary before work was stopped to
avoid colliding with another task sharing the GPU:

- exact Axolotl, Transformers, and FA4 revisions and both project patches passed;
- PyTorch 2.11.0+cu128, CUDA 12.8, one SM90 H100, 85,017,493,504 bytes HBM,
  and gated-model access passed;
- FA2 and FA4 were proven unsafe to co-install because both own `flash_attn`, so
  the runner/bootstrap now require separate baseline and project venvs;
- Axolotl accepted the plugin and the explicitly registered custom backend after
  its plugin-time canonical-backend allowlist was extended;
- the real-model launch then stopped before weight loading because the unified
  processor required `torchvision`, which is now included in the bootstrap.

No training step, correctness comparison, or performance measurement completed.
The table therefore remains pending and no speed claim is made. A compatibility
route may still be slower because it performs 31B-head-count attention work.

## Record

```bash
python scripts/record_result.py EXP-0036 \
  --kernel axolotl-gemma4-12b-compat --arch sm_90 \
  --decision refine --hypothesis '<measured result>' --bench <jsonl>
```
