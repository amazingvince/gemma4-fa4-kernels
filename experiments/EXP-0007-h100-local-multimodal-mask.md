# EXP-0007: H100 exact local multimodal mask

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-fwd / local-d256-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar path and revision: pinned `flash_attn/cute/mask.py`,
  `flash_fwd_sm90.py`, `flash_bwd_sm90.py`, `interface.py`, and
  `tests/cute/mask_mod_definitions.py`
- Design brief: `docs/h100-m1-local-multimodal-design.md`

## Invariant changed

Add one static, hash-stable SM90 mask callable and one runtime contiguous
INT32 `(S,)` vision-ID auxiliary tensor to the accepted local d256 path. The
public B=1 `(1,S)` IDs are normalized to that private layout. The mask callable, rather
than FA4's native causal/local flags, owns the complete Gemma predicate:

```text
k > q - 1024
AND
(k <= q OR same nonnegative vision block)
```

No WGMMA atom, tile ownership, TMA pipeline, barrier, stage, accumulation
type, epilogue, scale, or prepared-Q/K/V invariant changes.

## Hypothesis

The pinned SM90 custom-mask path, called with `causal=False` and no native
window, will apply Gemma's exact local vision overlay in both forward and
transposed backward and pass the fixed-length B1/32Q/16KV/d256 matrix without
changing the accepted text path.

The hypothesis is falsified by a compile/launch error, a mask boundary or
gradient mismatch, a cache variant keyed by runtime IDs/length, a sanitizer
report, or an unexpected pipeline/resource change.

## Single change

One mask/auxiliary dispatch path for fixed-length local multimodal attention.
Text dispatch remains byte-for-byte semantically equivalent to EXP-0001 and
EXP-0004.

## Correctness evidence

- [ ] locked contract and optional HF oracle
- [ ] fake-tensor compile
- [ ] boundary/tail/vision-pattern matrix
- [ ] O and FP32 LSE
- [ ] separate dQ, dK, and dV under the EXP-0004 numerical policy
- [ ] invalid/fallback cases
- [ ] repeated-run and nondefault-stream checks

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck
- [ ] IR/PTX/SASS observation
- [ ] registers/spills/SMEM recorded

## Measurement

No performance measurement is authorized for this correctness experiment.

## Decision

**OPEN.** Record ACCEPT, REJECT, or REFINE only after the declared H100 gates.

## Record

Append through `scripts/record_result.py` only after a source commit and strict
environment artifact exist.
