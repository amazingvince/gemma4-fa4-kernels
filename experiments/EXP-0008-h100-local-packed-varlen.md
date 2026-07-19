# EXP-0008: H100 local packed variable length

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-varlen-fwd / local-d256-varlen-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Design brief: `docs/h100-m1-local-varlen-design.md`

## Invariant changed

Add packed Tq/Tk inputs, CUDA INT32 cumulative lengths, lower-right query
coordinates, and globally indexed K-stream vision/document auxiliaries to the
accepted local d256 family. No model mask, scale, prepared-Q/K/V, WGMMA, TMA,
barrier, accumulation, or gradient-separation invariant changes.

## Hypothesis

The pinned public SM90 varlen autograd path will apply Gemma's exact local
predicate independently within every packed sequence in forward and
transposed backward, including Sq less than Sk, without keying compilation on
runtime cumulative values or metadata contents.

Falsification is a compile/launch error, cross-sequence/document leakage,
lower-right or W1024 boundary mismatch, O/LSE/gradient failure, runtime-value
cache growth, sanitizer finding, or unrecorded resource/protocol change.

## Single change

One validated Gemma self-attention packed adapter and one global-index custom
predicate.
Text-only varlen uses FA4's native causal/local path; fixed paths remain
unchanged.

## Correctness evidence

- [ ] CPU packed reference and validation suite
- [ ] exact text/custom fake compilation
- [ ] equal and lower-right length matrices
- [ ] O and FP32 LSE
- [ ] separate dQ, dK, and dV
- [ ] LSE-only and combined gradients
- [ ] vision/document/packed-boundary isolation
- [ ] structured ownership
- [ ] repeats and nondefault stream

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck
- [ ] PTX/cubin/SASS observation
- [ ] registers/spills/SMEM and cache keys recorded

## Measurement

No performance measurement or speed claim is authorized for EXP-0008.

## Decision

**OPEN.** Record ACCEPT, REJECT, or REFINE only after the declared H100 gates.

## Record

Append through `scripts/record_result.py` only after an implementation commit
and strict H100 evidence exist.
