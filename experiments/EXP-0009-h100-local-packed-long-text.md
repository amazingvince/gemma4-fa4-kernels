# EXP-0009: H100 packed local text at production lengths

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-fwd and local-d256-bwd integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar path and revision: pinned
  `flash_attn/cute/{interface,flash_fwd_sm90,flash_bwd_sm90,block_info}.py`
- Design brief: `docs/h100-m1-local-long-context-design.md`

## Invariant changed

The adapter's native packed-text admission bound changes from the EXP-0008
evidence ceiling of K1025 to the locked Gemma maximum K262144. The native
lower-right causal W1024 scheduler, mask, accumulation, tensor geometry,
pipeline, and generated kernels do not change. Metadata-bearing calls retain
the S1025 ceiling until an exact long-context sparse schedule is separately
proved.

## Hypothesis

The pinned SM90 native packed local forward/backward path uses runtime
sequence-local INT32 coordinates and a bounded W1024 tile schedule, so widening
only text admission through K262144 preserves exact lower-right O/LSE/dQ/dK/dV
semantics without making sequence values part of the compile cache key.

## Single change

Admit nonempty native packed text with per-sequence
`1 <= Sq <= Sk <= 262144`; preserve the custom metadata `Sk <= 1025` guard.
No kernel, tile, stage, barrier, or mask callable changes in this experiment.

## Correctness evidence

- [ ] locked contract and optional HF oracle
- [ ] CPU admission/rejection tests
- [ ] long fake forward/backward/LSE-only compilation
- [ ] tractable >1025 equal/lower-right reference matrix
- [ ] exact Q1/K262144 far-offset O/LSE/dV sentinel
- [ ] ragged packed-boundary isolation
- [ ] repeated-run and nondefault-stream checks
- [ ] memory preflight and full-Q long smoke

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck
- [ ] cache-key boundedness across long runtime lengths
- [ ] PTX/cubin/SASS and registers/spills/SMEM recorded

## Measurement

No performance measurement or speed claim is authorized for EXP-0009.

## Decision

**OPEN.** Record ACCEPT, REJECT, or REFINE only after the declared H100 gates.

## Record

Append through `scripts/record_result.py` only after an implementation commit
and strict H100 evidence exist.
