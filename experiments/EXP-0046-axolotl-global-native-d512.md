# EXP-0046: Axolotl native global d512 production route

- Date / author: 2026-07-21 / Codex
- Kernel family: integration
- Architecture: sm_90
- Official FA4 base: `2409214a03797b168f648ea30df1adbc09ce658a`
- Candidate FA4 revision: `17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1`
- CuTe DSL / CUDA / PyTorch: CuTe DSL `4.6.0.dev0`, CUDA `12.8`,
  PyTorch `2.11.0+cu128`
- Model: cached `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`

## Invariant changed

EXP-0045 routed all 48 text layers through FA4. This experiment changes only
the local implementation: the 40 native d256 sliding layers use upstream FA2,
while the 8 native symmetric-d512 global layers use the generic FA4 fork.
The model-facing tensors remain `16Q/8KV/d256` locally and `16Q/1KV/d512`
globally. No Q/KV head padding or KV repetition is permitted.

## Hypothesis

Using the mature FA2 local path and the fork only where d512 is required will
remove EXP-0045's S1024 integration regression without weakening the prepared
Q/K/V forward or separate-gradient correctness contract.

## Single change

Add the explicit `gemma4_fa4_h100_global_native` backend. It accepts only the
exact route mix `fa2_local/{fixed,varlen}` for sliding layers and
`fa4_native/global_{fixed,varlen}` for global layers. Deterministic FA4
backward becomes opt-in through `FLASH_ATTENTION_DETERMINISTIC=1`, matching the
public library policy. Repeated packedness and all-valid-padding checks cache
their results on the exact tensor shape/stride/device/dtype/version identity.

## Correctness evidence

- [x] Full local suite: `510 passed, 106 skipped`; all skips are declared
  optional-oracle or hardware-only cases.
- [x] Ruff, compileall, and `git diff --check` pass.
- [x] One real S1024 run covers exactly 320 local FA2 calls and 64 global FA4
  calls across the eight training steps.
- [x] Every prepared K and V is distinct; the global geometry is native
  `16Q/1KV/d512` with no padding or repetition.
- [x] Prepared forward checks pass on actual local and global model tensors.
  Global FA4 and BF16 SDPA each have maximum absolute error `0.125` against
  the FP32 reference (declared limit `0.25`).
- [x] Prepared global backward checks pass independently for dQ, dK, and dV
  against FP32, with BF16 SDPA defining the low-precision envelope.
- [x] Exact candidate-tip FA4 tests report `98 passed, 2 skipped` on H100 and
  `97 passed, 3 skipped` under fake compilation.
- [x] Exact fixed and varlen racecheck probes report zero hazards, errors, and
  warnings. The existing fixed backward memcheck/synccheck matrix remains
  clean because the benchmark-only tip amendment did not alter kernel code.
- [ ] Whole-model numerical-envelope gate. Against an identical full-SDPA
  replay, the hybrid gradient sketch has cosine `0.677894` and relative L2
  `0.76213`; the candidate has cosine `0.478484` and relative L2 `1.25293`.
  Prepared attention-level checks pass, but this stricter heuristic is not
  weakened or silently waived.

## Measurement

Three fresh-process B1/S1024 pairs used three warmups and five measured steps.
The pod denied application clock locking; observed runs were P0 at 1980 MHz
graphics and 2619 MHz memory. Every job held the exclusive H100 lock.

| Pair | hybrid median ms | candidate median ms | candidate delta |
|---:|---:|---:|---:|
| 1 | 257.990 | 259.192 | +0.466% |
| 2 | 265.474 | 266.242 | +0.289% |
| 3 | 263.073 | 260.284 | -1.060% |

All pairs satisfy the no-greater-than-3% step-time gate. Peak allocated memory
is identical at `24,114,589,696` bytes; candidate/hybrid peak reserved memory
is `40,045,117,440` bytes.

The exact-tip direct d512 benchmark uses 5 warmups and 50 samples per point.
Combined forward+backward medians are `1.687 ms` at 12B/S1024, `19.014 ms` at
12B/S8192, `1.320 ms` at 31B/S1024, and `37.178 ms` at 31B/S8192. Every point
beats both native PyTorch SDPA-GQA and explicitly expanded-KV SDPA with
non-overlapping IQRs. These are attention-kernel measurements, not 31B
whole-model claims.

Interleaved same-process base-versus-candidate checks keep existing paths
inside the 3% regression gate. Paired median deltas are +0.39% for d64,
+0.74% for d128, +0.47% for d192 forward, and +0.39% for d256; every paired
ratio IQR lies inside +/-2.1%. Upstream-base d192 backward does not compile in
this exact dependency environment and is not represented as a candidate
regression.

## Decision

**REFINE / KEEP FORK PR DRAFT.** The mixed route closes the measured S1024
performance and memory gates, and the prepared attention-level forward and
backward oracles pass. The fork PR must remain draft until either the declared
whole-model numerical-envelope gate is explained and accepted or the
candidate is improved to satisfy it.

Gemma 4 31B QLoRA also remains unrun: the pod has no cached checkpoint and no
Hugging Face authentication for the gated repository. Native
`32Q/4KV/d512` fixed/varlen tests and direct benchmarks pass, but they are not
a substitute for a 31B training claim. Nsight Compute counters remain blocked
by host policy (`ERR_NVGPUCTRPERM`).

## Reproduction outline

Select `fa4_harness_backend: global_native` in the existing Axolotl smoke
configuration, install candidate `flash-attn-4` at the exact revision above,
and run all GPU commands while holding `/workspace/.h100-codex.lock`:

```bash
axolotl train configs/axolotl/gemma4-12b-smoke.yaml --launcher python
```

Set `GEMMA4_FA4_PREPARED_ORACLE=1` and
`GEMMA4_FA4_PREPARED_BACKWARD_ORACLE=1` for the actual-layer forward/backward
oracle. The direct benchmark is
`benchmarks/benchmark_fa4_sm90_d512.py` in the FlashAttention fork and records
the command, commit, dirty state, every sample, quartiles, runtime memory, and
software/GPU provenance in JSON.
