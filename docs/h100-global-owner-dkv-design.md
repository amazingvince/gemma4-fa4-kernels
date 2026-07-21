# H100 global owner-computes dK/dV design brief

This is the pre-implementation ownership and synchronization gate for
EXP-0040. It removes the whole-sequence FP32 dK/dV accumulation buffers without
changing the accepted EXP-0038 dQ or EXP-0039 deterministic route.
It is the first bounded-memory slice; owner-computes dQ remains a separate
experiment.

## 1. Scope and invariants

- Target: H100 SM90a, exact BF16 global-causal Gemma 4 attention.
- Geometry: 32 Q heads, 4 KV heads, GQA 8, Q/K/V/D=512, scale exactly 1.0.
- Inputs remain distinct prepared Q/K/V; outputs remain separate dQ/dK/dV.
- Fixed BSHD and native packed THD retain their current admitted envelopes.
- Forward and the dQ kernel are unchanged.
- The candidate has its own compile/application cache key. After its declared
  gates passed, it became the fast default with an environment rollback.

## 2. Ownership

The existing dKV grid is K-major but schedules one CTA for every Q head. Eight
CTAs therefore contribute to each KV-head tile through whole-sequence FP32
bulk reductions. The candidate elects one CTA for each `(batch, kv_head,
N32)` tile. That CTA visits the eight Q heads in increasing order and visits
all causal M64 blocks for each head before writing the tile once.

The dK and low/high dV accumulators stay FP32 in registers for the complete
owned tile. dK receives scale 1.0 once after the last Q head. The epilogue
converts both dV D256 halves and dK to BF16 in shared memory, then uses direct
TMA stores to the final distinct dK and dV tensors. No semaphore, atomic, or
whole-sequence FP32 dK/dV destination is required.

## 3. Pipeline and synchronization

The correctness-first implementation replays K/V loads for each owned Q head
so the existing producer/consumer pipeline generations stay paired. Removing
that replay is a later performance experiment, not part of the ownership
change.

- Only Q-head leaders `0, 8, 16, 24` execute; other scheduled Q-head tiles do
  no memory work and write nothing.
- Producer and consumer warpgroups use the same leader predicate and increasing
  Q-head loop, so every acquired Q/dO generation has exactly one consumer.
- The dKV epilogue runs once after all eight heads. Low and high dV register
  fragments fill disjoint halves of the shared V tile before one full-D TMA
  store; the dK store follows the existing epilogue barrier discipline.
- Empty-query packed segments execute the same single-owner zero epilogue and
  cannot modify another segment.

## 4. Memory and cache contract

The candidate removes these padded FP32 buffers:

```text
dK: 4 * padded_K * 512 * 4 bytes
dV: 4 * padded_K * 512 * 4 bytes
total: 16384 * padded_K bytes
```

Final BF16 dK/dV outputs remain required. Whole-sequence FP32 dQ and its
postprocess remain, so EXP-0040 is not the complete bounded-memory backward.
Runtime lengths and cumulative values stay out of compile keys; fixed and
packed SS/SM/MM scheduler classes remain bounded.

## 5. Gates and rollback

- Fixed and packed independent-reference matrices, gradient-source isolation,
  GQA ownership, empty-segment isolation, and nondefault stream.
- Five exact dK/dV repeats; dQ follows the selected existing fast or
  deterministic route.
- Fixed and packed memcheck, synccheck, and racecheck.
- One owner dKV main object per ABI, direct BF16 dK/dV stores in generated
  code, no dK/dV postprocess launches, bounded cache, and measured peak below
  the updated preflight estimate.
- S8K and S64K characterization against the unchanged EXP-0038 default. Reject
  correctness/sanitizer failures, spills, cache growth outside the declared
  ABI classes, or a candidate that allocates a full-sequence FP32 dK/dV tensor.

Rollback is immediate: set
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_OWNER_DKV=0` to restore the accepted
EXP-0038 dKV accumulation/postprocess route. `deterministic=True` remains the
accepted EXP-0039 route and never selects owner dKV.
