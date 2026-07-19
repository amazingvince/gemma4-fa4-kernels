# Kernel architecture plan

This plan starts at the prepared-Q/K/V FMHA boundary defined in
`docs/model-contract.md`. It does not assume K=V.

## 1. Two independent families

### Local d=256

- 50 layers, Q/KV heads 32/16, GQA 2.
- Window 1024; short mainloop and boundary-heavy masking.
- Text fast path and vision-block tile classifier.
- H100 generic SM90 path exists as a baseline.
- B300 dedicated d=256 path must gain local semantics; compare one-CTA and
  two-CTA schedules instead of assuming two-CTA wins.

### Global d=512

- 10 layers, Q/KV heads 32/4, GQA 8.
- Full causal; dominates attention arithmetic above modest sequence lengths.
- K and V are distinct prepared operands, so both need storage/dataflow and
  backward gradients.
- No current dense SM90/SM103 FA4 path covers the complete contract.

## 2. Global H100 starting design

Initial candidates:

```text
M64 × N32 or M64 × N64
one TMA producer warpgroup role
at least two consumer warpgroup roles
output-D split 256 + 256
FP32 accumulators
one CTA per SM initially
```

An M64×D512 FP32 output contains 32768 values. One 128-thread warpgroup would
hold 256 accumulator values per thread before score, softmax, address, and
pipeline state, which is structurally too close to the register limit. Split
D across consumer groups.

Shared-memory plans to compare:

1. Q plus one-stage K and V buffers;
2. K/V time-sharing after QK consumes K (storage reuse, not operand identity);
3. N32 with deeper staging;
4. smaller logical M packed across GQA heads.

## 3. Global B300 starting design

Compare:

- one CTA with M32/M64 and sequential 256- or 128-wide output slabs;
- a two-CTA cluster where each CTA owns half of output D;
- cooperative score production versus duplicated QK;
- static versus CLC scheduling by workload regime.

TMEM makes d=512 possible but does not remove S/O placement, cluster,
softmax, or epilogue constraints. SM103 uses native exponential and row-reduce
features; never inherit B200 tuning constants without measurement.

## 4. Backward ownership

Use three logical stages:

1. preprocess `D = sum(O * dO)` and internal LSE representation;
2. Q-major owner-computes dQ;
3. K-major owner-computes dK and dV, initially as separate kernels if live
   accumulator pressure is excessive.

Each output tile has one owner and accumulates in FP32 on-chip, avoiding whole
layer FP32 dQ/dK/dV workspaces at long context. GQA group ownership processes
several Q heads per KV head, reuses K/V, and reduces dK/dV contributions
on-chip. Tune packing factors 2, 4, and 8.

A temporary global FP32 accumulator path is acceptable for bring-up but is not
the intended maximum-context design.

## 5. Local mask specialization

Classify tiles as:

1. outside the window: skip;
2. fully valid causal interior: no element predicate;
3. same-vision-block interior: full tile inside the lower window;
4. causal diagonal, window edge, document edge, or vision edge: predicate.

Backward uses the transposed ownership/range calculation. Do not route the
whole workload through a generic per-element mask if tile classification can
prove an interior case.

## 6. Optional model-specific fusion after base FMHA

A later global kernel may consume the shared projection source Z and form:

```text
K = partial_RoPE(KNorm(Z))
V = VNorm(Z)
```

Its backward may return dZ only after applying both distinct preparation
adjoints. This is a separate fusion milestone with new tests and source-level
integration; it is not a shortcut for the base d=512 attention kernels.

## 7. Architecture and integration boundaries

- SM90 and SM103 share interfaces and references, not tuning tables or
  pipeline assumptions.
- Runtime sequence lengths, offsets, and vision IDs stay out of compile keys.
- Support fixed BSHD and packed varlen layouts through explicit adapters.
- The base checkpoint has no cross-layer KV reuse (`num_kv_shared_layers=0`); keep
  support for future variants outside the initial fast-path contract.
- Tensor-parallel or KV-replicated per-rank shapes can expose GQA ratios
  1/2/4/8, especially in the global family; dispatch and tests cover all four
  while preserving full-model ratios 2 and 8 as first-class cases.

## 8. Implementation order

1. Contract/oracle/benchmark rig.
2. B300 local d=256 correctness, one-CTA and two-CTA candidates.
3. H100 and B300 global d=512 forward storage prototypes.
4. Global d=512 forward O/LSE correctness.
5. Q-major dQ and K-major dK/dV.
6. Local multimodal/varlen backward.
7. framework dispatch, KV-sharing integration, context-parallel offsets.
8. projection/norm/RoPE fusion and lower precision only after BF16 evidence.
